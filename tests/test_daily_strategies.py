"""Daily families: causality (prefix invariance), value ranges and the documented rule on hand-made cases."""
import numpy as np
import pandas as pd
import pytest

import fxlab.strategies  # noqa: F401
from fxlab.evaluate import run_signals
from fxlab.registry import CATALOG

DAILY = sorted(k for k in CATALOG if k.startswith("S.DAILY."))


def _bars(n=1600, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n, tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)
    for k, v in zip("ohlc", (o, h, lo, c)):
        df[f"bid_{k}"], df[f"ask_{k}"] = v * (1 - 1e-4), v * (1 + 1e-4)
    vix = pd.Series(15 + 10 * np.abs(np.sin(np.arange(n) / 40)) + rng.normal(0, 1, n), index=idx)
    return df, pd.DataFrame({"VIX": vix})


@pytest.mark.parametrize("sid", DAILY)
def test_daily_family_is_causal_and_bounded(sid):
    df, panel = _bars()
    for p in ({}, {"short": 1} if "short" in CATALOG[sid].params else {}):
        s = run_signals(sid, df, panel, "X", p)["side"]
        assert s.index.equals(df.index) and np.isfinite(s).all() and (s.abs() <= 1 + 1e-12).all()
        k = 1100
        s2 = run_signals(sid, df.iloc[:k], panel.iloc[:k], "X", p)["side"]
        assert np.allclose(s.iloc[:k].to_numpy(), s2.to_numpy()), f"{sid} changes when future bars are removed"
        assert (s != 0).any()


def test_rules_on_hand_made_cases():
    df, panel = _bars()
    # Halloween: long exactly in November-April
    s = run_signals("S.DAILY.HALLOWEEN", df, panel, "X", {})["side"]
    assert (s[df.index.month.isin([11, 12, 1, 2, 3, 4])] == 1).all() and (s[df.index.month == 7] == 0).all()
    # OPEX: March 2016 expiration Friday is the 18th -> held at the closes of 11, 14, 15, 16, 17 March
    s = run_signals("S.DAILY.OPEX", df, panel, "X", {})["side"]
    m = s["2016-03-01":"2016-03-31"]
    assert list(m[m == 1].index.day) == [11, 14, 15, 16, 17]
    # IBS: a close at the low enters, a close at the high exits
    d2 = df.copy()
    t = d2.index[1300]
    d2.loc[t, ["close", "bid_c", "ask_c"]] = d2.loc[t, "low"]
    s = run_signals("S.DAILY.IBS", d2, panel, "X", {"lo": 0.2, "x": 0.7})["side"]
    assert s.loc[t] == 1
    # VOL_TARGET never exceeds 1 and falls when volatility rises
    s = run_signals("S.DAILY.VOL_TARGET", df, panel, "X", {"tv": 0.10})["side"]
    assert s.max() <= 1 and s.iloc[300:].min() > 0
