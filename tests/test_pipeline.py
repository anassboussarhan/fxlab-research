"""Data pipeline, orchestration (checkpoint/resume, failure capture) and
reference-value numerical tests for key estimators."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fxlab.data import store


# ------------------------------------------------------------------------------------------- data pipeline
@pytest.fixture(scope="module")
def m1():
    if not store.available("EURUSD"):
        pytest.skip("no data")
    return store.load_m1("EURUSD", "2012-03-01", "2012-03-31")


def test_m1_bid_ask_semantics(m1):
    assert (m1["ask_c"] >= m1["bid_c"]).all()
    assert (m1["bid_h"] >= m1["bid_l"]).all() and (m1["ask_h"] >= m1["ask_l"]).all()
    assert m1.index.is_monotonic_increasing and not m1.index.duplicated().any()
    assert str(m1.index.tz) == "UTC"


@pytest.mark.parametrize("tf", ["M5", "M15", "H1", "H4", "D1"])
def test_resample_preserves_ticks_and_extremes(m1, tf):
    b = store.resample(m1, tf)
    assert b["n_ticks"].sum() == m1["n_ticks"].sum()
    assert b["bid_h"].max() == m1["bid_h"].max() and b["ask_l"].min() == m1["ask_l"].min()
    assert (b["bid_h"] >= b[["bid_o", "bid_c"]].max(axis=1) - 1e-12).all()
    assert b["n_m1"].sum() == len(m1)  # every M1 bar lands in exactly one bar; no bar is invented


def test_d1_uses_new_york_rollover(m1):
    d = store.resample(m1, "D1")
    ny = d.index.tz_convert("America/New_York")
    assert set(ny.hour) == {17}


def test_alternative_bars(m1):
    for b in (store.tick_bars(m1, 2000), store.volume_bars(m1, 5000), store.volatility_bars(m1, 0.002),
              store.range_bars(m1, 0.0015)):
        assert len(b) > 10 and b["n_ticks"].sum() <= m1["n_ticks"].sum()
        assert b.index.is_monotonic_increasing


@pytest.mark.parametrize("tf", ["TICK60", "VOLA60", "LIQ60"])
def test_adaptive_event_bars_are_causal_and_complete(m1, tf):
    b = store.event_bars(m1, tf)
    assert b["n_m1"].sum() == len(m1) and b["n_ticks"].sum() == m1["n_ticks"].sum()
    assert b.index.is_monotonic_increasing and not b.index.duplicated().any()
    hours = (m1.index[-1] - m1.index[0]).total_seconds() / 3600 * 5 / 7
    assert 0.2 < len(b) / hours < 5          # roughly one bar per active hour on average
    # prefix invariance: bars that closed before the cut are identical when later data is removed
    cut = m1.index[len(m1) * 2 // 3]
    b2 = store.event_bars(m1[m1.index < cut], tf)
    closed = b2.iloc[:-1]                                  # last bar of the prefix may be incomplete
    pd.testing.assert_frame_equal(b.loc[closed.index], closed, check_freq=False)


def test_no_holdout_rows_in_store():
    for sym in ("EURUSD", "XAUUSD"):
        if store.available(sym):
            assert max(store.available(sym)) < 2026


# ------------------------------------------------------------------------------------------- orchestration
def _ok_runner(cfg, outdir):
    return {"metrics": {"x": cfg["v"] * 2}}


def _bad_runner(cfg, outdir):
    if cfg["v"] == 2:
        raise ValueError("boom")
    return {"metrics": {"x": 1}}


def test_campaign_resume_and_failure_capture(tmp_path, monkeypatch):
    from fxlab.orchestration import campaign
    monkeypatch.setattr(campaign, "DB_PATH", tmp_path / "t.duckdb")
    man = [{"kind": "unit", "target": "t", "v": i} for i in range(4)]
    s1 = campaign.run_campaign(man, "tests.test_pipeline._bad_runner", "unit", workers=1)
    assert s1 == {"ok": 3, "failed": 1}
    con = campaign.connect(tmp_path / "t.duckdb")
    err, tb, cfg = con.execute("SELECT error, traceback, config FROM experiments WHERE status='failed'").fetchone()
    assert "boom" in err and "Traceback" in tb and json.loads(cfg)["v"] == 2
    con.close()
    # resume: nothing re-runs unless failures are retried
    s2 = campaign.run_campaign(man, "tests.test_pipeline._ok_runner", "unit", workers=1)
    assert s2 == {"ok": 3, "failed": 1}
    s3 = campaign.run_campaign(man, "tests.test_pipeline._ok_runner", "unit", workers=1, retry_failed=True)
    assert s3 == {"ok": 4}


def test_experiment_id_deterministic():
    from fxlab.orchestration.campaign import exp_id
    assert exp_id({"a": 1, "b": [1, 2]}) == exp_id({"b": [1, 2], "a": 1})


# ------------------------------------------------------------------------------------------- numerical references
def _gbm(n=20000, sigma=0.001, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    # simulate 60 sub-steps per bar for realistic OHLC
    sub = rng.normal(0, sigma / np.sqrt(60), (n, 60))
    path = np.cumsum(sub.ravel()).reshape(n, 60)
    start = np.r_[0, path[:-1, -1]]
    o = np.exp(start); c = np.exp(path[:, -1])
    h = np.exp(np.maximum(path.max(1), start)); l = np.exp(np.minimum(path.min(1), start))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_range_estimators_recover_sigma():
    from fxlab.features import volatility as V
    df = _gbm()
    for f in (V.f_parkinson, V.f_gk, V.f_rs, V.f_yz):
        est = f(df, n=500).dropna().median()
        assert est == pytest.approx(0.001, rel=0.12), f.__name__


def test_hurst_and_dfa_random_walk_near_half():
    from fxlab.features.fractal import _hurst_rs, _dfa
    rng = np.random.default_rng(0)
    r = rng.normal(size=4096)
    assert 0.4 < _dfa(r) < 0.6
    assert 0.45 < _hurst_rs(r) < 0.65  # R/S is biased upward in small samples


def test_permutation_entropy_iid_near_one():
    from fxlab.features.information import _perm_entropy
    r = np.random.default_rng(0).normal(size=2000)
    assert _perm_entropy(r) > 0.98


def test_roll_spread_recovers_bounce():
    from fxlab.features.microstructure import f_roll
    rng = np.random.default_rng(0)
    n = 5000
    true = 1.0 + np.cumsum(rng.normal(0, 1e-5, n))
    s = 0.0002
    obs = true + s / 2 * rng.choice([-1, 1], n)
    df = pd.DataFrame({"close": obs}, index=pd.date_range("2015", periods=n, freq="min", tz="UTC"))
    est = f_roll(df, n=2000).dropna().median()
    assert est == pytest.approx(s, rel=0.15)


def test_sma_rsi_reference():
    from fxlab.features.trend import f_sma
    from fxlab.features.momentum import f_rsi
    df = _gbm(500)
    ref = np.log(df["close"] / df["close"].rolling(20).mean())
    assert np.allclose(f_sma(df, 20).dropna(), ref.dropna())
    r = f_rsi(df, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    up = pd.DataFrame({"close": np.linspace(1, 2, 100)}, index=df.index[:100])
    assert f_rsi(up, 14).iloc[-1] > 99


def test_garch_filter_matches_recursion():
    from fxlab.features.econometric import _filter_garch
    e = np.array([0.1, -0.2, 0.3])
    s2 = _filter_garch(e, 0.01, 0.1, 0.8, 0.0, 0, 0.05)
    exp0 = 0.01 + 0.1 * 0.01 + 0.8 * 0.05
    assert s2[0] == pytest.approx(exp0)
    assert s2[1] == pytest.approx(0.01 + 0.1 * 0.04 + 0.8 * exp0)


def test_bi5_decoder_roundtrip():
    """The direct Dukascopy decoder (network path untestable here) decodes a synthetic bi5 blob exactly."""
    import lzma, struct
    from datetime import datetime
    from fxlab.data.acquire_dukascopy import decode_bi5
    recs = [(0, 120365, 120362, 1.5, 2.25), (1500, 120366, 120363, 1.0, 0.75)]
    raw = lzma.compress(b"".join(struct.pack(">IIIff", *r) for r in recs), format=lzma.FORMAT_ALONE)
    df = decode_bi5(raw, datetime(2018, 1, 2, 8), "EURUSD")
    assert df["bid"].to_list() == [1.20362, 1.20363] and df["ask"].to_list() == [1.20365, 1.20366]
    assert str(df["ts"][1]) == "2018-01-02 08:00:01.500000+00:00" or df["ts"][1].microsecond == 500000
    jpy = decode_bi5(raw, datetime(2018, 1, 2, 8), "USDJPY")
    assert jpy["bid"][0] == 120.362
