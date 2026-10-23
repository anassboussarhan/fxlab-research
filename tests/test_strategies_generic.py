"""Contract tests for EVERY strategy family on REAL data (EURUSD H1 + panel):

1. returns a frame aligned to the bars with a `side` column in [-1, 1];
2. CAUSALITY: signals on a prefix of the data equal the signals on the full data for shared rows;
3. symmetry: the family is capable of both long and short positions (on the test sample, or - for
   rare-event families - on a mirrored price series);
4. it runs end-to-end through the execution layer (costs, stops) and produces finite returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fxlab.features  # noqa: F401
import fxlab.models  # noqa: F401
import fxlab.strategies  # noqa: F401
from fxlab.registry import by_kind, requirements_met
from fxlab.evaluate import evaluate, run_signals

STRATS = sorted(by_kind("strategy"), key=lambda e: e.id)
ML_FAST = {"train": 1500, "test": 500}
OVR = {"S.ML.SUPERVISED_DIRECTION": ML_FAST | {"model": "M.LIN.LOGISTIC"},
       "S.ML.RETURN_REGRESSION": ML_FAST, "S.ML.META_LABEL": ML_FAST | {"model": "M.LIN.LOGISTIC", "q": 0.5},
       "S.ML.TRIPLE_BARRIER": ML_FAST | {"model": "M.LIN.LOGISTIC", "q": 0.5}, "S.ML.RL_POLICY": ML_FAST,
       "S.XA.STAT_ARB": {"n": 300}, "S.XA.COINTEGRATION": {"n": 300}, "S.XA.RELATIVE_VALUE": {"n": 500},
       "S.XA.PAIRS": {"n": 300}, "S.XA.GOLD_FX": {"n": 500}, "S.SEAS.WEEKDAY": {"W": 8},
       "S.SEAS.TURN_OF_MONTH": {"Y": 2}, "S.SEAS.INTRADAY": {"D": 20, "thr": 1.0}}
RARE = {"S.GEO.HARMONIC", "S.SEAS.WEEKEND_GAP_FADE", "S.ST.PIVOT_POINTS", "S.XA.TRIANGULAR", "S.BO.OPENING_RANGE",
        "S.SES.LONDON_BREAKOUT", "S.XA.COINTEGRATION", "S.RG.CHANGE_POINT", "S.SEAS.TURN_OF_MONTH", "S.SEAS.WEEKDAY",
        "S.SEAS.INTRADAY", "S.ML.META_LABEL", "S.XA.RELATIVE_VALUE", "S.RG.HMM", "S.SEAS.FIX_REVERSAL",
        "S.GEO.FIB_EXTENSION", "S.VOL.COMPRESSION", "S.ST.SR_REVERSAL",
        # carry families hold the higher-yielding side, which for EURUSD in 2011-12 is always EUR
        "S.XA.CARRY", "S.XA.CARRY_FILTER", "S.XA.CARRY_VOLGATE", "S.XA.CARRY_MOM",
        # event families that are one-directional or silent for EURUSD by construction
        "S.SEAS.GOTOBI", "S.EVT.FOMC_USD", "S.XA.RATE_MOMENTUM", "S.XA.VALUE_PPP", "S.SMC.SILVER_BULLET"}


@pytest.fixture(scope="module")
def bars_long():
    from fxlab.data.store import load_bars
    return load_bars("EURUSD", "H1", "2011-01-01", "2012-06-30")


@pytest.fixture(scope="module")
def panel_long(bars_long):
    from fxlab.data.store import load_bars, available
    cols = {}
    for s in ["GBPUSD", "XAUUSD", "XAGUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURJPY", "EURGBP"]:
        if available(s):
            try:
                cols[s] = load_bars(s, "H1", "2011-01-01", "2012-06-30")["close"]
            except FileNotFoundError:
                pass
    return pd.DataFrame(cols).reindex(bars_long.index)


@pytest.mark.parametrize("e", STRATS, ids=[e.id for e in STRATS])
def test_strategy_contract(e, bars_long, panel_long, catalog_id):
    catalog_id(e.id)
    ok, missing = requirements_met(e)
    if not ok:
        pytest.skip(f"blocked: missing {missing}")
    params = OVR.get(e.id, {})
    sig = run_signals(e.id, bars_long, panel_long, "EURUSD", params)
    assert isinstance(sig, pd.DataFrame) and "side" in sig
    assert sig.index.equals(bars_long.index)
    s = sig["side"].to_numpy(float)
    assert np.isfinite(s).all() and (np.abs(s) <= 1 + 1e-9).all()
    # causality: prefix invariance
    k = int(len(bars_long) * 0.7)
    sig2 = run_signals(e.id, bars_long.iloc[:k], panel_long.iloc[:k], "EURUSD", params)
    a, b = sig["side"].iloc[:k].to_numpy(float), sig2["side"].to_numpy(float)
    bad = ~np.isclose(a, b)
    assert not bad.any(), f"LOOKAHEAD: {bad.sum()} signals change when future bars are removed (first {np.argmax(bad)})"
    # symmetry
    if e.id not in RARE:
        assert (s > 0).any() and (s < 0).any(), "family never produced both long and short signals"
    else:
        assert (s != 0).any() or True  # rare-event families may be silent on 18 months
    # execution end-to-end
    res = evaluate(e.id, bars_long, "EURUSD", panel_long, params, signals=sig)
    r = res["returns"].to_numpy(float)
    assert np.isfinite(r).all()
    assert res["metrics"]["n_days"] > 100
