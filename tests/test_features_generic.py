"""Generic contract tests applied to EVERY registered feature on REAL EURUSD H1 data:

1. executes and returns a Series/DataFrame aligned to the input index;
2. CAUSALITY (lookahead detection): output on a prefix of the data equals the
   output on the full data for every shared row (two cut points). Any use of
   future bars changes earlier values when the future is removed and fails;
3. determinism (implied by 2: shared rows are computed twice);
4. non-degeneracy: after warm-up, a material share of finite, non-constant values;
5. no infinities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fxlab.features  # noqa: F401  (registers catalog)
from fxlab.registry import by_kind

FEATURES = sorted(by_kind("feature"), key=lambda e: e.id)

# Test-time parameter overrides: shrink windows/schedules so heavy methods are
# exercised end-to-end on ~3.6k bars. The algorithm is unchanged.
OVERRIDES = {
    "F.EC.HAR_RV": {"window_days": 40, "refit_days": 10},
    "F.EC.GARCH": {"window": 800, "refit": 400}, "F.EC.GJR_GARCH": {"window": 800, "refit": 400},
    "F.EC.EGARCH": {"window": 800, "refit": 400},
    "F.RG.HMM": {"window": 1000, "refit": 500}, "F.RG.GMM": {"window": 1000, "refit": 500},
    "F.RG.DRAWDOWN_REGIME": {"n": 1500}, "F.RG.VOL_PERCENTILE": {"n": 500}, "F.RG.LIQUIDITY_REGIME": {"n": 500},
    "F.LF.PCA_EMBED": {"W": 1000, "R": 500}, "F.LF.KERNEL_PCA": {"W": 1000, "R": 500}, "F.LF.ICA": {"W": 1000, "R": 500},
    "F.LF.DICT_LEARNING": {"W": 1000, "R": 500}, "F.LF.KMEANS_CODEBOOK": {"W": 1000, "R": 500},
    "F.LF.AUTOENCODER": {"W": 1000, "R": 500, "epochs": 5}, "F.LF.CONTRASTIVE": {"W": 1000, "R": 500, "epochs": 3},
    "F.LF.GRU_EMBED": {"W": 1000, "R": 500, "epochs": 2},
    "F.FR.RQA": {"step": 10}, "F.FR.LYAPUNOV": {"step": 10}, "F.FR.CORR_DIM": {"step": 10},
    "F.FR.SIMPLEX_SKILL": {"step": 10}, "F.IT.MULTISCALE": {"step": 10}, "F.IT.SAMPLE_ENTROPY": {"step": 5},
    "F.IT.APPROX_ENTROPY": {"step": 5}, "F.SP.EMD": {"step": 5}, "F.CA.TRANSFER_ENTROPY": {"n": 300},
    "F.CA.COINTEGRATION": {"n": 300}, "F.CA.PCA_FACTOR": {"n": 300}, "F.CA.CORR_LEVEL": {"n": 200},
    "F.CA.GRANGER": {"n": 300},
    "F.TS.SEASONAL_RETURN": {"D": 20}, "F.IN.REGIME_GATE": {"n": 500},
}
# features whose output is legitimately sparse/constant on a 7-month H1 sample
SPARSE_OK = {"F.ST.CANDLE_3BAR", "F.ST.CANDLE_2BAR", "F.ST.CANDLE_1BAR", "F.TS.HOLIDAYS", "F.TS.NFP_WINDOW",
             "F.TS.DST_TRANSITION", "F.TS.FIX_WINDOW", "F.MO.RSI_DIVERGENCE", "F.ST.FALSE_BREAKOUT",
             "F.ST.LIQ_SWEEP", "F.ST.INSIDE_OUTSIDE", "F.ES.PRIME_INTERVAL", "F.TS.MONTH_CALENDAR",
             "F.ST.NARROW_RANGE", "F.ST.FAIR_VALUE_GAP", "F.TS.SESSION_CLOCK", "F.FG.HARMONIC"}


def run(e, df, panel):
    kw = dict(e.params) | OVERRIDES.get(e.id, {})
    if e.multi_asset:
        out = e.fn(df, panel=panel, symbol="EURUSD", **kw)
    else:
        out = e.fn(df, **kw)
    if isinstance(out, pd.Series):
        out = out.to_frame("v")
    return out


@pytest.mark.parametrize("e", FEATURES, ids=[e.id for e in FEATURES])
def test_feature_contract(e, real_h1, real_panel, catalog_id):
    catalog_id(e.id)
    df = real_h1
    full = run(e, df, real_panel)
    assert isinstance(full, pd.DataFrame)
    assert full.index.equals(df.index), "output not aligned to input index"
    vals = full.to_numpy(float)
    assert not np.isinf(vals).any(), "infinite values"

    # causality / prefix invariance at two cut points
    n = len(df)
    for k in (n - 37, int(n * 0.62)):
        part = run(e, df.iloc[:k], real_panel.iloc[:k])
        a = full.iloc[:k].to_numpy(float)
        b = part.to_numpy(float)
        assert a.shape == b.shape
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-6, atol=1e-9)
        bad = ~(both_nan | close)
        assert not bad.any(), (f"LOOKAHEAD/nondeterminism: {bad.sum()} mismatching values at cut {k}; "
                               f"first bad row {np.argwhere(bad)[0][0]}")

    # non-degeneracy after warm-up
    tail = full.iloc[n // 2:]
    finite = np.isfinite(tail.to_numpy(float))
    assert finite.mean() > 0.3, f"too few finite values ({finite.mean():.2f})"
    if e.id not in SPARSE_OK and "H1" in supported(e.frequencies):
        nunique = max(tail[c].nunique() for c in tail.columns)
        assert nunique >= 2, "output constant"


ORDER = ["M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "H8", "D1"]


def supported(freq: str) -> list[str]:
    lo, hi = freq.split("..")
    return ORDER[ORDER.index(lo):ORDER.index(hi) + 1]
