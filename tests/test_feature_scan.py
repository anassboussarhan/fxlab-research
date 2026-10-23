"""The fast selection kernel of scripts/feature_scan.py must (1) price positions exactly like the vector
engine, (2) be causal (future values of the feature never change earlier targets) and (3) only use training
data for its choices."""
import importlib.util
import os

import numpy as np
import pandas as pd

from fxlab.execution.costs import BASE
from fxlab.execution.simulator import vector_backtest, rollover_counts
from fxlab.validation.metrics import trading_day

_spec = importlib.util.spec_from_file_location(
    "feature_scan", os.path.join(os.path.dirname(__file__), "..", "scripts", "feature_scan.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def _bars(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2006-01-02", periods=n, freq="4h", tz="UTC")
    mid = 1.2 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    sp = 1e-4 * (1 + rng.random(n))
    b = pd.DataFrame(index=idx)
    for side, sgn in (("bid", -1), ("ask", 1)):
        for k in "ohlc":
            b[f"{side}_{k}"] = mid + sgn * sp / 2
    b["close"] = mid
    return b


def _arrays(b, pip=1e-4):
    bo, ao = b["bid_o"].to_numpy(), b["ask_o"].to_numpy()
    mo = (bo + ao) / 2
    mc = ((b["bid_c"] + b["ask_c"]) / 2).to_numpy()
    r = np.r_[mo[1:], mc[-1]] / mo - 1
    sp = ao - bo
    c = (sp / 2 + BASE.slip_spread_frac * sp + BASE.slip_fixed_pips * pip) / mo + BASE.commission_bps_per_side * 1e-4
    roll = rollover_counts(b.index)
    fin = roll * BASE.swap_pips_per_night * pip / mo
    day = pd.factorize(trading_day(b.index))[0].astype(np.int64)
    return r, c, fin, roll, day


def test_kernel_costs_match_vector_engine():
    b = _bars()
    r, c, fin, roll, _ = _arrays(b)
    x = np.random.default_rng(1).normal(size=len(b))
    n = len(b)
    T, G, K = np.zeros(n), np.zeros(n), np.zeros(n)
    fs._sim(x, -0.8, 0.8, 3, r, c, fin, 0, n, T, G, K)
    bt = vector_backtest(b, pd.Series(T, index=b.index), 1e-4, BASE, roll)
    assert np.allclose(G - K, bt["net"].to_numpy(), atol=1e-12)


def test_scan_is_causal_and_train_only():
    b = _bars(9000)
    r, c, fin, roll, day = _arrays(b)
    x = np.random.default_rng(2).normal(size=len(b))
    lo, mid, hi = fs.folds(b.index, first_test="2008-01-01", train_years=1, test_months=6)
    assert len(lo) >= 3 and (mid > lo).all() and (hi > mid).all() and (mid[1:] >= hi[:-1]).all()

    def run(xx):
        thr = np.full((len(lo), len(fs.QS), 2), np.nan)
        for f in range(len(lo)):
            w = xx[lo[f]:mid[f]]
            thr[f, :, 0] = np.quantile(w, fs.QS)
            thr[f, :, 1] = np.quantile(w, 1 - fs.QS)
        return fs._scan_column(xx, r, c, fin, day, lo, mid, hi, thr, fs.HS)

    t1, ch1 = run(x)
    k = int(hi[1])          # perturb everything after the end of fold 2's test window
    x2 = x.copy()
    x2[k:] = np.random.default_rng(3).normal(size=len(x) - k) * 5
    t2, ch2 = run(x2)
    assert np.array_equal(np.nan_to_num(t1[:k], nan=-9), np.nan_to_num(t2[:k], nan=-9))
    assert np.array_equal(ch1[:2], ch2[:2])
    assert np.isnan(t1[: mid[0]]).all()     # nothing is traded before the first test window
