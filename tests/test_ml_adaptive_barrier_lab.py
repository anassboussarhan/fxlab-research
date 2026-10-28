import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import flows_lab as FL  # noqa: E402
import ml_adaptive_barrier_lab as A  # noqa: E402
from test_untested_intraday_lab import _m1  # noqa: E402


def _book(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-11 00:00", periods=n, freq="1min", tz="UTC")
    s = pd.Series(1.1 * np.exp(np.cumsum(rng.normal(0, 0.0003, n))), index=idx)
    bk = FL.Book(_m1(s, 0.00004), 1e-4)
    bk._hl()
    return bk


def test_numba_trade_matches_the_flows_lab_bracket():
    bk = _book()
    arr = tuple(x.astype(float) for x in (bk.bo, bk.ao, bk.bh, bk.bl, bk.ah, bk.al))
    rc = np.zeros(len(bk.idx), dtype=np.int64)
    sw = np.zeros(len(bk.idx))
    rng = np.random.default_rng(1)
    for _ in range(200):
        i = int(rng.integers(0, 2500))
        j = i + int(rng.integers(5, 300))
        side = 1 if rng.random() < 0.5 else -1
        tp, sl = float(rng.choice([0.0005, 0.001, 1e18])), float(rng.choice([0.0005, 0.002, 1e18]))
        net, gross, k = A.sim(*arr, i, j, side, tp, sl, bk.sf, bk.sp * bk.pip, bk.comm, rc, sw, sw)
        ref = bk.bracket(i, j, side, min(tp, 1e9), min(sl, 1e9))
        assert abs(net - ref[0]) < 1e-12


def test_rollover_index_counts_wednesday_three_times():
    idx = pd.date_range("2024-03-12 20:00", "2024-03-15 23:00", freq="1h", tz="UTC")   # Tue-Fri (EDT: 17:00 NY = 21:00 UTC)
    rc = A.rollover_index(idx)
    pos = {str(t): r for t, r in zip(idx, rc)}
    assert pos["2024-03-12 22:00:00+00:00"] - pos["2024-03-12 20:00:00+00:00"] == 1     # Tuesday 17:00 NY
    assert pos["2024-03-13 22:00:00+00:00"] - pos["2024-03-13 20:00:00+00:00"] == 3     # Wednesday, triple


def test_grid_has_the_pure_time_exit_and_both_sides():
    g = A.GRID
    assert len(g) == 490 and set(g[:, 0]) == {1.0, -1.0}
    assert ((~np.isfinite(g[:, 1])) & (~np.isfinite(g[:, 2]))).sum() == 10
