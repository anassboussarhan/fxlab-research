import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import ml_adaptive_barrier_lab as A  # noqa: E402
import ml_triple_barrier_lab as TB  # noqa: E402
import passive_entry_lab as P  # noqa: E402
from test_ml_adaptive_barrier_lab import _book  # noqa: E402


def _arr(bk):
    return tuple(x.astype(float) for x in (bk.bo, bk.ao, bk.bh, bk.bl, bk.ah, bk.al))


def test_atr_matches_the_triple_barrier_lab():
    rng = np.random.default_rng(2)
    n = 800
    idx = pd.date_range("2024-03-11", periods=n, freq="15min", tz="UTC")
    mid = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    sp = 0.00005
    hi, lo = mid * (1 + np.abs(rng.normal(0, 5e-4, n))), mid * (1 - np.abs(rng.normal(0, 5e-4, n)))
    bars = pd.DataFrame({"bid_o": mid - sp, "ask_o": mid + sp, "bid_h": hi - sp, "ask_h": hi + sp, "bid_l": lo - sp,
                         "ask_l": lo + sp, "bid_c": mid - sp, "ask_c": mid + sp}, index=idx)
    for c in ("n_ticks", "spread_max", "spread_min", "bid_liq", "ask_liq", "up_ticks", "down_ticks", "spread_mean"):
        bars[c] = rng.uniform(1, 2, n)
    _, atr, *_ = TB.features(bars)
    assert np.allclose(P.atr14(bars), np.asarray(atr), equal_nan=True)


def test_unfilled_when_the_ask_never_reaches_the_bid():
    bk = _book(3, 600)
    arr = list(_arr(bk))
    al = arr[5].copy()
    al[100:110] = arr[0][100] + 1e-3                 # ask low stays above the bid of minute 100
    arr[5] = al
    rc = np.zeros(len(bk.idx), np.int64)
    sw = np.zeros(len(bk.idx))
    r = P.sim_passive(*arr, 100, 110, 300, 310, 1.0, 1e9, 1e9, bk.sf, bk.sp * bk.pip, bk.comm, rc, sw, sw, 0.0, False)
    assert r[3] == -1 and np.isnan(r[0])


def test_filled_trade_saves_the_entry_cost_against_the_market_rule():
    bk = _book(4, 800)
    arr = list(_arr(bk))
    al = arr[5].copy()
    al[200] = arr[0][200] - 1e-6                      # the ask touches the bid within the entry minute
    arr[5] = al
    rc = np.zeros(len(bk.idx), np.int64)
    sw = np.zeros(len(bk.idx))
    sf, sp, comm = bk.sf, bk.sp * bk.pip, bk.comm
    net, gross, k, xf = P.sim_passive(*arr, 200, 210, 400, 410, 1.0, 1e9, 1e9, sf, sp, comm, rc, sw, sw, 0.0, False)
    assert xf == 200 and k == 400
    bo, ao = arr[0], arr[1]
    exit_px = bo[400] - (sf * (ao[400] - bo[400]) + sp)
    assert abs(net - (exit_px / bo[200] - 1 - 2 * comm)) < 1e-12
    mkt, _, _ = A.sim(*arr, 200, 400, 1.0, 1e18, 1e18, sf, sp, comm, rc, sw, sw)
    E_mkt = ao[200] + sf * (ao[200] - bo[200]) + sp
    assert abs(mkt - (exit_px / E_mkt - 1 - 2 * comm)) < 1e-12 and net > mkt


def test_stop_in_the_fill_minute_counts_but_the_target_does_not():
    bk = _book(5, 800)
    arr = list(_arr(bk))
    L = arr[0][300]
    arr[5] = arr[5].copy(); arr[5][300] = L - 1e-6        # filled in minute 300
    arr[2] = arr[2].copy(); arr[2][300] = L + 0.01         # bid high reaches a target in the fill minute
    rc = np.zeros(len(bk.idx), np.int64)
    sw = np.zeros(len(bk.idx))
    r = P.sim_passive(*arr, 300, 310, 500, 510, 1.0, 0.005, 1e9, bk.sf, bk.sp * bk.pip, bk.comm, rc, sw, sw, 0.0,
                      False)
    assert r[2] != 300                                     # the target is not taken in the fill minute
