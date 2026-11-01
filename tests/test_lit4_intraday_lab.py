import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import flows_lab as FL  # noqa: E402
import lit4_intraday_lab as L  # noqa: E402


def _book(mid, start="2015-03-02 00:00", spread=0.2, pip=0.01, hl=0.05):
    idx = pd.date_range(start, periods=len(mid), freq="1min", tz="UTC")
    mid = np.asarray(mid, float)
    b, a = mid - spread / 2, mid + spread / 2
    m = pd.DataFrame({"bid_o": b, "bid_h": b + hl, "bid_l": b - hl, "bid_c": b,
                      "ask_o": a, "ask_h": a + hl, "ask_l": a - hl, "ask_c": a}, index=idx)
    bk = FL.Book(m, pip, None)
    bk.comm = 0.0
    return bk


def test_breakout_long_hits_target():
    # range 1000-1010 (bid), then a rally to 1050: long from ~1010.1, stop 1000 (R ~ 10), target 2.5R
    path = np.r_[np.linspace(1000, 1010, 50), np.full(10, 1005), np.linspace(1005, 1060, 200)]
    bk = _book(path)
    bk._hl()
    r = L.lb_trade(bk, 60, len(path) - 1, hi=1010.0, lo=1000.0)
    net, gross, side, rm = r
    assert side == 1
    assert 2.4 < rm <= 2.5 + 1e-9               # target reached (R measured from the filled price)
    assert net > 0 and gross > net


def test_breakout_moves_to_break_even():
    # range 1000-1005 (cap not binding): long from ~1005.15, R ~5.15, 1.5R at ~1012.9, 2.5R at ~1018 (not reached)
    up = np.linspace(1003, 1015, 100)
    down = np.linspace(1015, 990, 100)
    path = np.r_[np.full(60, 1003), up, down]
    bk = _book(path)
    bk._hl()
    net, gross, side, rm = L.lb_trade(bk, 60, len(path) - 1, hi=1005.0, lo=1000.0)
    assert side == 1 and -0.2 < rm <= 0.0


def test_breakout_short_stopped_at_cap():
    # sell triggers at ~999.9, stop is the range high 1010 but capped at 0.6% (~6.0): stopped near 1005.9
    path = np.r_[np.full(60, 1003), np.linspace(1003, 995, 20), np.linspace(995, 1012, 100)]
    bk = _book(path)
    bk._hl()
    net, gross, side, rm = L.lb_trade(bk, 60, len(path) - 1, hi=1010.0, lo=1000.0)
    assert side == -1 and -1.1 < rm < -0.95


def test_smash_signals_use_previous_days_only():
    D = pd.DataFrame({"h": [10, 10, 10, 10, 9.0], "l": [9, 9, 9, 9, 7.0], "c": [9.5, 9.5, 9.5, 9.5, 8.0]},
                     index=pd.date_range("2020-01-06", periods=5, freq="B"))
    s = L.smash_signals(D, 3)
    assert list(s) == [0, 0, 0, 0, 1]             # close 8 below the lowest low (9) of the 3 previous days


def test_atr_is_known_only_after_the_hour():
    mid = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.01, 60 * 30))
    bk = _book(mid, pip=0.01, hl=0.0)
    a = L.atr_h1(bk)
    assert a.index[0] == bk.idx[0] + pd.Timedelta(hours=1)
    assert a.dropna().index[0] >= bk.idx[0] + pd.Timedelta(hours=14)
