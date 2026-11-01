import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import flows_lab as FL  # noqa: E402
import lit5_lab as L  # noqa: E402


def _book(mid, start="2015-03-02 00:00", spread=0.0002, pip=1e-4, hl=0.0):
    idx = pd.date_range(start, periods=len(mid), freq="1min", tz="UTC")
    mid = np.asarray(mid, float)
    b, a = mid - spread / 2, mid + spread / 2
    m = pd.DataFrame({"bid_o": b, "bid_h": b + hl, "bid_l": b - hl, "bid_c": b,
                      "ask_o": a, "ask_h": a + hl, "ask_l": a - hl, "ask_c": a, "n_ticks": 1.0}, index=idx)
    bk = FL.Book(m, pip, None)
    bk.comm = 0.0
    bk._hl()
    bk.mc = 0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy())
    return bk


def test_profile_poc_and_value_area():
    lo = np.r_[np.full(80, 1.0), np.full(20, 0.0)]
    hi = np.r_[np.full(80, 1.1), np.full(20, 2.0)]
    vol = np.ones(100)
    poc, val, vah = L.profile(lo, hi, vol, nb=20)
    assert 1.0 <= poc <= 1.1
    assert val <= 1.0 + 1e-9 and vah >= 1.1 - 1e-9 and vah - val < 1.0


def test_bos_and_order_block_are_found_from_known_swings_only():
    # up to a swing high at bar 5, pull back with a down candle at the low (bar 9), then break above at bar 13
    c = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2.5, 3, 4, 5.5, 7, 7.5, 8, 8.5]
    o = [x - 0.2 for x in c]
    o[9] = 3.0                                             # bar 9 is a down candle (close 2.5 < open 3)
    idx = pd.date_range("2015-03-02", periods=len(c), freq="15min", tz="UTC")
    B = pd.DataFrame({"o": o, "h": [x + 0.1 for x in c], "l": [min(x, y) - 0.1 for x, y in zip(c, o)], "c": c,
                      "k": np.arange(len(c)) * 15}, index=idx)
    bos, sw = L.structure_events(B, w=3)
    ups = [e for e in bos if e["d"] == 1]
    assert ups and ups[0]["t"] == 13 and ups[0]["ob"] == 9


def test_sweep_is_a_wick_through_without_a_close():
    c = [1, 2, 3, 4, 5, 6, 5, 4, 3, 3.5, 4, 5, 5.5]
    h = [x + 0.1 for x in c]
    h[12] = 6.5                                            # takes the swing high (6.1) but closes at 5.5
    idx = pd.date_range("2015-03-02", periods=len(c), freq="15min", tz="UTC")
    B = pd.DataFrame({"o": c, "h": h, "l": [x - 0.1 for x in c], "c": c, "k": np.arange(len(c))}, index=idx)
    bos, sw = L.structure_events(B, w=3)
    assert [e["t"] for e in sw if e["d"] == 1] == [12]


def test_limit_fill_then_target():
    path = np.r_[np.full(10, 1.1050), np.linspace(1.1050, 1.1000, 20), np.linspace(1.1000, 1.1200, 100)]
    bk = _book(path)
    net, gross = L.limit_trade(bk, 0, 60, +1, 1.1010, stop=1.0990, tp=1.1060, hold=120)
    k = int(np.flatnonzero(bk.al <= 1.1010)[0])
    E = min(1.1010, bk.ao[k])                               # at the limit, or at the open when it gaps through
    assert abs(net - (1.1060 / E - 1)) < 1e-9 and net >= 1.1060 / 1.1010 - 1
    none = L.limit_trade(bk, 0, 5, +1, 1.0900, stop=1.0890, tp=1.1000, hold=10)
    assert none is None


def test_stop_in_the_fill_bar_counts_as_a_loss():
    path = np.r_[np.full(5, 1.1050), [1.1000], np.full(10, 1.0950)]
    bk = _book(path, hl=0.0060)                             # the fill bar's range covers the stop
    net, gross = L.limit_trade(bk, 0, 10, +1, 1.1010, stop=1.0995, tp=1.1100, hold=10)
    assert net < 0


def test_setup_void_when_the_stop_trades_before_a_fill_is_allowed():
    # price falls through the limit and the stop while fills are not allowed; later it is allowed and back at L
    path = np.r_[np.full(5, 1.1050), np.linspace(1.1050, 1.0950, 20), np.linspace(1.0950, 1.1010, 20), np.full(20, 1.1010)]
    bk = _book(path)
    allowed = np.zeros(len(path), bool)
    allowed[40:] = True
    assert L.limit_trade(bk, 0, 60, +1, 1.1010, stop=1.0990, tp=1.1100, hold=10, allowed=allowed) is None


def test_gap_through_the_stop_exits_at_the_open_not_the_stop():
    path = np.r_[np.full(5, 1.1050), [1.0900], np.full(5, 1.0900)]
    bk = _book(path)
    net, gross = L.limit_trade(bk, 0, 10, +1, 1.1010, stop=1.0990, tp=1.1100, hold=5)
    assert net < 0                                          # filled at the open below the stop and out at the bid
