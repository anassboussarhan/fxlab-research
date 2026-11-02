import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import flows_lab as FL  # noqa: E402
import gotobi_accuracy_lab as A  # noqa: E402


def _book(mid, spread=0.0):
    idx = pd.date_range("2015-03-02 00:00", periods=len(mid), freq="1min", tz="UTC")
    mid = np.asarray(mid, float)
    b, a = mid - spread / 2, mid + spread / 2
    m = pd.DataFrame({"bid_o": b, "bid_h": b, "bid_l": b, "bid_c": b, "ask_o": a, "ask_h": a, "ask_l": a, "ask_c": a},
                     index=idx)
    bk = FL.Book(m, 0.01, None)
    bk.comm, bk.sf, bk.sp = 0.0, 0.0, 0.0
    bk._hl()
    return bk


H = {1: 10, 3: 20, 5: 30, 7: 40, 8: 50, 9: 60}


def test_none_and_stop():
    path = np.r_[np.full(10, 100.0), np.full(10, 99.0), np.full(81, 100.5)]   # dips 1.0, recovers to +0.5
    bk = _book(path)
    r = A.sim(bk, 0, 100, 1.0, {}, H)
    assert abs(r["net"] - 0.005) < 1e-9 and abs(r["mae"] + 0.01) < 1e-9 and not r["stopped"]
    s = A.sim(bk, 0, 100, 1.0, {"sl_w": 0.5}, H)
    assert s["stopped"] and abs(s["net"] + 0.01) < 1e-9                      # gap through the stop fills at the open
    w = A.sim(bk, 0, 100, 1.0, {"sl_w": 1.5}, H)
    assert not w["stopped"] and abs(w["net"] - 0.005) < 1e-9


def test_reentry_at_entry_price_and_at_time():
    path = np.r_[np.full(10, 100.0), np.full(10, 99.0), np.full(81, 100.5)]
    bk = _book(path)
    r = A.sim(bk, 0, 100, 1.0, {"sl_w": 0.5, "re": "entry", "re_until": 9}, H)
    # stopped at 99 (-1%), re-bought at the first price >= 100 (100.5), held to 100.5 -> 0
    assert r["stopped"] and abs(r["net"] + 0.01) < 1e-9
    t = A.sim(bk, 0, 100, 1.0, {"sl_w": 0.5, "re": "time", "re_at": 8}, H)
    assert abs(t["net"] + 0.01) < 1e-9


def test_target_partial_and_window():
    path = np.r_[np.full(30, 100.0), np.full(10, 101.0), np.full(61, 99.0)]   # up 1, then down 1
    bk = _book(path)
    f = A.sim(bk, 0, 100, 1.0, {"tp_w": 1.0}, H)
    assert f["target"] and abs(f["net"] - 0.01) < 1e-9
    h = A.sim(bk, 0, 100, 1.0, {"tp_w": 1.0, "part": 0.5}, H)
    assert abs(h["net"] - 0.0) < 1e-9                                         # +1% on half, -1% on half
    late = A.sim(bk, 0, 100, 1.0, {"tp_w": 1.0, "tp_from": 8}, H)             # target active only from bar 50
    assert not late["target"] and abs(late["net"] + 0.01) < 1e-9


def test_scale_in_and_dip_entry():
    path = np.r_[np.full(10, 100.0), np.full(10, 99.0), np.full(81, 100.0)]
    bk = _book(path)
    s = A.sim(bk, 0, 100, 1.0, {"add_w": 1.0, "add_until": 8}, H)
    assert s["filled"] and abs(s["net"] - 0.5 * (100 / 99 - 1)) < 1e-9
    d = A.sim(bk, 0, 100, 1.0, {"dip_w": 1.0, "dip_until": 7}, H)
    assert abs(d["net"] - (100 / 99 - 1)) < 1e-9
    assert A.sim(bk, 0, 100, 1.0, {"dip_w": 2.0, "dip_until": 7}, H) is None


def test_stop_windows_and_close_stop():
    path = np.r_[np.full(10, 100.0), np.full(10, 98.9), np.full(81, 100.0)]  # dip in bars 10-19 (before "03:00")
    bk = _book(path)
    late = A.sim(bk, 0, 100, 1.0, {"sl_w": 1.0, "sl_from": 3}, H)
    assert not late["stopped"]
    early = A.sim(bk, 0, 100, 1.0, {"sl_w": 1.0, "sl_until": 3}, H)
    assert early["stopped"]
    c = A.sim(bk, 0, 100, 1.0, {"sl_w": 1.0, "close_n": 15}, H)                # closes below at minute 14 -> exit at 15
    assert c["stopped"] and abs(c["net"] - (98.9 / 100 - 1)) < 1e-9
