import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lit6_lab as L  # noqa: E402


def test_supertrend_follows_a_trend_and_flips():
    c = np.r_[np.linspace(100, 120, 60), np.linspace(120, 90, 60)]
    idx = pd.date_range("2015-03-02", periods=len(c), freq="1h", tz="UTC")
    B = pd.DataFrame({"o": c, "h": c + 0.2, "l": c - 0.2, "c": c, "k": np.arange(len(c))}, index=idx)
    d = L.supertrend_dir(B)
    assert d[40] == 1 and d[-1] == -1
    flip = int(np.flatnonzero(d[60:] == -1)[0]) + 60
    assert 60 < flip < 80                                   # flips after the top, not before


def test_seg5_selects_bars_inside_the_window():
    idx = pd.date_range("2015-03-02 00:00", periods=48, freq="5min", tz="UTC")
    B = pd.DataFrame({"h": np.arange(48.0), "l": np.arange(48.0), "c": np.arange(48.0)}, index=idx)
    t, h, lo, c = L.seg5(B, idx[2], idx[10])
    assert t[0] == idx[2] and t[-1] == idx[9] and len(c) == 8
