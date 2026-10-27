import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import flows_lab as FL  # noqa: E402
import cost_tuning_lab as C  # noqa: E402
from test_untested_intraday_lab import _m1, _flat  # noqa: E402


def test_orb_settings_reproduce_the_preregistered_rule():
    import untested_intraday_lab as U
    day = pd.Timestamp("2024-03-12")
    s = _flat("2024-03-12 13:00", "2024-03-12 20:30", 100.0)
    s.loc["2024-03-12 13:30":"2024-03-12 13:59"] = np.where(np.arange(30) % 2 == 0, 100.1, 99.9)
    s.loc["2024-03-12 14:00":"2024-03-12 14:04"] = 100.0
    s.loc["2024-03-12 14:05":"2024-03-12 15:04"] = np.linspace(100.15, 101.1, 60)
    s.loc["2024-03-12 15:05":"2024-03-12 16:04"] = np.linspace(101.1, 99.0, 60)
    s.loc["2024-03-12 16:05":] = 99.0
    bk = FL.Book(_m1(s, 0.005), 0.1)
    ref = U.orb_trail(bk, day, "America/New_York", (9, 30), (16, 0))
    got = C.orb_p(bk, 30, "pre", "close")
    assert len(got) == 1 and abs(got.net.iloc[0] - ref[0]) < 1e-12
    none = C.orb_p(bk, 30, "none", "close").net.iloc[0]
    assert none < got.net.iloc[0]                    # without trailing the reversal goes to the close


def test_walk_forward_uses_only_earlier_years():
    ix = lambda y: pd.date_range(f"{y}-01-05", periods=40, freq="W-MON")  # noqa: E731
    good_late = pd.concat([pd.DataFrame({"net": -0.001, "gross": 0.0}, index=ix(y)) for y in (2010, 2011)] +
                          [pd.DataFrame({"net": 0.002, "gross": 0.0}, index=ix(y)) for y in (2012, 2013)])
    good_early = pd.concat([pd.DataFrame({"net": 0.001, "gross": 0.0}, index=ix(y)) for y in (2010, 2011)] +
                           [pd.DataFrame({"net": -0.002, "gross": 0.0}, index=ix(y)) for y in (2012, 2013)])
    O, chosen, Pre, (hk, hf) = C.walk_forward({"A": good_late, "B": good_early}, 2011, "A")
    assert chosen[2011] == "B" and chosen[2012] == "B"      # B led on the earlier years
    assert abs(O[O.index.year == 2012].net.mean() + 0.002) < 1e-12   # and its later losses count out of sample
    assert hk == "A" or hk == "B"


def test_night_settings_run():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-03-11 18:00", "2024-03-15 02:00", freq="1min", tz="UTC", inclusive="left")
    s = pd.Series(1.0 + 0.0001 * np.cumsum(rng.standard_normal(len(idx))) * 0.3, index=idx)
    bk = FL.Book(_m1(s, 0.00003), 1e-4)
    a = C.night_p(bk, 21, 2.0, "mid", 0.0)
    b = C.night_p(bk, 23, 3.0, "opp", 10.0)
    assert len(a) >= len(b)
