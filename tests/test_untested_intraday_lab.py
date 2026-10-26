import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402


def _m1(mid, half_spread):
    """M1 bid/ask bars from a minute mid path (open = previous close, or the bar's own close after a data gap;
    high/low = the two)."""
    mid = pd.Series(mid, dtype=float)
    cont = pd.Series(mid.index, index=mid.index).diff() == pd.Timedelta(minutes=1)
    o = mid.shift(1).where(cont, mid)
    hi, lo = np.maximum(o, mid), np.minimum(o, mid)
    d = {}
    for side, sgn in (("bid", -1), ("ask", +1)):
        d[f"{side}_o"] = o + sgn * half_spread
        d[f"{side}_h"] = hi + sgn * half_spread
        d[f"{side}_l"] = lo + sgn * half_spread
        d[f"{side}_c"] = mid + sgn * half_spread
    return pd.DataFrame(d, index=mid.index)


def _flat(start, end, level, freq="1min"):
    idx = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    return pd.Series(level, index=idx, dtype=float)


def test_orb_trailing_locks_profit_after_a_run_and_reversal():
    # New York 2024-03-12 (EDT): open 13:30 UTC; the first 30 minutes range 99.9-100.1; break up at 14:05,
    # run to 101.1 (R = about 0.2, so +5 R), then fall back to 99 -> the trailing stop exits with a profit.
    day = pd.Timestamp("2024-03-12")
    s = _flat("2024-03-12 13:00", "2024-03-12 20:30", 100.0)
    s.loc["2024-03-12 13:30":"2024-03-12 13:59"] = np.where(np.arange(30) % 2 == 0, 100.1, 99.9)
    s.loc["2024-03-12 14:00":"2024-03-12 14:04"] = 100.0
    up = np.linspace(100.15, 101.1, 60)
    s.loc["2024-03-12 14:05":"2024-03-12 15:04"] = up
    s.loc["2024-03-12 15:05":"2024-03-12 16:04"] = np.linspace(101.1, 99.0, 60)
    s.loc["2024-03-12 16:05":] = 99.0
    bk = FL.Book(_m1(s, 0.005), 0.1)
    r = U.orb_trail(bk, day, "America/New_York", (9, 30), (16, 0))
    assert r is not None
    # entry 100.15 + half spread + slippage = 100.1675, R = 0.2675; best bid 101.095 = +3.47 R -> 1 R behind is
    # +2.47 R, rounded down to 0.5 R steps = +2 R -> exit at 100.70 (about +0.53%) less slippage and commission
    assert 0.0048 < r < 0.0055


def test_orb_stop_loss_is_minus_one_r():
    day = pd.Timestamp("2024-03-12")
    s = _flat("2024-03-12 13:00", "2024-03-12 20:30", 100.0)
    s.loc["2024-03-12 13:30":"2024-03-12 13:59"] = np.where(np.arange(30) % 2 == 0, 100.1, 99.9)
    s.loc["2024-03-12 14:00":"2024-03-12 14:04"] = 100.0
    s.loc["2024-03-12 14:05"] = 100.15
    s.loc["2024-03-12 14:06":] = 99.5                         # straight through the stop at 99.9
    bk = FL.Book(_m1(s, 0.005), 0.1)
    r = U.orb_trail(bk, day, "America/New_York", (9, 30), (16, 0))
    # entry ~100.15 + costs, stop 99.9, the gap bar opens at 100.15 -> the stop fills at 99.9 minus slippage
    assert -0.0035 < r < -0.0020


def test_weekend_gap_fade_hits_friday_close():
    # 25 full weekdays of a 1.0000 +/- 0.0010 zig-zag (daily range 0.0020), then a Sunday gap to 1.0040
    # (= 2 x the range) that fills back to 1.0000 within the day -> the short fade earns about 0.4%.
    idx = pd.date_range("2024-01-01 00:00", "2024-02-02 21:00", freq="1min", tz="UTC", inclusive="left")
    idx = idx[(idx.weekday < 5)]
    zz = 1.0 + 0.0010 * np.sin(np.arange(len(idx)) * 2 * np.pi / 240)
    fri = pd.Series(zz, index=idx)
    fri.iloc[-1] = 1.0
    sun = pd.date_range("2024-02-04 22:00", "2024-02-05 23:00", freq="1min", tz="UTC", inclusive="left")
    path = np.r_[np.full(30, 1.0040), np.linspace(1.0040, 0.9990, 300), np.full(len(sun) - 330, 0.9990)]
    s = pd.concat([fri, pd.Series(path, index=sun)])
    bk = FL.Book(_m1(s, 0.00005), 1e-4)
    x = U.weekend_gap("TEST", bk)
    assert len(x) == 1
    assert 0.0030 < x.iloc[0] < 0.0041                        # target = Friday's close, less spread and costs


def test_night_scalp_short_above_band_exits_at_the_middle():
    # Tuesday night: flat 1.0000 with noise, a spike to 1.0030 on the 21:30 UTC five-minute close, back to the mean
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-03-12 18:00", "2024-03-13 02:00", freq="1min", tz="UTC", inclusive="left")
    s = pd.Series(1.0 + 0.00005 * rng.standard_normal(len(idx)), index=idx)
    s.loc["2024-03-12 21:30":"2024-03-12 21:34"] = 1.0030
    s.loc["2024-03-12 21:35":"2024-03-12 21:39"] = 1.0028
    bk = FL.Book(_m1(s, 0.00003), 1e-4)
    x = U.night_scalp("TEST", bk)
    assert len(x) >= 1
    first = x.index[0]
    assert first == pd.Timestamp("2024-03-12 21:35", tz="UTC")
    assert x.iloc[0] > 0.0015                                 # short at ~1.0028, out near 1.0000


def test_pin_bar_fade_wins_when_the_pin_fails():
    # H1 bars: a bullish pin at 10:00 (low 1.0000, open 1.0080, close 1.0090, high 1.0095); then the price falls to
    # 0.9990 -> the retail long stops out (1 pip under the low) and the fade short reaches its target.
    idx = pd.date_range("2024-03-11 00:00", "2024-03-12 12:00", freq="1min", tz="UTC", inclusive="left")
    s = pd.Series(1.0080, index=idx)
    pin = pd.date_range("2024-03-11 10:00", periods=60, freq="1min", tz="UTC")
    s.loc[pin] = np.r_[1.0080, np.linspace(1.0080, 1.0000, 20), np.linspace(1.0000, 1.0095, 30),
                       np.full(9, 1.0090)]
    after = pd.date_range("2024-03-11 11:00", periods=120, freq="1min", tz="UTC")
    s.loc[after] = np.linspace(1.0090, 0.9990, 120)
    s.loc["2024-03-11 13:00":] = 0.9990
    bk = FL.Book(_m1(s, 0.00002), 1e-4)
    res = U.candle_fade(bk)
    x = res["I8a EURUSD pin-bar fade (H1)"][0]
    assert len(x) >= 1 and x.loc[pd.Timestamp("2024-03-11 11:00", tz="UTC")] > 0.0080


def test_timed_tolerance_and_holdout_end():
    s = _flat("2024-03-12 00:00", "2024-03-13 00:00", 150.0)
    s = s[(s.index.hour != 0)]                                 # the mirror's missing first hour
    bk = FL.Book(_m1(s, 0.005), 0.01)
    d = [pd.Timestamp("2024-03-12")]
    strict = U.timed(bk, d, lambda x: (U.utc(x, 0), U.utc(x, 7)), +1)
    loose = U.timed(bk, d, lambda x: (U.utc(x, 0), U.utc(x, 7)), +1, tol_in=65)
    assert len(strict) == 0 and len(loose) == 1
    assert U.END.startswith("2025-12-31")


def test_by_period_splits_and_handles_tz():
    ix = pd.date_range("2017-01-02", "2020-12-31", freq="W-MON", tz="UTC")
    x = pd.Series(0.001, index=ix)
    x.iloc[::3] = -0.001
    bp = U.by_period(x, 52)
    assert bp["2007_2018"]["n"] + bp["2019_2025"]["n"] == bp["all"]["n"]
    assert bp["2019_2025"]["first"].startswith("2019")
