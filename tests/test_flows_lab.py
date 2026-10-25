"""scripts/flows_lab.py: calendars, local-time execution, costs and no look-ahead in the month-end signal."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


@pytest.fixture(scope="module")
def F():
    import flows_lab as m
    return m


def _book(F, start="2013-03-04", minutes=3 * 1440, bid=1.3000, spread=0.0002, drift=0.0):
    idx = pd.date_range(start, periods=minutes, freq="min", tz="UTC")
    b = bid + drift * np.arange(minutes)
    df = pd.DataFrame({"bid_o": b, "ask_o": b + spread, "bid_h": b, "bid_l": b, "ask_h": b + spread,
                       "ask_l": b + spread}, index=idx)
    return F.Book(df, 0.0001)


def test_gotobi_calendar_moves_weekends_and_holidays_back(F):
    g = F.gotobi_days("2013-03-01", "2013-03-31", F.jp_holidays([2013]))
    # 10th = Sunday -> Fri 8th; 20th = Vernal Equinox Day -> 19th; 31st = Sunday -> Fri 29th
    assert [d.day for d in g] == [5, 8, 15, 19, 25, 29]


def test_local_time_conversion_follows_daylight_saving(F):
    assert F.local_ts(pd.Timestamp("2013-01-15"), "Europe/London", 16, 0) == pd.Timestamp("2013-01-15 16:00", tz="UTC")
    assert F.local_ts(pd.Timestamp("2013-07-15"), "Europe/London", 16, 0) == pd.Timestamp("2013-07-15 15:00", tz="UTC")
    assert F.local_ts(pd.Timestamp("2013-07-15"), "Asia/Tokyo", 9, 55) == pd.Timestamp("2013-07-15 00:55", tz="UTC")


def test_trade_pays_spread_slippage_and_commission(F):
    bk = _book(F)
    t0, t1 = pd.Timestamp("2013-03-04 10:00", tz="UTC"), pd.Timestamp("2013-03-04 12:00", tz="UTC")
    net, gross, d_in, d_out, nights, flat, rate = bk.trade(t0, t1, +1)
    s = bk.sf * 0.0002 + bk.sp * 0.0001
    assert gross == 0.0 and d_in == 0 and d_out == 0 and nights == 0 and flat == 0.0
    assert abs(net - ((1.3000 - s) / (1.3002 + s) - 1 - 2 * bk.comm)) < 1e-15
    net_s = bk.trade(t0, t1, -1)[0]
    assert net_s < 0 and abs(net_s - net) < 1e-6                   # symmetric cost on a flat price
    assert bk.trade(t0, pd.Timestamp("2013-03-20", tz="UTC"), +1) is None     # no bar within tolerance


def test_month_end_signal_ignores_prices_after_the_previous_close(F, monkeypatch):
    """The equity signal for the fix day uses closes up to the previous business day only: changing every index
    level from the fix day on must not change any trade."""
    idx = pd.bdate_range("2012-12-01", "2013-02-28")
    rng = np.random.default_rng(0)
    eq = {c: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx) for c in ("USD", "EUR")}
    bk = _book(F, start="2013-01-28", minutes=6 * 1440, drift=1e-7)

    def run(mut):
        E = {c: s.copy() for c, s in eq.items()}
        if mut:
            for c in E:
                E[c][E[c].index >= "2013-01-31"] *= (5.0 if c == "EUR" else 0.2)
        monkeypatch.setattr(F, "equity", lambda ccy: (E[ccy], "test"))
        rows, _ = F.month_end(bk, "EURUSD", "2013-01-01", "2013-01-31")
        return [(r["test"], r["day"], round(r["net"], 12), r["signal"]) for r in rows]
    a, b = run(False), run(True)
    assert a and a == b


def test_fomc_fallback_dates_are_scheduled_wednesdays_or_tuesdays(F):
    days = [pd.Timestamp(f"{y}-{x}") for y, s in F.FOMC_FALLBACK.items() for x in s.split()]
    assert len(days) == 16 * 8 - 1                                 # 2020 has 7 scheduled statements in the list
    assert all(d.weekday() in (1, 2, 3) for d in days)


def test_rollover_count_charges_wednesday_three_times(F):
    ny = "America/New_York"
    tue_16 = pd.Timestamp("2013-03-05 16:00", tz=ny).tz_convert("UTC")
    wed_18 = pd.Timestamp("2013-03-06 18:00", tz=ny).tz_convert("UTC")
    assert F.Book.rollovers(tue_16, wed_18) == 1 + 3
    fri_16 = pd.Timestamp("2013-03-08 16:00", tz=ny).tz_convert("UTC")
    mon_09 = pd.Timestamp("2013-03-11 09:00", tz=ny).tz_convert("UTC")
    assert F.Book.rollovers(fri_16, mon_09) == 1
    at_17 = pd.Timestamp("2013-03-05 17:00", tz=ny).tz_convert("UTC")
    assert F.Book.rollovers(at_17, at_17 + pd.Timedelta(hours=23)) == 0     # entry exactly at the rollover
    assert F.Book.rollovers(at_17 - pd.Timedelta(hours=1), at_17) == 0      # exit exactly at the rollover
    assert F.Book.rollovers(at_17, at_17 + pd.Timedelta(hours=24, minutes=1)) == 3   # holds through Wednesday 17:00


def test_flat_swap_is_charged_per_night_on_either_side(F):
    bk = _book(F, start="2013-03-05", minutes=3 * 1440)
    ny = "America/New_York"
    t0 = pd.Timestamp("2013-03-05 16:00", tz=ny).tz_convert("UTC")
    t1 = pd.Timestamp("2013-03-05 18:00", tz=ny).tz_convert("UTC")
    for d in (+1, -1):
        r = bk.trade(t0, t1, d)
        assert r[4] == 1 and abs(r[5] + 0.5 * 0.0001 / 1.3001) < 1e-12


def _path_book(F, mids, spread=0.0002, hi=None, lo=None):
    idx = pd.date_range("2013-03-05 10:00", periods=len(mids), freq="min", tz="UTC")
    m = np.asarray(mids, float)
    h = m if hi is None else np.asarray(hi, float)
    l_ = m if lo is None else np.asarray(lo, float)
    df = pd.DataFrame({"bid_o": m - spread / 2, "ask_o": m + spread / 2, "bid_h": h - spread / 2,
                       "bid_l": l_ - spread / 2, "ask_h": h + spread / 2, "ask_l": l_ + spread / 2}, index=idx)
    return F.Book(df, 0.0001)


def test_bracket_target_fills_at_its_price_on_the_bid_for_longs(F):
    bk = _path_book(F, [1.3000, 1.3000, 1.3010, 1.3030, 1.3030, 1.3030])
    s0 = bk.sf * 0.0002 + bk.sp * 0.0001
    E = 1.3001 + s0
    r = bk.bracket(0, 5, +1, tp=0.0020, sl=0.0020)
    # target E + 0.0020 is reached when the bid high >= target: bar 3 (bid 1.3029 >= 1.3021 + s0)
    assert abs(r[0] - ((E + 0.0020) / E - 1 - 2 * bk.comm)) < 1e-12


def test_bracket_stop_wins_a_bar_that_touches_both_and_gaps_exit_at_the_open(F):
    # bar 1 touches both levels -> stop
    bk = _path_book(F, [1.3000, 1.3000, 1.3000], hi=[1.3000, 1.3050, 1.3000], lo=[1.3000, 1.2950, 1.3000])
    r = bk.bracket(0, 2, +1, tp=0.0020, sl=0.0020)
    s0 = bk.sf * 0.0002 + bk.sp * 0.0001
    E = 1.3001 + s0
    assert abs(r[0] - ((E - 0.0020 - s0) / E - 1 - 2 * bk.comm)) < 1e-12
    # bar 1 opens below the stop -> exit at that open (bid) minus slippage, not at the stop
    bk = _path_book(F, [1.3000, 1.2950, 1.2950])
    r = bk.bracket(0, 2, +1, tp=0.0020, sl=0.0020)
    assert abs(r[0] - ((1.2949 - s0) / E - 1 - 2 * bk.comm)) < 1e-12
    # short: the stop is a buy stop on the ask high
    bk = _path_book(F, [1.3000, 1.3000, 1.3000], hi=[1.3000, 1.3030, 1.3000])
    r = bk.bracket(0, 2, -1, tp=0.0050, sl=0.0020)
    Es = 1.2999 - s0
    assert abs(r[0] - (1 - (Es + 0.0020 + s0) / Es - 2 * bk.comm)) < 1e-12


def test_random_window_entries_stay_inside_the_window_and_are_deterministic(F):
    bk = _book(F, start="2013-03-01", minutes=40 * 1440, drift=1e-8)
    day = pd.Timestamp("2013-03-29")
    w0, w1 = pd.Timestamp("2013-03-28 16:00", tz="UTC"), pd.Timestamp("2013-03-29 15:30", tz="UTC")
    t_exit = pd.Timestamp("2013-03-29 16:00", tz="UTC")
    a = F.random_window(bk, ("X", "Xb"), "EURUSD", day, w0, w1, t_exit, +1.0)
    b = F.random_window(bk, ("X", "Xb"), "EURUSD", day, w0, w1, t_exit, +1.0)
    assert a == b and len(a) == F.SEEDS * 4
    lead_max = (t_exit - w0).total_seconds() / 60
    lead_min = (t_exit - w1).total_seconds() / 60
    assert all(lead_min < r["lead_min"] <= lead_max for r in a)
    assert {r["test"] for r in a} == {"X", "Xb", "X_control", "Xb_control"}
