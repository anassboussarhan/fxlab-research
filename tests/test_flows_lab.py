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
    df = pd.DataFrame({"bid_o": b, "ask_o": b + spread}, index=idx)
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
    net, gross, d_in, d_out = bk.trade(t0, t1, +1)
    s = bk.sf * 0.0002 + bk.sp * 0.0001
    assert gross == 0.0 and d_in == 0 and d_out == 0
    assert abs(net - ((1.3000 - s) / (1.3002 + s) - 1 - 2 * bk.comm)) < 1e-15
    net_s, _, _, _ = bk.trade(t0, t1, -1)
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
