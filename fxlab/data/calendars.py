"""Scheduled-event calendars that are known in advance (legitimately usable at decision time).

FOMC: scheduled meeting statement days 2007-2022 from the Federal Reserve's FOMC historical pages
(federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm, fomccalendars.htm), read with the web-fetch tool on
2026-10-23. Unscheduled meetings / conference calls are EXCLUDED (not known in advance). Statement time is
taken as 14:00 New York (14:15 before 2013, inside the same hourly bar); the 2011-2012 press-conference
meetings released at 12:30 NY, which the windows below also cover.

Gotobi (Japan): days 5, 10, 15, 20, 25, 30 of each month; a gotobi falling on a Saturday or Sunday moves to the
preceding Friday (Japanese business-day convention). Japanese public holidays other than 1-3 January are not
modelled (documented limitation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FOMC_STATEMENT_DAYS = [
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
]
FOMC_DAYS = pd.DatetimeIndex(pd.to_datetime(FOMC_STATEMENT_DAYS))
FOMC_LAST_KNOWN = pd.Timestamp("2022-12-31")      # calendar coverage; later dates are simply not flagged


def local(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return index.tz_convert(tz).tz_localize(None)


def is_fomc_day(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the New York calendar date of the timestamp is a scheduled FOMC statement day."""
    d = local(index, "America/New_York").normalize()
    return np.isin(d.values.astype("datetime64[D]"), FOMC_DAYS.values.astype("datetime64[D]"))


def gotobi_dates(start: str = "2004-01-01", end: str = "2026-12-31") -> pd.DatetimeIndex:
    days = []
    for m in pd.period_range(start, end, freq="M"):
        n = m.days_in_month
        for dd in (5, 10, 15, 20, 25, 30):
            if dd > n:
                continue
            t = pd.Timestamp(year=m.year, month=m.month, day=dd)
            while t.dayofweek >= 5:            # Saturday/Sunday -> preceding Friday
                t -= pd.Timedelta(days=1)
            if not (t.month == 1 and t.day <= 3):
                days.append(t)
    return pd.DatetimeIndex(sorted(set(days)))


_GOTOBI = gotobi_dates()


def is_gotobi(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the Tokyo calendar date of the timestamp is a gotobi business day."""
    d = local(index, "Asia/Tokyo").normalize()
    return np.isin(d.values.astype("datetime64[D]"), _GOTOBI.values.astype("datetime64[D]"))


def is_last_business_day(index: pd.DatetimeIndex, tz: str = "Europe/London") -> np.ndarray:
    d = local(index, tz).normalize()
    last = (d + pd.offsets.BMonthEnd(0)).normalize()
    return (d == last).to_numpy() if hasattr(d == last, "to_numpy") else np.asarray(d == last)
