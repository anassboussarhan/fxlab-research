"""7.14 Time, sessions and seasonality. Calendar facts that are known in advance
(sessions, holidays, scheduled releases, DST dates) are legitimately usable at
decision time; seasonal statistics use only past occurrences."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

from fxlab.registry import feature
from fxlab.features._util import EPS, bars_per_day

F = "time_seasonality"
SESS = {"tokyo": ("Asia/Tokyo", 9, 18), "london": ("Europe/London", 8, 17), "newyork": ("America/New_York", 8, 17)}


def _local_hours(idx: pd.DatetimeIndex, tz: str) -> np.ndarray:
    l = idx.tz_convert(tz)
    return l.hour.to_numpy() + l.minute.to_numpy() / 60.0


@feature("F.TS.HOUR_CYCLE", F, "time of day (UTC) encoded as sin/cos of 2 pi minute/1440", inputs=())
def f_hour(df):
    m = df.index.hour * 60 + df.index.minute
    return pd.DataFrame({"sin": np.sin(2 * np.pi * m / 1440), "cos": np.cos(2 * np.pi * m / 1440)}, index=df.index)


@feature("F.TS.WEEKDAY", F, "FX trading weekday (NY 17:00 rollover) as sin/cos of 2 pi d/5 plus Monday/Friday flags", inputs=())
def f_weekday(df):
    ny = df.index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    d = ny.dayofweek.to_numpy()
    return pd.DataFrame({"sin": np.sin(2 * np.pi * d / 5), "cos": np.cos(2 * np.pi * d / 5),
                         "monday": (d == 0).astype(float), "friday": (d == 4).astype(float)}, index=df.index)


@feature("F.TS.MONTH_CALENDAR", F, "month-of-year sin/cos, turn-of-month flag (last 2 or first 3 business days), "
         "quarter-end month flag", inputs=())
def f_month(df):
    idx = df.index
    mo = idx.month.to_numpy()
    bd_end = (idx + pd.offsets.BMonthEnd(0)).normalize()
    days_to_end = np.busday_count(idx.tz_localize(None).normalize().values.astype("datetime64[D]"),
                                  bd_end.tz_localize(None).values.astype("datetime64[D]"))
    bd_start = (idx - pd.offsets.BMonthBegin(1)).normalize()
    days_from_start = np.busday_count(bd_start.tz_localize(None).values.astype("datetime64[D]"),
                                      idx.tz_localize(None).normalize().values.astype("datetime64[D]"))
    tom = ((days_to_end <= 1) | (days_from_start <= 2)).astype(float)
    return pd.DataFrame({"sin": np.sin(2 * np.pi * mo / 12), "cos": np.cos(2 * np.pi * mo / 12), "turn_of_month": tom,
                         "quarter_end_month": np.isin(mo, [3, 6, 9, 12]).astype(float)}, index=idx)


@feature("F.TS.SESSION_CLOCK", F, "DST-aware session clock: membership of Tokyo 9-18 JST, London 8-17 local, New York 8-17 "
         "local; London-NY and Tokyo-London overlaps; hours elapsed since London and NY opens (clipped)", inputs=())
def f_sessions(df):
    out = {}
    hrs = {}
    for k, (tz, a, b) in SESS.items():
        h = _local_hours(df.index, tz)
        hrs[k] = h
        out[k] = ((h >= a) & (h < b)).astype(float)
    out["ldn_ny"] = out["london"] * out["newyork"]
    out["tky_ldn"] = out["tokyo"] * out["london"]
    out["since_ldn_open"] = np.clip(hrs["london"] - 8, -1, 12)
    out["since_ny_open"] = np.clip(hrs["newyork"] - 8, -1, 12)
    return pd.DataFrame(out, index=df.index)


def _session_mask(idx, which):
    if which == "asia":
        h = _local_hours(idx, "Europe/London")
        return (h >= 0) & (h < 7), "Europe/London"
    if which == "london_open":
        h = _local_hours(idx, "Europe/London")
        return (h >= 8) & (h < 9), "Europe/London"
    if which == "ny_open":
        h = _local_hours(idx, "America/New_York")
        return (h >= 8) & (h < 9), "America/New_York"
    raise ValueError(which)


@feature("F.TS.SESSION_RANGE", F, "time-window range of the current local day (asia 00-07 London time, London first hour, "
         "NY first hour); after the window closes: position of C in the range and breakout distance in range units; "
         "NaN while the window is still forming", params={"window": "asia"},
         space={"window": ["asia", "london_open", "ny_open"]}, inputs=("high", "low", "close"), frequencies="M1..H1")
def f_session_range(df, window="asia"):
    mask, tz = _session_mask(df.index, window)
    day = df.index.tz_convert(tz).tz_localize(None).normalize()
    hi = df["high"].where(mask).groupby(day).cummax()
    lo = df["low"].where(mask).groupby(day).cummin()
    # forward-fill within the day only, and only after the window ended
    hi = hi.groupby(day).ffill(); lo = lo.groupby(day).ffill()
    started = pd.Series(mask, index=df.index).groupby(day).cummax().astype(bool)
    after = started & ~mask
    rng = (hi - lo)
    pos = ((df["close"] - lo) / (rng + EPS)).where(after)
    brk = (np.maximum(df["close"] - hi, 0) - np.maximum(lo - df["close"], 0)) / (rng + EPS)
    return pd.DataFrame({"pos": pos, "breakout": brk.where(after), "range_bps": (1e4 * rng / df["close"]).where(after)})


def _holidays(years):
    out = []
    for y in years:
        e = pd.Timestamp(easter(y))
        out += [pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26), e - pd.Timedelta(days=2),
                e + pd.Timedelta(days=1), pd.Timestamp(y, 7, 4)]
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="W-THU")
        out.append(nov[3])  # US Thanksgiving
    return pd.DatetimeIndex(sorted(set(out)))


@feature("F.TS.HOLIDAYS", F, "calendar distance (days) to next and since last major FX holiday (New Year, Good Friday, Easter "
         "Monday, US Independence Day, Thanksgiving, Christmas, Boxing Day); low-liquidity flag within 1 day", inputs=())
def f_holidays(df):
    d = df.index.tz_convert("UTC").tz_localize(None).normalize()
    H = _holidays(range(d.min().year - 1, d.max().year + 2)).values
    pos = np.searchsorted(H, d.values)
    nxt = (H[np.clip(pos, 0, len(H) - 1)] - d.values) / np.timedelta64(1, "D")
    prv = (d.values - H[np.clip(pos - 1, 0, len(H) - 1)]) / np.timedelta64(1, "D")
    return pd.DataFrame({"to_next": nxt, "since_last": prv, "near": ((nxt <= 1) | (prv <= 1)).astype(float)}, index=df.index)


def _first_fridays(start, end):
    months = pd.date_range(start.replace(day=1), end + pd.offsets.MonthEnd(1), freq="MS")
    out = []
    for m in months:
        f = m + pd.offsets.Week(weekday=4) if m.dayofweek != 4 else m
        t = pd.Timestamp(f.year, f.month, f.day, 8, 30).tz_localize("America/New_York").tz_convert("UTC")
        out.append(t)
    return pd.DatetimeIndex(out)


@feature("F.TS.NFP_WINDOW", F, "scheduled US payrolls proxy: first Friday of each month 08:30 New York (rule-based "
         "approximation of the BLS calendar); flag within +-30 min and hours to the next release", inputs=())
def f_nfp(df):
    idx = df.index
    ev = _first_fridays(idx.min().tz_localize(None), idx.max().tz_localize(None))
    ev_ns = ev.as_unit("ns").asi8                 # explicit ns: M1 indexes load as ms/us under pandas >= 2
    t = idx.as_unit("ns").asi8
    pos = np.searchsorted(ev_ns, t)
    nxt = np.where(pos < len(ev_ns), (ev_ns[np.clip(pos, 0, len(ev_ns) - 1)] - t) / 3.6e12, np.nan)
    prv = np.where(pos > 0, (t - ev_ns[np.clip(pos - 1, 0, len(ev_ns) - 1)]) / 3.6e12, np.nan)
    near = ((nxt <= 0.5) | (prv <= 0.5)).astype(float)
    return pd.DataFrame({"hours_to_next": nxt, "hours_since": prv, "window": near}, index=idx)


@feature("F.TS.DST_TRANSITION", F, "DST regime: NY-London offset (normally 5h, 4h in mismatch weeks), US and UK DST flags, "
         "days since the most recent DST change in either zone", inputs=())
def f_dst(df):
    idx = df.index
    ny_off = idx.tz_convert("America/New_York").map(lambda x: x.utcoffset().total_seconds() / 3600)
    ld_off = idx.tz_convert("Europe/London").map(lambda x: x.utcoffset().total_seconds() / 3600)
    ny_off, ld_off = np.asarray(ny_off, float), np.asarray(ld_off, float)
    change = (pd.Series(ny_off).diff().fillna(0) != 0) | (pd.Series(ld_off).diff().fillna(0) != 0)
    ns = idx.as_unit("ns").asi8
    last = pd.Series(np.where(change, ns, np.nan)).ffill().to_numpy()
    since = (ns - last) / 8.64e13
    return pd.DataFrame({"ny_ldn_gap": ld_off - ny_off, "us_dst": (ny_off == -4).astype(float),
                         "uk_dst": (ld_off == 1).astype(float), "days_since_change": since}, index=idx)


def _minute_key(idx):
    return idx.hour * 60 + idx.minute


@feature("F.TS.SEASONAL_VOL", F, "intraday seasonal volatility: |l_t| divided by the mean |l| at the same UTC minute-of-day "
         "over the previous D days (current day excluded)", params={"D": 20}, space={"D": [10, 20, 60]},
         frequencies="M1..H4")
def f_seasonal_vol(df, D=20):
    a = np.log(df["close"]).diff().abs()
    key = _minute_key(df.index)
    base = a.groupby(key).transform(lambda s: s.shift(1).rolling(D, min_periods=max(3, D // 2)).mean())
    return pd.DataFrame({"ratio": a / (base + EPS), "seasonal_level": base})


@feature("F.TS.SEASONAL_RETURN", F, "intraday seasonal drift: mean l at the same minute-of-day over previous D days, as a "
         "t-statistic", params={"D": 60}, space={"D": [20, 60, 120]}, frequencies="M5..H4")
def f_seasonal_ret(df, D=60):
    l = np.log(df["close"]).diff()
    key = _minute_key(df.index)
    g = l.groupby(key)
    m = g.transform(lambda s: s.shift(1).rolling(D, min_periods=D // 2).mean())
    s = g.transform(lambda s: s.shift(1).rolling(D, min_periods=D // 2).std())
    return m / (s + EPS) * np.sqrt(D)


@feature("F.TS.FIX_WINDOW", F, "London 4pm WM/R fix: minutes to/from 16:00 London (clipped +-120), in-window flag "
         "(15:55-16:05), and last-business-day-of-month flag (month-end fix flows)", inputs=())
def f_fix(df):
    idx = df.index
    h = _local_hours(idx, "Europe/London")
    dm = np.clip((h - 16.0) * 60, -120, 120)
    lbd = (idx.tz_convert("Europe/London").tz_localize(None).normalize() ==
           (idx.tz_convert("Europe/London").tz_localize(None) + pd.offsets.BMonthEnd(0)).normalize())
    return pd.DataFrame({"min_to_fix": dm, "in_window": (np.abs(dm) <= 5).astype(float),
                         "month_end": np.asarray(lbd, float)}, index=idx)


@feature("F.TS.GOTOBI", F, "gotobi day (5/10/15/20/25/30, weekend-adjusted; Tokyo calendar) flag, and hours until the "
         "09:55 JST Tokyo fix on gotobi days (negative after the fix, NaN on other days)", inputs=())
def f_gotobi(df):
    from fxlab.data import calendars as cal
    g = cal.is_gotobi(df.index).astype(float)
    jst = cal.local(df.index, "Asia/Tokyo")
    h = jst.hour + jst.minute / 60.0
    to_fix = np.where(g > 0, (9 + 55 / 60) - h, np.nan)
    return pd.DataFrame({"flag": g, "hours_to_fix": to_fix}, index=df.index)


@feature("F.TS.FOMC", F, "scheduled FOMC statement day (New York calendar; Fed calendar 2007-2022) flag and hours to "
         "the ~14:00 NY statement on that day (NaN on other days)", inputs=())
def f_fomc(df):
    from fxlab.data import calendars as cal
    on = cal.is_fomc_day(df.index).astype(float)
    ny = cal.local(df.index, "America/New_York")
    h = ny.hour + ny.minute / 60.0
    return pd.DataFrame({"flag": on, "hours_to_statement": np.where(on > 0, 14.0 - h, np.nan)}, index=df.index)


@feature("F.TS.MONTH_END_FIX", F, "last London business day of the month flag and hours to the 16:00 London fix on "
         "that day (NaN otherwise)", inputs=())
def f_month_end_fix(df):
    from fxlab.data import calendars as cal
    on = cal.is_last_business_day(df.index).astype(float)
    ldn = cal.local(df.index, "Europe/London")
    h = ldn.hour + ldn.minute / 60.0
    return pd.DataFrame({"flag": on, "hours_to_fix": np.where(on > 0, 16.0 - h, np.nan)}, index=df.index)
