"""Daily-bar families popular on stock indices, ETFs, commodities and crypto that the FX catalog did not have.
All decisions use the close of day t and are executed at the next open by the simulator.

S.DAILY.IBS            internal bar strength (C - L) / (H - L) reversion (Quantpedia / ETF folklore)
S.DAILY.DOUBLE_SEVEN   Connors "double 7s": buy an n-day closing low above the 200-day average, sell an n-day high
S.DAILY.CUM_RSI        Connors cumulative RSI(2): sum of the last k RSI(2) values below lo above the 200-day average
S.DAILY.THREE_DOWN     n consecutive down closes -> long, out at the first up close (optionally above the 200 SMA)
S.DAILY.HIGH52         52-week-high proximity breakout (George & Hwang), out below the m-day average
S.DAILY.FABER          Faber trend timing: long above the n-day average, flat (or short) below
S.DAILY.ABS_MOM        absolute (time-series) momentum, re-decided on each month's first close: L-day excess return > 0
S.DAILY.VOL_TARGET     long exposure min(1, target vol / 20-day realised vol) (Moreira & Muir form, daily)
S.DAILY.HALLOWEEN      "sell in May": long while the month is November-April, flat (or short) May-October
S.DAILY.OPEX           long over the week of the monthly option expiration (third Friday)
S.DAILY.VIX_SPIKE      long after the VIX closes above its trailing 1-year q-quantile, for k days (needs panel VIX)

Long-only by default where the published rule is long-only (short=0); short=1 adds the mirrored short.
Calendar families decide from the current bar's date only (no knowledge of the next bar), so a month's first
trading day is the decision day where the published rule uses the month end."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.registry import strategy
from fxlab.strategies.base import out, latch
from fxlab.features._util import sma, EPS

ST = {"mode": "state"}
D1 = "D1..D1"


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def _none(c):
    return pd.Series(False, index=c.index)


@strategy("S.DAILY.IBS", "daily", "internal bar strength IBS = (C - L) / (H - L): long when IBS < lo, out when IBS > x; "
          "short mirrored (IBS > 1 - lo, out below 1 - x) when short = 1", params={"lo": 0.2, "x": 0.7, "short": 0},
          space={"lo": [0.1, 0.2, 0.3], "x": [0.5, 0.7], "short": [0, 1]}, frequencies=D1, extra=ST)
def s_ibs(df, lo=0.2, x=0.7, short=0):
    ibs = (df["close"] - df["low"]) / (df["high"] - df["low"] + EPS)
    es = (ibs > 1 - lo) if short else _none(ibs)
    return out(latch(ibs < lo, es, ibs > x, ibs < 1 - x))


@strategy("S.DAILY.DOUBLE_SEVEN", "daily", "Connors double 7s: long when C > SMA200 and C is the lowest close of n days; "
          "exit when C is the highest close of n days; short mirrored below the SMA200 when short = 1",
          params={"n": 7, "short": 0}, space={"n": [5, 7, 10], "short": [0, 1]}, frequencies=D1, extra=ST)
def s_double7(df, n=7, short=0):
    c = df["close"]
    s200 = sma(c, 200)
    lo_n, hi_n = c.rolling(n, min_periods=n).min(), c.rolling(n, min_periods=n).max()
    el = (c > s200) & (c <= lo_n)
    es = ((c < s200) & (c >= hi_n)) if short else _none(c)
    return out(latch(el, es, c >= hi_n, c <= lo_n))


@strategy("S.DAILY.CUM_RSI", "daily", "Connors cumulative RSI: long when C > SMA200 and the sum of the last k RSI(2) "
          "values < lo; exit when RSI(2) > x; short mirrored (sum > 100 k - lo below the SMA200, exit RSI(2) < 100 - x)",
          params={"k": 2, "lo": 35, "x": 65, "short": 0},
          space={"k": [2, 3], "lo": [10, 35, 50], "x": [55, 65], "short": [0, 1]}, frequencies=D1, extra=ST)
def s_cumrsi(df, k=2, lo=35, x=65, short=0):
    c = df["close"]
    r2 = _rsi(c, 2)
    cum = r2.rolling(k, min_periods=k).sum()
    s200 = sma(c, 200)
    el = (c > s200) & (cum < lo)
    es = ((c < s200) & (cum > 100 * k - lo)) if short else _none(c)
    return out(latch(el, es, r2 > x, r2 < 100 - x))


@strategy("S.DAILY.THREE_DOWN", "daily", "n consecutive lower closes -> long, exit at the first higher close; trend "
          "filter f = 1 requires C > SMA200; short mirrored (n higher closes) when short = 1",
          params={"n": 3, "f": 1, "short": 0}, space={"n": [2, 3, 4], "f": [0, 1], "short": [0, 1]},
          frequencies=D1, extra=ST)
def s_three_down(df, n=3, f=1, short=0):
    c = df["close"]
    d = np.sign(c.diff())
    dn = (d < 0).astype(int).rolling(n, min_periods=n).sum() == n
    up = (d > 0).astype(int).rolling(n, min_periods=n).sum() == n
    s200 = sma(c, 200)
    el = dn & ((c > s200) if f else True)
    es = (up & ((c < s200) if f else True)) if short else _none(c)
    return out(latch(el, es, d > 0, d < 0))


@strategy("S.DAILY.HIGH52", "daily", "52-week-high breakout: long when C >= (1 - tol) x the highest close of 252 days; "
          "exit below SMA(m); short mirrored at the 52-week low (exit above SMA(m)) when short = 1",
          params={"tol": 0.0, "m": 50, "short": 0}, space={"tol": [0.0, 0.02, 0.05], "m": [50, 100], "short": [0, 1]},
          frequencies=D1, extra=ST)
def s_high52(df, tol=0.0, m=50, short=0):
    c = df["close"]
    hi, lo = c.rolling(252, min_periods=252).max(), c.rolling(252, min_periods=252).min()
    sm = sma(c, m)
    el = c >= (1 - tol) * hi
    es = (c <= (1 + tol) * lo) if short else _none(c)
    return out(latch(el, es, c < sm, c > sm))


@strategy("S.DAILY.FABER", "daily", "Faber timing: long when C > SMA(n), flat below (short below when short = 1)",
          params={"n": 200, "short": 0}, space={"n": [100, 150, 200, 250], "short": [0, 1]}, frequencies=D1, extra=ST)
def s_faber(df, n=200, short=0):
    c = df["close"]
    s = sma(c, n)
    side = np.where(c > s, 1.0, np.where(c < s, -1.0 if short else 0.0, 0.0))
    side = pd.Series(side, index=c.index).where(s.notna(), 0.0)
    return out(side)


def _months(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.tz_convert("UTC") if index.tz is not None else index).month.to_numpy()


def _month_start_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """First bar of each calendar month (known at that bar: its month differs from the previous bar's)."""
    i = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    m = i.to_period("M").astype(str).to_numpy()
    return np.r_[True, m[1:] != m[:-1]]


@strategy("S.DAILY.ABS_MOM", "daily", "absolute momentum: at each month's first close, long if the L-day return of the "
          "(excess-return) price is > 0, else flat (short when short = 1); held until the next month's first close",
          params={"L": 252, "short": 0}, space={"L": [63, 126, 252], "short": [0, 1]}, frequencies=D1, extra=ST)
def s_absmom(df, L=252, short=0):
    c = df["close"]
    r = c / c.shift(L) - 1
    dec = pd.Series(np.where(r > 0, 1.0, -1.0 if short else 0.0), index=c.index).where(r.notna())
    side = dec.where(_month_start_mask(c.index)).ffill().fillna(0.0)
    return out(side)


@strategy("S.DAILY.VOL_TARGET", "daily", "volatility-targeted long: exposure min(1, tv / annualised 20-day realised "
          "volatility), re-set daily", params={"tv": 0.15}, space={"tv": [0.10, 0.15, 0.20, 0.30]},
          frequencies=D1, extra=ST)
def s_voltarget(df, tv=0.15):
    c = df["close"]
    rv = np.log(c).diff().rolling(20, min_periods=20).std() * np.sqrt(252)
    w = (tv / (rv + EPS)).clip(upper=1.0)
    return out(w.fillna(0.0))


@strategy("S.DAILY.HALLOWEEN", "daily", "Halloween / sell-in-May: long while the current month is November-April, "
          "flat (short when short = 1) in May-October; exposure therefore runs from the second trading day of "
          "November to the first of May", params={"short": 0}, space={"short": [0, 1]}, frequencies=D1, extra=ST)
def s_halloween(df, short=0):
    mon = _months(df.index)
    side = np.where(np.isin(mon, [11, 12, 1, 2, 3, 4]), 1.0, -1.0 if short else 0.0)
    return out(pd.Series(side, index=df.index))


@strategy("S.DAILY.OPEX", "daily", "option-expiration week: long at the closes from the Friday before the week of the "
          "month's third Friday to that week's Thursday, i.e. held from Monday's open to the open after the expiration "
          "Friday", params={}, space={}, frequencies=D1, extra=ST)
def s_opex(df):
    idx = df.index
    d = pd.DatetimeIndex((idx.tz_convert("UTC") if idx.tz is not None else idx).tz_localize(None)).normalize()
    first = d.to_period("M").start_time
    third = first + pd.to_timedelta((4 - first.weekday) % 7, unit="D") + pd.Timedelta(days=14)
    hold = (d >= third - pd.Timedelta(days=7)) & (d <= third - pd.Timedelta(days=1))
    return out(pd.Series(np.asarray(hold, float), index=idx))


@strategy("S.DAILY.VIX_SPIKE", "daily", "fear buying: long when the VIX close is above its trailing 252-day q-quantile, "
          "held k days after the last such close (panel column VIX)", params={"q": 0.9, "k": 10},
          space={"q": [0.8, 0.9, 0.95], "k": [5, 10, 20]}, frequencies=D1, multi_asset=True, extra=ST)
def s_vix_spike(df, panel=None, symbol=None, q=0.9, k=10):
    if panel is None or "VIX" not in panel:
        return out(pd.Series(0.0, index=df.index))
    v = panel["VIX"].reindex(df.index).ffill()
    thr = v.rolling(252, min_periods=200).quantile(q).shift(1)
    hit = (v > thr).astype(float)
    side = hit.rolling(k, min_periods=1).max().fillna(0.0)
    return out(side)
