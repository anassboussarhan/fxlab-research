"""Strategy families added after the gap review (round 3).

Event / calendar flows : S.SEAS.GOTOBI, S.EVT.FOMC_USD, S.SEAS.MONTH_END_FIX
Macro (rates)          : S.XA.RATE_MOMENTUM
Indicator systems      : S.TREND.INDICATOR_SIGNAL (MACD signal cross, Supertrend, Parabolic SAR, Ichimoku TK+cloud,
                         RSI-50, Aroon, Vortex - the variant is a walk-forward parameter)
Smart-money concepts   : S.SMC.FVG_RETRACE, S.SMC.SWEEP_REVERSAL, S.SMC.SILVER_BULLET
Retail money management: S.RETAIL.GRID, S.RETAIL.MARTINGALE, S.RETAIL.PYRAMID

Timing contract as everywhere: `side` at bar t is the position wanted after the close of t (executed at the next
open), so time-window rules test the START of the NEXT bar. Calendar facts (gotobi days, scheduled FOMC days,
month ends) are known in advance; everything else uses data up to t only (prefix-invariance tested)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import strategy
from fxlab.strategies.base import out, latch, edge, compute
from fxlab.features._util import atr, ema, sma, EPS
from fxlab.data import calendars as cal

ST = {"mode": "state"}
TR = {"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 1.5}, "target": "affine",
                                "target_params": {"a": 0.0, "b": 2.0}, "max_hold": 48}}
PANEL = dict(multi_asset=True, instruments="panel")


def _c(df):
    return df["close"]


def _bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 3:
        return pd.Timedelta(hours=1)
    return pd.Series(index[: min(len(index), 5000)]).diff().dropna().median()


def _next_start(df) -> pd.DatetimeIndex:
    return df.index + _bar_delta(df.index)


def _usd_sign(symbol: str) -> float:
    """+1 if a LONG position in `symbol` is short USD (XXXUSD, XAU/XAG), -1 for USDXXX, 0 for crosses."""
    if symbol.endswith("USD"):
        return 1.0
    if symbol.startswith("USD"):
        return -1.0
    return 0.0


# ------------------------------------------------------------------------------------------- event / calendar flows
@strategy("S.SEAS.GOTOBI", "seasonality", "gotobi-day yen selling (Japanese importers buy USD before the 09:55 JST "
          "Tokyo fix on the 5th/10th/15th/20th/25th/30th, weekend-adjusted): long XXXJPY from start JST to 09:55 "
          "JST on Tuesday-Friday gotobi days; optional short leg 09:55-12:00 JST (post-fix reversal). Other pairs "
          "flat", params={"start": 3, "short_leg": 0}, space={"start": [3, 5, 7], "short_leg": [0, 1]},
          extra=ST, frequencies="M5..H1", **PANEL)
def s_gotobi(df, panel=None, symbol="USDJPY", start=3, short_leg=0, placebo_offset=0):
    """placebo_offset (not searched): shift the gotobi calendar by n business days - robustness checks only."""
    if not symbol.endswith("JPY"):
        return out(pd.Series(0.0, index=df.index))
    nxt = _next_start(df)
    g = cal.is_gotobi(nxt, placebo_offset)
    jst = cal.local(nxt, "Asia/Tokyo")
    h = jst.hour + jst.minute / 60.0
    wd = jst.dayofweek
    ok = g & (wd >= 1) & (wd <= 4)
    fix = 9 + 55 / 60
    side = np.where(ok & (h >= start) & (h < fix), 1.0, 0.0)
    if short_leg:
        side = np.where(ok & (h >= fix) & (h < 12.0), -1.0, side)
    return out(pd.Series(side, index=df.index))


@strategy("S.EVT.FOMC_USD", "event", "pre-/post-FOMC dollar weakness (Mueller, Tahbaz-Salehi & Vedolin): on scheduled "
          "FOMC statement days hold short USD from `start` (NY hour) until `end` (NY hour; statement ~14:00 NY); "
          "XXXUSD and metals long, USDXXX short, crosses flat", params={"start": 8, "end": 17},
          space={"start": [3, 8, 12], "end": [15, 17]}, extra=ST, frequencies="M5..H4", **PANEL)
def s_fomc(df, panel=None, symbol="EURUSD", start=8, end=17):
    sg = _usd_sign(symbol)
    if sg == 0:
        return out(pd.Series(0.0, index=df.index))
    nxt = _next_start(df)
    ny = cal.local(nxt, "America/New_York")
    h = ny.hour + ny.minute / 60.0
    on = cal.is_fomc_day(nxt) & (h >= start) & (h < end)
    return out(pd.Series(np.where(on, sg, 0.0), index=df.index))


@strategy("S.SEAS.MONTH_END_FIX", "seasonality", "month-end rebalancing proxy: on the last London business day of the "
          "month, from `start` London time until the 16:00 London fix, trade AGAINST the pair's month-to-date move "
          "(portfolio rebalancing sells recent winners). Proxy only: the documented signal uses relative equity "
          "returns, which are not available here", params={"start": 12}, space={"start": [8, 12, 14]}, extra=ST,
          frequencies="M5..H1")
def s_month_end(df, start=12):
    c = _c(df)
    nxt = _next_start(df)
    ldn = cal.local(nxt, "Europe/London")
    h = ldn.hour + ldn.minute / 60.0
    last = cal.is_last_business_day(nxt)
    month = pd.Series(cal.local(df.index, "Europe/London").to_period("M"), index=df.index)
    first_close = c.groupby(month.values).transform("first")
    mtd = np.log(c / first_close)
    on = last & (h >= start) & (h < 16.0)
    return out(pd.Series(np.where(on, -np.sign(mtd.to_numpy()), 0.0), index=df.index))


# ------------------------------------------------------------------------------------------------------- macro / rates
@strategy("S.XA.RATE_MOMENTUM", "carry", "interest-rate momentum: hold the currency whose 3-month rate differential "
          "has WIDENED over the last n months (policy-divergence trend), when |change| > thr. Point-in-time monthly "
          "rates (used from the following month)", params={"n": 3, "thr": 0.0},
          space={"n": [1, 3, 6, 12], "thr": [0.0, 0.0025]}, requires=("fxlab.data.rates",), extra=ST, **PANEL)
def s_rate_mom(df, panel=None, symbol="EURUSD", n=3, thr=0.0):
    from fxlab.data.rates import rate_diff, has_rates
    if not has_rates(symbol):
        return out(pd.Series(0.0, index=df.index))
    d_now = rate_diff(symbol, df.index).to_numpy()
    d_then = rate_diff(symbol, df.index - pd.DateOffset(months=n)).to_numpy()
    ch = d_now - d_then
    side = np.where(ch > thr, 1.0, np.where(ch < -thr, -1.0, 0.0))
    return out(pd.Series(np.nan_to_num(side), index=df.index))


# ------------------------------------------------------------------------------------------------- indicator systems
INDICATORS = ["macd", "supertrend", "psar", "ichimoku", "rsi50", "aroon", "vortex"]


@strategy("S.TREND.INDICATOR_SIGNAL", "trend", "classic indicator trading systems, always in the market: MACD above/"
          "below its signal line, Supertrend direction, Parabolic SAR side, Ichimoku (Tenkan>Kijun AND close above the "
          "cloud, mirrored; flat otherwise), RSI above/below 50, Aroon oscillator sign, Vortex VI+ vs VI-",
          params={"kind": "macd"}, space={"kind": INDICATORS}, extra=ST)
def s_indicator(df, kind="macd"):
    if kind == "macd":
        s = np.sign(compute(df, "F.MO.MACD", col="hist"))
    elif kind == "supertrend":
        s = np.sign(compute(df, "F.TR.SUPERTREND", col="dir"))
    elif kind == "psar":
        s = np.sign(compute(df, "F.TR.PSAR", col="dir"))
    elif kind == "ichimoku":
        tk = compute(df, "F.TR.ICHIMOKU", col="tk")
        cp = compute(df, "F.TR.ICHIMOKU", col="cloud_pos")
        s = pd.Series(np.where((tk > 0) & (cp > 0), 1.0, np.where((tk < 0) & (cp < 0), -1.0, 0.0)), index=df.index)
    elif kind == "rsi50":
        s = np.sign(compute(df, "F.MO.RSI", col="v") - 50.0)
    elif kind == "aroon":
        s = np.sign(compute(df, "F.TR.AROON", col="osc"))
    elif kind == "vortex":
        s = np.sign(compute(df, "F.TR.VORTEX", col="diff"))
    else:
        raise ValueError(kind)
    return out(pd.Series(s, index=df.index).fillna(0.0))


# ------------------------------------------------------------------------------------------------ smart-money concepts
@njit(cache=False)
def _active_fvg(h, l, max_age):
    """Side (+1 bull / -1 bear) and midpoint of the most recent UNFILLED fair-value gap known at each bar."""
    n = len(h)
    side = np.zeros(n); mid = np.full(n, np.nan)
    s = 0.0; m = np.nan; born = -1
    for i in range(2, n):
        if l[i] > h[i - 2]:
            s = 1.0; m = (l[i] + h[i - 2]) / 2; born = i
        elif h[i] < l[i - 2]:
            s = -1.0; m = (h[i] + l[i - 2]) / 2; born = i
        elif born >= 0 and ((s > 0 and l[i] <= m) or (s < 0 and h[i] >= m)):
            born = -1                                       # filled (traded through the midpoint) on this bar
        if born >= 0 and i - born <= max_age:
            side[i] = s; mid[i] = m
    return side, mid


@strategy("S.SMC.FVG_RETRACE", "smart_money", "ICT fair-value-gap retracement: with swing structure in one direction "
          "(higher highs & lows, or lower), enter when price retraces to within `tol` ATR of the latest unfilled FVG "
          "of the same direction; ATR stop, 2R target", params={"tol": 0.25, "max_age": 30},
          space={"tol": [0.1, 0.25, 0.5], "max_age": [10, 30]}, extra=TR)
def s_fvg_retrace(df, tol=0.25, max_age=30):
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), _c(df).to_numpy(float)
    side, mid = _active_fvg(h, l, int(max_age))
    trend = compute(df, "F.ST.STRUCTURE_STATE", col="v").to_numpy()
    a = atr(df, 14).to_numpy()
    near = np.abs(c - mid) <= tol * a
    sig = np.where((side == trend) & (side != 0) & near, side, 0.0)
    return out(edge(pd.Series(np.nan_to_num(sig), index=df.index)))


@strategy("S.SMC.SWEEP_REVERSAL", "smart_money", "liquidity-sweep reversal (stop hunt): a wick through the last "
          "confirmed swing high/low that closes back inside -> trade the reversal; optional filter requiring the "
          "higher-level structure to agree", params={"k": 3, "filt": 0}, space={"k": [3, 5, 8], "filt": [0, 1]},
          extra=TR)
def s_sweep(df, k=3, filt=0):
    sw = compute(df, "F.ST.LIQ_SWEEP", col="v", k=k).fillna(0.0)
    if filt:
        st_ = compute(df, "F.ST.STRUCTURE_STATE", col="v", k=max(5, 2 * k)).fillna(0.0)
        sw = sw.where(st_ == sw, 0.0)
    return out(sw)


@strategy("S.SMC.SILVER_BULLET", "smart_money", "ICT 'silver bullet': inside the 10:00-11:00 New York window (or the "
          "03:00-04:00 London-open window), take the first fair-value gap formed in the window in its direction; ATR "
          "stop, 2R target", params={"window": "ny"}, space={"window": ["ny", "london"]}, extra=TR,
          frequencies="M1..M15")
def s_silver_bullet(df, window="ny"):
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    ny = cal.local(df.index, "America/New_York")
    hr = ny.hour.to_numpy()
    w0 = 10 if window == "ny" else 3
    inwin = hr == w0
    day = ny.normalize().to_numpy()
    sig = np.zeros(len(h))
    taken_day = None
    for i in range(2, len(h)):
        if not inwin[i]:
            continue
        if taken_day is not None and day[i] == taken_day:
            continue
        if l[i] > h[i - 2]:
            sig[i] = 1.0; taken_day = day[i]
        elif h[i] < l[i - 2]:
            sig[i] = -1.0; taken_day = day[i]
    return out(pd.Series(sig, index=df.index))


# ------------------------------------------------------------------------------------- retail money-management systems
@strategy("S.RETAIL.GRID", "retail_mm", "mean-reversion grid: hold -k/N units when price is k grid steps (step x ATR) "
          "above its n-bar EMA, +k/N below (buy dips / sell rallies in steps, take profit as price returns), capped "
          "at N units; fractional positions", params={"n": 100, "step": 1.0, "N": 4},
          space={"n": [50, 200], "step": [0.5, 1.0, 2.0], "N": [3, 5]}, extra=ST)
def s_grid(df, n=100, step=1.0, N=4):
    c = _c(df)
    k = np.round((c - ema(c, n)) / (step * atr(df, 14) + EPS))
    return out((-k.clip(-N, N) / N).fillna(0.0))


@njit(cache=False)
def _martingale(c, z, z_in, K):
    n = len(c)
    pos = np.zeros(n)
    side = 0.0; entry = 0.0; losses = 0
    for i in range(n):
        if side != 0.0:
            if (side > 0 and z[i] >= 0) or (side < 0 and z[i] <= 0):    # exit at the mean
                pnl = side * (c[i] - entry)
                losses = 0 if pnl > 0 else min(losses + 1, K)
                side = 0.0
        if side == 0.0 and z[i] == z[i]:
            if z[i] < -z_in:
                side = 1.0; entry = c[i]
            elif z[i] > z_in:
                side = -1.0; entry = c[i]
        pos[i] = side * (2.0 ** losses) / (2.0 ** K)
    return pos


@strategy("S.RETAIL.MARTINGALE", "retail_mm", "martingale mean reversion: fade z-score extremes (close vs 20-bar mean "
          "in std units) and DOUBLE the size after each losing trade (reset after a win), up to K doublings; size "
          "normalised so the maximum is 1 unit", params={"z_in": 2.0, "K": 4}, space={"z_in": [1.5, 2.0, 2.5],
                                                                                    "K": [3, 5]}, extra=ST)
def s_martingale(df, z_in=2.0, K=4):
    c = _c(df)
    z = ((c - sma(c, 20)) / (c.rolling(20, min_periods=20).std() + EPS)).to_numpy()
    return out(pd.Series(_martingale(c.to_numpy(float), z, float(z_in), int(K)), index=df.index))


@njit(cache=False)
def _pyramid(c, hi_n, lo_n, hi_x, lo_x, a, N):
    n = len(c)
    pos = np.zeros(n)
    side = 0.0; units = 0; last_add = 0.0
    for i in range(n):
        if side > 0 and c[i] < lo_x[i]:
            side = 0.0; units = 0
        elif side < 0 and c[i] > hi_x[i]:
            side = 0.0; units = 0
        if side == 0.0:
            if c[i] > hi_n[i]:
                side = 1.0; units = 1; last_add = c[i]
            elif c[i] < lo_n[i]:
                side = -1.0; units = 1; last_add = c[i]
        elif units < N and a[i] == a[i] and side * (c[i] - last_add) >= a[i]:
            units += 1; last_add = c[i]                                # add to the winner
        pos[i] = side * units / N
    return pos


@strategy("S.RETAIL.PYRAMID", "retail_mm", "pyramiding trend: Donchian(n) breakout entry with 1 unit, add 1 unit after "
          "every further +1 ATR move in favour up to N units, exit everything on the opposite Donchian(n/2) break",
          params={"n": 55, "N": 4}, space={"n": [20, 55, 100], "N": [2, 4]}, extra=ST)
def s_pyramid(df, n=55, N=4):
    c = _c(df)
    hi_n = df["high"].rolling(n, min_periods=n).max().shift(1)
    lo_n = df["low"].rolling(n, min_periods=n).min().shift(1)
    m = max(2, n // 2)
    hi_x = df["high"].rolling(m, min_periods=m).max().shift(1)
    lo_x = df["low"].rolling(m, min_periods=m).min().shift(1)
    a = atr(df, 20)
    f = lambda s: s.to_numpy(float)                                       # noqa: E731
    return out(pd.Series(_pyramid(f(c), f(hi_n), f(lo_n), f(hi_x), f(lo_x), f(a), int(N)), index=df.index))


# ------------------------------------------------------------------------------------------------- currency value (PPP)
@strategy("S.XA.VALUE_PPP", "value", "currency value (PPP reversion, Asness-Moskowitz-Pedersen / Menkhoff et al.): "
          "real exchange rate q = log(price) + log(CPI_base) - log(CPI_quote) with point-in-time CPIs (2-month "
          "publication lag; quarterly AUD/NZD 4 months); hold the currency that is cheap versus its trailing W-month "
          "average real rate when the z-score exceeds thr", params={"W": 36, "thr": 1.0},
          space={"W": [24, 36, 60], "thr": [0.5, 1.0, 1.5]}, requires=("fxlab.data.macro",), extra=ST,
          frequencies="H1..D1", **PANEL)
def s_value(df, panel=None, symbol="EURUSD", W=36, thr=1.0):
    from fxlab.data.macro import cpi_known_at, has_cpi
    if not has_cpi(symbol):
        return out(pd.Series(0.0, index=df.index))
    cpi = cpi_known_at(df.index)
    q = np.log(_c(df)) + np.log(cpi[symbol[:3]]) - np.log(cpi[symbol[3:]])
    # month-end samples of q (known at the time) define the trailing reference window
    month = q.index.to_period("M")
    qm = q.groupby(month).last()
    mu = qm.rolling(W, min_periods=max(12, W // 2)).mean().shift(1)
    sd = qm.rolling(W, min_periods=max(12, W // 2)).std().shift(1)
    z = (q - mu.reindex(month).to_numpy()) / (sd.reindex(month).to_numpy() + EPS)
    side = np.where(z < -thr, 1.0, np.where(z > thr, -1.0, 0.0))     # cheap base -> long, expensive -> short
    return out(pd.Series(np.nan_to_num(side), index=df.index))
