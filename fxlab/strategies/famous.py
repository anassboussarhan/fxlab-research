"""Round 4: published systems of well-known traders, popular forum systems, documented order-flow effects and
esoteric (astrological) rules - all mechanised and put through the same walk-forward / cost / multiple-testing
pipeline. Rules follow the published descriptions; simplifications are stated in each docstring.

S.FAMOUS.TURTLE            Dennis/Eckhardt Turtle rules (System 1 20/10 with skip-after-winner, System 2 55/20),
                           N = 20-bar Wilder ATR, 2N stop, pyramid +1 unit every 0.5N up to 4 units
S.FAMOUS.WILLIAMS_VBO      Larry Williams volatility breakout (day open +- k x previous day range, out at day end)
S.FAMOUS.CRABEL_NR         Toby Crabel NR4/NR7 opening-range breakout
S.FAMOUS.HOLY_GRAIL        Linda Raschke Holy Grail (ADX > 30 trend, pullback to the 20 EMA, re-entry)
S.FAMOUS.RASCHKE_8020      Linda Raschke 80-20 reversal
S.FAMOUS.CONNORS_RSI2      Larry Connors RSI(2) with 200-bar trend filter and 5-bar mean exit
S.FAMOUS.TTM_SQUEEZE       John Carter TTM Squeeze (Bollinger inside Keltner, trade the release with momentum)
S.FAMOUS.TD_SEQUENTIAL     Tom DeMark TD Sequential setup-9 exhaustion reversal
S.FAMOUS.ALLIGATOR         Bill Williams Alligator (jaw/teeth/lips smoothed and shifted) trend system
S.FORUM.THREE_DUCKS        BabyPips "3 Ducks" (price above the 60-SMA on three timeframes)
S.FORUM.NEWS_STRADDLE      NFP straddle: trade the first break of +-k ATR after 08:30 New York on payroll Fridays
S.MS.ROUND_NUMBER          Osler (NY Fed): stop-loss cascades after crossing round numbers / reversals at them
S.EVT.GOLD_FIX             gold/silver London fix window (AM 10:30, PM 15:00 London; direction walk-forward)
S.WEIRD.LUNAR              lunar phase windows (new / full moon; direction walk-forward)
S.WEIRD.MERCURY_RETRO      Mercury-retrograde folklore: trend-follow only outside (or only inside) retrograde"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import strategy
from fxlab.strategies.base import out, latch, edge, compute
from fxlab.features._util import atr, ema, sma, rma, EPS
from fxlab.data import calendars as cal

ST = {"mode": "state"}
TR = {"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 1.5}, "target": "affine",
                                "target_params": {"a": 0.0, "b": 2.0}, "max_hold": 24}}


def _c(df):
    return df["close"]


def _fx_day(index: pd.DatetimeIndex) -> np.ndarray:
    """FX trading day (17:00 New York rollover) as integer day numbers."""
    ny = index.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    return ny.normalize().values.astype("datetime64[D]").astype(np.int64)


def _day_frame(df):
    """Per bar: current day's open and the previous complete day's high/low/range (all known at the bar)."""
    d = _fx_day(df.index)
    g = pd.Series(d, index=df.index)
    day_open = df["open"].groupby(d).transform("first") if "open" in df else _c(df).groupby(d).transform("first")
    agg = pd.DataFrame({"h": df["high"].groupby(d).max(), "l": df["low"].groupby(d).min()})
    prev = agg.shift(1)
    ph = g.map(prev["h"]).to_numpy()
    pl = g.map(prev["l"]).to_numpy()
    return d, day_open.to_numpy(), ph, pl


# ----------------------------------------------------------------------------------------------------- Turtle rules
@njit(cache=False)
def _turtle(c, h, l, hi_e, lo_e, hi_x, lo_x, N, skip_after_win):
    n = len(c)
    pos = np.zeros(n)
    side = 0.0; units = 0; last_add = 0.0; stop = 0.0
    # hypothetical breakout trades for the System-1 skip rule
    h_side = 0.0; h_entry = 0.0; h_stop = 0.0; last_was_win = False
    for i in range(n):
        if N[i] != N[i] or hi_e[i] != hi_e[i]:
            pos[i] = side * units / 4.0
            continue
        # --- hypothetical (every breakout) trade bookkeeping
        if h_side != 0.0:
            exit_h = (h_side > 0 and (c[i] < lo_x[i] or c[i] <= h_stop)) or \
                     (h_side < 0 and (c[i] > hi_x[i] or c[i] >= h_stop))
            if exit_h:
                last_was_win = h_side * (c[i] - h_entry) > 0
                h_side = 0.0
        # --- live position: exits
        if side > 0 and (c[i] < lo_x[i] or c[i] <= stop):
            side = 0.0; units = 0
        elif side < 0 and (c[i] > hi_x[i] or c[i] >= stop):
            side = 0.0; units = 0
        # --- breakouts
        brk = 1.0 if c[i] > hi_e[i] else (-1.0 if c[i] < lo_e[i] else 0.0)
        if brk != 0.0 and h_side == 0.0:
            take = not (skip_after_win and last_was_win)
            h_side = brk; h_entry = c[i]; h_stop = c[i] - brk * 2.0 * N[i]
            if side == 0.0 and take:
                side = brk; units = 1; last_add = c[i]; stop = c[i] - side * 2.0 * N[i]
        # --- pyramiding: +1 unit every 0.5N in favour, stops raised to 2N below the last unit
        if side != 0.0 and units < 4 and side * (c[i] - last_add) >= 0.5 * N[i]:
            units += 1; last_add = c[i]; stop = c[i] - side * 2.0 * N[i]
        pos[i] = side * units / 4.0
    return pos


@strategy("S.FAMOUS.TURTLE", "famous", "Turtle trading rules (Dennis/Eckhardt, as published by C. Faith): System 1 = "
          "20-bar breakout entry, 10-bar exit, skip a breakout if the previous breakout trade was a winner; System 2 "
          "= 55/20 without skip; N = 20-bar Wilder ATR; 2N stop; add a unit every 0.5N up to 4 units; position = "
          "units/4. Simplification: stops/exits evaluated on closes", params={"system": 1},
          space={"system": [1, 2]}, extra=ST, frequencies="H1..D1")
def s_turtle(df, system=1):
    ne, nx = (20, 10) if system == 1 else (55, 20)
    f = lambda s: s.to_numpy(float)                                          # noqa: E731
    hi_e = df["high"].rolling(ne, min_periods=ne).max().shift(1)
    lo_e = df["low"].rolling(ne, min_periods=ne).min().shift(1)
    hi_x = df["high"].rolling(nx, min_periods=nx).max().shift(1)
    lo_x = df["low"].rolling(nx, min_periods=nx).min().shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - _c(df).shift()).abs(), (df["low"] - _c(df).shift()).abs()],
                   axis=1).max(axis=1)
    N = rma(tr, 20)
    p = _turtle(f(_c(df)), f(df["high"]), f(df["low"]), f(hi_e), f(lo_e), f(hi_x), f(lo_x), f(N), system == 1)
    return out(pd.Series(p, index=df.index))


# ------------------------------------------------------------------------------------------ day-structure breakouts
@strategy("S.FAMOUS.WILLIAMS_VBO", "famous", "Larry Williams volatility breakout: long once price closes above the "
          "day's open + k x previous day's range, short below open - k x range; flat at the end of the FX day",
          params={"k": 0.5}, space={"k": [0.3, 0.5, 0.7, 1.0]}, extra=ST, frequencies="M5..H4")
def s_williams_vbo(df, k=0.5):
    d, o, ph, pl = _day_frame(df)
    c = _c(df).to_numpy(float)
    r = ph - pl
    return out(pd.Series(_day_breakout(d, c, o + k * r, o - k * r, np.ones(len(c), np.bool_)), index=df.index))


@njit(cache=False)
def _day_breakout(d, c, up, dn, allowed):
    """First close beyond up/dn in a day sets the side for the rest of that day; one trade per day."""
    n = len(c)
    pos = np.zeros(n)
    side = 0.0; done = False
    for i in range(n):
        if i == 0 or d[i] != d[i - 1]:
            side = 0.0; done = False
        nxt_same_day = i + 1 < n and d[i + 1] == d[i]
        if not done and allowed[i] and up[i] == up[i]:
            if c[i] > up[i]:
                side = 1.0; done = True
            elif c[i] < dn[i]:
                side = -1.0; done = True
        pos[i] = side if nxt_same_day else 0.0          # never carry into the next FX day
    return pos


@strategy("S.FAMOUS.CRABEL_NR", "famous", "Toby Crabel narrow-range breakout: after an NR-k day (smallest daily range of "
          "the last k days), trade the first close beyond the day's open +- x x previous range; out at day end",
          params={"k": 7, "x": 0.3}, space={"k": [4, 7], "x": [0.2, 0.3, 0.5]}, extra=ST, frequencies="M5..H4")
def s_crabel(df, k=7, x=0.3):
    d, o, ph, pl = _day_frame(df)
    c = _c(df).to_numpy(float)
    day_rng = pd.Series(ph - pl, index=df.index).groupby(d).first()
    is_nr = (day_rng <= day_rng.rolling(k, min_periods=k).min()).reindex(d).to_numpy()  # previous day was NR-k
    allowed = np.asarray(is_nr, dtype=np.bool_)
    r = ph - pl
    return out(pd.Series(_day_breakout(d, c, o + x * r, o - x * r, allowed), index=df.index))


# --------------------------------------------------------------------------------------------------- Linda Raschke
@strategy("S.FAMOUS.HOLY_GRAIL", "famous", "Linda Raschke 'Holy Grail': with ADX(14) > thr and +DI > -DI (uptrend), "
          "enter long when price pulls back to the 20 EMA and closes back above it (mirrored for downtrends); ATR "
          "stop, 2R target", params={"thr": 30}, space={"thr": [25, 30, 35]}, extra=TR)
def s_holy_grail(df, thr=30):
    a = compute(df, "F.TR.ADX", col="adx")
    pdi = compute(df, "F.TR.ADX", col="pdi")
    mdi = compute(df, "F.TR.ADX", col="mdi")
    e = ema(_c(df), 20)
    up = (a > thr) & (pdi > mdi) & (df["low"] <= e) & (_c(df) > e)
    dn = (a > thr) & (mdi > pdi) & (df["high"] >= e) & (_c(df) < e)
    return out(pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=df.index))


@strategy("S.FAMOUS.RASCHKE_8020", "famous", "Linda Raschke 80-20: a bar that opens in the top 20% of its range and "
          "closes in the bottom 20%, followed by a bar that trades below its low and closes back above it -> long "
          "reversal (mirrored); ATR stop, 2R target", params={"q": 0.2}, space={"q": [0.2, 0.25]}, extra=TR)
def s_8020(df, q=0.2):
    h, l, c = df["high"], df["low"], _c(df)
    o = df["open"] if "open" in df else c.shift(1)
    r = (h - l) + EPS
    down_bar = ((o - l) / r >= 1 - q) & ((c - l) / r <= q)
    up_bar = ((o - l) / r <= q) & ((c - l) / r >= 1 - q)
    lng = down_bar.shift(1, fill_value=False) & (l < l.shift(1)) & (c > l.shift(1))
    sht = up_bar.shift(1, fill_value=False) & (h > h.shift(1)) & (c < h.shift(1))
    return out(pd.Series(np.where(lng, 1.0, np.where(sht, -1.0, 0.0)), index=df.index))


# ---------------------------------------------------------------------------------------------- Connors / Carter / DeMark
@strategy("S.FAMOUS.CONNORS_RSI2", "famous", "Larry Connors RSI(2): long when close > 200-bar SMA and RSI(2) < lo, exit "
          "when close > 5-bar SMA; short mirrored (RSI(2) > 100 - lo below the 200 SMA, exit below the 5 SMA)",
          params={"lo": 10}, space={"lo": [5, 10, 20]}, extra=ST)
def s_connors(df, lo=10):
    c = _c(df)
    r2 = compute(df, "F.MO.RSI", col="v", n=2)
    s200, s5 = sma(c, 200), sma(c, 5)
    return out(latch((c > s200) & (r2 < lo), (c < s200) & (r2 > 100 - lo), c > s5, c < s5))


@strategy("S.FAMOUS.TTM_SQUEEZE", "famous", "John Carter TTM Squeeze: squeeze ON while Bollinger(20, 2) lies inside "
          "Keltner(20, k ATR); when it turns OFF after >= m squeeze bars, enter with the sign of the momentum "
          "(20-bar linear-regression value of close minus the Donchian/SMA midline); exit when momentum falls for 2 "
          "bars", params={"k": 1.5, "m": 6}, space={"k": [1.5, 2.0], "m": [3, 6, 12]}, extra=ST)
def s_ttm(df, k=1.5, m=6):
    c = _c(df)
    mid = sma(c, 20)
    sd = c.rolling(20, min_periods=20).std()
    a = atr(df, 20)
    on = ((mid + 2 * sd) < (mid + k * a)) & ((mid - 2 * sd) > (mid - k * a))
    run = on.astype(int).groupby((~on).cumsum()).cumsum()
    fired = (~on) & (run.shift(1) >= m)
    base = (df["high"].rolling(20, min_periods=20).max() + df["low"].rolling(20, min_periods=20).min()) / 2
    x = c - (base + mid) / 2
    xs = np.arange(20, dtype=float) - 9.5
    mom = x.rolling(20, min_periods=20).apply(lambda w: float(np.dot(xs, w) / np.dot(xs, xs) * 9.5 + w.mean()),
                                              raw=True)
    falling = (mom < mom.shift(1)) & (mom.shift(1) < mom.shift(2))
    rising = (mom > mom.shift(1)) & (mom.shift(1) > mom.shift(2))
    return out(latch(fired & (mom > 0), fired & (mom < 0), falling, rising))


@strategy("S.FAMOUS.TD_SEQUENTIAL", "famous", "Tom DeMark TD Sequential setup: 9 consecutive closes below the close 4 "
          "bars earlier = buy setup (exhaustion) -> long; 9 above -> short; ATR stop, 2R target",
          params={"n": 9}, space={"n": [9, 13]}, extra=TR)
def s_td(df, n=9):
    c = _c(df)
    dn = (c < c.shift(4)).astype(int)
    up = (c > c.shift(4)).astype(int)
    cnt_dn = dn.groupby((dn == 0).cumsum()).cumsum()
    cnt_up = up.groupby((up == 0).cumsum()).cumsum()
    return out(pd.Series(np.where(cnt_dn == n, 1.0, np.where(cnt_up == n, -1.0, 0.0)), index=df.index))


@strategy("S.FAMOUS.ALLIGATOR", "famous", "Bill Williams Alligator: jaw = SMMA13 shifted 8 bars, teeth = SMMA8 shifted "
          "5, lips = SMMA5 shifted 3 (shifts are lags, hence causal); long while lips > teeth > jaw and close > lips "
          "(the alligator 'eating' up), short mirrored, flat otherwise", params={"scale": 1},
          space={"scale": [1, 2, 3]}, extra=ST)
def s_alligator(df, scale=1):
    med = (df["high"] + df["low"]) / 2
    jaw = rma(med, 13 * scale).shift(8 * scale)
    teeth = rma(med, 8 * scale).shift(5 * scale)
    lips = rma(med, 5 * scale).shift(3 * scale)
    c = _c(df)
    lng = (lips > teeth) & (teeth > jaw) & (c > lips)
    sht = (lips < teeth) & (teeth < jaw) & (c < lips)
    return out(pd.Series(np.where(lng, 1.0, np.where(sht, -1.0, 0.0)), index=df.index))


# --------------------------------------------------------------------------------------------------------- forums
@strategy("S.FORUM.THREE_DUCKS", "forum", "BabyPips '3 Ducks': long only when price is above the 60-period SMA on three "
          "timeframes (approximated on one bar series by SMAs of 60 x {mult^2, mult, 1} bars), short when below all "
          "three; exit when price crosses the base 60-SMA", params={"mult": 4}, space={"mult": [3, 4]}, extra=ST)
def s_three_ducks(df, mult=4):
    c = _c(df)
    s1, s2, s3 = sma(c, 60), sma(c, 60 * mult), sma(c, 60 * mult * mult)
    return out(latch((c > s1) & (c > s2) & (c > s3), (c < s1) & (c < s2) & (c < s3), c < s1, c > s1))


def _first_fridays_ny(index: pd.DatetimeIndex) -> np.ndarray:
    ny = index.tz_convert("America/New_York")
    return np.asarray((ny.dayofweek == 4) & (ny.day <= 7))


@strategy("S.FORUM.NEWS_STRADDLE", "forum", "news straddle on US payrolls (first-Friday rule): from 08:30 New York, the "
          "first close more than k ATR away from the pre-release close sets the side; held for `hold` hours",
          params={"k": 0.5, "hold": 2}, space={"k": [0.3, 0.5, 1.0], "hold": [1, 2, 4]}, extra=ST,
          frequencies="M5..H1")
def s_straddle(df, k=0.5, hold=2):
    c = _c(df).to_numpy(float)
    a = atr(df, 14).to_numpy(float)
    ny = df.index.tz_convert("America/New_York")
    hr = (ny.hour + ny.minute / 60.0).to_numpy()
    nfp = _first_fridays_ny(df.index)
    pos = np.zeros(len(c))
    ref = np.nan; side = 0.0; until = -1.0; today = None
    days = ny.normalize()
    for i in range(len(c)):
        if days[i] != today:
            today = days[i]; ref = np.nan; side = 0.0
        if not nfp[i]:
            continue
        if hr[i] < 8.5:
            ref = c[i]                                    # last close before the release
        elif ref == ref and side == 0.0 and hr[i] < 8.5 + hold and a[i] == a[i]:
            if c[i] > ref + k * a[i]:
                side = 1.0; until = 8.5 + hold
            elif c[i] < ref - k * a[i]:
                side = -1.0; until = 8.5 + hold
        if side != 0.0 and hr[i] < until:
            pos[i] = side
    return out(pd.Series(pos, index=df.index))


# --------------------------------------------------------------------------------------------- order-flow effects
def _round_step(symbol: str) -> float:
    if symbol.startswith("XAU"):
        return 10.0
    if symbol.startswith("XAG"):
        return 0.5
    return 1.0 if symbol.endswith("JPY") else 0.01


@strategy("S.MS.ROUND_NUMBER", "microstructure", "Osler (NY Fed SR150): stop-loss orders cluster just beyond round "
          "numbers (00/50), take-profits exactly at them. 'break': after a close crosses a round level, follow the "
          "move for h bars (stop cascade); 'fade': when price comes within d x ATR of a round level without crossing, "
          "trade away from it for h bars (take-profit wall)", params={"mode": "break", "h": 2, "level": "00"},
          space={"mode": ["break", "fade"], "h": [1, 2, 4], "level": ["00", "50"]}, extra=ST, frequencies="M5..H1",
          multi_asset=True, instruments="panel")
def s_round(df, panel=None, symbol="EURUSD", mode="break", h=2, level="00"):
    step = _round_step(symbol) * (0.5 if level == "50" else 1.0)
    c = _c(df)
    prev = c.shift(1)
    lvl_prev = np.floor(prev / step)
    lvl_now = np.floor(c / step)
    if mode == "break":
        trig = np.sign(lvl_now - lvl_prev).fillna(0.0)
    else:
        a = atr(df, 14)
        above = np.ceil(c / step) * step - c          # distance to the next level up
        below = c - np.floor(c / step) * step
        trig = pd.Series(np.where((above < 0.25 * a) & (lvl_now == lvl_prev) & (c > prev), -1.0,
                                  np.where((below < 0.25 * a) & (lvl_now == lvl_prev) & (c < prev), 1.0, 0.0)),
                         index=df.index)
    side = _hold(trig.to_numpy(float), int(h))
    return out(pd.Series(side, index=df.index))


@njit(cache=False)
def _hold(trig, h):
    n = len(trig)
    out_ = np.zeros(n)
    s = 0.0; cnt = 0
    for i in range(n):
        if trig[i] != 0.0:
            s = trig[i]; cnt = h
        elif cnt > 0:
            cnt -= 1
            if cnt == 0:
                s = 0.0
        out_[i] = s
    return out_


@strategy("S.EVT.GOLD_FIX", "event", "London gold/silver fix window (Abrantes-Metz & Metz reported mostly downward PM-fix "
          "moves 2004-2013; disputed): hold `dir` from 30 minutes before to 30 minutes after the AM (10:30) or PM "
          "(15:00) London fix; metals only", params={"fix": "pm", "dir": -1}, space={"fix": ["am", "pm"],
                                                                                    "dir": [-1, 1]},
          extra=ST, frequencies="M5..H1", multi_asset=True, instruments="panel")
def s_gold_fix(df, panel=None, symbol="XAUUSD", fix="pm", dir=-1):
    if not symbol.startswith(("XAU", "XAG")):
        return out(pd.Series(0.0, index=df.index))
    step = pd.Series(df.index[: min(len(df), 5000)]).diff().dropna().median() if len(df) > 2 else pd.Timedelta(hours=1)
    ldn = (df.index + step).tz_convert("Europe/London")
    h = ldn.hour + ldn.minute / 60.0
    t0 = 10.5 if fix == "am" else 15.0
    on = (h >= t0 - 0.5) & (h < t0 + 0.5) & (ldn.dayofweek < 5)
    return out(pd.Series(np.where(on, float(dir), 0.0), index=df.index))


# ------------------------------------------------------------------------------------------------------- esoteric
@strategy("S.WEIRD.LUNAR", "esoteric", "lunar phase (Dichev & Janes; Yuan, Zheng & Zhu found higher stock returns around "
          "new moons): hold `dir` within +-w days of the new moon (or full moon); PyEphem, offline and known in "
          "advance", params={"phase": "new", "w": 3, "dir": 1},
          space={"phase": ["new", "full"], "w": [2, 4], "dir": [-1, 1]}, requires=("ephem",), extra=ST,
          tags=("speculative",))
def s_lunar(df, phase="new", w=3, dir=1):
    from fxlab.data.astro import moon_age
    step = pd.Series(df.index[: min(len(df), 5000)]).diff().dropna().median() if len(df) > 2 else pd.Timedelta(hours=1)
    age = moon_age(df.index + step)
    syn = 29.53
    dist = np.minimum(age, syn - age) if phase == "new" else np.abs(age - syn / 2)
    return out(pd.Series(np.where(dist <= w, float(dir), 0.0), index=df.index))


@strategy("S.WEIRD.MERCURY_RETRO", "esoteric", "Mercury-retrograde folklore ('things go wrong'): EMA(20/100) trend "
          "following only while Mercury is direct ('normal'), or only while retrograde ('inverse'); PyEphem "
          "geocentric longitude", params={"mode": "normal"}, space={"mode": ["normal", "inverse"]},
          requires=("ephem",), extra=ST, tags=("speculative",))
def s_mercury(df, mode="normal"):
    from fxlab.data.astro import mercury_retro
    step = pd.Series(df.index[: min(len(df), 5000)]).diff().dropna().median() if len(df) > 2 else pd.Timedelta(hours=1)
    retro = mercury_retro(df.index + step)
    tr = np.sign(ema(_c(df), 20) - ema(_c(df), 100)).fillna(0.0).to_numpy()
    keep = ~retro if mode == "normal" else retro
    return out(pd.Series(np.where(keep, tr, 0.0), index=df.index))
