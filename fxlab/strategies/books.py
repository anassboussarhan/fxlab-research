"""Round 15 book and proprietary systems not in the catalog before (docs/round15_preregistration.md, part B).

Each follows its published rule on generic bars (they are run on D1, W1 and MN). State rules give the position after
each bar's close; trade rules give entry signals with their own stop distance (price units), target and hold.

Larry Williams   : S.BOOK.WILLIAMS_OOPS, S.BOOK.WILLIAMS_SMASH, S.BOOK.WILLIAMS_R
Linda Raschke    : S.BOOK.TURTLE_SOUP, S.BOOK.RASCHKE_ANTI
Larry Connors    : S.BOOK.CONNORS_CRSI, S.BOOK.CONNORS_3DAY, S.BOOK.CONNORS_R3, S.BOOK.CONNORS_PCTB
Bill Williams    : S.BOOK.BW_AO, S.BOOK.BW_FRACTAL
Welles Wilder    : S.BOOK.WILDER_DMI, S.BOOK.WILDER_VOLSYS
Kathy Lien       : S.BOOK.LIEN_DBB, S.BOOK.LIEN_PERFECT_ORDER, S.BOOK.LIEN_INSIDE2, S.BOOK.LIEN_FADER, S.BOOK.LIEN_CHANNEL
Ichimoku         : S.BOOK.ICHIMOKU_KUMO
Charting         : S.BOOK.HEIKIN_ASHI, S.BOOK.PNF_DOUBLE, S.BOOK.DOUBLE_TOP_BOTTOM, S.BOOK.HEAD_SHOULDERS
Others           : S.BOOK.GANN_HILO, S.BOOK.ELDER_IMPULSE, S.BOOK.PRIOR_PERIOD_BO
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import strategy
from fxlab.strategies.base import out, latch
from fxlab.features._util import atr, ema, sma, rma, EPS

ST = {"mode": "state"}
TR = {"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine",
                                "target_params": {"a": 0.0, "b": 2.0}, "max_hold": 10}}
FR = "M1..D1"


def _ohlc(df):
    o = df["open"] if "open" in df else df["close"].shift(1)
    return o, df["high"], df["low"], df["close"]


def _rsi(c, n):
    d = c.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / (dn + EPS))


def _adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = rma(pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(),
                        (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1), n)
    pdi, mdi = 100 * rma(pdm, n) / (tr + EPS), 100 * rma(mdm, n) / (tr + EPS)
    return pdi, mdi, rma(100 * (pdi - mdi).abs() / (pdi + mdi + EPS), n)


def _sig(lng, sht, idx):
    return pd.Series(np.where(lng, 1.0, np.where(sht, -1.0, 0.0)), index=idx)


# ------------------------------------------------------------------------------------------------ Larry Williams
@strategy("S.BOOK.WILLIAMS_OOPS", "books", "Larry Williams 'Oops!': a bar opens below the previous low and closes back "
          "above it -> long (mirrored above the previous high); stop below the bar's low", params={"k": 1.0},
          space={"k": [1.0, 2.0]}, frequencies=FR, extra=TR)
def s_oops(df, k=1.0):
    o, h, l, c = _ohlc(df)
    a = atr(df, 14)
    lng = (o < l.shift(1)) & (c > l.shift(1))
    sht = (o > h.shift(1)) & (c < h.shift(1))
    return out(_sig(lng, sht, df.index), stop_dist=(c - l).clip(lower=0) + k * 0.1 * a)


@strategy("S.BOOK.WILLIAMS_SMASH", "books", "Larry Williams Smash Day: a down close below the previous low (smash), then a "
          "close above the smash bar's high -> long (mirrored); stop at the smash low", params={"n": 1},
          space={"n": [1, 3]}, frequencies=FR, extra=TR)
def s_smash(df, n=1):
    o, h, l, c = _ohlc(df)
    smash_dn = (c < c.shift(1)) & (c < l.shift(1).rolling(n).min())
    smash_up = (c > c.shift(1)) & (c > h.shift(1).rolling(n).max())
    lng = smash_dn.shift(1, fill_value=False) & (c > h.shift(1))
    sht = smash_up.shift(1, fill_value=False) & (c < l.shift(1))
    stop = np.where(lng, c - l.shift(1), np.where(sht, h.shift(1) - c, np.nan))
    return out(_sig(lng, sht, df.index), stop_dist=pd.Series(stop, index=df.index).clip(lower=0))


@strategy("S.BOOK.WILLIAMS_R", "books", "Williams %R(n) oversold below -80 with the close above its m-bar average -> long, "
          "out above -20 (mirrored)", params={"n": 14, "m": 100}, space={"n": [10, 14], "m": [50, 100]},
          frequencies=FR, extra=ST)
def s_willr(df, n=14, m=100):
    o, h, l, c = _ohlc(df)
    hh, ll = h.rolling(n).max(), l.rolling(n).min()
    r = -100 * (hh - c) / (hh - ll + EPS)
    s = sma(c, m)
    return out(latch((r < -80) & (c > s), (r > -20) & (c < s), r > -20, r < -80))


# ------------------------------------------------------------------------------------------------ Linda Raschke
@strategy("S.BOOK.TURTLE_SOUP", "books", "Raschke Turtle Soup: a new n-bar low at least 4 bars after the previous n-bar "
          "low that closes back above the prior n-bar low -> long (mirrored); stop below the bar's low",
          params={"n": 20}, space={"n": [20, 55]}, frequencies=FR, extra=TR)
def s_turtle_soup(df, n=20):
    o, h, l, c = _ohlc(df)
    prior_lo, prior_hi = l.shift(1).rolling(n).min(), h.shift(1).rolling(n).max()
    newlo, newhi = l < prior_lo, h > prior_hi
    age_lo = pd.Series(np.arange(len(l)), index=l.index) - pd.Series(np.where(newlo, np.arange(len(l)), np.nan), index=l.index).shift(1).ffill()
    age_hi = pd.Series(np.arange(len(h)), index=h.index) - pd.Series(np.where(newhi, np.arange(len(h)), np.nan), index=h.index).shift(1).ffill()
    lng = newlo & (age_lo >= 4) & (c > prior_lo)
    sht = newhi & (age_hi >= 4) & (c < prior_hi)
    stop = np.where(lng, c - l, np.where(sht, h - c, np.nan))
    return out(_sig(lng, sht, df.index), stop_dist=pd.Series(stop, index=df.index).clip(lower=0) + 0.1 * atr(df, 14))


@strategy("S.BOOK.RASCHKE_ANTI", "books", "Raschke 'Anti': slow %D(10) rising over 3 bars and %K(7) hooking up after "
          "dipping toward %D -> long (mirrored); ATR stop, 2R target, 10 bars", params={"k": 7, "d": 10},
          space={"k": [7], "d": [10, 14]}, frequencies=FR, extra=TR)
def s_anti(df, k=7, d=10):
    o, h, l, c = _ohlc(df)
    K = 100 * (c - l.rolling(k).min()) / (h.rolling(k).max() - l.rolling(k).min() + EPS)
    D = sma(K, d)
    lng = (D > D.shift(3)) & (K.shift(1) < K.shift(2)) & (K > K.shift(1)) & (K.shift(1) <= D.shift(1) + 5)
    sht = (D < D.shift(3)) & (K.shift(1) > K.shift(2)) & (K < K.shift(1)) & (K.shift(1) >= D.shift(1) - 5)
    return out(_sig(lng, sht, df.index))


# ------------------------------------------------------------------------------------------------ Larry Connors
def _streak(c):
    s = np.zeros(len(c))
    v = c.to_numpy(float)
    for i in range(1, len(v)):
        if v[i] > v[i - 1]:
            s[i] = s[i - 1] + 1 if s[i - 1] > 0 else 1
        elif v[i] < v[i - 1]:
            s[i] = s[i - 1] - 1 if s[i - 1] < 0 else -1
    return pd.Series(s, index=c.index)


@strategy("S.BOOK.CONNORS_CRSI", "books", "ConnorsRSI = mean(RSI(3), RSI(2) of the up/down streak, percent rank(100) of "
          "the 1-bar change); long below lo with the close above its 200-bar average, out above 70 (mirrored with "
          "short = 1)", params={"lo": 10, "short": 0}, space={"lo": [5, 10, 15], "short": [0, 1]}, frequencies=FR, extra=ST)
def s_crsi(df, lo=10, short=0):
    c = df["close"]
    roc = c.pct_change()
    pr = roc.rolling(100).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    crsi = (_rsi(c, 3) + _rsi(_streak(c), 2) + pr) / 3
    s200 = sma(c, 200)
    es = (crsi > 100 - lo) & (c < s200) if short else pd.Series(False, index=c.index)
    return out(latch((crsi < lo) & (c > s200), es, crsi > 70, crsi < 30))


@strategy("S.BOOK.CONNORS_3DAY", "books", "Connors 3-day high / low: three lower highs and lower lows with the close above "
          "the 200-bar average -> long, out above the 5-bar average (mirrored with short = 1)",
          params={"short": 0}, space={"short": [0, 1]}, frequencies=FR, extra=ST)
def s_3day(df, short=0):
    o, h, l, c = _ohlc(df)
    s200, s5 = sma(c, 200), sma(c, 5)
    dn3 = (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3)) & (l < l.shift(1)) & (l.shift(1) < l.shift(2)) & (l.shift(2) < l.shift(3))
    up3 = (h > h.shift(1)) & (h.shift(1) > h.shift(2)) & (h.shift(2) > h.shift(3)) & (l > l.shift(1)) & (l.shift(1) > l.shift(2)) & (l.shift(2) > l.shift(3))
    es = (up3 & (c < s200)) if short else pd.Series(False, index=c.index)
    return out(latch(dn3 & (c > s200), es, c > s5, c < s5))


@strategy("S.BOOK.CONNORS_R3", "books", "Connors R3: RSI(2) falls three bars in a row, the first below 60, and ends below "
          "10 with the close above the 200-bar average -> long, out when RSI(2) > 70 (mirrored with short = 1)",
          params={"short": 0}, space={"short": [0, 1]}, frequencies=FR, extra=ST)
def s_r3(df, short=0):
    c = df["close"]
    r = _rsi(c, 2)
    s200 = sma(c, 200)
    dn = (r < r.shift(1)) & (r.shift(1) < r.shift(2)) & (r.shift(2) < r.shift(3)) & (r.shift(2) < 60) & (r < 10)
    up = (r > r.shift(1)) & (r.shift(1) > r.shift(2)) & (r.shift(2) > r.shift(3)) & (r.shift(2) > 40) & (r > 90)
    es = (up & (c < s200)) if short else pd.Series(False, index=c.index)
    return out(latch(dn & (c > s200), es, r > 70, r < 30))


@strategy("S.BOOK.CONNORS_PCTB", "books", "Connors %b: Bollinger %b(20, 2) below 0.2 for three bars with the close above "
          "the 200-bar average -> long, out above 0.8 (mirrored with short = 1)", params={"short": 0},
          space={"short": [0, 1]}, frequencies=FR, extra=ST)
def s_pctb(df, short=0):
    c = df["close"]
    m, s = sma(c, 20), c.rolling(20).std()
    b = (c - (m - 2 * s)) / (4 * s + EPS)
    s200 = sma(c, 200)
    lo3 = (b < 0.2) & (b.shift(1) < 0.2) & (b.shift(2) < 0.2)
    hi3 = (b > 0.8) & (b.shift(1) > 0.8) & (b.shift(2) > 0.8)
    es = (hi3 & (c < s200)) if short else pd.Series(False, index=c.index)
    return out(latch(lo3 & (c > s200), es, b > 0.8, b < 0.2))


# ------------------------------------------------------------------------------------------------ Bill Williams
@strategy("S.BOOK.BW_AO", "books", "Bill Williams Awesome Oscillator: SMA5 - SMA34 of the median price; long above zero, "
          "short below (always in)", params={}, space={}, frequencies=FR, extra=ST)
def s_ao(df):
    o, h, l, c = _ohlc(df)
    mp = (h + l) / 2
    ao = sma(mp, 5) - sma(mp, 34)
    return out(np.sign(ao).fillna(0.0))


@njit(cache=True)
def _fractals(h, l):
    n = len(h)
    uf = np.full(n, np.nan)
    lf = np.full(n, np.nan)
    last_u, last_l = np.nan, np.nan
    for i in range(4, n):
        j = i - 2                                   # a 5-bar fractal at j is known at i = j + 2
        if h[j] > h[j - 1] and h[j] > h[j - 2] and h[j] > h[j + 1] and h[j] > h[j + 2]:
            last_u = h[j]
        if l[j] < l[j - 1] and l[j] < l[j - 2] and l[j] < l[j + 1] and l[j] < l[j + 2]:
            last_l = l[j]
        uf[i] = last_u
        lf[i] = last_l
    return uf, lf


@strategy("S.BOOK.BW_FRACTAL", "books", "Bill Williams fractal breakout: a close above the last confirmed up-fractal while "
          "above the Alligator teeth (SMMA8 of the median, shifted 5) -> long, out below the teeth (mirrored)",
          params={}, space={}, frequencies=FR, extra=ST)
def s_fractal(df):
    o, h, l, c = _ohlc(df)
    uf, lf = _fractals(h.to_numpy(float), l.to_numpy(float))
    uf, lf = pd.Series(uf, index=df.index), pd.Series(lf, index=df.index)
    teeth = rma((h + l) / 2, 8).shift(5)
    return out(latch((c > uf) & (c > teeth), (c < lf) & (c < teeth), c < teeth, c > teeth))


# ------------------------------------------------------------------------------------------------ Welles Wilder
@strategy("S.BOOK.WILDER_DMI", "books", "Wilder directional movement: +DI > -DI with ADX(14) > thr -> long, -DI > +DI with "
          "ADX > thr -> short, else flat", params={"thr": 25}, space={"thr": [20, 25, 30]}, frequencies=FR, extra=ST)
def s_dmi(df, thr=25):
    pdi, mdi, a = _adx(df, 14)
    return out(_sig((pdi > mdi) & (a > thr), (mdi > pdi) & (a > thr), df.index))


@njit(cache=True)
def _volsys(c, a, k):
    n = len(c)
    pos = np.zeros(n)
    s, ext = 0.0, np.nan
    for i in range(n):
        if not np.isfinite(a[i]):
            continue
        if s == 0.0:
            s, ext = 1.0, c[i]
        elif s > 0:
            ext = max(ext, c[i])
            if c[i] < ext - k * a[i]:
                s, ext = -1.0, c[i]
        else:
            ext = min(ext, c[i])
            if c[i] > ext + k * a[i]:
                s, ext = 1.0, c[i]
        pos[i] = s
    return pos


@strategy("S.BOOK.WILDER_VOLSYS", "books", "Wilder Volatility System: stop and reverse when the close moves k x ATR(7) "
          "against the most favourable close since entry", params={"k": 3.0}, space={"k": [2.0, 3.0, 4.0]},
          frequencies=FR, extra=ST)
def s_volsys(df, k=3.0):
    return out(pd.Series(_volsys(df["close"].to_numpy(float), atr(df, 7).to_numpy(float), float(k)), index=df.index))


# ------------------------------------------------------------------------------------------------ Kathy Lien
@strategy("S.BOOK.LIEN_DBB", "books", "Lien Double Bollinger Bands: a close above the upper 1-sd band (20) -> long while it "
          "stays in the 1-2 sd zone, out below the 1-sd band (mirrored)", params={"n": 20}, space={"n": [20, 50]},
          frequencies=FR, extra=ST)
def s_dbb(df, n=20):
    c = df["close"]
    m, s = sma(c, n), c.rolling(n).std()
    return out(latch(c > m + s, c < m - s, c < m + s, c > m - s))


@strategy("S.BOOK.LIEN_PERFECT_ORDER", "books", "Lien Perfect Order: 10 > 20 > 50 > 100 > 200-bar averages with ADX(14) > "
          "20 -> long, out when the 10 falls below the 20 (mirrored)", params={"thr": 20}, space={"thr": [20, 25]},
          frequencies=FR, extra=ST)
def s_perfect(df, thr=20):
    c = df["close"]
    m = [sma(c, n) for n in (10, 20, 50, 100, 200)]
    _, _, a = _adx(df, 14)
    up = (m[0] > m[1]) & (m[1] > m[2]) & (m[2] > m[3]) & (m[3] > m[4]) & (a > thr)
    dn = (m[0] < m[1]) & (m[1] < m[2]) & (m[2] < m[3]) & (m[3] < m[4]) & (a > thr)
    return out(latch(up, dn, m[0] < m[1], m[0] > m[1]))


@strategy("S.BOOK.LIEN_INSIDE2", "books", "Lien double inside day: two inside bars; the next close beyond the mother bar's "
          "range gives the direction; stop at the mother bar's other side, 2R target", params={}, space={},
          frequencies=FR, extra=TR)
def s_inside2(df):
    o, h, l, c = _ohlc(df)
    ins = (h < h.shift(1)) & (l > l.shift(1))
    two = ins & ins.shift(1, fill_value=False)
    mh, ml = h.shift(2), l.shift(2)
    setup = two.shift(1, fill_value=False)
    mh1, ml1 = mh.shift(1), ml.shift(1)
    lng = setup & (c > mh1)
    sht = setup & (c < ml1)
    stop = np.where(lng, c - ml1, np.where(sht, mh1 - c, np.nan))
    return out(_sig(lng, sht, df.index), stop_dist=pd.Series(stop, index=df.index).clip(lower=0))


@strategy("S.BOOK.LIEN_FADER", "books", "Lien 'Fader': with ADX(14) < 35, a bar that breaks the previous two bars' high by "
          "15% of their range and closes back below that high -> short (mirrored); stop beyond the break",
          params={}, space={}, frequencies=FR, extra=TR)
def s_fader(df):
    o, h, l, c = _ohlc(df)
    _, _, a = _adx(df, 14)
    h2, l2 = h.shift(1).rolling(2).max(), l.shift(1).rolling(2).min()
    rg = h2 - l2
    sht = (a < 35) & (h > h2 + 0.15 * rg) & (c < h2)
    lng = (a < 35) & (l < l2 - 0.15 * rg) & (c > l2)
    stop = np.where(sht, h - c, np.where(lng, c - l, np.nan))
    return out(_sig(lng, sht, df.index), stop_dist=pd.Series(stop, index=df.index).clip(lower=0) + 0.1 * atr(df, 14))


@strategy("S.BOOK.LIEN_CHANNEL", "books", "Lien range channel: with ADX(14) < thr, buy a close below the lower 2-sd band, "
          "sell above the upper; out at the middle band", params={"thr": 20}, space={"thr": [20, 25]},
          frequencies=FR, extra=ST)
def s_channel(df, thr=20):
    c = df["close"]
    m, s = sma(c, 20), c.rolling(20).std()
    _, _, a = _adx(df, 14)
    return out(latch((a < thr) & (c < m - 2 * s), (a < thr) & (c > m + 2 * s), c >= m, c <= m))


# ------------------------------------------------------------------------------------------------ Ichimoku, charting
@strategy("S.BOOK.ICHIMOKU_KUMO", "books", "Ichimoku cloud breakout: close above both spans (span A = mean of the 9 / 26 "
          "midpoints, span B = the 52 midpoint, both shifted 26) -> long, below -> short, inside -> flat",
          params={}, space={}, frequencies=FR, extra=ST)
def s_kumo(df):
    o, h, l, c = _ohlc(df)
    mid = lambda n: (h.rolling(n).max() + l.rolling(n).min()) / 2       # noqa: E731
    a = ((mid(9) + mid(26)) / 2).shift(26)
    b = mid(52).shift(26)
    top, bot = np.maximum(a, b), np.minimum(a, b)
    return out(_sig(c > top, c < bot, df.index).where(b.notna(), 0.0))


@strategy("S.BOOK.HEIKIN_ASHI", "books", "Heikin-Ashi trend: two consecutive bullish HA candles without a lower shadow -> "
          "long, out at the first bearish HA candle (mirrored)", params={}, space={}, frequencies=FR, extra=ST)
def s_heikin(df):
    o, h, l, c = _ohlc(df)
    hc = (o + h + l + c) / 4
    ho = np.zeros(len(c))
    hcv, ov, cv = hc.to_numpy(float), o.to_numpy(float), c.to_numpy(float)
    ho[0] = (ov[0] + cv[0]) / 2 if np.isfinite(ov[0]) else cv[0]
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hcv[i - 1]) / 2
    ho = pd.Series(ho, index=c.index)
    hl, hh = np.minimum(l, np.minimum(ho, hc)), np.maximum(h, np.maximum(ho, hc))
    bull, bear = hc > ho, hc < ho
    nolow = (np.minimum(ho, hc) - hl) <= 1e-12 * c
    nohigh = (hh - np.maximum(ho, hc)) <= 1e-12 * c
    lng = bull & bull.shift(1, fill_value=False) & nolow
    sht = bear & bear.shift(1, fill_value=False) & nohigh
    return out(latch(lng, sht, bear, bull))


@njit(cache=True)
def _pnf(c, box, rev):
    n = len(c)
    pos = np.zeros(n)
    col, top, bot = 0, np.nan, np.nan
    last_xtop, last_obot = np.nan, np.nan
    s = 0.0
    for i in range(n):
        b = box[i]
        if not np.isfinite(b) or b <= 0 or not np.isfinite(c[i]):
            pos[i] = s
            continue
        if col == 0:
            col, top, bot = 1, c[i], c[i]
        elif col > 0:
            if c[i] >= top + b:
                top = top + b * np.floor((c[i] - top) / b)
            elif c[i] <= top - rev * b:
                last_xtop = top
                col = -1
                bot = top - b * np.floor((top - c[i]) / b)
        else:
            if c[i] <= bot - b:
                bot = bot - b * np.floor((bot - c[i]) / b)
            elif c[i] >= bot + rev * b:
                last_obot = bot
                col = 1
                top = bot + b * np.floor((c[i] - bot) / b)
        if col > 0 and np.isfinite(last_xtop) and top > last_xtop:
            s = 1.0                                  # double-top breakout
        elif col < 0 and np.isfinite(last_obot) and bot < last_obot:
            s = -1.0                                 # double-bottom breakdown
        pos[i] = s
    return pos


@strategy("S.BOOK.PNF_DOUBLE", "books", "Point and figure on closes (box = 1 x ATR(14), 3-box reversal): long on a "
          "double-top breakout (an X column above the previous X column's top), short on a double-bottom breakdown",
          params={"rev": 3}, space={"rev": [2, 3]}, frequencies=FR, extra=ST)
def s_pnf(df, rev=3):
    return out(pd.Series(_pnf(df["close"].to_numpy(float), atr(df, 14).to_numpy(float), float(rev)), index=df.index))


@njit(cache=True)
def _swings(h, l, k):
    """Confirmed swing highs / lows (k bars each side), known k bars later: arrays of (value at confirmation)."""
    n = len(h)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(2 * k, n):
        j = i - k
        ok_h, ok_l = True, True
        for m in range(j - k, j + k + 1):
            if m != j:
                if h[m] >= h[j]:
                    ok_h = False
                if l[m] <= l[j]:
                    ok_l = False
        if ok_h:
            sh[i] = h[j]
        if ok_l:
            sl[i] = l[j]
    return sh, sl


@njit(cache=True)
def _patterns(c, sh, sl, a, kind):
    """kind 0: double top / bottom; 1: head and shoulders. Entry signal on the neckline close-break."""
    n = len(c)
    sig = np.zeros(n)
    stop = np.full(n, np.nan)
    H = np.full(3, np.nan)
    L = np.full(3, np.nan)
    neck_s, neck_l = np.nan, np.nan          # neckline for a top (short) / bottom (long) pattern
    stop_s, stop_l = np.nan, np.nan
    for i in range(n):
        if np.isfinite(sh[i]):
            H[0], H[1], H[2] = H[1], H[2], sh[i]
            if kind == 0 and np.isfinite(H[1]) and np.isfinite(a[i]) and abs(H[2] - H[1]) <= 0.5 * a[i] and np.isfinite(L[2]):
                neck_s, stop_s = L[2], max(H[1], H[2])
            if kind == 1 and np.isfinite(H[0]) and np.isfinite(a[i]) and H[1] > H[0] and H[1] > H[2] and abs(H[2] - H[0]) <= 0.5 * a[i] and np.isfinite(L[2]):
                neck_s, stop_s = L[2], H[2]
        if np.isfinite(sl[i]):
            L[0], L[1], L[2] = L[1], L[2], sl[i]
            if kind == 0 and np.isfinite(L[1]) and np.isfinite(a[i]) and abs(L[2] - L[1]) <= 0.5 * a[i] and np.isfinite(H[2]):
                neck_l, stop_l = H[2], min(L[1], L[2])
            if kind == 1 and np.isfinite(L[0]) and np.isfinite(a[i]) and L[1] < L[0] and L[1] < L[2] and abs(L[2] - L[0]) <= 0.5 * a[i] and np.isfinite(H[2]):
                neck_l, stop_l = H[2], L[2]
        if np.isfinite(neck_s) and c[i] < neck_s:
            sig[i], stop[i] = -1.0, stop_s - c[i]
            neck_s = np.nan
        elif np.isfinite(neck_l) and c[i] > neck_l:
            sig[i], stop[i] = 1.0, c[i] - stop_l
            neck_l = np.nan
    return sig, stop


@strategy("S.BOOK.DOUBLE_TOP_BOTTOM", "books", "Double top / bottom (Bulkowski): two confirmed swing highs (k-bar) within "
          "0.5 ATR, a close below the swing low between them -> short (mirrored); stop above the tops, 2R target",
          params={"k": 3}, space={"k": [3, 5]}, frequencies=FR, extra=TR)
def s_double(df, k=3):
    o, h, l, c = _ohlc(df)
    sh, sl = _swings(h.to_numpy(float), l.to_numpy(float), int(k))
    sig, st = _patterns(c.to_numpy(float), sh, sl, atr(df, 14).to_numpy(float), 0)
    return out(pd.Series(sig, index=df.index), stop_dist=pd.Series(st, index=df.index).clip(lower=0))


@strategy("S.BOOK.HEAD_SHOULDERS", "books", "Head and shoulders (and inverse): three confirmed swing highs with the middle "
          "highest and the shoulders within 0.5 ATR, a close below the last swing low -> short (mirrored); stop above "
          "the right shoulder, 2R target", params={"k": 3}, space={"k": [3, 5]}, frequencies=FR, extra=TR)
def s_hs(df, k=3):
    o, h, l, c = _ohlc(df)
    sh, sl = _swings(h.to_numpy(float), l.to_numpy(float), int(k))
    sig, st = _patterns(c.to_numpy(float), sh, sl, atr(df, 14).to_numpy(float), 1)
    return out(pd.Series(sig, index=df.index), stop_dist=pd.Series(st, index=df.index).clip(lower=0))


# ------------------------------------------------------------------------------------------------ others
@strategy("S.BOOK.GANN_HILO", "books", "Gann HiLo activator: close above the n-bar average of highs -> long, below the "
          "average of lows -> short (stop and reverse)", params={"n": 3}, space={"n": [3, 10]}, frequencies=FR, extra=ST)
def s_ganhilo(df, n=3):
    o, h, l, c = _ohlc(df)
    return out(latch(c > sma(h, n).shift(1), c < sma(l, n).shift(1), c < sma(l, n).shift(1), c > sma(h, n).shift(1)))


@strategy("S.BOOK.ELDER_IMPULSE", "books", "Elder Impulse system: EMA(13) and the MACD(12, 26, 9) histogram both rising -> "
          "long, both falling -> short, otherwise flat", params={}, space={}, frequencies=FR, extra=ST)
def s_impulse(df):
    c = df["close"]
    e13 = ema(c, 13)
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    return out(_sig((e13 > e13.shift(1)) & (hist > hist.shift(1)), (e13 < e13.shift(1)) & (hist < hist.shift(1)), df.index))


@strategy("S.BOOK.PRIOR_PERIOD_BO", "books", "Prior-period breakout: a close above the high of the previous m bars as one "
          "block (e.g. last week on daily bars, m = 5) -> long, out below that block's low (mirrored)",
          params={"m": 5}, space={"m": [5, 20]}, frequencies=FR, extra=ST)
def s_prior(df, m=5):
    o, h, l, c = _ohlc(df)
    blk = pd.Series(np.arange(len(c)) // m, index=c.index)
    bh, bl = h.groupby(blk).transform("max"), l.groupby(blk).transform("min")
    first = blk != blk.shift(1)
    ph = bh.shift(1).where(first).ffill()                                  # previous block's high, known at the block start
    pl = bl.shift(1).where(first).ffill()
    return out(latch(c > ph, c < pl, c < pl, c > ph))
