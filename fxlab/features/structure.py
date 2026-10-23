"""7.7 Market structure and price action. All swing points are *confirmed*
causally (a pivot k bars back is only known once k further bars exist)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, atr, as_series, causal_pivots, zigzag, sma

F = "market_structure"
OHLC = ("open", "high", "low", "close")


def fx_day_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """FX trading day (17:00 New York rollover) for each timestamp."""
    ny = index.tz_convert("America/New_York").tz_localize(None)
    return (ny + pd.Timedelta(hours=7)).floor("D")


@feature("F.ST.SWING_DIST", F, "distance of C to last confirmed swing high / low (k-bar fractal, confirmed k bars later), "
         "in ATR, plus bars elapsed since each", params={"k": 3}, space={"k": [2, 3, 5, 8]}, inputs=OHLC)
def f_swing_dist(df, k=3):
    sh, sl, shi, sli = causal_pivots(df, k)
    a = atr(df, 14).to_numpy(float)
    idx = np.arange(len(df))
    c = df["close"].to_numpy(float)
    return pd.DataFrame({"to_high": (c - sh) / (a + EPS), "to_low": (c - sl) / (a + EPS),
                         "age_high": np.where(shi >= 0, idx - shi, np.nan),
                         "age_low": np.where(sli >= 0, idx - sli, np.nan)}, index=df.index)


@njit(cache=True)
def _structure(sh, sl):
    n = len(sh)
    out = np.full(n, np.nan)
    h1 = np.nan; h0 = np.nan; l1 = np.nan; l0 = np.nan
    for i in range(n):
        if not np.isnan(sh[i]) and sh[i] != h1:
            h0, h1 = h1, sh[i]
        if not np.isnan(sl[i]) and sl[i] != l1:
            l0, l1 = l1, sl[i]
        if not (np.isnan(h0) or np.isnan(l0)):
            s = 0.0
            s += 1.0 if h1 > h0 else -1.0
            s += 1.0 if l1 > l0 else -1.0
            out[i] = s / 2.0
    return out


@feature("F.ST.STRUCTURE_STATE", F, "swing structure: +1 higher-high & higher-low, -1 lower-high & lower-low, 0 mixed, "
         "from the last two confirmed swing highs and lows", params={"k": 3}, space={"k": [2, 3, 5]}, inputs=OHLC)
def f_structure(df, k=3):
    sh, sl, _, _ = causal_pivots(df, k)
    return as_series(_structure(sh, sl), df.index)


@njit(cache=True)
def _sr(c, a, sh, sl, shi, sli, lookback, tol):
    n = len(c)
    touches = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    hs = np.full(4096, np.nan); hi = np.zeros(4096, np.int64); m = 0
    last_h, last_l = -1, -1
    for i in range(n):
        if shi[i] >= 0 and shi[i] != last_h:
            hs[m % 4096] = sh[i]; hi[m % 4096] = shi[i]; m += 1; last_h = shi[i]
        if sli[i] >= 0 and sli[i] != last_l:
            hs[m % 4096] = sl[i]; hi[m % 4096] = sli[i]; m += 1; last_l = sli[i]
        if np.isnan(a[i]) or m == 0:
            continue
        best = np.inf; cnt = 0
        for j in range(min(m, 4096)):
            if i - hi[j] > lookback:
                continue
            d = abs(c[i] - hs[j]) / a[i]
            if d < best:
                best = d
            if d <= tol:
                cnt += 1
        touches[i] = cnt
        dist[i] = best if best < np.inf else np.nan
    return touches, dist


@feature("F.ST.SR_LEVELS", F, "support/resistance from clustered confirmed swing points within lookback: number of swing "
         "levels within tol*ATR of C (touches) and distance to the nearest level in ATR",
         params={"k": 3, "lookback": 500, "tol": 0.5}, space={"k": [3, 5], "lookback": [250, 500, 1000], "tol": [0.25, 0.5]},
         inputs=OHLC, complexity="O(n·m)")
def f_sr(df, k=3, lookback=500, tol=0.5):
    sh, sl, shi, sli = causal_pivots(df, k)
    t, d = _sr(df["close"].to_numpy(float), atr(df, 14).to_numpy(float), sh, sl, shi, sli, lookback, tol)
    return pd.DataFrame({"touches": t, "nearest": d}, index=df.index)


@feature("F.ST.FLOOR_PIVOTS", F, "classic floor pivots from the previous completed FX day: P=(H+L+C)/3, R1=2P-L, S1=2P-H; "
         "outputs (C-P)/ATR and position within [S1,R1]", inputs=OHLC, frequencies="M1..H4")
def f_floor_pivots(df):
    key = fx_day_key(df.index)
    g = df.groupby(key.values)
    day = pd.DataFrame({"H": g["high"].max(), "L": g["low"].min(), "C": g["close"].last()})
    prev = day.shift(1)  # previous completed day
    P = (prev["H"] + prev["L"] + prev["C"]) / 3
    R1, S1 = 2 * P - prev["L"], 2 * P - prev["H"]
    Pm = pd.Series(P.reindex(key.values).values, index=df.index)
    R1m = pd.Series(R1.reindex(key.values).values, index=df.index)
    S1m = pd.Series(S1.reindex(key.values).values, index=df.index)
    a = atr(df, 14)
    return pd.DataFrame({"to_p": (df["close"] - Pm) / (a + EPS), "pos": (df["close"] - S1m) / (R1m - S1m + EPS)})


@feature("F.ST.BREAKOUT", F, "n-bar breakout: (C_t - max_{t-n..t-1} H)/ATR if positive, (C_t - min_{t-n..t-1} L)/ATR if "
         "negative, else 0", params={"n": 20}, space={"n": [10, 20, 55, 100]}, inputs=OHLC)
def f_breakout(df, n=20):
    hh = df["high"].shift(1).rolling(n, min_periods=n).max()
    ll = df["low"].shift(1).rolling(n, min_periods=n).min()
    a = atr(df, 14)
    up = (df["close"] - hh).clip(lower=0)
    dn = (df["close"] - ll).clip(upper=0)
    return ((up + dn) / (a + EPS)).where(hh.notna())


@feature("F.ST.FALSE_BREAKOUT", F, "failed breakout: within the last m bars H exceeded the prior n-bar high (measured "
         "before the probe) but C_t is back below that level -> -1; mirrored for lows -> +1",
         params={"n": 20, "m": 3}, space={"n": [20, 55], "m": [1, 3, 5]}, inputs=OHLC)
def f_false_breakout(df, n=20, m=3):
    ref_h = df["high"].shift(m + 1).rolling(n, min_periods=n).max()
    ref_l = df["low"].shift(m + 1).rolling(n, min_periods=n).min()
    probe_h = df["high"].rolling(m, min_periods=m).max()
    probe_l = df["low"].rolling(m, min_periods=m).min()
    bear = (probe_h > ref_h) & (df["close"] < ref_h)
    bull = (probe_l < ref_l) & (df["close"] > ref_l)
    return (bull.astype(float) - bear.astype(float)).where(ref_h.notna())


@feature("F.ST.LIQ_SWEEP", F, "liquidity sweep: bar wick pierces the last confirmed swing high but closes below it -> -1 "
         "(sweep of buy stops), mirrored for swing low -> +1", params={"k": 3}, space={"k": [3, 5, 8]}, inputs=OHLC)
def f_sweep(df, k=3):
    sh, sl, _, _ = causal_pivots(df, k)
    sh = pd.Series(sh, index=df.index).shift(1)  # level known before this bar
    sl = pd.Series(sl, index=df.index).shift(1)
    bear = (df["high"] > sh) & (df["close"] < sh)
    bull = (df["low"] < sl) & (df["close"] > sl)
    return (bull.astype(float) - bear.astype(float)).where(sh.notna() | sl.notna())


@njit(cache=True)
def _fvg(h, l, c, a, max_age):
    n = len(c)
    dist = np.full(n, np.nan); size = np.zeros(n)
    lvl = np.nan; side = 0.0; born = -1
    for i in range(2, n):
        if l[i] > h[i - 2]:
            lvl = (l[i] + h[i - 2]) / 2; side = 1.0; born = i
            size[i] = (l[i] - h[i - 2]) / a[i] if a[i] > 0 else 0.0
        elif h[i] < l[i - 2]:
            lvl = (h[i] + l[i - 2]) / 2; side = -1.0; born = i
            size[i] = -(l[i - 2] - h[i]) / a[i] if a[i] > 0 else 0.0
        if born >= 0 and i - born <= max_age and a[i] > 0:
            # gap considered filled once price trades through its midpoint
            if (side > 0 and l[i] <= lvl and i > born) or (side < 0 and h[i] >= lvl and i > born):
                born = -1
            else:
                dist[i] = side * (c[i] - lvl) / a[i]
    return size, dist


@feature("F.ST.FAIR_VALUE_GAP", F, "3-candle imbalance (FVG): gap size L_t - H_{t-2} (bull) or H_t - L_{t-2} (bear) in ATR, "
         "and signed distance to the most recent unfilled gap midpoint", params={"max_age": 50}, space={"max_age": [20, 50, 200]},
         inputs=OHLC)
def f_fvg(df, max_age=50):
    a = atr(df, 14)
    s, d = _fvg(df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float),
                a.to_numpy(float), max_age)
    return pd.DataFrame({"gap_size": s, "gap_dist": d}, index=df.index).where(a.notna())


@feature("F.ST.NARROW_RANGE", F, "volatility contraction NR-k: 1 if (H-L)_t is the smallest of the last k ranges; "
         "plus range_t / mean_k range", params={"k": 7}, space={"k": [4, 7, 14]}, inputs=OHLC)
def f_nr(df, k=7):
    r = df["high"] - df["low"]
    mn = r.rolling(k, min_periods=k).min()
    return pd.DataFrame({"nr": (r <= mn).astype(float).where(mn.notna()), "ratio": r / (r.rolling(k, min_periods=k).mean() + EPS)})


@njit(cache=True)
def _inside_outside(h, l):
    n = len(h)
    run = np.zeros(n); out = np.zeros(n)
    for i in range(1, n):
        if h[i] <= h[i - 1] and l[i] >= l[i - 1]:
            run[i] = run[i - 1] + 1
        if h[i] > h[i - 1] and l[i] < l[i - 1]:
            out[i] = 1.0
    return run, out


@feature("F.ST.INSIDE_OUTSIDE", F, "consecutive inside-bar count and outside-bar flag", inputs=OHLC)
def f_inside(df):
    r, o = _inside_outside(df["high"].to_numpy(float), df["low"].to_numpy(float))
    return pd.DataFrame({"inside_run": r, "outside": o}, index=df.index)


@njit(cache=True)
def _trendlines(h, l, c, a, k):
    n = len(c)
    sup = np.full(n, np.nan); res = np.full(n, np.nan)
    # running last two confirmed swing lows/highs (index, price)
    l0i = -1; l1i = -1; h0i = -1; h1i = -1
    for i in range(n):
        cc = i - k
        if cc - k >= 0:
            is_h = True; is_l = True
            for j in range(cc - k, i + 1):
                if h[j] > h[cc]:
                    is_h = False
                if l[j] < l[cc]:
                    is_l = False
            if is_h:
                h0i, h1i = h1i, cc
            if is_l:
                l0i, l1i = l1i, cc
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        if l0i >= 0 and l1i > l0i:
            slope = (l[l1i] - l[l0i]) / (l1i - l0i)
            sup[i] = (c[i] - (l[l1i] + slope * (i - l1i))) / a[i]
        if h0i >= 0 and h1i > h0i:
            slope = (h[h1i] - h[h0i]) / (h1i - h0i)
            res[i] = (c[i] - (h[h1i] + slope * (i - h1i))) / a[i]
    return sup, res


@feature("F.ST.TRENDLINES", F, "trendlines through the last two confirmed swing lows (support) and highs (resistance), "
         "extrapolated to t; signed distance of C in ATR (negative support distance = break)",
         params={"k": 5}, space={"k": [3, 5, 8]}, inputs=OHLC)
def f_trendlines(df, k=5):
    s, r = _trendlines(df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float),
                       atr(df, 14).to_numpy(float), k)
    return pd.DataFrame({"support": s, "resistance": r}, index=df.index)


@njit(cache=True)
def _order_blocks(o, h, l, c, a, m, mult):
    n = len(c)
    bull = np.full(n, np.nan); bear = np.full(n, np.nan)
    bz = np.nan; sz = np.nan
    for i in range(m + 1, n):
        if np.isnan(a[i]):
            continue
        move = c[i] - c[i - m]
        if move > mult * a[i]:
            for j in range(i - m, max(-1, i - m - 10), -1):  # last bearish candle before the impulse
                if c[j] < o[j]:
                    bz = (h[j] + l[j]) / 2
                    break
        elif move < -mult * a[i]:
            for j in range(i - m, max(-1, i - m - 10), -1):
                if c[j] > o[j]:
                    sz = (h[j] + l[j]) / 2
                    break
        if not np.isnan(bz):
            bull[i] = (c[i] - bz) / a[i]
        if not np.isnan(sz):
            bear[i] = (c[i] - sz) / a[i]
    return bull, bear


@feature("F.ST.ORDER_BLOCK", F, "order-block proxy: midpoint of the last opposite candle preceding an impulse "
         "|C_t - C_{t-m}| > mult*ATR; distance of C to latest bullish / bearish block in ATR",
         params={"m": 3, "mult": 2.0}, space={"m": [3, 5], "mult": [1.5, 2.0, 3.0]}, inputs=OHLC)
def f_order_block(df, m=3, mult=2.0):
    b, s = _order_blocks(*(df[k].to_numpy(float) for k in OHLC), atr(df, 14).to_numpy(float), m, mult)
    return pd.DataFrame({"bull_ob": b, "bear_ob": s}, index=df.index)


def _anat(df):
    o, h, l, c = (df[k] for k in OHLC)
    body = c - o
    rng = (h - l).replace(0, np.nan)
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l
    return o, h, l, c, body, rng, up, lo


@feature("F.ST.CANDLE_1BAR", F, "single-bar patterns on mid OHLC: doji |B|<=.1R; hammer lower>=2|B| & upper<=.1R; "
         "shooting star (mirror); marubozu sign(B) if |B|>=.9R; spinning top |B|<=.3R & both wicks>=|B|", inputs=OHLC)
def f_candle1(df):
    o, h, l, c, B, R, U, L = _anat(df)
    aB = B.abs()
    return pd.DataFrame({"doji": (aB <= 0.1 * R).astype(float),
                         "hammer": ((L >= 2 * aB) & (U <= 0.1 * R)).astype(float),
                         "shooting_star": ((U >= 2 * aB) & (L <= 0.1 * R)).astype(float),
                         "marubozu": np.sign(B) * (aB >= 0.9 * R),
                         "spinning_top": ((aB <= 0.3 * R) & (U >= aB) & (L >= aB)).astype(float)}).where(R.notna(), 0.0)


@feature("F.ST.CANDLE_2BAR", F, "two-bar patterns (+1 bull / -1 bear): engulfing, harami, piercing/dark-cloud, tweezer "
         "(|L_t-L_{t-1}|<=.05ATR)", inputs=OHLC)
def f_candle2(df):
    o, h, l, c, B, R, U, L = _anat(df)
    po, pc, pB = o.shift(1), c.shift(1), B.shift(1)
    a = atr(df, 14)
    eng = ((pB < 0) & (B > 0) & (o <= pc) & (c >= po)).astype(float) - ((pB > 0) & (B < 0) & (o >= pc) & (c <= po)).astype(float)
    har = ((pB < 0) & (B > 0) & (np.maximum(o, c) <= po) & (np.minimum(o, c) >= pc)).astype(float) - \
          ((pB > 0) & (B < 0) & (np.maximum(o, c) <= pc) & (np.minimum(o, c) >= po)).astype(float)
    mid = (po + pc) / 2
    pierce = ((pB < 0) & (B > 0) & (o <= pc) & (c > mid) & (c < po)).astype(float) - \
             ((pB > 0) & (B < 0) & (o >= pc) & (c < mid) & (c > po)).astype(float)
    tw = ((pB < 0) & (B > 0) & ((l - l.shift(1)).abs() <= 0.05 * a)).astype(float) - \
         ((pB > 0) & (B < 0) & ((h - h.shift(1)).abs() <= 0.05 * a)).astype(float)
    return pd.DataFrame({"engulfing": eng, "harami": har, "piercing": pierce, "tweezer": tw}).where(pB.notna())


@feature("F.ST.CANDLE_3BAR", F, "three-bar patterns (+1/-1): morning/evening star, three soldiers/crows, three inside up/down",
         inputs=OHLC)
def f_candle3(df):
    o, h, l, c, B, R, U, L = _anat(df)
    B1, B2, R1, R2 = B.shift(1), B.shift(2), R.shift(1), R.shift(2)
    o1, o2, c1, c2 = o.shift(1), o.shift(2), c.shift(1), c.shift(2)
    mid2 = (o2 + c2) / 2
    star = ((B2 < 0) & (B2.abs() > 0.6 * R2) & (B1.abs() < 0.3 * R1) & (B > 0) & (c > mid2)).astype(float) - \
           ((B2 > 0) & (B2.abs() > 0.6 * R2) & (B1.abs() < 0.3 * R1) & (B < 0) & (c < mid2)).astype(float)
    sold = ((B > 0) & (B1 > 0) & (B2 > 0) & (c > c1) & (c1 > c2) & (o > o1) & (o < c1) & (o1 > o2) & (o1 < c2)).astype(float) - \
           ((B < 0) & (B1 < 0) & (B2 < 0) & (c < c1) & (c1 < c2) & (o < o1) & (o > c1) & (o1 < o2) & (o1 > c2)).astype(float)
    inside_up = ((B2 < 0) & (B1 > 0) & (np.maximum(o1, c1) <= o2) & (np.minimum(o1, c1) >= c2) & (c > o2)).astype(float) - \
                ((B2 > 0) & (B1 < 0) & (np.maximum(o1, c1) <= c2) & (np.minimum(o1, c1) >= o2) & (c < o2)).astype(float)
    return pd.DataFrame({"star": star, "soldiers": sold, "three_inside": inside_up}).where(B2.notna())


@njit(cache=True)
def _heikin(o, h, l, c):
    n = len(c)
    ho = np.full(n, np.nan); hc = np.full(n, np.nan); streak = np.zeros(n)
    for i in range(n):
        hc[i] = (o[i] + h[i] + l[i] + c[i]) / 4
        ho[i] = (o[i] + c[i]) / 2 if i == 0 else (ho[i - 1] + hc[i - 1]) / 2
        col = 1.0 if hc[i] > ho[i] else -1.0
        if i > 0 and np.sign(streak[i - 1]) == col:
            streak[i] = streak[i - 1] + col
        else:
            streak[i] = col
    return ho, hc, streak


@feature("F.ST.HEIKIN_ASHI", F, "Heikin-Ashi: HA_C=(O+H+L+C)/4, HA_O=(HA_O'+HA_C')/2; signed colour streak and HA body/ATR",
         inputs=OHLC)
def f_heikin(df):
    ho, hc, s = _heikin(*(df[k].to_numpy(float) for k in OHLC))
    return pd.DataFrame({"streak": s, "body": (hc - ho) / (atr(df, 14).to_numpy(float) + EPS)}, index=df.index)


@njit(cache=True)
def _renko(c, a, mult):
    n = len(c)
    d = np.zeros(n); run = np.zeros(n)
    base = np.nan; dirn = 0.0; r = 0.0
    for i in range(n):
        if np.isnan(a[i]):
            continue
        if np.isnan(base):
            base = c[i]
        b = mult * a[i]
        while c[i] >= base + b:
            base += b
            r = r + 1 if dirn > 0 else 1.0
            dirn = 1.0
        while c[i] <= base - b:
            base -= b
            r = r + 1 if dirn < 0 else 1.0
            dirn = -1.0
        d[i] = dirn; run[i] = r * dirn
    return d, run


@feature("F.ST.RENKO", F, "causal Renko with brick = mult*ATR_14 at formation: current brick direction and signed run length",
         params={"mult": 1.0}, space={"mult": [0.5, 1.0, 2.0]}, inputs=OHLC)
def f_renko(df, mult=1.0):
    d, r = _renko(df["close"].to_numpy(float), atr(df, 14).to_numpy(float), mult)
    return pd.DataFrame({"dir": d, "run": r}, index=df.index)


@feature("F.ST.ZIGZAG_LEG", F, "causal % zigzag (threshold thr): size of the developing leg since the last confirmed pivot and of "
         "the last completed leg (both in ATR), and current leg direction", params={"thr": 0.005}, space={"thr": [0.002, 0.005, 0.01]})
def f_zigzag_leg(df, thr=0.005):
    pp, pi, d = zigzag(df["close"], thr)
    c = df["close"].to_numpy(float)
    a = atr(df, 14).to_numpy(float) if {"high", "low"} <= set(df.columns) else df["close"].diff().abs().rolling(14).mean().to_numpy()
    leg = (c - pp[:, 3]) / (a + EPS)
    prev_leg = (pp[:, 3] - pp[:, 2]) / (a + EPS)
    return pd.DataFrame({"leg": leg, "prev_leg": prev_leg, "dir": d}, index=df.index)


@njit(cache=True)
def _profile(h, l, v, c, w, nb):
    n = len(c)
    poc = np.full(n, np.nan); va = np.full(n, np.nan)
    for i in range(w - 1, n):
        lo = l[i - w + 1:i + 1].min(); hi = h[i - w + 1:i + 1].max()
        if hi <= lo:
            continue
        hist = np.zeros(nb)
        for j in range(i - w + 1, i + 1):
            a = int((l[j] - lo) / (hi - lo) * (nb - 1)); b = int((h[j] - lo) / (hi - lo) * (nb - 1))
            k = b - a + 1
            for q in range(a, b + 1):
                hist[q] += v[j] / k
        p = np.argmax(hist)
        poc_px = lo + (p + 0.5) * (hi - lo) / nb
        tot = hist.sum(); acc = hist[p]; a_ = p; b_ = p
        while acc < 0.7 * tot:
            left = hist[a_ - 1] if a_ > 0 else -1.0
            right = hist[b_ + 1] if b_ < nb - 1 else -1.0
            if right >= left:
                b_ += 1; acc += right
            else:
                a_ -= 1; acc += left
        val = lo + a_ * (hi - lo) / nb; vah = lo + (b_ + 1) * (hi - lo) / nb
        poc[i] = (c[i] - poc_px) / (hi - lo)
        va[i] = (c[i] - val) / (vah - val) if vah > val else np.nan
    return poc, va


@feature("F.ST.VOLUME_PROFILE", F, "tick-volume profile over trailing w bars (nb price bins): position of C relative to "
         "point of control and to the 70% value area", params={"w": 100, "nb": 30}, space={"w": [50, 100, 250], "nb": [30]},
         inputs=("high", "low", "close", "n_ticks"), complexity="O(n·w·nb)")
def f_profile(df, w=100, nb=30):
    p, v = _profile(df["high"].to_numpy(float), df["low"].to_numpy(float), df["n_ticks"].to_numpy(float),
                    df["close"].to_numpy(float), w, nb)
    return pd.DataFrame({"poc": p, "value_area": v}, index=df.index)
