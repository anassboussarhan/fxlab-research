"""Round 12 part A: the nine untested rule families (docs/anatomy_preregistration.md), default parameters only.

Each function takes bars (UTC index = bar open, mid OHLC, as fxlab.data.store.resample) and returns either
- ("state", side Series) - a position held after the close of each bar (entered at the next bar's open), or
- ("specs", DataFrame) - explicit trades: t0 (order / entry time), d, etype ('mkt' | 'lim' | 'stp'), px (order
  price for lim / stp), t_valid (order expiry), sl / tp (price distances from the fill, NaN = none), hold (Timedelta
  after the fill) or t_end (absolute time exit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.features._util import atr, ema, sma, wma
from fxlab.strategies.famous import _fx_day, _day_breakout

TF = {"H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}


def _swings(x: np.ndarray, k: int, kind: str) -> np.ndarray:
    """Bars that are a strict-or-equal extreme of the k bars on each side (confirmed k bars later)."""
    n = len(x)
    out = np.zeros(n, bool)
    for i in range(k, n - k):
        w = x[i - k:i + k + 1]
        if (kind == "max" and x[i] == w.max()) or (kind == "min" and x[i] == w.min()):
            out[i] = True
    return out


def _daily_context(bars: pd.DataFrame):
    """Per bar: FX-day id, the day's last bar flag, previous complete day's H, L, C and the D1 storyline."""
    d = _fx_day(bars.index)
    g = pd.DataFrame({"h": bars["high"], "l": bars["low"], "c": bars["close"]}).groupby(d)
    day = pd.DataFrame({"h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last()})
    prev = day.shift(1)
    s = pd.Series(d, index=bars.index)
    ph, pl, pc = s.map(prev["h"]).to_numpy(), s.map(prev["l"]).to_numpy(), s.map(prev["c"]).to_numpy()
    story = (day["c"] > day["c"].rolling(20).mean()).astype(float) * 2 - 1
    story = story.where(day["c"].rolling(20).mean().notna())
    st = s.map(story.shift(1)).to_numpy()                                 # yesterday's close vs its 20-day mean
    last = np.r_[d[1:] != d[:-1], True]
    return d, last, ph, pl, pc, st


def _day_end_times(bars: pd.DataFrame, d: np.ndarray, tf: str) -> np.ndarray:
    """Time exit at the close of each bar's FX day (open time of its last bar + one bar)."""
    s = pd.Series(bars.index, index=bars.index)
    last_open = s.groupby(d).transform("max")
    return (last_open + TF[tf]).to_numpy()


# --------------------------------------------------------------------------------------------- MS1 Malaysian SNR
def ms1(bars, tf):
    c, h, l = bars["close"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    a = atr(bars, 14).to_numpy()
    _, _, _, _, _, st = _daily_context(bars)
    pk, vl = _swings(c, 3, "max"), _swings(c, 3, "min")
    act_a, act_v = [], []                                                # [level, active_from_bar, formed_bar]
    rows = []
    idx = bars.index
    for b in range(len(c)):
        if b >= 3:
            k = b - 3
            if pk[k]:
                act_a.append([c[k], b + 1, k])
            if vl[k]:
                act_v.append([c[k], b + 1, k])
        if not np.isfinite(a[b]):
            continue
        new_a = []
        for lev in act_a:
            if b >= lev[1] and h[b] >= lev[0]:                           # first revisit of a fresh A-level
                if st[b] == -1:
                    targets = [v[0] for v in act_v if v[1] <= b and v[0] < lev[0]]
                    sl = 0.5 * a[b - 1] if b > 0 and np.isfinite(a[b - 1]) else 0.5 * a[b]
                    tgt = (lev[0] - max(targets)) if targets else np.nan
                    tp = tgt if (np.isfinite(tgt) and tgt >= sl) else 2 * sl
                    rows.append((idx[b], -1, "lim", lev[0], idx[b] + TF[tf], sl, tp, 48 * TF[tf]))
                continue                                                 # touched: no longer fresh
            if b - lev[2] < 500:
                new_a.append(lev)
        act_a = new_a
        new_v = []
        for lev in act_v:
            if b >= lev[1] and l[b] <= lev[0]:
                if st[b] == 1:
                    targets = [x[0] for x in act_a if x[1] <= b and x[0] > lev[0]]
                    sl = 0.5 * a[b - 1] if b > 0 and np.isfinite(a[b - 1]) else 0.5 * a[b]
                    tgt = (min(targets) - lev[0]) if targets else np.nan
                    tp = tgt if (np.isfinite(tgt) and tgt >= sl) else 2 * sl
                    rows.append((idx[b], 1, "lim", lev[0], idx[b] + TF[tf], sl, tp, 48 * TF[tf]))
                continue
            if b - lev[2] < 500:
                new_v.append(lev)
        act_v = new_v
    return "specs", pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold"])


# --------------------------------------------------------------------------------------------- QM1 Quasimodo
def qm1(bars, tf):
    h, l = bars["high"].to_numpy(), bars["low"].to_numpy()
    a = atr(bars, 14).to_numpy()
    sh, sl_ = _swings(h, 3, "max"), _swings(l, 3, "min")
    pts = []                                                             # alternating zigzag (bar, kind, price)
    rows = []
    idx = bars.index
    for b in range(3, len(h)):
        k = b - 3
        for kind, flag, price in (("H", sh[k], h[k]), ("L", sl_[k], l[k])):
            if not flag:
                continue
            if pts and pts[-1][1] == kind:
                if (kind == "H" and price > pts[-1][2]) or (kind == "L" and price < pts[-1][2]):
                    pts[-1] = (k, kind, price)
                continue
            pts.append((k, kind, price))
            if len(pts) < 4 or b + 1 >= len(h) or not np.isfinite(a[b]):
                continue
            p1, p2, p3, p4 = pts[-4:]
            if [p[1] for p in (p1, p2, p3, p4)] == ["H", "L", "H", "L"] and p3[2] > p1[2] and p4[2] < p2[2]:
                px, stop = p1[2], p3[2] + 0.2 * a[b]                    # bearish QM: sell the left shoulder
                if stop > px:
                    rows.append((idx[b + 1], -1, "lim", px, idx[b + 1] + 48 * TF[tf], stop - px, 2 * (stop - px), 48 * TF[tf]))
            if [p[1] for p in (p1, p2, p3, p4)] == ["L", "H", "L", "H"] and p3[2] < p1[2] and p4[2] > p2[2]:
                px, stop = p1[2], p3[2] - 0.2 * a[b]
                if stop < px:
                    rows.append((idx[b + 1], 1, "lim", px, idx[b + 1] + 48 * TF[tf], px - stop, 2 * (px - stop), 48 * TF[tf]))
    return "specs", pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold"])


# --------------------------------------------------------------------------------------------- CL1 Dual Thrust
def dual_thrust(bars, tf):
    d = _fx_day(bars.index)
    g = pd.DataFrame({"h": bars["high"], "l": bars["low"], "c": bars["close"], "o": bars["open"]}).groupby(d)
    day = pd.DataFrame({"h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(), "o": g["o"].first()})
    hh, ll = day["h"].rolling(4).max().shift(1), day["l"].rolling(4).min().shift(1)
    hc, lc = day["c"].rolling(4).max().shift(1), day["c"].rolling(4).min().shift(1)
    rng = np.maximum(hh - lc, hc - ll)
    s = pd.Series(d, index=bars.index)
    up = (s.map(day["o"]) + 0.5 * s.map(rng)).to_numpy()
    dn = (s.map(day["o"]) - 0.5 * s.map(rng)).to_numpy()
    c = bars["close"].to_numpy()
    pos, side = np.zeros(len(c)), 0.0
    for i in range(len(c)):
        if np.isfinite(up[i]):
            if c[i] > up[i]:
                side = 1.0
            elif c[i] < dn[i]:
                side = -1.0
        pos[i] = side
    return "state", pd.Series(pos, index=bars.index)


# --------------------------------------------------------------------------------------------- CL2 R-Breaker / Fiali
def r_breaker(bars, tf):
    d, last, ph, pl, pc, _ = _daily_context(bars)
    P = (ph + pl + pc) / 3
    bbreak, sbreak = ph + 2 * (P - pl), pl - 2 * (ph - P)
    ssetup, senter = P + (ph - pl), 2 * P - pl
    benter, bsetup = 2 * P - ph, P - (ph - pl)
    c, hi, lo = bars["close"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy()
    pos, side, dh, dl = np.zeros(len(c)), 0.0, -np.inf, np.inf
    for i in range(len(c)):
        if i == 0 or d[i] != d[i - 1]:
            side, dh, dl = 0.0, -np.inf, np.inf
        dh, dl = max(dh, hi[i]), min(dl, lo[i])
        if np.isfinite(P[i]):
            if c[i] > bbreak[i]:
                side = 1.0
            elif c[i] < sbreak[i]:
                side = -1.0
            elif dh > ssetup[i] and c[i] < senter[i]:
                side = -1.0
            elif dl < bsetup[i] and c[i] > benter[i]:
                side = 1.0
        pos[i] = 0.0 if last[i] else side
    return "state", pd.Series(pos, index=bars.index)


def fiali(bars, tf):
    d, last, ph, pl, _, _ = _daily_context(bars)
    c = bars["close"].to_numpy(float)
    pos = _day_breakout(d, c, ph.astype(float), pl.astype(float), np.ones(len(c), np.bool_))
    return "state", pd.Series(pos, index=bars.index)


# --------------------------------------------------------------------------------------------- CL3 Aberration / King Keltner
def _latch(el, es, xl, xs):
    pos, s = np.zeros(len(el)), 0.0
    for i in range(len(el)):
        if s > 0 and xl[i]:
            s = 0.0
        elif s < 0 and xs[i]:
            s = 0.0
        if el[i] and not es[i]:
            s = 1.0
        elif es[i] and not el[i]:
            s = -1.0
        pos[i] = s
    return pos


def aberration(bars, tf):
    c = bars["close"]
    m, s = sma(c, 35), c.rolling(35).std()
    pos = _latch((c > m + 2 * s).to_numpy(), (c < m - 2 * s).to_numpy(), (c < m).to_numpy(), (c > m).to_numpy())
    return "state", pd.Series(pos, index=bars.index)


def king_keltner(bars, tf):
    c = bars["close"]
    tp = (bars["high"] + bars["low"] + c) / 3
    avg = sma(tp, 40)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - c.shift()).abs(), (bars["low"] - c.shift()).abs()], axis=1).max(axis=1)
    band = sma(tr, 40)
    rising, falling = avg > avg.shift(1), avg < avg.shift(1)
    pos = _latch(((c > avg + band) & rising).to_numpy(), ((c < avg - band) & falling).to_numpy(),
                 (c < avg).to_numpy(), (c > avg).to_numpy())
    return "state", pd.Series(pos, index=bars.index)


# --------------------------------------------------------------------------------------------- CL4 Thermostat / Bollinger Bandit
def thermostat(bars, tf):
    c, o, h, l = bars["close"], bars["open"], bars["high"], bars["low"]
    cmi = (c - c.shift(30)).abs() / (h.rolling(30).max() - l.rolling(30).min())
    choppy = (cmi < 0.2).to_numpy()
    lowb, highb = sma(l, 3).shift(1).to_numpy(), sma(h, 3).shift(1).to_numpy()
    rng = (h - l).rolling(30).mean().shift(1).to_numpy()
    m50 = sma(c, 50).to_numpy()
    cc, oo = c.to_numpy(), o.to_numpy()
    pos, s, mode = np.zeros(len(cc)), 0.0, None
    for i in range(len(cc)):
        if not (np.isfinite(cmi.iloc[i]) and np.isfinite(rng[i]) and np.isfinite(m50[i]) and np.isfinite(lowb[i])):
            pos[i] = s
            continue
        if s != 0:                                                       # exits by the mode the trade was opened in
            if mode == "range" and ((s > 0 and cc[i] > highb[i]) or (s < 0 and cc[i] < lowb[i])):
                s = 0.0
            elif mode == "trend" and ((s > 0 and cc[i] < m50[i]) or (s < 0 and cc[i] > m50[i])):
                s = 0.0
        if s == 0:
            if choppy[i]:
                if cc[i] < lowb[i]:
                    s, mode = 1.0, "range"
                elif cc[i] > highb[i]:
                    s, mode = -1.0, "range"
            else:
                if cc[i] > oo[i] + 0.5 * rng[i]:
                    s, mode = 1.0, "trend"
                elif cc[i] < oo[i] - 0.5 * rng[i]:
                    s, mode = -1.0, "trend"
        pos[i] = s
    return "state", pd.Series(pos, index=bars.index)


def bollinger_bandit(bars, tf):
    c = bars["close"].to_numpy()
    cs = bars["close"]
    m50, sd50 = sma(cs, 50).to_numpy(), cs.rolling(50).std().to_numpy()
    csum = np.r_[0.0, np.cumsum(np.nan_to_num(c))]
    pos, s, n_in = np.zeros(len(c)), 0.0, 0
    for i in range(len(c)):
        if not (np.isfinite(m50[i]) and i >= 30):
            pos[i] = s
            continue
        if s != 0:
            n_in += 1
            L = max(10, 50 - n_in)
            ma = (csum[i + 1] - csum[i + 1 - L]) / L
            if (s > 0 and c[i] < ma) or (s < 0 and c[i] > ma):
                s = 0.0
        if s == 0:
            if c[i] > m50[i] + 1.25 * sd50[i] and c[i] > c[i - 30]:
                s, n_in = 1.0, 0
            elif c[i] < m50[i] - 1.25 * sd50[i] and c[i] < c[i - 30]:
                s, n_in = -1.0, 0
        pos[i] = s
    return "state", pd.Series(pos, index=bars.index)


# --------------------------------------------------------------------------------------------- CL5 Camarilla
def camarilla(bars, tf):
    d, last, ph, pl, pc, _ = _daily_context(bars)
    rng = ph - pl
    H3, H4, L3, L4 = pc + 1.1 * rng / 4, pc + 1.1 * rng / 2, pc - 1.1 * rng / 4, pc - 1.1 * rng / 2
    h, l, c = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    tend = _day_end_times(bars, d, tf)
    idx = bars.index
    rows, done = [], set()
    first = np.r_[True, d[1:] != d[:-1]]
    for i in range(len(c) - 1):
        if not np.isfinite(H3[i]):
            continue
        key = d[i]
        if first[i]:
            # fades: limits resting at H3 / L3 from the day's first bar to its end, filled on the first touch
            rows.append((idx[i], -1, "lim", H3[i], tend[i], H4[i] - H3[i], H3[i] - pc[i], pd.NaT, tend[i]))
            rows.append((idx[i], 1, "lim", L3[i], tend[i], L3[i] - L4[i], pc[i] - L3[i], pd.NaT, tend[i]))
        if last[i]:
            continue
        # breakouts: a close beyond H4 / L4, entered at the next bar's open, stop at H3 / L3, target 1R
        if (key, "bl") not in done and c[i] > H4[i]:
            done.add((key, "bl"))
            sl = c[i] - H3[i]
            rows.append((idx[i + 1], 1, "mkt", np.nan, pd.NaT, sl, sl, pd.NaT, tend[i]))
        if (key, "bs") not in done and c[i] < L4[i]:
            done.add((key, "bs"))
            sl = L3[i] - c[i]
            rows.append((idx[i + 1], -1, "mkt", np.nan, pd.NaT, sl, sl, pd.NaT, tend[i]))
    D = pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold", "t_end"])
    return "specs", D[(D.sl > 0) & (D.tp > 0)]


# --------------------------------------------------------------------------------------------- CL6 Elder triple screen
def elder(bars, tf):
    c = bars["close"]
    wk = c.resample("W-FRI").last().dropna()
    macd = ema(wk, 12) - ema(wk, 26)
    hist = macd - ema(macd, 9)
    slope = np.sign(hist.diff())
    # a week's slope is known after its last D1 bar closes (Friday 17:00 New York, before 23:00 UTC)
    slope.index = slope.index + pd.Timedelta(hours=23)
    sl_map = slope.reindex(c.index, method="ffill")
    vol = bars["volume"] if "volume" in bars else pd.Series(1.0, index=c.index)
    fi2 = ema(c.diff() * vol, 2)
    h, l = bars["high"], bars["low"]
    idx = bars.index
    rows = []
    for i in range(1, len(c) - 1):
        s = sl_map.iloc[i]
        if not np.isfinite(s) or not np.isfinite(fi2.iloc[i]):
            continue
        t0, tv = idx[i + 1], idx[i + 1] + TF[tf]
        if s > 0 and fi2.iloc[i] < 0:
            px = h.iloc[i] * (1 + 1e-6)
            rows.append((t0, 1, "stp", px, tv, px - l.iloc[i], np.nan, 10 * TF[tf]))
        elif s < 0 and fi2.iloc[i] > 0:
            px = l.iloc[i] * (1 - 1e-6)
            rows.append((t0, -1, "stp", px, tv, h.iloc[i] - px, np.nan, 10 * TF[tf]))
    D = pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold"])
    return "specs", D[D.sl > 0]


# --------------------------------------------------------------------------------------------- CL7 Puria
def puria(bars, tf):
    c, lo = bars["close"], bars["low"]
    l75, l85, e5 = wma(lo, 75), wma(lo, 85), ema(c, 5)
    macd = ema(c, 15) - ema(c, 26)
    top, bot = np.maximum(l75, l85), np.minimum(l75, l85)
    up = (e5 > top) & (e5.shift(1) <= top.shift(1)) & (macd > 0)
    dn = (e5 < bot) & (e5.shift(1) >= bot.shift(1)) & (macd < 0)
    a = atr(bars, 14)
    idx = bars.index
    rows = []
    for i in np.flatnonzero((up | dn).to_numpy()):
        if i + 1 >= len(idx) or not np.isfinite(a.iloc[i]):
            continue
        d = 1 if up.iloc[i] else -1
        rows.append((idx[i + 1], d, "mkt", np.nan, pd.NaT, 2 * a.iloc[i], 3 * a.iloc[i], 48 * TF[tf]))
    return "specs", pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold"])


SYSTEMS = {
    "MS1": (ms1, ("H1", "H4")),
    "QM1": (qm1, ("H1", "H4")),
    "CL1_DUAL_THRUST": (dual_thrust, ("H1",)),
    "CL2_R_BREAKER": (r_breaker, ("H1",)),
    "CL2_FIALI": (fiali, ("H1",)),
    "CL3_ABERRATION": (aberration, ("H4", "D1")),
    "CL3_KING_KELTNER": (king_keltner, ("H4", "D1")),
    "CL4_THERMOSTAT": (thermostat, ("H4", "D1")),
    "CL4_BOLLINGER_BANDIT": (bollinger_bandit, ("H4", "D1")),
    "CL5_CAMARILLA": (camarilla, ("H1",)),
    "CL6_ELDER": (elder, ("D1",)),
    "CL7_PURIA": (puria, ("H1", "H4")),
}
