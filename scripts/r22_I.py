"""Round 22 part I: ICT / SMC and level rules with every parameter optimised walk-forward
(docs/round22_preregistration.md). 128 scrambled-Sobol configurations per family and instrument; the timeframe
(M15 / H1 for FX and metals; M5 / M15 session bars for the indices) is one of the parameters.

Families: LVL (sweeps / Turtle Soup / Asia sweep / prior-day-week runs / session breakouts), FVG, OB, OTE, BRK,
CBDR (FX), SES (Judas / midnight open / power of three), SMT (EUR/GBP, US500/NAS100).
Usage: python scripts/r22_I.py --symbol SYM --out DIR        (SMT: --symbol EURUSD+GBPUSD or USA500IDXUSD+USATECHIDXUSD)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402

NY = "America/New_York"
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
FUT_PTS = {"USA500IDXUSD": 0.25 + 2.5 / 50, "USATECHIDXUSD": 0.25 + 2.5 / 20}
N_CFG = 128


def load(sym):
    from fxlab.data.store import load_m1
    m = load_m1(sym, "2007-01-01", "2025-12-31", columns=COLS)
    mid = pd.DataFrame({k: 0.5 * (m[f"bid_{k}"] + m[f"ask_{k}"]) for k in "ohlc"})
    mid["spr"] = (m.ask_o - m.bid_o).clip(lower=0)
    mid = mid[np.isfinite(mid.o) & (mid.o > 0)]
    if "IDX" in sym:
        pip = float(1e-4 * np.nanmedian(mid.c))
    elif sym.startswith("XAU"):
        pip = 0.1
    elif sym.startswith("XAG"):
        pip = 0.01
    else:
        pip = 0.01 if sym.endswith("JPY") else 1e-4
    return mid, pip


def bars(mid, rule, session=False):
    x = mid
    if session:                                                        # index cash session 09:30-16:00 New York
        ny = x.index.tz_convert(NY)
        mm = ny.hour * 60 + ny.minute
        x = x[(mm >= 570) & (mm < 960)]
    g = x.resample(rule, label="left", closed="left")
    b = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "spr": g.spr.mean()}).dropna()
    return b


class Bars:
    def __init__(self, b, sym, pip):
        self.idx = b.index
        self.o, self.h, self.l, self.c, self.spr = (b[k].to_numpy(float) for k in ("o", "h", "l", "c", "spr"))
        pc = np.r_[self.c[0], self.c[:-1]]
        tr = np.maximum(np.maximum(self.h - self.l, np.abs(self.h - pc)), np.abs(self.l - pc))
        self.atr = pd.Series(tr).rolling(20, min_periods=10).mean().shift().to_numpy()
        ny = self.idx.tz_convert(NY)
        self.nyh = (ny.hour + ny.minute / 60.0).to_numpy()
        self.uth = (self.idx.hour + self.idx.minute / 60.0).to_numpy()
        self.day = ny.normalize().tz_localize(None)                          # New York date
        roll = (self.idx + pd.Timedelta(hours=7)).tz_convert(NY).normalize().tz_localize(None)   # 17:00 NY roll
        self.tday = roll
        self.sym, self.pip = sym, pip
        self.n = len(self.o)
        codes, uniq = pd.factorize(pd.Index(self.day))
        order = np.argsort(codes, kind="stable")
        cuts = np.flatnonzero(np.diff(codes[order])) + 1
        self.days_u = list(uniq)
        self.day_pos = np.split(order, cuts)
        self.fut = FUT_PTS.get(sym)

    def cost(self, i, px):
        if self.fut:
            return self.fut / px
        from fxlab.execution.costs import BASE
        s = self.spr[min(i, self.n - 1)]
        return (0.5 * s + BASE.slip_spread_frac * s + BASE.slip_fixed_pips * self.pip) / px + BASE.commission_bps_per_side * 1e-4


def trade(B, i0, d, entry, stop, tgt, max_hold, limit=None, expiry=0, cancel=None, t_exit=None):
    """Market entry at bar i0's open (entry=None) or a limit at `limit` from bar i0 to i0+expiry.
    stop / tgt are absolute prices. Returns (i_in, net, gross) or None."""
    o, h, l, c = B.o, B.h, B.l, B.c
    n = B.n
    if limit is None:
        if i0 >= n:
            return None
        j, E = i0, o[i0]
    else:
        j = None
        for k in range(i0, min(n, i0 + expiry + 1)):
            if cancel is not None and ((d > 0 and h[k] >= cancel) or (d < 0 and l[k] <= cancel)):
                return None
            if (d > 0 and l[k] <= limit) or (d < 0 and h[k] >= limit):
                j, E = k, (min(o[k], limit) if d > 0 else max(o[k], limit))
                break
        if j is None:
            return None
    if stop is None or not np.isfinite(stop):
        return None
    if (d > 0 and stop >= E) or (d < 0 and stop <= E):
        return None
    X, kx = None, None
    first = j if limit is not None else j
    for k in range(first, min(n, j + max_hold + 1)):
        if t_exit is not None and k > j and B.day[k] != B.day[j]:
            X, kx = o[k], k
            break
        if t_exit is not None and B.nyh[k] >= t_exit and k > j:
            X, kx = o[k], k
            break
        if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
            X, kx = (min(o[k], stop) if (d > 0 and k > j) else (max(o[k], stop) if (d < 0 and k > j) else stop)), k
            break
        if k > j and tgt is not None and ((d > 0 and h[k] >= tgt) or (d < 0 and l[k] <= tgt)):
            X, kx = (max(o[k], tgt) if d > 0 else min(o[k], tgt)), k
            break
    if X is None:
        kx = min(n - 1, j + max_hold)
        X = c[kx]
    g = d * (X / E - 1)
    return j, g - B.cost(j, E) - B.cost(kx, X), g, kx


# ============================================================================================ families
def pick(u, opts):
    return opts[min(int(u * len(opts)), len(opts) - 1)]


def rng(u, lo, hi):
    return lo + u * (hi - lo)


def fam_LVL(B, u, levels):
    kind = pick(u[0], ["PD", "PW", "ASIA", "OR", "FR"])
    trig = pick(u[1], ["wick", "close"])
    act = pick(u[2], [-1, 1])                                            # -1 fade, +1 follow
    retest = u[3] < 0.5
    r_off = rng(u[4], 0, 1)
    s_atr = rng(u[5], 0.5, 3)
    R = rng(u[6], 0.5, 5)
    w0 = int(rng(u[7], 0, 24))
    wl = int(rng(u[8], 2, 24))
    mh = int(rng(u[9], 2, 96))
    orm = int(rng(u[10], 15, 120))
    frn = int(rng(u[11], 2, 10))
    hi_l, lo_l = levels(kind, orm, frn)
    trades = []
    used_up, used_dn = set(), set()
    h, l, c = B.h, B.l, B.c
    for i in range(21, B.n - 2):
        hr = B.uth[i]
        if not (((hr - w0) % 24) < wl):
            continue
        a = B.atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        td = B.tday[i]
        for side, lev, used in ((1, hi_l[i], used_up), (-1, lo_l[i], used_dn)):
            if not np.isfinite(lev) or (td, lev) in used:
                continue
            x = (h[i] if trig == "wick" else c[i]) if side > 0 else (l[i] if trig == "wick" else c[i])
            if (side > 0 and x <= lev) or (side < 0 and x >= lev):
                continue
            used.add((td, lev))
            d = side * act
            if retest:
                lim = lev + (-d) * r_off * a if act > 0 else lev
                E_ = lim
                stop = E_ - d * s_atr * a
                r = trade(B, i + 1, d, None, stop, E_ + d * R * s_atr * a, mh, limit=lim, expiry=12)
            else:
                E_ = B.o[i + 1]
                stop = E_ - d * s_atr * a
                r = trade(B, i + 1, d, None, stop, E_ + d * R * s_atr * a, mh)
            if r:
                trades.append(r)
    return trades


def make_levels(B):
    """Functions returning per-bar prior-day / week / Asian / opening-range / fractal levels (known at the bar)."""
    td = B.tday
    df = pd.DataFrame({"h": B.h, "l": B.l, "td": td})
    dh, dl = df.groupby("td").h.max(), df.groupby("td").l.min()
    pdh = pd.Series(td).map(dh.shift()).to_numpy()
    pdl = pd.Series(td).map(dl.shift()).to_numpy()
    wk = pd.DatetimeIndex(td).to_period("W-FRI")
    wh, wl = df.groupby(wk).h.max(), df.groupby(wk).l.min()
    pwh = pd.Series(wk).map(wh.shift()).to_numpy()
    pwl = pd.Series(wk).map(wl.shift()).to_numpy()
    asia = (B.uth >= 23) | (B.uth < 7)                                       # Asian session 23:00-07:00 UTC
    ah = pd.Series(np.where(asia, B.h, np.nan)).groupby(np.asarray(td)).cummax().to_numpy()
    al = pd.Series(np.where(asia, B.l, np.nan)).groupby(np.asarray(td)).cummin().to_numpy()
    ah = pd.Series(np.where(asia, np.nan, ah)).groupby(np.asarray(td)).ffill().to_numpy()
    al = pd.Series(np.where(asia, np.nan, al)).groupby(np.asarray(td)).ffill().to_numpy()
    asia_after = ~asia
    cache = {}

    def opening_range(minutes):
        key = ("OR", minutes)
        if key in cache:
            return cache[key]
        start = 9.5 if B.fut or "IDX" in B.sym else 3.0                     # NY open for indices; London 03:00 NY
        inr = (B.nyh >= start) & (B.nyh < start + minutes / 60.0)
        hh = pd.Series(np.where(inr, B.h, np.nan)).groupby(np.asarray(B.day)).cummax()
        ll = pd.Series(np.where(inr, B.l, np.nan)).groupby(np.asarray(B.day)).cummin()
        after = B.nyh >= start + minutes / 60.0
        hh = hh.groupby(np.asarray(B.day)).ffill().to_numpy()
        ll = ll.groupby(np.asarray(B.day)).ffill().to_numpy()
        cache[key] = (np.where(after, hh, np.nan), np.where(after, ll, np.nan))
        return cache[key]

    def fractal(N):
        key = ("FR", N)
        if key in cache:
            return cache[key]
        h, l = B.h, B.l
        hs = pd.Series(h).rolling(2 * N + 1, center=True).max().to_numpy() == h
        ls = pd.Series(l).rolling(2 * N + 1, center=True).min().to_numpy() == l
        fh = pd.Series(np.where(hs, h, np.nan)).shift(N).ffill().to_numpy()     # known N bars later
        fl = pd.Series(np.where(ls, l, np.nan)).shift(N).ffill().to_numpy()
        cache[key] = (fh, fl)
        return cache[key]

    def levels(kind, orm, frn):
        if kind == "PD":
            return pdh, pdl
        if kind == "PW":
            return pwh, pwl
        if kind == "ASIA":
            return np.where(asia_after, ah, np.nan), np.where(asia_after, al, np.nan)
        if kind == "OR":
            return opening_range(orm)
        return fractal(frn)
    return levels


def window_ok(B, i, win):
    if win == "any":
        return True
    t = B.nyh[i]
    return {"london": 2 <= t < 5, "nyam": 8.5 <= t < 11, "sb": 10 <= t < 11}[win]


def fam_FVG(B, u):
    gmin = rng(u[0], 0, 1.5)
    withd = u[1] < 0.5
    ent = pick(u[2], ["near", "mid", "far"])
    se = rng(u[3], 0, 1)
    R = rng(u[4], 1, 5)
    win = pick(u[5], ["any", "london", "nyam", "sb"])
    exp = int(rng(u[6], 3, 48))
    mh = int(rng(u[7], 6, 96))
    trades, busy = [], -1
    h, l = B.h, B.l
    for k in range(22, B.n - 2):
        if k <= busy or not window_ok(B, k, win):
            continue
        a = B.atr[k]
        if not np.isfinite(a) or a <= 0:
            continue
        if l[k] > h[k - 2] and l[k] - h[k - 2] >= gmin * a:
            top, bot, gd, c1 = l[k], h[k - 2], 1, l[k - 2]
        elif h[k] < l[k - 2] and l[k - 2] - h[k] >= gmin * a:
            top, bot, gd, c1 = l[k - 2], h[k], -1, h[k - 2]
        else:
            continue
        d = gd if withd else -gd
        lvl = {"near": top if gd > 0 else bot, "mid": 0.5 * (top + bot), "far": bot if gd > 0 else top}[ent]
        if d == gd:
            stop = c1 - gd * se * a
        else:                                                              # against the gap (inversion): stop beyond the gap
            stop = (top if gd > 0 else bot) + gd * (se * a + 1e-12) * 1.0
            stop = stop if d < 0 else stop
        if (d > 0 and stop >= lvl) or (d < 0 and stop <= lvl):
            continue
        risk = abs(lvl - stop)
        r = trade(B, k + 1, d, None, stop, lvl + d * R * risk, mh, limit=lvl, expiry=exp)
        if r:
            trades.append(r)
            busy = r[3]
    return trades


def fam_OB(B, u):
    disp = rng(u[0], 1, 4)
    zone = pick(u[1], ["body", "wick"])
    ent = pick(u[2], ["edge", "mid"])
    se = rng(u[3], 0, 1)
    R = rng(u[4], 1, 5)
    exp = int(rng(u[5], 5, 100))
    mh = int(rng(u[6], 6, 96))
    o, h, l, c = B.o, B.h, B.l, B.c
    trades, busy = [], -1
    for k in range(25, B.n - 2):
        if k <= busy:
            continue
        a = B.atr[k]
        if not np.isfinite(a) or a <= 0 or (h[k] - l[k]) < disp * a:
            continue
        up = c[k] > o[k]
        j = None
        for q in range(k - 1, max(k - 6, 0), -1):
            if (up and c[q] < o[q]) or ((not up) and c[q] > o[q]):
                j = q
                break
        if j is None:
            continue
        zt, zb = (max(o[j], c[j]), min(o[j], c[j])) if zone == "body" else (h[j], l[j])
        d = 1 if up else -1
        lvl = (zt if d > 0 else zb) if ent == "edge" else 0.5 * (zt + zb)
        stop = (zb - se * a) if d > 0 else (zt + se * a)
        risk = abs(lvl - stop)
        if risk <= 0:
            continue
        r = trade(B, k + 1, d, None, stop, lvl + d * R * risk, mh, limit=lvl, expiry=exp)
        if r:
            trades.append(r)
            busy = r[3]
    return trades


def fam_OTE(B, u):
    disp = rng(u[0], 1.5, 4)
    lb = int(rng(u[1], 3, 30))
    ret = rng(u[2], 0.5, 0.85)
    se = rng(u[3], 0, 1)
    ext = rng(u[4], 0, 1)
    exp = int(rng(u[5], 3, 48))
    mh = int(rng(u[6], 6, 96))
    o, h, l, c = B.o, B.h, B.l, B.c
    trades, busy = [], -1
    for k in range(lb + 21, B.n - 2):
        if k <= busy:
            continue
        a = B.atr[k]
        rg = h[k] - l[k]
        if not np.isfinite(a) or a <= 0 or rg < disp * a:
            continue
        pos = (c[k] - l[k]) / rg
        if pos >= 0.75:
            d, L, H = 1, l[k - lb:k + 1].min(), h[k]
        elif pos <= 0.25:
            d, L, H = -1, h[k - lb:k + 1].max(), l[k]
        else:
            continue
        leg = H - L
        lvl = H - ret * leg
        stop = L - d * se * a
        tgt = H + ext * leg
        r = trade(B, k + 1, d, None, stop, tgt, mh, limit=lvl, expiry=exp, cancel=H if ext == 0 else None)
        if r:
            trades.append(r)
            busy = r[3]
    return trades


def fam_BRK(B, u):
    N = int(rng(u[0], 2, 8))
    bw = int(rng(u[1], 6, 96))
    ent = pick(u[2], ["edge", "mid"])
    sa = rng(u[3], 0, 2)
    R = rng(u[4], 1, 5)
    exp = int(rng(u[5], 12, 48))
    mh = int(rng(u[6], 6, 96))
    h, l, c = B.h, B.l, B.c
    hs = np.flatnonzero(pd.Series(h).rolling(2 * N + 1, center=True).max().to_numpy() == h)
    ls = np.flatnonzero(pd.Series(l).rolling(2 * N + 1, center=True).min().to_numpy() == l)
    trades, busy = [], -1
    for kind, P, Q in (("bear", hs, ls), ("bull", ls, hs)):
        for i in range(1, len(P)):
            s, sp = P[i], P[i - 1]
            if not ((h[s] > h[sp]) if kind == "bear" else (l[s] < l[sp])):
                continue
            qs = Q[(Q < s) & (Q > sp)]
            if not len(qs):
                continue
            q = int(qs[-1])
            conf = s + N
            lv = l[q] if kind == "bear" else h[q]
            brk = None
            for t in range(conf, min(B.n - 1, conf + bw)):
                if (kind == "bear" and c[t] < lv) or (kind == "bull" and c[t] > lv):
                    brk = t
                    break
            if brk is None or brk <= busy or not np.isfinite(B.atr[brk]):
                continue
            d = -1 if kind == "bear" else 1
            zt, zb = h[q], l[q]
            lvl = (zb if d < 0 else zt) if ent == "edge" else 0.5 * (zt + zb)
            stop = (zt + sa * B.atr[brk]) if d < 0 else (zb - sa * B.atr[brk])
            risk = abs(stop - lvl)
            if risk <= 0:
                continue
            r = trade(B, brk + 1, d, None, stop, lvl + d * R * risk, mh, limit=lvl, expiry=exp)
            if r:
                trades.append(r)
                busy = r[3]
    return trades


def fam_CBDR(B, u):
    s0 = rng(u[0], 12, 16)
    e0 = rng(u[1], 18, 21)
    mx = rng(u[2], 0.2, 1.0)
    en = rng(u[3], 1, 4)
    st = en + rng(u[4], 1, 4)
    tg = pick(u[5], ["mid", "edge", "one"])
    fl = rng(u[6], 8, 14)
    days = np.asarray(B.day)
    df = pd.DataFrame({"h": B.h, "l": B.l, "d": days})
    adr = (df.groupby("d").h.max() - df.groupby("d").l.min()).rolling(20).mean().shift()
    trades = []
    ud = B.days_u
    pos = dict(zip(B.days_u, B.day_pos))
    for j in range(1, len(ud)):
        d0, d1 = ud[j - 1], ud[j]
        ii = pos[d0]
        sel = ii[(B.nyh[ii] >= s0) & (B.nyh[ii] < e0)]
        if len(sel) < 3 or not np.isfinite(adr.get(d0, np.nan)):
            continue
        CH, CL = B.h[sel].max(), B.l[sel].min()
        rgn = CH - CL
        if rgn <= 0 or rgn > mx * adr[d0]:
            continue
        nxt = pos[d1]
        w = nxt[B.nyh[nxt] < fl]
        if len(w) < 2:
            continue
        for d, lvl, stp in ((-1, CH + en * rgn, CH + st * rgn), (1, CL - en * rgn, CL - st * rgn)):
            tgt = {"mid": 0.5 * (CH + CL), "edge": CH if d < 0 else CL, "one": lvl + d * rgn}[tg]
            r = trade(B, int(w[0]), d, None, stp, tgt, int(w[-1] - w[0]), limit=lvl, expiry=int(w[-1] - w[0]), t_exit=fl)
            if r:
                trades.append(r)
    return trades


def fam_SES(B, u, index):
    start = 9.5 if index else 3.0
    pw = rng(u[0], 0.25, 1.5)
    ref = pick(u[1], ["midnight", "open"])
    act = pick(u[2], [-1, 1])
    stop_k = pick(u[3], ["probe", 1.0, 2.0, 3.0])
    ex = rng(u[4], 11, 16) if index else rng(u[4], 8, 12)
    trades = []
    for ii in B.day_pos:
        if len(ii) < 5:
            continue
        mo_i = ii[B.nyh[ii] >= 0][0]
        so = ii[B.nyh[ii] >= start]
        if not len(so):
            continue
        so_i = so[0]
        pr = ii[(B.nyh[ii] >= start) & (B.nyh[ii] < start + pw)]
        if len(pr) < 1 or pr[-1] + 1 >= B.n:
            continue
        refp = B.o[mo_i] if ref == "midnight" else B.o[so_i]
        hi, lo = B.h[pr].max(), B.l[pr].min()
        cl = B.c[pr[-1]]
        up_probe = hi > max(refp, B.o[so_i]) and cl < B.o[so_i]
        dn_probe = lo < min(refp, B.o[so_i]) and cl > B.o[so_i]
        if up_probe == dn_probe:
            continue
        probe_dir = 1 if up_probe else -1
        dd = -probe_dir if act < 0 else probe_dir
        i0 = pr[-1] + 1
        a = B.atr[i0]
        E = B.o[i0]
        if stop_k == "probe":
            stop = hi if dd < 0 else lo
        else:
            if not np.isfinite(a):
                continue
            stop = E - dd * stop_k * a
        r = trade(B, i0, dd, None, stop, None, 400, t_exit=ex)
        if r:
            trades.append(r)
    return trades


def fam_SMT(A, Bb, u):
    """A, Bb: aligned Bars; signal on A vs Bb."""
    lb = int(rng(u[0], 6, 72))
    hold = int(rng(u[1], 3, 48))
    mode = pick(u[2], ["fade_leader", "follow_laggard"])
    hA = pd.Series(A.h).rolling(lb).max().shift().to_numpy()
    lA = pd.Series(A.l).rolling(lb).min().shift().to_numpy()
    hB = pd.Series(Bb.h).rolling(lb).max().shift().to_numpy()
    lB = pd.Series(Bb.l).rolling(lb).min().shift().to_numpy()
    trades, busy = [], -1
    for b in range(lb + 1, A.n - hold - 1):
        if b <= busy:
            continue
        up = A.h[b] > hA[b] and Bb.h[b] <= hB[b]
        dn = A.l[b] < lA[b] and Bb.l[b] >= lB[b]
        if up == dn:
            continue
        if mode == "fade_leader":
            X, d = A, (-1 if up else 1)
        else:
            X, d = Bb, (-1 if up else 1)
        E = X.o[b + 1]
        x_ = X.o[b + hold] if b + hold < X.n else X.c[-1]
        if (X.idx[b + hold] - X.idx[b + 1]) > pd.Timedelta(hours=60):
            continue
        g = d * (x_ / E - 1)
        trades.append((b + 1, g - X.cost(b + 1, E) - X.cost(b + hold, x_), g, b + hold))
        busy = b + hold
    return trades


FAMS = {"LVL": 12, "FVG": 8, "OB": 7, "OTE": 7, "BRK": 7, "CBDR": 7, "SES": 5}


def daily(trades, B, cal):
    s = pd.Series(0.0, index=cal)
    if not trades:
        return s
    d = pd.DatetimeIndex([B.day[t[0]] for t in trades])
    v = pd.Series([t[1] for t in trades], index=d).groupby(level=0).sum()
    return s.add(v.reindex(cal).fillna(0.0), fill_value=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_CFG)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows, ser = [], {}
    if "+" in a.symbol:
        s1, s2 = a.symbol.split("+")
        (m1, p1), (m2, p2) = load(s1), load(s2)
        index = "IDX" in s1
        tfs = ("5min", "15min") if index else ("15min", "1h")
        BB = {}
        for tf in tfs:
            b1, b2 = bars(m1, tf, index), bars(m2, tf, index)
            ix = b1.index.intersection(b2.index)
            BB[tf] = (Bars(b1.reindex(ix), s1, p1), Bars(b2.reindex(ix), s2, p2))
        cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]][0].day)))
        U = qmc.Sobol(4, scramble=True, seed=22).random(a.n)
        M = []
        for u in U:
            tf = pick(u[3], list(tfs))
            A_, B_ = BB[tf]
            M.append(daily(fam_SMT(A_, B_, u), A_, cal).to_numpy())
        groups = {"SMT": np.vstack(M)}
        sym_label = a.symbol
    else:
        mid, pip = load(a.symbol)
        index = "IDX" in a.symbol
        tfs = ("5min", "15min") if index else ("15min", "1h")
        BB = {tf: Bars(bars(mid, tf, index), a.symbol, pip) for tf in tfs}
        if index:                                                          # full-day bars for level families
            BBfull = {tf: Bars(bars(mid, tf, False), a.symbol, pip) for tf in tfs}
        else:
            BBfull = BB
        cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]].day)))
        LV = {tf: make_levels(BBfull[tf]) for tf in tfs}
        groups = {}
        for fam, dims in FAMS.items():
            if fam == "CBDR" and index:
                continue
            U = qmc.Sobol(dims + 1, scramble=True, seed=22 + len(fam)).random(a.n)
            M = []
            for u in U:
                tf = pick(u[-1], list(tfs))
                Bx = BBfull[tf] if fam in ("LVL", "CBDR", "SES") else BB[tf]
                if fam == "LVL":
                    tr = fam_LVL(Bx, u, LV[tf])
                elif fam == "FVG":
                    tr = fam_FVG(Bx, u)
                elif fam == "OB":
                    tr = fam_OB(Bx, u)
                elif fam == "OTE":
                    tr = fam_OTE(Bx, u)
                elif fam == "BRK":
                    tr = fam_BRK(Bx, u)
                elif fam == "CBDR":
                    tr = fam_CBDR(Bx, u)
                else:
                    tr = fam_SES(Bx, u, index)
                M.append(daily(tr, Bx, cal).to_numpy())
            groups[fam] = np.vstack(M)
            print(a.symbol, fam, "done", round(time.time() - t0), "s", flush=True)
        sym_label = a.symbol
    for fam, M in groups.items():
        first = max(2010, int(cal.year.min()) + 3)
        s5, s1_, ch = wf22.walk_forward(M, cal, first, min_active=20)
        st, st1 = wf22.stream_stats(s5), wf22.stream_stats(s1_)
        gross_note = float(np.nanmean(M))
        rows.append({"part": "I", "family": f"I-{fam}", "symbol": sym_label, "placebo": False, "n_cfg": int(M.shape[0]), **st,
                     "top1_p2_mean": st1["p2_mean"], "top1_p2_t": st1["p2_t"], "own": wf22.counts_rule(st),
                     "mean_all_cfg": gross_note, "choices": json.dumps({str(k): v["cfg"] for k, v in ch.items()})})
        ser[f"I-{fam}"] = s5
    tag = sym_label.replace("+", "_")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"I_rows_{tag}.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, f"I_series_{tag}.parquet"))
    print(pd.DataFrame(rows)[["family", "symbol", "p1_mean", "p2_mean", "p2_t", "own"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
