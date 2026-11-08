"""Round 14 analysis (docs/trend14_preregistration.md).

A: an independent daily-bar engine (no round-13 engine, indicator or system code) re-runs the round-13 systems and
is compared with round 13; fills at the close; ECB reference rates 1999-2025.
B: trend variants (speeds, long-or-flat, currency-strength indices, trend + carry, high-volatility only).
C: can trend be predicted (direction models; trendiness predictors and switching models), walk-forward.

Usage: python scripts/trend14.py --inp DIR_WITH_d1p --r13 A_monthly_by_symbol.csv --r13diag diag_groups.json
       --ecb exr.csv --out DIR [--parts A,B,C]"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

DEV_END = pd.Timestamp("2025-12-31")
FIT = ("2008-01", "2018-12")
CHK = ("2019-01", "2025-12")
RISK, COMM = 0.005, 0.35e-4
CUR8 = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")       # market order of quotation
FX = sorted({a + b for i, a in enumerate(CUR8) for b in CUR8[i + 1:]})
NONFX = ["XAUUSD", "XAGUSD", "USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY"]


# ================================================================================ statistics
def nw_t(x, lags=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else np.nan


def month_grid(a="2008-01", b="2025-12"):
    return pd.period_range(a, b, freq="M").astype(str)


def bh_pass(p, q=0.10):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok])]
    m = len(ok)
    thr = q * np.arange(1, m + 1) / m
    passed = np.flatnonzero(p[o] <= thr)
    if len(passed):
        out[o[:passed.max() + 1]] = True
    return out


def pstats(m: pd.Series, fit=FIT, chk=CHK):
    """m: monthly returns indexed by 'YYYY-MM' (missing months = 0). Scaled to 10% annual vol on the fit period."""
    m = m.reindex(month_grid(fit[0], chk[1])).fillna(0.0)
    f, c = m[m.index <= fit[1]], m[m.index >= chk[0]]
    sd = f.std()
    sc = 0.10 / (sd * math.sqrt(12)) if sd > 0 else np.nan
    return {"fit_mean_pct": f.mean() * sc * 100, "fit_t": nw_t(f), "check_mean_pct": c.mean() * sc * 100,
            "check_t": nw_t(c), "check_sharpe": c.mean() / c.std() * math.sqrt(12) if c.std() > 0 else np.nan}, sc


# ================================================================================ data
def load_panels(inp):
    P = {}
    for f in sorted(glob.glob(os.path.join(inp, "**", "d1p_*.parquet"), recursive=True)):
        sym = os.path.basename(f)[4:-8]
        d = pd.read_parquet(f)
        d["tday"] = pd.to_datetime(d["tday"])
        d = d[d.tday <= DEV_END].copy()
        d["x_mid"] = 0.5 * (d.x_bid + d.x_ask)
        d["x_half"] = 0.5 * (d.x_ask - d.x_bid)
        d["c_mid"] = 0.5 * (d.c_bid + d.c_ask)
        d["x_cost"] = (d.x_slip + d.x_half) / d.x_mid + COMM
        d["month"] = d.tday.dt.to_period("M").astype(str)
        P[sym] = d
    return P


def as_f(x):
    return np.ascontiguousarray(np.asarray(x, np.float64))


# ================================================================================ engines (new code)
@njit(cache=True)
def stop_engine(o_bid, o_ask, pre_bh, pre_bl, pre_ah, pre_al, x_ok, x_bid, x_ask, x_slip,
                post_bh, post_bl, post_ah, post_al, c_mid, roll, swl, sws, N,
                mode, tgt, ent, mxl, mxs, el, es, xl, xs, upd, risk, stop_mult, trail_mult, max_units, add_step, comm,
                worst=True):
    """Daily-bar engine: each day = pre-fill segment, the 08:00 London fill, post-fill segment, close."""
    nb = len(c_mid)
    pnl = np.zeros(nb)
    gpnl = np.zeros(nb)
    swp = np.zeros(nb)
    TR = np.zeros((20000, 8))   # entry k, exit k, dir, units, net, gross, swap, hold
    nt = 0
    pos = 0
    nu = 0
    e = np.zeros(4)
    w = np.zeros(4)
    mk = np.zeros(4)
    gm = np.zeros(4)
    stop = np.nan
    hc = 0.0
    N0 = 0.0
    R0 = 0.0
    add_lv = np.nan
    ek = 0
    t_net = 0.0
    t_gr = 0.0
    t_sw = 0.0
    slip = 0.0
    half = 0.0
    for k in range(nb):
        if x_ok[k]:
            slip = x_slip[k]
            half = 0.5 * (x_ask[k] - x_bid[k])
        exited = False
        for ph in range(3):
            if ph == 1:
                # --------------------------------------------------- the fill minute: market orders
                if not x_ok[k]:
                    continue
                want = 0
                xp = np.nan
                if mode == 0:
                    s = tgt[k]
                    if np.isfinite(s):
                        if pos != 0 and s != pos:
                            xp = x_bid[k] - slip if pos > 0 else x_ask[k] + slip
                        if s != 0:
                            want = int(s)
                else:
                    if pos != 0 and ((pos > 0 and mxl[k]) or (pos < 0 and mxs[k])):
                        xp = x_bid[k] - slip if pos > 0 else x_ask[k] + slip
                    if np.isfinite(ent[k]) and ent[k] != 0:
                        want = int(ent[k])
                if np.isfinite(xp) and pos != 0:
                    gx = xp + pos * (slip + half)
                    for u in range(nu):
                        v = w[u] * pos * (xp - mk[u]) / e[u] - comm * w[u]
                        g = w[u] * pos * (gx - gm[u]) / e[u]
                        pnl[k] += v
                        gpnl[k] += g
                        t_net += v
                        t_gr += g
                    if nt < 20000:
                        TR[nt, 0] = ek
                        TR[nt, 1] = k
                        TR[nt, 2] = pos
                        TR[nt, 3] = nu
                        TR[nt, 4] = t_net
                        TR[nt, 5] = t_gr
                        TR[nt, 6] = t_sw
                        TR[nt, 7] = k - ek
                        nt += 1
                    pos = 0
                    nu = 0
                    exited = True
                if pos == 0 and want != 0 and np.isfinite(N[k]) and N[k] > 0:
                    fill = x_ask[k] + slip if want > 0 else x_bid[k] - slip
                    pos = want
                    N0 = N[k]
                    R0 = stop_mult * N0 if stop_mult > 0 else 2.0 * N0
                    stop = fill - pos * R0 if stop_mult > 0 else np.nan
                    nu = 1
                    e[0] = fill
                    w[0] = risk * fill / R0
                    mk[0] = fill
                    gm[0] = fill - pos * (slip + half)
                    pnl[k] -= comm * w[0]
                    t_net = -comm * w[0]
                    t_gr = 0.0
                    t_sw = 0.0
                    hc = gm[0]
                    add_lv = fill + pos * add_step * N0 if max_units > 1 else np.nan
                    ek = k
                continue
            # ------------------------------------------------------- a price segment (pre-fill or post-fill)
            if ph == 0:
                if not np.isfinite(pre_bh[k]):
                    continue
                sob, soa, bh_, bl_, ah_, al_ = o_bid[k], o_ask[k], pre_bh[k], pre_bl[k], pre_ah[k], pre_al[k]
            else:
                if not np.isfinite(post_bh[k]):
                    continue
                sob, soa, bh_, bl_, ah_, al_ = x_bid[k], x_ask[k], post_bh[k], post_bl[k], post_ah[k], post_al[k]
            entered = False
            for step in range(2):
                if pos == 0 and step == 0 and mode == 1 and not exited:
                    lt = np.isfinite(el[k]) and ah_ >= el[k]
                    st_ = np.isfinite(es[k]) and bl_ <= es[k]
                    if (lt != st_) and np.isfinite(N[k]) and N[k] > 0:
                        d = 1 if lt else -1
                        fill = max(el[k], soa) + slip if d > 0 else min(es[k], sob) - slip
                        pos = d
                        N0 = N[k]
                        R0 = stop_mult * N0 if stop_mult > 0 else 2.0 * N0
                        stop = fill - pos * R0 if stop_mult > 0 else np.nan
                        nu = 1
                        e[0] = fill
                        w[0] = risk * fill / R0
                        mk[0] = fill
                        gm[0] = fill - pos * (slip + half)
                        pnl[k] -= comm * w[0]
                        t_net = -comm * w[0]
                        t_gr = 0.0
                        t_sw = 0.0
                        hc = gm[0]
                        add_lv = fill + pos * add_step * N0 if max_units > 1 else np.nan
                        ek = k
                        entered = True
                if pos == 0:
                    continue
                # adds (worst case: added, then the stop is checked against the whole segment)
                added = False
                while nu < max_units and np.isfinite(add_lv) and ((pos > 0 and ah_ >= add_lv) or (pos < 0 and al_ <= add_lv)):
                    px = max(add_lv, soa) + slip if pos > 0 else min(add_lv, sob) - slip
                    e[nu] = px
                    w[nu] = risk * px / R0
                    mk[nu] = px
                    gm[nu] = px - pos * (slip + half)
                    pnl[k] -= comm * w[nu]
                    t_net -= comm * w[nu]
                    nu += 1
                    add_lv = add_lv + pos * add_step * N0
                    added = True
                    if stop_mult > 0:
                        stop = px - pos * stop_mult * N0
                if not worst and (entered or added):
                    break
                # exits: stop or resting channel level
                if pos > 0:
                    lev = stop
                    if np.isfinite(xl[k]) and (not np.isfinite(lev) or xl[k] > lev):
                        lev = xl[k]
                    hit = np.isfinite(lev) and (bl_ <= lev)
                    xp = (sob - slip if (sob <= lev and not entered) else lev - slip) if hit else np.nan
                else:
                    lev = stop
                    if np.isfinite(xs[k]) and (not np.isfinite(lev) or xs[k] < lev):
                        lev = xs[k]
                    hit = np.isfinite(lev) and (ah_ >= lev)
                    xp = (soa + slip if (soa >= lev and not entered) else lev + slip) if hit else np.nan
                if hit:
                    gx = xp + pos * (slip + half)
                    for u in range(nu):
                        v = w[u] * pos * (xp - mk[u]) / e[u] - comm * w[u]
                        g = w[u] * pos * (gx - gm[u]) / e[u]
                        pnl[k] += v
                        gpnl[k] += g
                        t_net += v
                        t_gr += g
                    if nt < 20000:
                        TR[nt, 0] = ek
                        TR[nt, 1] = k
                        TR[nt, 2] = pos
                        TR[nt, 3] = nu
                        TR[nt, 4] = t_net
                        TR[nt, 5] = t_gr
                        TR[nt, 6] = t_sw
                        TR[nt, 7] = k - ek
                        nt += 1
                    pos = 0
                    nu = 0
                    exited = True
                break
        # ----------------------------------------------------------- close: mark, trail, swap into the next bar
        if pos != 0:
            for u in range(nu):
                v = w[u] * pos * (c_mid[k] - mk[u]) / e[u]
                g = w[u] * pos * (c_mid[k] - gm[u]) / e[u]
                pnl[k] += v
                gpnl[k] += g
                t_net += v
                t_gr += g
                mk[u] = c_mid[k]
                gm[u] = c_mid[k]
            if k + 1 < nb:
                if upd[k]:
                    if pos > 0:
                        hc = max(hc, c_mid[k])
                        if trail_mult > 0 and np.isfinite(N[k + 1]):
                            ts = hc - trail_mult * N[k + 1]
                            if not np.isfinite(stop) or ts > stop:
                                stop = ts
                    else:
                        hc = min(hc, c_mid[k])
                        if trail_mult > 0 and np.isfinite(N[k + 1]):
                            ts = hc + trail_mult * N[k + 1]
                            if not np.isfinite(stop) or ts < stop:
                                stop = ts
                W = 0.0
                for u in range(nu):
                    W += w[u]
                s_ = W * (swl[k] if pos > 0 else sws[k]) * roll[k + 1]
                pnl[k] += s_
                swp[k] += s_
                t_net += s_
                t_sw += s_
    return pnl, gpnl, swp, TR[:nt]


@njit(cache=True)
def cont_engine(x_ok, x_mid, x_cost, c_mid, roll, swl, sws, ptg, buf):
    """Continuous notional, marked at closes: on day k the old position earns close(k-1) -> fill(k), the new one
    fill(k) -> close(k); a change pays its cost at the fill; swap for the position held at each rollover."""
    nb = len(c_mid)
    pnl = np.zeros(nb)
    gpnl = np.zeros(nb)
    swp = np.zeros(nb)
    p = 0.0
    for k in range(nb):
        if k > 0 and p != 0.0:
            s_ = abs(p) * (swl[k - 1] if p > 0 else sws[k - 1]) * roll[k]
            pnl[k - 1] += s_
            swp[k - 1] += s_
        newp = p
        if x_ok[k] and np.isfinite(ptg[k]):
            b = buf[k] if np.isfinite(buf[k]) else 0.0
            if abs(ptg[k] - p) > b:
                newp = ptg[k]
        if k == 0:
            p = newp
            continue
        if x_ok[k]:
            r_old = p * (x_mid[k] / c_mid[k - 1] - 1.0)
            r_new = newp * (c_mid[k] / x_mid[k] - 1.0)
            pnl[k] += r_old + r_new - abs(newp - p) * x_cost[k]
            gpnl[k] += r_old + r_new
        else:
            r = p * (c_mid[k] / c_mid[k - 1] - 1.0)
            pnl[k] += r
            gpnl[k] += r
        p = newp
    return pnl, gpnl, swp


def run_stop(d: pd.DataFrame, N, mode=1, tgt=None, ent=None, mxl=None, mxs=None, el=None, es=None, xl=None, xs=None,
             upd=None, stop_mult=2.0, trail_mult=0.0, max_units=1, add_step=0.5, fill="x", worst=True):
    nb = len(d)
    nan = np.full(nb, np.nan)
    f = lambda a: as_f(nan if a is None else a)                          # noqa: E731
    b = lambda a: np.ascontiguousarray(np.zeros(nb, np.bool_) if a is None else np.asarray(a, np.bool_))  # noqa: E731
    if fill == "x":
        seg = [d.o_bid, d.o_ask, d.pre_bh, d.pre_bl, d.pre_ah, d.pre_al, d.x_ok, d.x_bid, d.x_ask, d.x_slip,
               d.post_bh, d.post_bl, d.post_ah, d.post_al]
    else:   # A2: fill at the day's open (the previous close), with the 08:00 cost; one segment for the whole day
        om = 0.5 * (d.o_bid + d.o_ask)
        hb = np.fmax(d.pre_bh, d.post_bh)
        lb = np.fmin(d.pre_bl, d.post_bl)
        ha = np.fmax(d.pre_ah, d.post_ah)
        la = np.fmin(d.pre_al, d.post_al)
        half = d.x_half.ffill().bfill()
        sl = d.x_slip.ffill().bfill()
        seg = [d.o_bid, d.o_ask, nan, nan, nan, nan, np.ones(nb, bool), om - half, om + half, sl, hb, lb, ha, la]
    seg = [as_f(s) if i != 6 else np.ascontiguousarray(np.asarray(s, np.bool_)) for i, s in enumerate(seg)]
    return stop_engine(*seg, as_f(d.c_mid), np.ascontiguousarray(d.roll.to_numpy(np.int64)), as_f(d.swl), as_f(d.sws),
                       as_f(N), int(mode), f(tgt), f(ent), b(mxl), b(mxs), f(el), f(es), f(xl), f(xs),
                       b(np.ones(nb, bool) if upd is None else upd), RISK, float(stop_mult), float(trail_mult),
                       int(max_units), float(add_step), COMM, bool(worst))


def run_cont(d: pd.DataFrame, ptg, buf=None, fill="x"):
    nb = len(d)
    if fill == "x":
        xo, xm, xc = d.x_ok, d.x_mid, d.x_cost
    else:
        xo = np.ones(nb, bool)
        xm = 0.5 * (d.o_bid + d.o_ask)
        xc = d.x_cost.ffill().bfill()
    return cont_engine(np.ascontiguousarray(np.asarray(xo, np.bool_)), as_f(xm), as_f(xc), as_f(d.c_mid),
                       np.ascontiguousarray(d.roll.to_numpy(np.int64)), as_f(d.swl), as_f(d.sws), as_f(ptg),
                       as_f(np.zeros(nb) if buf is None else buf))


# ================================================================================ indicators (new code)
def turtleN(d, n=20):
    c1 = d.close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - c1).abs(), (d.low - c1).abs()))
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def prior_max(x, n):
    return x.shift(1).rolling(n, min_periods=n).max()


def prior_min(x, n):
    return x.shift(1).rolling(n, min_periods=n).min()


def ewvol_ann(c, com=60):
    r = np.log(c / c.shift(1))
    return np.sqrt((r * r).ewm(com=com, min_periods=60).mean() * 261)


def month_end(d):
    m = d.month.to_numpy()
    return pd.Series(np.r_[m[1:] != m[:-1], True], index=d.index)


def monthly_pos(d, sig, vol, tv=0.40):
    """sig at month-end closes -> target notional held from the next fill for the month."""
    me = month_end(d)
    v = (sig * tv / vol).where(me).ffill()
    return v.shift(1).to_numpy()


def ewmac_fc(c, speeds=((8, 32, 5.3), (16, 64, 3.75), (32, 128, 2.65), (64, 256, 1.87)), fdm=1.1):
    sp = c.diff().ewm(span=36, min_periods=36).std()
    parts = []
    for f, s, k in speeds:
        parts.append(((c.ewm(span=f, min_periods=s).mean() - c.ewm(span=s, min_periods=s).mean()) / sp * k).clip(-20, 20))
    return (pd.concat(parts, axis=1).mean(axis=1, skipna=False) * fdm).clip(-20, 20)


def vol36(c):
    r = np.log(c / c.shift(1))
    return np.sqrt((r * r).ewm(span=36, min_periods=36).mean() * 256)


def weekly(d):
    """Weekly closes / highs / lows / N; for each D1 bar the index of the last completed week (-1 before)."""
    iso = d.tday.dt.isocalendar()
    key = (iso.year * 100 + iso.week).to_numpy()
    wend = np.r_[key[1:] != key[:-1], True]
    g = pd.DataFrame({"k": key, "high": d.high.to_numpy(), "low": d.low.to_numpy(), "close": d.close.to_numpy()})
    W = g.groupby("k", sort=True).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    cw = np.cumsum(wend)
    last_done = np.r_[-1, cw[:-1] - 1]
    fresh = np.r_[False, last_done[1:] != last_done[:-1]]
    return W.reset_index(drop=True), last_done, fresh, wend


def wmap(x, last_done):
    x = np.asarray(x, float)
    out = np.full(len(last_done), np.nan)
    ok = last_done >= 0
    out[ok] = x[last_done[ok]]
    return out


# ================================================================================ systems
def sys_round13(d, rng, fill="x"):
    """The round-13 systems, rebuilt from their written rules."""
    c, h, lo = d.close, d.high, d.low
    N = turtleN(d).shift(1).to_numpy()
    R = {}
    R["L1"] = run_stop(d, N, el=prior_max(h, 55), es=prior_min(lo, 55), xl=prior_min(lo, 20), xs=prior_max(h, 20),
                       stop_mult=2, max_units=4, add_step=0.5, fill=fill)
    R["L2"] = run_stop(d, N, el=prior_max(h, 20), es=prior_min(lo, 20), xl=prior_min(lo, 10), xs=prior_max(h, 10),
                       stop_mult=2, max_units=4, add_step=0.5, fill=fill)
    R["L4"] = run_stop(d, N, el=prior_max(h, 100), es=prior_min(lo, 100), xl=prior_min(lo, 50), xs=prior_max(h, 50),
                       stop_mult=0, fill=fill)
    m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
    R["L5"] = run_stop(d, N, mode=0, tgt=np.sign(m50 - m200).where(m200.notna()).shift(1), stop_mult=0, fill=fill)
    vol = ewvol_ann(c)
    r12, r3, r1 = c / c.shift(252) - 1, c / c.shift(63) - 1, c / c.shift(21) - 1
    R["L6"] = run_cont(d, monthly_pos(d, np.sign(r12), vol), fill=fill)
    R["L7"] = run_cont(d, monthly_pos(d, ((np.sign(r12) + np.sign(r3) + np.sign(r1)) / 3).where(r12.notna()), vol), fill=fill)
    me = month_end(d)
    cm = c[me]
    fab = pd.Series(np.nan, index=c.index)
    fab[me] = np.sign(cm - cm.rolling(10).mean()).to_numpy()
    R["L8"] = run_cont(d, monthly_pos(d, fab, vol), fill=fill)
    fc = ewmac_fc(c)
    v36 = vol36(c)
    R["L9"] = run_cont(d, (fc / 10 * 0.20 / v36).shift(1).to_numpy(), (0.1 * 0.20 / v36).shift(1).to_numpy(), fill=fill)
    s20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    c1 = c.shift(1)
    atr20 = np.maximum(h - lo, np.maximum((h - c1).abs(), (lo - c1).abs())).rolling(20).mean()
    sq = 2 * sd20 < 1.5 * atr20
    rel = sq.shift(1, fill_value=False).astype(bool) & ~sq.astype(bool)
    R["L10"] = run_stop(d, N, ent=pd.Series(np.where(rel, np.sign(c - s20), 0.0), index=c.index).shift(1),
                        stop_mult=2, trail_mult=3, fill=fill)
    W, last_done, fresh, wend = weekly(d)
    wN = turtleN(W).to_numpy()
    lg = (W.close > prior_max(W.high, 20)).astype(float) - (W.close < prior_min(W.low, 20)).astype(float)
    xlg = (W.close < prior_min(W.low, 10)).to_numpy()
    xsh = (W.close > prior_max(W.high, 10)).to_numpy()
    ent = wmap(lg, last_done)
    ent[~fresh] = np.nan
    mxl = np.nan_to_num(wmap(xlg.astype(float), last_done)) > 0
    mxs = np.nan_to_num(wmap(xsh.astype(float), last_done)) > 0
    mxl[~fresh] = False
    mxs[~fresh] = False
    R["L11"] = run_stop(d, wmap(wN, last_done), ent=ent, mxl=mxl, mxs=mxs, stop_mult=2, upd=wend, fill=fill)
    R["BH"] = run_cont(d, monthly_pos(d, pd.Series(1.0, index=c.index).where(vol.notna()), vol), fill=fill)
    rs = pd.Series(np.where(rng.random(len(c)) < 0.5, 1.0, -1.0), index=c.index).where(r12.notna())
    R["PL2"] = run_cont(d, monthly_pos(d, rs, vol), fill=fill)
    # Double Seven on weekly bars, its own exit (state: in from a 7-week closing low above the 200-week average,
    # out at a 7-week closing high), sized at a nominal 2 weekly N
    wc = W.close
    el7 = (wc > wc.rolling(200).mean()) & (wc <= wc.rolling(7).min())
    ex7 = wc >= wc.rolling(7).max()
    state = np.zeros(len(wc))
    s = 0.0
    for i in range(len(wc)):
        if s == 0 and el7.iloc[i]:
            s = 1.0
        elif s == 1 and ex7.iloc[i]:
            s = 0.0
        state[i] = s
    tg = wmap(state, last_done)
    R["D7W"] = run_stop(d, wmap(wN, last_done), mode=0, tgt=tg, stop_mult=0, upd=wend, fill=fill)
    return R


def monthly(d, res):
    pnl, gp, sw = res[0], res[1], res[2]
    df = pd.DataFrame({"pnl": pnl, "gpnl": gp, "swp": sw, "month": d.month.to_numpy()})
    return df.groupby("month").sum()


def portfolio(Mlist):
    if not Mlist:
        return pd.DataFrame(columns=["pnl", "gpnl", "swp"])
    return pd.concat(Mlist).groupby(level=0).sum()


# ================================================================================ part A
def part_A(P, args, out):
    rng = np.random.default_rng(14)
    t0 = time.time()
    rows, store, trades = [], {}, []
    for fill in ("x", "open"):
        per = {}
        for sym, d in P.items():
            R = sys_round13(d, rng, fill=fill)
            for k, res in R.items():
                per.setdefault(k, {})[sym] = monthly(d, res)
                if fill == "x" and len(res) > 3 and len(res[3]):
                    T = pd.DataFrame(res[3], columns=["ek", "xk", "d", "units", "net", "gross", "swap", "hold"])
                    T["system"], T["symbol"] = k, sym
                    T["exit_month"] = d.month.to_numpy()[T.xk.astype(int)]
                    trades.append(T)
        store[fill] = per
        print("part A fills", fill, round(time.time() - t0), "s", flush=True)
    r13 = pd.read_csv(args.r13) if args.r13 and os.path.exists(args.r13) else None
    for k in store["x"]:
        for grp, syms in (("FX", FX), ("NONFX", NONFX), ("ALL", FX + NONFX)):
            mine = portfolio([v for s, v in store["x"][k].items() if s in syms])
            mopen = portfolio([v for s, v in store["open"][k].items() if s in syms])
            if not len(mine):
                continue
            st, sc = pstats(mine.pnl)
            stg, _ = pstats(mine.gpnl)
            sto, _ = pstats(mopen.pnl)
            row = {"system": k, "row": grp, **{f"new_{a}": b for a, b in st.items()},
                   "new_check_gross_t": stg["check_t"], "open_fill_check_mean_pct": sto["check_mean_pct"],
                   "open_fill_check_t": sto["check_t"], "open_fill_fit_t": sto["fit_t"]}
            if r13 is not None and k in set(r13.system):
                o = r13[(r13.system == k) & r13.symbol.isin(syms)].groupby("month")[["pnl", "gpnl", "swp"]].sum()
                if len(o):
                    so, _ = pstats(o.pnl)
                    sog, _ = pstats(o.gpnl)
                    g = month_grid()
                    a1, b1 = mine.gpnl.reindex(g).fillna(0), o.gpnl.reindex(g).fillna(0)
                    a2, b2 = mine.pnl.reindex(g).fillna(0), o.pnl.reindex(g).fillna(0)
                    row.update({"r13_fit_t": so["fit_t"], "r13_check_mean_pct": so["check_mean_pct"],
                                "r13_check_t": so["check_t"], "r13_check_gross_t": sog["check_t"],
                                "corr_gross": float(a1.corr(b1)), "corr_net": float(a2.corr(b2)),
                                "sum_gross_new": float(a1.sum()), "sum_gross_r13": float(b1.sum()),
                                "sum_net_new": float(a2.sum()), "sum_net_r13": float(b2.sum())})
                    row["agree"] = bool(row["corr_gross"] >= 0.9 and np.sign(so["check_mean_pct"]) == np.sign(st["check_mean_pct"]))
            rows.append(row)
    A = pd.DataFrame(rows)
    # Double Seven against round 13's breakdown (sums in R and t by group)
    if args.r13diag and os.path.exists(args.r13diag):
        dg = json.load(open(args.r13diag)).get("S.DAILY.DOUBLE_SEVEN|W1|T0", {})
        for grp, syms in (("FX", FX), ("ALL", FX + NONFX)):
            mine = portfolio([v for s, v in store["x"]["D7W"].items() if s in syms]).pnl.reindex(month_grid()).fillna(0)
            key = "all" if grp == "ALL" else "FX"
            i = A.index[(A.system == "D7W") & (A.row == grp)]
            if len(i) and key in dg:
                A.loc[i, "r13_check_sum_R"] = dg[key]["check_sum_R"]
                A.loc[i, "r13_check_t"] = dg[key]["check_t"]
                A.loc[i, "r13_fit_sum_R"] = dg[key]["fit_sum_R"]
                A.loc[i, "new_check_sum_R"] = mine[mine.index >= CHK[0]].sum() / RISK
                A.loc[i, "new_fit_sum_R"] = mine[mine.index <= FIT[1]].sum() / RISK
                A.loc[i, "new_check_t_raw"] = nw_t(mine[mine.index >= CHK[0]])
    A.to_csv(os.path.join(out, "A1_A2_compare.csv"), index=False)
    TT = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    if len(TT):
        TT.to_parquet(os.path.join(out, "A_trades_new.parquet"), index=False)
    return A, store


def part_A3(args, out, store):
    """ECB reference rates: the 28 crosses, close-only systems, gross, 1-day lag."""
    if not args.ecb or not os.path.exists(args.ecb):
        return None
    E = pd.read_csv(args.ecb, usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    E["TIME_PERIOD"] = pd.to_datetime(E.TIME_PERIOD)
    E = E[E.TIME_PERIOD <= DEV_END]
    X = E.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
    X["EUR"] = 1.0
    rows, mon = [], {}
    for sym in FX:
        a, b = sym[:3], sym[3:]
        px = (X[b] / X[a]).dropna()
        px = px[px > 0]
        r = px.pct_change()
        res = {}
        m50, m200 = px.rolling(50).mean(), px.rolling(200).mean()
        res["L5"] = np.sign(m50 - m200).where(m200.notna()) * 0.005 / (2 * (px.diff().abs().rolling(20).mean() / px))
        vol = np.sqrt((np.log(px / px.shift(1)) ** 2).ewm(com=60, min_periods=60).mean() * 261)
        me = pd.Series(px.index.to_period("M"), index=px.index)
        mend = me != me.shift(-1)
        r12, r3, r1 = px / px.shift(252) - 1, px / px.shift(63) - 1, px / px.shift(21) - 1
        mp = lambda s: (s * 0.40 / vol).where(mend).ffill()                       # noqa: E731
        res["L6"] = mp(np.sign(r12))
        res["L7"] = mp(((np.sign(r12) + np.sign(r3) + np.sign(r1)) / 3).where(r12.notna()))
        cm = px[mend]
        fb = pd.Series(np.nan, index=px.index)
        fb[mend] = np.sign(cm - cm.rolling(10).mean()).to_numpy()
        res["L8"] = mp(fb)
        res["L9"] = ewmac_fc(px) / 10 * 0.20 / vol36(px)
        # close Donchian 100 / 50 (state)
        hi100, lo100 = px.shift(1).rolling(100).max(), px.shift(1).rolling(100).min()
        hi50, lo50 = px.shift(1).rolling(50).max(), px.shift(1).rolling(50).min()
        st = np.zeros(len(px))
        s = 0.0
        pv, a1, b1, a2, b2 = px.to_numpy(), hi100.to_numpy(), lo100.to_numpy(), hi50.to_numpy(), lo50.to_numpy()
        for i in range(len(pv)):
            if s >= 0 and np.isfinite(a1[i]) and pv[i] > a1[i]:
                s = 1.0
            elif s <= 0 and np.isfinite(b1[i]) and pv[i] < b1[i]:
                s = -1.0
            elif s > 0 and pv[i] < b2[i]:
                s = 0.0
            elif s < 0 and pv[i] > a2[i]:
                s = 0.0
            st[i] = s
        nominal = 0.005 / (2 * (px.diff().abs().rolling(20).mean() / px))
        res["L4c"] = pd.Series(st, index=px.index) * nominal
        for k, pos in res.items():
            g = (pos.shift(2) * r).fillna(0.0)                 # decided at fixing t, traded at t+1
            mon.setdefault(k, []).append(g.groupby(g.index.to_period("M").astype(str)).sum())
    for k, lst in mon.items():
        port = pd.concat(lst, axis=1).sum(axis=1)
        pre, _ = pstats(port, fit=("1999-03", "2003-12"), chk=("2004-01", "2007-12"))
        full = port.reindex(month_grid("1999-03", "2007-12")).fillna(0)
        st, _ = pstats(port)
        row = {"system": k, "ecb_1999_2007_mean_monthly_raw": float(full.mean()), "ecb_1999_2007_t": nw_t(full),
               "ecb_fit_t": st["fit_t"], "ecb_check_t": st["check_t"],
               "ecb_by_year_raw": {y: round(float(v), 4) for y, v in port.groupby(port.index.str[:4]).sum().items()}}
        if k in store.get("x", {}):
            mine = portfolio([v for s, v in store["x"][k].items() if s in FX]).gpnl.reindex(month_grid()).fillna(0)
            row["corr_with_store_gross_2008_2025"] = float(mine.corr(port.reindex(month_grid()).fillna(0)))
        rows.append(row)
    A3 = pd.DataFrame(rows)
    A3.to_csv(os.path.join(out, "A3_ecb.csv"), index=False)
    return A3


# ================================================================================ part B
def b_family_runs(d, sym, idx_pos=None):
    """Every member of the B1 speed grids, B2 long-or-flat, B4, B5 for one instrument. Returns {(row, member): res}."""
    c, h, lo = d.close, d.high, d.low
    N = turtleN(d).shift(1).to_numpy()
    out = {}
    for n in (10, 20, 50, 100, 200, 250):
        out[("B1a", n)] = run_stop(d, N, el=prior_max(h, n), es=prior_min(lo, n), xl=prior_min(lo, n // 2),
                                   xs=prior_max(h, n // 2), stop_mult=2)
    for f_, s_ in ((5, 20), (10, 50), (20, 100), (50, 200), (100, 400)):
        out[("B1b", f"{f_}/{s_}")] = run_stop(d, N, mode=0, tgt=np.sign(c.rolling(f_).mean() - c.rolling(s_).mean())
                                              .where(c.rolling(s_).mean().notna()).shift(1), stop_mult=0)
    vol = ewvol_ann(c)
    for m in (1, 3, 6, 9, 12, 24):
        out[("B1c", m)] = run_cont(d, monthly_pos(d, np.sign(c / c.shift(21 * m) - 1), vol))
    v36 = vol36(c)
    buf = (0.1 * 0.20 / v36).shift(1).to_numpy()
    for (f_, s_, k) in ((2, 8, 10.6), (4, 16, 7.5), (8, 32, 5.3), (16, 64, 3.75), (32, 128, 2.65), (64, 256, 1.87)):
        fc = ewmac_fc(c, speeds=((f_, s_, k),), fdm=1.0)
        out[("B1d", f"{f_}/{s_}")] = run_cont(d, (fc / 10 * 0.20 / v36).shift(1).to_numpy(), buf)
    fc9 = ewmac_fc(c)
    if sym in NONFX:
        m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
        out[("B2_L5", "lf")] = run_stop(d, N, mode=0, tgt=(np.sign(m50 - m200).clip(lower=0)).where(m200.notna()).shift(1),
                                       stop_mult=0)
        out[("B2_L6", "lf")] = run_cont(d, monthly_pos(d, np.sign(c / c.shift(252) - 1).clip(lower=0), vol))
        me = month_end(d)
        cm = c[me]
        fab = pd.Series(np.nan, index=c.index)
        fab[me] = np.sign(cm - cm.rolling(10).mean()).clip(lower=0).to_numpy()
        out[("B2_L8", "lf")] = run_cont(d, monthly_pos(d, fab, vol))
        out[("B2_L9", "lf")] = run_cont(d, (fc9.clip(lower=0) / 10 * 0.20 / v36).shift(1).to_numpy(), buf)
        out[("BH", "bh")] = run_cont(d, monthly_pos(d, pd.Series(1.0, index=c.index).where(vol.notna()), vol))
    if sym in FX:
        vol_a = np.sqrt((np.log(c / c.shift(1)) ** 2).ewm(span=36, min_periods=36).mean() * 256)
        carry_fc = (pd.Series(d.rd.to_numpy(), index=c.index) / vol_a * 30).clip(-20, 20)
        comb = (fc9 + carry_fc) / 2
        out[("B4", "ewmac+carry")] = run_cont(d, (comb / 10 * 0.20 / v36).shift(1).to_numpy(), buf)
    # B5: L9 only when 21-day vol at the month-end is above its trailing 252-day median
    r = np.log(c / c.shift(1))
    v21 = r.rolling(21).std()
    on = (v21 > v21.rolling(252).median()).astype(float).where(v21.rolling(252).median().notna())
    me = month_end(d)
    reg = on.where(me).ffill().shift(1)
    out[("B5", "highvol")] = run_cont(d, ((fc9 / 10 * 0.20 / v36).shift(1) * reg).to_numpy(), buf)
    return out


def strength_positions(P):
    """B3: the eight currency-strength indices and the cross positions they imply, per system."""
    cl = {}
    for s in FX:
        if s in P:
            d = P[s]
            cl[s] = pd.Series(d.close.to_numpy(), index=d.tday)
    C = pd.DataFrame(cl).sort_index()
    C = C[~C.index.duplicated()]
    LR = np.log(C / C.shift(1))
    idx = {}
    for cur in CUR8:
        parts = []
        for s in LR.columns:
            if s[:3] == cur:
                parts.append(LR[s])
            elif s[3:] == cur:
                parts.append(-LR[s])
        if parts:
            idx[cur] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    I = pd.DataFrame(idx).fillna(0.0)
    lvl = np.exp(I.cumsum())
    q = {}
    for cur in I.columns:
        L = lvl[cur]
        vol = np.sqrt((I[cur] ** 2).ewm(com=60, min_periods=60).mean() * 261)
        months = pd.Series(L.index.to_period("M"), index=L.index)
        mend = months != months.shift(-1)
        q.setdefault("B3_L6", {})[cur] = (np.sign(L / L.shift(252) - 1) * 0.40 / vol).where(mend).ffill()
        v36_ = np.sqrt((I[cur] ** 2).ewm(span=36, min_periods=36).mean() * 256)
        q.setdefault("B3_L9", {})[cur] = ewmac_fc(L) / 10 * 0.20 / v36_
        hi, lo_, hi2, lo2 = L.shift(1).rolling(50).max(), L.shift(1).rolling(50).min(), L.shift(1).rolling(25).max(), L.shift(1).rolling(25).min()
        st = np.zeros(len(L))
        s = 0.0
        lv, a1, b1, a2, b2 = L.to_numpy(), hi.to_numpy(), lo_.to_numpy(), hi2.to_numpy(), lo2.to_numpy()
        for i in range(len(lv)):
            if s >= 0 and np.isfinite(a1[i]) and lv[i] > a1[i]:
                s = 1.0
            elif s <= 0 and np.isfinite(b1[i]) and lv[i] < b1[i]:
                s = -1.0
            elif s > 0 and np.isfinite(b2[i]) and lv[i] < b2[i]:
                s = 0.0
            elif s < 0 and np.isfinite(a2[i]) and lv[i] > a2[i]:
                s = 0.0
            st[i] = s
        q.setdefault("B3_BO", {})[cur] = pd.Series(st, index=L.index) * 0.20 / v36_
    pos = {}
    for k, qq in q.items():
        Q = pd.DataFrame(qq)
        for s in FX:
            if s in P:
                qa = Q[s[:3]] if s[:3] in Q else 0.0
                qb = Q[s[3:]] if s[3:] in Q else 0.0
                pos.setdefault(k, {})[s] = ((qa - qb) / 7.0)
    return pos


def part_B(P, out):
    t0 = time.time()
    mem = {}
    for sym, d in P.items():
        for (row, member), res in b_family_runs(d, sym).items():
            mem.setdefault((row, member), {})[sym] = monthly(d, res)
    sp = strength_positions(P)
    for k, per in sp.items():
        for sym, ps in per.items():
            d = P[sym]
            ptg = ps.reindex(d.tday).shift(1).to_numpy()
            res = run_cont(d, ptg, None)
            mem.setdefault((k, "idx"), {})[sym] = monthly(d, res)
    print("part B runs", round(time.time() - t0), "s", flush=True)
    surf = []
    bh = portfolio([v for s, v in mem.get(("BH", "bh"), {}).items()])
    bh_st, bh_sc = pstats(bh.pnl) if len(bh) else ({}, np.nan)
    for (row, member), per in mem.items():
        for grp, syms in (("FX", FX), ("NONFX", NONFX)):
            m = portfolio([v for s, v in per.items() if s in syms])
            if not len(m):
                continue
            st, sc = pstats(m.pnl)
            stg, _ = pstats(m.gpnl)
            rec = {"row": row, "member": str(member), "grp": grp, **st, "check_gross_t": stg["check_t"]}
            if grp == "NONFX" and len(bh):
                g = month_grid()
                ex = m.pnl.reindex(g).fillna(0) * sc - bh.pnl.reindex(g).fillna(0) * bh_sc
                rec["excess_check_t"] = nw_t(ex[ex.index >= CHK[0]])
                rec["excess_check_mean_pct"] = float(ex[ex.index >= CHK[0]].mean() * 100)
                if row.startswith("B2"):
                    import statsmodels.api as sm
                    y = m.pnl.reindex(g).fillna(0) * sc
                    x = bh.pnl.reindex(g).fillna(0) * bh_sc
                    for per_, sel in (("fit", y.index <= FIT[1]), ("check", y.index >= CHK[0])):
                        o = sm.OLS(y[sel].to_numpy(), sm.add_constant(x[sel].to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
                        rec[f"alpha_{per_}_pct"] = float(o.params[0] * 100)
                        rec[f"alpha_{per_}_t"] = float(o.tvalues[0])
                        rec[f"beta_{per_}"] = float(o.params[1])
            surf.append(rec)
    S = pd.DataFrame(surf)
    S.to_csv(os.path.join(out, "B_surface.csv"), index=False)
    # primary rows (pre-registered)
    prim = []
    def pick(row, grp):
        g = S[(S.row == row) & (S.grp == grp)]
        if not len(g):
            return None
        r = g.loc[g.fit_t.idxmax()].to_dict()
        r["rowname"] = f"{row}|{grp}"
        r["members"] = int(len(g))
        return r
    for row in ("B1a", "B1b", "B1c", "B1d"):
        for grp in ("FX", "NONFX"):
            r = pick(row, grp)
            if r:
                prim.append(r)
    for row in ("B2_L5", "B2_L6", "B2_L8", "B2_L9"):
        r = pick(row, "NONFX")
        if r:
            prim.append(r)
    for row in ("B3_L6", "B3_L9", "B3_BO", "B4"):
        r = pick(row, "FX")
        if r:
            prim.append(r)
    for grp in ("FX", "NONFX"):
        r = pick("B5", grp)
        if r:
            prim.append(r)
    return pd.DataFrame(prim)


# ================================================================================ part C
def features(d):
    c = d.close
    r = np.log(c / c.shift(1))
    v63 = r.rolling(63).std()
    v252 = r.rolling(252).std()
    F = pd.DataFrame(index=d.index)
    for m in (1, 3, 6, 12):
        F[f"r{m}"] = np.log(c / c.shift(21 * m)) / (v63 * math.sqrt(21 * m))
    rd = pd.Series(d.rd.to_numpy(), index=d.index)
    F["carry"] = rd
    F["drd3"] = rd - rd.shift(63)
    F["drd12"] = rd - rd.shift(252)
    F["value"] = np.log(c) - np.log(c).rolling(756, min_periods=500).mean()
    F["vol63"] = v63 * math.sqrt(261)
    F["vratio"] = v63 / v252
    F["skew63"] = r.rolling(63).skew()
    hi, lo = c.rolling(252).max(), c.rolling(252).min()
    F["rangepos"] = (c - lo) / (hi - lo)
    # trendiness predictors
    r5 = np.log(c / c.shift(5))
    G = pd.DataFrame(index=d.index)
    G["vr5"] = r5.rolling(252).var() / (5 * r.rolling(252).var())
    G["ac1d"] = r.rolling(252).corr(r.shift(1))
    G["ac1w"] = r5.rolling(260).corr(r5.shift(5))
    G["er63"] = (c - c.shift(63)).abs() / c.diff().abs().rolling(63).sum()
    G["er252"] = (c - c.shift(252)).abs() / c.diff().abs().rolling(252).sum()
    h, lo1 = d.high, d.low
    up, dn = h.diff(), -lo1.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    c1 = c.shift(1)
    tr = np.maximum(h - lo1, np.maximum((h - c1).abs(), (lo1 - c1).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    G["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / 14, adjust=False).mean()
    G["tstrength"] = np.log(c / c.shift(252)).abs() / (v252 * math.sqrt(252))
    G["vr1_12"] = r.rolling(21).std() / v252
    return F, G, v63


def monthly_panel(P):
    rows = []
    fc_all = {}
    for sym, d in P.items():
        F, G, v63 = features(d)
        me = month_end(d).to_numpy()
        c = d.close
        fc = ewmac_fc(c)
        idx = np.flatnonzero(me)
        cm = c.to_numpy()[idx]
        nxt = np.r_[cm[1:], np.nan]
        dn = np.r_[np.diff(idx), np.nan]                       # trading days in the next month
        ret_next = np.log(nxt / cm) / (v63.to_numpy()[idx] * np.sqrt(dn))
        df = pd.concat([F.iloc[idx].reset_index(drop=True), G.iloc[idx].reset_index(drop=True)], axis=1)
        df["sym"] = sym
        df["month"] = d.month.to_numpy()[idx]
        df["y_dir"] = ret_next
        df["fc9_sign"] = np.sign(fc.to_numpy()[idx])
        df["y_trend"] = df.fc9_sign * df.y_dir
        df["bar"] = idx
        rows.append(df)
        fc_all[sym] = fc
    M = pd.concat(rows, ignore_index=True)
    M["fmonth"] = (pd.PeriodIndex(M.month, freq="M") + 1).astype(str)        # the month being forecast
    # trailing 12-month realised trend P&L (known at the month-end)
    M = M.sort_values(["sym", "month"])
    M["past_trend"] = M.groupby("sym").y_trend.transform(lambda s: s.shift(1).rolling(12, min_periods=6).sum())
    # market-level predictors: FX average pairwise correlation (63 days) and cross-sectional dispersion
    cl = pd.DataFrame({s: pd.Series(P[s].close.to_numpy(), index=P[s].tday) for s in FX if s in P}).sort_index()
    cl = cl[~cl.index.duplicated()]
    lr = np.log(cl / cl.shift(1))
    mon = pd.Series(lr.index.to_period("M").astype(str), index=lr.index)
    ac = {}
    for mth, g in lr.groupby(mon):
        end = g.index[-1]
        win = lr.loc[:end].tail(63)
        cm_ = win.corr().to_numpy()
        iu = np.triu_indices_from(cm_, 1)
        ac[mth] = float(np.nanmean(cm_[iu]))
    M["avgcorr"] = M.month.map(ac)
    M["disp"] = M.groupby("month").r3.transform("std")
    return M, fc_all


FEAT_DIR = ["r1", "r3", "r6", "r12", "carry", "drd3", "drd12", "value", "vol63", "vratio", "skew63", "rangepos"]
FEAT_TR = ["vr5", "ac1d", "ac1w", "er63", "er252", "adx", "tstrength", "vr1_12", "past_trend", "avgcorr", "disp"]


def walk_forward(M, feats, target, model, years=range(2011, 2026)):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    pred = pd.Series(np.nan, index=M.index)
    for Y in years:
        tr = M[(M.fmonth < f"{Y}-01") & (M.fmonth >= "2008-01")].dropna(subset=feats + [target])
        te = M[(M.fmonth >= f"{Y}-01") & (M.fmonth <= f"{Y}-12")]
        te = te.dropna(subset=feats)
        if len(tr) < 200 or not len(te):
            continue
        X, y = tr[feats].to_numpy(), tr[target].clip(-5, 5).to_numpy()
        if model == "ridge":
            mu, sd = X.mean(0), X.std(0) + 1e-12
            m = Ridge(alpha=10.0 * len(feats)).fit((X - mu) / sd, y)
            pred[te.index] = m.predict((te[feats].to_numpy() - mu) / sd)
        else:
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=50,
                                              random_state=0).fit(X, y)
            pred[te.index] = m.predict(te[feats].to_numpy())
    return pred


def part_C(P, out):
    import statsmodels.api as sm
    t0 = time.time()
    M, fc_all = monthly_panel(P)
    M.to_parquet(os.path.join(out, "C_panel.parquet"), index=False)
    print("panel", M.shape, round(time.time() - t0), "s", flush=True)
    rows = []
    # C2a: single trendiness predictors (pooled slope, month-clustered)
    single = []
    for f in FEAT_TR:
        g = M.dropna(subset=[f, "y_trend"])
        fit = g[(g.fmonth >= "2008-01") & (g.fmonth <= "2018-12")]
        chk = g[(g.fmonth >= "2019-01") & (g.fmonth <= "2025-12")]
        if len(fit) < 200 or len(chk) < 100:
            continue
        mu, sd = fit[f].mean(), fit[f].std()
        res = {}
        for nm, s in (("fit", fit), ("check", chk)):
            z = (s[f] - mu) / sd
            o = sm.OLS(s.y_trend.clip(-5, 5).to_numpy(), sm.add_constant(z.to_numpy())).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(s.fmonth)[0]})
            res[nm] = (float(o.params[1]), float(o.tvalues[1]))
        sgn = np.sign(res["fit"][0])
        single.append({"predictor": f, "fit_slope": res["fit"][0], "fit_t": res["fit"][1], "check_slope": res["check"][0],
                       "check_t": res["check"][1], "check_t_in_fit_sign": sgn * res["check"][1]})
    S = pd.DataFrame(single)
    from scipy.stats import norm
    S["p"] = 1 - norm.cdf(S.check_t_in_fit_sign)
    S["bh_pass"] = bh_pass(S.p.to_numpy())
    S.to_csv(os.path.join(out, "C2a_single.csv"), index=False)
    # walk-forward models
    for model in ("ridge", "gbm"):
        M[f"pd_{model}"] = walk_forward(M, FEAT_DIR, "y_dir", model)
        M[f"pt_{model}"] = walk_forward(M, FEAT_TR, "y_trend", model)
    M.to_parquet(os.path.join(out, "C_panel_pred.parquet"), index=False)
    print("models", round(time.time() - t0), "s", flush=True)
    # trade them
    mon = {}
    for sym, d in P.items():
        c = d.close
        vol = ewvol_ann(c)
        g = M[M.sym == sym].set_index("bar")
        v36 = vol36(c)
        p9 = (fc_all[sym] / 10 * 0.20 / v36)
        buf = (0.1 * 0.20 / v36).shift(1).to_numpy()
        for model in ("ridge", "gbm"):
            s = pd.Series(np.nan, index=range(len(d)))
            s[g.index] = np.sign(g[f"pd_{model}"]).to_numpy()
            sig = pd.Series(s.to_numpy(), index=d.index)
            mon.setdefault(("C1", model), {})[sym] = monthly(d, run_cont(d, monthly_pos(d, sig, vol)))
            sw = pd.Series(np.nan, index=range(len(d)))
            sw[g.index] = np.where(g[f"pt_{model}"].isna(), np.nan, np.where(g[f"pt_{model}"] > 0, 1.0, -1.0))
            swm = pd.Series(sw.to_numpy(), index=d.index).ffill().shift(1)
            mon.setdefault(("C2b_rev", model), {})[sym] = monthly(d, run_cont(d, (p9.shift(1) * swm).to_numpy(), buf))
            mon.setdefault(("C2b_flat", model), {})[sym] = monthly(d, run_cont(d, (p9.shift(1) * swm.clip(lower=0)).to_numpy(), buf))
        mon.setdefault(("C_L9ref", "-"), {})[sym] = monthly(d, run_cont(d, p9.shift(1).to_numpy(), buf))
    for (row, model), per in mon.items():
        groups = (("FX", FX), ("ALL", FX + NONFX)) if row == "C1" else (("ALL", FX + NONFX),)
        for grp, syms in groups:
            m = portfolio([v for s, v in per.items() if s in syms])
            st, _ = pstats(m.pnl, fit=("2011-01", "2018-12"))
            stg, _ = pstats(m.gpnl, fit=("2011-01", "2018-12"))
            rows.append({"rowname": f"{row}|{model}|{grp}", "row": row, "member": model, "grp": grp, **st,
                         "check_gross_t": stg["check_t"]})
    # prediction quality (information): pooled rank correlation of prediction and target by period
    q = []
    for col, tgt in (("pd_ridge", "y_dir"), ("pd_gbm", "y_dir"), ("pt_ridge", "y_trend"), ("pt_gbm", "y_trend")):
        for nm, lo_, hi_ in (("fit_oos", "2011-01", "2018-12"), ("check", "2019-01", "2025-12")):
            g = M[(M.fmonth >= lo_) & (M.fmonth <= hi_)].dropna(subset=[col, tgt])
            ics = g.groupby("fmonth").apply(lambda x: x[col].rank().corr(x[tgt].rank()) if len(x) > 5 else np.nan)
            q.append({"pred": col, "period": nm, "mean_rank_ic": float(ics.mean()), "ic_t": nw_t(ics.dropna()),
                      "hit_rate": float((np.sign(g[col]) == np.sign(g[tgt])).mean())})
    pd.DataFrame(q).to_csv(os.path.join(out, "C_prediction_quality.csv"), index=False)
    return pd.DataFrame(rows), S


# ================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--r13", default="")
    ap.add_argument("--r13diag", default="")
    ap.add_argument("--ecb", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="A,B,C")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = load_panels(a.inp)
    print("panels", len(P), sorted(P), flush=True)
    parts = set(a.parts.split(","))
    summ = {}
    store = {}
    if "A" in parts:
        A, store = part_A(P, a, a.out)
        cols = [c for c in ["system", "row", "new_fit_t", "new_check_mean_pct", "new_check_t", "new_check_gross_t",
                            "r13_fit_t", "r13_check_mean_pct", "r13_check_t", "r13_check_gross_t", "corr_gross", "corr_net",
                            "agree", "open_fill_check_t", "r13_check_sum_R", "new_check_sum_R"] if c in A.columns]
        pd.set_option("display.width", 250)
        print(A[cols].round(2).to_string(index=False), flush=True)
        a3 = part_A3(a, a.out, store)
        if a3 is not None:
            print(a3.drop(columns=["ecb_by_year_raw"]).round(3).to_string(index=False), flush=True)
        if "agree" in A:
            summ["A_agree"] = A[A.row.isin(["FX", "NONFX"])].dropna(subset=["agree"]).groupby("system").agree.all().to_dict()
    prim = []
    if "B" in parts:
        prim.append(part_B(P, a.out))
    if "C" in parts:
        Cr, S = part_C(P, a.out)
        prim.append(Cr[Cr.row != "C_L9ref"])
        summ["C2a"] = S[["predictor", "fit_t", "check_t_in_fit_sign", "bh_pass"]].round(3).to_dict("records")
        Cr.to_csv(os.path.join(a.out, "C_rows.csv"), index=False)
    if prim:
        from scipy.stats import norm
        R = pd.concat(prim, ignore_index=True)
        R["p"] = 1 - norm.cdf(R.check_t.fillna(-9))
        R["bh_pass"] = bh_pass(R.p.to_numpy())
        rule = (R.fit_mean_pct > 0) & (R.check_mean_pct > 0) & (R.check_t >= 1.645)
        nonfx = R.grp.eq("NONFX") & ~R.row.str.startswith("B2")
        if "excess_check_t" in R:
            rule &= ~nonfx | (R.excess_check_t >= 1.645)
        if "alpha_check_t" in R:
            b2 = R.row.str.startswith("B2")
            rule &= ~b2 | ((R.alpha_fit_pct > 0) & (R.alpha_check_t >= 1.645))
        R["meets_rule"] = rule
        R["counts"] = rule & R.bh_pass
        R.to_csv(os.path.join(a.out, "BC_primary.csv"), index=False)
        cols = [c for c in ["rowname", "member", "fit_mean_pct", "fit_t", "check_mean_pct", "check_t", "check_gross_t",
                            "excess_check_t", "alpha_fit_t", "alpha_check_t", "bh_pass", "meets_rule", "counts"] if c in R]
        print(R[cols].round(2).to_string(index=False), flush=True)
        summ["BC_rows"] = int(len(R))
        summ["BC_counts"] = int(R.counts.sum())
        summ["BC_meets_rule"] = int(R.meets_rule.sum())
    json.dump(summ, open(os.path.join(a.out, "trend14_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
