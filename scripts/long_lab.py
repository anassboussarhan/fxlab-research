"""Round 13 (docs/long_preregistration.md): long trades that let winners run.

Per instrument: the long-trend systems L1-L12, the controls PL1 / PL2 / BH, and every D1-allowed catalog strategy on
D1 and W1 bars with the exit variants T0-T4, all priced on the M1 bid / ask path with swaps. Output per instrument:
daily P&L of part A, part-A trade records, monthly P&L and trade statistics of part B.

Usage: python scripts/long_lab.py --symbol EURUSD --out DIR [--only L1,L2] [--no-catalog]"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import signal
import sys
import time
import traceback

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
FX8 = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
NONFX = ("XAUUSD", "XAGUSD", "USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY")
FX_MARKUP, NONFX_MARKUP = 0.0075, 0.025
RISK = 0.005                       # equity fraction at the initial stop per unit
SLIP_FRAC, SLIP_PIPS, COMM = 0.25, 0.1, 0.35e-4
MAXTR = 20000
EXCL_FAMILY = {"machine_learning", "retail_mm", "esoteric"}
KEEP_MULTI = {"S.MS.ROUND_NUMBER", "S.EVT.GOLD_FIX"}
TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "H8", "H12", "D1"]
LOG = {}


def fx28():
    return [a + b for i, a in enumerate(FX8) for b in FX8[i + 1:]]


def market_symbol(a_b: str) -> str:
    """The market convention of a pair of the 8 currencies (EURUSD not USDEUR)."""
    order = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
    a, b = a_b[:3], a_b[3:]
    return a + b if order.index(a) < order.index(b) else b + a


FX_SYMBOLS = sorted({market_symbol(s) for s in fx28()})
ALL_SYMBOLS = FX_SYMBOLS + list(NONFX)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# ============================================================================================ market data
class Mkt:
    """M1 arrays plus the D1 grid: bar of each minute, first minute and the 08:00 London fill minute of each day,
    rollover counts, swap rates per night for a long and a short, and the D1 mid bars used for signals."""

    def __init__(self, m1: pd.DataFrame, sym: str, pip: float):
        from fxlab.data.store import resample
        m1 = m1[m1.index <= DEV_END]
        ok = (m1["ask_o"] >= m1["bid_o"]) & (m1["bid_o"] > 0) & np.isfinite(m1["bid_o"]) & np.isfinite(m1["ask_o"])
        m1 = m1[ok]
        self.sym, self.pip = sym, pip
        self.t = m1.index.as_unit("ns").asi8
        f = lambda c: np.ascontiguousarray(m1[c].to_numpy(np.float64))     # noqa: E731
        self.bo, self.bh, self.bl, self.bc = f("bid_o"), f("bid_h"), f("bid_l"), f("bid_c")
        self.ao, self.ah, self.al, self.ac = f("ask_o"), f("ask_h"), f("ask_l"), f("ask_c")
        self.spr = np.maximum(self.ao - self.bo, 0.0)
        self.slip = SLIP_FRAC * self.spr + SLIP_PIPS * pip
        d1 = resample(m1, "D1")
        d1 = d1[np.isfinite(d1["close"])]
        self.d1 = d1
        dt = d1.index.as_unit("ns").asi8
        self.dk = np.ascontiguousarray(np.searchsorted(dt, self.t, "right") - 1).astype(np.int64)
        nb = len(d1)
        self.first = np.searchsorted(self.dk, np.arange(nb), "left").astype(np.int64)
        last = np.r_[self.first[1:] - 1, len(self.t) - 1]
        self.mclose = np.ascontiguousarray(0.5 * (self.bc[last] + self.ac[last]))
        from fxlab.validation.metrics import trading_day
        self.tday = pd.DatetimeIndex(trading_day(d1.index))                 # naive trading dates
        x = (self.tday + pd.Timedelta(hours=8)).tz_localize("Europe/London", nonexistent="shift_forward",
                                                             ambiguous=True).tz_convert("UTC").as_unit("ns").asi8
        j = np.searchsorted(self.t, x, "left")
        jj = np.minimum(j, len(self.t) - 1)
        good = (j < len(self.t)) & (self.t[jj] - x <= 30 * 60_000_000_000) & (self.dk[jj] == np.arange(nb))
        self.exec = np.where(good, jj, -1).astype(np.int64)
        # rollovers between consecutive bars: weekday 17:00 New York instants (Wednesday x3)
        days = pd.date_range(self.tday[0] - pd.Timedelta(days=3), self.tday[-1] + pd.Timedelta(days=1))
        days = days[days.weekday < 5]
        inst = (days + pd.Timedelta(hours=17)).tz_localize("America/New_York", nonexistent="shift_forward",
                                                            ambiguous="NaT").tz_convert("UTC")
        okk = ~inst.isna()
        it = inst[okk].as_unit("ns").asi8
        wc = np.r_[0, np.cumsum(np.where(days[okk].weekday == 2, 3, 1))]
        tf_ = self.t[self.first]
        tl_ = self.t[last]
        roll = np.zeros(nb, np.int64)
        a = np.searchsorted(it, tl_[:-1], "right")
        b = np.searchsorted(it, tf_[1:], "right")
        roll[1:] = wc[b] - wc[a]
        self.roll = roll
        # swap per night for a long and a short, per bar (known at the bar)
        from fxlab.data.rates import known_at, rate_diff
        idx = d1.index
        if sym[:3] in FX8 and sym[3:] in FX8:
            rd = rate_diff(sym, idx).to_numpy(float)
            self.rd = rd
            self.swl = (rd - FX_MARKUP) / 365.0
            self.sws = (-rd - FX_MARKUP) / 365.0
        else:
            q = sym[-3:] if sym[-3:] in FX8 else "USD"           # quote currency (HKD and US cents -> USD)
            rq = known_at(idx)[q].to_numpy(float)
            self.rd = -rq
            self.swl = (-rq - NONFX_MARKUP) / 365.0
            self.sws = (rq - NONFX_MARKUP) / 365.0
        self.swl = np.nan_to_num(self.swl, nan=-FX_MARKUP / 365)
        self.sws = np.nan_to_num(self.sws, nan=-FX_MARKUP / 365)
        ex = np.maximum(self.exec, 0)
        self.me = 0.5 * (self.bo[ex] + self.ao[ex])
        self.mcost = (self.slip[ex] + 0.5 * self.spr[ex]) / self.me + COMM
        self.mcost[self.exec < 0] = np.nan


# ============================================================================================ event engine
@njit(cache=True)
def ev_run(bo, bh, bl, ao, ah, al, slip, spr, dk, first, ex_m1, roll, swl, sws, mclose, N,
           mode, tgt, ent, mxl, mxs, el, es, xl, xs, sld, tpd, hld, upd,
           risk, stop_mult, trail_mult, max_units, add_step, tp_R, be_R, part_R, max_hold, comm, pip):
    n, nb = len(bo), len(mclose)
    pnl = np.zeros(nb)
    gpnl = np.zeros(nb)
    swp = np.zeros(nb)
    T = np.zeros((20000, 12))          # ek, xk, d, units, net, gross, swap, pips, R/price, hold bars, ei, xi
    nt = 0
    pos = 0
    e = np.zeros(4)
    w = np.zeros(4)
    mk = np.zeros(4)
    gm = np.zeros(4)
    nu = 0
    stop = np.nan
    hc = 0.0
    N0 = 0.0
    R0 = 0.0
    e_first = 0.0
    g_first = 0.0
    tp_lv = np.nan
    be_lv = np.nan
    pt_lv = np.nan
    add_lv = np.nan
    ent_i = -1
    ent_k = -1
    hold_lim = -1
    tr_net = 0.0
    tr_gross = 0.0
    tr_swap = 0.0
    kprev = -1
    for i in range(n):
        k = dk[i]
        if k != kprev:
            if kprev >= 0 and pos != 0:
                for u in range(nu):
                    v = w[u] * pos * (mclose[kprev] - mk[u]) / e[u]
                    g = w[u] * pos * (mclose[kprev] - gm[u]) / e[u]
                    pnl[kprev] += v
                    gpnl[kprev] += g
                    tr_net += v
                    tr_gross += g
                    mk[u] = mclose[kprev]
                    gm[u] = mclose[kprev]
                W = 0.0
                for u in range(nu):
                    W += w[u]
                rate = swl[kprev] if pos > 0 else sws[kprev]
                s_ = W * rate * roll[k]
                pnl[kprev] += s_
                swp[kprev] += s_
                tr_net += s_
                tr_swap += s_
                if upd[kprev]:
                    if pos > 0:
                        if mclose[kprev] > hc:
                            hc = mclose[kprev]
                        if trail_mult > 0 and np.isfinite(N[k]):
                            ts = hc - trail_mult * N[k]
                            if not np.isfinite(stop) or ts > stop:
                                stop = ts
                    else:
                        if mclose[kprev] < hc:
                            hc = mclose[kprev]
                        if trail_mult > 0 and np.isfinite(N[k]):
                            ts = hc + trail_mult * N[k]
                            if not np.isfinite(stop) or ts < stop:
                                stop = ts
            kprev = k
        # ---------------------------------------------------------------- open position: stops, limits, adds
        xfill = np.nan
        xcost = 0.0
        if pos != 0 and i > ent_i:
            if pos > 0:
                s_eff = stop
                if np.isfinite(xl[k]) and (not np.isfinite(s_eff) or xl[k] > s_eff):
                    s_eff = xl[k]
                if np.isfinite(s_eff):
                    if bo[i] <= s_eff:
                        xfill = bo[i] - slip[i]
                        xcost = slip[i] + 0.5 * spr[i]
                    elif bl[i] <= s_eff:
                        xfill = s_eff - slip[i]
                        xcost = slip[i] + 0.5 * spr[i]
                if not np.isfinite(xfill) and np.isfinite(tp_lv) and bh[i] >= tp_lv:
                    xfill = max(tp_lv, bo[i])
                    xcost = 0.5 * spr[i]
            else:
                s_eff = stop
                if np.isfinite(xs[k]) and (not np.isfinite(s_eff) or xs[k] < s_eff):
                    s_eff = xs[k]
                if np.isfinite(s_eff):
                    if ao[i] >= s_eff:
                        xfill = ao[i] + slip[i]
                        xcost = slip[i] + 0.5 * spr[i]
                    elif ah[i] >= s_eff:
                        xfill = s_eff + slip[i]
                        xcost = slip[i] + 0.5 * spr[i]
                if not np.isfinite(xfill) and np.isfinite(tp_lv) and al[i] <= tp_lv:
                    xfill = min(tp_lv, ao[i])
                    xcost = 0.5 * spr[i]
            if not np.isfinite(xfill):
                # partial take-profit (limit), then breakeven and pyramid adds (effective from the next minute)
                if np.isfinite(pt_lv) and ((pos > 0 and bh[i] >= pt_lv) or (pos < 0 and al[i] <= pt_lv)):
                    px = max(pt_lv, bo[i]) if pos > 0 else min(pt_lv, ao[i])
                    gx = px + pos * 0.5 * spr[i]
                    for u in range(nu):
                        hw = 0.5 * w[u]
                        v = hw * pos * (px - mk[u]) / e[u] - comm * hw
                        g = hw * pos * (gx - gm[u]) / e[u]
                        pnl[k] += v
                        gpnl[k] += g
                        tr_net += v
                        tr_gross += g
                        w[u] = hw
                    pt_lv = np.nan
                    if pos > 0 and (not np.isfinite(stop) or e_first > stop):
                        stop = e_first
                    if pos < 0 and (not np.isfinite(stop) or e_first < stop):
                        stop = e_first
                if np.isfinite(be_lv) and ((pos > 0 and bh[i] >= be_lv) or (pos < 0 and al[i] <= be_lv)):
                    if pos > 0 and (not np.isfinite(stop) or e_first > stop):
                        stop = e_first
                    if pos < 0 and (not np.isfinite(stop) or e_first < stop):
                        stop = e_first
                    be_lv = np.nan
                if nu < max_units and np.isfinite(add_lv):
                    hit = False
                    px = 0.0
                    if pos > 0 and ah[i] >= add_lv:
                        hit = True
                        px = max(add_lv, ao[i]) + slip[i]
                    elif pos < 0 and al[i] <= add_lv:
                        hit = True
                        px = min(add_lv, bo[i]) - slip[i]
                    if hit:
                        wu = risk / (R0 / px)
                        e[nu] = px
                        w[nu] = wu
                        mk[nu] = px
                        gm[nu] = px - pos * (slip[i] + 0.5 * spr[i])
                        pnl[k] -= comm * wu
                        tr_net -= comm * wu
                        nu += 1
                        add_lv = add_lv + pos * add_step * N0
                        if stop_mult > 0:
                            stop = px - pos * stop_mult * N0
                # time exit at the day's market fill
                if hold_lim > 0 and i == ex_m1[k] and (k - ent_k) >= hold_lim:
                    if pos > 0:
                        xfill = bo[i] - slip[i]
                    else:
                        xfill = ao[i] + slip[i]
                    xcost = slip[i] + 0.5 * spr[i]
        # ---------------------------------------------------------------- market orders at the day's fill minute
        want = 0
        if i == ex_m1[k]:
            if mode == 0:
                s = tgt[k]
                if np.isfinite(s):
                    if pos != 0 and s != pos and not np.isfinite(xfill):
                        xfill = bo[i] - slip[i] if pos > 0 else ao[i] + slip[i]
                        xcost = slip[i] + 0.5 * spr[i]
                    if s != 0:
                        want = int(s)
            else:
                if pos != 0 and not np.isfinite(xfill) and ((pos > 0 and mxl[k]) or (pos < 0 and mxs[k])):
                    xfill = bo[i] - slip[i] if pos > 0 else ao[i] + slip[i]
                    xcost = slip[i] + 0.5 * spr[i]
                if np.isfinite(ent[k]) and ent[k] != 0:
                    want = int(ent[k])
        # ---------------------------------------------------------------- book an exit
        if np.isfinite(xfill) and pos != 0:
            gx = xfill + pos * xcost
            W = 0.0
            for u in range(nu):
                v = w[u] * pos * (xfill - mk[u]) / e[u] - comm * w[u]
                g = w[u] * pos * (gx - gm[u]) / e[u]
                pnl[k] += v
                gpnl[k] += g
                tr_net += v
                tr_gross += g
                W += w[u]
            if nt < 20000:
                T[nt, 0] = ent_k
                T[nt, 1] = k
                T[nt, 2] = pos
                T[nt, 3] = nu
                T[nt, 4] = tr_net
                T[nt, 5] = tr_gross
                T[nt, 6] = tr_swap
                T[nt, 7] = pos * (gx - g_first) / pip
                T[nt, 8] = R0 / e_first
                T[nt, 9] = k - ent_k
                T[nt, 10] = ent_i
                T[nt, 11] = i
                nt += 1
            pos = 0
            nu = 0
        # ---------------------------------------------------------------- entries
        d_new = 0
        fill = 0.0
        if pos == 0:
            if want != 0 and i == ex_m1[k]:
                d_new = want
                fill = ao[i] + slip[i] if want > 0 else bo[i] - slip[i]
            elif mode == 1 and i >= first[k]:
                lt = np.isfinite(el[k]) and ah[i] >= el[k]
                st_ = np.isfinite(es[k]) and al[i] <= es[k]
                if lt and not st_:
                    d_new = 1
                    fill = max(el[k], ao[i]) + slip[i]
                elif st_ and not lt:
                    d_new = -1
                    fill = min(es[k], bo[i]) - slip[i]
        if d_new != 0 and np.isfinite(N[k]) and N[k] > 0:
            N0 = N[k]
            if np.isfinite(sld[k]) and sld[k] > 0:
                R0 = sld[k]
                stop = fill - d_new * R0
            elif stop_mult > 0:
                R0 = stop_mult * N0
                stop = fill - d_new * R0
            else:
                R0 = 2.0 * N0
                stop = np.nan
            pos = d_new
            nu = 1
            e[0] = fill
            w[0] = risk / (R0 / fill)
            mk[0] = fill
            gm[0] = fill - pos * (slip[i] + 0.5 * spr[i])
            e_first = fill
            g_first = gm[0]
            pnl[k] -= comm * w[0]
            tr_net = -comm * w[0]
            tr_gross = 0.0
            tr_swap = 0.0
            hc = 0.5 * (bo[i] + ao[i])
            if np.isfinite(tpd[k]) and tpd[k] > 0:
                tp_lv = fill + pos * tpd[k]
            elif tp_R > 0:
                tp_lv = fill + pos * tp_R * R0
            else:
                tp_lv = np.nan
            be_lv = fill + pos * be_R * R0 if be_R > 0 else np.nan
            pt_lv = fill + pos * part_R * R0 if part_R > 0 else np.nan
            add_lv = fill + pos * add_step * N0 if max_units > 1 else np.nan
            ent_i = i
            ent_k = k
            if np.isfinite(hld[k]) and hld[k] > 0:
                hold_lim = int(hld[k])
            else:
                hold_lim = max_hold
    # open position at the end: mark to the last close (not recorded as a trade)
    if pos != 0 and kprev >= 0:
        for u in range(nu):
            pnl[kprev] += w[u] * pos * (mclose[kprev] - mk[u]) / e[u]
            gpnl[kprev] += w[u] * pos * (mclose[kprev] - gm[u]) / e[u]
    return pnl, gpnl, swp, T[:nt]


# ============================================================================================ continuous engine
def cont_run_close(M: Mkt, p_target: np.ndarray, buf: np.ndarray):
    """Round 17: as cont_run (decided on the previous close, traded at each day's 08:00 London fill), but marked at
    the closes: on day k the old position earns close(k-1) -> fill(k) and the new one fill(k) -> close(k). Day k's
    P&L is then the close-to-close return, aligned with close-of-day features (cont_run's fill-to-fill marks start
    14 hours before the previous close)."""
    nb = len(M.mclose)
    pnl, gpnl, swp = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    p, changes = 0.0, 0
    for k in range(nb):
        if k > 0 and p != 0.0:
            s_ = abs(p) * (M.swl[k - 1] if p > 0 else M.sws[k - 1]) * M.roll[k]
            pnl[k] += s_
            swp[k] += s_
        newp = p
        if M.exec[k] >= 0:
            tg = p_target[k]
            if np.isfinite(tg) and abs(tg - p) > (buf[k] if np.isfinite(buf[k]) else 0.0):
                newp = float(tg)
        if k == 0:
            p = newp
            continue
        if M.exec[k] >= 0 and newp != p:
            r_old = p * (M.me[k] / M.mclose[k - 1] - 1.0)
            r_new = newp * (M.mclose[k] / M.me[k] - 1.0)
            pnl[k] += r_old + r_new - abs(newp - p) * M.mcost[k]
            gpnl[k] += r_old + r_new
            changes += 1
        else:
            r = p * (M.mclose[k] / M.mclose[k - 1] - 1.0)
            pnl[k] += r
            gpnl[k] += r
        p = newp
    return pnl, gpnl, swp, changes


def cont_run(M: Mkt, p_target: np.ndarray, buf: np.ndarray):
    """Target notional p (equity fraction, signed) decided on information to the previous close, traded at each
    day's 08:00 London fill when the change exceeds buf. Returns daily net, gross, swap and number of changes."""
    nb = len(M.mclose)
    pnl, gpnl, swp = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    p, kp, changes = 0.0, -1, 0
    for k in range(nb):
        if M.exec[k] < 0:
            continue
        if kp >= 0 and p != 0.0:
            r = p * (M.me[k] / M.me[kp] - 1.0)
            pnl[k] += r
            gpnl[k] += r
            for j in range(kp + 1, k + 1):
                s_ = abs(p) * (M.swl[j - 1] if p > 0 else M.sws[j - 1]) * M.roll[j]
                pnl[k] += s_
                swp[k] += s_
        tg = p_target[k]
        if np.isfinite(tg) and abs(tg - p) > (buf[k] if np.isfinite(buf[k]) else 0.0):
            pnl[k] -= abs(tg - p) * M.mcost[k]
            p = float(tg)
            changes += 1
        kp = k
    return pnl, gpnl, swp, changes


# ============================================================================================ indicators
def turtle_n(b: pd.DataFrame, n=20) -> pd.Series:
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def known(s: pd.Series) -> np.ndarray:
    """Value usable during bar k = value at the close of bar k-1."""
    return np.ascontiguousarray(s.shift(1).to_numpy(np.float64))


def month_end_mask(M: Mkt) -> np.ndarray:
    td = M.tday
    nxt = np.r_[td[1:].to_period("M").asi8, -1]
    return td.to_period("M").asi8 != nxt


def week_end_mask(M: Mkt) -> np.ndarray:
    iso = M.tday.isocalendar()
    key = (iso.year.to_numpy() * 100 + iso.week.to_numpy())
    return np.r_[key[1:] != key[:-1], True]


def weekly_bars(M: Mkt) -> tuple[pd.DataFrame, np.ndarray]:
    """W1 bars of Monday-Friday weeks from the D1 bars; returns the bars (indexed by the week's first D1 bar time)
    and, for every D1 bar, the index of the last completed week (-1 before the first)."""
    d1 = M.d1
    iso = M.tday.isocalendar()
    key = pd.Series(iso.year.to_numpy() * 100 + iso.week.to_numpy(), index=d1.index)
    agg = {"bid_o": "first", "bid_h": "max", "bid_l": "min", "bid_c": "last", "ask_o": "first", "ask_h": "max",
           "ask_l": "min", "ask_c": "last", "n_ticks": "sum", "spread_mean": "mean", "rv_intrabar": "sum", "n_m1": "sum"}
    agg = {c: f for c, f in agg.items() if c in d1.columns}
    mids = {"open", "high", "low", "close", "spread", "volume"}
    for c in d1.columns:                       # tick statistics some strategies read (sums over the week)
        if c not in agg and c not in mids and pd.api.types.is_numeric_dtype(d1[c]):
            agg[c] = "sum"
    kv = key.to_numpy()
    w = d1.groupby(kv, sort=True).agg(agg)
    starts = np.r_[0, np.flatnonzero(kv[1:] != kv[:-1]) + 1]
    assert len(starts) == len(w) and (np.diff(kv[starts]) > 0).all()
    w.index = d1.index[starts]
    from fxlab.data.store import add_mid
    w = add_mid(w)
    wend = week_end_mask(M)
    cw = np.cumsum(wend)                      # weeks completed at the close of each D1 bar
    last_done = np.r_[-1, cw[:-1] - 1]       # usable during bar k: weeks completed by the close of k-1
    return w, last_done.astype(np.int64)


def map_weekly(x: np.ndarray, last_done: np.ndarray) -> np.ndarray:
    out = np.full(len(last_done), np.nan)
    ok = last_done >= 0
    out[ok] = x[last_done[ok]]
    return out


# ============================================================================================ part A systems
def ev_args(M: Mkt, N: np.ndarray, **kw):
    nb = len(M.mclose)
    nan = np.full(nb, np.nan)
    z = np.zeros(nb, np.bool_)
    a = dict(mode=1, tgt=nan, ent=nan, mxl=z, mxs=z, el=nan, es=nan, xl=nan, xs=nan, sld=nan, tpd=nan, hld=nan,
             upd=np.ones(nb, np.bool_), risk=RISK, stop_mult=2.0, trail_mult=0.0, max_units=1, add_step=0.5,
             tp_R=0.0, be_R=0.0, part_R=0.0, max_hold=0)
    a.update(kw)
    for c in ("tgt", "ent", "el", "es", "xl", "xs", "sld", "tpd", "hld"):
        a[c] = np.ascontiguousarray(np.asarray(a[c], np.float64))
    for c in ("mxl", "mxs", "upd"):
        a[c] = np.ascontiguousarray(np.asarray(a[c], np.bool_))
    return a


def run_ev(M: Mkt, N: np.ndarray, **kw):
    a = ev_args(M, N, **kw)
    return ev_run(M.bo, M.bh, M.bl, M.ao, M.ah, M.al, M.slip, M.spr, M.dk, M.first, M.exec, M.roll,
                  np.ascontiguousarray(M.swl), np.ascontiguousarray(M.sws), M.mclose, np.ascontiguousarray(N),
                  a["mode"], a["tgt"], a["ent"], a["mxl"], a["mxs"], a["el"], a["es"], a["xl"], a["xs"],
                  a["sld"], a["tpd"], a["hld"], a["upd"], a["risk"], a["stop_mult"], a["trail_mult"],
                  int(a["max_units"]), a["add_step"], a["tp_R"], a["be_R"], a["part_R"], int(a["max_hold"]), COMM,
                  M.pip)


def ann_vol_ewm(close: pd.Series, com=60) -> pd.Series:
    r = np.log(close).diff()
    return np.sqrt((r ** 2).ewm(com=com, min_periods=60).mean() * 261.0)


def monthly_target(M: Mkt, sig_at_close: pd.Series, vol: pd.Series, tv=0.40) -> np.ndarray:
    """Target notional fixed at each month-end close (sign x tv / vol), applied from the next fill, held a month."""
    me = month_end_mask(M)
    v = (sig_at_close * tv / vol).where(me)
    v = v.ffill()
    return known(v)


def systems_A(M: Mkt, rng: np.random.Generator, l1_rate: float | None = None):
    """Yields (name, kind, result). kind 'ev' -> (pnl, gpnl, swp, trades); 'ct' -> (pnl, gpnl, swp, changes)."""
    b = M.d1
    c, h, lo = b["close"], b["high"], b["low"]
    Ns = turtle_n(b)
    N = known(Ns)
    hh = lambda n: known(h.rolling(n).max())          # noqa: E731  prior n-day high, usable during bar k
    ll = lambda n: known(lo.rolling(n).min())         # noqa: E731
    out = {}
    out["L1"] = ("ev", run_ev(M, N, el=hh(55), es=ll(55), xl=ll(20), xs=hh(20), stop_mult=2.0, max_units=4,
                              add_step=0.5))
    out["L2"] = ("ev", run_ev(M, N, el=hh(20), es=ll(20), xl=ll(10), xs=hh(10), stop_mult=2.0, max_units=4,
                              add_step=0.5))
    ma50, ma100, ma200 = c.rolling(50).mean(), c.rolling(100).mean(), c.rolling(200).mean()
    up = (ma50 > ma100) & (c >= c.rolling(50).max())
    dn = (ma50 < ma100) & (c <= c.rolling(50).min())
    out["L3"] = ("ev", run_ev(M, N, ent=known(pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=b.index)),
                              stop_mult=3.0, trail_mult=3.0))
    out["L4"] = ("ev", run_ev(M, N, el=hh(100), es=ll(100), xl=ll(50), xs=hh(50), stop_mult=0.0))
    side = pd.Series(np.sign(ma50 - ma200), index=b.index).where(ma200.notna())
    out["L5"] = ("ev", run_ev(M, N, mode=0, tgt=known(side), stop_mult=0.0))
    vol = ann_vol_ewm(c)
    r252, r63, r21 = c / c.shift(252) - 1, c / c.shift(63) - 1, c / c.shift(21) - 1
    nb = len(c)
    zero_buf = np.zeros(nb)
    out["L6"] = ("ct", cont_run(M, monthly_target(M, np.sign(r252), vol), zero_buf))
    blend = (np.sign(r252) + np.sign(r63) + np.sign(r21)) / 3.0
    out["L7"] = ("ct", cont_run(M, monthly_target(M, blend.where(r252.notna()), vol), zero_buf))
    me = month_end_mask(M)
    cm = c[me]
    sma10 = cm.rolling(10).mean()
    fab = pd.Series(np.nan, index=c.index)
    fab[me] = np.sign(cm - sma10).where(sma10.notna()).to_numpy()
    out["L8"] = ("ct", cont_run(M, monthly_target(M, fab, vol), zero_buf))
    fc = ewmac(c)
    vol36 = np.sqrt((np.log(c).diff() ** 2).ewm(span=36, min_periods=36).mean() * 256.0)
    p9 = known(fc / 10.0 * 0.20 / vol36)
    buf9 = known(0.1 * 0.20 / vol36)
    out["L9"] = ("ct", cont_run(M, p9, buf9))
    if M.sym in FX_SYMBOLS:
        rd = pd.Series(M.rd, index=c.index)
        fc12 = fc.where(np.sign(fc) == np.sign(rd), 0.0)
        out["L12"] = ("ct", cont_run(M, known(fc12 / 10.0 * 0.20 / vol36), buf9))
    # L10 squeeze release
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    sq = (sma20 + 2 * sd20 < sma20 + 1.5 * atr20) & (sma20 - 2 * sd20 > sma20 - 1.5 * atr20)
    rel = sq.shift(1, fill_value=False) & ~sq
    d10 = pd.Series(np.where(rel, np.sign(c - sma20), 0.0), index=c.index)
    out["L10"] = ("ev", run_ev(M, N, ent=known(d10), stop_mult=2.0, trail_mult=3.0))
    # L11 weekly breakout
    w, last_done = weekly_bars(M)
    wN = turtle_n(w).to_numpy()
    wc, wh, wl = w["close"], w["high"], w["low"]
    lg = (wc > wh.rolling(20).max().shift(1)).to_numpy(float)
    sh = (wc < wl.rolling(20).min().shift(1)).to_numpy(float)
    xlg = (wc < wl.rolling(10).min().shift(1)).to_numpy()
    xsh = (wc > wh.rolling(10).max().shift(1)).to_numpy()
    fresh = np.zeros(len(last_done), bool)
    fresh[1:] = last_done[1:] != last_done[:-1]                       # first D1 bar after a weekly close
    entw = map_weekly(lg - sh, last_done)
    entw[~fresh] = np.nan
    mxl_ = np.nan_to_num(map_weekly(xlg.astype(float), last_done)) > 0
    mxs_ = np.nan_to_num(map_weekly(xsh.astype(float), last_done)) > 0
    mxl_[~fresh] = False
    mxs_[~fresh] = False
    out["L11"] = ("ev", run_ev(M, map_weekly(wN, last_done), ent=entw, mxl=mxl_, mxs=mxs_, stop_mult=2.0,
                               upd=week_end_mask(M)))
    # controls
    rate = l1_rate if l1_rate else max(len(out["L1"][1][3]), 1) / max(nb, 1)
    rnd = np.where(rng.random(nb) < rate, np.where(rng.random(nb) < 0.5, 1.0, -1.0), 0.0)
    out["PL1"] = ("ev", run_ev(M, N, ent=rnd, stop_mult=2.0, trail_mult=3.0))
    rs = pd.Series(np.where(rng.random(nb) < 0.5, 1.0, -1.0), index=c.index).where(r252.notna())
    out["PL2"] = ("ct", cont_run(M, monthly_target(M, rs, vol), zero_buf))
    out["BH"] = ("ct", cont_run(M, monthly_target(M, pd.Series(1.0, index=c.index).where(vol.notna()), vol), zero_buf))
    return out


def ewmac(c: pd.Series) -> pd.Series:
    sig_p = c.diff().ewm(span=36, min_periods=36).std()
    fcs = []
    for f, s, sc in ((8, 32, 5.3), (16, 64, 3.75), (32, 128, 2.65), (64, 256, 1.87)):
        raw = (c.ewm(span=f, min_periods=s).mean() - c.ewm(span=s, min_periods=s).mean()) / sig_p
        fcs.append((raw * sc).clip(-20, 20))
    return (pd.concat(fcs, axis=1).mean(axis=1, skipna=False) * 1.1).clip(-20, 20)


# ============================================================================================ part B
def catalog_units_d1():
    import fxlab.strategies.daily  # noqa: F401
    import fxlab.strategies.extra  # noqa: F401
    import fxlab.strategies.famous  # noqa: F401
    import fxlab.strategies.rules  # noqa: F401
    from fxlab.registry import CATALOG
    units = []
    for e in CATALOG.values():
        if e.kind != "strategy" or e.family in EXCL_FAMILY:
            continue
        if e.multi_asset and e.id not in KEEP_MULTI:
            continue
        lo, hi = (e.frequencies.split("..") + [None])[:2]
        hi = hi or lo
        if TFS.index(lo) <= TFS.index("D1") <= TFS.index(hi):
            units.append(e)
    return units


def exit_variants(M: Mkt, base: dict, Nk: np.ndarray, hh20: np.ndarray, ll20: np.ndarray, upd: np.ndarray):
    """The five exit variants on fresh entries (T0 is the strategy's own behaviour, passed in `base`)."""
    ent = base["fresh"]
    v = {"T0": base["T0"]}
    v["T1"] = dict(ent=ent, stop_mult=2.0, trail_mult=3.0, upd=upd)
    v["T2"] = dict(ent=ent, stop_mult=2.0, xl=ll20, xs=hh20, upd=upd)
    v["T3"] = dict(ent=ent, stop_mult=2.0, trail_mult=3.0, be_R=1.0, upd=upd)
    v["T4"] = dict(ent=ent, stop_mult=2.0, trail_mult=3.0, part_R=2.0, upd=upd)
    return v


def to_monthly(M: Mkt, pnl, gpnl, swp, T):
    """Monthly sums of daily P&L and closed-trade statistics by the month of the exit."""
    per = M.tday.to_period("M")
    df = pd.DataFrame({"pnl": pnl, "gpnl": gpnl, "swp": swp}, index=per)
    g = df.groupby(level=0).sum()
    if len(T):
        xm = per[T[:, 1].astype(int)]
        t = pd.DataFrame({"n": 1, "win": (T[:, 4] > 0).astype(int), "sw": np.where(T[:, 4] > 0, T[:, 4], 0.0),
                          "sl": np.where(T[:, 4] <= 0, T[:, 4], 0.0), "hold": T[:, 9], "pips": T[:, 7],
                          "apips": np.abs(T[:, 7])}, index=xm)
        g = g.join(t.groupby(level=0).sum(), how="left")
    for col in ("n", "win", "sw", "sl", "hold", "pips", "apips"):
        if col not in g:
            g[col] = 0.0
    g = g.fillna(0.0)
    g.index = pd.Index(g.index.astype(str), name="month")
    return g


def run_B(M: Mkt, rng, per_unit_s=120, budget_s=15000, t_job=None, only=None):
    from fxlab.features._util import atr
    rows = []
    b1 = M.d1
    w, last_done = weekly_bars(M)
    wend = week_end_mask(M)
    nb = len(b1)
    fresh_w = np.zeros(nb, bool)
    fresh_w[1:] = last_done[1:] != last_done[:-1]
    grids = {
        "D1": dict(bars=b1, N=known(turtle_n(b1)), hh=known(b1["high"].rolling(20).max()),
                   ll=known(b1["low"].rolling(20).min()), upd=np.ones(nb, bool), atr=known(atr(b1, 14)), step=1),
        "W1": dict(bars=w, N=map_weekly(turtle_n(w).to_numpy(), last_done),
                   hh=map_weekly(w["high"].rolling(20).max().to_numpy(), last_done),
                   ll=map_weekly(w["low"].rolling(20).min().to_numpy(), last_done), upd=wend,
                   atr=map_weekly(atr(w, 14).to_numpy(), last_done), step=5),
    }
    units = [(e.id, e) for e in catalog_units_d1()]
    units += [(f"PB_RANDOM_{r}", None) for r in (10, 20, 40)]
    if only:
        units = [u for u in units if u[0] in only]
    t_job = t_job or time.time()
    for uid, e in units:
        for tf, G in grids.items():
            t0 = time.time()
            key = f"{uid}|{tf}"
            if t0 - t_job > budget_s:
                LOG[key] = {"error": "budget"}
                continue
            try:
                signal.alarm(per_unit_s)
                bars = G["bars"]
                if e is None:
                    r = int(uid.split("_")[-1])
                    s = np.where(rng.random(len(bars)) < 1.0 / r, np.where(rng.random(len(bars)) < 0.5, 1.0, -1.0), 0.0)
                    side = pd.Series(s, index=bars.index)
                    trade_mode, sig = True, None
                else:
                    kw = dict(e.params)
                    if "symbol" in inspect.signature(e.fn).parameters:
                        kw["symbol"] = M.sym
                    if e.id == "S.EVT.GOLD_FIX" and not M.sym.startswith("XAU"):
                        signal.alarm(0)
                        continue
                    sig = e.fn(bars.copy(), **kw)
                    side = sig["side"].fillna(0.0)
                    trade_mode = e.extra.get("mode") == "trade"
                sd = np.sign(side.to_numpy(float))
                prev = np.r_[0.0, sd[:-1]]
                fr = np.where((sd != 0) & (sd != prev), sd, 0.0)
                if tf == "D1":
                    fresh = np.r_[np.nan, fr[:-1]]
                    target = np.r_[np.nan, sd[:-1]]
                else:
                    fresh = map_weekly(fr, last_done)
                    fresh[~fresh_w] = np.nan
                    target = map_weekly(sd, last_done)
                if e is None:
                    T0 = dict(ent=fresh, stop_mult=0.0, max_hold=10 * G["step"])
                elif trade_mode:
                    risk = e.extra.get("risk") or {}
                    mult = (risk.get("stop_params") or {}).get("mult", 2.0)
                    tpp = risk.get("target_params") or {"a": 0.0, "b": 1.5}
                    tgt_kind = risk.get("target", "affine")
                    mh = risk.get("max_hold", 48)
                    sl_bar = sig["stop_dist"].to_numpy(float) if "stop_dist" in sig else np.full(len(bars), np.nan)
                    if tf == "D1":
                        sl_k = np.r_[np.nan, sl_bar[:-1]]
                        a_k = G["atr"]
                    else:
                        sl_k = map_weekly(sl_bar, last_done)
                        a_k = G["atr"]
                    sl_k = np.where(np.isfinite(sl_k) & (sl_k > 0), sl_k, mult * np.asarray(a_k))
                    if "tp_dist" in sig:
                        tpb = sig["tp_dist"].to_numpy(float)
                        tp_k = np.r_[np.nan, tpb[:-1]] if tf == "D1" else map_weekly(tpb, last_done)
                    else:
                        tp_k = np.full(nb, np.nan)
                    tp_k = np.where(np.isfinite(tp_k), tp_k, np.nan if tgt_kind == "none" else
                                    tpp.get("a", 0.0) + tpp.get("b", 1.5) * sl_k)
                    T0 = dict(ent=fresh, sld=sl_k, tpd=tp_k, hld=np.full(nb, float(mh * G["step"])), stop_mult=2.0)
                else:
                    T0 = dict(mode=0, tgt=target, stop_mult=0.0)
                V = exit_variants(M, {"fresh": fresh, "T0": T0}, G["N"], G["hh"], G["ll"], G["upd"])
                for vname, kw in V.items():
                    kw = dict(kw)
                    if "upd" not in kw:
                        kw["upd"] = G["upd"]
                    pnl, gp, sw, T = run_ev(M, G["N"], **kw)
                    g = to_monthly(M, pnl, gp, sw, T)
                    g["unit"], g["tf"], g["variant"] = uid, tf, vname
                    rows.append(g.reset_index())
                signal.alarm(0)
                LOG[key] = {"s": round(time.time() - t0, 1)}
            except Timeout:
                LOG[key] = {"error": "timeout"}
            except Exception as ex:                                   # noqa: BLE001
                signal.alarm(0)
                LOG[key] = {"error": repr(ex)[:200]}
                traceback.print_exc(limit=1)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================================================ per instrument
TCOLS = ["ek", "xk", "d", "units", "net", "gross", "swap", "pips", "R_frac", "hold", "ei", "xi"]


def run_symbol(sym, out_dir, only=None, catalog=True, m1=None, pip=None):
    t_job = time.time()
    if m1 is None:
        import edges_lab2 as E2
        bk = E2.book(sym)
        if bk is None:
            print("no data", sym)
            return
        m1, pip = bk.m, bk.pip
    M = Mkt(m1, sym, pip)
    print(sym, "M1", len(M.t), "D1", len(M.d1), M.tday[0].date(), M.tday[-1].date(), "fills", int((M.exec >= 0).sum()),
          flush=True)
    import zlib
    rng = np.random.default_rng(zlib.crc32(sym.encode()))          # stable across processes
    os.makedirs(out_dir, exist_ok=True)
    A = systems_A(M, rng)
    daily, trades = [], []
    for name, (kind, res) in A.items():
        if only and name not in only:
            continue
        pnl, gp, sw = res[0], res[1], res[2]
        daily.append(pd.DataFrame({"system": name, "date": M.tday, "pnl": pnl, "gpnl": gp, "swp": sw}))
        if kind == "ev":
            T = pd.DataFrame(res[3], columns=TCOLS)
            T["system"] = name
            T["t_in"] = pd.to_datetime(M.t[T.ei.astype(int)], utc=True) if len(T) else pd.Series(dtype="datetime64[ns, UTC]")
            T["t_out"] = pd.to_datetime(M.t[T.xi.astype(int)], utc=True) if len(T) else pd.Series(dtype="datetime64[ns, UTC]")
            trades.append(T)
            LOG[name] = {"trades": int(len(T))}
        else:
            LOG[name] = {"changes": int(res[3])}
        print(sym, name, LOG[name], "net %.4f gross %.4f swap %.4f" % (pnl.sum(), gp.sum(), sw.sum()), flush=True)
    D = pd.concat(daily, ignore_index=True)
    D["symbol"] = sym
    D.to_parquet(os.path.join(out_dir, f"A_daily_{sym}.parquet"), index=False)
    if trades:
        TT = pd.concat(trades, ignore_index=True)
        TT["symbol"] = sym
        TT.to_parquet(os.path.join(out_dir, f"A_trades_{sym}.parquet"), index=False)
    if catalog:
        signal.signal(signal.SIGALRM, _alarm)
        B = run_B(M, rng, t_job=t_job, only=only)
        if len(B):
            B["symbol"] = sym
            for col in B.columns:
                if B[col].dtype == np.float64:
                    B[col] = B[col].astype(np.float32)
            B.to_parquet(os.path.join(out_dir, f"B_monthly_{sym}.parquet"), index=False)
        print(sym, "part B rows", len(B), flush=True)
    json.dump(LOG, open(os.path.join(out_dir, f"log_{sym}.json"), "w"), indent=0, default=str)
    print("done", sym, round(time.time() - t_job), "s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-catalog", action="store_true")
    a = ap.parse_args()
    run_symbol(a.symbol, a.out, only=set(a.only.split(",")) if a.only else None, catalog=not a.no_catalog)


if __name__ == "__main__":
    main()
