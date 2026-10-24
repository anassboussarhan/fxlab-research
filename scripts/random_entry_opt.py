"""Best trade-management strategy and timeframe for RANDOM entries, fully optimised and tested out of sample.

Entries are random: on every bar a new position is opened with probability p (only when flat) in a coin-flip
direction (the entry rate is itself optimised). Everything else is the strategy and is optimised:
  bracket      ATR stop 0.25-10, take-profit none / 0.2-8 R, trailing none / 0.25-10 ATR, breakeven none /
               0.2-4 R, half take-profit none / 0.2-4 R, time exit none / 1-500 bars, sizing notional or 0.5% risk
  martingale   bracket (stop, target, time exit) with position size x f (1.2-3) after a loss (martingale) or
               after a win (anti-martingale), up to K = 1-6 steps
  pyramid      add a unit every +0.25-5 ATR in favour up to N = 2-8 units, trailing exit on a 0.5-10 ATR give-back,
               initial stop 0.5-10 ATR
  grid         averaging down: add every 0.25-5 ATR against (sizes x g = 1-2 per level, N = 2-10 levels), basket
               take-profit 0.1-5 ATR beyond the average price, basket stop 0.5-10 ATR beyond the last level
  zone         zone recovery (hedging martingale): zone 0.25-5 ATR, take-profit 0.25-5 ATR, up to K = 1-8 legs
  hedged_grid  bi-directional grid with locked hedges (no entries needed): step 0.25-5 ATR, basket take-profit
               0.5-10 steps, stop-out 2-50 steps, N = 3-20 positions per side
Costs: quoted bid/ask, slippage 0.25 x spread + 0.1 pip, commission 0.35 bp per side, swap 0.5 pip per night on
every open position (both legs of a hedge). Stop-first inside a bar; on M5+ a bar holding both the stop and the
target of a bracket is resolved from the M1 path.

Evaluation (development data only, 2007-2018; EURUSD to 2022):
  WALK-FORWARD   3-year training / 6-month test from 2010. In each fold TPE (Optuna, 60 trials) maximises the mean
                 daily net Sharpe over 6 random-entry seeds on the training window (common random numbers, >= 30
                 trades per seed). The best parameters then trade the test window with 16 NEW random seeds.
                 Reported: the median single-trader out-of-sample Sharpe (over the 16 seeds), and the seed-average
                 stream (the strategy's expected-value stream) with its t-statistic.
  HINDSIGHT      TPE (120 trials) on the whole sample, then re-run with 16 fresh seeds on the SAME data: does the
                 in-sample optimum survive new random entries (price-path structure) or not (entry overfitting)?
Usage: python scripts/random_entry_opt.py --pair EURUSD --tf H1 [--out DIR]"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

N_TRIALS = int(os.environ.get("REO_TRIALS", 60))
N_HIND = int(os.environ.get("REO_HIND", 120))
TRAIN_SEEDS = 6
TEST_SEEDS = 16
BIG = 20_000_000           # safety valve: fewer trials / seeds for extremely long series
MIN_TRADES = 30
FAMILIES = os.environ.get("REO_FAMILIES", "bracket,martingale,pyramid,grid,zone,hedged_grid").split(",")


# ------------------------------------------------------------------------------------------------ kernels
@njit(cache=False)
def _bracket(bo, bh, bl, bc, ao, ah, al, ac, atr, roll, ent, sl, tp, trail, be, part_r, tmax, size_mode, risk,
             levcap, mg_mode, mg_f, mg_k, slip_frac, slip_pips, pip, comm, swap_pips, m1o, mbh, mbl, mah, mal, use_m1):
    n = len(bo)
    bar = np.zeros(n)
    ntr = 0
    pos = 0.0; epx = 0.0; stop = 0.0; tgt = np.nan; trl = 0.0; sd = 0.0; ebar = -1; f = 0.0; last = 0.0
    part_done = False; lvl = 0; tpnl = 0.0
    for j in range(1, n):
        slip = slip_frac * (ao[j] - bo[j]) + slip_pips * pip
        if pos == 0.0 and ent[j - 1] != 0 and atr[j - 1] == atr[j - 1] and atr[j - 1] > 0:
            d = float(ent[j - 1])
            epx = ao[j] + slip if d > 0 else bo[j] - slip
            sd = sl * atr[j - 1]
            stop = epx - d * sd
            tgt = epx + d * tp * sd if tp > 0 else np.nan
            trl = trail * atr[j - 1] if trail > 0 else 0.0
            mult = mg_f ** lvl if mg_mode != 0 else 1.0
            base = 1.0 if size_mode == 0 else min(levcap, risk / (sd / epx))
            f = base * mult
            pos = d; ebar = j; part_done = False; last = 0.0; tpnl = -f * comm
            bar[j] -= f * comm
            ntr += 1
        if pos == 0.0:
            continue
        d = pos
        done = False; px = 0.0
        if ebar != j:
            if d > 0:
                if bo[j] <= stop:
                    px = bo[j] - slip; done = True
                elif tgt == tgt and bo[j] >= tgt:
                    px = bo[j]; done = True
            else:
                if ao[j] >= stop:
                    px = ao[j] + slip; done = True
                elif tgt == tgt and ao[j] <= tgt:
                    px = ao[j]; done = True
        if not done:
            if d > 0:
                hs = bl[j] <= stop
                ht = tgt == tgt and bh[j] >= tgt
            else:
                hs = ah[j] >= stop
                ht = tgt == tgt and al[j] <= tgt
            if hs and ht and use_m1:
                first = 1
                for q in range(m1o[j], m1o[j + 1]):
                    if d > 0:
                        if mbl[q] <= stop:
                            first = 1
                            break
                        if mbh[q] >= tgt:
                            first = 2
                            break
                    else:
                        if mah[q] >= stop:
                            first = 1
                            break
                        if mal[q] <= tgt:
                            first = 2
                            break
                px = stop - d * slip if first == 1 else tgt
                done = True
            elif hs:
                px = stop - d * slip; done = True
            elif ht:
                px = tgt; done = True
        if not done and part_r > 0 and not part_done:
            lv = epx + d * part_r * sd
            if (d > 0 and bh[j] >= lv) or (d < 0 and al[j] <= lv):
                fp = 0.5 * f
                r = d * (lv - epx) / epx
                bar[j] += fp * (r - last) - fp * comm
                tpnl += fp * (r - last) - fp * comm
                f -= fp; part_done = True
                if be > 0:
                    stop = max(stop, epx) if d > 0 else min(stop, epx)
        if not done and tmax > 0 and j - ebar + 1 >= tmax:
            px = bc[j] - slip if d > 0 else ac[j] + slip
            done = True
        if done:
            r = d * (px - epx) / epx
            bar[j] += f * (r - last) - f * comm
            tpnl += f * (r - last) - f * comm
            pos = 0.0
            if mg_mode == 1:                         # martingale: bigger after a loss
                lvl = 0 if tpnl > 0 else (lvl + 1 if lvl < mg_k else 0)
            elif mg_mode == 2:                       # anti-martingale: bigger after a win
                lvl = (lvl + 1 if lvl < mg_k else 0) if tpnl > 0 else 0
            continue
        mark = bc[j] if d > 0 else ac[j]
        val = d * (mark - epx) / epx
        bar[j] += f * (val - last); tpnl += f * (val - last)
        last = val
        if roll[j] > 0:
            fc = f * roll[j] * swap_pips * pip / epx
            bar[j] -= fc; tpnl -= fc
        if trl > 0:
            if d > 0:
                stop = max(stop, bh[j] - trl)
            else:
                stop = min(stop, al[j] + trl)
        if be > 0:
            if d > 0 and bh[j] - epx >= be * sd:
                stop = max(stop, epx)
            elif d < 0 and epx - al[j] >= be * sd:
                stop = min(stop, epx)
    return bar, ntr


@njit(cache=False)
def _pyramid(bo, bh, bl, bc, ao, ah, al, ac, atr, roll, ent, add_atr, N, give_atr, stop_atr, slip_frac, slip_pips,
             pip, comm, swap_pips):
    """Notional w = units/N of equity; marked at mid; fills pay half spread + slippage + commission."""
    n = len(bo)
    bar = np.zeros(n)
    ntr = 0
    d = 0.0; w = 0.0; units = 0; a0 = 0.0; S = 0.0; best = 0.0; last_add = 0.0; pend = False
    for j in range(1, n):
        mo = 0.5 * (bo[j] + ao[j]); mc = 0.5 * (bc[j] + ac[j]); mpc = 0.5 * (bc[j - 1] + ac[j - 1])
        hs = 0.5 * (ao[j] - bo[j]); slip = slip_frac * (ao[j] - bo[j]) + slip_pips * pip
        if w != 0.0:
            bar[j] += w * d * (mo / mpc - 1.0)                     # overnight / gap part to this open
        if w == 0.0 and ent[j - 1] != 0 and atr[j - 1] == atr[j - 1] and atr[j - 1] > 0:
            d = float(ent[j - 1]); units = 1; w = 1.0 / N; a0 = atr[j - 1]
            S = mo - d * stop_atr * a0; best = mo; last_add = mo; pend = False
            bar[j] -= w * ((hs + slip) / mo + comm)
            ntr += 1
        elif w != 0.0 and pend:
            units += 1; w = units / N; pend = False
            bar[j] -= (1.0 / N) * ((hs + slip) / mo + comm)
            last_add = mo
        if w == 0.0:
            continue
        stopped = (d > 0 and 0.5 * (bl[j] + al[j]) <= S) or (d < 0 and 0.5 * (bh[j] + ah[j]) >= S)
        if stopped:
            ex = min(S, mo) if d > 0 else max(S, mo)
            bar[j] += w * d * (ex / mo - 1.0) - w * ((hs + slip) / mo + comm)
            w = 0.0; units = 0
            continue
        bar[j] += w * d * (mc / mo - 1.0)
        if roll[j] > 0:
            bar[j] -= w * roll[j] * swap_pips * pip / mc
        if d * (mc - best) > 0:
            best = mc
        S = max(S, best - give_atr * a0) if d > 0 else min(S, best + give_atr * a0)
        if units < N and d * (mc - last_add) >= add_atr * a0:
            pend = True
    return bar, ntr


@njit(cache=False)
def _grid(bo, bh, bl, bc, ao, ah, al, ac, atr, roll, ent, step_atr, N, g, tp_atr, sl_atr, slip_frac, slip_pips, pip,
          comm, swap_pips, m1o, mbh, mbl, mah, mal, use_m1):
    """Averaging-down grid with resting limit adds (filled at the level), basket take-profit (limit, at the level)
    and basket stop (market). Size of level k = g^k, normalised so all N levels = 1 notional. Inside a bar the
    order of events is unknown: when more than one level lies in the bar's range the M1 path is replayed
    (use_m1); within one sub-bar (or without M1) adds come first, then the stop, and no take-profit on a bar segment
    in which an add filled (pessimistic)."""
    n = len(bo)
    bar = np.zeros(n)
    ntr = 0
    tot = 0.0
    for k in range(N):
        tot += g ** k
    d = 0.0; w = 0.0; lev = 0; a0 = 0.0; avg = 0.0; nxt = 0.0; stop = 0.0; tpl = 0.0
    for j in range(1, n):
        mo = 0.5 * (bo[j] + ao[j]); mc = 0.5 * (bc[j] + ac[j]); mpc = 0.5 * (bc[j - 1] + ac[j - 1])
        hs = 0.5 * (ao[j] - bo[j]); slip = slip_frac * (ao[j] - bo[j]) + slip_pips * pip
        if w != 0.0:
            bar[j] += w * d * (mo / mpc - 1.0)
        if w == 0.0 and ent[j - 1] != 0 and atr[j - 1] == atr[j - 1] and atr[j - 1] > 0:
            d = float(ent[j - 1]); a0 = atr[j - 1]; lev = 1; w = 1.0 / tot; avg = mo
            nxt = mo - d * step_atr * a0
            stop = mo - d * (step_atr * (N - 1) + sl_atr) * a0
            tpl = avg + d * tp_atr * a0
            bar[j] -= w * ((hs + slip) / mo + comm)
            ntr += 1
        if w == 0.0:
            continue
        lo_b = 0.5 * (bl[j] + al[j]); hi_b = 0.5 * (bh[j] + ah[j])
        if d > 0:
            nlev = (1 if (lev < N and lo_b < nxt) else 0) + (1 if lo_b <= stop else 0) + (1 if hi_b >= tpl else 0)
        else:
            nlev = (1 if (lev < N and hi_b > nxt) else 0) + (1 if hi_b >= stop else 0) + (1 if lo_b <= tpl else 0)
        rep = use_m1 and nlev >= 1 and m1o[j + 1] > m1o[j]     # any level in range: replay (post-add targets move)
        q0 = m1o[j] if rep else 0
        q1 = m1o[j + 1] if rep else 1
        ref = mo
        closed = False
        for q in range(q0, q1):
            if rep:
                lo = 0.5 * (mbl[q] + mal[q]); hi = 0.5 * (mbh[q] + mah[q])
            else:
                lo = lo_b; hi = hi_b
            filled_add = False
            while lev < N and ((d > 0 and lo < nxt) or (d < 0 and hi > nxt)):
                size = g ** lev / tot
                bar[j] += w * d * (nxt / ref - 1.0)
                ref = nxt
                avg = (avg * w + nxt * size) / (w + size)
                w += size; lev += 1
                bar[j] -= size * comm
                nxt = nxt - d * step_atr * a0
                tpl = avg + d * tp_atr * a0
                filled_add = True
            if (d > 0 and lo <= stop) or (d < 0 and hi >= stop):
                bar[j] += w * d * (stop / ref - 1.0) - w * ((hs + slip) / stop + comm)
                w = 0.0; closed = True
                break
            if not filled_add and ((d > 0 and hi >= tpl) or (d < 0 and lo <= tpl)):
                bar[j] += w * d * (tpl / ref - 1.0) - w * comm
                w = 0.0; closed = True
                break
        if closed:
            continue
        bar[j] += w * d * (mc / ref - 1.0)
        if roll[j] > 0:
            bar[j] -= w * roll[j] * swap_pips * pip / mc
    return bar, ntr


@njit(cache=False)
def _zone(c, a, ent, zone_k, tp_k, K):
    """Zone recovery with a random first-leg direction per cycle (prices mirrored for shorts). Returns net units,
    gross open units and gross units traded per decision bar (evaluated by _eval_units)."""
    n = len(c)
    net = np.zeros(n); gross_open = np.zeros(n); traded = np.zeros(n)
    active = False; P = 0.0; z = 0.0; tp = 0.0; k = 0; last_long = True; L = 0.0; S = 0.0; dd = 1.0
    ntr = 0
    for t in range(n):
        tr = 0.0
        x = dd * c[t]
        if active:
            if x >= P + tp or x <= P - z - tp:
                tr += L + S; L = 0.0; S = 0.0; active = False
            elif last_long and x <= P - z:
                if k >= K:
                    tr += L + S; L = 0.0; S = 0.0; active = False
                else:
                    need = (L * (z + tp) + tp) / tp
                    add = max(need - S, 0.0); S += add; tr += add; k += 1; last_long = False
            elif (not last_long) and x >= P:
                if k >= K:
                    tr += L + S; L = 0.0; S = 0.0; active = False
                else:
                    need = (S * (z + tp) + tp) / tp
                    add = max(need - L, 0.0); L += add; tr += add; k += 1; last_long = True
        if not active and ent[t] != 0 and a[t] == a[t] and a[t] > 0:
            dd = float(ent[t]); x = dd * c[t]
            active = True; k = 0; z = zone_k * a[t]; tp = tp_k * a[t]; P = x; L = 1.0; S = 0.0
            last_long = True; tr += 1.0; ntr += 1
        net[t] = dd * (L - S)
        gross_open[t] = L + S
        traded[t] = tr
    return net, gross_open, traded, ntr


@njit(cache=False)
def _hgrid(c, a, step_k, basket_k, stop_k, N):
    n = len(c)
    net = np.zeros(n); gross_open = np.zeros(n); traded = np.zeros(n)
    buys = np.zeros(4 * N + 4); sells = np.zeros(4 * N + 4)
    nb = 0; ns = 0; anchor = np.nan; g = 0.0; up = 0; dn = 0; ncyc = 0
    for t in range(n):
        tr = 0.0
        if anchor != anchor:
            if a[t] == a[t] and a[t] > 0:
                anchor = c[t]; g = step_k * a[t]; up = 1; dn = 1; ncyc += 1
            continue
        i = 0
        while i < nb:
            if c[t] >= buys[i] + g:
                buys[i] = buys[nb - 1]; nb -= 1; tr += 1
            else:
                i += 1
        i = 0
        while i < ns:
            if c[t] <= sells[i] - g:
                sells[i] = sells[ns - 1]; ns -= 1; tr += 1
            else:
                i += 1
        while c[t] >= anchor + up * g and nb < 4 * N:
            buys[nb] = anchor + up * g; nb += 1; up += 1; tr += 1
        while c[t] <= anchor - dn * g and ns < 4 * N:
            sells[ns] = anchor - dn * g; ns += 1; dn += 1; tr += 1
        if c[t] < anchor + (up - 1) * g:
            up = max(1, int(np.floor((c[t] - anchor) / g)) + 1)
        if c[t] > anchor - (dn - 1) * g:
            dn = max(1, int(np.floor((anchor - c[t]) / g)) + 1)
        fl = 0.0
        for i in range(nb):
            fl += c[t] - buys[i]
        for i in range(ns):
            fl += sells[i] - c[t]
        fl /= g
        if (nb + ns > 0) and (fl >= basket_k or fl <= -stop_k or nb > N or ns > N):
            tr += nb + ns; nb = 0; ns = 0
            anchor = c[t]; g = step_k * a[t] if a[t] == a[t] else g; up = 1; dn = 1; ncyc += 1
        net[t] = nb - ns; gross_open[t] = nb + ns; traded[t] = tr
    return net, gross_open, traded, ncyc


@njit(cache=False)
def _eval_units(net_units, gross_open, traded, unit, mo, mc, spr, roll, slip_frac, slip_pips, pip, comm, swap_pips):
    """vector_backtest convention: position decided at close t-1 is held from open t to open t+1; every unit traded
    pays half spread + slippage + commission (both legs of locked hedges); swap on gross open units."""
    n = len(mo)
    out = np.zeros(n)
    for t in range(1, n):
        held = net_units[t - 1] * unit
        nx = mo[t + 1] if t + 1 < n else mc[t]
        out[t] = held * (nx / mo[t] - 1.0)
        trd = traded[t - 1] * unit
        out[t] -= trd * ((0.5 * spr[t] + slip_frac * spr[t] + slip_pips * pip) / mo[t] + comm)
        out[t] -= gross_open[t - 1] * unit * roll[t] * swap_pips * pip / mc[t]
    return out


# ------------------------------------------------------------------------------------------------ framework
class Data:
    def __init__(self, bars, pip, m1=None):
        from fxlab.features._util import atr
        from fxlab.execution.simulator import rollover_counts, _m1_path
        from fxlab.validation.metrics import trading_day
        f = lambda c: bars[c].to_numpy(np.float64)                             # noqa: E731
        self.A = tuple(f(c) for c in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"))
        mid = pd.DataFrame({k: (bars[f"bid_{c}"] + bars[f"ask_{c}"]) / 2 for k, c in
                            (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"))}, index=bars.index)
        self.atr = atr(mid, 14).to_numpy(np.float64)
        self.mo, self.mc = mid["open"].to_numpy(), mid["close"].to_numpy()
        self.spr = self.A[7] - self.A[3]
        self.roll = rollover_counts(bars.index).astype(np.float64)
        self.day = pd.factorize(trading_day(bars.index))[0].astype(np.int64)
        self.n = len(bars)
        self.pip = pip
        self.m1o, self.mbh, self.mbl, self.mah, self.mal, self.use_m1 = _m1_path(bars, m1)
        self.Amid = (self.mo, mid["high"].to_numpy(), mid["low"].to_numpy(), self.mc,
                     self.mo, mid["high"].to_numpy(), mid["low"].to_numpy(), self.mc)


def daily_sharpe(b, day, i0):
    """Daily Sharpe of bar returns b aligned to bars [i0, i0 + len(b))."""
    dd = day[i0:i0 + len(b)] - day[i0]
    s = np.bincount(dd, weights=b)
    if len(s) < 20 or s.std() == 0:
        return 0.0, s
    return float(s.mean() / s.std() * math.sqrt(260)), s


WARM = 300


def entries(p, seed, w0, i0, i1):
    """Random entries for the kernel slice [w0, i1): none in the warm-up part [w0, i0)."""
    rng = np.random.default_rng(seed)
    m = i1 - i0
    u = rng.random(m)
    sg = np.where(rng.random(m) < 0.5, -1, 1).astype(np.int8)
    e = np.zeros(i1 - w0, np.int8)
    e[i0 - w0:] = np.where(u < p, sg, 0)
    return e


def run_family(fam, P, D, e, i0, i1, gross=False):
    """Bar returns for bars [i0, i1) (the kernel runs on [i0 - WARM, i1) with no entries in the warm-up)."""
    from fxlab.execution.costs import BASE
    w0 = max(0, i0 - WARM)
    sl_ = slice(w0, i1)
    A = D.Amid if gross else D.A
    A = tuple(x[sl_] for x in A)
    slip_frac, slip_pips, comm, swap = (0.0, 0.0, 0.0, 0.0) if gross else (
        BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4, BASE.swap_pips_per_night)
    at = D.atr[sl_]; ro = D.roll[sl_]
    if fam in ("bracket", "martingale"):
        mg = {"none": 0, "martingale": 1, "anti": 2}[P.get("mg_mode", "none")]
        bar, ntr = _bracket(*A, at, ro, e, P["sl"], P.get("tp", 0.0), P.get("trail", 0.0), P.get("be", 0.0),
                            P.get("part", 0.0), P.get("tmax", 0), 1 if P.get("size") == "risk" else 0, 0.005, 10.0,
                            mg, P.get("mg_f", 1.0), P.get("mg_k", 0), slip_frac, slip_pips, D.pip, comm, swap,
                            D.m1o[w0:i1 + 1], D.mbh, D.mbl, D.mah, D.mal, D.use_m1 and not gross)
    elif fam == "pyramid":
        bar, ntr = _pyramid(*A, at, ro, e, P["add"], P["N"], P["give"], P["stop"], slip_frac, slip_pips, D.pip, comm, swap)
    elif fam == "grid":
        bar, ntr = _grid(*A, at, ro, e, P["step"], P["N"], P["g"], P["tp"], P["sl"], slip_frac, slip_pips, D.pip, comm, swap,
                         D.m1o[w0:i1 + 1], D.mbh, D.mbl, D.mah, D.mal, D.use_m1)
    else:
        mc = D.mc[sl_]
        if fam == "zone":
            nu, go, tr, ntr = _zone(mc, at, e, P["zone"], P["tp"], P["K"])
            unit = 0.25
        else:
            nu, go, tr, ntr = _hgrid(mc, at, P["step"], P["basket"], P["stop"], P["N"])
            unit = 1.0 / P["N"]
        sp = np.zeros(len(mc)) if gross else D.spr[sl_]
        bar = _eval_units(nu, go, tr, unit, D.mo[sl_], mc, sp, ro, slip_frac, slip_pips, D.pip, comm, swap)
    return bar[i0 - w0:], int(ntr)


def suggest(trial, fam, bpd):
    P = {"rate": trial.suggest_float("rate", 0.1, max(0.2, min(50.0, bpd)), log=True)} if fam != "hedged_grid" else {}
    if fam == "bracket":
        P["sl"] = trial.suggest_float("sl", 0.25, 10.0, log=True)
        if trial.suggest_categorical("use_tp", [False, True]):
            P["tp"] = trial.suggest_float("tp", 0.2, 8.0, log=True)
        if trial.suggest_categorical("use_trail", [False, True]):
            P["trail"] = trial.suggest_float("trail", 0.25, 10.0, log=True)
        if trial.suggest_categorical("use_be", [False, True]):
            P["be"] = trial.suggest_float("be", 0.2, 4.0, log=True)
        if trial.suggest_categorical("use_part", [False, True]):
            P["part"] = trial.suggest_float("part", 0.2, 4.0, log=True)
        if trial.suggest_categorical("use_time", [False, True]):
            P["tmax"] = trial.suggest_int("tmax", 1, 500, log=True)
        P["size"] = trial.suggest_categorical("size", ["notional", "risk"])
    elif fam == "martingale":
        P["sl"] = trial.suggest_float("sl", 0.25, 10.0, log=True)
        P["tp"] = trial.suggest_float("tp", 0.2, 8.0, log=True)
        if trial.suggest_categorical("use_time", [False, True]):
            P["tmax"] = trial.suggest_int("tmax", 1, 500, log=True)
        P["mg_mode"] = trial.suggest_categorical("mg_mode", ["martingale", "anti"])
        P["mg_f"] = trial.suggest_float("mg_f", 1.2, 3.0)
        P["mg_k"] = trial.suggest_int("mg_k", 1, 6)
        P["size"] = "notional"
    elif fam == "pyramid":
        P["add"] = trial.suggest_float("add", 0.25, 5.0, log=True)
        P["N"] = trial.suggest_int("N", 2, 8)
        P["give"] = trial.suggest_float("give", 0.5, 10.0, log=True)
        P["stop"] = trial.suggest_float("stop", 0.5, 10.0, log=True)
    elif fam == "grid":
        P["step"] = trial.suggest_float("step", 0.25, 5.0, log=True)
        P["N"] = trial.suggest_int("N", 2, 10)
        P["g"] = trial.suggest_float("g", 1.0, 2.0)
        P["tp"] = trial.suggest_float("tp", 0.1, 5.0, log=True)
        P["sl"] = trial.suggest_float("sl", 0.5, 10.0, log=True)
    elif fam == "zone":
        P["zone"] = trial.suggest_float("zone", 0.25, 5.0, log=True)
        P["tp"] = trial.suggest_float("tp", 0.25, 5.0, log=True)
        P["K"] = trial.suggest_int("K", 1, 8)
    else:
        P["step"] = trial.suggest_float("step", 0.25, 5.0, log=True)
        P["basket"] = trial.suggest_float("basket", 0.5, 10.0, log=True)
        P["stop"] = trial.suggest_float("stop", 2.0, 50.0, log=True)
        P["N"] = trial.suggest_int("N", 3, 20)
    return P


def evaluate(fam, P, D, i0, i1, seeds, bpd, gross=False):
    """Mean over seeds of the daily net Sharpe on [i0, i1), per-seed bar returns, mean trades."""
    p = min(1.0, P.get("rate", 1.0) / bpd)
    sh, bars_, trades = [], [], []
    w0 = max(0, i0 - WARM)
    for s in (seeds if fam != "hedged_grid" else seeds[:1]):
        e = entries(p, s, w0, i0, i1)
        b, ntr = run_family(fam, P, D, e, i0, i1, gross)
        x, _ = daily_sharpe(b, D.day, i0)
        sh.append(x); bars_.append(b); trades.append(ntr)
    return float(np.mean(sh)), bars_, float(np.mean(trades))


def optimise(fam, D, i0, i1, seeds, bpd, n_trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        P = suggest(trial, fam, bpd)
        try:
            s, _, tr = evaluate(fam, P, D, i0, i1, seeds, bpd)
        except Exception:
            return -10.0
        return s if (tr >= MIN_TRADES and np.isfinite(s)) else -10.0
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    vals = np.array([t.value for t in st.trials if t.value is not None and t.value > -10])
    return suggest(optuna.trial.FixedTrial(st.best_params), fam, bpd), float(st.best_value), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "random_entry_opt"))
    a = ap.parse_args()
    import feature_scan as FS
    from fxlab.data.store import load_m1, M1_PATH_COLUMNS, TF_MINUTES
    from fxlab.data.instruments import get as instrument
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.tf == "M1":
        bars = load_m1(a.pair, a.start, a.end, columns=["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"])
        m1 = None
    else:
        bars = FS.cached_bars(a.pair, a.tf, a.start, a.end)
        m1 = load_m1(a.pair, a.start, a.end, columns=M1_PATH_COLUMNS)
    D = Data(bars, instrument(a.pair).pip, m1)
    del m1
    global N_TRIALS, N_HIND, TRAIN_SEEDS, TEST_SEEDS
    if D.n > BIG:
        N_TRIALS, N_HIND, TRAIN_SEEDS, TEST_SEEDS = min(N_TRIALS, 25), min(N_HIND, 40), 3, 8
    bpd = 1440.0 / TF_MINUTES.get(a.tf, 60) if a.tf in TF_MINUTES else 24.0
    tr_lo, tr_hi, te_hi = FS.folds(bars.index)
    out = {"pair": a.pair, "tf": a.tf, "bars": D.n, "folds": int(len(tr_lo)), "families": {}}
    daily = {}
    for fam in FAMILIES:
        tf0 = time.time()
        test_bar = np.zeros((TEST_SEEDS, D.n), np.float32); test_gross = np.zeros((TEST_SEEDS, D.n), np.float32)
        mask = np.zeros(D.n, bool)
        sel = []
        for f in range(len(tr_lo)):
            i0, i1, i2 = int(tr_lo[f]), int(tr_hi[f]), int(te_hi[f])
            P, s_in, _ = optimise(fam, D, i0, i1, [1000 * f + k for k in range(TRAIN_SEEDS)], bpd, N_TRIALS, seed=f)
            tseeds = [10 ** 6 + 1000 * f + k for k in range(TEST_SEEDS)]
            _, bs, ntr = evaluate(fam, P, D, i1, i2, tseeds, bpd)
            _, bg, _ = evaluate(fam, P, D, i1, i2, tseeds, bpd, gross=True)
            for k in range(len(bs)):
                test_bar[k, i1:i2] = bs[k]
                test_gross[k, i1:i2] = bg[k]
            if fam == "hedged_grid":                       # deterministic: one path
                test_bar[1:, i1:i2] = test_bar[0, i1:i2]; test_gross[1:, i1:i2] = test_gross[0, i1:i2]
            mask[i1:i2] = True
            sel.append({"fold": f, "train_sharpe": s_in, "params": P, "test_trades_mean": ntr})
        idx = np.flatnonzero(mask)
        i0a, i1a = int(idx[0]), int(idx[-1]) + 1
        per_seed = [daily_sharpe(test_bar[k, i0a:i1a], D.day, i0a)[0] for k in range(TEST_SEEDS)]
        per_seed_g = [daily_sharpe(test_gross[k, i0a:i1a], D.day, i0a)[0] for k in range(TEST_SEEDS)]
        ev = test_bar.mean(0)[i0a:i1a]
        ev_sh, ev_d = daily_sharpe(ev, D.day, i0a)
        ev_t = float(ev_d.mean() / ev_d.std() * math.sqrt(len(ev_d))) if ev_d.std() > 0 else 0.0
        evg_sh, _ = daily_sharpe(test_gross.mean(0)[i0a:i1a], D.day, i0a)
        # hindsight: optimise on the whole tested span, then re-randomise the entries on the same data
        Ph, sh_in, vals = optimise(fam, D, i0a, i1a, [7 * 10 ** 6 + k for k in range(TRAIN_SEEDS)], bpd, N_HIND, seed=999)
        sh_re, _, _ = evaluate(fam, Ph, D, i0a, i1a, [9 * 10 ** 6 + k for k in range(TEST_SEEDS)], bpd)
        bench = float("nan")      # (the trial distribution is far from normal; the re-randomised run is the test)
        out["families"][fam] = {
            "wf": {"single_trader_median_sharpe": float(np.median(per_seed)), "single_trader_sharpes": per_seed,
                   "single_trader_median_gross": float(np.median(per_seed_g)),
                   "expected_stream_sharpe": ev_sh, "expected_stream_t": ev_t, "expected_stream_gross_sharpe": evg_sh,
                   "expected_stream_ann_return": float(ev_d.mean() * 260), "days": int(len(ev_d)),
                   "median_train_sharpe": float(np.median([s["train_sharpe"] for s in sel]))},
            "hindsight": {"in_sample_best": sh_in, "rerandomised": sh_re, "deflation_benchmark": bench, "params": Ph},
            "selection": sel, "seconds": round(time.time() - tf0, 1)}
        daily[fam] = pd.Series(ev_d, dtype="float32")
        print(f"{a.pair} {a.tf} {fam:11s}: WF single-trader median {np.median(per_seed):+.2f} (gross {np.median(per_seed_g):+.2f}) "
              f"| expected stream {ev_sh:+.2f} t={ev_t:+.1f} gross {evg_sh:+.2f} | train median "
              f"{out['families'][fam]['wf']['median_train_sharpe']:+.2f} | hindsight in-sample {sh_in:+.2f} "
              f"re-randomised {sh_re:+.2f} (deflation bench {bench:.2f}) | {time.time() - tf0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, f"{a.pair}_{a.tf}.json"), "w"), indent=1, default=float)
    pd.DataFrame(daily).to_parquet(os.path.join(a.out, f"{a.pair}_{a.tf}_daily.parquet"))
    print(f"done {a.pair} {a.tf} in {out['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
