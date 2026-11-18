"""Round 23 part IC (docs/round23_preregistration.md): the round-22 ICT / SMC families on a causal simulator.

The families of scripts/r22_I.py are run with `trade` replaced by a recorder, so every signal's order is kept and
none is resolved. A causal event-driven engine then executes the orders bar by bar:
  - an order is active from its first bar i0 for expiry bars (market orders: bar i0 only, at its open);
  - a cancel level touched first removes it; a limit fills when the bar trades at it (at the better of the open
    and the limit);
  - one position at a time: on a fill every other order is cancelled, and orders created while in a position are
    ignored (only orders with i0 after the exit bar are eligible);
  - a fill at an open beyond the stop is closed at once at that open (round 22 dropped it);
  - exits as round 22's trade(): stop first (at the stop on the entry bar, at the worse of the open and the stop
    later), target from the next bar (at the better of the open and the target), time exits, max hold at the close.
Timeframes: US index CFDs 5min / 15min / 30min / 1h of the 09:30-16:00 New York session; other instruments
15min / 1h / 4h of the full day. Placebo: the same orders with a random side (levels mirrored), same engine.
Usage: python scripts/r23_IC.py --symbol SYM [--n 128] --out DIR      (SYM may be "A+B" for SMT)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numba as nb
import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r22_I as I  # noqa: E402
import wf23  # noqa: E402

US_IDX = ("USA500IDXUSD", "USATECHIDXUSD", "USA30IDXUSD", "USSC2000IDXUSD")
ORD = []


def recorder(B, i0, d, entry, stop, tgt, max_hold, limit=None, expiry=0, cancel=None, t_exit=None):
    if stop is None or not np.isfinite(stop) or i0 >= B.n:
        return None
    ORD.append((int(i0), int(d), 0 if limit is None else 1, np.nan if limit is None else float(limit), float(stop),
                np.nan if tgt is None else float(tgt), int(max_hold), int(expiry), np.nan if cancel is None else float(cancel),
                np.nan if t_exit is None else float(t_exit)))
    return None


@nb.njit(cache=True)
def engine(o, h, l, c, spr, day, nyh, fut, pip, ssf, sfp, com, i0, d, islim, lvl, stp, tgt, mh, exp, can, tex):
    n, m = len(o), len(i0)
    J = np.empty(m, np.int64)
    NET = np.empty(m)
    cnt = 0
    t = 0
    q0 = 0                                                                 # orders are sorted by i0
    while t < n:
        while q0 < m and i0[q0] + max(exp[q0], 0) < t:
            q0 += 1
        fill = -1
        q = q0
        while q < m and i0[q] <= t:                                        # market orders due at this open first
            if islim[q] == 0 and i0[q] == t:
                fill = q
                break
            q += 1
        if fill < 0:
            q = q0
            while q < m and i0[q] <= t:                                    # then resting limits, oldest first
                if islim[q] == 1 and i0[q] + exp[q] >= t:
                    dq = d[q]
                    cancelled = False
                    if can[q] == can[q]:
                        for k in range(i0[q], t + 1):                      # cancel level touched first
                            if (dq > 0 and h[k] >= can[q]) or (dq < 0 and l[k] <= can[q]):
                                cancelled = True
                                break
                    if not cancelled and ((dq > 0 and l[t] <= lvl[q]) or (dq < 0 and h[t] >= lvl[q])):
                        fill = q
                        break
                q += 1
        if fill < 0:
            t += 1
            continue
        q = fill
        dq = d[q]
        if islim[q] == 0:
            E = o[t]
        else:
            E = min(o[t], lvl[q]) if dq > 0 else max(o[t], lvl[q])
        j = t
        if (dq > 0 and stp[q] >= E) or (dq < 0 and stp[q] <= E):
            kx, X = j, o[j]
        else:
            X = np.nan
            kx = -1
            end = min(n - 1, j + mh[q])
            for k in range(j, end + 1):
                if tex[q] == tex[q] and k > j and day[k] != day[j]:
                    X, kx = o[k], k
                    break
                if tex[q] == tex[q] and nyh[k] >= tex[q] and k > j:
                    X, kx = o[k], k
                    break
                if (dq > 0 and l[k] <= stp[q]) or (dq < 0 and h[k] >= stp[q]):
                    if k == j:
                        X = stp[q]
                    elif dq > 0:
                        X = min(o[k], stp[q])
                    else:
                        X = max(o[k], stp[q])
                    kx = k
                    break
                if k > j and tgt[q] == tgt[q] and ((dq > 0 and h[k] >= tgt[q]) or (dq < 0 and l[k] <= tgt[q])):
                    X = max(o[k], tgt[q]) if dq > 0 else min(o[k], tgt[q])
                    kx = k
                    break
            if kx < 0:
                kx = end
                X = c[kx]
        g = dq * (X / E - 1.0)
        if fut == fut:
            cost = fut / E + fut / X
        else:
            s0, s1 = spr[min(j, n - 1)], spr[min(kx, n - 1)]
            cost = (0.5 * s0 + ssf * s0 + sfp * pip) / E + com + (0.5 * s1 + ssf * s1 + sfp * pip) / X + com
        J[cnt] = j
        NET[cnt] = g - cost
        cnt += 1
        t = kx + 1
        while q0 < m and i0[q0] < t:                                       # orders from before the exit are gone
            q0 += 1
    return J[:cnt], NET[:cnt]


def run_orders(B, orders, placebo_seed=None):
    if not orders:
        return np.empty(0, np.int64), np.empty(0)
    A = np.array(orders, dtype=np.float64)
    A = A[np.lexsort((np.arange(len(A)), A[:, 0]))]
    i0, d, islim = A[:, 0].astype(np.int64), A[:, 1].astype(np.int64), A[:, 2].astype(np.int64)
    lvl, stp, tgt = A[:, 3].copy(), A[:, 4].copy(), A[:, 5].copy()
    mh, exp, can, tex = A[:, 6].astype(np.int64), A[:, 7].astype(np.int64), A[:, 8].copy(), A[:, 9].copy()
    if placebo_seed is not None:                                           # random side, levels mirrored
        rs = np.random.default_rng(placebo_seed).choice([-1, 1], len(d))
        flip = rs != d
        ref = np.where(islim == 1, lvl, B.o[np.minimum(i0, B.n - 1)])
        stp = np.where(flip, 2 * ref - stp, stp)
        tgt = np.where(flip, 2 * ref - tgt, tgt)
        can = np.where(flip, np.nan, can)
        d = rs
    from fxlab.execution.costs import BASE
    day = pd.factorize(pd.Index(B.day))[0].astype(np.int64)
    fut = B.fut if B.fut else np.nan
    return engine(B.o, B.h, B.l, B.c, B.spr, day, B.nyh, float(fut), float(B.pip), BASE.slip_spread_frac,
                  BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4, i0, d, islim, lvl, stp, tgt, mh, exp, can, tex)


def daily(J, NET, B, cal):
    s = pd.Series(0.0, index=cal)
    if len(J) == 0:
        return s
    v = pd.Series(NET, index=pd.DatetimeIndex(np.asarray(B.day)[J])).groupby(level=0).sum()
    return s.add(v.reindex(cal).fillna(0.0), fill_value=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    I.trade = recorder
    t0 = time.time()
    groups = {}
    if "+" in a.symbol:
        s1, s2 = a.symbol.split("+")
        (m1, p1), (m2, p2) = I.load(s1), I.load(s2)
        session = s1 in US_IDX
        tfs = ("5min", "15min", "30min", "1h") if session else ("15min", "1h", "4h")
        BB = {}
        for tf in tfs:
            b1, b2 = I.bars(m1, tf, session), I.bars(m2, tf, session)
            ix = b1.index.intersection(b2.index)
            BB[tf] = (I.Bars(b1.reindex(ix), s1, p1), I.Bars(b2.reindex(ix), s2, p2))
        cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]][0].day)))
        U = qmc.Sobol(4, scramble=True, seed=2322).random(a.n)
        M, Mp = [], []
        for ci, u in enumerate(U):
            tf = I.pick(u[3], list(tfs))
            A_, B_ = BB[tf]
            ORD.clear()
            I.fam_SMT(A_, B_, u)
            od = list(ORD)
            M.append(daily(*run_orders(A_, od), A_, cal).to_numpy())
            Mp.append(daily(*run_orders(A_, od, placebo_seed=ci), A_, cal).to_numpy())
        groups["SMT"] = (np.vstack(M), np.vstack(Mp))
    else:
        mid, pip = I.load(a.symbol)
        session = a.symbol in US_IDX
        tfs = ("5min", "15min", "30min", "1h") if session else ("15min", "1h", "4h")
        BB = {tf: I.Bars(I.bars(mid, tf, session), a.symbol, pip) for tf in tfs}
        BBfull = {tf: I.Bars(I.bars(mid, tf, False), a.symbol, pip) for tf in tfs} if session else BB
        cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]].day)))
        LV = {tf: I.make_levels(BBfull[tf]) for tf in tfs}
        for fam, dims in I.FAMS.items():
            if fam == "CBDR" and session:
                continue
            U = qmc.Sobol(dims + 1, scramble=True, seed=2300 + len(fam)).random(a.n)
            M, Mp = [], []
            for ci, u in enumerate(U):
                tf = I.pick(u[-1], list(tfs))
                Bx = BBfull[tf] if fam in ("LVL", "CBDR", "SES") else BB[tf]
                ORD.clear()
                if fam == "LVL":
                    I.fam_LVL(Bx, u, LV[tf])
                elif fam == "SES":
                    I.fam_SES(Bx, u, session)
                else:
                    getattr(I, f"fam_{fam}")(Bx, u)
                od = list(ORD)
                M.append(daily(*run_orders(Bx, od), Bx, cal).to_numpy())
                Mp.append(daily(*run_orders(Bx, od, placebo_seed=ci), Bx, cal).to_numpy())
            groups[fam] = (np.vstack(M), np.vstack(Mp))
            print(a.symbol, fam, "done", round(time.time() - t0), "s", flush=True)
    rows, ser = [], {}
    first = max(2010, int(cal.year.min()) + 3)
    for fam, (M, Mp) in groups.items():
        for plc, X in ((False, M), (True, Mp)):
            res = wf23.both(wf23.Agg(cal, P=X), first, min_active=20, min_valid=50)
            for kind, (s, st, own, ch) in res.items():
                lab = f"IC-{fam}{'-PLACEBO' if plc else ''}-{kind}"
                rows.append({"part": "IC", "family": f"IC-{fam}", "symbol": a.symbol, "placebo": plc, "selection": kind,
                             "n_cfg": int(X.shape[0]), **st, "own": own, "mean_all_cfg": float(np.nanmean(X)),
                             "choices": json.dumps({str(k): v for k, v in list(ch.items())[-2:]})[:1500]})
                ser[lab] = s
    tag = a.symbol.replace("+", "_")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"IC_rows_{tag}.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, f"IC_series_{tag}.parquet"))
    print(pd.DataFrame(rows)[["family", "placebo", "selection", "p1_mean", "p2_mean", "p2_t", "own"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
