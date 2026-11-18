"""Round 22 part P: options on real end-of-day quotes (DoltHub, 2019-2025) with every parameter optimised
walk-forward (docs/round22_preregistration.md).

Per underlying, 128 scrambled-Sobol configurations of:
  structure  short put, short strangle, iron condor, put credit spread, covered call, delta-hedged short straddle,
             delta-hedged short strangle
  delta 0.05-0.50, wing 0.05-0.20 delta, maturity 14-60 days (nearest listed), take-profit none / 20-90% of the
  credit, stop none / 1-4 x the credit, exit 0-30 days before expiry, hedge band 0.05-0.50 delta (hedged ones).
Entries monthly (the engine's entry calendar for the maturity). Fills at mid +- 0.75 x half-spread, USD 0.65 a
contract; hedges 1 bp (SPY, DIA) or 3 bp of traded notional. Monthly series by entry month, P&L per unit of the
underlying's notional. Walk-forward: training from 2019, out-of-sample 2021-2025.
Usage: python scripts/r22_P.py --chain DIR --closes CSV --symbol SYM --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import options_strategy_lab as L  # noqa: E402
import options_edges_lab as OE  # noqa: E402
import wf22  # noqa: E402

F = L.PRIMARY_FILL
STRUCT = ["SP", "STRG", "IC", "PCS", "CC", "HSTRD", "HSTRG"]


def legs_of(kind, dl, w):
    lo = max(dl - w, 0.02)
    return {"SP": [("P", dl, -1)], "STRG": [("P", dl, -1), ("C", dl, -1)],
            "IC": [("P", dl, -1), ("P", lo, 1), ("C", dl, -1), ("C", lo, 1)], "PCS": [("P", dl, -1), ("P", lo, 1)],
            "CC": [("S", None, 1), ("C", dl, -1)], "HSTRD": [("P", "atm", -1), ("C", "atm", -1)],
            "HSTRG": [("P", dl, -1), ("C", dl, -1)]}[kind]


def evaluate_g(bk, d, legs, front, T, tp, sl, dx):
    """legs from L.build_position; generic management. Returns pnl per unit notional or None."""
    S0 = bk.und.px(d)
    opt = [(i, l) for i, l in enumerate(legs) if l[0] == "O"]
    stk = [l for l in legs if l[0] == "S"]
    if T.empty or any(not np.isfinite(T.iloc[0].get(f"m{i}", np.nan)) for i, _ in opt):
        return None
    row0 = T.iloc[0]
    entry_cf = 0.0
    for i, (_, cp, e, k, q) in opt:
        entry_cf += -q * (row0[f"m{i}"] + np.sign(q) * F * row0[f"h{i}"])
    comm = L.COMM * sum(abs(l[4]) for _, l in opt)
    credit = entry_cf

    def close_value(row):
        v = 0.0
        for i, (_, cp, e, k, q) in opt:
            m, h = row[f"m{i}"], row[f"h{i}"]
            if not np.isfinite(m):
                return np.nan
            v += q * (m - np.sign(q) * F * h)
        return v

    exit_d, exit_val = None, None
    for dd, row in T.iloc[1:].iterrows():
        if dd >= front:
            break
        left = (front - dd).days
        v = close_value(row)
        if not np.isfinite(v):
            continue
        pnl_now = entry_cf + v - 2 * comm
        if dx and left <= dx:
            exit_d, exit_val = dd, v
            break
        if tp is not None and credit > 0 and pnl_now >= tp * credit:
            exit_d, exit_val = dd, v
            break
        if sl is not None and credit > 0 and pnl_now <= -sl * credit:
            exit_d, exit_val = dd, v
            break
    comm_close = comm
    if exit_d is None:
        exit_d = front
        Sx = bk.und.px(front)
        exit_val = sum(q * (max(Sx - k, 0) if cp == "C" else max(k - Sx, 0)) for _, (_, cp, e, k, q) in opt)
        comm_close = 0.0
    pnl = entry_cf + exit_val - comm - comm_close
    for (_, _, _, _, q) in stk:
        r = bk.und.tr(d, exit_d)
        pnl += q * S0 * r - abs(q) * S0 * 2 * L.STK_COST
    return pnl / S0


def hedged_g(bk, d, front, legs3, band, hedge_cost, und):
    """Delta-hedged short options held to expiry; re-hedge when |net delta - hedge| > band."""
    S0 = und.px(d)
    days = [pd.Timestamp(t) for t in bk.dates if np.datetime64(d) <= t < np.datetime64(front)]
    if not days or days[0] != d or not np.isfinite(S0):
        return None
    paths = {}
    for cp, k, q in legs3:
        p = bk.path(front, cp, k, d, front)
        if not len(p) or p.index[0] != d:
            return None
        paths[(cp, k)] = p.reindex(days).ffill()
    cash = -sum(q * (float(paths[(cp, k)].mid.iloc[0]) + np.sign(q) * F * float(paths[(cp, k)].half.iloc[0]))
                for cp, k, q in legs3)
    cash -= L.COMM * sum(abs(q) for _, _, q in legs3)
    stock_pnl, cost, held = 0.0, 0.0, 0.0
    for i, t in enumerate(days):
        St = und.px(t)
        dl = 0.0
        for cp, k, q in legs3:
            x = OE.smile_delta(bk, t, front, cp, k)
            if x is None:
                x = (1.0 if St > k else 0.0) if cp == "C" else (-1.0 if St < k else 0.0)
            dl += q * x
        target = -dl
        if abs(target - held) > band or i == 0:
            cost += abs(target - held) * St * hedge_cost
            held = target
        t1 = days[i + 1] if i + 1 < len(days) else front
        a0, a1 = und.adj.loc[:t].iloc[-1], und.adj.loc[:t1].iloc[-1]
        stock_pnl += held * St * (a1 / a0 - 1.0)
    cost += abs(held) * und.px(front) * hedge_cost
    Sx = und.px(front)
    settle = sum(q * (max(Sx - k, 0.0) if cp == "C" else max(k - Sx, 0.0)) for cp, k, q in legs3)
    return (cash + settle + stock_pnl - cost) / S0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--closes", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= L.END]
    sym = a.symbol
    und = L.Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
    C = L.load_chain(a.chain, sym)
    C = L.trim_chain(C, und)
    bk = L.Book(C, und)
    hc = 1e-4 if sym in ("SPY", "DIA") else 3e-4
    print(sym, "chain rows", len(C), "dates", len(bk.dates), round(time.time() - t0), "s", flush=True)
    U = qmc.Sobol(8, scramble=True, seed=61).random(a.n)
    months = pd.period_range("2019-01", "2025-12", freq="M")
    M, desc = [], []
    entry_cache = {}
    for ci, u in enumerate(U):
        kind = STRUCT[min(int(u[0] * len(STRUCT)), len(STRUCT) - 1)]
        dl = 0.05 + u[1] * 0.45
        w = 0.05 + u[2] * 0.15
        dte = int(14 + u[3] * 46)
        tp = None if u[4] < 0.25 else 0.2 + (u[4] - 0.25) / 0.75 * 0.7
        sl = None if u[5] < 0.25 else 1.0 + (u[5] - 0.25) / 0.75 * 3.0
        dx = int(u[6] * 30) if u[6] >= 0.25 else 0
        band = 0.05 + u[7] * 0.45
        struct = legs_of(kind, round(dl, 2), round(w, 2))
        lo, hi = max(7, dte - 10), dte + 12
        if dte not in entry_cache:
            entry_cache[dte] = L.entry_dates(bk, dte if dte >= 21 else L.WEEKLY_DTE)
        vals = {}
        for d in entry_cache[dte]:
            S = und.px(d)
            if not np.isfinite(S):
                continue
            exps = bk.expiries(d)
            cand = [e for e in exps if lo <= (e - d).days <= hi and e <= L.END]
            if not cand:
                continue
            front = min(cand, key=lambda e: abs((e - d).days - dte))
            if und.has_event(d, front):
                continue
            try:
                if kind in ("HSTRD", "HSTRG"):
                    ks = []
                    for cp, tgt, q in struct:
                        k = OE.atm_strike(bk, d, front, S) if tgt == "atm" else bk.pick(d, front, cp, tgt, S)
                        if k is None:
                            ks = None
                            break
                        ks.append((cp, k, q))
                    if not ks:
                        continue
                    p = hedged_g(bk, d, front, ks, band, hc, und)
                else:
                    bp = L.build_position(bk, d, struct, dte if dte >= 21 else L.WEEKLY_DTE, S)
                    if bp is None:
                        continue
                    legs, fr = bp
                    T = L.cycle_paths(bk, d, legs, fr)
                    p = evaluate_g(bk, d, legs, fr, T, tp, sl, dx)
            except Exception:                                              # noqa: BLE001
                p = None
            if p is not None and np.isfinite(p):
                vals[pd.Timestamp(d).to_period("M")] = vals.get(pd.Timestamp(d).to_period("M"), 0.0) + p
        s = pd.Series(vals).reindex(months).fillna(0.0)
        M.append(s.to_numpy())
        desc.append({"struct": kind, "delta": round(dl, 2), "wing": round(w, 2), "dte": dte, "tp": None if tp is None else round(tp, 2),
                     "sl": None if sl is None else round(sl, 2), "exit_days": dx, "band": round(band, 2), "cycles": len(vals)})
        if ci % 16 == 0:
            print(sym, ci, kind, len(vals), round(time.time() - t0), "s", flush=True)
    Mx = np.vstack(M)
    dates = months.to_timestamp(how="end").normalize()
    s5, s1, ch = wf22.walk_forward(Mx, dates, 2021, min_active=12, min_train_years=2, min_valid=12)
    st, st1 = wf22.stream_stats(s5), wf22.stream_stats(s1)
    row = {"part": "P", "family": "P-options", "symbol": sym, "placebo": False, "n_cfg": int(Mx.shape[0]), **st,
           "top1_p2_mean": st1["p2_mean"], "top1_p2_t": st1["p2_t"], "own": wf22.counts_rule(st),
           "choices": json.dumps({str(k): [desc[i] for i in v["cfg"]] for k, v in ch.items()}, default=str)[:4000]}
    row["selection"] = "default"
    prow, pser = [row], {f"P-options|{sym}": s5}
    if os.environ.get("R23_NESTED"):                                       # round 23 part N
        import wf23
        rn, sn = wf23.nested_row(row, Mx, dates, 2021, min_active=12, min_train_years=2, min_valid=12)
        prow.append(rn)
        pser[f"P-options|{sym}|nested"] = sn
    pd.DataFrame(prow).to_csv(os.path.join(a.out, f"P_row_{sym}.csv"), index=False)
    pd.DataFrame(pser).to_parquet(os.path.join(a.out, f"P_series_{sym}.parquet"))
    pd.DataFrame(Mx, columns=[str(m) for m in months]).assign(desc=[json.dumps(d) for d in desc]).to_csv(
        os.path.join(a.out, f"P_matrix_{sym}.csv.gz"), index=False)
    print(json.dumps({k: row[k] for k in ("symbol", "oos_first", "p2_mean", "p2_t", "own")}), round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
