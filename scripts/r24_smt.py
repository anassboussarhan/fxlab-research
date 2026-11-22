"""Round 24 R3 (docs/round24_preregistration.md): IC-SMT ported to the order recorder of scripts/r23_IC.py.

Signal as round 22's `fam_SMT`: instrument A makes a new lb-bar high (low) that B does not confirm. Then either fade
A ("fade_leader") or trade B against the same move ("follow_laggard"), entering at the next open and leaving after
`hold` bars. Trades spanning more than 60 hours are skipped.

Orders go through the causal engine of round 23:
- A's orders on A's bars, B's orders on B's bars;
- the stop sits 50% away, so the exit is in effect the time exit, as in round 22.
Placebos: the engine's random-side placebo. Usage: python scripts/r24_smt.py --symbol A+B --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r23_IC as IC  # noqa: E402
import wf23  # noqa: E402

I = IC.I


def smt_orders(A, Bb, u):
    lb = int(I.rng(u[0], 6, 72))
    hold = int(I.rng(u[1], 3, 48))
    mode = I.pick(u[2], ["fade_leader", "follow_laggard"])
    hA = pd.Series(A.h).rolling(lb).max().shift().to_numpy()
    lA = pd.Series(A.l).rolling(lb).min().shift().to_numpy()
    hB = pd.Series(Bb.h).rolling(lb).max().shift().to_numpy()
    lB = pd.Series(Bb.l).rolling(lb).min().shift().to_numpy()
    oa, ob, busy = [], [], -1
    for b in range(lb + 1, A.n - hold - 1):
        if b <= busy:
            continue
        up = A.h[b] > hA[b] and Bb.h[b] <= hB[b]
        dn = A.l[b] < lA[b] and Bb.l[b] >= lB[b]
        if up == dn:
            continue
        X, lst = (A, oa) if mode == "fade_leader" else (Bb, ob)
        d = -1 if up else 1
        if (X.idx[b + hold] - X.idx[b + 1]) > pd.Timedelta(hours=60):
            continue
        E = X.o[b + 1]
        IC.ORD.clear()
        IC.recorder(X, b + 1, d, None, E * (1 - 0.5 * d), None, hold - 1)
        lst += list(IC.ORD)
        busy = b + hold
    return oa, ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s1, s2 = a.symbol.split("+")
    (m1, p1), (m2, p2) = I.load(s1), I.load(s2)
    session = s1 in IC.US_IDX
    tfs = ("5min", "15min", "30min", "1h") if session else ("15min", "1h", "4h")
    BB = {}
    for tf in tfs:
        b1, b2 = I.bars(m1, tf, session), I.bars(m2, tf, session)
        ix = b1.index.intersection(b2.index)
        BB[tf] = (I.Bars(b1.reindex(ix), s1, p1), I.Bars(b2.reindex(ix), s2, p2))
    cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]][0].day)))
    U = qmc.Sobol(4, scramble=True, seed=2422).random(a.n)
    M, Mp, n_tr = [], [], []
    for ci, u in enumerate(U):
        tf = I.pick(u[3], list(tfs))
        A_, B_ = BB[tf]
        oa, ob = smt_orders(A_, B_, u)
        n_tr.append(len(oa) + len(ob))
        M.append((IC.daily(*IC.run_orders(A_, oa), A_, cal) + IC.daily(*IC.run_orders(B_, ob), B_, cal)).to_numpy())
        Mp.append((IC.daily(*IC.run_orders(A_, oa, placebo_seed=ci), A_, cal)
                   + IC.daily(*IC.run_orders(B_, ob, placebo_seed=10_000 + ci), B_, cal)).to_numpy())
    rows, ser = [], {}
    first = max(2010, int(cal.year.min()) + 3)
    for plc, X in ((False, np.vstack(M)), (True, np.vstack(Mp))):
        res = wf23.both(wf23.Agg(cal, P=X), first, min_active=20, min_valid=50)
        for kind, (s, st, own, ch) in res.items():
            rows.append({"part": "IC", "family": "IC-SMT", "symbol": a.symbol, "placebo": plc, "selection": kind,
                         "n_cfg": int(X.shape[0]), **st, "own": own, "mean_all_cfg": float(np.nanmean(X)),
                         "choices": json.dumps({str(k): v for k, v in list(ch.items())[-2:]})[:1500]})
            ser[f"IC-SMT{'-PLACEBO' if plc else ''}-{kind}"] = s
    tag = a.symbol.replace("+", "_")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"IC_rows_{tag}.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, f"IC_series_{tag}.parquet"))
    print("orders per configuration: median", int(np.median(n_tr)), "min", min(n_tr), flush=True)
    print(pd.DataFrame(rows)[["family", "placebo", "selection", "p1_mean", "p2_mean", "p2_t", "own"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
