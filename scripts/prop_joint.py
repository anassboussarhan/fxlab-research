"""Several prop accounts at once on the same Gotobi nights (ledger item 3): FTMO 2-Step 5x, FundedNext Stellar 1-Step
3x, The5ers High Stakes Classic 4x, 100k each, the same simulated trader path (same start night) in every account.
Reported: P(at least one account paid), P(all paid), mean total net, fees, at the 2019-2025, 2026-shifted and zero
edges. Usage: python scripts/prop_joint.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_best as PB  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = (("FTMO 2-Step", 5, 100, 621.0), ("FundedNext Stellar 1-Step", 3, 100, 569.99), ("The5ers High Stakes Classic", 4, 100, 455.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()
    B = pd.read_csv(os.path.join(ROOT, "docs/gotobi_losses/trades_basket_stop_80_bp.csv"), index_col=0, parse_dates=True)
    B.index = pd.DatetimeIndex(B.index).normalize()
    B = B[(B.index >= "2019-01-01") & (B.index <= "2025-12-31")][["net", "mae"]].dropna()
    res = {}
    for scen, target in (("2019-2025", None), ("2026", 2.04e-4), ("zero", 0.0)):
        g = B.copy()
        if target is not None:
            d = g.net.mean() - target
            g["net"] = g.net - d
            g["mae"] = np.minimum(g.mae - d, np.minimum(g.net, 0.0))
        X, M = PB.daily(g)
        paid_any, paid_all, tot = None, None, 0.0
        per = {}
        for name, lev, size, fee in SETUP:
            r = PB.P[name]
            paid, npay, passed, first, fail = PB.simulate(X, M, r, a.n, 1, lev, lev)   # same seed = same start nights
            refund = (npay >= r["refund_after"]) if r["refund_after"] else np.zeros(a.n, bool)
            net = paid * size * 1000 + fee * refund - fee
            p = npay > 0
            per[name] = float(p.mean())
            paid_any = p if paid_any is None else (paid_any | p)
            paid_all = p if paid_all is None else (paid_all & p)
            tot = tot + net
        res[scen] = {"paid_each": per, "paid_at_least_one": float(paid_any.mean()), "paid_all": float(paid_all.mean()),
                     "mean_total_net": float(np.mean(tot)), "median_total_net": float(np.median(tot)),
                     "share_total_positive": float((tot > 0).mean()), "fees": sum(s[3] for s in SETUP)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "prop_joint.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
