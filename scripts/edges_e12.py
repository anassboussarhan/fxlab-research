"""E12 of docs/edges_batch_preregistration.md: FTMO 2-Step, Gotobi basket with the 80 bp stop on 2019-2025 nights;
5x dropping to 2.5x while equity is more than 4% below the phase / payout start (back to 5x within 2%), against a
fixed 5x and a fixed 3x. Counts if P(paid) at the 2026-shifted edge is at least 3 points above both.
Usage: python scripts/edges_e12.py --out DIR"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
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
        for name, le, lf, rule in (("fixed 5x", 5, 5, None), ("fixed 3x", 3, 3, None), ("5x, 2.5x after -4%", 5, 5, (2.5, 0.04, 0.02))):
            paid, npay, passed, first, fail = PB.simulate(X, M, PB.P["FTMO 2-Step"], a.n, 1, le, lf, rule)
            net = paid * 1e5 + 621 * (npay >= 1) - 621
            f = first.astype(float)
            f[f < 0] = np.inf
            res[f"{scen}|{name}"] = {"pass": float(passed.mean()), "paid": float((npay > 0).mean()),
                                     "mean_net_100k": float(net.mean()), "paid_within_12m": float((f <= 252).mean()),
                                     "fail_daily": float((fail == 1).mean()), "fail_max": float((fail == 2).mean())}
    p = {k.split("|")[1]: v["paid"] for k, v in res.items() if k.startswith("2026|")}
    res["counts"] = bool(p["5x, 2.5x after -4%"] >= max(p["fixed 5x"], p["fixed 3x"]) + 0.03)
    json.dump(res, open(os.path.join(a.out, "e12.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
