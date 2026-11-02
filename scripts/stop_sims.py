"""Prop-firm and own-account simulations of the Gotobi basket with and without the chosen stop (stop 0.75 W, chosen
on 2007-2018 in scripts/gotobi_risk_lab.py), at the development edge and at the 2026 edge (every variant shifted by
the same amount that brings the no-stop basket to its 2026 mean, +2.04 bp). Rules and firms as in final_sims.py.
Usage: python scripts/stop_sims.py --dir DIR_WITH_trades_basket_*.csv --out DIR"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_sims as FS  # noqa: E402
import gotobi_opt_sims as GS  # noqa: E402
import prop_firms_ev as PF  # noqa: E402

VARIANTS = ("none", "stop_0.75_W", "stop_40_bp")
LEV_PASS = (3, 5, 8, 10, 15, 20)
LEV_FUND = (3, 5, 8, 10, 15)


def load(d, v):
    f = pd.read_csv(os.path.join(d, f"basket_{v}.csv"), index_col=0, parse_dates=True)
    f.index = pd.DatetimeIndex(f.index).normalize()
    return f[["net", "mae"]].dropna().sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = {v: load(a.dir, v) for v in VARIANTS}
    shift = B["none"].net.mean() - FS.TARGET_2026
    rows, acc = [], []
    for v, f in B.items():
        for s, g in (("dev", f), ("2026", FS.shifted(f, f.net.mean() - shift))):
            X, M = FS.daily(g)
            per_year = len(g) / ((g.index.max() - g.index.min()).days / 365.25)
            for firm, rules in FS.FIRMS.items():
                for lp, lf in itertools.product(LEV_PASS, LEV_FUND):
                    r = PF.simulate(X, rules, n=a.n, seed=1, M=M, scale_eval=lp, scale_funded=lf)
                    rows.append({"variant": v, "scenario": s, "firm": firm, "lev_pass": lp, "lev_funded": lf, **r})
            for L in (5, 10, 15, 20, 30, 40):
                for yrs in (1, 3):
                    acc.append({"variant": v, "scenario": s, **GS.account(g.net.to_numpy(), g.mae.to_numpy(), L, yrs, per_year)})
            print(v, s, "mean bp", round(g.net.mean() * 1e4, 2), flush=True)
    P, A = pd.DataFrame(rows), pd.DataFrame(acc)
    P.to_csv(os.path.join(a.out, "prop.csv"), index=False)
    A.to_csv(os.path.join(a.out, "account.csv"), index=False)


if __name__ == "__main__":
    main()
