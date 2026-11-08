"""Round 13 diagnostic (information only; it cannot change a verdict): where the part-B units at the top of the
check period earn their money, by instrument and by instrument group, against the long-only benchmark, and how
often they are long. Usage: python scripts/long_diag.py --inp DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_combine as LC                                                   # noqa: E402
import long_lab as LL                                                       # noqa: E402

UNITS = [("S.DAILY.DOUBLE_SEVEN", "W1", "T0"), ("S.DAILY.CUM_RSI", "W1", "T0"), ("S.FAMOUS.CONNORS_RSI2", "W1", "T0"),
         ("S.DAILY.ABS_MOM", "D1", "T0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.inp, "**", "B_monthly_*.parquet"), recursive=True))],
                  ignore_index=True)
    if "month" not in B.columns:
        B = B.rename(columns={"ts": "month", "index": "month"})
    A = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.inp, "**", "A_daily_*.parquet"), recursive=True))],
                  ignore_index=True)
    A["month"] = pd.to_datetime(A.date).dt.to_period("M").astype(str)
    bh = A[A.system == "BH"].groupby(["symbol", "month"]).pnl.sum()
    groups = {"FX": LL.FX_SYMBOLS, "metals": ["XAUUSD", "XAGUSD"], "indices": ["USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY"]}
    out, per = {}, []
    for uid, tf, v in UNITS:
        g = B[(B.unit == uid) & (B.tf == tf) & (B.variant == v)]
        key = f"{uid}|{tf}|{v}"
        res = {}
        for gname, syms in list(groups.items()) + [("all", LL.ALL_SYMBOLS)]:
            x = g[g.symbol.isin(syms)].groupby("month")[["pnl", "gpnl", "n"]].sum()
            f, c = LC.split(x.pnl)
            bf, bc = LC.split(bh[bh.index.get_level_values(0).isin(syms)].groupby(level=1).sum())
            res[gname] = {"fit_sum_R": f.sum() / LL.RISK, "check_sum_R": c.sum() / LL.RISK, "fit_t": LC.nw_t(f),
                          "check_t": LC.nw_t(c), "check_trades": float(x[x.index >= LC.CHK[0]].n.sum()),
                          "corr_check_with_long_only": float(np.corrcoef(c.to_numpy(), bc.reindex(c.index).fillna(0).to_numpy())[0, 1])
                          if c.std() > 0 and bc.std() > 0 else None}
        out[key] = res
        for s, h in g.groupby("symbol"):
            x = h.set_index("month")
            f, c = LC.split(x.pnl)
            per.append({"unit": key, "symbol": s, "fit_sum_R": f.sum() / LL.RISK, "check_sum_R": c.sum() / LL.RISK,
                        "check_t": LC.nw_t(c), "check_trades": float(x[x.index >= LC.CHK[0]].n.sum())})
    P = pd.DataFrame(per)
    P.to_csv(os.path.join(a.out, "diag_by_symbol.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "diag_groups.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 200)
    for u, h in P.groupby("unit"):
        print(u)
        print(h.sort_values("check_sum_R", ascending=False).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
