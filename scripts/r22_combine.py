"""Round 22 combine: every walk-forward stream, pooled family streams, Benjamini-Hochberg over all non-placebo
streams (docs/round22_preregistration.md).
Usage: python scripts/r22_combine.py --inp DIR --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf22  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = lambda p: sorted(glob.glob(os.path.join(a.inp, "**", p), recursive=True))      # noqa: E731
    parts = []
    for pat in ("L_rows.csv", "T_row_*.csv", "I_rows_*.csv", "X_rows_*.csv", "D_rows.csv", "MB_rows.csv", "C_rows.csv",
                "C1_rows.csv", "S_rows.csv", "K_rows.csv", "V_rows.csv", "E_rows.csv", "EF_rows.csv", "P_row_*.csv"):
        for f in g(pat):
            try:
                parts.append(pd.read_csv(f))
            except Exception as ex:                                        # noqa: BLE001
                print("skip", f, repr(ex)[:80])
    R = pd.concat(parts, ignore_index=True)
    # pooled streams for T, I, X, P (equal weight across instruments per period)
    pooled = []
    for part, pat, key in (("T", "T_series_*.parquet", None), ("I", "I_series_*.parquet", None),
                           ("X", "X_series_*.parquet", None), ("P", "P_series_*.parquet", None)):
        S = {}
        for f in g(pat):
            x = pd.read_parquet(f)
            for c in x.columns:
                if c.endswith("_top1"):
                    continue
                fam = c.split("|")[0]
                S.setdefault(fam, []).append(x[c].dropna())
        for fam, lst in S.items():
            if len(lst) < 2:
                continue
            df = pd.concat(lst, axis=1)
            s = df.mean(axis=1, skipna=True).dropna()
            st = wf22.stream_stats(s)
            pooled.append({"part": part, "family": fam if part != "T" else "T_time_windows", "symbol": "POOLED", "placebo": False,
                           "n_cfg": len(lst), **st, "own": wf22.counts_rule(st)})
    if pooled:
        R = pd.concat([R, pd.DataFrame(pooled)], ignore_index=True)
    R["placebo"] = R["placebo"].fillna(False).astype(bool)
    real = R[~R.placebo & R.p2_t.notna()].copy()
    real["p"] = 1 - norm.cdf(real.p2_t.astype(float))
    p = real.p.to_numpy()
    o = np.argsort(p)
    thr = 0.10 * np.arange(1, len(p) + 1) / max(len(p), 1)
    passed = np.flatnonzero(p[o] <= thr)
    real["bh"] = False
    if len(passed):
        real.iloc[o[: passed.max() + 1], real.columns.get_loc("bh")] = True
    real["counts"] = real.own.astype(bool) & real.bh
    R = R.merge(real[["p", "bh", "counts"]], left_index=True, right_index=True, how="left")
    R.drop(columns=["choices"], errors="ignore").to_csv(os.path.join(a.out, "r22_streams.csv"), index=False)
    pl = R[R.placebo & R.p2_t.notna()]
    summ = {"streams": int(len(R)), "real_streams_tested": int(len(real)), "own_rule": int(real.own.sum()),
            "bh": int(real.bh.sum()), "counts": int(real.counts.sum()), "first_bh_threshold": 0.10 / max(len(real), 1),
            "by_part": real.groupby("part").agg(streams=("p", "size"), own=("own", "sum"), med_p2_t=("p2_t", "median"),
                                                max_p2_t=("p2_t", "max")).round(3).reset_index().to_dict("records"),
            "top_real": real.sort_values("p2_t", ascending=False).head(40).drop(columns=["choices"], errors="ignore")
            .round(6).to_dict("records"),
            "placebo": {"streams": int(len(pl)), "own_rule": int(pl.own.sum()) if len(pl) else 0,
                        "median_p2_t": float(pl.p2_t.median()) if len(pl) else None,
                        "p95_p2_t": float(pl.p2_t.quantile(0.95)) if len(pl) else None},
            "real_own_share": float(real.own.mean()) if len(real) else None,
            "counting": real[real.counts].drop(columns=["choices"], errors="ignore").round(6).to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "r22_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps({k: summ[k] for k in ("streams", "real_streams_tested", "own_rule", "bh", "counts", "first_bh_threshold",
                                          "placebo", "real_own_share")}, indent=1, default=str))
    print(pd.DataFrame(summ["by_part"]).to_string())
    cols = ["part", "family", "symbol", "oos_first", "p1_mean", "p2_mean", "p2_t", "own", "bh"]
    print(real.sort_values("p2_t", ascending=False)[cols].head(40).to_string())


if __name__ == "__main__":
    main()
