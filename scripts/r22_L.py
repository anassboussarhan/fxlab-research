"""Round 22 part L: the strategy libraries re-selected every year (docs/round22_preregistration.md).

Reads per-configuration daily net P&L files (daily17_SYM.npz from intraday17, daily15c_SYM.npz / daily15_SYM.npz
from sweep15) one artifact at a time, and runs the walk-forward on:
  - each rule x timeframe x direction (its configurations are its grid), and
  - ALL: every real configuration of the instrument (the whole library is the grid).
Placebo units get the same treatment (reported as the null).
Usage: python scripts/r22_L.py --files DIR --out DIR   (DIR holds the npz files of one or more symbols)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf22  # noqa: E402


def symbol_of(f):
    b = os.path.basename(f)[:-4]
    return b.split("_", 1)[1]


def run_symbol(files, out_rows, pooled, placebo_rows):
    mats, metas, days_ref = [], [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        eng = "D" if os.path.basename(f).startswith("daily15") else "I"
        days = pd.to_datetime(pd.Series(z["days"]).astype(str).str[:10])
        U = pd.DataFrame({"unit": z["unit"].astype(str), "cfg": z["cfg"].astype(str), "dir": z["dir"].astype(int)})
        P = z["pnl"].astype(np.float64)
        S = pd.DataFrame(P.T, index=pd.DatetimeIndex(days))
        S = S[~S.index.duplicated(keep="last")]
        mats.append((eng, U, S))
    cal = sorted(set().union(*[set(S.index) for _, _, S in mats]))
    cal = pd.DatetimeIndex(cal)
    cal = cal[cal <= wf22.DEV_END]
    rows_all, keys = [], []
    for eng, U, S in mats:
        S = S.reindex(cal)
        keep = (U.cfg != "terciles").to_numpy()
        for i in np.flatnonzero(keep):
            keys.append((eng, U.unit.iloc[i], int(U.dir.iloc[i]), U.cfg.iloc[i]))
        rows_all.append(S.to_numpy().T[keep])
    M = np.vstack(rows_all)                                   # configs x days; NaN where the engine had no day
    K = pd.DataFrame(keys, columns=["engine", "unit", "dir", "cfg"])
    K["placebo"] = K.unit.str.startswith("PLACEBO")
    return K, M, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.files, "**", "daily1[57]*_*.npz"), recursive=True))
    bysym = {}
    for f in files:
        bysym.setdefault(symbol_of(f), []).append(f)
    rows = []
    pool_sum, pool_n = {}, {}
    for sym, fs in sorted(bysym.items()):
        K, M, cal = run_symbol(fs, None, None, None)
        first = max(2010, int(cal.year.min()) + 3)
        groups = [("ALL", np.flatnonzero(~K.placebo.to_numpy()))]
        groups += [("PLACEBO_ALL", np.flatnonzero(K.placebo.to_numpy()))]
        for (eng, unit, d), g in K.groupby(["engine", "unit", "dir"]):
            groups.append((f"{eng}:{unit}:{d}", g.index.to_numpy()))
        ser = {}
        for fam, ii in groups:
            if not len(ii):
                continue
            s5, s1, ch = wf22.walk_forward(M[ii], cal, first)
            st = wf22.stream_stats(s5)
            st1 = wf22.stream_stats(s1)
            rows.append({"part": "L", "family": fam, "symbol": sym, "placebo": fam.startswith("PLACEBO") or ":PLACEBO" in fam,
                         "n_cfg": int(len(ii)), **st, "top1_p2_mean": st1["p2_mean"], "top1_p2_t": st1["p2_t"],
                         "own": wf22.counts_rule(st),
                         "choices": json.dumps({str(k): [K.cfg.iloc[ii[j]] + "|" + K.unit.iloc[ii[j]] for j in v["cfg"][:5]]
                                                for k, v in ch.items()})[:2000] if fam == "ALL" else ""})
            rows[-1]["selection"] = "default"
            if os.environ.get("R23_NESTED"):                               # round 23 part N
                import wf23
                rn, sn = wf23.nested_row(rows[-1], M[ii], cal, first)
                rows.append(rn)
                if fam in ("ALL", "PLACEBO_ALL"):
                    ser[fam + "|nested"] = sn
                nk = fam + "|nested"
                if nk in pool_sum:
                    pool_sum[nk] = pool_sum[nk].add(sn, fill_value=0.0)
                    pool_n[nk] = pool_n[nk].add(sn.notna().astype(float), fill_value=0.0)
                elif len(sn):
                    pool_sum[nk], pool_n[nk] = sn.copy(), sn.notna().astype(float)
            if fam in ("ALL", "PLACEBO_ALL"):
                ser[fam] = s5
            key = fam.split(":", 1)[1] if ":" in fam else fam          # pooled across instruments (both engines' units)
            key = fam
            if key in pool_sum:
                pool_sum[key] = pool_sum[key].add(s5, fill_value=0.0)
                pool_n[key] = pool_n[key].add(s5.notna().astype(float), fill_value=0.0)
            else:
                pool_sum[key], pool_n[key] = s5.copy(), s5.notna().astype(float)
        pd.DataFrame(ser).to_parquet(os.path.join(a.out, f"L_series_{sym}.parquet"))
        print(sym, "configs", M.shape, "first OOS", first, "families", len(groups), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "L_rows_partial.csv"), index=False)
    for fam in pool_sum:
        n = pool_n[fam]
        s5 = (pool_sum[fam] / n.where(n > 0)).dropna()
        st = wf22.stream_stats(s5)
        base = fam[:-len("|nested")] if fam.endswith("|nested") else fam
        rows.append({"part": "L", "family": base, "symbol": "POOLED", "placebo": base.startswith("PLACEBO") or ":PLACEBO" in base,
                     "selection": "nested" if fam.endswith("|nested") else "default",
                     "n_cfg": int(n.max()), **st, "own": wf22.counts_rule(st), "choices": ""})
        if fam in ("ALL", "PLACEBO_ALL"):
            pd.DataFrame({fam: s5}).to_parquet(os.path.join(a.out, f"L_series_POOLED_{fam}.parquet"))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "L_rows.csv"), index=False)
    print(R[R.family == "ALL"][["symbol", "n_cfg", "oos_first", "p1_mean", "p2_mean", "p2_t", "own"]].round(6).to_string())


if __name__ == "__main__":
    main()
