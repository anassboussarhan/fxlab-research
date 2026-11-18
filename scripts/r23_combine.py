"""Round 23 combine (docs/round23_preregistration.md): every unit stream (default and nested), pooled streams,
Benjamini-Hochberg at q = 0.10 over all real streams, placebo rates.
Usage: python scripts/r23_combine.py --inp DIR --out DIR
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


def stats_row(part, family, symbol, sel, s, placebo=False, n=0):
    s = s.dropna()
    st = wf22.stream_stats(s if len(s) else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    return {"part": part, "family": family, "symbol": symbol, "selection": sel, "placebo": placebo, "n_cfg": n, **st,
            "own": wf22.counts_rule(st) if len(s) else False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    def g(p):
        """Files matching p. Artifacts sit at inp/RUN/ARTIFACT/...: when the same artifact file exists in two runs
        (a job rerun), only the later run's copy is used."""
        fs = sorted(glob.glob(os.path.join(a.inp, "**", p), recursive=True))
        best = {}
        for f in fs:
            rel = os.path.relpath(f, a.inp).split(os.sep)
            run, rest = (rel[0], os.sep.join(rel[1:])) if rel[0].isdigit() else ("0", os.sep.join(rel))
            if rest not in best or int(run) > int(best[rest][0]):
                best[rest] = (run, f)
        if len(best) < len(fs):
            print(p, "files replaced by a later run:", len(fs) - len(best))
        return sorted(f for _, f in best.values())
    parts = []
    pats = {"LW_rows_*.csv": "LW", "IC_rows_*.csv": "IC", "T2_rows_*.csv": "T2", "C30_rows.csv": "C30", "ETF_rows.csv": "ETF",
            "D_rows.csv": "N", "MB_rows.csv": "N", "C_rows.csv": "N", "C1_rows.csv": "N", "S_rows.csv": "N", "K_rows.csv": "N",
            "V_rows.csv": "N", "E_rows.csv": "N", "EF_rows.csv": "N", "X_rows_*.csv": "N", "P_row_*.csv": "N", "L_rows.csv": "N"}
    for pat, grp in pats.items():
        for f in g(pat):
            try:
                x = pd.read_csv(f)
            except Exception as ex:                                        # noqa: BLE001
                print("skip", f, repr(ex)[:80])
                continue
            x["group"] = grp
            if pat == "L_rows.csv":
                x = x[x.symbol != "POOLED"]                                # per-instrument jobs: pooled below
            parts.append(x)
    R = pd.concat(parts, ignore_index=True)
    R["selection"] = R.get("selection", pd.Series("default", index=R.index)).fillna("default")
    R["placebo"] = R["placebo"].fillna(False).astype(bool)
    pooled = []
    # LW: each strategy pooled over instruments and timeframes (sum / count of unit streams per day)
    acc = {}
    for f in g("LW_pooled_*.parquet"):
        x = pd.read_parquet(f)
        for c in x.columns:
            key, kind, what = c.rsplit("|", 2)
            d = acc.setdefault((key, kind), {"sum": None, "n": None})
            v = x[c].astype(np.float64)
            d[what] = v if d[what] is None else d[what].add(v, fill_value=0.0)
    plc = {}
    for (key, kind), d in acc.items():
        n = d["n"]
        s = (d["sum"] / n.where(n > 0)).dropna()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        is_p = key.startswith("PLACEBO")
        pooled.append(stats_row("LW", f"LW-{key}", "POOLED", kind, s, is_p, int(n.max())))
        if is_p:
            plc.setdefault(kind, []).append(s)
    # IC, T2, X, P and L-ALL pooled across instruments (equal weight per day)
    def parse(pat, c):
        sel = "nested" if c.endswith(("nested", "|nested")) else "default"
        if pat.startswith("IC_"):
            base = c.rsplit("-", 1)[0]                                     # IC-FAM[-PLACEBO]
            return base.replace("-PLACEBO", ""), sel, "-PLACEBO" in c
        if pat.startswith("T2_"):
            return "T2_time_windows", sel, False
        if pat.startswith("P_"):
            return "P-options", sel, False
        base = c.replace("|nested", "")
        return base, sel, base.startswith("PLACEBO")
    for part, pat in (("IC", "IC_series_*.parquet"), ("T2", "T2_series_*.parquet"), ("N", "X_series_*.parquet"),
                      ("N", "P_series_*.parquet"), ("N", "L_series_*.parquet")):
        S = {}
        for f in g(pat):
            if "POOLED" in os.path.basename(f):
                continue
            x = pd.read_parquet(f)
            for c in x.columns:
                if c.endswith("_top1"):
                    continue
                S.setdefault(parse(pat, c), []).append(x[c].dropna())
        for (fam, sel, is_p), lst in S.items():
            if len(lst) < 2:
                continue
            sx = pd.concat(lst, axis=1).mean(axis=1, skipna=True).dropna()
            pooled.append(stats_row(part, fam, "POOLED", sel, sx, is_p, len(lst)))
    if pooled:
        P = pd.DataFrame(pooled)
        P["group"] = P["part"].where(P["part"].isin(["LW", "IC", "T2"]), "N")
        R = pd.concat([R, P], ignore_index=True)
    if "counted" in R:
        R["counted"] = R["counted"].fillna(True).astype(bool)
    else:
        R["counted"] = True
    real = R[~R.placebo & R.counted & R.p2_t.notna()].copy()
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
    R.drop(columns=["choices"], errors="ignore").to_csv(os.path.join(a.out, "r23_streams.csv.gz"), index=False)
    pl = R[R.placebo & R.p2_t.notna()]
    keep = ["group", "part", "family", "symbol", "unit", "selection", "oos_first", "p1_mean", "p2_mean", "p2_t", "own", "bh"]
    summ = {"streams": int(len(R)), "real_streams_tested": int(len(real)), "own_rule": int(real.own.sum()),
            "bh": int(real.bh.sum()), "counts": int(real.counts.sum()), "first_bh_threshold_p": 0.10 / max(len(real), 1),
            "first_bh_threshold_t": float(norm.ppf(1 - 0.10 / max(len(real), 1))),
            "by_group": real.groupby(["group", "selection"]).agg(streams=("p", "size"), own=("own", "sum"),
                                                                 med_p2_t=("p2_t", "median"), max_p2_t=("p2_t", "max"))
            .round(3).reset_index().to_dict("records"),
            "top_real": real.sort_values("p2_t", ascending=False).head(60)[[c for c in keep if c in real]].round(6).to_dict("records"),
            "placebo": {"streams": int(len(pl)), "own_rule": int(pl.own.sum()) if len(pl) else 0,
                        "own_share": float(pl.own.mean()) if len(pl) else None,
                        "median_p2_t": float(pl.p2_t.median()) if len(pl) else None,
                        "p95_p2_t": float(pl.p2_t.quantile(0.95)) if len(pl) else None,
                        "max_p2_t": float(pl.p2_t.max()) if len(pl) else None},
            "real_own_share": float(real.own.mean()) if len(real) else None,
            "counting": real[real.counts][[c for c in keep if c in real]].round(6).to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "r23_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps({k: summ[k] for k in ("streams", "real_streams_tested", "own_rule", "bh", "counts", "first_bh_threshold_t",
                                          "placebo", "real_own_share")}, indent=1, default=str))
    print(pd.DataFrame(summ["by_group"]).to_string())
    print(real.sort_values("p2_t", ascending=False)[[c for c in keep if c in real]].head(60).to_string())


if __name__ == "__main__":
    main()
