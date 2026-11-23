"""Round 24 combine (docs/round24_preregistration.md): every stream of R1-R11, pooled LW streams per part,
Benjamini-Hochberg at q = 0.10 over all round-24 real streams, placebo rates per part, and the sanity flags:
- R1 units of the two session rules (they look ahead with free constants; replaced by R2) and of TTM are void;
- units on instruments that fail the X10 data rule are not counted;
- streams with 2019-2025 t above 8 are flagged for examination.
Artifacts sit at inp/RUN/ARTIFACT/...; when the same file exists in two runs, the later run's copy is used.
Usage: python scripts/r24_combine.py --inp DIR --out DIR [--x10 FILE]
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

VOID_R1 = {"S.SES.SESSION_MOMENTUM", "S.SES.NY_OPEN_REVERSAL", "S.FAMOUS.TTM_SQUEEZE"}


def stats_row(part, family, symbol, sel, s, placebo=False, n=0):
    s = s.dropna()
    st = wf22.stream_stats(s if len(s) else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    return {"part": part, "family": family, "symbol": symbol, "selection": sel, "placebo": placebo, "n_cfg": n, **st,
            "own": wf22.counts_rule(st) if len(s) else False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--x10", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    unused = set()
    if a.x10 and os.path.exists(a.x10):
        unused = {k for k, v in json.load(open(a.x10)).items() if not v.get("used", False)}
    print("X10 instruments not used:", sorted(unused))

    def g(p):
        fs = sorted(glob.glob(os.path.join(a.inp, "**", p), recursive=True))
        best = {}
        for f in fs:
            rel = os.path.relpath(f, a.inp).split(os.sep)
            run, rest = (rel[0], os.sep.join(rel[1:])) if rel[0].isdigit() else ("0", os.sep.join(rel))
            if rest not in best or int(run) > int(best[rest][0]):
                best[rest] = (run, f)
        if len(best) < len(fs):
            print(p, "files replaced by a later run:", len(fs) - len(best))
        return sorted((f, rest.split(os.sep)[0]) for rest, (_, f) in best.items())

    def lw_part(art):
        return "R2" if art.startswith("r24LW-R2-") else "R1"

    parts = []
    for f, art in g("LW_rows_*.csv"):
        x = pd.read_csv(f)
        prt = lw_part(art)
        x["part"] = prt
        x["family"] = x["family"].str.replace("LW-", f"{prt}-", regex=False)
        sid = x["unit"].astype(str).str.split("|").str[0]
        x["counted"] = True
        if prt == "R1":
            x.loc[sid.isin(VOID_R1), "counted"] = False
            x["void"] = sid.isin(VOID_R1)
        sym = os.path.basename(f)[len("LW_rows_"):-4]
        if sym in unused:
            x["counted"] = False
        parts.append(x)
    for pat in ("IC_rows_*.csv", "R24_rows_*.csv"):
        for f, art in g(pat):
            if os.path.getsize(f) < 5:                                     # a part that did not run (R10)
                print("empty rows file (part not run):", f)
                continue
            x = pd.read_csv(f)
            if pat.startswith("IC_"):
                x["part"] = "R3"
            x["counted"] = True
            parts.append(x)
    R = pd.concat(parts, ignore_index=True)
    R["selection"] = R["selection"].fillna("default")
    R["placebo"] = R["placebo"].fillna(False).astype(bool)
    R["void"] = R.get("void", pd.Series(False, index=R.index)).fillna(False).astype(bool)
    # pooled LW streams per part (each strategy over instruments and timeframes)
    pooled, acc = [], {}
    for f, art in g("LW_pooled_*.parquet"):
        sym = os.path.basename(f)[len("LW_pooled_"):-8]
        if sym in unused:
            continue
        prt = lw_part(art)
        x = pd.read_parquet(f)
        for c in x.columns:
            key, kind, what = c.rsplit("|", 2)
            if prt == "R1" and key in VOID_R1:
                continue
            d = acc.setdefault((prt, key, kind), {"sum": None, "n": None})
            v = x[c].astype(np.float64)
            d[what] = v if d[what] is None else d[what].add(v, fill_value=0.0)
    for (prt, key, kind), d in acc.items():
        n = d["n"]
        s = (d["sum"] / n.where(n > 0)).dropna()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        pooled.append(stats_row(prt, f"{prt}-{key}", "POOLED", kind, s, key.startswith("PLACEBO"), int(n.max())))
    if pooled:
        P = pd.DataFrame(pooled)
        P["counted"] = True
        P["void"] = False
        R = pd.concat([R, P], ignore_index=True)
    R["counted"] = R["counted"].fillna(True).astype(bool)
    # Newey-West t of 2019-2025 (round 25 part HAC) for every stream whose series is stored (R3-R11; LW units keep
    # no series, their HAC t comes from the round-25 reruns)
    import wf25
    ser = {}
    for f, art in g("R24_series_*.parquet"):
        x = pd.read_parquet(f)
        for c in x.columns:
            ser[c] = x[c]
    for f, art in g("IC_series_*.parquet"):
        tag = os.path.basename(f)[len("IC_series_"):-8]
        x = pd.read_parquet(f)
        for c in x.columns:
            ser[f"{c}@{tag}"] = x[c]

    def hac(r):
        if r.get("part") == "R3":
            k = f"IC-SMT{'-PLACEBO' if r['placebo'] else ''}-{r['selection']}@{str(r['symbol']).replace('+', '_')}"
        else:
            k = f"{r['family']}|{r['symbol']}|{r['selection']}{'|placebo' if r['placebo'] else ''}"
        s_ = ser.get(k)
        if s_ is None:
            return np.nan
        s_ = s_.dropna()
        s_.index = pd.DatetimeIndex(pd.to_datetime(s_.index))
        b = s_[(s_.index >= wf22.SPLIT) & (s_.index <= wf22.DEV_END)]
        return wf25.nw_t(b.to_numpy()) if len(b) else np.nan
    R["p2_hac_t"] = [hac(r) for r in R.to_dict("records")]
    print("streams with a Newey-West t:", int(R.p2_hac_t.notna().sum()), "of", len(R), flush=True)
    real = R[~R.placebo & R.counted & ~R.void & R.p2_t.notna()].copy()
    real["p"] = 1 - norm.cdf(real.p2_t.astype(float))
    p = real.p.to_numpy()
    o = np.argsort(p)
    m = max(len(p), 1)
    thr = 0.10 * np.arange(1, len(p) + 1) / m
    ok = np.flatnonzero(p[o] <= thr)
    real["bh"] = False
    if len(ok):
        real.iloc[o[: ok.max() + 1], real.columns.get_loc("bh")] = True
    real["counts"] = real.own.astype(bool) & real.bh
    real["flag_t8"] = real.p2_t > 8
    R = R.merge(real[["p", "bh", "counts", "flag_t8"]], left_index=True, right_index=True, how="left")
    R.drop(columns=["choices"], errors="ignore").to_csv(os.path.join(a.out, "r24_streams.csv.gz"), index=False)
    keep = ["part", "family", "symbol", "unit", "selection", "oos_first", "p1_mean", "p2_mean", "p2_t", "p2_hac_t", "own", "bh"]
    by_part = []
    for prt, grp in R[R.p2_t.notna()].groupby("part"):
        rl = grp[~grp.placebo & grp.counted & ~grp.void]
        pl = grp[grp.placebo]
        by_part.append({"part": prt, "real_streams": int(len(rl)), "real_own": int(rl.own.sum()),
                        "real_own_share": float(rl.own.mean()) if len(rl) else None,
                        "real_max_t": float(rl.p2_t.max()) if len(rl) else None,
                        "placebo_streams": int(len(pl)), "placebo_own": int(pl.own.sum()) if len(pl) else 0,
                        "placebo_own_share": float(pl.own.mean()) if len(pl) else None,
                        "placebo_max_t": float(pl.p2_t.max()) if len(pl) else None,
                        "counts": int(real[real.part == prt].counts.sum()), "void": int(grp.void.sum())})
    summ = {"x10_not_used": sorted(unused), "streams": int(len(R)), "real_streams_tested": int(len(real)),
            "own_rule": int(real.own.sum()), "bh": int(real.bh.sum()), "counts": int(real.counts.sum()),
            "first_bh_threshold_t": float(norm.ppf(1 - 0.10 / m)), "by_part": by_part,
            "flag_t8": real[real.flag_t8][[c for c in keep if c in real]].round(6).to_dict("records")[:100],
            "top_real": real.sort_values("p2_t", ascending=False).head(60)[[c for c in keep if c in real]].round(6).to_dict("records"),
            "counting": real[real.counts][[c for c in keep if c in real]].round(6).to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "r24_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps({k: summ[k] for k in ("streams", "real_streams_tested", "own_rule", "bh", "counts", "first_bh_threshold_t")}, indent=1))
    print(pd.DataFrame(by_part).to_string())
    print(real.sort_values("p2_t", ascending=False)[[c for c in keep if c in real]].head(40).to_string())


if __name__ == "__main__":
    main()
