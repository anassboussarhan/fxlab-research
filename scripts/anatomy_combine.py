"""Round 12 combine step (docs/anatomy_preregistration.md): part A rows, the failure diagnosis of every unit, the
one fix per unit chosen on 2007-2018 and judged once on 2019-2025, Benjamini-Hochberg, placebo calibration.
Usage: python scripts/anatomy_combine.py --inp DIR_WITH_trades_*.parquet --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import queue_lab as Q                                                        # noqa: E402

FIT_END = pd.Timestamp("2018-12-31 23:59", tz="UTC")
TERC = ["f_atr", "f_tstr", "f_ext", "f_round", "f_spread"]
CATS = {"f_with": [-1.0, 1.0], "f_sess": [0, 1, 2, 3], "f_wday": [0, 1, 2], "f_cal": [0.0, 1.0]}
VARS = [f"n_{e}{x}" for e in ("E0", "E1", "E2", "E3") for x in ("X05", "X1", "X2")] + ["n_FLIP"]


def kind_of(u):
    n = u.split("|")[0]
    return "placebo" if n.startswith("P") and "_" in n and n[1].isdigit() else ("new" if not n.startswith("S.") else "catalog")


def tstat(x):
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * math.sqrt(len(x)))


def s_month(vals, times):
    s = pd.Series(vals, index=pd.DatetimeIndex(times)).dropna()
    return Q.summ(s)


def bh(p, q=0.10):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    passed = np.zeros(m, bool)
    thr = q * (np.arange(1, m + 1)) / m
    ok = p[order] <= thr
    if ok.any():
        k = np.flatnonzero(ok).max()
        passed[order[:k + 1]] = True
    return passed


def masks(fitdf, alldf):
    """Filter masks over alldf rows with cut points from fitdf: (name, mask)."""
    out = [("none", np.ones(len(alldf), bool))]
    for f in TERC:
        if f not in alldf:
            continue
        v = fitdf[f].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < 30:
            continue
        q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
        x = alldf[f].to_numpy(float)
        b = np.where(~np.isfinite(x), -1, np.where(x <= q1, 0, np.where(x <= q2, 1, 2)))
        for k in range(3):
            out.append((f"{f}=t{k}", b == k))
            out.append((f"{f}!=t{k}", (b != k) & (b >= 0)))
    for f, vals in CATS.items():
        if f not in alldf:
            continue
        x = alldf[f].to_numpy(float)
        for v in vals:
            out.append((f"{f}={v:g}", x == v))
            if len(vals) > 2:
                out.append((f"{f}!={v:g}", (x != v) & np.isfinite(x)))
    return out


def diagnose(fit):
    n = fit["n_E0X1"].to_numpy(float)
    g = fit["g_E0X1"].to_numpy(float)
    win = n[n > 0]
    loss = n[n <= 0]
    mfe = fit["mfe_c"].to_numpy(float)
    d = {"n_fit": int(len(n)), "net_bp": float(np.nanmean(n) * 1e4), "gross_bp": float(np.nanmean(g) * 1e4),
         "gross_t": tstat(g), "net_t": tstat(n), "cost_bp": float(np.nanmean(fit["cost"]) * 1e4),
         "win_rate": float((n > 0).mean()), "avg_win_bp": float(win.mean() * 1e4) if len(win) else np.nan,
         "avg_loss_bp": float(loss.mean() * 1e4) if len(loss) else np.nan,
         "mfe_c_median": float(np.nanmedian(mfe)), "mae_c_median": float(np.nanmedian(fit["mae_c"])),
         "exit_problem": float(np.mean(mfe[n <= 0] > 1)) if len(loss) else np.nan,
         "entry_problem": float(np.mean(mfe < 1)), "hold_h_median": float(np.nanmedian(fit["hold_min"]) / 60)}
    if not np.isfinite(d["gross_t"]) or d["gross_t"] < 1:
        lab = "no gross edge"
    elif d["gross_bp"] > 0 and d["net_bp"] <= 0:
        lab = "cost-bound"
    else:
        lab = "weak"
    if d["win_rate"] >= 0.6 and np.isfinite(d["avg_loss_bp"]) and np.isfinite(d["avg_win_bp"]) and \
            -d["avg_loss_bp"] >= 2 * d["avg_win_bp"]:
        lab = "skew trap" if lab != "cost-bound" else lab
    d["label"] = lab
    return d


def search(U):
    fit, chk = U[U.t0 <= FIT_END], U[U.t0 > FIT_END]
    nb = int(fit["n_E0X1"].notna().sum())
    if nb < 30 or len(chk) < 10:
        return None
    min_n = max(30, nb // 4)
    M = masks(fit, U)
    isfit = (U.t0 <= FIT_END).to_numpy()
    best = None
    for v in VARS:
        if v not in U:
            continue
        x = U[v].to_numpy(float)
        ok = np.isfinite(x)
        for name, m in M:
            sel = ok & m & isfit
            k = int(sel.sum())
            if k < min_n:
                continue
            xs = x[sel]
            sd = xs.std()
            if sd <= 0:
                continue
            t = xs.mean() / sd * math.sqrt(k)
            if best is None or t > best[0]:
                best = (t, v, name, k)
    if best is None:
        return None
    t, v, name, k = best
    m = dict(M)[name]
    x = U[v].to_numpy(float)
    sel_c = np.isfinite(x) & m & ~isfit
    sel_f = np.isfinite(x) & m & isfit
    fs = s_month(x[sel_f], U.t0[sel_f].dt.tz_convert("UTC").dt.tz_localize(None))
    cs = s_month(x[sel_c], U.t0[sel_c].dt.tz_convert("UTC").dt.tz_localize(None))
    return {"fix_variant": v, "fix_filter": name, "fix_fit_trade_t": float(t), "fix_fit_n": k,
            "fix_fit_bp": fs.get("mean_bp"), "fix_fit_month_t": fs.get("t"),
            "fix_check_n": cs.get("n", 0), "fix_check_bp": cs.get("mean_bp"), "fix_check_t": cs.get("t"),
            "fix_check_by_year": cs.get("by_year_bp")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.inp, "**", "trades_*.parquet"), recursive=True))
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    D["t0"] = pd.to_datetime(D["t0"], utc=True)
    for c in D.columns:
        if D[c].dtype == np.float32:
            D[c] = D[c].astype(float)
    print("trades", len(D), "units", D.unit.nunique(), "files", len(files), flush=True)
    try:
        import fxlab.strategies.daily, fxlab.strategies.extra, fxlab.strategies.famous, fxlab.strategies.rules  # noqa
        from fxlab.registry import CATALOG
        fam = {k: v.family for k, v in CATALOG.items()}
    except Exception:                                                    # noqa: BLE001
        fam = {}
    rows = []
    for u, U in D.groupby("unit"):
        U = U.sort_values("t0")
        k = kind_of(u)
        fit, chk = U[U.t0 <= FIT_END], U[U.t0 > FIT_END]
        r = {"unit": u, "kind": k, "family": fam.get(u.split("|")[0], "classic" if k == "new" else k),
             "symbols": int(U.sym.nunique())}
        if len(fit) >= 10:
            r.update(diagnose(fit))
        base_f = s_month(fit["n_E0X1"].to_numpy(float), fit.t0.dt.tz_convert("UTC").dt.tz_localize(None))
        base_c = s_month(chk["n_E0X1"].to_numpy(float), chk.t0.dt.tz_convert("UTC").dt.tz_localize(None))
        r.update({"base_fit_bp": base_f.get("mean_bp"), "base_fit_t": base_f.get("t"),
                  "base_check_n": base_c.get("n", 0), "base_check_bp": base_c.get("mean_bp"), "base_check_t": base_c.get("t"),
                  "base_check_gross_bp": float(chk["g_E0X1"].mean() * 1e4) if len(chk) else np.nan})
        if r.get("net_bp") is not None and r.get("base_fit_bp") is not None and np.isfinite(r["base_fit_bp"] or np.nan):
            if r["base_fit_bp"] > 0 and (r["base_check_bp"] is None or r["base_check_bp"] <= 0):
                r["label"] = "decay"
        s = search(U)
        if s:
            r.update(s)
        rows.append(r)
        print(u, r.get("label"), r.get("base_check_bp"), r.get("fix_variant"), r.get("fix_filter"), r.get("fix_check_bp"),
              r.get("fix_check_t"), flush=True)
    R = pd.DataFrame(rows)

    def p1(t):
        return float(1 - norm.cdf(t)) if t is not None and np.isfinite(t) else 1.0
    # part A: the nine families as published
    A = R[R.kind == "new"].copy()
    A["p"] = A.base_check_t.map(p1)
    A["bh"] = bh(A.p.values)
    A["counts"] = A.bh & (A.base_check_bp > 0) & (A.base_check_t >= 1.645) & (A.base_fit_bp > 0)
    # part B: the fixes
    B = R[(R.kind != "placebo") & R.fix_check_t.notna()].copy()
    B["p"] = B.fix_check_t.map(p1)
    B["bh"] = bh(B.p.values)
    B["counts"] = B.bh & (B.fix_check_bp > 0) & (B.fix_check_t >= 1.645)
    P = R[(R.kind == "placebo") & R.fix_check_t.notna()]
    # per-instrument placebo calibration: the same selection on each placebo x instrument
    pc = []
    for u, U in D[D.unit.map(kind_of) == "placebo"].groupby("unit"):
        for sym, V in U.groupby("sym"):
            s = search(V.sort_values("t0"))
            if s and s.get("fix_check_t") is not None and np.isfinite(s["fix_check_t"]):
                pc.append({"unit": u, "sym": sym, **s})
    PC = pd.DataFrame(pc)
    PC.to_csv(os.path.join(a.out, "anatomy_placebo_by_symbol.csv"), index=False)
    summ = {"units": int(len(R)), "part_A_rows": int(len(A)), "part_A_counts": int(A.counts.sum()),
            "part_B_units": int(len(B)), "part_B_counts": int(B.counts.sum()),
            "part_B_check_t_ge_1645": int((B.fix_check_t >= 1.645).sum()),
            "part_B_share_t_ge_1645": float((B.fix_check_t >= 1.645).mean()) if len(B) else None,
            "placebo_units": int(len(P)), "placebo_t_ge_1645": int((P.fix_check_t >= 1.645).sum()),
            "placebo_by_symbol_runs": int(len(PC)),
            "placebo_by_symbol_t_ge_1645": int((PC.fix_check_t >= 1.645).sum()) if len(PC) else 0,
            "placebo_by_symbol_share": float((PC.fix_check_t >= 1.645).mean()) if len(PC) else None,
            "fix_fit_t_median": float(B.fix_fit_trade_t.median()) if len(B) else None,
            "fix_check_t_median": float(B.fix_check_t.median()) if len(B) else None,
            "labels": R[R.kind != "placebo"].label.value_counts().to_dict(),
            "fix_variants": B.fix_variant.value_counts().to_dict(),
            "fix_filters": B.fix_filter.str.split("[=!]", regex=True).str[0].value_counts().to_dict()}
    R.to_csv(os.path.join(a.out, "anatomy_units.csv"), index=False)
    A.to_csv(os.path.join(a.out, "anatomy_partA.csv"), index=False)
    B.to_csv(os.path.join(a.out, "anatomy_partB.csv"), index=False)
    json.dump(summ, open(os.path.join(a.out, "anatomy_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(A[["unit", "n_fit", "base_fit_bp", "base_check_n", "base_check_bp", "base_check_t", "gross_bp", "label", "counts"]].to_string())
    print(B.sort_values("p").head(25)[["unit", "label", "base_check_bp", "fix_variant", "fix_filter", "fix_fit_bp",
                                       "fix_check_n", "fix_check_bp", "fix_check_t", "bh", "counts"]].to_string())


if __name__ == "__main__":
    main()
