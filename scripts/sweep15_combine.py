"""Round 15 combine (docs/round15_preregistration.md, parts A-C): portfolios (FX primary, non-FX reported), selection
of each unit's configuration on 2008-2018, one check on 2019-2025, the strict and regime tracks with
Benjamini-Hochberg over all real units of parts A-C, and the placebo pass rates.
Usage: python scripts/sweep15_combine.py --inp DIR --r13 A_monthly_by_symbol.csv --out DIR"""
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
import long_lab as LL                                                       # noqa: E402

FIT = ("2008-01", "2018-12")
CHK = ("2019-01", "2025-12")
MONTHS = pd.period_range(FIT[0], CHK[1], freq="M").astype(str)


def nw_t(x, lags=3):
    x = np.asarray(x, float)
    n = len(x)
    if n < 12 or not np.isfinite(x).all():
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else np.nan


def bh(p, q=0.10):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok])]
    thr = q * np.arange(1, len(ok) + 1) / len(ok)
    passed = np.flatnonzero(p[o] <= thr)
    if len(passed):
        out[o[: passed.max() + 1]] = True
    return out


def stats_block(P: pd.DataFrame, prefix: str):
    """P: rows (unit, cfg, dir) x columns months (pnl). Returns per-row fit / check mean and t."""
    X = P.reindex(columns=MONTHS).fillna(0.0).to_numpy(float)
    fm = np.array([m <= FIT[1] for m in MONTHS])
    cm = ~fm
    res = pd.DataFrame(index=P.index)
    res[f"{prefix}fit_mean"] = X[:, fm].mean(1)
    res[f"{prefix}check_mean"] = X[:, cm].mean(1)
    res[f"{prefix}fit_t"] = [nw_t(r) for r in X[:, fm]]
    res[f"{prefix}check_t"] = [nw_t(r) for r in X[:, cm]]
    res[f"{prefix}fit_sd"] = X[:, fm].std(1)
    return res


def accumulate(inp, syms):
    """Sum monthly pnl / gpnl / n over the given instruments, one file at a time."""
    acc = {"pnl": None, "gpnl": None, "n": None}
    nfiles = 0
    for pat in ("SAB_*.parquet", "SC_*.parquet"):
        for f in sorted(glob.glob(os.path.join(inp, "**", pat), recursive=True)):
            sym = os.path.basename(f).split("_", 1)[1][:-8]
            if sym not in syms:
                continue
            d = pd.read_parquet(f)
            d = d[d.month.astype(str) >= FIT[0]]
            for c in ("unit", "cfg"):
                d[c] = d[c].astype(str)
            d["month"] = d["month"].astype(str)
            for col in ("pnl", "gpnl", "n"):
                pv = d.pivot_table(index=["unit", "cfg", "dir"], columns="month", values=col, aggfunc="sum").astype(float)
                acc[col] = pv if acc[col] is None else acc[col].add(pv, fill_value=0.0)
            nfiles += 1
            print("read", f, len(d), flush=True)
            del d
    return acc, nfiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--r13", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fx, nonfx = LL.FX_SYMBOLS, list(LL.NONFX)
    F, nf_fx = accumulate(a.inp, set(fx))
    Nf, nf_nf = accumulate(a.inp, set(nonfx))
    print("files FX", nf_fx, "non-FX", nf_nf, flush=True)
    Pf = F["pnl"]
    Pg = F["gpnl"]
    Pn = F["n"].reindex(columns=MONTHS).fillna(0)
    S = stats_block(Pf, "")
    G = stats_block(Pg, "gross_")
    S["check_gross_t"] = G["gross_check_t"]
    fit_cols = [m for m in MONTHS if m <= FIT[1]]
    S["fit_trades"] = Pn[fit_cols].sum(1)
    S["fit_active_months"] = (Pn[fit_cols] > 0).sum(1)
    S = S.reset_index()
    S["is_feature"] = S.unit.astype(str).str.startswith(("FEAT:", "PLACEBOF_"))
    S["is_placebo"] = S.unit.astype(str).str.startswith(("PLACEBO_", "PLACEBOF_"))
    S["tf"] = S.unit.astype(str).str.split("|").str[-1]
    # non-FX block for the selected rows, and the long-only benchmark for the excess
    Pnf = Nf["pnl"]
    S.to_csv(os.path.join(a.out, "all_configs_fx.csv.gz"), index=False)
    # selection on the fit period
    elig = np.where(S.is_feature, S.fit_active_months >= 24, S.fit_trades >= 30) & np.isfinite(S.fit_t)
    E = S[elig]
    sel = E.loc[E.groupby("unit", observed=True).fit_t.idxmax()].copy()
    sel["p"] = 1 - norm.cdf(sel.check_t.fillna(-9))
    real = ~sel.is_placebo
    sel["bh_pass"] = False
    sel.loc[real, "bh_pass"] = bh(sel.loc[real, "p"].to_numpy())
    sel["regime_rule"] = (sel.check_mean > 0) & (sel.check_t >= 1.645)
    sel["strict_rule"] = sel.regime_rule & (sel.fit_mean > 0)
    # placebo calibration per track
    pl = sel[sel.is_placebo]
    rates = {}
    for tr in ("strict_rule", "regime_rule"):
        rates[tr] = {"real": float(sel.loc[real, tr].mean()), "placebo": float(pl[tr].mean()) if len(pl) else None,
                     "real_rules": float(sel.loc[real & ~sel.is_feature, tr].mean()),
                     "placebo_rules": float(pl.loc[~pl.is_feature, tr].mean()) if len(pl) else None,
                     "real_features": float(sel.loc[real & sel.is_feature, tr].mean()),
                     "placebo_features": float(pl.loc[pl.is_feature, tr].mean()) if len(pl) else None}
    sel["counts_strict"] = real & sel.strict_rule & sel.bh_pass
    sel["regime_candidate"] = real & sel.regime_rule & sel.bh_pass
    # non-FX view of the selected configuration (information)
    if Pnf is not None:
        key = list(zip(sel.unit, sel.cfg, sel.dir))
        idx = Pnf.index
        nf = Pnf.reindex(pd.MultiIndex.from_tuples(key, names=idx.names))
        st = stats_block(nf, "nonfx_")
        sel["nonfx_check_t"] = st["nonfx_check_t"].to_numpy()
        sel["nonfx_fit_t"] = st["nonfx_fit_t"].to_numpy()
    sel.to_csv(os.path.join(a.out, "selected.csv"), index=False)
    # whole-surface summaries (information)
    surf = S[~S.is_placebo].groupby(["is_feature", "tf"]).agg(configs=("unit", "size"), median_check_t=("check_t", "median"),
                                                             share_check_t_ge_1645=("check_t", lambda s: float((s >= 1.645).mean())))
    surf.to_csv(os.path.join(a.out, "surface_summary.csv"))
    plsurf = S[S.is_placebo].groupby(["is_feature", "tf"]).agg(configs=("unit", "size"), median_check_t=("check_t", "median"),
                                                              share_check_t_ge_1645=("check_t", lambda s: float((s >= 1.645).mean())))
    plsurf.to_csv(os.path.join(a.out, "surface_summary_placebo.csv"))
    summ = {"units_real": int(real.sum()), "units_placebo": int((~real).sum()),
            "counts_strict": int(sel.counts_strict.sum()), "regime_candidates": int(sel.regime_candidate.sum()),
            "rates": rates,
            "strict_list": sel[sel.counts_strict][["unit", "cfg", "dir", "fit_t", "check_t", "check_gross_t"]].round(3).to_dict("records"),
            "regime_list": sel[sel.regime_candidate][["unit", "cfg", "dir", "fit_t", "check_t", "check_gross_t"]].round(3).to_dict("records"),
            "top_check": sel[real].nlargest(25, "check_t")[["unit", "cfg", "dir", "fit_t", "check_t", "p", "bh_pass"]].round(4).to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "summary15.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
