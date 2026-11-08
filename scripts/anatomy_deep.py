"""Round 12 deeper look across all units (information only; nothing here is a test that can count):
does a filter, an entry method, an exit or reversing that helps in 2007-2018 also help in 2019-2025, on average
over the strategies? Usage: python scripts/anatomy_deep.py --inp DIR --units anatomy_units.csv --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anatomy_combine as AC                                                 # noqa: E402

FIT_END = AC.FIT_END


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.inp, "**", "trades_*.parquet"), recursive=True))],
                  ignore_index=True)
    D["t0"] = pd.to_datetime(D["t0"], utc=True)
    for c in D.columns:
        if D[c].dtype == np.float32:
            D[c] = D[c].astype(float)
    D = D[D.unit.map(AC.kind_of) != "placebo"]
    U = pd.read_csv(a.units).set_index("unit")
    feat_rows, var_rows = [], []
    for u, G in D.groupby("unit"):
        fit, chk = G[G.t0 <= FIT_END], G[G.t0 > FIT_END]
        if len(fit) < 100 or len(chk) < 50:
            continue
        base_f, base_c = fit["n_E0X1"].mean(), chk["n_E0X1"].mean()
        # (a) every filter: effect in fit vs effect in check
        Mf = dict(AC.masks(fit, G))
        isfit = (G.t0 <= FIT_END).to_numpy()
        x = G["n_E0X1"].to_numpy(float)
        for name, m in Mf.items():
            if name == "none":
                continue
            mf, mc = m & isfit & np.isfinite(x), m & ~isfit & np.isfinite(x)
            if mf.sum() < 30 or mc.sum() < 15:
                continue
            feat_rows.append({"unit": u, "filter": name, "feature": name.split("=")[0].rstrip("!"),
                              "fit_effect_bp": (x[mf].mean() - base_f) * 1e4, "check_effect_bp": (x[mc].mean() - base_c) * 1e4,
                              "fit_n": int(mf.sum()), "check_n": int(mc.sum())})
        # (b) every entry / exit variant and the reversal: change against the base in fit vs check
        for v in AC.VARS:
            if v == "n_E0X1" or v not in G:
                continue
            ff, cc = fit[[v, "n_E0X1"]].dropna(), chk[[v, "n_E0X1"]].dropna()
            if len(ff) < 30 or len(cc) < 15:
                continue
            var_rows.append({"unit": u, "variant": v, "fit_delta_bp": (ff[v] - ff["n_E0X1"]).mean() * 1e4,
                             "check_delta_bp": (cc[v] - cc["n_E0X1"]).mean() * 1e4,
                             "fit_level_bp": ff[v].mean() * 1e4, "check_level_bp": cc[v].mean() * 1e4,
                             "fill_share_fit": float(fit[v].notna().mean())})
    Fr, Vr = pd.DataFrame(feat_rows), pd.DataFrame(var_rows)
    Fr.to_csv(os.path.join(a.out, "deep_filters.csv"), index=False)
    Vr.to_csv(os.path.join(a.out, "deep_variants.csv"), index=False)
    out = {}
    # does a filter's fit-period effect predict its check-period effect?
    fs = {}
    for f, g in Fr.groupby("feature"):
        fs[f] = {"pairs": int(len(g)), "corr_fit_check": float(g.fit_effect_bp.corr(g.check_effect_bp)),
                 "same_sign_share": float((np.sign(g.fit_effect_bp) == np.sign(g.check_effect_bp)).mean()),
                 "top_decile_fit_mean_check_bp": float(g[g.fit_effect_bp >= g.fit_effect_bp.quantile(0.9)].check_effect_bp.mean())}
    out["filters"] = fs
    out["filters_all"] = {"pairs": int(len(Fr)), "corr_fit_check": float(Fr.fit_effect_bp.corr(Fr.check_effect_bp)),
                          "same_sign_share": float((np.sign(Fr.fit_effect_bp) == np.sign(Fr.check_effect_bp)).mean())}
    vs = {}
    for v, g in Vr.groupby("variant"):
        vs[v] = {"units": int(len(g)), "median_fit_delta_bp": float(g.fit_delta_bp.median()),
                 "median_check_delta_bp": float(g.check_delta_bp.median()),
                 "share_better_fit": float((g.fit_delta_bp > 0).mean()), "share_better_check": float((g.check_delta_bp > 0).mean()),
                 "corr_fit_check": float(g.fit_delta_bp.corr(g.check_delta_bp)),
                 "units_positive_level_both": int(((g.fit_level_bp > 0) & (g.check_level_bp > 0)).sum())}
    out["variants"] = vs
    # cost-bound strategies: how far from paying
    cb = U[(U.label == "cost-bound") & (U.kind != "placebo")]
    out["cost_bound"] = [{"unit": i, "gross_fit_bp": r.gross_bp, "cost_bp": r.cost_bp, "net_fit_bp": r.net_bp,
                          "gross_check_bp": r.base_check_gross_bp, "net_check_bp": r.base_check_bp,
                          "cost_cut_needed_pct": float(100 * (1 - r.gross_bp / r.cost_bp)) if r.cost_bp > 0 else None}
                         for i, r in cb.iterrows()]
    # by family
    fam = U[U.kind != "placebo"].groupby("family").agg(units=("label", "size"),
                                                         no_gross=("label", lambda s: int((s == "no gross edge").sum())),
                                                         cost_bound=("label", lambda s: int((s == "cost-bound").sum())),
                                                         decay=("label", lambda s: int((s == "decay").sum())),
                                                         skew=("label", lambda s: int((s == "skew trap").sum())),
                                                         gross_fit_bp=("gross_bp", "median"), cost_bp=("cost_bp", "median"),
                                                         win_rate=("win_rate", "median"), mfe_c=("mfe_c_median", "median"),
                                                         exit_problem=("exit_problem", "median"), entry_problem=("entry_problem", "median"),
                                                         base_check_bp=("base_check_bp", "median"), fix_check_t=("fix_check_t", "median"))
    fam.to_csv(os.path.join(a.out, "deep_families.csv"))
    json.dump(out, open(os.path.join(a.out, "deep_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:6000])
    pd.set_option("display.width", 250)
    print(fam.round(2).to_string())


if __name__ == "__main__":
    main()
