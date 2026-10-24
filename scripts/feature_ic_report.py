"""Aggregate the feature predictability lab (scripts/feature_ic_lab.py) over all instruments, timeframes and horizons.

A test = one feature column x instrument x timeframe x horizon. One-sided p-value from z (the sign was learned on
training data, so information shows up as z > 0); Benjamini-Hochberg across ALL tests. Per feature (all its output
columns, raw and |d20): number of tests, share significant after BH (q < 0.05), share with z > 3 (about 0.13%
expected by chance), median out-of-sample IC, best test, and the economic size of the best significant test
(decile spread vs round-trip cost). Writes results/reports/feature_ic_summary/{features.csv, by_tf.csv,
by_category.csv, summary.json}."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg  # noqa: E402

IN = Path(os.environ.get("FR_IN", RESULTS_DIR / "reports" / "feature_ic"))
OUT = Path(os.environ.get("FR_OUT", RESULTS_DIR / "reports" / "feature_ic_summary"))
TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1", "TICK60", "VOLA60"]
CAT = {"PR": "Price/returns", "TR": "Trend", "MO": "Momentum/oscillators", "VO": "Volatility", "DI": "Distribution",
       "MS": "Microstructure/spread/ticks", "ST": "Structure/levels", "SP": "Spectral/cycles", "FR": "Fractal/chaos",
       "IT": "Information theory", "EC": "Econometrics", "TS": "Time/calendar", "RG": "Regime", "XA": "Cross-asset",
       "ES": "Esoteric", "LF": "Learned representations", "FG": "Fibonacci/geometry", "CA": "Cross-asset/currency",
       "IN": "Interactions"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(IN / "*.csv.gz")))
    if not files:
        json.dump({"tests": 0}, open(OUT / "summary.json", "w"))
        print("no feature IC files")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[np.isfinite(df.z)]
    df["p"] = st.norm.sf(df.z)
    df["p_bh"] = benjamini_hochberg(df.p.to_numpy())
    df["sig"] = df.p_bh < 0.05
    df["edge_over_cost"] = (df.spread_bps / 2) / df.cost_rt_bps        # one leg of the decile spread vs a round trip
    try:
        import fxlab.features  # noqa: F401
        from fxlab.registry import by_kind
        names = {e.id: e.name for e in by_kind("feature")}
        spec = {e.id: ("speculative" in (e.tags or ())) for e in by_kind("feature")}
    except Exception:
        names, spec = {}, {}
    rows = []
    for fid, g in df.groupby("feature"):
        sig = g[g.sig]
        b = g.loc[g.z.idxmax()]
        bs = sig.loc[sig.edge_over_cost.idxmax()] if len(sig) else None
        by_tf = {tf: float(x.sig.mean()) for tf, x in g.groupby("tf")}
        rows.append({
            "feature": fid, "name": names.get(fid, ""), "category": CAT.get(fid.split(".")[1], fid.split(".")[1]),
            "speculative": spec.get(fid, False), "tests": len(g), "share_significant_bh": float(g.sig.mean()),
            "share_z_gt_3": float((g.z > 3).mean()), "median_ic": float(g.ic_oos.median()),
            "median_z": float(g.z.median()),
            **{f"sig_{tf}": by_tf.get(tf, np.nan) for tf in TF_ORDER if tf in set(df.tf)},
            "best_z": float(b.z), "best_test": f"{b.pair} {b.tf} h={int(b.h)} {b.column.split(':', 1)[-1]}",
            "best_ic": float(b.ic_oos),
            "best_sig_edge_over_cost": float(bs.edge_over_cost) if bs is not None else np.nan,
            "best_sig_edge_test": (f"{bs.pair} {bs.tf} h={int(bs.h)} {bs.column.split(':', 1)[-1]}: "
                                   f"{bs.spread_bps / 2:.2f}bp vs {bs.cost_rt_bps:.2f}bp") if bs is not None else ""})
    F = pd.DataFrame(rows)

    def verdict(x):
        if x.share_significant_bh >= 0.05:
            return "predictive (significant in >=5% of its tests)"
        if x.share_significant_bh >= 0.01:
            return "weakly predictive (1-5% of tests)"
        if x.share_significant_bh > 0:
            return "rarely predictive (<1% of tests)"
        return "no detectable predictive information"
    F["verdict"] = F.apply(verdict, axis=1)
    F["covers_costs"] = np.where(F.best_sig_edge_over_cost >= 1, "in its best significant test", "no")
    F = F.sort_values(["share_significant_bh", "median_z"], ascending=False).reset_index(drop=True)
    F.insert(0, "rank", range(1, len(F) + 1))
    F.to_csv(OUT / "features.csv", index=False)
    T = df.groupby("tf").agg(tests=("z", "size"), share_sig=("sig", "mean"), share_z3=("z", lambda z: float((z > 3).mean())),
                             median_ic=("ic_oos", "median"), best_ic=("ic_oos", "max"),
                             median_cost=("cost_rt_bps", "median"),
                             sig_median_edge_over_cost=("edge_over_cost", lambda x: float(x[df.loc[x.index, "sig"]].median())
                                                        if df.loc[x.index, "sig"].any() else np.nan),
                             sig_covering_cost=("edge_over_cost", lambda x: int((x[df.loc[x.index, "sig"]] >= 1).sum()))).reset_index()
    T["o"] = T.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    T = T.sort_values("o").drop(columns="o")
    T.to_csv(OUT / "by_tf.csv", index=False)
    C = F.groupby("category").agg(features=("feature", "size"), predictive=("verdict", lambda v: int(v.str.startswith("predictive").sum())),
                                  weak=("verdict", lambda v: int(v.str.startswith("weakly").sum())),
                                  median_share_sig=("share_significant_bh", "median")).sort_values("median_share_sig", ascending=False)
    C.to_csv(OUT / "by_category.csv")
    summ = {"tests": int(len(df)), "features": int(F.feature.nunique()), "pairs": int(df.pair.nunique()),
            "tfs": [t for t in TF_ORDER if t in set(df.tf)], "horizons": sorted(df.h.unique().tolist()),
            "significant_tests": int(df.sig.sum()), "share_significant": float(df.sig.mean()),
            "significant_covering_cost": int((df.sig & (df.edge_over_cost >= 1)).sum()),
            "verdicts": F.verdict.value_counts().to_dict()}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(T.round(4).to_string())
    print(C.to_string())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
