"""Aggregate the projection lab (scripts/projection_lab.py) over all instruments and timeframes.

Every row is one projection on one instrument x timeframe. It is NON-RANDOM (statistically) when the
out-of-sample correlation of its forecast with the next move is significant after Benjamini-Hochberg across
ALL rows (two-sided p from t = ic x sqrt(n)) AND it beats its sign-randomised surrogates (z_sur > 3). It is
TRADABLE when, in addition, the gross edge per bar exceeds the round-trip cost and the cost-aware rule has a
positive net Sharpe. Writes results/reports/projection_lab_summary/{rows.csv, by_tf_projection.csv,
by_tf.csv, summary.json}."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg  # noqa: E402

IN = RESULTS_DIR / "reports" / "projection_lab"
OUT = RESULTS_DIR / "reports" / "projection_lab_summary"
TF_ORDER = ["M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1", "TICK60", "VOLA60"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(IN / "*.json"))):
        rows += json.load(open(f)).get("rows", [])
    if not rows:
        json.dump({"rows": 0}, open(OUT / "summary.json", "w"))
        print("no projection rows")
        return
    df = pd.DataFrame(rows)
    df["p"] = 2 * st.norm.sf(np.abs(df["t"].fillna(0)))
    df["p_bh"] = benjamini_hochberg(df["p"].to_numpy())
    df["non_random"] = (df["p_bh"] < 0.05) & (df["z_sur"] > 3)
    df["edge_over_cost"] = df["edge_bps"].abs() / df["cost_rt_bps"]
    df["tradable"] = df["non_random"] & (df["edge_bps"] > df["cost_rt_bps"]) & (df["net_sharpe"] > 0)
    df["tf_order"] = df["tf"].map({t: i for i, t in enumerate(TF_ORDER)})
    df.sort_values(["tf_order", "projection", "pair"]).drop(columns="tf_order").to_csv(OUT / "rows.csv", index=False)
    agg = dict(pairs=("pair", "nunique"), median_ic=("ic", "median"), median_t=("t", "median"),
               median_hit=("hit", "median"), median_info_bits=("info_bits", "median"),
               share_non_random=("non_random", "mean"), median_edge_bps=("edge_bps", "median"),
               median_cost_bps=("cost_rt_bps", "median"), median_edge_over_cost=("edge_over_cost", "median"),
               median_gross_sharpe=("gross_sharpe", "median"), median_net_sharpe=("net_sharpe", "median"),
               tradable=("tradable", "sum"))
    g = df.groupby(["tf", "projection"]).agg(**agg).reset_index()
    g["tf_order"] = g["tf"].map({t: i for i, t in enumerate(TF_ORDER)})
    g.sort_values(["tf_order", "median_t"], ascending=[True, False]).drop(columns="tf_order").to_csv(
        OUT / "by_tf_projection.csv", index=False)
    t = df.groupby("tf").agg(**agg).reset_index()
    t["tf_order"] = t["tf"].map({x: i for i, x in enumerate(TF_ORDER)})
    t = t.sort_values("tf_order").drop(columns="tf_order")
    t.to_csv(OUT / "by_tf.csv", index=False)
    nr = df[df.non_random].sort_values("t", key=np.abs, ascending=False)
    summ = {"rows": int(len(df)), "pairs": int(df.pair.nunique()), "tfs": [x for x in TF_ORDER if x in set(df.tf)],
            "projections": sorted(df.projection.unique().tolist()),
            "non_random_rows": int(df.non_random.sum()), "tradable_rows": int(df.tradable.sum()),
            "non_random_by_projection": df.groupby("projection").non_random.sum().astype(int).to_dict(),
            "non_random_by_tf": df.groupby("tf").non_random.sum().astype(int).to_dict(),
            "strongest_non_random": nr.head(15)[["pair", "tf", "projection", "ic", "t", "hit", "info_bits",
                                                  "z_sur", "edge_bps", "cost_rt_bps", "gross_sharpe",
                                                  "net_sharpe"]].to_dict("records"),
            "best_edge_over_cost": df.sort_values("edge_over_cost", ascending=False).head(10)[
                ["pair", "tf", "projection", "ic", "t", "edge_bps", "cost_rt_bps", "net_sharpe", "non_random"]].to_dict("records")}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(t.round(4).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k not in ("strongest_non_random", "best_edge_over_cost")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
