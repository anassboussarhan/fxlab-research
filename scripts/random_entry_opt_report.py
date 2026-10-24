"""Aggregate scripts/random_entry_opt.py: which management strategy and timeframe is best for RANDOM entries?

Per instrument x timeframe x family the walk-forward results (parameters optimised on training data with random
entries, tested on later data with new random entries), the hindsight check (optimum on the whole sample re-run
with fresh entries) and gross vs net. Ranking by the median, across instruments, of the single-trader
out-of-sample Sharpe. Significance: one-sided t of the expected-value stream (mean over the test seeds),
Benjamini-Hochberg over all instrument x timeframe x family cells. Portfolios: equal weight across instruments.
Writes results/reports/random_entry_opt_summary/{cells.csv, ranking.csv, by_tf.csv, portfolios.csv, params.csv,
summary.json}."""
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

IN = Path(os.environ.get("RO_IN", RESULTS_DIR / "reports" / "random_entry_opt"))
OUT = Path(os.environ.get("RO_OUT", RESULTS_DIR / "reports" / "random_entry_opt_summary"))
TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1", "TICK60", "VOLA60"]


def sharpe_t(d):
    d = pd.Series(d).dropna()
    if len(d) < 20 or d.std() == 0:
        return 0.0, 0.0
    return float(d.mean() / d.std() * np.sqrt(260)), float(d.mean() / d.std() * np.sqrt(len(d)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, prm, streams = [], [], {}
    for f in sorted(glob.glob(str(IN / "*.json"))):
        J = json.load(open(f))
        dp = f[:-5] + "_daily.parquet"
        Dd = pd.read_parquet(dp) if os.path.exists(dp) else pd.DataFrame()
        for fam, R in J.get("families", {}).items():
            w, h = R["wf"], R["hindsight"]
            rows.append({"pair": J["pair"], "tf": J["tf"], "family": fam,
                         "single_trader_median": w["single_trader_median_sharpe"],
                         "single_trader_share_positive": float(np.mean(np.array(w["single_trader_sharpes"]) > 0)),
                         "single_trader_gross": w["single_trader_median_gross"],
                         "expected_stream_sharpe": w["expected_stream_sharpe"], "t": w["expected_stream_t"],
                         "expected_stream_gross": w["expected_stream_gross_sharpe"], "days": w["days"],
                         "ann_return": w["expected_stream_ann_return"], "train_median": w["median_train_sharpe"],
                         "hindsight_in_sample": h["in_sample_best"], "hindsight_rerandomised": h["rerandomised"]})
            for s in R["selection"]:
                prm.append({"tf": J["tf"], "family": fam, **{k: v for k, v in s["params"].items()}})
            if fam in Dd:
                streams[(J["pair"], J["tf"], fam)] = Dd[fam].dropna().reset_index(drop=True)
    if not rows:
        json.dump({"cells": 0}, open(OUT / "summary.json", "w"))
        print("no results")
        return
    C = pd.DataFrame(rows)
    C["p"] = st.t.sf(C.t, df=np.maximum(C.days - 1, 1))
    C["p_bh"] = benjamini_hochberg(C.p.to_numpy())
    C["o"] = C.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    C.sort_values(["o", "family", "pair"]).drop(columns="o").to_csv(OUT / "cells.csv", index=False)
    agg = dict(pairs=("pair", "nunique"), median_single_trader=("single_trader_median", "median"),
               share_pairs_positive=("single_trader_median", lambda x: float((x > 0).mean())),
               median_gross=("single_trader_gross", "median"), median_expected_stream=("expected_stream_sharpe", "median"),
               median_t=("t", "median"), bh_survivors=("p_bh", lambda p: int((p < 0.10).sum())),
               median_train=("train_median", "median"), median_hindsight_in_sample=("hindsight_in_sample", "median"),
               median_hindsight_rerandomised=("hindsight_rerandomised", "median"))
    R = C.groupby(["tf", "family"]).agg(**agg).reset_index()
    # equal-weight portfolio across instruments of the expected-value streams (aligned by position in the OOS span;
    # streams of one timeframe share the same calendar, so this is a reasonable diversification estimate)
    port = []
    for (tf, fam), g in R.groupby(["tf", "family"]):
        S = [v for k, v in streams.items() if k[1] == tf and k[2] == fam]
        if S:
            L = min(len(x) for x in S)
            M = np.mean([x.to_numpy()[-L:] for x in S], axis=0)
            sh, t = sharpe_t(M)
            port.append({"tf": tf, "family": fam, "streams": len(S), "portfolio_sharpe": sh, "portfolio_t": t})
    PF = pd.DataFrame(port)
    if len(PF):
        R = R.merge(PF, on=["tf", "family"], how="left")
    R["o"] = R.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    R = R.sort_values("median_single_trader", ascending=False).drop(columns="o")
    R.to_csv(OUT / "ranking.csv", index=False)
    T = R.loc[R.groupby("tf").median_single_trader.idxmax()].copy()
    T["o"] = T.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    T.sort_values("o").drop(columns="o").to_csv(OUT / "by_tf.csv", index=False)
    PF.to_csv(OUT / "portfolios.csv", index=False)
    P = pd.DataFrame(prm)
    if len(P):
        num = P.select_dtypes("number").columns
        P.groupby(["tf", "family"])[list(num)].median().to_csv(OUT / "params.csv")
    best = R.iloc[0].to_dict()
    summ = {"cells": int(len(C)), "pairs": int(C.pair.nunique()), "tfs": [t for t in TF_ORDER if t in set(C.tf)],
            "families": sorted(C.family.unique().tolist()), "bh_survivors_q10": int((C.p_bh < 0.10).sum()),
            "share_cells_single_trader_positive": float((C.single_trader_median > 0).mean()),
            "best_tf_family": best, "median_single_trader_all": float(C.single_trader_median.median()),
            "median_gross_all": float(C.single_trader_gross.median())}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(R.head(20).round(3).to_string())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
