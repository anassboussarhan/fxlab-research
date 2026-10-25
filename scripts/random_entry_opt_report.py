"""Aggregate scripts/random_entry_opt.py: which management strategy and timeframe is best for RANDOM entries?

Per instrument x timeframe x family the walk-forward results (parameters optimised on training data with random
entries, tested on later data with new random entries), the hindsight check (optimum on the whole sample re-run
with fresh entries) and gross vs net. Ranking by the median, across instruments, of the single-trader
out-of-sample Sharpe. Significance: one-sided t of the expected-value stream (mean over the test seeds),
Benjamini-Hochberg over all instrument x timeframe x family cells. Portfolios: equal weight across instruments on
common trading days (streams written without dates by older versions borrow the calendar of a dated stream of
the same instrument and timeframe - every family shares the walk-forward span - or are aligned by their start).
Writes results/reports/random_entry_opt_summary/{cells.csv, ranking.csv, by_tf.csv, best_tf_by_family.csv,
portfolios.csv, params.csv, choices.csv, summary.json}."""
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
CATEGORICAL = ("etype", "size", "mg_mode", "vol", "sar", "flat_roll")


def sharpe_t(d):
    d = pd.Series(d).dropna()
    if len(d) < 20 or d.std() == 0:
        return 0.0, 0.0
    return float(d.mean() / d.std() * np.sqrt(260)), float(d.mean() / d.std() * np.sqrt(len(d)))


def load_streams(files):
    """{(pair, tf, family): daily Series}; dateless series borrow a dated calendar of the same pair x tf."""
    raw, cal = {}, {}
    for f in files:
        dp = f[:-5] + "_daily.parquet"
        if not os.path.exists(dp):
            continue
        J = json.load(open(f))
        D = pd.read_parquet(dp)
        if isinstance(D.index, pd.DatetimeIndex):
            cal.setdefault((J["pair"], J["tf"]), D.index)
        for fam in D.columns:
            raw[(J["pair"], J["tf"], fam)] = D[fam]
    out = {}
    for k, s in raw.items():
        if not isinstance(s.index, pd.DatetimeIndex):
            c = cal.get(k[:2])
            if c is not None and len(c) == len(s):
                s = pd.Series(s.to_numpy(), index=c)
        out[k] = s
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, prm = [], []
    files = sorted(glob.glob(str(IN / "*.json")))
    for f in files:
        J = json.load(open(f))
        for fam, R in J.get("families", {}).items():
            w, h = R["wf"], R["hindsight"]
            sel = R["selection"]
            rows.append({"pair": J["pair"], "tf": J["tf"], "family": fam,
                         "single_trader_median": w["single_trader_median_sharpe"],
                         "single_trader_share_positive": float(np.mean(np.array(w["single_trader_sharpes"]) > 0)),
                         "single_trader_gross": w["single_trader_median_gross"],
                         "expected_stream_sharpe": w["expected_stream_sharpe"], "t": w["expected_stream_t"],
                         "expected_stream_gross": w["expected_stream_gross_sharpe"], "days": w["days"],
                         "ann_return": w["expected_stream_ann_return"], "train_median": w["median_train_sharpe"],
                         "test_trades_per_fold": float(np.mean([s.get("test_trades_mean", np.nan) for s in sel])),
                         "hindsight_in_sample": h["in_sample_best"], "hindsight_rerandomised": h["rerandomised"],
                         "seconds": R.get("seconds")})
            for s in sel:
                p = {k: v for k, v in s["params"].items()}
                p.update({"tf": J["tf"], "family": fam, "session": "h0" in p, "uses_tp": "tp" in p and p.get("tp", 0) > 0,
                          "uses_trail": "trail" in p, "uses_be": "be" in p, "uses_part": "part" in p,
                          "uses_time": "tmax" in p})
                prm.append(p)
    if not rows:
        json.dump({"cells": 0}, open(OUT / "summary.json", "w"))
        print("no results")
        return
    streams = load_streams(files)
    C = pd.DataFrame(rows)
    # tail risk of the expected-value stream (returns are fractions of starting equity, summed): max drawdown,
    # worst day, skew, ruin (cumulative loss of half the account at some point), share of calendar years positive
    risk = []
    for (pair, tf, fam), s in streams.items():
        x = s.dropna().astype(float)
        if len(x) < 20:
            continue
        cum = x.cumsum()
        r = {"pair": pair, "tf": tf, "family": fam, "max_drawdown": float((cum - cum.cummax().clip(lower=0)).min()),
             "worst_day": float(x.min()), "skew": float(x.skew()), "ruin50": bool(cum.min() <= -0.5)}
        if isinstance(x.index, pd.DatetimeIndex):
            yr = x.groupby(x.index.year).sum()
            r["share_years_positive"] = float((yr > 0).mean())
        risk.append(r)
    if risk:
        C = C.merge(pd.DataFrame(risk), on=["pair", "tf", "family"], how="left")
    C["p"] = st.t.sf(C.t, df=np.maximum(C.days - 1, 1))
    C["p_bh"] = benjamini_hochberg(C.p.to_numpy())
    C["o"] = C.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    C.sort_values(["o", "family", "pair"]).drop(columns="o").to_csv(OUT / "cells.csv", index=False)
    agg = dict(pairs=("pair", "nunique"), median_single_trader=("single_trader_median", "median"),
               share_pairs_positive=("single_trader_median", lambda x: float((x > 0).mean())),
               median_gross=("single_trader_gross", "median"), median_expected_stream=("expected_stream_sharpe", "median"),
               median_expected_gross=("expected_stream_gross", "median"),
               median_t=("t", "median"), bh_survivors=("p_bh", lambda p: int((p < 0.10).sum())),
               median_train=("train_median", "median"), median_trades_per_fold=("test_trades_per_fold", "median"),
               median_hindsight_in_sample=("hindsight_in_sample", "median"),
               median_hindsight_rerandomised=("hindsight_rerandomised", "median"))
    if "max_drawdown" in C:
        agg.update(median_max_drawdown=("max_drawdown", "median"), worst_max_drawdown=("max_drawdown", "min"),
                   share_ruin50=("ruin50", "mean"), median_skew=("skew", "median"))
    if "share_years_positive" in C:
        agg["median_share_years_positive"] = ("share_years_positive", "median")
    R = C.groupby(["tf", "family"]).agg(**agg).reset_index()
    # equal-weight portfolio across instruments of the expected-value streams, on common trading days
    port, pser = [], {}
    for (tf, fam) in R[["tf", "family"]].itertuples(index=False):
        S = {k[0]: v for k, v in streams.items() if k[1] == tf and k[2] == fam}
        if not S:
            continue
        if all(isinstance(v.index, pd.DatetimeIndex) for v in S.values()):
            M = pd.DataFrame(S).mean(axis=1, skipna=True)
            aligned = "dates"
        else:                                    # fallback: align by the start of the walk-forward span
            L = min(len(x) for x in S.values())
            M = pd.Series(np.mean([x.to_numpy()[:L] for x in S.values()], axis=0))
            aligned = "start"
        sh, t = sharpe_t(M)
        pser[(tf, fam)] = M
        port.append({"tf": tf, "family": fam, "streams": len(S), "portfolio_sharpe": sh, "portfolio_t": t,
                     "portfolio_ann_return": float(M.mean() * 260), "aligned_by": aligned})
    PF = pd.DataFrame(port)
    if len(PF):
        PF["portfolio_p"] = st.norm.sf(PF.portfolio_t)
        PF["portfolio_p_bh"] = benjamini_hochberg(PF.portfolio_p.to_numpy())
        R = R.merge(PF[["tf", "family", "portfolio_sharpe", "portfolio_t", "portfolio_p_bh"]], on=["tf", "family"], how="left")
    R = R.sort_values("median_single_trader", ascending=False)
    R.to_csv(OUT / "ranking.csv", index=False)
    T = R.loc[R.groupby("tf").median_single_trader.idxmax()].copy()
    T["o"] = T.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    T.sort_values("o").drop(columns="o").to_csv(OUT / "by_tf.csv", index=False)
    F = R.loc[R.groupby("family").median_single_trader.idxmax()].sort_values("median_single_trader", ascending=False)
    F.to_csv(OUT / "best_tf_by_family.csv", index=False)
    if len(PF):
        PF.sort_values("portfolio_sharpe", ascending=False).to_csv(OUT / "portfolios.csv", index=False)
    yrs = []                                     # year by year: the 10 best-ranked timeframe x family portfolios
    for tf, fam in R.head(10)[["tf", "family"]].itertuples(index=False):
        M = pser.get((tf, fam))
        if M is None or not isinstance(M.index, pd.DatetimeIndex):
            continue
        for y, g in M.groupby(M.index.year):
            yrs.append({"tf": tf, "family": fam, "year": int(y), "return": float(g.sum()), "sharpe": sharpe_t(g)[0]})
    if yrs:
        pd.DataFrame(yrs).to_csv(OUT / "years.csv", index=False)
    P = pd.DataFrame(prm)
    if len(P):
        num = [c for c in P.select_dtypes("number").columns if c not in ("h0", "width")]
        P.groupby(["tf", "family"])[num].median().to_csv(OUT / "params.csv")
        ch = []
        for (tf, fam), g in P.groupby(["tf", "family"]):
            r = {"tf": tf, "family": fam, "folds": len(g)}
            for c in CATEGORICAL:
                if c in g and g[c].notna().any():
                    for v, s in g[c].value_counts(normalize=True).items():
                        r[f"{c}={v}"] = float(s)
            for c in ("session", "uses_tp", "uses_trail", "uses_be", "uses_part", "uses_time"):
                r[c] = float(g[c].mean())
            ch.append(r)
        pd.DataFrame(ch).to_csv(OUT / "choices.csv", index=False)
    best = R.iloc[0].to_dict()
    top = C.sort_values("t", ascending=False).head(10)[["pair", "tf", "family", "single_trader_median",
                                                         "expected_stream_sharpe", "t", "p", "p_bh"]]
    summ = {"cells": int(len(C)), "pairs": int(C.pair.nunique()), "tfs": [t for t in TF_ORDER if t in set(C.tf)],
            "families": sorted(C.family.unique().tolist()), "bh_survivors_q10": int((C.p_bh < 0.10).sum()),
            "share_cells_single_trader_positive": float((C.single_trader_median > 0).mean()),
            "share_cells_expected_stream_positive": float((C.expected_stream_sharpe > 0).mean()),
            "share_cells_t_above_1.645": float((C.t > 1.645).mean()),
            "best_tf_family": best, "median_single_trader_all": float(C.single_trader_median.median()),
            "median_gross_all": float(C.single_trader_gross.median()),
            "median_train_all": float(C.train_median.median()),
            "hindsight": {"median_in_sample": float(C.hindsight_in_sample.median()),
                          "median_rerandomised": float(C.hindsight_rerandomised.median()),
                          "share_rerandomised_positive": float((C.hindsight_rerandomised > 0).mean())},
            "portfolio_bh_survivors_q10": int((PF.portfolio_p_bh < 0.10).sum()) if len(PF) else 0,
            "best_portfolio": PF.sort_values("portfolio_sharpe", ascending=False).head(1).to_dict("records") if len(PF) else None,
            "top_cells_by_t": top.to_dict("records")}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(R.head(25).round(3).to_string())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
