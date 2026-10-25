"""Summaries of an options-strategies run (branch options-strategies) for docs/options_strategies.md.

Reads combos_primary.csv.gz, combos_fill_sensitivity.csv.gz, summary.json, walkforward_*.csv, top30_drawdowns.csv,
prop_top30.csv and synth/*; writes the tables below to OUT and prints them.
  structures.csv   per structure and days to expiry, no filter: SPY, DIA, EW stocks, the IV-RV quintile portfolios
                   and the median over single stocks (best management rule and hold-to-expiry)
  management.csv   change in Sharpe from each management rule vs holding to expiry, same combination
  filters.csv      change in Sharpe from each entry filter vs no filter, same combination
  fills.csv        mean P&L at the three fill assumptions, hold-to-expiry and 50% / 21-day
  walkforward.csv  out-of-sample result of choosing the best combination each year, per family and criterion
Usage: python scripts/options_strategy_report.py --run DIR --out DIR"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

PORTF = ("EW", "XSQ5", "XSQ1", "GS_LS")
ETFS = ("SPY", "DIA", "XLF", "XLE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(os.path.join(a.run, "combos_primary.csv.gz"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    stocks = T[~T.symbol.isin(PORTF + ETFS)]

    # 1. structures, no filter
    base = T[T["filter"] == "ALL"]
    rows = []
    for (s, dte), g in base.groupby(["struct", "dte"]):
        r = {"struct": s, "dte": dte}
        for sym in ("SPY", "DIA", "XLF", "EW", "XSQ5", "XSQ1"):
            h = g[g.symbol == sym]
            if len(h):
                hh = h[h.mgmt == "H"]
                b = h.sort_values("sharpe", ascending=False).iloc[0]
                r[f"{sym}_sharpe_H"] = float(hh.sharpe.iloc[0]) if len(hh) else np.nan
                r[f"{sym}_best_mgmt"] = b.mgmt
                r[f"{sym}_sharpe_best"] = float(b.sharpe)
                r[f"{sym}_mean_H"] = float(hh["mean"].iloc[0]) if len(hh) else np.nan
                r[f"{sym}_worst_H"] = float(hh.worst.iloc[0]) if len(hh) else np.nan
        sg = g[~g.symbol.isin(PORTF + ETFS) & (g.mgmt == "H")]
        r["stocks_n"] = int(len(sg))
        r["stocks_median_sharpe_H"] = float(sg.sharpe.median()) if len(sg) else np.nan
        r["stocks_share_sharpe_pos_H"] = float((sg.sharpe > 0).mean()) if len(sg) else np.nan
        r["stocks_median_worst_H"] = float(sg.worst.median()) if len(sg) else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out, "structures.csv"), index=False)
    cols = ["struct", "dte", "SPY_sharpe_H", "SPY_sharpe_best", "SPY_best_mgmt", "SPY_mean_H", "SPY_worst_H",
            "DIA_sharpe_H", "EW_sharpe_H", "XSQ5_sharpe_H", "XSQ1_sharpe_H", "stocks_median_sharpe_H",
            "stocks_share_sharpe_pos_H", "stocks_median_worst_H"]
    print("== structures (no filter, Sharpe by product)")
    print(S[[c for c in cols if c in S.columns]].round(3).to_string(index=False))

    # 2. management vs hold, 3. filters vs none
    key = ["symbol", "struct", "dte", "filter"]
    H = T[T.mgmt == "H"].set_index(key).sharpe
    M = T[T.mgmt != "H"].join(H.rename("sharpe_H"), on=key)
    M["d"] = M.sharpe - M.sharpe_H
    mg = M.groupby(["mgmt", M.symbol.isin(PORTF + ETFS).map({True: "etf/portfolio", False: "stock"})]).d.agg(
        ["count", "mean", "median", lambda x: (x > 0).mean()]).rename(columns={"<lambda_0>": "share_better"})
    mg.to_csv(os.path.join(a.out, "management.csv"))
    print("\n== management rule vs hold to expiry (change in Sharpe, same combination)")
    print(mg.round(3).to_string())
    key2 = ["symbol", "struct", "dte", "mgmt"]
    A0 = T[T["filter"] == "ALL"].set_index(key2).sharpe
    Fz = T[T["filter"] != "ALL"].join(A0.rename("sharpe_all"), on=key2)
    Fz["d"] = Fz.sharpe - Fz.sharpe_all
    fl = Fz.groupby("filter").d.agg(["count", "mean", "median", lambda x: (x > 0).mean()]).rename(
        columns={"<lambda_0>": "share_better"})
    fl.to_csv(os.path.join(a.out, "filters.csv"))
    print("\n== entry filter vs no filter (change in Sharpe, same combination)")
    print(fl.round(3).to_string())

    # 4. fills
    fpath = os.path.join(a.run, "combos_fill_sensitivity.csv.gz")
    if os.path.exists(fpath):
        Fs = pd.read_csv(fpath)
        both = pd.concat([T[T.mgmt.isin(["H", "T50D21", "T50"])], Fs])
        both = both[both["filter"] == "ALL"]
        fi = both.pivot_table(index=["struct"], columns="fill", values="mean", aggfunc="median")
        fi.to_csv(os.path.join(a.out, "fills.csv"))
        print("\n== median mean P&L per cycle (fraction of notional) by fill assumption, no filter")
        print((fi * 1e4).round(1).to_string())

    # 5. walk-forward
    summ = json.load(open(os.path.join(a.run, "summary.json")))
    W = pd.DataFrame([{"family_criterion": k, **{kk: v for kk, v in d.items() if kk not in ("years", "picks")},
                       "picks": " ; ".join(d.get("picks", []))} for k, d in summ["walk_forward"].items()])
    W.to_csv(os.path.join(a.out, "walkforward.csv"), index=False)
    print("\n== walk-forward (pick the best on earlier years, trade the next year)")
    print(W.drop(columns="picks").round(4).to_string(index=False))
    for k, d in summ["walk_forward"].items():
        print(" ", k, d.get("picks"))
    print("\n== rank stability (2019-2022 vs 2023-2025)", summ.get("rank_stability"))
    print("combinations:", summ.get("n_combinations_primary"), "BH survivors mean / alpha:",
          summ.get("bh_survivors_mean"), summ.get("bh_survivors_alpha"))
    b = T.sort_values("sharpe", ascending=False)
    print("best full-sample Sharpe:", b.iloc[0][["symbol", "struct", "dte", "mgmt", "filter", "cycles", "sharpe"]].to_dict(),
          "deflated Sharpe probability:", b.deflated_sharpe_prob.dropna().tolist())

    # 6. robust combinations: many cycles, positive in most years
    R = T[(T.cycles >= 48)].copy()
    ycols = [c for c in T.columns if c.startswith("y20")]
    R["years_neg"] = (R[ycols] < 0).sum(axis=1)
    R = R.sort_values("t_nw", ascending=False)
    R.head(40).to_csv(os.path.join(a.out, "robust_top40.csv"), index=False)
    print("\n== top 25 by t with 48+ cycles")
    print(R[["symbol", "struct", "dte", "mgmt", "filter", "cycles", "mean", "t_nw", "sharpe", "worst", "alpha",
             "t_alpha", "years_neg", "sharpe_is", "sharpe_oos", "bh_q_mean"]].head(25).round(4).to_string(index=False))

    # 7. drawdowns and provider simulation
    for f in ("top30_drawdowns.csv", "prop_top30.csv"):
        p = os.path.join(a.run, f)
        if os.path.exists(p):
            X = pd.read_csv(p)
            print(f"\n== {f}")
            if f.startswith("prop"):
                X = X.sort_values("mean_net", ascending=False)
                print(X[["combo", "provider", "daily_sd_target", "notional_per_account", "worst_day_account", "starts",
                         "pass_rate", "paid_rate", "mean_net", "median_net", "p10_net", "breach_rate"]].head(30)
                      .round(3).to_string(index=False))
            else:
                print(X.round(4).to_string(index=False))
    # 8. synthetic layer
    sp = os.path.join(a.run, "synth", "summary_synth.json")
    if os.path.exists(sp):
        ss = json.load(open(sp))
        print("\n== synthetic layer validation", ss.get("validation"))
        pre = os.path.join(a.run, "synth", "combos_synth_pre2019.csv.gz")
        if os.path.exists(pre):
            P = pd.read_csv(pre)
            P = P[(P["filter"] == "ALL") & (P.fill == 0.75)]
            P.to_csv(os.path.join(a.out, "synth_pre2019_nofilter.csv"), index=False)
            print(P.sort_values("sharpe", ascending=False)[["symbol", "struct", "dte", "mgmt", "cycles", "mean",
                                                             "sharpe", "worst", "worst_year", "y2008" if "y2008" in P.columns else "first"]]
                  .head(25).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
