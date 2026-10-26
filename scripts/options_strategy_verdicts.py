"""Per-structure summary of an options-strategies run (every structure, first and second batch): for SPY / DIA
the best of 14 / 30 / 45 days held to expiry without a filter (Sharpe, mean, worst cycle, beta, alpha t, the
2023-2025 Sharpe), for single stocks the median across names, and for the equal-weight stock portfolio the alpha t;
plus the change in Sharpe from each management rule and filter, and the Benjamini-Hochberg survivors by structure.
Usage: python scripts/options_strategy_verdicts.py --combos combos_primary.csv.gz --out DIR"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_strategy_lab as L  # noqa: E402

PORTF = ["EW", "XSQ5", "XSQ1", "GS_LS"]
ETFS = ["SPY", "DIA", "XLF", "XLE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(a.combos)
    T = T[T.fill == L.PRIMARY_FILL]
    H = T[(T.mgmt == "H") & (T["filter"] == "ALL")]
    rows = []
    for s in sorted(H.struct.unique()):
        r = {"struct": s, "batch": "first" if s in L.BASE_STRUCTS or s == "GSLS" else "second",
             "defined_risk": s in L.DEFINED_RISK, "short_premium": s in L.SHORT_PREMIUM}
        for u in ("SPY", "DIA"):
            x = H[(H.symbol == u) & (H.struct == s)]
            if len(x):
                b = x.loc[x.sharpe.idxmax()]
                r.update({f"{u}_dte": int(b.dte), f"{u}_sharpe": b.sharpe, f"{u}_mean": b["mean"], f"{u}_worst": b.worst,
                          f"{u}_beta": b.beta, f"{u}_alpha_t": b.t_alpha, f"{u}_sharpe_2023_25": b.sharpe_oos,
                          f"{u}_alpha_t_all_dtes": ";".join(f"{int(v.dte)}:{v.t_alpha:.1f}" for _, v in x.sort_values("dte").iterrows())})
        st_ = H[(~H.symbol.isin(ETFS + PORTF)) & (H.struct == s)]
        if len(st_):
            by = st_.groupby("dte").agg(med_sharpe=("sharpe", "median"), med_alpha_t=("t_alpha", "median"),
                                        share_alpha_t2=("t_alpha", lambda v: float((v > 2).mean())), n=("symbol", "size"))
            best = by.med_sharpe.idxmax()
            r.update({"stocks_best_dte": int(best), "stocks_median_sharpe": by.loc[best, "med_sharpe"],
                      "stocks_median_alpha_t": by.loc[best, "med_alpha_t"],
                      "stocks_share_alpha_t_over_2": by.loc[best, "share_alpha_t2"], "stocks_n": int(by.loc[best, "n"]),
                      "stocks_median_sharpe_all_dtes": ";".join(f"{d}:{v:.2f}" for d, v in by.med_sharpe.items())})
        ew = H[(H.symbol == "EW") & (H.struct == s)]
        if len(ew):
            b = ew.loc[ew.sharpe.idxmax()]
            r.update({"EW_dte": int(b.dte), "EW_sharpe": b.sharpe, "EW_alpha_t": b.t_alpha})
        allc = T[T.struct == s]
        r.update({"combos": int(len(allc)), "bh_mean_survivors": int((allc.bh_q_mean < 0.05).sum()),
                  "bh_alpha_survivors": int((allc.bh_q_alpha < 0.05).sum())})
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out, "verdicts_by_structure.csv"), index=False)
    # management and filters: change in Sharpe against hold-to-expiry / no filter, same combination
    key = ["symbol", "struct", "dte", "filter"]
    base = T[T.mgmt == "H"].set_index(key).sharpe
    M = T[T.mgmt != "H"].join(base.rename("sharpe_H"), on=key).dropna(subset=["sharpe", "sharpe_H"])
    M["d"] = M.sharpe - M.sharpe_H
    M["grp"] = np.where(M.symbol.isin(ETFS + PORTF), "etf/portfolio", "stock")
    mg = M.groupby(["mgmt", "grp"]).d.agg(mean="mean", median="median", improved=lambda v: float((v > 0).mean()),
                                          n="size").reset_index()
    mg.to_csv(os.path.join(a.out, "management_v2.csv"), index=False)
    key2 = ["symbol", "struct", "dte", "mgmt"]
    nb = T[T["filter"] == "ALL"].set_index(key2).sharpe
    F = T[T["filter"] != "ALL"].join(nb.rename("sharpe_all"), on=key2).dropna(subset=["sharpe", "sharpe_all"])
    F["d"] = F.sharpe - F.sharpe_all
    fl = F.groupby("filter").d.agg(mean="mean", median="median", improved=lambda v: float((v > 0).mean()),
                                   n="size").reset_index()
    fl.to_csv(os.path.join(a.out, "filters_v2.csv"), index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 200)
    cols = ["struct", "batch", "SPY_dte", "SPY_sharpe", "SPY_alpha_t", "SPY_alpha_t_all_dtes", "SPY_sharpe_2023_25",
            "DIA_sharpe", "DIA_alpha_t", "stocks_median_sharpe", "stocks_median_alpha_t", "stocks_share_alpha_t_over_2",
            "EW_alpha_t", "bh_alpha_survivors"]
    print(S[[c for c in cols if c in S.columns]].round(2).to_string())
    print(mg.round(3).to_string())
    print(fl.round(3).to_string())


if __name__ == "__main__":
    main()
