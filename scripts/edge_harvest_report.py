"""Aggregate scripts/edge_harvest.py over instruments, timeframes and horizons, plus the diversified ("hedged")
portfolio layer: equal-weight portfolios of the selected-policy out-of-sample streams across all instruments
(per timeframe and horizon, and over everything), which nets opposite exposures and diversifies risk.
Significance: one-sided t of daily net returns, Benjamini-Hochberg over all instrument x timeframe x horizon
streams. Writes results/reports/edge_harvest_summary/{streams.csv, by_tf.csv, portfolios.csv, choices.csv,
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

IN = Path(os.environ.get("EH_IN", RESULTS_DIR / "reports" / "edge_harvest"))
OUT = Path(os.environ.get("EH_OUT", RESULTS_DIR / "reports" / "edge_harvest_summary"))
TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H2", "H4", "H8", "D1", "TICK60", "VOLA60"]


def sharpe(d):
    d = d.dropna()
    return (float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0,
            float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 20 and d.std() > 0 else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, choices, daily = [], [], {}
    for f in sorted(glob.glob(str(IN / "*.json"))):
        J = json.load(open(f))
        P = J.get("policy_grid", [])
        dp = f[:-5] + "_daily.parquet"
        D = pd.read_parquet(dp) if os.path.exists(dp) else pd.DataFrame()
        for h, H in J.get("horizons", {}).items():
            s = H["selected"]
            rows.append({"pair": J["pair"], "tf": J["tf"], "h": int(h), "ic_oos": H["ic_oos"], "hit": H["hit_rate"],
                         "cost_rt_bps": H["cost_rt_bps"], "net_sharpe": s["net_sharpe"], "gross_sharpe": s["gross_sharpe"],
                         "cost_x2_sharpe": s["cost_x2_sharpe"], "ann_return": s["ann_return"], "t": s["t"], "days": s["days"],
                         "hindsight_best_sharpe": H["best_in_hindsight"]["sharpe"]})
            for sel in H["selection"]:
                p = P[sel["policy"]] if sel["policy"] < len(P) else {}
                choices.append({"tf": J["tf"], "h": int(h), "kind": p.get("kind"), "exec": p.get("exec", p.get("entry")),
                                "theta": p.get("theta"), "exit": p.get("exit", p.get("risk")), "size": p.get("size"),
                                "pyramid": p.get("pyramid")})
            col = f"h{h}"
            if col in D:
                daily[(J["pair"], J["tf"], int(h))] = D[col]
    if not rows:
        json.dump({"streams": 0}, open(OUT / "summary.json", "w"))
        print("no edge harvest results")
        return
    S = pd.DataFrame(rows)
    S["p"] = st.t.sf(S.t, df=np.maximum(S.days - 1, 1))
    S["p_bh"] = benjamini_hochberg(S.p.to_numpy())
    S["o"] = S.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    S.sort_values(["o", "h", "net_sharpe"], ascending=[True, True, False]).drop(columns="o").to_csv(OUT / "streams.csv", index=False)
    T = S.groupby(["tf", "h"]).agg(pairs=("pair", "nunique"), median_ic=("ic_oos", "median"), median_hit=("hit", "median"),
                                   median_gross=("gross_sharpe", "median"), median_net=("net_sharpe", "median"),
                                   share_net_positive=("net_sharpe", lambda x: float((x > 0).mean())),
                                   best_net=("net_sharpe", "max"), median_x2=("cost_x2_sharpe", "median"),
                                   bh_survivors=("p_bh", lambda p: int((p < 0.10).sum())),
                                   median_hindsight_best=("hindsight_best_sharpe", "median")).reset_index()
    T["o"] = T.tf.map({t: i for i, t in enumerate(TF_ORDER)})
    T = T.sort_values(["o", "h"]).drop(columns="o")
    T.to_csv(OUT / "by_tf.csv", index=False)
    C = pd.DataFrame(choices)
    if len(C):
        C["label"] = C.kind.astype(str) + ":" + C["exec"].astype(str)
        CC = C.groupby(["tf", "label"]).size().unstack(fill_value=0)
        CC = CC.div(CC.sum(1), axis=0)
        CC.to_csv(OUT / "choices.csv")
    # diversified / hedged portfolios: equal weight across instruments
    port = []
    if daily:
        idx = sorted(set().union(*[set(v.index) for v in daily.values()]))
        for (tf, h) in sorted({(k[1], k[2]) for k in daily}):
            M = pd.DataFrame({k[0]: v for k, v in daily.items() if k[1] == tf and k[2] == h}).reindex(idx)
            pr = M.mean(axis=1, skipna=True)
            sh, t = sharpe(pr)
            port.append({"portfolio": f"{tf} h={h}", "streams": M.shape[1], "sharpe": sh, "t": t,
                         "ann_return": float(pr.mean() * 260)})
        A = pd.DataFrame({f"{k[0]}_{k[1]}_{k[2]}": v for k, v in daily.items()}).reindex(idx)
        pr = A.mean(axis=1, skipna=True)
        sh, t = sharpe(pr)
        port.append({"portfolio": "ALL instruments x timeframes x horizons", "streams": A.shape[1], "sharpe": sh, "t": t,
                     "ann_return": float(pr.mean() * 260)})
    PF = pd.DataFrame(port)
    PF.to_csv(OUT / "portfolios.csv", index=False)
    summ = {"streams": int(len(S)), "pairs": int(S.pair.nunique()), "tfs": [t for t in TF_ORDER if t in set(S.tf)],
            "median_net_sharpe": float(S.net_sharpe.median()), "share_net_positive": float((S.net_sharpe > 0).mean()),
            "median_gross_sharpe": float(S.gross_sharpe.median()), "bh_survivors_q10": int((S.p_bh < 0.10).sum()),
            "best_stream": S.sort_values("net_sharpe", ascending=False).head(1)[["pair", "tf", "h", "net_sharpe", "p_bh"]].to_dict("records"),
            "best_portfolio": PF.sort_values("sharpe", ascending=False).head(1).to_dict("records") if len(PF) else None}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(T.round(3).to_string())
    print(PF.round(3).to_string())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
