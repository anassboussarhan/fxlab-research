"""Program-wide multiple-testing summary across EVERYTHING tested on development data (2007-2025).

Collects: every strategy experiment in the results DB (all campaigns, all timeframes, entry types, financing
modes), every stream of the all-feature scan, every all-feature ML run. For each stream: OOS net Sharpe and a
one-sided t-test p-value of the daily mean. Then
  * Benjamini-Hochberg (FDR) and Holm (FWER) over ALL streams of the program together;
  * Deflated Sharpe of the single best stream with N = total number of configurations tried in the program
    (strategy candidates x folds are not counted separately - the number of distinct candidates is used, which
    is the lenient choice) and the cross-sectional variance of OOS Sharpe ratios;
  * the pre-declared survival bar (docs/methodology.md): Sharpe > 0, BH p < 0.10, DSR > 0.95, Sharpe > 0 at
    2x costs. Survivors would be frozen for the 2026 holdout; none are frozen by this script.
Output: results/reports/program/summary.json, streams.csv, by_campaign.csv"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg, holm, dsr  # noqa: E402
from fxlab.validation.metrics import daily  # noqa: E402

OUT = RESULTS_DIR / "reports" / "program"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect(read_only=True)
    rows = con.execute("SELECT id, campaign, target, symbol, tf, config, metrics, artifacts FROM experiments "
                       "WHERE status='ok' AND kind='strategy'").fetchall()
    con.close()
    S = []
    for eid, camp, tgt, sym, tf, cfg, m, art in rows:
        m, cfg = json.loads(m), json.loads(cfg)
        x2 = (m.get("stress") or {}).get("cost_x2.0") or {}
        ex = "grid" if cfg.get("exec_grid") else (cfg.get("exec") or {}).get("type", "market")
        S.append({"source": "strategy", "campaign": camp, "name": tgt, "symbol": sym, "tf": tf, "exec": ex,
                  "financing": cfg.get("financing", "flat"), "sharpe": m.get("sharpe"), "p": m.get("ttest_p"),
                  "n_days": m.get("n_days"), "cost_x2_sharpe": x2.get("sharpe") if isinstance(x2, dict) else None,
                  "n_candidates": m.get("n_candidates", 1), "artifact": json.loads(art).get("oos_returns")})
    fs = RESULTS_DIR / "reports" / "feature_scan"
    for d in sorted(fs.glob("*_*")):
        if (d / "summary.csv").exists():
            sym, tf = d.name.rsplit("_", 1)
            x = pd.read_csv(d / "summary.csv")
            x = x[~x.placebo]
            for r in x.itertuples():
                S.append({"source": "feature_scan", "campaign": "feature_scan", "name": r.column, "symbol": sym,
                          "tf": tf, "exec": "market", "financing": "flat", "sharpe": r.oos_sharpe,
                          "p": float(st.t.sf(r.t_stat, max(r.oos_days - 1, 1))), "n_days": r.oos_days,
                          "cost_x2_sharpe": None, "n_candidates": 48, "artifact": f"fs::{d}::{r.column}"})
    ml = RESULTS_DIR / "reports" / "ml_allfeatures"
    for d in sorted(ml.glob("*")):
        if (d / "summary.json").exists():
            j = json.loads((d / "summary.json").read_text())
            dn = pd.read_parquet(d / "oos_daily.parquet")["net"]
            t = dn.mean() / dn.std() * np.sqrt(len(dn)) if dn.std() > 0 else 0.0
            S.append({"source": "ml_allfeatures", "campaign": "ml_allfeatures", "name": f"LGBM_h{j['h']}",
                      "symbol": j["symbol"], "tf": j["tf"], "exec": "market", "financing": "flat",
                      "sharpe": j["oos_net_sharpe"], "p": float(st.t.sf(t, max(len(dn) - 1, 1))), "n_days": len(dn),
                      "cost_x2_sharpe": j["cost_x2_sharpe"], "n_candidates": 8, "artifact": None})
    T = pd.DataFrame(S)
    T["p"] = T["p"].fillna(1.0).clip(0, 1)
    T["p_bh"] = benjamini_hochberg(T.p.to_numpy())
    T["p_holm"] = holm(T.p.to_numpy())
    n_trials = int(T.n_candidates.fillna(1).sum())
    var_sr = float((T.sharpe.dropna() / np.sqrt(260)).var())
    best = T.sort_values("sharpe", ascending=False).iloc[0]

    def stream(a):
        if not isinstance(a, str):
            return None
        if a.startswith("fs::"):
            _, d, col = a.split("::", 2)
            return pd.read_parquet(f"{d}/oos_daily.parquet")[col].dropna().to_numpy(float)
        return daily(pd.read_parquet(a)["ret"]).to_numpy() if os.path.exists(a) else None

    best_dsr = {}
    for src, g in T.groupby("source"):
        b = g.sort_values("sharpe", ascending=False).iloc[0]
        r = stream(b.artifact)
        if r is not None and len(r) > 50:
            best_dsr[src] = {"name": b["name"], "symbol": b.symbol, "tf": b.tf, "sharpe": float(b.sharpe),
                             "dsr": float(dsr(r, n_trials, var_sr))}
    T["survives_bar"] = (T.sharpe > 0) & (T.p_bh < 0.10) & (T.cost_x2_sharpe.fillna(-1) > 0)
    by = T.groupby(["source", "campaign", "tf", "exec", "financing"]).agg(
        streams=("sharpe", "size"), median_sharpe=("sharpe", "median"), share_positive=("sharpe", lambda s: (s > 0).mean()),
        best_sharpe=("sharpe", "max"), bh10=("p_bh", lambda p: int((p < 0.10).sum())),
        holm05=("p_holm", lambda p: int((p < 0.05).sum()))).reset_index()
    by.to_csv(OUT / "by_campaign.csv", index=False)
    T.drop(columns=["artifact"]).sort_values("sharpe", ascending=False).to_csv(OUT / "streams.csv", index=False)
    res = {"streams_total": int(len(T)), "by_source": T.source.value_counts().to_dict(),
           "configurations_tried": n_trials, "median_oos_sharpe": float(T.sharpe.median()),
           "share_positive": float((T.sharpe > 0).mean()), "bh_significant_10pct": int((T.p_bh < 0.10).sum()),
           "holm_significant_5pct": int((T.p_holm < 0.05).sum()), "survivors_of_predeclared_bar": int(T.survives_bar.sum()),
           "best_stream": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in
                           best.drop(labels=["artifact"]).to_dict().items()},
           "best_stream_deflated_sharpe": best_dsr,
           "top10": T.sort_values("sharpe", ascending=False).head(10)[["source", "name", "symbol", "tf", "exec",
                                                                       "financing", "sharpe", "p", "p_bh"]].round(4).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print(by.sort_values("median_sharpe", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
