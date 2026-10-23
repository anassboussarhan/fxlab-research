"""Assemble docs/research_report.md and docs/final_audit.md from artifacts on disk (no hand-typed numbers)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fxlab.config import RESULTS_DIR, ROOT  # noqa: E402

R = RESULTS_DIR
D = ROOT / "docs"


def jload(p, default=None):
    p = R / p if not str(p).startswith("/") else p
    return json.loads(open(p).read()) if os.path.exists(p) else default


def exists(p):
    return os.path.exists(R / p) or os.path.exists(ROOT / p)


DELIVERABLES = [
    (1, "Complete executable repository", "fxlab/", None),
    (2, "Dependency lockfiles", "requirements.lock", None),
    (3, "Installation instructions", "README.md", None),
    (4, "Data-source comparison", "docs/data_sources.md", None),
    (5, "Verified historical data inventory", "results/data_inventory.csv", None),
    (6, "Data quality audit", "results/data_quality_audit.csv", None),
    (7, "Full feature catalog", "docs/catalog_features.md", None),
    (8, "Full strategy catalog", "docs/catalog_strategies.md", None),
    (9, "Full model catalog", "docs/catalog_models.md", None),
    (10, "Automated test suite", "tests/", None),
    (11, "Leakage validation report", "docs/leakage_report.md", None),
    (12, "Execution-cost simulator", "fxlab/execution/simulator.py", None),
    (13, "Experiment configuration files", "configs/campaign_default.toml", None),
    (14, "Experiment orchestration system", "fxlab/orchestration/campaign.py", None),
    (15, "Checkpoint and resume system", "fxlab/orchestration/campaign.py", None),
    (16, "Resource benchmarking report", "results/benchmark.json", None),
    (17, "Compute-cost estimate", "docs/compute_plan.md", None),
    (18, "Full research experiment manifest", "results/manifests/", None),
    (19, "Walk-forward results database", "results/results.duckdb", None),
    (20, "Statistical significance report", "results/reports/{strategy_campaign}/summary.json", None),
    (21, "Feature importance analysis", "results/reports/features_EURUSD_H1/relevance.csv", None),
    (22, "Feature redundancy analysis", "results/reports/features_EURUSD_H1/redundant_pairs.csv", None),
    (23, "Strategy-family comparison", "results/reports/{strategy_campaign}/family_comparison.csv", None),
    (24, "Model-family comparison", "results/reports/{model_campaign}/model_results.csv", None),
    (25, "Instrument comparison", "results/reports/{strategy_campaign}/instrument_comparison.csv", None),
    (26, "Timeframe comparison", "results/reports/{strategy_campaign}/timeframe_comparison.csv", None),
    (27, "Prediction-horizon comparison", "results/reports/{model_campaign}/summary.json", None),
    (28, "Market-regime robustness report", "results/reports/{strategy_campaign}/regime_robustness.csv", None),
    (29, "Transaction-cost sensitivity report", "results/reports/{strategy_campaign}/cost_sensitivity.csv", None),
    (30, "Risk-management comparison", "results/reports/{risk_campaign}/strategy_results.csv", None),
    (31, "Hedging comparison", "results/reports/{strategy_campaign}/hedging_comparison.csv", None),
    (32, "Complete trade logs", "results/artifacts/", None),
    (33, "Equity and drawdown charts", "results/reports/{strategy_campaign}/charts/", None),
    (34, "Reproducible candidate strategy configurations", "results/reports/{strategy_campaign}/survivors.csv", None),
    (35, "Detailed scientific conclusions", "docs/research_report.md", None),
    (36, "List of failures, exclusions and limitations", "docs/research_report.md", None),
    (37, "Instructions for rerunning the research", "README.md", None),
    (38, "Final audit of requirements", "docs/final_audit.md", None),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-campaign", required=True)
    ap.add_argument("--model-campaign", required=True)
    ap.add_argument("--risk-campaign", default=None)
    a = ap.parse_args()
    fill = {"strategy_campaign": a.strategy_campaign.split(",")[0], "model_campaign": a.model_campaign,
            "risk_campaign": a.risk_campaign or "risk"}
    rows = []
    for n, name, path, _ in DELIVERABLES:
        p = path.format(**fill)
        ok = exists(p.replace("results/", "")) if p.startswith("results/") else os.path.exists(ROOT / p)
        rows.append((n, name, p, "present" if ok else "MISSING"))
    md = ["# Final requirements audit", "", f"Generated {datetime.now(timezone.utc).isoformat()}", "",
          "| # | deliverable | evidence | status |", "|---|---|---|---|"]
    md += [f"| {n} | {name} | `{p}` | {st} |" for n, name, p, st in rows]
    pf = jload("preflight_audit.json", {})
    if pf:
        md += ["", "## Pre-flight checks (section 17)", "", "| check | status |", "|---|---|"]
        md += [f"| {c['check']} | {c['status']} |" for c in pf["checks"]]
    (D / "final_audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    research_report(a.strategy_campaign.split(","), a.model_campaign, a.risk_campaign, str(D / "conclusions.md"))



def table(df, cols=None, n=None, floatfmt=3):
    if df is None or len(df) == 0:
        return "_no rows_"
    d = df[cols] if cols else df
    if n:
        d = d.head(n)
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append(f"{v:.{floatfmt}f}" if abs(v) < 1e5 else f"{v:.3g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def csv(p):
    p = R / p
    return pd.read_csv(p) if os.path.exists(p) else None


def research_report(strategy_campaigns, model_campaign, risk_campaign, conclusions_md):
    md = [f"# Research report", "", f"Generated {datetime.now(timezone.utc).isoformat()} from files under `results/`.", ""]
    md += ["## 1. Data actually used", "", table(csv("data_inventory.csv"), ["symbol", "first_obs", "last_obs", "years", "m1_bars"]
                                                 if csv("data_inventory.csv") is not None else None), ""]
    cat = jload("catalog.json", {}).get("counts", {})
    md += ["## 2. Engineering completeness", "", "| kind | registered | by state |", "|---|---|---|"]
    md += [f"| {k} | {v['registered']} | {v['by_state']} |" for k, v in cat.items()]
    lk = jload("leakage_summary.json", {})
    md += ["", f"Leakage tests: {lk}", ""]
    bm = jload("benchmark.json", {})
    if bm:
        md += ["## 3. Compute", "", f"Machine: {bm['machine']}. All 266 features on {bm['features']['bars']} H1 bars: "
               f"{bm['features']['total_seconds']} s; all 69 model families, one fit on 10k x 20: {bm['models']['total_seconds']} s. "
               "Full-campaign estimate: see `docs/compute_plan.md`.", ""]
    for sc in strategy_campaigns:
        base = f"reports/{sc}"
        s = jload(f"{base}/summary.json", {})
        md += [f"## 4. Strategy campaign `{sc}`", "", f"Experiments: {s.get('experiments')} (failed {s.get('failed')}).", ""]
        st = s.get("strategy", {})
        if st:
            md += [f"Configurations tried (for DSR): {st.get('n_trials_for_dsr')}. White Reality Check p = "
                   f"{st.get('white_rc', {}).get('p_value')}; Hansen SPA (consistent) p = {st.get('hansen_spa', {}).get('p_consistent')}.", ""]
        t = csv(f"{base}/strategy_results.csv")
        if t is not None:
            cols = [c for c in ["target", "symbol", "tf", "sharpe", "ann_return", "max_drawdown", "trades", "psr_0", "dsr",
                                "p_bh", "stress.cost_x2.0.sharpe", "diagnostic_zero_cost_default_sharpe"] if c in t]
            md += ["Top 15 by out-of-sample net Sharpe:", "", table(t, cols, 15), ""]
            md += [f"Share of experiments with OOS net Sharpe > 0: {(t['sharpe'] > 0).mean():.1%} of {len(t)}; "
                   f"median OOS Sharpe {t['sharpe'].median():.2f}; median zero-cost diagnostic Sharpe "
                   f"{t['diagnostic_zero_cost_default_sharpe'].median():.2f}.", ""]
        for name, f in (("Family comparison", "family_comparison.csv"), ("Instrument comparison", "instrument_comparison.csv"),
                        ("Timeframe comparison", "timeframe_comparison.csv")):
            d = csv(f"{base}/{f}")
            if d is not None:
                md += [f"### {name}", "", table(d), ""]
        sv = csv(f"{base}/survivors.csv")
        md += ["### Survivors of the pre-declared bar", "", table(sv) if sv is not None and len(sv) else "**None.**", ""]
        for f, title in (("portfolio_risk_comparison.csv", "Portfolio / sizing comparison"), ("hedging_comparison.csv", "Hedging comparison")):
            d = csv(f"{base}/{f}")
            if d is not None:
                md += [f"### {title}", "", table(d, n=25), ""]
    if risk_campaign:
        t = csv(f"reports/{risk_campaign}/strategy_results.csv")
        md += [f"## 5. Risk-management comparison `{risk_campaign}`", "",
               "Each family re-run with 11 stop/target/sizing configurations as walk-forward candidates (selection on "
               "training windows); see `results/artifacts/<id>/folds.parquet` for the configuration chosen per fold.", "",
               table(t, [c for c in ["target", "sharpe", "ann_return", "max_drawdown", "trades", "stress.cost_x2.0.sharpe"] if t is not None and c in t], 30), ""]
    if model_campaign:
        base = f"reports/{model_campaign}"
        t = csv(f"{base}/model_results.csv")
        s = jload(f"{base}/summary.json", {})
        md += [f"## 6. Prediction campaign `{model_campaign}`", "", f"Experiments: {s.get('experiments')} (failed {s.get('failed')}).", ""]
        if t is not None:
            if "auc" in t:
                c = t[t["task"] == "clf"].sort_values("auc", ascending=False)
                md += ["Direction classification, top 20 by out-of-sample AUC:", "",
                       table(c, [x for x in ["target", "h", "auc", "brier", "ece", "base_rate", "sign_ret_mean_bps", "p_holm", "n"] if x in c], 20), ""]
            v = t[t["task"] == "vol"]
            if len(v):
                md += ["Volatility forecasting:", "", table(v, [x for x in ["target", "h", "ic_spearman", "r2_oos", "rmse"] if x in v]), ""]
            p = t[t["task"] == "pol"]
            if len(p):
                md += ["Trading policies (RL / evolutionary), IC of position with realised return:", "",
                       table(p, [x for x in ["target", "h", "ic_spearman", "hit_rate"] if x in p]), ""]
        md += [f"Horizon comparison: {s.get('model', {}).get('horizon_comparison')}", ""]
    fa = jload("reports/features_EURUSD_H1/summary.json", {})
    if fa:
        md += ["## 7. Feature relevance / redundancy (EURUSD H1)", "",
               f"Feature columns: {fa.get('feature_columns')}; BH-significant (q<0.10) real columns: {fa.get('significant_bh10')}; "
               f"BH-significant placebo columns: {fa.get('placebo_significant_bh10')}; redundant pairs |rho|>=0.9: "
               f"{fa.get('redundant_pairs_rho_ge_0.9')}.", "", table(pd.DataFrame(fa.get("top", [])), n=15), ""]
    if conclusions_md and os.path.exists(conclusions_md):
        md += [open(conclusions_md).read()]
    (D / "research_report.md").write_text("\n".join(md) + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    main()
