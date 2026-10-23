"""Extrapolate measured throughput (results/benchmark.json) to the full directive-scale campaign and compare
execution environments. Writes docs/compute_plan.md and results/compute_estimate.json.

Prices are list prices looked up on 2026-10-21 (AWS c7a.16xlarge us-east-1 via Holori; Hetzner AX162-R via
WHTop) and must be re-checked before authorising spend."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fxlab.config import RESULTS_DIR, ROOT  # noqa: E402

BARS_PER_YEAR = {"M5": 74_000, "M15": 24_700, "H1": 6_200, "H4": 1_560}


def main():
    bm = json.loads((RESULTS_DIR / "benchmark.json").read_text())
    feat_s_per_bar = bm["features"]["total_seconds"] / bm["features"]["bars"]          # all 266 features, 1 core
    st = {k: v["seconds"] for k, v in bm["strategy_wf"].items() if "seconds" in v and "error" not in v}
    st_mean = sum(st.values()) / max(1, len(st))
    h1_bars_bench = 12 * BARS_PER_YEAR["H1"]
    model_s = bm["models"]["total_seconds"]  # all 69 models, one fit+predict on 10k x 20
    # ---- directive-scale plan (hierarchical, see docs/methodology.md)
    years = 19                 # 2007-2025 (2003-2025 would add ~20%)
    instruments = 40
    tfs = ["M5", "M15", "H1", "H4"]
    strategies = 66
    trials_screen, trials_opt = 6, 30
    est = {}
    # Phase 2/5 features: all features per instrument x timeframe (cached once)
    feat_core_h = sum(feat_s_per_bar * BARS_PER_YEAR[tf] * years for tf in tfs) * instruments / 3600
    est["features_core_hours"] = feat_core_h
    # Phase 5 strategy screening: cost scales ~linearly in bars; benchmark measured on 12y H1 with 6 trials
    per_bar = st_mean / h1_bars_bench
    screen_h = sum(per_bar * BARS_PER_YEAR[tf] * years for tf in tfs) * instruments * strategies / 3600
    est["strategy_screen_core_hours"] = screen_h
    # Phase 6 optimisation: top 10% re-run with 30 Optuna trials (5x a 6-trial run)
    est["strategy_optimise_core_hours"] = 0.10 * screen_h * trials_opt / trials_screen
    # Phase 7 models: 69 families x 3 tasks x 9 horizons x 10 instruments x 2 timeframes x ~8 folds,
    # each fold ~ measured single fit scaled to 30k rows (x3) - neural/RL dominate
    est["model_core_hours"] = model_s * 3 * 3 * 9 * 10 * 2 * 8 / 3600
    # Phase 8-10 robustness: cross-market, cost/latency, seeds, perturbation on survivors ~ 25% of screening
    est["robustness_core_hours"] = 0.25 * screen_h
    total = sum(est.values())
    est["total_core_hours"] = total
    envs = {
        "this sandbox (2 vCPU, 8 GB, no GPU)": {"cores": 2, "usd_per_hour": 0.0},
        "AWS c7a.16xlarge on-demand (64 vCPU, 128 GiB)": {"cores": 64, "usd_per_hour": 3.2845},
        "AWS c7a.16xlarge spot (64 vCPU, 128 GiB)": {"cores": 64, "usd_per_hour": 0.95},
        "Hetzner AX162-R dedicated (48 cores, 256 GB, EUR199/mo + EUR542 setup)": {"cores": 48, "usd_per_month": 215.0, "setup_usd": 585.0},
    }
    rows = []
    for name, e in envs.items():
        wall_h = total / (e["cores"] * 0.85)  # 85% parallel efficiency
        if "usd_per_hour" in e:
            cost = wall_h * e["usd_per_hour"]
        else:
            months = max(1, -(-wall_h // 720))
            cost = e["setup_usd"] + months * e["usd_per_month"]
        rows.append({"environment": name, "wall_clock_days": wall_h / 24, "est_cost_usd": cost})
    res = {"benchmark_inputs": {"feature_seconds_per_bar_all_features": feat_s_per_bar, "strategy_wf_seconds": st,
                                "models_all_families_seconds": model_s}, "core_hours": est, "environments": rows}
    (RESULTS_DIR / "compute_estimate.json").write_text(json.dumps(res, indent=1))
    md = ["# Compute plan and cost estimate", "",
          "Measured on this machine (`results/benchmark.json`), extrapolated linearly in bars. Scope: 40 instruments, "
          "4 intraday timeframes (M5, M15, H1, H4), 2007-2025, 66 strategy families (6-trial screen, 30-trial Optuna "
          "re-optimisation of the top 10%), 69 model families x 3 tasks x 9 horizons on 10 instruments x 2 timeframes, "
          "robustness passes on survivors.", "", "| phase | core-hours |", "|---|---|"]
    for k, v in est.items():
        md.append(f"| {k} | {v:,.0f} |")
    md += ["", "| environment | wall-clock (days) | estimated cost (USD) |", "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['environment']} | {r['wall_clock_days']:,.1f} | {r['est_cost_usd']:,.0f} |")
    md += ["", "Recommendation: CPU-bound workload (tree models, simulators, numba features); GPUs only help the "
           "neural/RL share. A single dedicated 48-core server or spot 64-vCPU instances with the existing "
           "checkpoint/resume orchestration are the most cost-effective options. **No spend has been incurred; "
           "any paid environment needs explicit authorisation.** Prices: AWS c7a.16xlarge us-east-1 on-demand "
           "$3.2845/h, spot ~$0.95/h (Holori, Oct 2026); Hetzner AX162-R EUR199/mo + EUR542 setup (WHTop listing, "
           "Feb 2026) - re-check before purchase."]
    (ROOT / "docs" / "compute_plan.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
