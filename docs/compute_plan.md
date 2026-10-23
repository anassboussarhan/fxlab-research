# Compute plan and cost estimate

Measured on this machine (`results/benchmark.json`), extrapolated linearly in bars. Scope: 40 instruments, 4 intraday timeframes (M5, M15, H1, H4), 2007-2025, 66 strategy families (6-trial screen, 30-trial Optuna re-optimisation of the top 10%), 69 model families x 3 tasks x 9 horizons on 10 instruments x 2 timeframes, robustness passes on survivors.

| phase | core-hours |
|---|---|
| features_core_hours | 680 |
| strategy_screen_core_hours | 652 |
| strategy_optimise_core_hours | 326 |
| model_core_hours | 1,654 |
| robustness_core_hours | 163 |
| total_core_hours | 3,474 |

| environment | wall-clock (days) | estimated cost (USD) |
|---|---|---|
| this sandbox (2 vCPU, 8 GB, no GPU) | 85.1 | 0 |
| AWS c7a.16xlarge on-demand (64 vCPU, 128 GiB) | 2.7 | 210 |
| AWS c7a.16xlarge spot (64 vCPU, 128 GiB) | 2.7 | 61 |
| Hetzner AX162-R dedicated (48 cores, 256 GB, EUR199/mo + EUR542 setup) | 3.5 | 800 |

Recommendation: CPU-bound workload (tree models, simulators, numba features); GPUs only help the neural/RL share. A single dedicated 48-core server or spot 64-vCPU instances with the existing checkpoint/resume orchestration are the most cost-effective options. **No spend has been incurred; any paid environment needs explicit authorisation.** Prices: AWS c7a.16xlarge us-east-1 on-demand $3.2845/h, spot ~$0.95/h (Holori, Oct 2026); Hetzner AX162-R EUR199/mo + EUR542 setup (WHTop listing, Feb 2026) - re-check before purchase.
