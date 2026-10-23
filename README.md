# fxlab — FX & precious-metals intraday research platform

A reproducible research system for discovering (or failing to discover) robust intraday edges in FX
and metals: real bid/ask tick data, a tested catalog of features / strategy families / model
families, a cost-aware execution simulator, leakage-safe validation and a resumable experiment
orchestrator. Engineering completeness and research completeness are reported separately in
`docs/preflight_audit.md` and `results/reports/`.

## Install

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.lock          # exact versions used to build and test (CPU-only torch wheel works)
export POLARS_MAX_THREADS=1               # optional: leave a core free on small machines
```

## Data

```bash
# 1) reachable-everywhere mirror (Dukascopy ticks on GitHub; 2007-2018, EURUSD to 2022)
python -m fxlab.data.acquire_fxdata_git              # resumable; writes data/m1/<PAIR>/<PAIR>_<YEAR>.parquet
# 2) on a machine with normal internet: extend to 2003-2025 directly from Dukascopy (same format)
python -m fxlab.data.acquire_dukascopy --pairs EURUSD GBPUSD XAUUSD --start 2019-01-01 --end 2025-12-31
# audit + inventory (actual first/last observation per instrument)
python -c "from fxlab.data.quality import run_audit, inventory; inventory(run_audit())"
```

Every acquired pair-year is logged in `data/manifests/acquisition.jsonl` with the source commit, raw/clean
tick counts, drop reasons, first/last timestamp and the parquet SHA-256 (provenance + dataset versioning).
M1 bars carry bid & ask OHLC, quote count, up/down mid-quote ticks, spread mean/min/max and quoted
liquidity. Higher timeframes (M2…D1, D1 on the 17:00 New-York rollover), tick-count, liquidity,
volatility and range bars are built from M1 only — nothing is interpolated.

## The 2026 holdout

Development data ends 2025-12-31. Every loader refuses timestamps >= 2026-01-01 unless
`results/frozen_candidates.json` exists (written once by `scripts/freeze_candidates.py`, which refuses to
overwrite) **and** `FXLAB_HOLDOUT_UNLOCK` holds the passphrase whose SHA-256 the authoriser recorded there.

## Tests

```bash
pytest -q tests/                # ~25 min on 2 CPUs; writes results/catalog_test_status.json
python scripts/export_catalog.py  # -> results/catalog.json, docs/catalog_{features,strategies,models}.md
```

* every feature: prefix-invariance (lookahead) test on real EURUSD H1, determinism, non-degeneracy;
* every strategy: prefix-invariance of signals, long/short capability, end-to-end execution;
* every model: fit/predict contract, prediction causality, planted-signal learnability;
* execution simulator: bid/ask fills, gaps, stop-before-target ambiguity rule, trailing/time stops,
  commission, financing, rollover calendar, latency;
* validation: purging, embargo, nested walk-forward, holdout guard, PSR/DSR, White RC, Hansen SPA,
  multiple-testing corrections; orchestration: checkpoint/resume and failure capture.

## Running research

```bash
python scripts/benchmark.py                                   # throughput on this machine -> results/benchmark.json
python scripts/campaign.py plan --name screen_h1 --kind strategy --symbols EURUSD GBPUSD XAUUSD --tfs H1
python scripts/campaign.py run  --name screen_h1 --workers 2 --time-budget-min 110   # rerun to resume
python scripts/campaign.py plan --name models_h1 --kind model --symbols EURUSD --tfs H1 --horizons 1 5 20
python scripts/campaign.py run  --name models_h1
python scripts/analyze.py --name screen_h1                    # DSR, White RC, SPA, BH/Holm, comparisons
```

Strategy experiments use nested walk-forward (default: 3-year rolling train, 6-month test; parameters chosen
on the training window only) with realistic costs, then re-simulate the selected configurations under
2x/3x costs and +1 bar latency. Model experiments evaluate the *prediction* problem (AUC, Brier,
calibration, IC per fold and per year) separately from trading.

## Layout

```
fxlab/data          acquisition (mirror + direct), store (PIT loaders, resampling, alt bars), quality audit
fxlab/features      266 feature methods in 18 families (registry-driven)
fxlab/strategies    67 strategy families (rules + model-driven)
fxlab/models        69 model families (linear ... RL)
fxlab/execution     costs, bid/ask simulators, risk (stops, targets, sizing, exposure, hedging)
fxlab/validation    splits (walk-forward, purged k-fold, nested), metrics, statistics
fxlab/orchestration campaign DB (DuckDB), checkpoint/resume, runners
scripts/            benchmark, campaign, analyze, export_catalog, preflight, freeze_candidates
docs/               data-source comparison, catalogs, preflight audit, compute plan
```
