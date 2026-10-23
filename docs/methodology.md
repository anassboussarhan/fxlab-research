# Research protocol

## Timing and execution contract
* A bar is labelled by its open time (UTC) and is known only at its close.
* Strategies decide at the close of bar *t*; the simulator executes at the **open of bar t+1+latency**.
* Longs buy the ask and sell the bid; shorts the reverse. Quoted spreads come from the real tick data
  of the execution bar; slippage (fraction of spread + fixed pips), commission (bps of notional per
  side) and financing (per New-York 17:00 rollover, x3 on Wednesday) are added on top.
* Stops are evaluated against the bid (longs) / ask (shorts) extremes; a gap through a stop fills at the
  worse open; a gap through a target fills at the better open; when both stop and target lie inside one
  bar the stop is assumed to have been hit first (conservative).
* Zero-cost results exist only as a labelled diagnostic and are never the primary result.

## Leakage controls (all automated tests)
| control | where | test |
|---|---|---|
| point-in-time features | every feature | prefix-invariance on real data at two cut points (any use of bar > t changes earlier values) |
| point-in-time signals | every strategy | prefix-invariance of the `side` series |
| prediction causality | every model | predicting a prefix of the test block reproduces the same outputs; no test-block normalisation |
| scheduled refits | GARCH/HMM/GMM/HAR/learned features | refits anchored to absolute bar positions on trailing windows; filtered (never smoothed) posteriors |
| label overlap | labels return `label_end`; splits purge overlapping training labels and embargo the boundary | `tests/test_validation.py` |
| holdout | loaders refuse >= 2026-01-01; unlock needs frozen candidates + passphrase hash | `tests/test_validation.py::test_loader_refuses_holdout` |
| selection | parameters chosen on the training window of each fold only; stream selection for portfolios uses trailing OOS performance only | runners / portfolio scripts |

## Evaluation layers (kept separate, as required)
1. **Prediction** (`model_wf`): purged walk-forward per model family, horizon and task; AUC, PR-AUC, Brier,
   calibration error, log-loss, IC, R2_oos per fold and per year; Holm-adjusted test of AUC > 0.5.
2. **Trading** (`strategy_wf`): nested walk-forward (rolling 3-year train / 6-month test by default);
   parameters selected by training-window net Sharpe (min trade count); concatenated test windows form
   the OOS record; re-simulated under 1.5x/2x/3x costs and +1 bar latency; Sharpe CI by stationary
   block bootstrap; PSR; per-year / volatility-regime / session breakdowns; full trade logs.
3. **Campaign** (`scripts/analyze.py`): Deflated Sharpe with N = all configurations tried and the
   cross-sectional variance of OOS Sharpe; White's Reality Check and Hansen's SPA across all OOS streams;
   Benjamini-Hochberg and Holm adjustment of per-experiment p-values; family / instrument / timeframe /
   cost-sensitivity / regime tables.
4. **Portfolio, sizing, hedging** (`scripts/portfolio.py`, `scripts/hedging.py`): on OOS streams only.

## Pre-declared survival bar (Phase 10 -> 11)
A configuration is a *candidate* only if, out of sample: Sharpe > 0, BH-adjusted p < 0.10, DSR > 0.95,
Sharpe still > 0 at 2x costs, and it is not a single narrow peak (parameter stability recorded). Candidates
are then frozen (`scripts/freeze_candidates.py`) before any holdout access. A technically complete system
that finds **no** candidate is a valid, reportable outcome.

## Extended search (second research round)
All of the following use only 2007-2025 development data and the same timing/cost contract.
* **Entry order types** (`execution.simulator.order_backtest`): market; limit at the current price
  (`limit_touch`); limit k x ATR better than the decision close (`limit_pullback`); stop k x ATR beyond it
  (`stop_breakout`); each with a time-to-live after which the order is cancelled and the signal skipped.
  Limit fills need a strict trade-through (bid low < level for buys); stops fill at the worse of level and
  open plus slippage; exits are market orders. Two uses: (a) post-hoc re-execution of saved OOS positions
  (`scripts/entry_types.py`, descriptive only, adds 7x tests), (b) **walk-forward selection** of the entry
  type per fold on training data jointly with the parameters (`--exec grid`, `cfg["exec_grid"]`).
* **Cost-gated composite** (`scripts/composite.py`): families chosen on 2010-2013 gross OOS results only;
  equal-weight composite; entry/exit thresholds with hysteresis, volatility-percentile gate and entry type
  chosen walk-forward from 2014; abstention when no rule is profitable in training.
* **All-feature scan** (`scripts/feature_scan.py`): every output column of every feature becomes a
  quantile-threshold rule (48 rules per fold: 4 quantiles x 6 holding periods x 2 directions), chosen on the
  training window, abstaining when none is profitable after costs; reported numbers re-run through the
  vector engine. 20 noise + 20 time-shuffled placebo columns per instrument go through the same pipeline.
  Program-level statistics (`scripts/feature_scan_report.py`): BH/Holm across all streams, Hansen SPA and
  White RC per instrument, cross-instrument averages, and a meta walk-forward that trades the best streams
  of all *previous* OOS windows only.
* **Information-driven bars** (`data.store.event_bars`): tick-count, absolute-move and quoted-liquidity bars
  with an adaptive, causal threshold (previous-20-day average per 60 minutes), so bar duration stays near
  one hour despite the large growth in tick counts over 2007-2025. Prefix-invariance tested.

## Speculative hypotheses
Esoteric features/strategies are tagged `speculative`; they pass through exactly the same tests and
multiple-testing corrections, and are reported separately.
