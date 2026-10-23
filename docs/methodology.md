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
  quantile-threshold rule (60 rules per fold: 5 quantiles incl. the median sign rule x 6 holding periods x 2 directions, on every column and on its deviation from its own 20-bar EMA), chosen on the
  training window, abstaining when none is profitable after costs; reported numbers re-run through the
  vector engine. 20 noise + 20 time-shuffled placebo columns per instrument go through the same pipeline.
  Program-level statistics (`scripts/feature_scan_report.py`): BH/Holm across all streams, Hansen SPA and
  White RC per instrument, cross-instrument averages, and a meta walk-forward that trades the best streams
  of all *previous* OOS windows only.
* **Information-driven bars** (`data.store.event_bars`): tick-count, absolute-move and quoted-liquidity bars
  with an adaptive, causal threshold (previous-20-day average per 60 minutes), so bar duration stays near
  one hour despite the large growth in tick counts over 2007-2025. Prefix-invariance tested.

## Joint optimisation of parameters x entry x exit (third round)
`orchestration.runners.strategy_wf_joint`, enabled with `campaign.py plan --joint` (cloud: `sweep_job --joint`,
H1/H4/D1). Per instrument (one experiment = one family x pair x timeframe) the search space is the family's
parameter grid x 8 entry types (`JOINT_EXEC`: market, two pullback limits, stop breakout, limit at the close,
one-bar confirmation, London/NY-hours filter, scale-in) x 14 exits (`JOINT_RISK`: the family's own exit plus 13
stop/target/trail/breakeven/partial/time-exit/sizing configurations run in the trade engine). Model-driven
families keep fixed parameters (retraining cost); entry and exit are still searched.
* Every candidate is simulated once over the full history at base costs and stored compactly (cumulative log
  returns + trade entry positions). Training scores use only bars inside the training window (the trading day
  that straddles the train/test boundary is cut at the boundary).
* Per fold a TPE study (Optuna, fixed seed) is warm-started with every candidate simulated so far, re-scored on
  the current training window (earlier folds only ever used earlier data), then asks 16 new candidates. The best
  training candidate (daily Sharpe, >= 10 trades) trades the next test window. Test
  `test_joint_walk_forward_selection_uses_only_past_data`: replacing all data after a cut point leaves earlier
  folds' selections and OOS returns unchanged.
* Trade-engine entry types: pending limit / stop orders placed at the decision close, filled with a strict
  trade-through (limits) or at the worse of level and open plus slippage (stops); stops and targets are set from
  the actual fill. Orders of continuous-position families are cancelled when the signal changes; orders of
  one-bar (event) signals live until filled, expired (ttl) or reversed.
* Diagnostics, never the primary stream: (i) abstain (flat) when the best training Sharpe <= 0; (ii) abstain
  when it does not exceed the deflation benchmark, the expected maximum Sharpe of that many zero-skill trials
  given the cross-candidate variance (Bailey & Lopez de Prado). Also reported: selection frequency of every
  entry / exit, the configuration that would be carried forward (best on the last 3 years of dev data), and the
  number of distinct candidates simulated, which enters the program-wide DSR.
* Limitation: rate-differential financing is applied by the vector engine (market entry, own exit); the trade
  engine uses the flat swap charge.

## Speculative hypotheses
Esoteric features/strategies are tagged `speculative`; they pass through exactly the same tests and
multiple-testing corrections, and are reported separately.
