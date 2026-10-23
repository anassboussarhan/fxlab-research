# Round 2: more timeframes, every feature, every entry type, carry

All results are out of sample (walk-forward, parameters chosen on training windows only), after retail ECN costs
(real quoted bid/ask, slippage 0.25 x spread + 0.1 pip, 0.35 bp commission per side, overnight financing), on
2007-2025 development data (actual coverage 2007-2018 for most pairs, EURUSD to 2022). The 2026 holdout was never
loaded. Numbers come from `results/reports/*` and the results database; the program-wide tables are produced by
`scripts/program_report.py` and `scripts/family_portfolios.py`.

## What was added
| addition | code | scale |
|---|---|---|
| Entry order types (market, limit at price, limit pullback k x ATR, stop breakout k x ATR, time-to-live) | `execution.simulator.order_backtest` | engine + tests |
| Entry type chosen walk-forward per fold, jointly with parameters | `--exec grid` | H1, 43 families x 7 instruments (301 runs) |
| Every feature column as a walk-forward rule (48 rules per fold, abstain if none profitable in training) + 40 placebo columns per instrument | `scripts/feature_scan.py` | 14 instruments x H1/H4/D1: 17,580 real + 1,680 placebo streams |
| All-features gradient boosting with a cost-aware trading threshold (purged inner validation) | `scripts/ml_allfeatures.py` | 7 instruments x H1 x horizons 4, 24 |
| Cost-gated composite of the best families (chosen on 2010-13 only) | `scripts/composite.py` | H1, 7 instruments |
| New timeframes | `data.store` | H2 (7 instr.), H6/H12 (check), H8 (14), M30 (4), M15 (2), M5 (EURUSD) |
| Information-driven bars with adaptive causal threshold (tick count, absolute move, quoted liquidity) | `data.store.event_bars` | TICK60 / VOLA60 x 4 instruments, all families |
| Point-in-time interest rates (8 currencies) and real overnight financing | `data.rates`, `financing="rates"` | all families D1 x 12 FX pairs; carry families D1/H4 |
| Carry families (pure, trend-filtered, volatility-gated, momentum-confirmed) + classic carry factor | `strategies.rules`, `scripts/carry_portfolio.py` | 12 pairs; 7-pair currency factor |
| Cross-instrument family portfolios + program-wide multiple testing | `scripts/family_portfolios.py`, `scripts/program_report.py` | every family x timeframe x execution x financing |

## Results
**Every feature as a rule (17,580 streams).** Median OOS net Sharpe -0.28 (H1), -0.20 (H4), -0.16 (D1); 12-23%
positive. 0 streams significant after Benjamini-Hochberg or Holm. Hansen SPA per instrument and timeframe:
lowest p = 0.46 of 42. The best real column is no better than the best placebo on D1 (0.955 vs 0.966). Trading
the best columns of all previous OOS windows forward ("trailing winners", top 1/5/20) loses on every timeframe
(Sharpe -0.13 to -0.89).

**Entry types.** Re-executing saved H1 positions: limit pullback 0.5 ATR / 6 bars is the best entry (median
-0.63 vs -1.01 at market). Choosing the entry type walk-forward per fold: median -0.91 -> -0.70, better than
market for 75% of streams; limit pullback 0.5 ATR chosen in 51% of folds. Still only 9.5% of streams positive.
Resting passive limit orders on the gross-positive families (strict trade-through fills): 0 of 84 positive,
median -1.94 (market -2.33). Fills arrive mostly when price moves through the order (adverse selection).

**All-features ML with cost-aware threshold (H1).** Gross Sharpe positive for all 7 instruments at h = 4 bars
(0.12 to 1.07), OOS rank IC 0.007-0.029. Net of costs negative in 12 of 14 runs; one run +0.10 (EURUSD h = 24)
and one never traded (USDCHF h = 24). The model abstained in 65-100% of folds: predicted moves rarely exceed the
round-trip cost.

**Composite.** Families selected on 2010-13 gross results only (3 ML families + London-fix reversal); threshold,
hysteresis, volatility gate and entry type chosen walk-forward from 2014: OOS net Sharpe -0.77 (95% CI -1.42 to
-0.28). From 2014 no configuration had a positive training Sharpe after costs.

**Timeframes (all strategy experiments in the database, both rounds).**

| timeframe | experiments | median OOS net Sharpe | share positive | best single stream |
|---|---|---|---|---|
| M5 (EURUSD) | 68 | -3.04 | 1.5% | 0.17 |
| M15 (EURUSD, USDJPY) | 138 | -1.49 | 3% | 0.67 |
| M30 | 273 | -1.15 | 6% | 0.61 |
| H1 (incl. entry grid, passive) | 1,364 | -0.95 | 8% | 0.60 |
| tick bars (TICK60) | 262 | -0.64 | 7% | 0.63 |
| volatility bars (VOLA60) | 262 | -0.61 | 7% | 0.40 |
| H2 | 406 | -0.47 | 14% | 0.64 |
| H4 | 865 | -0.40 | 18% | 0.74 |
| H8 | 784 | -0.26 | 21% | 1.28 |
| D1 (flat and real financing) | 1,478 | -0.22 | 24% | 0.76 |
| H6 / H12 (change-point check only) | 14 / 14 | 0.13 / 0.32 | 71% / 71% | 0.62 / 1.00 |

Lower frequency loses less to costs but has no more gross signal. Tick/volatility bars beat H1 time bars for 61%
of paired streams, never enough to turn positive.

**Carry.** Classic currency carry factor (long top-k / short bottom-k of 8 currencies by 3-month rate, monthly,
7 USD pairs, real spreads, rate-differential swaps minus 0.75% mark-up): net Sharpe -0.10 (k = 3) and +0.01
(k = 2) over 2007-03 to 2018-05, -21% / -33% in 2008, ~0 since. Pair carry families: median Sharpe -0.02 to
-0.49 (D1/H4), 17-50% of pairs positive. Real financing for all D1 families moves the median from -0.23 to -0.20.

**The one lead and its pre-registered check.** S.RG.CHANGE_POINT (Bayesian change-point, trade the fresh
regime's drift) on H8 was positive on 13 of 14 instruments (median 0.29, portfolio 0.31, p = 0.12, not
significant after correction). Pre-registered test (`results/preregistration/change_point_h8.md`): unseen H12
passed (portfolio 0.35), unseen H6 failed (portfolio -0.25), H8 with stop/target chosen walk-forward failed
(-0.41). Verdict: not robust; not a holdout candidate.

**Cross-instrument family portfolios.** 578 (family, timeframe, execution, financing) equal-weight portfolios:
median Sharpe -0.78, 5.4% positive, 0 significant after BH or Holm. Best: change point H12 0.35 (p = 0.09),
volatility expansion on tick bars 0.33, change point H8 0.31, change point H2 0.26. The change-point family is the
most frequent near the top, but it failed its pre-registered check and is not significant anywhere after
correction.

**Program-wide (scripts/program_report.py).** 23,477 out-of-sample streams (5,882 strategy experiments, 17,580
feature rules, 15 all-feature ML runs) covering about 876,000 configurations. Median OOS net Sharpe -0.26; 17.5%
positive. Benjamini-Hochberg (10%): 0. Holm (5%): 0. Streams meeting the pre-declared survival bar: 0. Best
stream: change point on XAUUSD H8, Sharpe 1.28 (p = 0.002 unadjusted, 1.0 adjusted), Deflated Sharpe ~ 0.

## Why the answer is negative, and what it implies
1. **Gross information exists but is smaller than retail cost.** About half of the H1 streams are profitable
   before costs, and boosted models predict the sign of the next hours with IC 0.01-0.04. The median break-even
   cost of the gross-positive streams is 0.43 bp per side, against 0.67 bp (EURUSD) to 2.2 bp (NZDUSD) retail.
   At 0.1 bp per side (inter-dealer level), 282 of 602 streams would clear costs.
2. **Waiting for better prices does not fix it.** Limit entries cut costs but the fills are adversely selected.
3. **Slower trading cuts costs but the gross signal also disappears.** D1/H8 are the least bad timeframes but
   no family is significant there either.
4. **The documented risk premium (carry) did not pay in 2007-2018** and carries crash risk (2008).

## Not done / limitations
* M15 ran on EURUSD/USDJPY and M5 on EURUSD only (68 of 69 families; S.ML.RL_POLICY not run: memory limits on
  this 8 GB machine allow one M5 process at a time and a session restart interrupted the last pass). The
  all-feature scan was not run below H1 (about 1 hour per instrument here). M5/M15 numbers in this table were
  completed after the program-wide statistics were computed; they only add negative streams.
* S.ML.RL_POLICY was not completed on H2, M30, event bars and the H1 entry grid (about 21 runs; 10-20 minutes
  each here); it learned nothing usable on every timeframe where it ran (H1, H4, D1, H8).
* Rates were transcribed from FRED table pages via a web-fetch tool and rounded to 2 decimals (see
  `docs/data_sources.md`).
* Data still ends in 2018 for most pairs; 2019-2025 needs an unrestricted network (Dukascopy) or uploaded files.
