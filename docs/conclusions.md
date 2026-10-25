## 8. Scientific conclusions

All figures below are out-of-sample walk-forward results with costs: real quoted bid/ask, slippage, commission and financing. They come from `results/reports/*/summary.json` and `strategy_results.csv` / `model_results.csv`.

### 8.1 Headline
No strategy family, instrument, timeframe, risk configuration or portfolio construction produced a trading edge that survives realistic costs and data-snooping control. The pre-declared survival bar (OOS Sharpe > 0, BH-adjusted p < 0.10, DSR > 0.95, Sharpe > 0 at 2x costs) was met by **zero** of 1,107 strategy experiments, covering 5,278 parameter/risk configurations. The engineering system is complete and working; the research answer on this data is **negative**.

### 8.1b Round 2 (more timeframes, every feature, every entry type, carry) - see `docs/round2_report.md`
Across the whole program, 23,477 out-of-sample streams (5,882 strategy experiments on M5-D1 and event bars,
17,580 single-feature rules, 15 all-feature ML runs; about 876,000 configurations) give a median net Sharpe of
-0.26 and **zero** results significant after Benjamini-Hochberg or Holm, and zero meeting the survival bar.
Walk-forward entry-type selection, real interest-rate financing, carry, information-driven bars and a
cost-aware all-feature model each improve or explain parts of the picture, but none produces a significant
edge. The one consistent lead (Bayesian change-point family on H8/H12) failed its pre-registered robustness
check. The gross signals that do exist are smaller than retail trading costs.

### 8.1c Full cloud sweep (runs 39964535615 + 39979322451; branches `results`, `results2`)
34 instruments (2007-2018, EURUSD to 2022), 10 timeframes (M5-D1, tick and volatility bars), 97 strategy
families, 269 features, 69 model families. Merged program: **214,090 OOS streams** (40,818 strategy walk-forwards
incl. 8,704 joint-optimisation streams, 173,204 feature rules, 68 all-feature ML runs) from about **11.4 million
configurations**. 16 strategy and 21 model experiments failed (logged); 6,711 model experiments are prediction
evaluations, not return streams.
* Median OOS net Sharpe -0.32; 14.9% positive; **0** significant after BH (10%) or Holm (5%); **0** meet the
  survival bar; best-stream deflated Sharpe ~ 0. Strategy medians by timeframe: D1 -0.22, H8 -0.29, H4 -0.38,
  H1 -0.81, M15 -2.4, M5 -5.4 (costs dominate at high frequency).
* Cross-instrument family portfolios: 929 tests, median Sharpe -1.14, 7% positive, 0 significant (best: D1
  Bollinger / z-score mean reversion ~0.5, p ~0.07-0.09).
* Feature rules vs placebo columns: 0.86% of real rules exceed OOS Sharpe 0.5 against 0.20% of placebo rules -
  a fatter right tail, but nothing survives correction.
* **Joint optimisation of parameters x entry x exit per pair** (H1/H4/D1, 96 families x 34 pairs, ~762,000
  candidates simulated). Figures from run 10, after the trade-engine fix of 2026-10-24 (a pending limit/stop
  entry that filled mid-bar could be credited a take-profit, trailing or breakeven from that bar's earlier
  extremes); run 2 (pre-fix) in brackets: median OOS Sharpe -0.41 [-0.38] (D1 -0.23 [-0.19], H4 -0.34 [-0.29],
  H1 -0.71 [-0.71]), 16% positive [18%]; better than the standard walk-forward of the same family/pair/timeframe
  in 59% of cases [60%] (median +0.11 [+0.13] Sharpe, mostly by losing less: pullback limits were chosen in 46%
  of folds), and **0** BH/Holm survivors. The deflation-benchmark abstention traded only 4.5% of folds: the best
  in-sample combination rarely beat what luck produces with ~100 candidates.
* Round-4 families (famous traders, forum methods, esoteric): every family's median OOS Sharpe is negative
  (lunar -0.24, Turtle -0.53, Connors RSI2 -0.96, Raschke 80-20 -2.4, round numbers -3.2).
* Gotobi USDJPY remains the strongest consistent lead on the same 2010-2018 data: H1 1.05 (0.36 at 2x costs),
  M5 0.88, M15 0.87, M30 0.78; joint optimisation lowers it to 0.77 (more choices, more noise). It would not
  pass the program-wide correction; only 2019-2025 data or the 2026 holdout can confirm it.

### 8.1d Trade management without prediction - see `docs/no_prediction.md`
Random entries managed with each of the 13 exit configurations, pyramiding or martingale sizing (504
direction-neutral runs per method; 14 instruments x H1/H4/D1 x 12 seeds): no exit rule has a positive gross
expectation; net Sharpe -0.26 to -6.6, at most 14% of runs net-positive. Hedged grids (median net -0.90) and
zone recovery (net -0.50; 52-61% of months positive, worst month -98%, 34% of runs lose half the account) confirm
the theory: management reshapes the distribution (win rate, skew) but cannot create expectancy without
predictability. The trade engine's stop-first rule for bars holding both stop and target was replaced by an
optional M1 path replay. Run 3 re-tested the trade-mode families and the joint optimisation with it: per-stream
change median 0.00 (mean +0.005, correlation 0.98 with stop-first), still 0 BH/Holm survivors. Run 10 repeated
run 3 after the mid-bar-fill fix: joint streams change by a median of 0.00 and a mean of -0.03 to -0.06 Sharpe
(correlation 0.92); trade-mode families with market entries are unchanged; still 0 survivors.

### 8.1e Searching for a non-random projection - see `docs/projections.md`
Linear projections of the currency space (Lo-MacKinlay maximally predictable portfolio, Box-Tiao most
mean-reverting basket), D1/H4 walk-forward with 20 placebos each: no genuine basket has a robust directional
forecast (best gross Sharpe 0.46 vs placebo 0.08 +- 0.24, net -0.05; OOS R^2 ~ 0.001). A basket with gross
Sharpe 1.56 that beat every placebo turned out to be the triangular identity of the crosses (quote noise
inside the spread, net -11.6). Volatility is the non-random projection (HAR OOS R^2 0.29-0.48) but carries no
direction; the only directional one is the gotobi calendar effect, still unconfirmed on new data.

### 8.1f Projection lab: where the price is not random - see `docs/projection_lab.md`
20 mathematical projections (ordinal patterns, Markov contexts, simplex projection, directional-change time, cycle
phase, fractional differencing, clock, visibility graph, path signature, record times) x 34 pairs x 14 timeframes
(9,520 tests, BH-corrected, against sign-randomised surrogates): the price is NOT random at short horizons -
86% of M1 tests and 83% of M5 tests are significant (median t 25 / 15, hit rate ~51.5-52%; mostly short-term mean
reversion, lag-1 autocorrelation -0.01 to -0.04) - but the predictable move is 0.5-2% of the round-trip cost.
The structure fades with the timeframe (35% at H1, 4% at H4, 0.3% at D1) faster than moves grow relative to
costs; the best non-random test reaches 0.61 of its cost. 0 of 9,520 are tradable.

### 8.1g Feature predictability - see `docs/feature_predictability.md`
Out-of-sample information coefficient of every column of all 269 features (raw and |d20) with the next 1/6/24-bar
return, 34 pairs x 10 timeframes (857,202 tests), against 20 circular-shift nulls with t(19) p-values and BH:
2.8% of tests are significant - 17.6% at M5, 1.0% at H1, 0.4% at D1. 68 features are predictive in >= 5% of their
tests (trend, momentum and price-shape features, almost entirely at M5), 58 weakly, 142 rarely, 1 never. The
predicted move of significant tests is 2-10% of the round-trip cost at M5-H1; the rare significant tests at H8/D1
are about cost-sized, concentrated in EURSEK/EURNOK, and none survived as trading rules after program-wide
correction.

### 8.1h Harvesting the feature edge with optimised management - see `docs/edge_harvest.md`
Ridge composite of the top-30 features per fold, with every trade-management lever searched continuously by TPE
on a pseudo-out-of-sample slice of each training window:
- threshold, exit, sizing, pyramiding;
- passive execution;
- stop, target, trailing, breakeven, partial, time exit;
- entry order type.

Run 9: 34 pairs x M5-D1 x 3 horizons, 816 streams.
- Median training Sharpe +1.41 against a test median of -0.36 (rank correlation 0.14).
- 17% of streams positive; 0 BH survivors; best USDNOK D1 +1.07 (t 2.9, q = 1.0).
- Best equal-weight portfolio: D1 h = 6, +0.07 (t 0.26).
- D1 comes closest (median -0.07 to -0.17; SEK/NOK pairs +0.09, metals +0.19), but nothing is significant.

Optimising everything mostly fits noise.

### 8.1i Best management of random entries, fully optimised - see `docs/random_entry_opt.md`
Seven families with every parameter and the entry rate optimised by TPE per walk-forward fold, and tested with new
random entries (runs 11-14: 34 pairs x 13 timeframes x 7 families = 3,094 cells):
- bracket;
- martingale / anti-martingale;
- pyramid;
- averaging-down grid;
- zone recovery;
- hedged grid;
- "smart": limit/stop entries, trading-hour window, volatility and spread filters, stop-and-reverse, flat before
  rollover.

Results:
- Median net Sharpe -0.28; 16% of cells positive; **0** BH survivors (cells and portfolios); 0.6% of cells with
  t > 1.645 (5% by chance).
- Best: an averaging-down grid on D1 (median single-trader +0.10, 62% of pairs positive, no ruin). But its
  expectancy (average over random traders) is +0.02 and the 34-pair portfolio +0.29 (t 1.0, p 0.15). It is the
  best of 91 timeframe x family combinations and fails the survival bar.
- Martingale and zone recovery come close to zero only by risking the account: half the account lost in 79-100%
  and 9-38% of cells.
- Momentum-type management (pyramid, trailing) loses even before costs. Mean-reversion-type management is slightly
  positive before costs.
- Every family is worst at M1-M5.

A grid fill error found on the way: limit orders were filled at mid without paying the spread. It produced all
five apparent BH survivors (AUDNZD M5 grid +3.5); it was fixed and the grid re-run in run 14.

### 8.2 Strategies (H1, 14 instruments, 2007-2018; EURUSD to 2022)
* 924 family x instrument experiments: 8.4% had positive OOS net Sharpe; median OOS Sharpe -1.02.
* White's Reality Check p = 1.0 in both screens; Hansen SPA p = 0.95 and 0.999. Every BH-adjusted p-value = 1.0; max Deflated Sharpe = 0.00.
* Costs are not the whole story. Even at zero cost (diagnostic only), only 42% of default configurations were profitable; median zero-cost Sharpe was -0.07. Most rules carry no gross information at H1.
* Families by median OOS Sharpe: cross-asset (-0.35, 21% positive) > trend (-0.60) > geometry (-0.72) > mean reversion (-0.81) > ... > session (-1.70) > model-driven (-1.87). Model-driven strategies are positive before costs (median zero-cost Sharpe +0.14) but the worst after costs, because they trade frequently.
* Isolated good-looking results did not replicate across instruments. Weekend-gap fade: Sharpe 0.75 on AUDUSD and 0.58 on EURJPY, but positive on only 3 of 14 instruments. Volatility expansion: positive on 8 of 14, median 0.08. Best single results rest on 40-120 trades. They are treated as multiple-testing artefacts, not candidates.
* Timeframes (EURUSD): H4 is least bad (median -0.36, 14% positive), H1 -0.75, M15 -1.52. Higher frequency loses more to costs. Best M15 result (volatility expansion, 0.73) is not significant after correction (SPA p = 0.20, WRC p = 0.995).

### 8.3 Risk management, portfolio, hedging (OOS streams only)
* 11 stop/target/sizing designs (fixed, ATR, realised-vol, structural, quantile stops; affine, power, log, piecewise, polynomial targets; trailing; time exit; vol-target sizing; wide catastrophic stop + time exit) selected per fold on EURUSD: best 0.19, 3 of 61 positive, none positive at 2x costs. Risk shaping does not create an edge where the entry signal has none.
* Portfolios of the 20 best streams by trailing OOS Sharpe, re-selected monthly, lose money under every allocation tested (equal, inverse-vol, currency-netted; Sharpe -0.72 to -1.62). Fractional Kelly comes closest (-0.17 / -0.43). Out-of-sample performance does not persist.
* Min-variance hedging with the most-correlated pair reduced volatility in 29 of 30 streams but improved Sharpe in 0 of 30 (median Sharpe 0.27 -> 0.00). Hedge-leg costs and basis risk outweigh the variance reduction.

### 8.4 Prediction (EURUSD H1, 198 model runs, 2007-2022)
* **Direction is weakly but genuinely predictable at 1 hour.** Best AUC 0.539 (extremely randomised trees), with 92 of 171 classifier runs significant after Holm correction. It is stable in every year 2008-2022 (per-year AUC about 0.52-0.56 for the leading models). It decays fast: median AUC 0.528 at 1 hour, 0.514 at 5 hours, 0.499 at 20 hours.
* Complexity does not help. Linear/logistic models (about 0.535) match trees, ensembles and genetic programming. Neural nets (MLP 0.532 best, LSTM/GRU/TCN/Transformers lower) and time-series models (median 0.499) do not beat them.
* The edge is economically too small. A sign-trading proxy earns about 0.3 bp per hour gross, against about 1.35 bp modelled round-trip cost on EURUSD (median spread 0.4 pip; median |1h return| 4.7 bp).
* **Volatility is strongly predictable.** HAR-RV rank correlation with realised volatility is 0.30 / 0.49 / 0.66 at 1 / 5 / 20 hours (R^2_oos 0.34 at 20h); GARCH-family models are close behind. This is useful for position sizing and risk limits, not as a directional signal.
* RL and evolutionary trading policies (DQN, PPO, discrete SAC, LinUCB, differential evolution) learned nothing usable: correlation of position with realised return |IC| <= 0.024 at every horizon.

### 8.5 Data findings
* The GitHub mirror stores JPY-quoted pairs and metals at 1/100 of the true price. This was detected (USDJPY financing came out at about 1% per night) and corrected in the store; 132 affected results were discarded and re-run. A price-plausibility check is now part of the audit.
* The audit after correction: 0 inconsistent bars, 0 implausible price levels, no Saturday data. The last months of each series (June 2018) and the first months (March 2007, Nov 2010-Feb 2011 for silver) are partial.

## 9. Failures, exclusions and limitations
* **Data depth:** 2007-2018 for most instruments (EURUSD to 2022-07, gold 2011-05 to 2018-06, silver 2010-11 to 2018-06, AUDJPY to 2015). 2019-2025 is not available here, so the development period ends in 2018/2022, not 2025.
* **Instruments:** 14 of the 38 requested were acquired. USDMXN, USDZAR, USDTRY and USDCNH are not in the mirror. The other 20 (remaining crosses and SEK/NOK pairs) are available in the mirror but were not acquired locally (time budget).
* **Macro inputs:** round 2 added point-in-time 3-month rates for 8 currencies (FRED/OECD, via the web-fetch tool, rounded to 2 decimals), which unblocked the carry families and real overnight financing (`financing="rates"`). Yields, equity indices and VIX remain unreachable. Round-1 results keep the conservative flat swap (0.5 pip/night both sides).
* **2026 holdout:** not evaluated. The data does not exist in this environment, and candidates cannot be frozen because none survived.
* **Scope run locally:** round 1: H1 for all 14 instruments, M15/H4 for EURUSD, models on EURUSD H1 (horizons 1, 5, 20). Round 2 added H4/D1/H8 for all 14, H2 for 7, M30 for 4, M15 for 2, M5 for EURUSD (partial), tick/volatility bars for 4, H6/H12 for one family, the all-feature scan on 14 x H1/H4/D1 and all-feature ML on 7 x H1. Still not run: M1-M10, model horizons 2/3/10/30/50/100, the all-feature scan below H1, Optuna re-optimisation and cross-market model transfer (2 CPUs, 8 GB).
* **Excluded experiments:** S.ML.RL_POLICY on M15 (compute) and S.XA.CARRY_FILTER (data), recorded in the manifests. No experiment failed (0 of 1,305).
* **Simplifications:** a reduced TFT (no static covariates or multi-horizon decoder); discrete-action SAC; Kelly sizing on portfolio streams only; intrabar ambiguity resolved stop-first (conservative).
* **Statistical caveat:** the DSR uses the cross-sectional variance of OOS Sharpe across all configurations; with many strongly negative strategies this variance is large, which makes DSR conservative. Conclusions do not depend on it: BH-adjusted p-values, White RC and SPA all agree.

## 10. What would change the conclusion
1. Full 2003-2025 Dukascopy history and TrueFX as an independent second source (run `fxlab/data/acquire_dukascopy.py` on an unrestricted machine) to test the post-2018 regime and source sensitivity.
2. The full directive-scale campaign on rented CPU (about 3,500 core-hours; see `docs/compute_plan.md`). Not authorised; nothing has been spent.
3. Using the one real predictive signal differently. The 1-hour directional information and the strong volatility forecasts could be used as filters on lower-turnover entries, or for execution timing, rather than as stand-alone hourly trading signals.
