# Cloud sweep: what it runs

GitHub Actions on the public repo (free runners). Development data only (2007-2025; the mirror covers
2007-2018, EURUSD to 2022). The 2026 holdout is never loaded.

## Run 1: `sweep.yml` (ci/sweep_config.json)

| Job | Matrix | What it does |
|---|---|---|
| plan | 1 | Builds the job matrices from the config. |
| data | 34 pairs | Gets Dukascopy ticks from the FX-Data mirror, builds M1 bid/ask bars, audits quality and writes closes per timeframe. M1 data is cached, so it is not downloaded again. |
| panel | 1 | Merges the closes of all 34 pairs into the cross-instrument panel used by the multi-asset families. |
| sweep_hi | 34 pairs x {H1, H2, H4, H8, D1, TICK60} = 204 | See "one sweep job" below. |
| sweep_lo | 34 pairs x {M30, M15, M5, VOLA60} = 136 | Same, on the lower timeframes; starts after sweep_hi. |
| models | 34 pairs | Every model family on H1. Tasks: direction classification, volatility, policy. Horizons: 1, 5 and 20 bars. Purged walk-forward. |
| collect | 1 | Merges every job, runs the program-wide statistics and publishes summaries to the `results` branch. |

**One sweep job** (one pair x one timeframe):

1. Every strategy family whose timeframe scope includes this timeframe goes through the walk-forward:
   - 3-year training window, 6-month test window;
   - parameters chosen on training data only;
   - net of spread, slippage, commission and financing;
   - re-run at 1.5x, 2x and 3x costs and with one bar of latency;
   - zero-cost diagnostic.
2. On H4 and D1 for FX pairs, overnight financing uses point-in-time interest-rate differentials.
3. On H1 only:
   - the entry-type grid: the 8 entry types, chosen per fold;
   - the all-feature LightGBM model at horizons 4 and 24, which is cost-aware.
4. On M30 to D1: the all-feature scan.
   - Every feature column (269 features) becomes quantile rules plus a feature-momentum variant (x minus its EWM over 20 bars).
   - Each rule goes through the walk-forward, alongside placebo columns.

**collect** produces:

- `program_report`: BH and Holm over every stream of the program, and the deflated Sharpe with the program-wide number of configurations;
- `family_portfolios`: cross-instrument portfolios per family;
- `feature_scan_report`;
- `joint_report`, when joint experiments exist;
- coverage tables, data-coverage tables and job-log tails.

## Run 2: `sweep2.yml` (ci/sweep2_config.json)

It has the same jobs and the same matrices, except that models are off. Its own concurrency group means it can be queued while run 1 is still going.

- **(a) Round-4 families on all 10 timeframes of run 1, standard walk-forward:**
  - famous-trader systems: Turtle, Williams VBO, Crabel NR, Holy Grail, Raschke 80-20, Connors RSI2, TTM Squeeze, TD Sequential, Alligator;
  - forum methods: three ducks, news straddle;
  - round numbers, gold fix;
  - lunar, Mercury retrograde.
- **(b) JOINT optimisation on H1, H4 and D1, for all 97 families and all 34 pairs** (`--joint`). The search space for each pair and timeframe is:
  - **strategy parameters:** the family's own grid. Model-driven families keep fixed parameters because retraining is expensive, so only entry and exit are tuned for them;
  - **entry, 8 options:**
    - market at the next open;
    - limit pullback 0.25 ATR (lives 3 bars);
    - limit pullback 0.5 ATR (6 bars);
    - stop breakout 0.25 ATR (3 bars);
    - limit at the signal close (3 bars);
    - wait for one confirming close;
    - London/NY hours only;
    - scale-in (half at market, half on a pullback);
  - **exit, 14 options:**
    - the family's own exit (signal flat or reversal, or its default stop and target);
    - fixed 20-pip stop with a 1.5R target;
    - ATR stops at 1.5, 2 and 3 with targets of 1R, 2R or none (with a trailing stop);
    - realised-volatility stop with a power-law target;
    - structural stop with a log target;
    - quantile stop with a piecewise target;
    - ATR stop with a polynomial target;
    - 24-bar time exit;
    - volatility-targeted sizing;
    - catastrophic 6-ATR stop plus a 12-bar time exit;
    - breakeven after +1R with a 3R target;
    - half off at +1R, then breakeven, then a 3-ATR trail.
  - **search, per walk-forward fold:**
    - a TPE (Optuna) study on the training window only;
    - it is warm-started with every candidate simulated so far, each re-scored on this fold's training window;
    - each fold adds 16 new trials;
    - the best training candidate (daily Sharpe, at least 10 trades) trades the next six months.
    - A test checks that later data cannot change earlier folds' choices.
  - **reported per pair:**
    - the OOS net stream and the stress tests;
    - how often each entry and exit was chosen;
    - the configuration that would be carried forward: the best on the last three years of dev data;
    - two abstention diagnostics: flat when the best training Sharpe is 0 or less, or when it is below the deflation benchmark (the expected maximum Sharpe of that many zero-skill trials);
    - the number of distinct candidates simulated. Every one of them counts in the program-wide multiple-testing correction.
- **collect** waits for run 1, merges both runs and publishes to the `results2` branch.

## Limitations

- Rate-differential financing applies to the vector (market-entry, own-exit) engine. The trade engine charges the flat swap.
- A joint stream's selection searches about 100 to 200 candidates per pair, timeframe and family, so its OOS Sharpe has to clear the program-wide BH and Holm corrections and the DSR before it counts.

## Follow-up runs

Each follow-up run has its own config (`ci/sweepN_config.json`), trigger file (`ci/RUNN`), concurrency group and
results branch (`resultsN`). Development data only (2007-2018, EURUSD to 2022), 34 pairs.

| Run | GitHub run | What it does | Write-up |
|---|---|---|---|
| 3 | 40040589503 | trade-mode families and the joint optimisation with M1 replay of same-bar stop/target ambiguity (H1/H4/D1) | `docs/no_prediction.md` |
| 4 + 5 | 40089114354, 40089996931 | projection lab, waves 1 and 2 (20 projections x 14 timeframes) | `docs/projection_lab.md` |
| 6 + 7 | 40098535811, 40110483807 | feature predictability lab (out-of-sample IC of every feature column, 10 timeframes) | `docs/feature_predictability.md` |
| 8 | 40117414725 | edge harvest with 86 fixed policies; superseded (ran before the mid-bar-fill fix of the trade engine) | - |
| 9 | 40119113962 | edge harvest with every management lever optimised by TPE (M5-D1, 3 horizons) | `docs/edge_harvest.md` |
| 10 | 40119113917 | run 3 repeated after the mid-bar-fill fix | `docs/no_prediction.md` |
| 11 | 40139466942 | optimised management of RANDOM entries: bracket, martingale, pyramid, grid, zone recovery, hedged grid (13 timeframes) | `docs/random_entry_opt.md` |
| 12 | 40141201668 | the "smart" execution family for random entries (limit/stop entries, sessions, filters, stop-and-reverse, flat before rollover); merges run 11 into one ranking | `docs/random_entry_opt.md` |
| 13 | 40145929636 | collect only: re-aggregates runs 9, 11 and 12 with the chosen parameter values (edge harvest) and tail risk / year-by-year results (random entries) | `docs/edge_harvest.md`, `docs/random_entry_opt.md` |
| 14 | 40152143260 | re-run of the random-entry grid after the limit-fill fix (limit orders pay the spread); merges runs 11-12 - the final random-entry ranking | `docs/random_entry_opt.md` |
| 15 | 40191195490 | random entries, mean-reversion harvest with pooled robust selection: 321 shared candidates x 34 pairs x H1-D1, six pre-declared selection rules, full / half / zero costs | `docs/random_entry_pool.md` |
| 16 | 40195407249 | run 15 extended to M1, M5, M15, M30, H2, tick and volatility bars; merges run 15 (78 tests over 13 timeframes) | `docs/random_entry_pool.md` |
| 17 | 40205212178 | flows and options lab: 17 registered tests of published flow, calendar and option-implied effects (Gotobi, month-end fix, local hours, around the clock, FOMC days, fix reversal, EVZ variance, VRP signal, VIX-filtered carry) on the 9 USD pairs | `docs/flows_lab.md` |
| 18 | 40206036189 | macro lab: economic momentum, output gap, Taylor-rule fundamentals and elastic net, 9 currencies vs USD, monthly. **No macro data was fetched (FRED timed out on every request), so nothing was tested** | `docs/macro_lab.md` |
| 19 | 40207062071 | run 17 repeated with swap on overnight trades, EVZ download fallbacks, random-date tests and random entry timing inside the event windows | `docs/flows_lab.md` |
| 20 | 40211584679 | run 18 repeated with the macro series from DBnomics (OECD MEI / IMF IFS mirrors; all 50 series fetched) | `docs/macro_lab.md` |
| 21 | 40212676021 | collect only: re-scores run 20 with the macro-report fix (a constant elastic-net forecast is held flat, not ranked in a fixed order), records every fit, publishes the inputs | `docs/macro_lab.md` |
| 22 | 40213570596 | round-2 literature tests: silver / gold fixes, gold Asian vs Western hours, gold seasonality, dollar carry, dollar- and carry-factor momentum, weekly momentum (11 registered tests) | `docs/lit2_lab.md` |
