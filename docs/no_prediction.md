# Can trade management alone make money? (no-prediction tests)

Many retail traders say they win without predicting anything, through entries and exits, pyramiding, grids,
hedging or martingale sizing. This page tests that claim directly. Everything uses development data only:
2007-2018 (EURUSD to 2022), 14 instruments, H1/H4/D1. Costs are the platform's: quoted bid/ask plus slippage
plus commission plus swaps.

## Why it cannot work in theory

If price changes are unpredictable from the past (a martingale), the gain of any strategy is the sum of
position x price change, where each position is decided before its price change. This is a martingale
transform, and its expected value is zero. That holds whatever the entry timing, stops, targets, trailing,
pyramiding, averaging down or position sizing.

The one classical exception is unbounded doubling with unlimited capital and time. With a finite account, it
turns into a small probability of ruin.

Management changes the **shape** of the result distribution:
- the win rate;
- the skew;
- how often you have a good month.

It does not change the **sign** of the expectation. Costs then make the expectation negative. Management can
only amplify a real edge or ruin it, so an edge needs some genuine predictability first: drift, momentum, mean
reversion or a risk premium.

## Test 1: random entries x every exit / pyramid / martingale

Setup:
- `scripts/random_entry_test.py`, results in `results/reports/random_entry/`;
- coin-flip timing, about one entry opportunity per day;
- long-only and short-only runs averaged, so a pair's drift cancels;
- 12 seeds x 14 instruments x 3 timeframes = 504 runs per method;
- the trade engine replays the M1 path when a bar contains both the stop and the target.

| Management on random entries | Gross Sharpe | Net Sharpe | Runs net-positive | Win rate |
|---|---|---|---|---|
| fixed 20-pip stop, 1.5R target | -3.50 | -6.59 | 0% | 34% |
| 1.5 ATR stop, 1R target | -0.25 | -1.07 | 0% | 48% |
| 2 ATR stop, 2R target | -0.10 | -0.53 | 1% | 32% |
| 3 ATR stop, 3 ATR trailing, no target | -0.05 | -0.44 | 1% | 35% |
| 2x realised-vol stop, power target | -0.04 | -0.30 | 1% | 46% |
| structural stop, log target | -0.13 | -0.69 | 7% | 41% |
| quantile stop, piecewise target | -0.03 | -0.26 | 3% | 39% |
| 2 ATR stop, polynomial target | -0.18 | -0.82 | 0% | 52% |
| 2 ATR stop, 1.5R, 24-bar time exit | -0.11 | -0.72 | 0% | 41% |
| 2 ATR stop, 1.5R, vol-target sizing | -0.12 | -0.60 | 0% | 39% |
| 6 ATR catastrophic stop + 12-bar time exit | -0.02 | -0.71 | 2% | 47% |
| 2 ATR, breakeven at +1R, 3R target | -0.08 | -0.51 | 1% | 17% |
| half off at +1R, breakeven, 3 ATR trail | -0.12 | -0.63 | 0% | 30% |
| martingale: TP/SL 1 ATR, double after loss (max 5x) | +0.08 | -0.32 | 14% | 51% |
| pyramid: +1 unit per +1 ATR (max 4), out on 2 ATR give-back | -0.03 | -0.39 | 4% | 18% |

"Gross" means no spread, slippage, commission or swap at the fill. Stops and targets still trigger on bid/ask,
so stops tighter than about the spread lose even "gross". That is the fixed 20-pip row on metals and D1.

Findings:
- No exit rule has a positive gross expectation on random entries.
- The martingale's small positive gross (about +0.08) reflects slight short-horizon mean reversion. It is not
  paid for by costs: net -0.32.

## Test 2: hedged grid and zone recovery

These systems hold longs and shorts at the same time. Setup:
- `scripts/no_prediction_systems.py`, results in `results/reports/no_prediction/`;
- every leg of a locked hedge pays spread, slippage and commission;
- swaps are charged on both legs, as retail brokers do;
- "ruined" means equity fell to 50% or less at some point.

| System (18 grid / 12 zone configurations x 14 instruments x 3 TFs) | Gross Sharpe | Net Sharpe | Runs net-positive | Months positive | Worst month | Runs ruined |
|---|---|---|---|---|---|---|
| Hedged (bi-directional) grid, basket TP / stop-out | -0.08 | -0.90 | 4% | 39% | -12% | 15% |
| Zone recovery (hedging martingale), random start | +0.15 | -0.50 | 11% | 52% (up to 61%) | -98% | 34% |

Zone recovery shows the pattern behind many "I make money every month" stories: most months are positive,
then one month takes most of the account.

## What else the review found

The trade engine used to assume the **stop was hit first** whenever a bar's range contained both the stop and
the target. That is conservative, but it penalises tight stop/target exits; on EURUSD H4 it took the fixed
20-pip exit from -0.28 to -1.15 gross. The engine can now replay the M1 path to resolve those bars
(`simulate_trades(..., m1=...)`, `cfg["intrabar"] = "m1"`).

Run 3 (`ci/sweep3_config.json`, branch `results3`) re-runs, with this resolution on all 34 pairs at H1/H4/D1:
- the 29 trade-mode families;
- the joint parameters x entry x exit optimisation.

## Run 3 result: does the M1 replay change the sweep? (branch `results3`, run 40040589503)
11,322 experiments (all ok): the 29 trade-mode families with their own stops/targets and the joint optimisation
(8,704 streams, 762,429 candidates), H1/H4/D1, 34 pairs, with M1 replay of same-bar stop/target ambiguity.
* Joint streams, M1 replay vs stop-first (same family x pair x timeframe): median change 0.00, mean +0.005,
  correlation 0.98. Medians: D1 -0.19 (was -0.19), H4 -0.28 (-0.29), H1 -0.71 (-0.71). 0 BH/Holm survivors
  (best q = 0.23: lead-lag EURGBP D1 1.36, liquidity-filtered AUDJPY H4 1.25).
* Tight exits are chosen slightly more often (fixed 20-pip stop 5.7% -> 6.5% of folds; 1.5 ATR / 1R 5.9% ->
  6.1%).
* Trade-mode families with their own exits: median change 0.00 per timeframe; the largest family gains are
  intraday breakouts (opening range +0.12, London breakout +0.14), which remain negative.
The stop-first rule was not what kept the sweep negative.

## Where retail "winners" come from

- **Survivorship and short track records** of negatively skewed systems (grid, martingale, zone recovery).
  These win most months until one month erases them.
- **Real risk premia.**
  - Carry, which is tested here: negative over 2007-2018 after crashes.
  - Selling volatility, which needs options data this platform does not have.
- **Business income rather than trading edge:**
  - signals, courses and copy-trading fees;
  - prop-firm challenge fees;
  - introducing-broker rebates.
- **Genuine edges that retail costs or access rule out:**
  - market making / spread capture;
  - order-flow information;
  - latency.
