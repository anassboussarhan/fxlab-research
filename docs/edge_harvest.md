# Harvesting the feature edge with optimised trade management (run 9)

`docs/feature_predictability.md` shows that the features carry a little real information about the next
return. This page asks whether trade management can turn it into net profit:
- entry thresholds;
- sizing;
- pyramiding;
- holding rules;
- stops, targets and trailing;
- passive (limit) execution.

Development data only: 2007-2018, EURUSD to 2022. The 2026 holdout is untouched.

## Design

**Forecast.** In each walk-forward fold (3-year training, 6-month test, first test 2010):
- take the 30 features with the highest training IC;
- fit a ridge forecast of the next h-bar return, for h = 1, 6 and 24 bars.

**Optimisation without look-ahead.**
1. Inside each training window, the model is fitted on the first 75% and forecasts the last 25%.
2. TPE (Optuna, 80 trials per fold and horizon) maximises the daily net Sharpe on that pseudo-out-of-sample slice
   (at least 10 trades).
3. The model is refitted on the whole training window.
4. The chosen parameters trade the next 6 months.

Script: `scripts/edge_harvest_opt.py`.

| Lever | Search range |
|---|---|
| mode | signal-managed position, or stop/target-managed trades (trade engine) |
| entry threshold | 0.2-5 x round-trip cost |
| exit | hysteresis (0-0.95 x threshold) or hold 1-4h bars |
| sizing | unit, or proportional to the forecast (scale 1-5, capped at 1 x notional) |
| pyramiding | off, or add 0.1-1 unit per further bar of signal |
| execution | market, or passive limit (0-0.5 spread towards mid; market after 1-12 bars; strict trade-through fills) |
| stops mode | stop 0.3-8 ATR, target none / 0.3-6 R, trailing none / 0.5-8 ATR, breakeven none / 0.3-3 R, half take-profit none / 0.3-3 R, time exit none / 2-100 bars, entry market / limit / pullback limit / breakout stop (life 1-12 bars), 0.5% risk per trade |

**Scope and costs.**
- 34 pairs x 8 timeframes (M5-D1) x 3 horizons = **816 out-of-sample streams** (run 40119113962, branch
  `results9`).
- Costs as everywhere: quoted bid/ask, slippage, commission and swap.
- The trade engine includes the mid-bar fill fix (see `docs/no_prediction.md`, run 10).

**Run 8 is superseded.** `scripts/edge_harvest.py` selected among 86 fixed policies, but it ran before the
trade-engine fix, so it is not reported. Run 9 covers the same levers continuously.

## Results

Median out-of-sample net Sharpe across the 34 pairs; "training" is the pseudo-out-of-sample Sharpe the optimiser
saw.

| Timeframe | h = 1 | h = 6 | h = 24 | Streams net-positive | Median gross | Median training |
|---|---|---|---|---|---|---|
| M5 | +0.00 | -0.47 | -0.47 | 15% | -0.01 | +1.36 |
| M15 | -0.51 | -0.60 | -0.35 | 14% | -0.20 | +1.50 |
| M30 | -0.67 | -0.54 | -0.35 | 11% | -0.09 | +1.39 |
| H1 | -0.40 | -0.43 | -0.43 | 17% | -0.14 | +1.44 |
| H2 | -0.36 | -0.47 | -0.26 | 14% | -0.07 | +1.40 |
| H4 | -0.41 | -0.28 | -0.38 | 12% | -0.08 | +1.42 |
| H8 | -0.37 | -0.34 | -0.23 | 21% | -0.03 | +1.50 |
| D1 | -0.10 | -0.17 | -0.07 | 37% | +0.07 | +1.30 |

M5 at h = 1 has a median of 0.00 because in 59% of those streams the forecast almost never exceeded even the
lowest threshold. No candidate reached 10 trades in training, and the chosen rule did not trade out of sample.

### Overall

| Measure | Value |
|---|---|
| Median net Sharpe | -0.36 |
| Streams net-positive | 17% |
| Median gross Sharpe | -0.07 (41% of streams positive) |
| Median at 2x costs | -0.81 to -0.86 (22 of 816 streams positive) |
| Significant | **0** after Benjamini-Hochberg (q = 10%) |
| Forecast quality | median IC 0.011, hit rate 50.4% |

### Optimisation fits noise

- The median training Sharpe is +1.41 and the median test Sharpe -0.36: a gap of 1.78.
- Across streams, the rank correlation between training and test Sharpe is only 0.14.

### The tails are not what chance would give

- Only 0.9% of the streams have t > 1.645; with no effect at all, about 5% would. 31% have t < -1.645: costs
  dominate.
- The best stream is USDNOK D1 h = 6: Sharpe +1.07, t 2.92, p = 0.002. Among 816 tests, its BH q is 1.0.

### Where it comes closest: D1

- Medians are -0.10, -0.17 and -0.07 for the three horizons, and 37% of streams are positive.
- The SEK/NOK pairs at D1 have a median of +0.09 (8 of 12 positive). Gold and silver at D1: +0.19 (4 of 6).
  The other pairs at D1: -0.15.
- This is where the feature IC lab found cost-sized signals. Still, nothing is significant.

### Portfolios

Equal weight across the 34 pairs per timeframe and horizon:
- best: D1 h = 6, Sharpe +0.07 (t 0.26);
- all 816 streams together: -1.90 (t -7.4).

### What the optimiser chose

- **M5-H2:** signal-managed positions with passive limit execution, in 41-54% of folds.
- **H4-H8:** stop/target trades with pullback limit entries, in 32-38% of folds.
- **Stop/target mode:** market entries were rarely chosen (1-4%).

Cheaper execution was chosen wherever it was available, but saving part of the spread does not make up for a
forecast whose predicted move is a few percent of the cost.

### The parameter values chosen

These come from run 13 (40145929636, branch `results13`), which re-collected run 9 with the chosen values per
fold (`sweep_results/edge_harvest_opt/params.csv`). There are 509-570 folds per timeframe and horizon.

| Lever | M5-H1 | H2-D1 |
|---|---|---|
| stop/target mode instead of signal-managed | 5-47% of folds (mostly 12-35%) | 29-72% |
| entry threshold (x round-trip cost), median | 0.3-3.4 | 1.0-2.6 |
| signal mode: passive execution | 39-70% | 49-82% (highest at D1, ~80%) |
| signal mode: forecast-proportional sizing / pyramiding | 22-57% / 48-58% | 30-52% / 52-73% |
| stops mode: ATR stop, median (interquartile range) | 1.3-4.9 (0.8-6.5) | 1.4-3.3 (0.7-5.5) |
| stops mode: target used, median R | 26-55%, 0.4-2.7 | 46-57%, 1.0-2.0 |
| stops mode: trailing used, median ATR | 23-49%, 2.1-5.7 | 37-61%, 1.5-3.2 |
| stops mode: breakeven / partial / time exit used | 34-68% / 33-50% / 33-62% | 45-59% / 45-56% / 44-54% |
| stops mode entry: pullback limit / breakout stop / market | 21-66% / 12-46% / 1-22% | 50-72% / 15-31% / 0-10% |

Three things stand out:
- **No consistent choice.** Every optional lever is switched on in roughly half the folds, and the interquartile
  ranges of the ATR stop span most of the search range. That is what fitting noise looks like: no setting is
  reliably better, so the choice flips from fold to fold.
- **The only consistent preferences are cheaper fills:** passive or limit execution and pullback entries.
- **Why that is not enough:** cheaper fills reduce the cost but cannot make up for a forecast that is smaller
  than it.

## Conclusion

Optimising every lever lifts the training Sharpe to about +1.4, but not the out-of-sample one. The
feature edge is real but smaller than retail costs, and the extra freedom mostly fits noise. The only area worth
watching is D1 on the Scandinavian pairs and metals. It would need new data (2019-2025) to be tested honestly.
