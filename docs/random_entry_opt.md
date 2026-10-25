# Best management of random entries, fully optimised (runs 11-12)

The question: if every entry is random (coin-flip direction at a random time), which trade-management strategy
and which timeframe are best once every parameter is optimised? And does any of them make money out of sample?

Development data only: 2007-2018, EURUSD to 2022. 34 pairs, 13 timeframes. The 2026 holdout is untouched.

## What to expect, and why

The profit of any strategy is the sum of position x next price change. With random entries, the position is a
coin flip times a function of the path since entry. Management (stops, targets, trailing, adding, averaging down,
sizing) decides that function.

Management is therefore a bet on the serial dependence of the path:
- **Trailing stops and pyramids** keep or enlarge positions whose path has moved in their favour. They gain when
  moves persist (momentum).
- **Grids, martingales, zone recovery and tight targets** keep or enlarge losing positions. They gain when moves
  revert (mean reversion).
- **On a martingale price**, every such rule has zero expectancy (optional stopping), so the result is minus the
  costs.

Management can only extract dependence that a direct momentum or mean-reversion strategy could also extract, and
it does so less efficiently, because the random timing dilutes it. Those direct strategies, and the short-horizon
mean reversion found by the projection lab, were all far below costs.

The expectation:
- no family is profitable after costs;
- the least bad is where cost per unit of risk is smallest: slow timeframes, wide stops, few trades.

A rough figure for D1:
- costs of about 3 pips per trade (spread, slippage, commission and a few nights of swap);
- against about 160 pips of risk per trade;
- 10-20 trades a year.

That gives an expected annual Sharpe of about -0.05 to -0.1. The sampling noise of an 8-year Sharpe is about
+-0.35, so many pairs will look positive by chance.

## The seven families

| Family | Parameters searched (all at once) |
|---|---|
| bracket | ATR stop 0.25-10; target none / 0.2-8 R; trailing none / 0.25-10 ATR; breakeven none / 0.2-4 R; half take-profit none / 0.2-4 R; time exit none / 1-500 bars; notional or 0.5%-risk sizing |
| martingale | stop, target, time exit; size x 1.2-3 after a loss (martingale) or after a win (anti-martingale), up to 1-6 steps |
| pyramid | add a unit every 0.25-5 ATR in favour, up to 2-8 units; exit on a 0.5-10 ATR give-back; initial stop 0.5-10 ATR |
| grid | averaging down every 0.25-5 ATR, 2-10 levels, size x 1-2 per level; basket target 0.1-5 ATR beyond the average; basket stop 0.5-10 ATR beyond the last level |
| zone | zone recovery (hedging martingale): zone 0.25-5 ATR, target 0.25-5 ATR, up to 1-8 legs |
| hedged_grid | bi-directional grid with locked hedges, needs no entries: step 0.25-5 ATR, basket target 0.5-10 steps, stop-out 2-50 steps, 3-20 positions per side |
| smart | the bracket plus: market / limit / stop entry 0.05-2 ATR from the price, live 1-50 bars; trading-hour window; volatility regime filter (ATR14/ATR100 above or below 0.6-1.6); spread cap; stop-and-reverse; flat before the daily rollover (no swap) |

The entry rate (0.1 to 50 entries a day, at most one per bar) is optimised in every family except the hedged
grid.

## Protocol

**Walk-forward.** 3-year training, 6-month test, from 2010.
- In each fold, TPE (Optuna) maximises the mean daily net Sharpe over 6 random-entry sequences on the training
  window: 60 trials, 90 for smart, at least 30 trades per sequence.
- The same 6 sequences are used for every candidate (common random numbers).
- The best parameters then trade the test window with 16 *new* random sequences.

**Measures.**
- **Single trader:** the median over the 16 test sequences of the out-of-sample Sharpe, i.e. what one trader using
  the strategy would have seen.
- **Expected-value stream:** the average over the 16 sequences, which is the strategy's expectancy with most of
  the entry noise removed. Its one-sided t-statistic, Benjamini-Hochberg corrected over all pair x timeframe x
  family cells.
- **Portfolio:** equal weight across pairs.
- **Tail risk** of the expected stream: maximum drawdown, and ruin (half the account lost at some point).

**Hindsight check.** TPE (120 trials, 180 for smart) on the whole sample, then the best settings are re-run with
16 fresh entry sequences on the same data.
- If the in-sample Sharpe survives, the price path has structure that management exploits.
- If it collapses, the optimiser fitted lucky entries.

**Gross.** The same strategies at mid prices with no costs, as a diagnostic only.

**Execution.**
- Quoted bid/ask; slippage 0.25 x spread + 0.1 pip; commission 0.35 bp per side; swap 0.5 pip per night on every
  open position, on both legs of a hedge.
- Stop-first inside a bar. On M5 and slower, bars that hold both the stop and the target of a bracket, and grid
  bars with any level in range, are replayed on the M1 path.
- For pending orders that fill mid-bar, the rest of the bar is replayed from the fill minute. Nothing (target,
  partial, trailing or breakeven) uses the fill bar's earlier extremes.
- Grid limit orders trade on their own side of the quote: a long basket's adds fill when the ask trades below the
  level, its take-profit when the bid reaches it (mirrored for shorts). A gap through the basket stop exits at the
  open.
- Tests: `tests/test_random_entry_opt.py` checks for no gain on zero-cost random walks. A deliberate fill-bar
  look-ahead shows up there as z = +54. Further tests check that a grid cannot trade quote noise inside the spread
  and that a gap through its stop exits at the open.

**Runs.**
- Run 11 (40139466942, branch `results11`): the first six families.
- Run 12 (40141201668, branch `results12`): smart, merged with run 11 into one ranking.
- Run 13 (40145929636, branch `results13`): re-collect with tail risk and year-by-year results.
- Run 14 (40152143260, branch **`results14`**, the final numbers): the grid re-run after the fill fix described
  below. It supersedes the 442 grid cells of run 11; the other 2,652 cells are identical in runs 12-14.

Script: `scripts/random_entry_opt.py`; report: `scripts/random_entry_opt_report.py`.

## The grid fill error found and fixed

In the first full results, 5 of the 3,094 cells passed Benjamini-Hochberg. All 5 were averaging-down grids on
wide-spread crosses at M1-H1. The top one was AUDNZD M5: expected-stream Sharpe 3.5, gross 8.8, about 1,400
baskets every six months.

The cause was the grid simulation, not the market. Its resting limit orders (the adds and the basket
take-profit) were triggered and filled at mid, so they never paid the spread. Price noise inside the spread could
even fill them.

After the fix (buy limits wait for the ask, sell limits for the bid), the five cells became:

| Cell | Expected-stream Sharpe before | After |
|---|---|---|
| AUDNZD M5 | +3.46 | -0.25 |
| AUDNZD M1 | +3.15 | -1.10 |
| NZDCAD M5 | +1.78 | -0.82 |
| EURSEK H1 | +1.37 | +0.31 |
| AUDNZD M15 | +1.37 | -1.16 |

All are now insignificant. The grid's median single-trader Sharpe changed from -0.20 to -0.35 at M1, -0.11 to
-0.21 at M5, -0.06 to -0.20 at M15, and +0.01 to -0.03 at H6. D1 moved from +0.07 to +0.10, because it was
re-optimised with the correct fills. With unchanged parameters, the fix changes only costs, not gross results;
the re-optimised parameters differ, so the gross figures moved slightly. The other kernels already traded on the
correct side of the quote.

## Results (3,094 cells: 34 pairs x 13 timeframes x 7 families)

| Measure | Value |
|---|---|
| Median single-trader Sharpe | -0.28 |
| Cells positive | 16% |
| Median gross Sharpe | +0.02 |
| Median training Sharpe | +0.36 |
| Significant | **0** after Benjamini-Hochberg (q = 10%), cells and portfolios alike |
| Expected-value streams with t > 1.645 | 0.6% (about 5% would by pure chance) |
| Expected-value streams with t < -1.645 | 48% |
| Hindsight check (all cells) | +0.21 in sample, -0.04 with fresh entries |

Median single-trader out-of-sample Sharpe, by timeframe and family (net of costs):

| Timeframe | bracket | grid | hedged grid | martingale | pyramid | smart | zone |
|---|---|---|---|---|---|---|---|
| M1 | -0.83 | -0.35 | -9.81 | -0.38 | -1.32 | -0.59 | -1.10 |
| M5 | -0.60 | -0.21 | -2.33 | -0.24 | -0.56 | -0.48 | -0.66 |
| M15 | -0.48 | -0.20 | -0.96 | -0.13 | -0.45 | -0.50 | -0.23 |
| M30 | -0.42 | -0.14 | -0.72 | -0.15 | -0.45 | -0.46 | -0.16 |
| H1 | -0.49 | -0.12 | -0.67 | -0.07 | -0.39 | -0.41 | -0.13 |
| H2 | -0.31 | -0.10 | -0.57 | -0.12 | -0.40 | -0.34 | -0.08 |
| H4 | -0.31 | -0.10 | -0.37 | -0.01 | -0.42 | -0.32 | -0.04 |
| H6 | -0.35 | -0.03 | -0.45 | -0.02 | -0.36 | -0.19 | -0.02 |
| H8 | -0.36 | -0.06 | -0.46 | -0.03 | -0.42 | -0.24 | -0.04 |
| H12 | -0.26 | -0.02 | -0.52 | -0.03 | -0.33 | -0.33 | -0.00 |
| D1 | -0.36 | **+0.10** | -0.78 | -0.08 | -0.48 | -0.24 | -0.06 |
| TICK60 | -0.46 | -0.17 | -0.42 | -0.09 | -0.44 | -0.41 | -0.09 |
| VOLA60 | -0.41 | -0.15 | -0.61 | -0.07 | -0.46 | -0.49 | -0.10 |

By family, over all timeframes:

| Family | Median net | Cells positive | Median gross | Hindsight: in sample / fresh entries | Ruin (median over timeframes) | Median max drawdown |
|---|---|---|---|---|---|---|
| grid | -0.13 | 29% | +0.05 | +0.47 / +0.18 | 0% | -4% |
| martingale | -0.10 | 30% | +0.05 | +0.27 / -0.06 | **88%** | **-203%** |
| zone | -0.11 | 24% | +0.22 | +0.20 / -0.04 | 24% | -27% |
| smart | -0.37 | 10% | -0.03 | +0.27 / -0.03 | 12% | -23% |
| bracket | -0.39 | 12% | -0.04 | +0.07 / -0.15 | 12% | -32% |
| pyramid | -0.46 | 4% | -0.15 | -0.03 / -0.26 | 0% | -22% |
| hedged grid | -0.69 | 6% | -0.06 | +0.32 / +0.32 (no entries) | 3% | -16% |

"Ruin" means the expected-value stream lost half the account at some point, through tail events or slow cost
bleed. Returns are summed fractions of starting equity, so a drawdown below -100% means the account was wiped out.

### The best strategy and timeframe: an averaging-down grid on D1 bars

Settings chosen (medians over pairs x folds):
- about one entry every three days (0.34 a day);
- an add every 1.1 ATR against the position, up to 9 levels, each 1.77x the previous, so the deep levels carry
  most of the size;
- take profit 0.5 ATR beyond the average entry;
- basket stop 2.8 ATR beyond the last level, about 11.5 ATR from the first entry.

The total position is capped at 1x equity.

| Out of sample, 2010-2018 (EURUSD to 2022) | Value |
|---|---|
| Median single-trader Sharpe | +0.10 (gross +0.23); 62% of pairs positive |
| Expected-value stream (mean over 16 random traders) | median **+0.02** (median t 0.05) |
| Equal-weight portfolio of 34 pairs | Sharpe +0.29, t 1.0, p 0.15 (BH q = 1.0); about 0.2% a year at this exposure |
| Tail risk | no ruin in any pair; median max drawdown -4% (worst -13%); median skew -0.2 |
| Years positive (portfolio) | 6 of 9 in 2010-2018 (2013-2015 negative); 2019-2022 is EURUSD only |
| Hindsight check | +0.60 in sample, +0.37 with fresh entries |

What this means:
- **The grid is the best of the 91 timeframe x family combinations, but not a demonstrated edge.**
  - The strategy's expectancy (the average over random traders) is about zero, and the 34-pair portfolio is not
    significant.
  - Some of the D1 grid's lead is selection. Each timeframe x family median has a standard error of about 0.06
    (the spread across pairs is 0.28). Picking the best of 91 combinations inflates it by roughly two to three
    standard errors, about 0.15.
  - The single-trader median (+0.10) is higher than the expectancy (+0.02) because averaging down has many small
    wins and few large losses.
- **The hindsight check finds real but unexploitable structure.** Settings fitted to the whole of 2010-2018 keep
  +0.37 with fresh entries, so daily FX prices in that period did have mean reversion that a grid can pick up.
  Settings chosen on the past three years did not capture it in the next six months.
- **It fails the pre-declared survival bar.** It is not a candidate for the 2026 holdout.

### Best by timeframe, and what the alternatives cost

- **Every timeframe, counting ruin:** the grid is the best family that never lost half the account. Its medians
  run from -0.35 at M1 to -0.02 / -0.03 at H12 / H6, and +0.10 at D1.
- **Without the ruin filter:** martingale (H1, H4, H8) and zone recovery (H6, H12) come within 0.01-0.07 of zero,
  still negative. Martingale loses half the account in 79-100% of the cells at every timeframe; the optimiser
  chose "increase after a loss" in about two-thirds of folds. Zone recovery loses half the account in 9-38% of
  the cells, depending on the timeframe.
- **Momentum-type management loses even before costs:** pyramid (gross -0.15) and trailing brackets.
- **Mean-reversion-type management is slightly positive before costs:** grid, zone, martingale. That fits the
  weak short-horizon mean reversion found by the projection lab, which is smaller than the spread.
- **The smart-entry levers do not help** (-0.37 against -0.39 for the plain bracket). Limit entries were chosen in
  40-53% of folds and stop entries in 22-33%. The trading-hour window, flat before rollover (23-39%),
  stop-and-reverse (45-72%) and the volatility filter were each on in about half the folds: no consistent choice,
  the signature of fitting noise.
- **At M1-M5 the bracket's optimiser avoided targets, trailing and time exits** (in 72-97% of folds) and chose
  wide stops: trading as rarely as possible is the best a random entry can do against costs.
- **Cost is the main gradient.** The median over all families is -0.83 at M1, -0.48 at M5, about -0.3 at
  M15-H1, and -0.15 to -0.23 from H4 to D1.

## Conclusion

With random entries, no trade-management strategy makes money after costs on 2007-2018 data. That holds with
every parameter optimised, every timeframe from M1 to D1, and limit/stop entries, sessions, filters,
stop-and-reverse, martingale, pyramiding, grids and hedges included.

The least bad is an averaging-down grid on daily bars, whose expectancy is about zero with low drawdown. The
styles that look close to zero by Sharpe (martingale, zone recovery) do so by risking the account.

This confirms the optional-stopping argument at the top of this page. Management reshapes the distribution of
outcomes; it only extracts serial dependence, and in FX that dependence (mean reversion) is too small to pay the
spread.
