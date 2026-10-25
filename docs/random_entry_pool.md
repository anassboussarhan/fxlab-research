# Random entries, mean-reversion harvest with pooled robust selection (run 15)

Runs 11-14 found one property of the price path that the management of random entries can harvest: daily mean
reversion.
- The D1 averaging grid earned +0.23 Sharpe gross.
- Settings fitted to 2010-2018 keep +0.37 with fresh random entries.

But re-tuning every pair every six months mostly fitted noise: training +0.84, test +0.10. Run 15 attacks the
tuning noise directly. Entries stay strictly random: a coin-flip direction at a random time.

Development data only (2007-2018, EURUSD to 2022); 34 pairs; H1, H4, H6, H8, H12, D1. Run 40191195490, branch
`results15`.

## Design

**Candidate set.** A shared, deterministic set of 321 candidates, the same for every pair and timeframe
(`scripts/random_entry_pool.py`):
- one a-priori grid fixed in advance: 0.5 entries a day, an add every 1 ATR, 5 levels x1.5, take-profit 0.5 ATR,
  stop 3 ATR beyond the last level;
- 224 random grids;
- 96 random brackets.

Each candidate is ungated, or opens trades only in calm markets (ATR14/ATR100 <= q) or range-bound markets
(|close - SMA100| <= m ATR).

**Evaluation.** For every pair, fold (3-year training, 6-month test) and candidate:
- training Sharpe over 8 random-entry seeds;
- the out-of-sample expected-value stream over 16 new seeds, at full costs, half costs and zero costs (gross).

**Selection rules.** Declared before the run and applied using training data only. A test checks that the
selection is unchanged by test-window data.

| Rule | What it does |
|---|---|
| apriori | the fixed grid, no selection |
| per_pair | each pair's own best (the run 11-14 style) |
| pooled | one candidate for all pairs: best mean training Sharpe across pairs |
| pooled_top10 | the 10 best pooled candidates, averaged |
| pooled_filter | pooled, traded only where it was profitable in training |
| pooled_mr | pooled over, and traded on, the more mean-reverting half of the pairs (training variance ratio) |

**Tests.** 36 equal-weight portfolio tests (6 timeframes x 6 rules), Benjamini-Hochberg corrected, at full costs
(primary) and at half costs.

## Results

Portfolio Sharpe of the expected-value stream, full costs:

| Timeframe | apriori | per_pair | pooled | pooled_top10 | pooled_filter | pooled_mr |
|---|---|---|---|---|---|---|
| H1 | -3.16 | -0.95 | -2.02 | -2.45 | -2.04 | -1.94 |
| H4 | -1.25 | -0.51 | -0.51 | -0.31 | -0.47 | -1.38 |
| H6 | -0.97 | -0.15 | -0.73 | -0.24 | -0.69 | -1.24 |
| H8 | -0.60 | -0.11 | -0.76 | -0.15 | -0.75 | -1.18 |
| H12 | -0.35 | +0.09 | -0.48 | -0.14 | -0.47 | -0.52 |
| D1 | -0.02 | +0.05 | +0.25 | -0.23 | +0.25 | -0.24 |

Half costs:

| Timeframe | apriori | per_pair | pooled | pooled_top10 | pooled_filter | pooled_mr |
|---|---|---|---|---|---|---|
| H1 | -1.46 | -0.29 | -0.43 | -0.97 | -0.48 | -0.22 |
| H4 | -0.65 | -0.13 | -0.11 | -0.04 | -0.08 | -0.33 |
| H6 | -0.53 | +0.07 | -0.18 | +0.09 | -0.15 | -0.63 |
| H8 | -0.22 | +0.21 | -0.28 | +0.03 | -0.28 | -0.92 |
| H12 | -0.06 | +0.30 | -0.06 | -0.00 | -0.06 | -0.40 |
| D1 | +0.22 | +0.22 | +0.32 | -0.08 | +0.32 | -0.10 |

Gross (no costs):

| Timeframe | apriori | per_pair | pooled | pooled_top10 | pooled_filter | pooled_mr |
|---|---|---|---|---|---|---|
| H1 | +0.38 | +0.39 | **+1.03** | +0.57 | +0.99 | +0.81 |
| H4 | -0.11 | +0.31 | +0.24 | +0.15 | +0.24 | **+1.09** |
| H6 | -0.03 | +0.24 | +0.23 | +0.45 | +0.23 | +0.25 |
| H8 | +0.32 | +0.38 | +0.57 | +0.19 | +0.56 | -0.20 |
| H12 | +0.21 | +0.58 | -0.03 | +0.04 | -0.02 | -0.31 |
| D1 | +0.40 | +0.28 | +0.28 | +0.01 | +0.28 | -0.07 |

- **No rule wins after costs.** 0 of 36 tests survive Benjamini-Hochberg at full costs, and 0 at half costs. The
  largest t-statistics are 0.88 at full costs and 1.16 at half costs.
- **The best, D1 pooled at +0.25 (t 0.9), is fragile.** It was positive in only 3 of 9 years in 2010-2018. Most of
  its total comes from 2018 and from 2019-2022, when only EURUSD has data.
- **Before costs, the mean reversion is real and significant.**
  - The pooled grid at H1 makes +1.03 gross (t 3.7). Pooled over the most mean-reverting pairs, H4 makes +1.09
    (t 3.5).
  - Random-entry grids do harvest a genuine property of FX prices: small swings of about 0.2-0.4 ATR revert.
  - But costs are about three times that edge. Interpolating gross, half and full costs puts break-even at roughly
    one third of the cost model used here: H1 pooled about 0.34x, H4 pooled_mr about 0.38x, H8 pooled about 0.33x.
- **Pooling made the choice more stable, but not more profitable.**
  - The pooled choice was always a grid. The same range-gated tight grid (step 0.36 ATR, 7 levels x1.95,
    take-profit 0.14 ATR, stop 1.6 ATR) was chosen in 15-18 of 26 half-years at H4-D1.
  - Its mean training Sharpe (+0.55 to +1.35) still did not carry into the test windows.
- **Single traders can look better than the average.** Grids have many small wins and rare large basket losses.
  The median single trader is often positive even where the average over traders loses (H1 pooled_mr: median
  trader +1.06, expectancy -1.94). This is the "most grid traders win for years" trap. Only the expectancy counts.

## Conclusion

Random entries cannot win at retail costs on this data, even with robust, pooled parameter choice. The one
exploitable property, short-swing mean reversion, is significant before costs at H1-H4. Paying it out would need
trading costs of roughly a third of a normal ECN retail account (spread, slippage, commission and swap), which is
market-maker or prime-broker territory.
