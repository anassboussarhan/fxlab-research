# Break-even costs of the best edges, and what cheaper execution would change

Script: `scripts/breakeven_edges.py`. Output: `docs/breakeven_edges.csv`.
- Inputs: the per-event series of run 19 (flows) and run 22 (round 2), and the rule table of run 16 (random
  entries).
- **Leverage does not enter.** It multiplies the edge, the costs and the risk alike, so t-values and break-even
  costs per unit of position are the same at any leverage.

## Cost levels compared (round trip, basis points of notional)

The platform's cost model per round trip is 1.5 x the quoted spread + 0.2 pip + 0.7 bp: the spread, slippage of
0.25 x spread + 0.1 pip per side, and commission of 0.35 bp per side. The quoted spread at the trade times is
recovered from each test's modelled cost.

| Level | Cost | Who pays it |
|---|---|---|
| A | 1.5 x spread + 0.2 pip + 0.7 bp | this platform (conservative retail) |
| B | spread + 0.7 bp | raw-spread ECN account: IC Markets Raw / Pepperstone Razor charge USD 3.50 per side per lot = 0.35 bp per side; EUR/USD raw spreads are 0.1-0.2 pip today |
| C | spread + 0.4 bp | Interactive Brokers: 0.20 bp of trade value per order, quotes from 17 dealers, "as narrow as 1/10 pip" |
| D | 0.5 x spread + 0.1 bp | institutional / passive execution |

The spreads are those of this data (2007-2018) at the trade times. Spreads at liquid hours are tighter today, so B
and C are conservative for today, if the effects still exist.

## Results (net t-value with swap; BE = break-even round-trip cost)

| Edge | Gross bp / trade (t) | Cost A, bp | BE, bp | Net t: A | B (ECN) | C (IBKR) | D (institutional) |
|---|---|---|---|---|---|---|---|
| **Gotobi 03:00 -> fix (USD/JPY)** | 5.57 (4.7) | 2.12 | 5.53 (5.5 pips) | **2.88** | **3.39** | **3.65** | **4.24** |
| FOMC day, high-yielders | 21.7 (2.8) | 6.49 | 21.8 | 1.97 | 2.23 | 2.27 | 2.55 |
| FOMC day, dollar portfolio | 17.9 (2.7) | 7.44 | 17.9 | 1.60 | 1.96 | 2.00 | 2.39 |
| 2 days before FOMC, long foreign | 15.3 (2.5) | 3.08 | 14.9 | 1.91 | 2.06 | 2.10 | 2.27 |
| Month-end fix, 24 h | 6.36 (2.5) | 2.58 | 6.12 | 1.41 | 1.70 | 1.82 | 2.17 |
| Local hours, EUR/USD | 1.83 (2.8) | 1.62 | 1.83 (2.3 pips) | 0.31 | 0.93 | 1.39 | 2.22 |
| Gotobi reversal after the fix | 2.06 (2.2) | 2.10 | 2.06 | -0.04 | 0.59 | 0.90 | 1.64 |
| Around the clock, EUR/GBP/CHF (2 trades/day) | 1.70 (2.3) | 5.84 | 1.69 | -5.56 | -3.28 | -2.48 | 0.17 |
| Gold Asian hours | 2.39 (2.2) | 7.22 | 2.39 | -4.42 | -2.42 | -2.14 | 0.11 |
| Gold AM fix | 1.05 (3.6) | 4.03 | 1.05 | -10.2 | -6.35 | -5.33 | -0.54 |
| 4 p.m. fix reversal | 0.32 (2.9) | 2.73 | 0.32 | -21.2 | -14.3 | -11.7 | -3.55 |
| Silver fix | 3.64 (5.1) | 25.6 | 3.64 | -29.8 | -18.5 | -18.1 | -6.42 |

Benjamini-Hochberg survivors (q < 0.10) at each cost level, within each run's registered family:

| Level | Run 19 (17 flow / options tests) | Run 22 (11 round-2 tests) |
|---|---|---|
| A | variance selling, Gotobi | none |
| B | + FOMC high-yielders | none |
| C | + FOMC high-yielders, FOMC dollar portfolio | none |
| D | + FOMC (both), local hours EUR/USD, month-end 24 h | none |

### What cheaper costs cannot fix: the effects that faded

Net at cost B (and D), by period:

| Edge | 2007-2010 | 2011-2018/22 |
|---|---|---|
| **Gotobi** | +0.5 bp, t 0.2 (D: +1.5, t 0.6) | **+5.8 bp, t 4.5 (D: +6.8, t 5.2)** |
| FOMC, dollar portfolio | +41 bp, t 2.6 | +3.2 bp, t 0.5 (D: t 0.9) |
| FOMC, high-yielders | +61 bp, t 3.1 | +2.6 bp, t 0.3 (D: t 0.7) |
| 2 days before FOMC | +35 bp, t 1.9 | +5.3 bp, t 1.0 (D: t 1.2) |
| Month-end, 24 h | +15 bp, t 3.2 | +0.4 bp, t 0.2 (D: t 0.6) |
| Local hours, EUR/USD | +0.2 bp, t 0.1 | +0.8 bp, t 1.1 (**D: +1.6 bp, t 2.3**) |

- **Gotobi gets stronger with cheaper execution**, and it comes almost entirely from 2011-2018. 2007-2009 were
  the losing years.
- **The FOMC and month-end effects are 2007-2010 effects.** Lower costs push their full-sample t over 2, but since
  2011 they are near zero even at institutional costs.
- **EUR/USD local trading hours** (Breedon & Ranaldo) is the one effect that becomes interesting since 2011 at
  institutional cost only: +1.6 bp a day, t 2.3, at about 0.36 bp per round trip.

### Random entries and the mean-reversion harvest (runs 15-16)

Break-even as a share of this platform's cost (linear in cost, from the Sharpe at full, half and zero cost):

| Rule | Gross Sharpe (t) | Net Sharpe at A | Break-even share of cost A | Share of cost A for t = 2 |
|---|---|---|---|---|
| H1 pooled grid | 1.03 (3.65) | -2.02 | 34% | 15% |
| H4 pooled, mean-reverting pairs | 1.09 (3.48) | -1.38 | 44% | 19% |
| H1 pooled with filter | 0.99 (3.49) | -2.04 | 33% | 14% |
| M1 pooled, mean-reverting pairs | 1.16 (3.37) | -7.48 | 13% | 5% |
| M5 a priori | 0.96 (3.41) | -12.5 | 7% | 3% |
| H12 per pair | 0.58 (2.26) | +0.09 | 119% | 14% |

A raw-ECN account costs about 70-80% of cost A on liquid pairs, IBKR about 55%, institutional about 20-30%. The
grids would need institutional costs, and even then they are borderline.

### Where costs are not the problem

The monthly signals cost only 1-2 bp a month, but their gross t is below 2, so no cost level makes them
significant:
- economic momentum (gross t 1.44);
- dollar-factor momentum (1.21);
- weekly momentum (gross -13 bp).

The hourly direction model (AUC 0.539) earns 0.3 bp an hour gross against 1.35 bp cost, so it breaks even at about
22% of cost A.

## Short-timeframe predictability under cheaper costs

The 857,202 feature tests of run 7 (`docs/feature_predictability.md`) were re-costed at levels A-D:
- significant = BH p < 0.05 with the out-of-sample sign of training;
- the edge is half the top-minus-bottom decile spread of the next move, out of sample;
- the full table is `docs/breakeven_lowtf_features.csv`.

| TF | Significant tests | Median edge / cost: A (ours) | B (ECN) | C (IBKR) | D (institutional) | Share covering cost at D |
|---|---|---|---|---|---|---|
| M5 | 14,161 | 0.05 | 0.07 | 0.08 | 0.17 | 4% |
| M15 | 3,905 | 0.07 | 0.10 | 0.11 | 0.24 | 16% |
| M30 | 1,799 | 0.14 | 0.21 | 0.23 | 0.50 | 36% |
| H1 | 845 | 0.19 | 0.28 | 0.31 | 0.67 | 40% |
| H2 | 576 | 0.47 | 0.69 | 0.76 | 1.65 | 62% |
| H4 | 476 | 0.76 | 1.09 | 1.31 | 2.80 | 78% |
| H8 | 449 | 2.13 | 3.06 | 3.51 | 7.93 | 88% |
| D1 | 379 | 1.96 | 2.91 | 3.01 | 6.18 | 81% |

- **The structure at M1-M30 is strong statistically but tiny in price.** At M5 the median significant signal is
  5% of this platform's cost, 7% of a raw-ECN account's and 17% of institutional cost.
- The M5-H1 signals that would cover institutional cost on paper (2,184) are mostly 24-bar horizons on crosses:
  EURCHF, AUDCAD, CHFJPY, EURSEK.
- From H2 up, the typical significant signal covers cost on paper, but there are few of them (about 0.5% of
  tests), and they failed as trading rules after correction.
- **Decisive:** the optimised harvest of the same forecasts (`docs/edge_harvest.md`) had a median out-of-sample
  Sharpe **before any cost** of -0.01 to -0.20 at M5-H1. Removing costs entirely does not make it pay.
- The projection lab puts the predictable part at 0.5% (M1) to about 2% (M30-H1) of the cost; the best tests are at
  2-12%.
- The hourly direction model (AUC 0.539) earns about 0.3 bp an hour gross, which is roughly the whole
  institutional round-trip cost on EUR/USD (half of a 0.4 bp spread + 0.1 bp).
- Random-entry mean-reversion grids at institutional cost (about 25% of cost A), interpolated:
  - H1 pooled: Sharpe 0.27, t 0.95;
  - H4 on the mean-reverting pairs: Sharpe 0.47, t 1.5;
  - M1: Sharpe -1.0.
- **Who can harvest it:** whoever *earns* the spread instead of paying it, i.e. market makers and HFT firms with
  the fastest connections. The short-horizon mean reversion is largely the bid-ask bounce and inventory effect
  they are paid for. This matches the evidence that latency rank drives HFT performance (Baron et al., JFQA 2019)
  and that the few persistently profitable day traders win by fast, aggressive orders (Barber et al.).

## Conclusion

1. **Gotobi is the edge that cheaper execution improves:** net +4.0 bp per trade (t 3.4) with a raw-ECN account
   and +4.3 bp (t 3.7) at IBKR, against +3.4 bp here. It rests on 2011-2018.
2. **Variance selling** is unaffected by spot costs. It needs options.
3. **The FOMC-day, pre-FOMC and month-end effects** become significant with cheaper costs only because of
   2007-2010. Since 2011 they are near zero at every cost level.
4. **EUR/USD local trading hours** is a real but small edge since 2011 that only institutional execution could
   harvest.
5. **Fix reversals, the metals fixes, gold sessions, around-the-clock and random-entry grids** stay negative at
   retail-ECN and IBKR costs. The fix reversal and the silver fix stay negative even at institutional costs.
