# Every options structure tested: 92 structures, 745,383 combinations

This is the final options strategy study. It merges the corrected first and second batches (62 structures plus the
wheel, run 40329232195) with the third batch (30 more named structures, run 40334638224).
- Merge: run 40370344366, branch `options-strategies-all`. Tables are in `docs/options_strategies_all/`.
- Data and costs as in `docs/options_strategies.md`:
  - real DoltHub end-of-day quotes, 2019-2025, 106 underlyings plus stock portfolios;
  - fills at mid +- 0.75 x half-spread; USD 0.65 per contract;
  - the database's corrupted implied-volatility snapshots repaired from the quotes.
- Every structure ran at 14 / 30 / 45 days with:
  - 6 management rules: hold, 50% take-profit, 21-day exit, both, 50% take-profit with a 2x stop, and rolling when
    the short strike is hit;
  - 6 entry filters: none, VIX contango, high IV, IV above realised, no earnings, 200-day trend.
- That is 16.9 million position cycles.

## 1. Optimising across everything still does not survive out of sample

Each year from 2021, the best combination on the earlier years only was picked and traded that year:

| Family (selection by Sharpe) | Months | Mean / month | t | Sharpe | Worst cycle | Picks |
|---|---|---|---|---|---|---|
| Everything (all 92 structures, all underlyings) | 28 | +0.28% | 0.58 | 0.38 | -9.3% | single-stock 16-delta short puts and one put spread (CVS, AMD, AVGO) |
| Defined risk only | 35 | -0.07% | -0.18 | -0.11 | -9.3% | put spreads, iron albatross |
| Second and third batches only | 43 | +0.38% | 0.89 | 0.47 | -14.9% | rolled short puts, naked 16-delta calls |
| Stock portfolios | 59 | +0.68% | 1.57 | 0.71 | -5.2% | covered calls, protective puts on the equal-weight book |
| **Index ETFs only** | 35 | **+0.96%** | **2.60** | **1.52** | -4.0% | SPY covered strangle, 16-delta short puts, iron albatross |

- The in-sample winners had Sharpes of 17-21. The 42 extra structures did not displace them, and out of sample the
  selection gives a Sharpe of 0.38 (t 0.6).
- The only significant family is again index ETFs. Every pick was on SPY in rising years:
  - a covered strangle (beta 1.1);
  - short puts (beta 0.2-0.4);
  - an iron albatross.
  That is equity and short-volatility exposure.
- In-sample and out-of-sample Sharpes have a rank correlation of 0.70 across 123,063 combinations. The top
  in-sample tenth averaged 0.29 out of sample, against -1.08 for all.
- What persists is the *kind* of structure (equity-exposed, short premium on the index), not the specific best one.

## 2. Every structure

For SPY the best of 14 / 30 / 45 days is shown, held to expiry without a filter. "Alpha t" is the return beyond the
structure's exposure to SPY over the same cycle, at each maturity. Full table: `verdicts_by_structure.csv`;
grouping: `groups.csv`.

- **No structure has a significant alpha on SPY at two or more maturities.** The two synthetics are the exception,
  and their "alpha" is the financing rate:
  - a synthetic short earns the short-sale interest embedded in the forward;
  - a synthetic long pays it;
  - the benchmark is the stock's total return, not an excess return.
- **Significant (t > 2) at one maturity only** (12 structures, mostly at 30 days):

  | Structure | Alpha t at 14 / 30 / 45 days | Note |
  |---|---|---|
  | collar 30/30 | 0.3 / 3.5 / 0.7 | DIA 30 days t 3.0 |
  | put backspread | 0.4 / 3.5 / -1.4 | |
  | call diagonal | -0.2 / 2.8 (14 / 30 only) | |
  | bearish risk reversal | 0.9 / 2.6 / 0.7 | |
  | iron albatross | -0.1 / 2.5 / 0.1 | |
  | short strangle 16 | 0.1 / 2.5 / 0.2 | |
  | short call 16 | 0.8 / 2.2 / 0.4 | |
  | short put ladder | -1.1 / 2.1 / -1.0 | |
  | short call butterfly | 45 days 2.0 | |
  | bearish seagull | 45 days 2.0 | |
  | long put 30 | 30 days 2.0 | |
  | call ratio diagonal | 30 days 2.0 | |

  - With about 260 SPY maturity tests, about 6 above t 2 would be expected by chance. The 30-day ones also share
    the same SPY cycles.
  - Most of them lean bearish (long puts, short calls). That matches the second edges batch: selling skew lost in
    2019-2025.
  - None is confirmed at a neighbouring maturity.
- **Market exposure** (positive return because they hold equity risk; SPY beta 0.36-1.14; alpha about zero or
  negative), 16 structures:
  - covered call; buy-write;
  - collar 16/16; covered strangle; covered short straddle;
  - protective puts (two);
  - long 30-delta / ATM / deep in-the-money calls;
  - short 30-delta and ATM puts;
  - bullish risk reversal; bullish seagull;
  - call ZEBRA; put ratio calendar;
  - the wheel.

  On single stocks their median Sharpe is 0.2-0.6, with median alpha t between -1.6 and +0.6.
- **Positive on SPY with a small beta but no alpha** (20 structures):
  - bull call spread;
  - big lizard; jade lizard;
  - broken-wing call butterfly;
  - call calendar; double calendar; calendar strangle;
  - long call and put condors; bull call ladder;
  - iron condor 16/5; unbalanced iron condor;
  - put credit spreads; 16-delta short put; put ratio;
  - put Christmas tree; short put butterfly;
  - strap; long 30-delta call.

  On stocks almost all of them lose: median alpha t -2.4 to +0.3, and the equal-weight portfolio's alpha t is
  -9.6 to +1.1.
- **Lose or show nothing** (41 structures):
  - long butterflies and long iron condors / butterflies;
  - short iron butterfly and iron condor 30/16;
  - broken-wing put butterfly and condor; double butterfly; call Christmas tree;
  - bear put spread; call credit spread; call ratio; call backspread;
  - ladders (bear put, short call); short call and put condors;
  - every calendar and diagonal on stocks; put calendar; reverse calendars; put diagonal; double diagonal;
    call ratio calendar;
  - long straddles, strangles, strips, strap strangles, long guts; short guts; short straddle (unhedged);
  - 30-delta short call; covered put; synthetic put; put ZEBRA; reverse lizards.

  The worst on stocks: reverse calendars (median alpha t -3.5), calendars and diagonals (-2.4 to -2.7), and
  butterflies and condors (-1.8 to -2.2).

## 3. Management and filters

Change in Sharpe against holding to expiry (or against no filter), for the same combination:

| Rule | ETFs / portfolios | Stocks | Improved |
|---|---|---|---|
| Take profit at 50% | -0.15 | -0.10 | 30-33% |
| 50% take-profit + 2x stop | -0.51 | -0.37 | 22-24% |
| Close at 21 days | -1.16 | -1.15 | 9-11% |
| Both | -1.14 | -1.11 | 10% |
| **Roll when tested (out to the same maturity, at most twice)** | **-1.50** | **-1.59** | **8-9%** |

| Filter | Mean change | Improved |
|---|---|---|
| VIX contango | -0.06 | 51% |
| IV above its one-year median | -0.06 | 44% |
| No earnings before expiry | -0.05 | 45% |
| IV above 21-day realised | -0.06 | 47% |
| 200-day trend | -0.10 | 43% |

- **Rolling a tested position is the worst rule of all.** It turns a capped loss into repeated losses in a trending
  move.
- **Holding to expiry is the best default.** No filter helps on average.

## 4. Capital providers and the long-history check

- **Provider simulation of the year-by-year picks** (29 monthly starts, the published rules):
  - Options Funding (monthly fee) loses in every case: mean -$1,412 to -$2,533.
  - Imperial (one-time fee) has a mean of -$357 to +$1,045 and a median of -$357 (the fee) in every case.
- **The 1994-2018 model layer** (first-batch structures) matches the real 2019-2025 cycles with a median
  correlation of 0.88 (13,865 pairs; same-sign means in 81%).

## 5. Conclusion

- Across 92 structures, 106 underlyings and 745,383 combinations, no options structure has an edge on 2019-2025 real
  quotes beyond the exposure it carries:
  - equity beta (covered calls, short puts, the wheel, collars);
  - or the index variance premium.
- The index variance premium is clearest when delta-hedged (`docs/options_edges.md`, `docs/options_edges2.md`).
- Single-stock premium selling loses after spreads.
- The popular management rules, rolling included, lower risk-adjusted returns.
- Optimising over everything picks noise, which then fails out of sample.
