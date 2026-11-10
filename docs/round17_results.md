# Round 17 results: regimes, harder (2026-11-10)

Rules: `docs/round17_preregistration.md`, with addendum 1 (implementation) and addendum 2 (a timing flaw and its
fix). Each was committed before the run it governs.
- **Code:**
  - libraries: `scripts/sweep15.py --daily` and `scripts/intraday17.py`;
  - the close-marked part C: `scripts/long_lab.py: cont_run_close`;
  - data: `scripts/r17_data.py`; engine: `scripts/regime17.py`;
  - tests: `tests/test_regime17.py` (19 pass);
  - follow-ups: `scripts/r17_explore.py`.
- **Runs:**
  - libraries: 42677782642 and 42690296819 (index CFDs), and 42709504619 (close-marked part C);
  - data: 42711158127;
  - engine: 42711912215, which replaces 42697499365.
- **Outputs:** branches `round17` and `round17-data`; tables in `docs/round17/`.

## The answer

**Nothing works, at any timeframe or switching speed.**
- **What was tested:**
  - 93 pre-registered tests: nine method families at daily, weekly and monthly switching over four libraries, and
    regimes per instrument for 33 instruments;
  - 486 timing tests of options and intraday-state features.
- **Result:**
  - 0 tests count;
  - none passes the correction for multiple tests;
  - none beats plain "pick the best past strategy" (the static benchmark) by t >= 1.645.
- **The surface:** of 21,247 settings, 0.5% reach t >= 1.645 on 2019-2025. Pure luck would give about 5%. Most
  settings lose, because switching costs and losing strategies dominate.

**Answers to the specific questions**
- **More timeframes.** Adding the intraday rules (H4, H1, M30, M15; 3,648 configurations, both directions) changes
  nothing. The libraries' best past strategies lose on 2019-2025 at daily and weekly switching (Sharpe -0.8 to
  -1.0).
- **Faster switching hurts.** Median 2019-2025 t by switching speed in the two book tiers:

  | Switching | Median t |
  |---|---|
  | Daily | -2.3 to -3.8 |
  | Weekly | -2.1 |
  | Monthly | -1.1 to -1.9 |

  Changing strategy often costs more than any regime information is worth.
- **Regimes per instrument** (each of 33 instruments choosing its own strategy from its own state): the best
  method has t 0.82 (Hedge, weekly). The methods chosen on 2011-2018 lose (t -1.36 and -1.55).
- **Adaptive learners and meta-models:**
  - Hedge with forgetting, which tracks the best strategy as it changes: median t -0.62;
  - change-point resets: -1.36;
  - pooled boosting across all strategies and states: -2.10.
- **Number of regimes:** no K from 2 to 8 is significantly better than no regimes, at any speed or tier.
- **Options and intraday state** (VIX family, CME euro / yen implied volatility and risk reversals, 5-minute
  volatility, jumps, autocorrelation, Asian-session share): 0 of 486 long-only timing tests count. Only 3.5% beat
  holding the style by t >= 1.645, fewer than chance.

## The timing flaw, found and fixed (addendum 2)

**What looked like an edge.** The first engine run (42697499365) had one close miss, in both book tiers: a daily
switch on 5-day cross-sectional dispersion (terciles).
- 2019-2025: Sharpe 0.76 (t 2.00) and 0.71 (t 1.79).
- It beat static selection by t 3.2 and 3.0, and beat its placebos.
- It failed only the multiple-testing correction.

**The check that exposed it.** With the feature lagged one more day, the Sharpe ratio fell to -3.2 and -2.5. A
genuine daily effect weakens with an extra day's lag; it does not reverse that sharply.

**The cause.**
- The 1,698 feature-tercile strategies of round 15 had their daily P&L marked from one 08:00 London fill to the
  next.
- The regime features are taken at the 17:00 New York close.
- So a daily decision already "saw" 14 of the 24 hours it was supposed to predict.
- Not affected: the rule strategies, the nine styles and the intraday library, which are all marked at the closes.
- Round 16 had the same overlap, but only about 2.6% of a month. Its null result stands, as noted there.

**The fix.**
- Those strategies were re-marked close to close: the same trades, `cont_run_close`.
- Every test was re-run with the rules unchanged.
- The dispersion split then has a Sharpe of -3.0 at lag 0, the same as with one or two extra days of lag
  (`docs/round17/dispersion_check_corrected.json`).
- Its gross gain was the look-ahead; its daily switching costs of about 0.1 volatility units a day remained.
- Every number below is from the corrected run. The first run's results are superseded.

## 1. The libraries

| | Configurations | Of which |
|---|---|---|
| BOOK-CFG (FX book, 28 crosses) | 7,788 | daily engine: 4,140 (D1 1,450, W1 1,412, MN 1,278); intraday: 3,648 (912 each at H4 / H1 / M30 / M15) |
| BOOK-UNIT | 3,098 units | unit x direction x engine, averaged over settings |
| FACTOR | 9 styles | carry, 12- and 3-month trend, EWMAC, weekly and daily reversal, weekly dip-buying, Donchian 20, the dollar |
| PAIR-UNIT | 46,200 | 33 instruments x rule unit x direction (both engines) |
| Placebo libraries | 900 book configurations, 6,600 instrument units | random-signal strategies priced the same way |

**Features:** 45 market-wide plus 11 per-instrument, all computed daily.
- **Coverage:** VIX3M, GVZ and OVX use Yahoo history before CBOE's files begin. EVZ is gone. CME IV and risk
  reversals cover 3,909 days from 2010-06. The intraday features come from 5-minute bars of every instrument.
- **Data gap:** the M1 store lacks AUDCAD 2018 (also true in round 15).

## 2. Results by tier and switching speed

The static benchmark B1 holds the best past strategies, with no regimes. The "best family" is the regime family with
the highest 2019-2025 t, which is information only, since it was not chosen in advance. The "pick" is the family
with the best 2011-2018 t, which is what a trader would have chosen.

| Tier | Switching | B1 Sharpe 2019-25 | Best family (t, Sharpe) | Its gain over B1, t | Pick on 2011-18 | Pick's t 2011-18 | Pick's t 2019-25 |
|---|---|---|---|---|---|---|---|
| FACTOR | daily | 0.18 | momentum (0.59, 0.24) | 0.21 | one feature (EPU) | 0.98 | 0.30 |
| FACTOR | weekly | 0.30 | clusters (0.77, 0.28) | -0.05 | one feature (carry trailing) | 0.58 | 0.61 |
| FACTOR | monthly | 0.51 | HMM, 4 states, options set (1.77, 0.58) | 0.48 | one feature (RV change) | 1.07 | 1.26 |
| BOOK-UNIT | daily | -0.94 | clusters (-1.74, -0.68) | 0.70 | one feature (10y-2y slope) | 2.00 | -2.47 |
| BOOK-UNIT | weekly | -0.77 | ridge (-1.20, -0.45) | 0.73 | ridge, intraday set | 1.99 | -1.20 |
| BOOK-UNIT | monthly | -0.78 | clusters (-0.20, -0.07) | 0.65 | one feature (RV21) | 2.43 | -0.44 |
| BOOK-CFG | daily | -0.99 | clusters (-1.31, -0.50) | 0.10 | one feature (slope) | 1.39 | -2.92 |
| BOOK-CFG | weekly | -0.79 | ridge (-0.91, -0.35) | 1.01 | ridge, intraday set | 0.77 | -0.91 |
| BOOK-CFG | monthly | 0.25 | clusters, intraday set (1.25, 0.45) | 1.07 | one feature (gold 63-day) | 1.67 | 0.06 |
| PAIR-UNIT | weekly | 0.13 | Hedge (0.82, 0.33) | 0.76 | change-point | 0.42 | -1.36 |
| PAIR-UNIT | monthly | -0.21 | one feature (0.00, 0.00) | 0.28 | momentum | 0.27 | -1.55 |

**Reading the table**
- **Only one chosen setting reached t >= 1.645:** the HMM on the nine styles, monthly, which is round 16's result
  again.
  - It fails the correction: p = 0.039 against a first threshold of 0.0011.
  - It does not beat static selection (t 0.48).
  - Its Sharpe is below its placebo's 95th percentile.
- **In the larger libraries, the 2011-2018 pick lost afterwards** in 7 of 8 cases. Book-tier picks with 2011-2018
  t of up to 2.4 ended between -2.9 and +0.06.

**Every family's median 2019-2025 t, over the 11 tier / speed combinations:**

| Family | Median t |
|---|---|
| Single-feature splits | -0.01 |
| Clusters | -0.20 |
| Hedge | -0.62 |
| Change-point resets | -1.36 |
| Strategy momentum | -1.55 |
| HMM | -1.56 |
| Ridge | -1.91 |
| Pooled boosting | -2.10 |
| Performance filter | -5.31 |

The full table of 93 tests is `docs/round17/chosen_all.csv`; the surface is `docs/round17/surface_summary.csv`.

**Number of regimes (the best 2011-2018 setting for each K, its 2019-2025 t)**

| | K=2 | K=3 | K=4 | K=6 | K=8 |
|---|---|---|---|---|---|
| FACTOR, monthly, clusters | 0.67 | 1.29 | 1.11 | 1.29 | 0.81 |
| FACTOR, monthly, HMM | 0.89 | 1.10 | 1.77 | | |
| BOOK-CFG, monthly, clusters | 1.25 | 0.61 | 0.61 | -0.54 | -0.65 |
| BOOK-CFG, daily, clusters | -1.31 | -3.87 | -2.20 | -2.09 | -2.52 |
| BOOK-UNIT, daily, HMM | -4.37 | -3.87 | -3.58 | | |

All 18 rows are in `docs/round17/k_curve.csv`; none is significant.

## 3. Options and intraday-state timing (long-only)

- **The test:** each of the nine styles, at each speed, was timed by each OPT, CME and INTRA feature (486 tests).
  It counts only if it beats holding the style at the same average weight.
- **Result:** 0 count. The smallest p is 0.0005, against a first BH threshold of 0.0002.
- **The best rows reduce exposure to styles that lose on their own.** Daily reversal timed by the FX volatility
  premium: difference t 3.27, but the timed rule itself lost (t -0.74). That is not a tradeable edge.

## Reproducibility note

FACTOR results reproduce exactly between the two runs. Two PAIR settings moved slightly (t changes up to 1.5, all
far from significance):
- momentum with a one-period lookback, from ties among equal scores (`argpartition`);
- pooled boosting, which is multithreaded.

No conclusion depends on them.

## What this means

**Seventeen rounds now agree.**
- The FX strategies people use (books, forums, EAs, the literature) were tried at every timeframe from M15 to
  monthly, in both directions and at every parameter setting.
- Their good and bad periods cannot be predicted from the market's state. That holds for the state variables
  tried: volatility, correlation, dispersion, trend strength, autocorrelation, carry, risk appetite, implied
  volatility, skew, rates and intraday microstructure.
- It holds daily, weekly or monthly, for the whole book or per instrument, with any of nine detection methods.

**The only lasting candidates remain the two forward tests:**
- the Gotobi trade;
- weekly dip-buying (Double Seven and its family).

**The useful by-product is a warning.** A 14-hour misalignment was enough to manufacture a Sharpe 0.76 "regime
edge" that beat its placebos. The one-day-lag check caught it. That check is now part of how any daily signal here
is judged.

Not financial advice.
