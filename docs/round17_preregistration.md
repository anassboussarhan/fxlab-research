# Round 17 pre-registration: regimes, harder (2026-11-09)

Written after round 16 (`docs/round16_results.md`) and before any round-17 data was built.

## Why this round

Round 16 found no tradeable regime effect. Its limits were:
- monthly decisions only;
- a library of D1 / W1 / MN rules only;
- regimes for the whole FX book only;
- six method families.

The user asked to try harder and with more timeframes. Round 17 removes each limit:
1. **Intraday strategies:** every catalog rule at H4, H1, M30 and M15, at its full parameter grid, in both
   directions.
2. **Faster switching:** decisions every day (D), every week (W) and every month (M).
3. **Regimes per instrument:** each of the 33 instruments (28 FX crosses, gold, silver, three index CFDs) chooses
   its own strategy from its own state and the market's.
4. **More detectors:**
   - online learning that tracks the best strategy as it changes (Hedge with forgetting);
   - change-point resets (CUSUM);
   - a pooled gradient-boosting meta-model that learns which kind of strategy works in which state, across all
     strategies at once.
5. **Features at several horizons (5 / 21 / 63 / 252 days), plus intraday state features** computed from 5-minute
   returns:
   - realised volatility;
   - jump share;
   - 5-minute autocorrelation;
   - the Asian session's share of the day's range.

**Known before this round:**
- **From round 16:** the monthly results, and the exploratory finding that yen-volatility regime effects repeat.
  Nothing here is chosen to exploit them. Every choice is made on 2011-2018, and 2019-2025 is looked at once.
- **From rounds 2-12:** intraday rules were tested on other designs, and most had no gross edge or were cost-bound.

## 1. Libraries

**Daily engine (D1, W1, MN).**
- **What:** the round-15 sweep (rules, book systems, feature terciles, placebos; 33 instruments), unchanged.
- **Change:** it now also saves daily P&L. Engine: round 13 (fills at 08:00 London on the M1 path, real swaps).

**Intraday engine (H4, H1, M30, M15).**
- **Rules:** every catalog strategy outside the ML and money-management families that runs on one instrument,
  including the time-of-day rules (excluded at D1+ in round 15). That is 105 strategies, 456 settings, each in both
  directions.
- **Engine:** `fxlab.evaluate`, the project's bar engine:
  - costs: the base cost model (quoted spread, slippage, commission);
  - state rules: the vector engine; trade rules: the trade engine, stop first when a bar holds both stop and target;
  - financing: flat;
  - the reversed rule uses the same stops and targets.
- **Placebos:** at each intraday timeframe, 10 random long / flat / short state series at each persistence
  p = 0.9, 0.98 and 0.995 per bar, both directions.
- **Aggregation:** P&L is summed by trading day, the day ending at 17:00 New York.

**Tiers.** FX book = the sum over the 28 crosses.

| Tier | What | Frequencies |
|---|---|---|
| FACTOR | the nine round-16 styles, now with daily P&L | D, W, M |
| BOOK-UNIT | each (unit, direction) of both engines on the FX book, averaged over its settings; a unit is a rule or feature at one timeframe | D, W, M |
| BOOK-CFG | every configuration of both engines on the FX book | D, W, M |
| PAIR-UNIT | instrument x rule unit x direction, rules only (no feature terciles), 33 instruments, both engines | W, M |

**Placebo libraries.** The placebos of both engines, aggregated the same way. A PAIR-UNIT placebo is the instrument
x placebo unit.

**Scaling.** Each strategy's period return is divided by the RMS of its last N periods.

| Frequency | N | Active periods needed |
|---|---|---|
| D | 252 | 60 |
| W | 104 | 26 |
| M | 36 | 12 |

z is winsorised at +-5 for estimates and raw for P&L.

## 2. Features (known at the period's last close)

**Market-wide (FX book), computed daily, sampled at each period end.**

| Set | Features |
|---|---|
| PRICE | RV5, RV21, RV_CHG (log RV21 / RV21 63 days earlier), RV252, CORR21, CORR63, DISP5, DISP21, ER21, ER63, VR5, AC_D (lag-1 autocorrelation of daily returns over 63 days), AC_W, TREND12, REV12, CARRY6 (style trailing returns, 12 / 12 / 6 months of daily P&L), CARRY_SD, USD5, USD63, SPX5, SPX63, SPX_RV21, GOLD63 |
| INTRA | RV_5M (21-day mean of daily realised volatility from 5-minute returns), JUMP (1 - bipower / realised variance, 21-day mean), AC_5M (lag-1 autocorrelation of 5-minute returns, 21-day mean), ASIA_SHARE (00:00-07:00 UTC range / day range, 21-day mean); each a median over the crosses |
| OPT | VIX, VIX_TS, VVIX, SKEW, GVZ, OVX, VRP_SPX, DVIX5 (5-day log change), DVIX21 |
| CME | EUR_IV, EUR_RR25, JPY_IV, JPY_RR25, FX_VRP, now daily |
| MACRO | US2Y_CHG63, SLOPE, NFCI (lagged 7 days), EPU (30-day mean, lagged 1 day) |

ALL = PRICE + INTRA + OPT + MACRO; ALLC = ALL + CME. The definitions and sources are those of round 16 unless
stated.

**Per instrument (PAIR tier).** The instrument's own RV21, RV_CHG, ER63, VR5, AC_D, AC_W, CARRY (its rate
differential; FX only), TS12 (252-day return), MA200Z ((close - 200-day mean) / (close x RV21 / sqrt(252))),
RV_5M and AC_5M. The market-wide sets are also available to the PAIR tier.

## 3. Walk-forward (as round 16, per frequency)

- **Timing:** the decision for a period is made at the close of the previous one. The out-of-sample period starts
  2011-01-01.
- **Inner and check periods:** the inner selection uses 2011-2018 and the check uses 2019-2025.
- **Refits:** every January.
- **Top-k:** among strategies with a score above 0, weight 1/k each; the rest of the weight is cash.
- **Switching cost per unit of weight moved.** The absolute cost equals round 16's monthly cost; in period z units
  it is:

| Frequency | Cost (period z) |
|---|---|
| M | 0.03 |
| W | 0.0625 |
| D | 0.137 |

- **Shrinkage n0:** 0 or one year of periods (12 / 52 / 252).
- **Minimum regime size:** 6 / 26 / 126 periods.
- **Long window:** 60 months / 260 weeks / 1,260 days.

**k grids**

| Tier | k |
|---|---|
| FACTOR | 1, 2, 3 |
| BOOK-UNIT | 1, 5, 20, 100 |
| BOOK-CFG and PAIR-UNIT | 1, 10, 50, 200 |

## 4. Methods

The lookbacks L in M1 are in periods:

| Frequency | L |
|---|---|
| M | 1, 3, 6, 12, 24, 36 |
| W | 1, 4, 13, 26, 52, 104 |
| D | 1, 5, 21, 63, 126, 252 |

M1b uses the 2nd, 4th and 5th of these.

| Family | Rule | Grid (besides k) | Tiers |
|---|---|---|---|
| B0 | equal weight over all eligible | - | all (benchmark) |
| B1 static | mean z over the training window | window: expanding, long | all (benchmark) |
| M1 momentum | mean z over the last L periods | L | all |
| M1b filter | equal weight over strategies with positive mean z over the last L periods | L (three values) | all |
| M2 clusters | as round 16 | k-means / diagonal Gaussian mixture x K 2, 3, 4, 6, 8 x set PRICE / INTRA / OPT / ALL / ALLC x n0; expanding | FACTOR, BOOK |
| M3 one feature | as round 16: quantile regimes of one feature | feature x K 2, 3 x n0 x window. PAIR tier: each per-instrument feature (each instrument uses its own) and each market feature | all |
| M4 HMM | as round 16 | K 2, 3, 4 x set PRICE / INTRA / OPT / ALL / ALLC / RET x n0 | FACTOR, BOOK |
| M5 ridge | as round 16 | set x alpha 10, 100, 1000 | FACTOR, BOOK |
| M6 Hedge | weights proportional to exp(eta x discounted sum of z) over every eligible strategy plus a cash expert (z = 0) | eta 0.1, 0.3, 1.0 x half-life: none, 3 years, 1 year. No top-k: the weights are used as they are | all |
| M8 change-point | a two-sided CUSUM on each strategy's z (drift 0.5) with threshold h; after an alarm, its mean restarts. The score is the mean z since the last alarm, needing at least 6 / 26 / 126 periods | h 4, 8 | all |
| M9 pooled boosting | one gradient-boosting model over all (period, strategy) rows | feature set: PRICE or ALLC | all |

**M9 in detail**
- **Target:** next-period z.
- **Inputs:**
  - the strategy's descriptors: timeframe, direction, catalog family, engine;
  - its trailing mean z over 3 / 12 / 36 (M), 4 / 26 / 104 (W) or 21 / 126 / 252 (D) periods;
  - its trailing volatility ratio and its current drawdown in z;
  - the market feature set (PRICE or ALLC);
  - for the PAIR tier, the instrument's own features.
- **Model:** depth 4, 300 trees, learning rate 0.05, at least 200 rows per leaf.
- **Training:** refitted each January on at most 400,000 randomly drawn earlier rows (fixed seed).

## 5. Tests

**One test per (tier, frequency, family).**

| Tier and frequencies | Regime families | Tests |
|---|---|---|
| FACTOR and the two BOOK tiers at D / W / M (9 combinations) | M1, M1b, M2, M3, M4, M5, M6, M8, M9 | 81 |
| PAIR-UNIT at W / M | M1, M1b, M3, M6, M8, M9 | 12 |
| **Total** | | **93** |

Each family's setting is chosen by the best Newey-West t on 2011-2018 (lags: 3 for M, 4 for W, 5 for D). It is
then judged once on 2019-2025.

**A method counts only if all four hold:**
- **P1:** the 2019-2025 mean is above 0, t >= 1.645, and it passes BH at q = 0.10 over the 93 tests.
- **P2:** its return minus that of B1 (inner-selected, same tier and frequency) has t >= 1.645.
- **P3:** its 2019-2025 Sharpe ratio is at or above the 95th percentile of 200 placebo runs. In each run the regime
  input is circularly shifted by a random offset (from 2 years to the sample length minus 2 years).
- **P4:** the same family's inner selection on the placebo library does not reach mean > 0 and t >= 1.645 on
  2019-2025. For the FACTOR tier P4 does not apply.

**Reported, not counted:**
- the whole surface;
- the K curves and the BIC;
- the regime maps;
- the "overall pick" per (tier, frequency);
- the options swap of round 16 (O2) for the chosen M2 / M4 / M5 settings.

**Correction for the timing tests.** Round 16's signed timing tests are replaced by long-only versions. The signed
version can short a style whose return is net of costs and so earn those costs. Each of the nine styles, at each
frequency, is timed by each OPT, CME and INTRA feature:
- the weight is max(0, sign(b) x clipped feature);
- b is fitted on 2008-2018;
- the timed style must beat holding it at the same average weight (difference t >= 1.645);
- BH at q = 0.10 across all of these tests.

## 6. After

- **If something counts:** a 2026 check is fixed in an addendum before any 2026 data is touched.
- **If nothing counts:** that is the answer for these methods at these timeframes.

**Leakage checks (on synthetic data, before the real run)**
- Perturbing data from a decision period on must not change any earlier decision, for every method.
- A planted look-ahead feature must give an obviously fake result.
- A planted regime must be recovered.

## 7. Code

| Purpose | File |
|---|---|
| Daily output of the round-15 sweep | `scripts/sweep15.py --daily` |
| Intraday library and intraday features | `scripts/intraday17.py` |
| Matrices and features | `scripts/r17_data.py` |
| Engine | `scripts/regime17.py` |
| Tests | `tests/test_regime17.py` |
| Write-up | `docs/round17_results.md` |

## Addendum 1 (2026-11-09, implementation; before the engine ran on real data)

These are implementation details. None changes a rule above.

**Placebos (P3 and P4)**
- **When they run:** only for chosen settings whose 2019-2025 mean is above 0 with t >= 1.645. A setting that
  fails this cannot count anyway, and the BH p-values do not depend on P3 or P4, so the verdicts are the same as
  computing them everywhere.
- **What is shifted:**
  - M1 / M1b / M6 / M8: the trailing-performance input;
  - M2 / M3 / M4 / M5: the feature rows;
  - M9: both, by the same offset.

**Tiers**
- **PAIR-UNIT returns:** the mean of the unit's settings' raw P&L on that instrument. BOOK-UNIT averages scaled
  returns, as in round 16.
- **Calendars:**
  - the FACTOR tier uses the FX-panel trading calendar;
  - the BOOK and PAIR tiers use the libraries' calendar;
  - features are taken at each period's end from the last value at most 7 days old.
- **Start:** periods start with the libraries (2007), so training uses every earlier month with data.
- **Known data gap:** the M1 store has no AUDCAD data for 2018 (also true in round 15).

**Methods**
- **HMM on the tier's own returns (RET):** uses at most 3,000 of the tier's strategies, drawn with a fixed seed, to
  keep the principal components tractable.
- **M9:** the catalog family is a categorical input. Training rows are drawn uniformly from all earlier (period,
  strategy) pairs with a finite target, at most 400,000.
- **M6:** includes the cash expert; its weights are used as they are.
- **Scores:** a static or regime score needs at least the minimum regime size of periods (6 / 26 / 126). Otherwise
  the strategy is not eligible that period.

## Addendum 2 (2026-11-09, after the first engine run): a timing flaw in part of the library, and the fix

**What was found.** In the first run (42697499365), the closest miss was the daily single-feature split on 5-day
dispersion. With the feature lagged one more day its Sharpe ratio fell from +0.76 to -3.2. That is not how a real
effect behaves.

**The cause.** The round-15 feature-tercile units and their placebos (sweep part C) were priced by
`long_lab.cont_run`, which marks P&L from one 08:00 London fill to the next.
- Day k's P&L therefore runs from 08:00 London of day k-1 to 08:00 London of day k.
- A decision taken at the close of day k-1 (17:00 New York) uses features that already contain 14 of those 24
  hours.
- Standalone tests (round 15) were not affected: there the position is decided at the previous close and earns
  only later fills.
- Combining these returns with close-of-day features gives a look-ahead of up to 14 hours. It is large for daily
  decisions, about 12% of a week for weekly ones and about 2.6% of a month for monthly ones (rounds 16 and 17).
- **Not affected:**
  - the rule units (`ev_run`), marked at the closes;
  - the nine styles (round-14 engine, marked at the closes);
  - the intraday library (bar marks);
  - the PAIR tier, which has no feature units.

**The fix.**
- Part C is re-run with `long_lab.cont_run_close`: the same decisions and fills, marked close to close (the old
  position earns the previous close to the fill, the new one the fill to the close).
- Its rows replace the fill-to-fill rows in the book matrices.
- Every BOOK test (D, W, M) is re-run, with all rules unchanged. FACTOR and PAIR are re-run as well and should
  reproduce.
- The second run's results replace the first's.
- **Round 16** used the same fill-to-fill monthly P&L for its feature units (an overlap of about 2.6% of a month).
  This can only have flattered its regime methods, so its null result stands. This is noted in its results.
