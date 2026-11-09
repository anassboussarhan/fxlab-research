# Round 16 pre-registration: market regimes, the best strategy for each regime, and what options data adds (2026-11-09)

Written before any of the new data below was downloaded and before any regime model was run.

## Why this round

Rounds 13-15 found strategies that win in one period and lose in another. The clearest case is weekly dip-buying
(Double Seven's family): every rule lost in 2008-2018 and won in 2019-2025 and 2026. Choosing a rule on the past
and trading it in the future cannot find such rules. The question here is whether **the market's state, measured
at the time**, tells us which strategy to run next. Sub-questions:
- how to detect the regime;
- how many regimes there are;
- which strategy, setting, variant and direction to run in each;
- whether options data (implied volatility, skew, the volatility premium) improves any of this.

**What is already known, and how it is handled.** Round 15 showed each library strategy's results on 2008-2018 and
2019-2025, so the decade-level split is known. To keep that knowledge out of the design:
- the features below are generic regime variables from the literature (volatility, correlation, trend strength,
  autocorrelation, carry, risk appetite, implied volatility, skew, rates), fixed now;
- every choice of method, number of regimes and setting is made on 2011-2018 results only;
- 2019-2025 is looked at once, at the end.

## 1. The strategy library (what a regime method can choose from)

Every strategy's monthly net P&L is computed on the 28 FX crosses as one book, 2008-01 to 2025-12, with the costs
and swaps of rounds 13-15. There are three tiers, from fine to coarse.

| Tier | What it holds | Size (about) |
|---|---|---|
| CFG | every real configuration of round 15 (parts A-C): each rule or feature unit (rule x timeframe D1 / W1 / MN) x each parameter setting x each direction | 4,140 |
| UNIT | each (unit, direction): the equal-weight average of its configurations' scaled returns | 2,140 |
| FACTOR | nine classic FX styles, coded below | 9 |

**The FACTOR tier.** These styles run on the round-14 daily panels with the round-14 continuous engine (fill at
08:00 London, costs at the fill, swaps). Each pair is sized at 10% / its 60-day EWMA volatility.

| Factor | Position on each cross |
|---|---|
| F1 CARRY | sign of the interest-rate differential; daily, 25% buffer |
| F2 TSMOM12 | sign of the 252-day return; month-end rebalance |
| F3 TSMOM3 | sign of the 63-day return; month-end rebalance |
| F4 EWMAC | sign of EMA(16) - EMA(64) of the close; daily, 25% buffer |
| F5 REV1W | minus the sign of the last 5-day return, set at each week's last close, held a week |
| F6 REV1D | minus the sign of the last day's return, daily |
| F7 DIP_W1 | the weekly dip-buying family: the equal-weight average of the seven Connors-family units at W1 (Double Seven, cumulative RSI, RSI(2), R3, %b, ConnorsRSI, three down closes), direction +1, at the default setting (all settings if the default is absent), from the library |
| F8 DONCH20 | always in: long after a close above the prior 20-day high, short after a close below the prior 20-day low |
| F9 USD | long the US dollar on the seven USD pairs |

**Scaling.** Each strategy's return in month t is divided by its trailing volatility. That volatility is the root
mean square over months t-36 to t-1, and needs at least 12 active months in that window. Strategies without it are
not eligible in month t.
- The scaled return z has a monthly standard deviation of about 1.
- Estimates use z winsorised at +-5; P&L uses the raw z.

**The placebo library.** The round-15 random-signal placebo units, rules and features, were priced exactly like
the real ones. They form a second library and go through every method below as a calibration.

## 2. Regime variables (monthly, known at the month's last close)

**PRICE: computed from the daily panels (28 crosses unless stated).**

| # | Feature | Definition |
|---|---|---|
| 1 | RV21 | median over crosses of the annualised 21-day realised volatility |
| 2 | RV_CHG | log(RV21 / RV21 three months earlier) |
| 3 | RV252 | median 252-day realised volatility |
| 4 | CORR63 | mean pairwise correlation over 63 days of the seven currencies' daily returns against the dollar |
| 5 | DISP21 | cross-sectional standard deviation of 21-day returns / (RV21 x sqrt(21/252)) |
| 6 | ER63 | median efficiency ratio over 63 days: \|net move\| / sum of \|daily moves\| |
| 7 | VR5 | median variance ratio over 252 days: variance of non-overlapping 5-day returns / (5 x daily variance) |
| 8 | AC_W | median lag-1 autocorrelation of weekly returns over the last 52 weeks |
| 9 | TREND12 | F2's mean scaled return over the last 12 months |
| 10 | REV12 | F5's mean scaled return over the last 12 months |
| 11 | CARRY_SD | cross-sectional standard deviation of the 28 interest-rate differentials |
| 12 | CARRY6 | F1's mean scaled return over the last 6 months |
| 13 | USD3 | 63-day log return of the dollar against the seven others (equal weight) |
| 14 | SPX3 | 63-day log return of the S&P 500 CFD |
| 15 | SPX_RV21 | 21-day realised volatility of the S&P 500 CFD |
| 16 | GOLD3 | 63-day log return of gold |

**OPT: free CBOE volatility indices, with FRED as the fallback source.**

| # | Feature | Definition |
|---|---|---|
| 17 | VIX | log VIX |
| 18 | VIX_TS | VIX / VIX3M (above 1 means an inverted term structure) |
| 19 | VVIX | log VVIX |
| 20 | SKEW | CBOE SKEW |
| 21 | GVZ | log gold volatility index |
| 22 | OVX | log oil volatility index |
| 23 | EVZ | log euro volatility index (FX implied volatility); used only if its history runs to 2025 |
| 24 | VRP_SPX | VIX / 100 - SPX_RV21 |
| 25 | DVIX | log change of VIX over the month |

**CME: the CME euro and yen options bought in round 9 (2010-06 to 2025).** Black-76 on daily settlements, at the
expiry nearest 30 days (10-60 days).

| # | Feature | Definition |
|---|---|---|
| 26 | EUR_IV | at-the-money implied volatility (strike nearest the future; mean of call and put) |
| 27 | EUR_RR25 | 25-delta call IV - 25-delta put IV |
| 28 | JPY_IV | as EUR_IV, for the yen |
| 29 | JPY_RR25 | as EUR_RR25, for the yen |
| 30 | FX_VRP | mean of (EUR_IV - EURUSD RV21) and (JPY_IV - USDJPY RV21) |

**MACRO**

| # | Feature | Definition |
|---|---|---|
| 31 | US2Y_CHG3 | 3-month change of the US 2-year Treasury yield (Treasury par curve) |
| 32 | SLOPE | US 10-year - 2-year yield |
| 33 | NFCI | Chicago Fed financial conditions index; the latest week ending at least 7 days before the month's end |
| 34 | EPU | US daily news-based economic policy uncertainty, 30-day mean |

A source that cannot be downloaded is dropped and reported. Nothing replaces it.

**Feature sets**

| Set | Contents |
|---|---|
| PRICE | 1-16 |
| OPT | 17-25 |
| ALL | PRICE + OPT + MACRO |
| ALLC | ALL + CME (26-30) |
| RET | for the HMM only: the first three principal components of the UNIT tier's scaled returns, fitted on the training months |

**Optional extra options data.** A free Databento cost check prices CME options on sterling, the Australian and
Canadian dollars and the Swiss franc (definitions and settlements). They are bought only if the total is at most
USD 10, which is inside what is left of the free credit. If bought, their IV and RR25 join the CME set as features
26b-29b. If not bought, they are reported as not affordable.

**Rules for every feature**
- **Standardisation:** by the training months' mean and standard deviation, at each refit.
- **Missing values:**
  - A month missing any feature of a set is left out of that set's training data.
  - A set needs at least 30 complete training months. Until it has them, or when the decision month's value is
    missing, the method falls back to the static score (B1 below) for that month.

## 3. Walk-forward rules (common to every method)

- **Timing:** the decision for month t is made at the close of month t-1, using only data up to then.
- **Out-of-sample period:** 2011-01 to 2025-12. The library starts in 2008-01, so the first decision has 36 months
  of history.
- **Training data:** all earlier months (expanding), or the last 60 months where a grid says so.
- **Refits:** regime models (clusters, HMM, ridge, boosting, quantile cut-points) are refitted each January on data
  to the previous December. Between refits, the fitted model labels new months. Regime-conditional means are
  updated every month.
- **Allocation:**
  - each method gives every eligible strategy a score;
  - the top k strategies with a score above 0 get weight 1/k each;
  - unused weight stays in cash.
- **Return and costs:** the method's month-t return is R_t = sum of w_i z_i - 0.03 x sum of \|w_i,t - w_i,t-1\|.
  The cost term is a switching cost of 0.03 standard deviations per unit of weight moved (about two round turns of
  spread on a full book).
- **Statistics:** the mean of R, the annualised Sharpe ratio, and the Newey-West t (3 lags).

**k grid**

| Tier | k |
|---|---|
| FACTOR | 1, 2, 3 |
| UNIT | 1, 5, 20, 100 |
| CFG | 1, 10, 50, 200 |

## 4. Methods

**Shrunk regime-conditional mean (used by M2-M4).** For strategy i in regime r the score is
(n_r m_ir + n0 m_i) / (n_r + n0), where:
- m_ir is the training mean of z in regime r, and m_i the unconditional training mean;
- n_r is the number (or the posterior weight) of training months in regime r;
- n0 is in {0, 12}.

A regime with fewer than 6 training months scores m_i.

| Method | Score of strategy i for month t | Grid (besides k) |
|---|---|---|
| B0 EW | 1 for every eligible strategy (no top-k) | - |
| B1 STATIC | training mean of z: unconditional, the "no regime" case (K = 1) | window: expanding, 60 |
| M1 MOM | mean z over the last L months | L: 1, 3, 6, 12, 24, 36 |
| M1b FILTER | equal weight over all strategies whose mean z over the last L months is above 0 (no top-k) | L: 3, 6, 12 |
| M2 CLUSTER | shrunk regime-conditional mean. Regimes come from k-means (20 starts) or a diagonal Gaussian mixture (5 starts) on the first three principal components of the standardised feature set; the decision month is classified by the fitted model | algorithm x K: 2, 3, 4, 5, 6, 8 x set: PRICE / OPT / ALL / ALLC x n0 x window |
| M3 SPLIT | shrunk regime-conditional mean, with regimes = quantile bins of one feature (cut-points from the training months) | feature: each of 1-34 x K: 2, 3 x n0 x window |
| M4 HMM | posterior-weighted shrunk conditional mean. A diagonal Gaussian HMM (10 starts, best likelihood) on the first three components of the set. The training months' weights are their smoothed posteriors. The decision month's regime probabilities are the filtered posterior at t-1 times the transition matrix | K: 2, 3, 4 x set: PRICE / OPT / ALL / ALLC / RET x n0 |
| M5 PREDICT | ridge regression of each strategy's z on the standardised set at t-1 (expanding training); score = prediction. FACTOR tier only: one pooled gradient-boosting model (depth 3, 200 trees, learning rate 0.05, at least 20 rows per leaf, factor one-hot plus the set) | ridge: set x alpha 10, 100, 1000; boosting: set |

## 5. Selection and the tests

**Inner selection.** In each (method family, tier), the setting with the highest Newey-West t on 2011-01 to
2018-12 is chosen. That one setting is then evaluated once on 2019-01 to 2025-12. B1 is chosen the same way and is
the benchmark.

**The tests**
- **Count:** 18 tests, one per (method family, tier). The families are M1, M1b, M2, M3, M4 and M5, each in three
  tiers.
- **p-value:** p = 1 - Phi(t) on 2019-2025.
- **Correction:** Benjamini-Hochberg at q = 0.10.

**A regime method works only if all four hold on 2019-2025:**
- **P1:** mean > 0, Newey-West t >= 1.645, and it passes BH.
- **P2:** it beats the static benchmark. Its return minus B1's (same tier, inner-selected) has Newey-West
  t >= 1.645.
- **P3:** it beats its placebo. Its Sharpe ratio is at or above the 95th percentile of 200 placebo runs of the same
  setting. In each placebo run, the method's regime input is shifted in time by a random offset of 24 to T-24
  months (circularly). For M2-M5 that input is the feature matrix; for M1 / M1b it is the trailing-performance
  scores.
- **P4:** the same method family and setting, applied to the placebo library, does not pass P1. This is a
  calibration check: if the placebo library passes, the machinery is biased and nothing counts.

**Reported, not counted**
- B1 against B0.
- The whole surface of settings on both periods, including the result for every number of regimes K. The
  2019-2025 surface is information only.
- The chosen setting's regime map at the last refit: each regime's feature profile and its top strategies.
- The BIC-optimal number of mixture components on 2008-2018.
- An "overall pick" per tier: the family with the best 2011-2018 t, with its 2019-2025 result.

## 6. Diagnostics that need no strategy selection (part A)

**A1, performance persistence.** For the UNIT and FACTOR tiers: the monthly cross-sectional Spearman correlation
between strategies' trailing 12-month mean z and their next-month z.
- The test is its mean over 2011-2025, with the Newey-West t.
- The same is reported for 36-month trailing means.

**A2, do regime effects repeat?** For each single feature (terciles cut on 2008-2018):
- compute each strategy's conditional mean minus its unconditional mean, per regime, separately on 2008-2018 and on
  2019-2025 (same cut-points);
- correlate the two across all (strategy, regime) cells;
- the p-value comes from 200 circular time shifts of the feature;
- BH at q = 0.10 across features, for the UNIT and FACTOR tiers.

A positive, significant correlation means regime effects found in one decade show up in the next. This answers the
question "do strategies fail and win in regimes that can be identified?" directly.

## 7. What options data adds (part O)

**O1, inventory (reported).** What options data exists and at what cost: the CME data we hold, the free CBOE
indices, FRED, the Databento cost check, and the earlier options tests (`docs/test_ledger.md`).

**O2, options in the regime methods.** For the chosen settings of M2, M4 and M5 in each tier, the feature set is
swapped (PRICE / OPT / ALL / ALLC) with everything else fixed.
- Reported: the 2019-2025 Sharpe ratios, and the difference t of OPT - PRICE, ALL - PRICE and ALLC - ALL.
- Options "help" only if a difference has t >= 1.645 and passes BH across those comparisons (3 methods x 3 tiers
  x 3 differences = 27).

**O3, options as timing signals for the factors.** For each factor (F1-F9) and each OPT and CME feature (17-30):
- the timing return is sign(b) x clip(standardised feature at t-1, -2, 2) x the factor's z in month t;
- b is the slope of the factor's z on the lagged feature over 2008-2018 (2010-07 to 2018 for CME);
- it counts if the 2019-2025 Newey-West t >= 1.645 and it passes BH at q = 0.10 across all factor x feature tests;
- the PRICE and MACRO features are run the same way and reported for comparison.

## 8. A filter for the weekly dip-buying trade (part D)

F7 is the forward-tested family. For each feature (1-34):
- the filter holds F7 only in the tercile where F7's 2008-2018 mean was highest;
- it counts if, on 2019-2025, the filtered return minus the unfiltered one has t >= 1.645 and passes BH across
  features.

If a filter counts and its feature can be computed in 2026 (vol indices, rates, or ECB-fixing volatility), it gets
one check on the 2026 family trades in `docs/round15/family_trades.csv`, by this rule:
- mean R of the trades the filter keeps, minus the mean R of all trades, > 0;
- kept trades: t >= 1.645, by month.

That would be the fourth use of the 2026 holdout.

## 9. What happens after

**If anything counts:**
- it is described with its regime map;
- before any 2026 data is touched, an addendum fixes its 2026 check;
- it can join the forward test.

**If nothing counts:** that is the answer. The library's wins and losses are not predictable from the market's
state with these variables and methods.

**Leakage checks (run first, on synthetic data)**
- A planted look-ahead feature (next month's returns) must give a huge, obviously fake result.
- Shifting it by one month must remove the result.
- A synthetic library with planted regimes must be recovered.

## 10. Code and runs

- **Data job:** `scripts/r16_data.py`, workflow `r16data`. Outputs:
  - an artifact (the library matrices, factors, features; not redistributed);
  - coverage reports on branch `round16-data`.
- **Engine:** `scripts/regime16.py`, workflow `r16run`, results on branch `round16`. Tests are in
  `tests/test_regime16.py`.
- **Write-up:** `docs/round16_results.md`.

## Addendum 1 (2026-11-09, data collection; before any regime model was run on the real library)

These are data-availability facts and fixes. No regime result had been computed when they were written.

**Sources**
- **FRED refuses connections from the GitHub runners** (HTTP/2 stream errors). The FRED fallbacks were replaced by
  the original publishers:
  - yields: the US Treasury par yield curve (2- and 10-year);
  - NFCI: the Chicago Fed's own file;
  - EPU: policyuncertainty.com (Baker, Bloom and Davis).
- **CBOE's own files for VIX3M, GVZ and OVX now start on 2009-09-18.** Their earlier history (VIX3M from 2006,
  GVZ from 2008-06, OVX from 2007-05) is the same index series from Yahoo Finance, used only before CBOE's first
  date.
- **EVZ is no longer served** (CBOE: access denied; Yahoo: no data). By the rule of feature 23 it is dropped.

**Extra CME options.** The Databento cost check priced sterling, Australian and Canadian dollar and Swiss franc
options (definitions and settlements) at **USD 60.29**. That is above the USD 10 limit, so nothing was bought.

**Fix to the FX features.** The first build sampled every symbol on the union of all symbols' days. The S&P 500 and
gold CFDs, and a few FX month ends, came out missing on days their own series had no close. Each symbol is now
sampled at its own last close on or before the FX month end, at most 7 days old. Month ends come from the FX
calendar: days with at least 20 of the 28 crosses.

**P4 in code.** The placebo library "passes P1" if its chosen setting has mean > 0 and t >= 1.645 on 2019-2025,
without BH. This is stricter than P1, so it can only remove a pass.
