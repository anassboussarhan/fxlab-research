# Round 14 results: re-checking round 13, missed trend variants, predicting trend, and Double Seven on 2026 (2026-11-09)

Rules: `docs/trend14_preregistration.md`, committed before the run.
- **Code:** `scripts/trend14_prep.py` (daily panels), `scripts/trend14.py` (an independent daily engine and parts
  A-C), `scripts/trend14_bracket.py` (one investigation) and `scripts/trend14_holdout.py` (part D).
- **Runs:** 42534301711 (A-C), the bracket and holdout misc jobs.
- **Tables:** `docs/trend14/`.

**Results:**
- **Round 13 holds up.** A second engine written from scratch reproduces it. A second data source (ECB fixings)
  agrees with it, and so does a different fill time.
- **No missed variant works on FX:** 18 rows, 0 count. Those rows cover:
  - 23 trend speeds;
  - long-or-flat timing of gold and indices;
  - currency-strength indices;
  - trend plus carry;
  - trend only when volatility is high.
- **Trend cannot be predicted here:** 8 model rows and 11 single predictors, 0 count. Out of sample, the models
  call the sign right 48-51% of the time.
- **Double Seven on weekly bars passed its 2026 check:** 33 trades, +0.78R a trade, t 5.4 with trades clustered
  by month. It is a real effect but a dangerous one. It lost heavily whenever markets trended hard (2008-2010,
  2015-2016).

## A. Double check of round 13

**A1. An independent engine.**
- The new engine works on daily bars split at the 08:00 London fill. It shares no engine, indicator or system
  code with round 13.
- On synthetic data with planted trends, both engines agreed (monthly correlation 0.98-1.00 on every system).
- On pure random walks, every new variant and model was null.
- On the real data, 32 of 33 system x market rows agree (monthly gross correlation 0.89-1.00, same sign of the
  2019-2025 mean):

| System | FX 2019-2025 t: new / round 13 | Non-FX 2019-2025 t: new / round 13 | Gross correlation FX / non-FX |
|---|---|---|---|
| L1 Turtle System 2 | -4.60 / -3.90 | +0.43 / +0.70 | 0.95 / 0.98 |
| L2 Turtle System 1 | -4.88 / -4.04 | -0.87 / +0.29 | 0.94 / **0.89** |
| L4 Donchian 100 / 50 | -1.75 / -1.74 | +1.17 / +1.26 | 1.00 / 1.00 |
| L5 MA 50 / 200 | -0.93 / -0.93 | +1.48 / +1.48 | 1.00 / 1.00 |
| L6 TSMOM 12 months | -0.82 / -0.93 | +1.43 / +1.37 | 0.97 / 0.96 |
| L7 TSMOM 1 / 3 / 12 | -0.96 / -1.01 | +0.62 / +0.55 | 0.97 / 0.95 |
| L8 Faber | -1.16 / -1.20 | +0.55 / +0.50 | 0.97 / 0.96 |
| L9 EWMAC | -1.21 / -1.22 | +1.19 / +1.16 | 0.97 / 0.96 |
| L10 Squeeze, trailing stop | -2.96 / -3.04 | +0.27 / +0.27 | 1.00 / 1.00 |
| L11 Weekly breakout | -1.29 / -1.28 | +0.90 / +0.89 | 1.00 / 1.00 |
| BH long-only | +0.27 / +0.19 | +2.61 / +2.53 | 0.98 / 0.95 |
| Double Seven W1 (FX, sum R; t) | +39.7R; 2.37 / +39.6R; 2.37 | all 33: +64.3R; 3.61 / +64.3R; 3.61 | - |

**The one disagreement, Turtle System 1 on non-FX, was investigated** (`docs/trend14/A1_bracket.csv`).
- A daily bar cannot tell whether a new entry or added unit came before or after a stop touched in the same part
  of the day. The new engine assumes the worse order. Run again assuming the better order, the two cases bracket
  the minute-exact round-13 result:

| | Worst order | Best order | Round 13 |
|---|---|---|---|
| L2 non-FX, 2019-2025 mean %/month (t) | -0.31 (-0.87) | +0.20 (0.53) | +0.10 (0.29) |
| L1 FX | -0.87 (-4.60) | -0.67 (-3.89) | -0.86 (-3.90) |

- The difference is the daily engine's simplification, not an error in round 13. With the better order, the
  gross correlation is 0.99.

**A2. Fill time.** Filling at the close instead of 08:00 London the next morning changes nothing that matters.
The FX 2019-2025 t moves between -0.3 and +0.3 for the slow systems. It gets worse for the Turtles (-5.3 and -5.5).

**A3. A second data source and a longer history (ECB fixings, gross).**
- **Data check:** the ECB-based monthly gross returns correlate 0.83-0.92 with the store-based ones over 2008-2025.
- **Before 2008:** FX trend was weak there too:

| System | 1999-2007 t | 2008-2018 t | 2019-2025 t |
|---|---|---|---|
| L5 MA 50 / 200 | 0.96 | -0.19 | -0.53 |
| L6 TSMOM 12 months | 1.03 | -0.55 | 0.10 |
| L7 TSMOM blend | 0.31 | -0.30 | -0.29 |
| L8 Faber | 0.13 | -0.26 | -0.88 |
| L9 EWMAC | -0.77 | -0.35 | -1.20 |
| Close Donchian 100 / 50 | -0.41 | 0.39 | -0.99 |

- In the 28 crosses, the strong trend years were 2002-2003 (the dollar's fall) and 2008. Few other years
  helped.

## B. Missed variants (18 rows; 0 count)

| Row | Best on 2008-2018 | Fit t | 2019-2025 mean %/month (t) | Gross t | vs long-only (t) |
|---|---|---|---|---|---|
| B1a breakout speed, FX | 50 days | -0.24 | -0.69 (-3.34) | -2.37 | - |
| B1a breakout speed, non-FX | 200 days | 1.12 | +1.00 (1.83) | 2.37 | -0.26 |
| B1b MA speed, FX | 20 / 100 | -0.35 | -0.71 (-2.18) | -1.44 | - |
| B1b MA speed, non-FX | 50 / 200 | 0.73 | +0.83 (1.48) | 2.05 | -0.65 |
| B1c TSMOM speed, FX | 1 month | 0.34 | -0.48 (-1.36) | -0.51 | - |
| B1c TSMOM speed, non-FX | 6 months | 0.81 | +0.18 (0.43) | 1.28 | -2.66 |
| B1d EWMAC speed, FX | 32 / 128 | -0.39 | -0.18 (-0.70) | -0.03 | - |
| B1d EWMAC speed, non-FX | 64 / 256 | 0.49 | +0.70 (1.65) | 2.39 | -1.06 |
| B2 long-or-flat L5, non-FX | - | 1.15 | +1.52 (2.30) | 2.86 | alpha t 0.85 |
| B2 long-or-flat L6, non-FX | - | 0.69 | +1.28 (2.45) | 3.29 | alpha t 0.57 |
| B2 long-or-flat L8, non-FX | - | 0.95 | +0.99 (2.02) | 2.91 | alpha t -0.12 |
| B2 long-or-flat L9, non-FX | - | -0.02 | +0.99 (1.95) | 2.70 | alpha t 0.52 |
| B3 currency strength, TSMOM | - | -1.15 | -0.29 (-0.89) | -0.29 | - |
| B3 currency strength, EWMAC | - | -1.80 | +0.05 (0.08) | 0.49 | - |
| B3 currency strength, breakout 50 / 25 | - | -0.16 | +0.15 (0.20) | 0.57 | - |
| B4 EWMAC + carry, FX | - | -0.80 | -0.30 (-0.75) | -1.13 | - |
| B5 trend in high volatility, FX | - | -1.50 | -0.31 (-1.12) | -0.29 | - |
| B5 trend in high volatility, non-FX | - | -1.48 | +0.56 (1.13) | 1.60 | -1.35 |

**FX: every speed loses.**
- Across all 23 breakout, moving-average, momentum and EWMAC speeds, the 2019-2025 FX t runs from -4.1 to +0.4.
- The best is the slowest moving-average cross, 100 / 400 (+0.42).
- None is positive in 2008-2018 except 1-month momentum (+0.34).

**Gold and indices: timing is beta, not skill.**
- Long-or-flat versions earn +1.0% to +1.5% a month on 2019-2025, with t 2.0-2.5.
- Regressed on holding long, their alpha t is -0.1 to +0.9, and their betas are 0.6-0.7 in 2008-2018 and
  0.7-1.1 in 2019-2025.
- They are a lower-risk way of being long. They are not a timing edge.

**Currency-strength indices** (trading each currency against the basket of the other seven, the way some FX funds
define FX trend) did not help either: t -1.8 to +0.2.

## C. Can trend be predicted? (8 rows and 11 predictors; 0 count)

**Direction models** (12 features: momentum at 1-12 months, carry, rate changes, value, volatility, skew, range
position; ridge and gradient boosting, retrained every year):

| Row | 2011-2018 (out of sample) t | 2019-2025 mean %/month (t) | Sign hit rate 2019-2025 (pooled, all 33) |
|---|---|---|---|
| Ridge, FX | -2.47 | -0.75 (-2.36) | 49.7% |
| Ridge, all 33 | -2.60 | -0.56 (-1.90) | 49.7% |
| Gradient boosting, FX | -1.68 | -0.77 (-2.71) | 49.8% |
| Gradient boosting, all 33 | -1.80 | -0.78 (-2.98) | 49.8% |

- The ridge model's cross-sectional ranking was slightly right in 2019-2025 (rank correlation +0.05, t 1.9). Its
  sign, which decides the trade, was not. Both models lost in both out-of-sample periods.

**When trends happen (trendiness).** Each predictor was tested on whether it predicts next month's trend-following
P&L:
- the variance ratio;
- autocorrelation of daily and weekly returns;
- the efficiency ratio at 63 and 252 days;
- ADX;
- trend strength;
- the volatility ratio;
- past trend P&L;
- average FX correlation;
- cross-sectional dispersion.

| Predictor | 2008-2018 t | 2019-2025 t in the same sign |
|---|---|---|
| Variance ratio (5-day) | -0.78 | -0.13 |
| Daily autocorrelation | -0.16 | 0.62 |
| Weekly autocorrelation | -0.00 | 0.48 |
| Efficiency ratio 63 days | 0.22 | -0.73 |
| Efficiency ratio 252 days | 0.69 | 0.15 |
| ADX(14) | 0.56 | -1.92 |
| Trend strength | 0.75 | 0.07 |
| Volatility ratio 1 / 12 months | 0.34 | -0.80 |
| Past 12-month trend P&L | -0.37 | 0.27 |
| Average FX correlation | -0.01 | 1.02 |
| Cross-sectional dispersion | 0.29 | -1.45 |

- None had any power even in 2008-2018 (all |t| < 0.8), and none in 2019-2025.
- ADX, the most common "is it trending?" gauge, points the wrong way out of sample.
- The models built on all of them lost when switching between trend and reversal (t -1.4 and -1.8). They also
  lost when switching between trend and flat (t -0.7 and -0.9).
- Past trend P&L does not predict future trend P&L. Trend did not come in predictable runs.

The 11 predictors were tested under Benjamini-Hochberg as one group of 11. The pre-registration said 12; the list
it gives has 11.

## D. Double Seven on weekly bars, 2026 (third use of the holdout)

**Rule, unchanged from round 13:**
- Buy when the weekly close is above its 200-week average and is the lowest close of 7 weeks.
- Sell after a 7-week highest close. No stop.
- Tested on the 28 FX crosses with ECB fixings, a 3 bp round trip and no swap, as pre-registered.

| Period | Trades | Mean R per trade | t (iid) | t (clustered by month) | Win rate | Worst trade |
|---|---|---|---|---|---|---|
| 2008-2018 | 331 | -0.36 | -2.40 | -1.42 | 61% | -15.7R |
| 2019-2025 (development comparator) | 242 | +0.40 | 4.11 | 3.20 | 70% | -7.9R |
| **2026, January-October (holdout)** | **33** | **+0.78** | **5.84** | **5.38** | **85%** | **-0.8R** |

**Verdict by the pre-registered rule: supported** (2026 mean > 0 with t >= 1.645).
- The rule is the frozen round-13 rule.
- The data were stored before this round, and the comparator was positive.
- 28 of the 33 trades won. By exit month, seven of the ten months were positive, October lost -0.7R, and two
  months closed no trade.

**Why it is still not something to trade at size:**
- **It loses big when markets trend.** The yearly results (by exit year, sum of R over the 28 crosses):

  | Years | Result |
  |---|---|
  | 2003-2007 | Positive every year, +12 to +49R |
  | 2008 / 2009 / 2010 | -60R / -45R / -15R |
  | 2011-2014 | +12, -1, -4, +29R |
  | 2015 / 2016 | -20R / -39R |
  | 2017-2025 | Positive in 7 of 9 years |
  | 2026 | +19R |

- **The worst trades** were dips bought in a crash and held for months without a stop:
  - GBP/USD from August 2008: -15.7R;
  - EUR/USD in 2014-15: -14.6R;
  - EUR/JPY, AUD/JPY, AUD/USD and CHF/JPY, all from August 2008: -11 to -13R.
- **The drawdown.** At 1R a trade across all crosses, the largest drawdown over 2008-2025 was 155R.
- **Prop rules.** With several correlated positions open at once, a 2008-type month would break any prop firm's
  drawdown limit unless the risk per trade were tiny.
- **Approximations in the 2026 test.** It used ECB fixings, an assumed cost, and no swap (swap is positive for long
  AUD, NZD or GBP against JPY or CHF, and negative for long EUR). It covered FX only; gold and the indices were not
  tested. 33 trades in 21 distinct weeks is still a small sample.
- **Stops don't fix it.** Round 13's versions with a 2N stop and trailing exits were weaker in both periods
  (2019-2025 t 1.2-1.8).
- **The design problem.** A stop that keeps the edge but caps the crash losses would be a new rule. Every year
  through 2026 has now been seen, so it can only be judged on live data.

**What it is:** weekly dip-buying in up-trending currencies. It earns in calm or range-bound years and gives back
years of gains in a crisis trend. It is the mirror image of trend following. It explains why trend lost in
2019-2026, and why trend's best years (2008, 2014-2016) were this rule's worst.

## What this means

- **Round 13 stands.** Long FX trend does not pay on these data. The finding survives:
  - a second engine;
  - a second data source;
  - a different fill time;
  - 23 speeds, a currency-strength version, a carry blend, a volatility filter;
  - two prediction approaches.
- **On gold and indices**, trend timing is a cheaper way to be long. It is not an edge over being long.
- **Trend cannot be predicted** with these tools: neither its direction a month ahead, nor whether next month
  will trend.
- **The only new survivor is the opposite trade:** weekly Double Seven on FX. It passed a pre-registered check on
  2026 data that no earlier round had used. It still carries crash risk that the data show clearly.
- **What it would take to trade it:** it would need a live forward test at a small size before any prop account.
  It would also need a written crash rule (for example a portfolio cap on open trades, or a far stop), judged on
  the live data.
