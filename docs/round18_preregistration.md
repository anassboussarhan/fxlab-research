# Round 18 pre-registration: what the full review found untested, and the regime question asked two new ways (2026-11-11)

Written after the full inventory (`docs/inventory.md`) and a new web search, and before any round-18 data was
read.

## Why this round

The user asked for four things:
- enumerate everything done;
- search the web again for what was missed;
- say why each failed strategy failed;
- think again about a regime detector that runs only the strategies that work in the current regime.

The review found three real gaps.

1. **Old-sample-only tests.** The published time-of-day and fix effects (runs 17-22) were tested on 2007-2018
   only. Several had a real gross edge that costs removed. Since then gold has roughly tripled in price, so its
   spread in basis points has fallen, and an outside study reports that gold's Asian-hours rise became much
   stronger after 2020. None of these effects was ever re-run on 2019-2025.
2. **Two published effects never tested here:**
   - the overnight-to-intraday reversal across currencies (Della Corte, Kosowski and Wang);
   - global currency volatility as a predictor of dollar returns over the following months (a 2026 paper).
3. **The regime question was only asked on 2008-2025.** That is one or two regime cycles. Two things were never
   done:
   - a feature-free test of whether strategy performance has persistent states at all;
   - a test on a long history (1975-2025) with many regime cycles.

**Corrections to the previous message to the user.** Three items listed there as "missed" had in fact been tested
in other forms:
- **Daily dip-buying on stock-index ETFs** (`docs/untested_batch.md`, D1 and 8.1): +42 bp a trade in 1993-2018
  (t 3.9) and +26 bp in 2019-2025 (t 1.4).
- **Crypto trend on daily bars** (same document): BTC 14 / 56-day trend, alpha t 1.65.
- **The Nasdaq opening-range breakout at futures costs** (same document, 9.12).

They are not repeated here.

## Part G: second-sample check of the cost-bound published effects (M1, real bid / ask)

The original code is used unchanged (`scripts/lit2_lab.py: metals`; `scripts/flows_lab.py: local_hours,
around_clock, fix_reversal`), on 2007-2025. Results are split at 2019-01-01. Net = the bid / ask fill with
slippage and commission, plus the rate-based swap.

| ID | Effect (as first tested) | 2007-2018 result here |
|---|---|---|
| G-A1 | Gold in Asian hours: long 18:00 New York to 08:00 London | gross +2.4 bp (t 2.2), net -4.8 bp |
| G-A1x | The same, without Sunday-evening entries (the outside study's version) | not run before |
| G-A2 | Gold in Western hours: short 08:00 London to 16:55 New York | gross +2.6 bp (t 1.2) |
| G-G2 | Gold AM fix: short 10:15 to 10:45 London | gross +1.0 bp (t 3.6) |
| G-G1 | Gold PM fix: short 14:45 to 15:15 London | gross +0.2 bp |
| G-S1 | Silver fix: short 11:45 to 12:15 London | gross +3.6 bp (t 5.1) |
| G-L1 | Sell each currency in its home hours (7 USD pairs) | positive gross in some pairs, negative net |
| G-K | Around the clock: long foreign currencies in New York hours, short overnight | gross t 2.3 on EUR / GBP / CHF |
| G-W1 | Fade the 15:30-16:00 London move after the 4 p.m. fix | negative net |

**A row counts if all hold on 2019-2025:**
- net mean > 0 with t >= 1.645 (t from daily values; rows with several pairs are averaged by day first);
- its 2007-2018 gross mean was positive (the direction is not chosen on the new data);
- BH at q = 0.10 over the round-18 family (below);
- **for G-A1 and G-A1x only, the anomaly test:** the Asian-hours gross return minus the Western-hours gross return
  (long 08:00 London to 16:55 New York) has t >= 1.645. Gold rose in 2019-2025, so any long window gains from the
  rise; the anomaly is that the rise comes in Asian hours.

**If G-A1 or G-A1x counts:** one check on 2026 gold data (Dukascopy, January to October 2026, to be downloaded
only then), with the same rule and costs.
- Supported: net mean > 0 with t >= 1.645.
- Inconclusive: positive with a smaller t.
- Contradicted: negative.

This would be the fifth use of the 2026 holdout.

## Part C: two published effects not tested before

**C1. Overnight-to-intraday reversal across currencies.**
- **Instruments:** the seven USD pairs, each expressed as the foreign currency against the dollar.
- **Signal:** each currency's return from 17:00 New York (previous business day) to 08:00 London.
- **Weights:** cross-sectional contrarian, w_i = -(r_i - mean r) / sum of |r_j - mean r|, so gross exposure is 1.
- **Holding:** 08:00 London to 16:55 New York, at bid / ask with costs.
- **Periods:** 2007-2018 and 2019-2025.
- **Counts if:** 2019-2025 net mean > 0 with t >= 1.645, 2007-2018 net mean > 0, and BH over the family.
- **Reported:** the gross result, and the time-series version (each pair fades its own overnight move).

**C2. VWAP trend on index CFDs (Zarattini and Aziz).**
- **Instruments:** US500 and NAS100 CFDs, 09:30-16:00 New York, 2019-2025.
- **Rule:** after each minute's close, long if the mid is above the session's VWAP and short if below; the position
  changes at the next minute's open; flat at 16:00.
- **VWAP weights:** the tick count per minute where the store has it, else equal weights.
- **Costs:** half the quoted spread plus slippage on every position change.
- **Counts if:** net mean > 0 with t >= 1.645, and BH over the family. The gross result is reported.

## Part R0: is there anything for a regime detector to find?

A regime detector can only help if a strategy's expected return moves between persistent states. That can be
tested without choosing any feature.

**Variance ratios.**
- **Statistic:** for each strategy, VR(K) = variance of non-overlapping K-period sums of its scaled return /
  (K x variance of the one-period return). With no persistent states VR(K) is about 1; with them it rises with
  K. The statistic is the median over strategies.
- **Data:**
  - monthly: the round-16 library (UNIT tier, 2,258 strategies; K = 3, 6, 12, 24 months);
  - daily: the round-17 library (BOOK-UNIT tier, 3,098 strategies; K = 5, 21, 63, 126, 252 days);
  - the nine styles, monthly and daily;
  - each placebo library, the same way.
- **Null:** 300 random permutations of the time order, the same permutation for every strategy. This keeps each
  strategy's average and the cross-section, and removes persistence.
- **p-value:** the share of permutations with a median VR at or above the real one.

**Persistence in rank.**
- **Statistic:** the mean cross-sectional rank correlation between a strategy's trailing mean (12 months, or 63
  days sampled every 21 days) and its next-period return.
- **Null:** the same permutation null.
- **Why:** round 16's A1 found 0.05, but lasting cost differences between strategies produce that without any
  regime. The permutation keeps those differences, so only the excess over the null is evidence of regimes.

**R0 says "there are states to detect" if** the median VR is above the null's 95th percentile at two or more
horizons for the real library, and not for its placebo library. Otherwise the answer is that performance has no
persistent states beyond chance, and no detector, with any features, can time these strategies.

## Part L: regimes on a long history (1975-2025, daily)

**Data.**
- **Exchange rates:** Bank of England daily spot rates from 1975 for USD, JPY, EUR (the Deutsche Mark / 1.95583
  before 1999), CHF, CAD, AUD and NZD against GBP, giving the 28 crosses of the eight currencies.
- **Interest rates:** OECD monthly 3-month interbank rates (DBnomics), lagged one month. Germany is used for the
  euro before 1999. Where 3-month rates are missing, the OECD short-term rate is used.
- **Costs:** 1.5 bp per unit of notional traded, per side (today's level; older spreads were wider, so early
  results are upper bounds).
- **Returns:** include the interest differential.

**Styles** (each cross sized at 10% / its 60-day EWMA volatility; the book is the sum over the 28 crosses; monthly
returns scaled as in round 16).

| Name | Rule |
|---|---|
| F1 CARRY | sign of the 3-month rate differential, set at each month end |
| F2 TREND | sign of the 252-day return, set at each month end |
| F5 REV1W | minus the sign of the last 5-day return, set at each week's last close |
| F7 DIP_W | the seven weekly Connors-family rules (published defaults, long only, close-only weekly bars); the position is the share of the seven that are long |

**L1 (reported).** Each style's Sharpe ratio by decade, and the correlation of DIP_W with TREND.

**L2, the regime detector.** Round 16's walk-forward machinery (`scripts/regime16.py`) is used: monthly decisions,
choose among the four styles and cash, top 1 or 2, switching cost 0.03.

| Family | Rule | Grid |
|---|---|---|
| B1 static | expanding or 60-month mean (the benchmark) | window; k 1, 2 |
| M1 momentum | trailing mean over L months | L 12, 36, 60; k 1, 2 |
| M3 one feature | quantile regimes of one feature | feature x K 2, 3 x n0 0, 12 x window; k 1, 2 |
| C1 literature rule | hold TREND only when the average forward discount is positive and both FX volatility and dispersion are below their expanding medians; else cash | none |

**Features (known at each month end).**
- **Volatility and co-movement:** RV63 (median 63-day realised volatility), RV_CHG (log RV63 / RV63 twelve months
  earlier), CORR126 (mean pairwise correlation of the seven currencies against the dollar), DISP63 (cross-sectional
  standard deviation of 63-day returns / RV63).
- **Trendiness:** ER252 (median efficiency ratio), VR5 (median weekly / daily variance ratio over two years).
- **Rates:** AFD (US 3-month rate minus the mean foreign rate), CARRY_SD (standard deviation of the eight rates),
  US3M_CHG12.
- **Dollar:** USD12 (12-month dollar return against the seven).
- **Style returns:** TREND12, REV12, CARRY6 (as in round 16), and TREND36, DIP36, CARRY36 (trailing 36-month means
  of the styles' scaled returns).

**Selection and verification, in reverse time order.**
- 1999-2025 has been looked at before for these styles (rounds 14-17); 1981-1998 never has.
- Each family's setting is therefore chosen by the best Newey-West t on 1999-2025 and verified once on 1981-1998.
  Decisions start in 1981, after five years of history.
- The walk-forward itself always runs forward in time.

**A family counts if, on 1981-1998:**
- mean > 0, t >= 1.645 and BH over the round-18 family;
- it beats B1 (chosen the same way) with difference t >= 1.645;
- for M3: its Sharpe ratio is at or above the 95th percentile of 200 time-shifted placebo runs. For C1: it beats
  always holding TREND (difference t >= 1.645).

**L3, global FX volatility and the dollar.**
- **Rule:** at each month end, long the seven foreign currencies against the dollar (each at 10% / volatility) if
  RV63 is above its expanding median, else short them.
- **Counts if:** positive in both halves (1981-1998 and 1999-2025), the pooled Newey-West t >= 1.645, and BH over
  the family.
- **Reported:** against always holding the basket long.

## The round-18 family (one BH correction, q = 0.10)

G-A1, G-A1x, G-A2, G-G2, G-G1, G-S1, G-L1, G-K, G-W1, C1, C2 (two indices), L2 (M1, M3, C1) and L3: 16 tests.
R0 is a diagnostic with its own permutation p-values.

## Code

| Part | File |
|---|---|
| G, C (per symbol) | `scripts/r18_m1.py` |
| R0 | `scripts/r18_regime0.py` |
| L | `scripts/r18_long.py` |
| Combine, BH | `scripts/r18_combine.py` |
| Workflow | `.github/workflows/r18.yml`, results on branch `round18` |
| Write-up | `docs/round18_results.md` |
