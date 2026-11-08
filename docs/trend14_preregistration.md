# Round 14: double-checking round 13, the trend variants it missed, and whether trend can be predicted (pre-registration, 2026-11-09)

Written before any data of this round is read. The user asked to:
- test everything left;
- double-check the round-13 results;
- look for missed variants;
- find out whether trend can be predicted.

Development data end on 2025-12-31. 2026 is read only in part D, a pre-registered single use of the holdout. Data,
costs, swaps, sizing, periods (fit 2008-2018, check 2019-2025), Newey-West t (3 lags) of monthly portfolio returns
and the 10%-volatility scaling are as in `docs/long_preregistration.md` unless stated.

## A. Double check of round 13

**A1. Independent re-implementation.**
- **What is new.** A second engine, written from scratch in `scripts/trend14.py`, uses daily bars instead of minute
  bars. It does not reuse any round-13 engine, indicator or system code. Each day is split at the 08:00 London
  fill: before the fill, the fill, after the fill, each with its own bid / ask high and low.
  - A stop or entry level reached inside a segment fills at the level, or at the segment's open if price gapped
    through.
  - If an entry and its stop are both reached in one segment, the trade is assumed stopped.
- **Systems re-run:** L1, L2, L4, L5, L6, L7, L8, L9, L10 and L11, the long-only benchmark BH, and Double Seven
  on weekly bars with its own exit.
- **Compared with round 13,** per system and per row (FX / non-FX): the correlation of monthly gross and net
  returns 2008-2025, and the check-period mean and t side by side.
- **The engines agree** for a system if the monthly gross correlation is >= 0.9 and the check-period means have
  the same sign. A system where they disagree is investigated before anything else is reported.

**A2. Fill time (information only).** The same systems with orders filled at the daily close (17:00 New York, mid
plus the same cost) instead of 08:00 London the next day.

**A3. A second data source and a longer history.**
- **Data:** ECB euro reference rates (14:15 CET, from 1999-01-04; stored in round 9, cut here at 2025-12-31). They
  give the 28 crosses of the eight currencies.
- **Systems:** close-only versions of L5, L6, L7, L8 and L9, and a close-based Donchian (close beyond the 100-day
  closing high or low, exit at the 50-day). All on gross returns: no costs, and no swap, because rates before
  2006 are not stored.
- **Uses:**
  - the correlation with the store-based gross results over 2008-2025 (a data check);
  - the gross returns over 1999-2007, to see whether FX trend paid before 2008 in these data (information only).

## B. Variants round 13 did not test

**Selection and count.**
- Each row picks its member by the 2008-2018 Newey-West t and is judged once on 2019-2025.
- A row counts under the round-13 part-A rule: check mean > 0 with t >= 1.645; fit mean > 0; Benjamini-Hochberg
  at q 0.10 across all part-B and part-C rows; and, for non-FX rows, an excess over BH with t >= 1.645.
- The whole speed surface is reported as well.

| Row | Rule |
|---|---|
| B1a breakout speed (FX; non-FX) | Stop-entry at the prior N-day high / low, N in {10, 20, 50, 100, 200, 250}; exit at the prior N/2-day opposite extreme; initial stop 2N; one unit |
| B1b moving-average speed | Always in, sign of the fast minus the slow average of closes, (fast, slow) in {(5, 20), (10, 50), (20, 100), (50, 200), (100, 400)} |
| B1c time-series momentum speed | As L6 with lookbacks of {1, 3, 6, 9, 12, 24} months |
| B1d EWMAC speed | One L9 crossover alone: {(2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256)}, scalars 10.6, 7.5, 5.3, 3.75, 2.65, 1.87 |
| B2 long-or-flat on non-FX (four rows: L5, L6, L8, L9) | The system long when its signal is long and flat otherwise (Faber's original form). This row counts only through its alpha against BH: monthly returns regressed on BH's, with alpha > 0 in the fit period and alpha t >= 1.645 in the check period |
| B3 currency-strength trend (three rows: L6, L9, breakout 50 / 25) | For each of the eight currencies, an index = the average daily log return against the other seven. The system runs on the eight indices, and each cross trades the difference of its two currencies' index positions, divided by 7 |
| B4 trend plus carry (FX) | Forecast = average of the L9 forecast and Carver's carry forecast (annualised rate differential / annualised volatility x 30, capped at +-20); sized and traded as L9 |
| B5 trend only in high volatility (FX; non-FX) | L9, but flat whenever the 1-month realised volatility at the month start is below its trailing 1-year median |

That gives 8 + 4 + 3 + 1 + 2 = 18 rows. The random-sign control PL2 runs alongside, as before.

## C. Can trend be predicted?

**Walk-forward for every model.**
- Retrained each January on all instrument-months before it.
- First predictions in 2011 (training from 2008). Predictions for 2011-2018 form the fit period, 2019-2025 the
  check.
- Count rule as part B: fit mean > 0, check mean > 0 with t >= 1.645, Benjamini-Hochberg across the B and C rows.

**C1. Direction.**
- **Target:** next month's return divided by the trailing 63-day volatility.
- **Features at the month-end:**
  - volatility-scaled returns over 1, 3, 6 and 12 months;
  - carry (the rate differential; minus the quote rate for non-FX);
  - the 3- and 12-month change in that differential;
  - value (log close minus its 756-day average);
  - 63-day volatility, and its ratio to the 252-day volatility;
  - 63-day skew of daily returns;
  - the close's position in its 252-day range.
- **Models:** ridge (standardised features; penalty fixed at 10 x the number of features) and gradient boosting
  (`HistGradientBoostingRegressor`: depth 3, 200 iterations, learning rate 0.05, at least 50 samples a leaf).
- **Trade:** sign of the prediction, sized as L6.
- **Rows:** 2 models x {FX, all 33}.

**C2. Trendiness (does trend following pay next month?).**
- **Target:** sign of the L9 forecast at the month-end x next month's volatility-scaled return.
- **Predictors at the month-end:**
  - the variance ratio of 5-day to 1-day returns over 252 days;
  - the lag-1 autocorrelation of daily returns (252 days) and of weekly returns (52 weeks);
  - the efficiency ratio (absolute 63-day net move / sum of absolute daily moves) at 63 and 252 days;
  - ADX(14);
  - trend strength (absolute 12-month return / volatility);
  - the 1- to 12-month volatility ratio;
  - the instrument's trailing 12-month trend P&L;
  - the FX-wide average pairwise correlation (63 days);
  - the cross-sectional dispersion of 3-month returns.
- **C2a, single predictors (information-ranked, not in the BH family above):**
  - For each predictor, the 2008-2018 pooled rank correlation with the target sets its sign.
  - The 2019-2025 monthly rank correlation is then tested in that sign: t >= 1.645, with BH at q 0.10 over the 12
    predictors.
- **C2b, switching:** ridge and gradient boosting on the predictors give the predicted trend P&L. Two actions:
  - hold the L9 trend position when the prediction is > 0, otherwise the reverse;
  - hold it when > 0, otherwise flat.
  - Rows: 2 models x 2 actions, all 33 instruments.

## D. Double Seven on weekly bars, 2026 (third use of the holdout, pre-registered)

**Rule.** Exactly the round-13 rule. The weekly close is above its 200-week average and is the lowest close of 7
weeks: buy. Sell after a 7-week highest close.

**Data and costs.**
- ECB reference rates for the 28 crosses, the only 2026 prices already stored.
- Gold and the indices are left out.
- Fill at the next ECB fixing after the weekly close; 3 bp a round trip; no swap (2026 rates are not stored).

**The development comparator** is the same ECB computation over 2019-2025. If it is not positive, the 2026 test is
moot.

**Verdicts, as in the holdout protocol, on the 2026 trades (entries from 2026-01-01, exits by 2026-10-31):**
- **Supported:** the 2026 mean net per trade is > 0 with t >= 1.645.
- **Contradicted:** the 2026 mean is below the development mean with t <= -1.645, using the 2026 standard
  deviation.
- **Otherwise inconclusive.** About 30 trades are expected.

## Order

1. A1 is run first. If the engines disagree, B and C wait.
2. D runs last, after A-C are committed.
