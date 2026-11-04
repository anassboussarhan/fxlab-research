# Queue batch Q: the testing list run on data in hand (pre-registration, 2026-11-04)

Written before any of these tests' data is read. 2007-2025 only; 2026 untouched. Fills and costs as in the flows
lab (`scripts/flows_lab.py` Book: cross the quoted spread, 25% of it plus 0.1 pip slippage a side, 0.35 bp
commission a side, point-in-time rate swaps over 17:00 New York, else the flat swap) unless a test says otherwise.
Month-clustered t; one-sided. Benjamini-Hochberg at q <= 0.10 over the primary rows of VS2-VS5, T1, BS1, C5E (VS1 is
a sizing rule on an existing edge and is judged on its own rule).

## The volatility forecast used everywhere (HAR)

For each instrument: daily realised variance RV_d = the sum of squared 5-minute mid log returns over the UTC day.
Forecast of day d+1, made at the end of day d: log RV_{d+1} = a + b1 log RV_d + b5 log mean(RV_{d-4..d}) + b22 log
mean(RV_{d-21..d}), fitted once by OLS on 2007-2018 days and then fixed. sigma_hat_d = sqrt(exp(forecast)).

## VS1. Gotobi sizing for a constant breach risk

- Nights: the frozen B0 basket (long USD/JPY + EUR/JPY, 23:00 -> 09:53 Tokyo, 80 bp stop), net of commission and
  swap, with each night's worst open loss.
- Night forecast: the mean of the two pairs' sigma_hat for the UTC day of entry (known at 14:00 UTC).
- Leverage L_t = k / sigma_hat_t, clipped to [1, 10]; k set so that the average L over the 2007-2018 nights is 4.
- Compared with fixed 3x, 4x and 5x through the five prop programmes of `scripts/final_sims.py`
  (`prop_firms_ev.simulate`, 8,000 starts), on the 2019-2025 nights, at the development edge and with every night
  shifted to the 2026 mean (+2.04 bp), as in the earlier sims.
- **Use it if** the five-firm average P(paid) of VS1 is at least 3 points above fixed 4x in both edge scenarios,
  and its average daily-loss failure rate is not higher.

## VS2. Choosing the gamma by the variance ratio

- Instruments: EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD, AUD/USD, NZD/USD, gold.
- Each weekday at 12:00 UTC: VR = variance of the daily (12:00 -> 12:00) returns over the previous 60 weekdays /
  (24 x variance of the hourly returns over the same days). Position for the next 24 hours: the sign of the
  previous 24-hour return if VR > 1 (trend), its opposite if VR < 1 (mean reversion). Nothing fitted.
- **Counts if** the pooled (equal weight per day) net is > 0 with t >= 1.645 on 2019-2025 and > 0 on 2007-2018.
  Always-follow and always-fade are reported for reference.

## VS3. Gold: implied minus realised volatility

- Monthly, at the first trading day of each month (15:00 UTC): VRP = GVZ close of the previous trading day minus
  the annualised realised volatility of gold's daily (15:00 UTC) returns over the previous 21 trading days.
  Signal = sign(VRP - median of VRP over the previous 36 months). Hold gold to the next month's first trading day.
- Direction (follow or fade the signal) from the fit months (60/40 by months, as GVZ starts in 2008).
- **Counts if** the check-months net is > 0 with t >= 1.645.

## VS4. Cost-to-volatility gate on the effects that are real before costs

- Effects (rules of `scripts/lit3_intraday_lab.py` and P4): E2 gold end-of-day momentum (13:00 -> 13:30 New York
  in the sign of the move from the previous 13:30); E3 FX end-of-day momentum (14:30 -> 15:00 New York in the sign of
  the move from the previous 15:00; seven majors, pooled); K3 the dollar around the ECB fix (long the dollar 13:15 ->
  14:15 Frankfurt, short 14:15 -> 15:15; EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, pooled); PF the post-fix
  reversal (short USD/JPY and EUR/JPY 09:55 -> 10:20 Tokyo on Gotobi days, pooled).
- Gate: trade only when sigma_hat of the day / the quoted relative spread at entry is above the 70th percentile of
  that ratio over the effect's 2007-2018 trades (per pair). Direction is the effect's own; only the cut is set on
  2007-2018.
- **Counts if** the gated pooled net on 2019-2025 is > 0 with t >= 1.645. Ungated 2019-2025 rows are reported.

## VS5. Volatility-forecast jumps and the carry crosses

- AUD/JPY and NZD/JPY (AUD/USD and NZD/USD if a cross is missing). Jump = log sigma_hat for the next day minus log
  sigma_hat for the current day; z against the previous 250 days. On days with z >= 2: short both pairs from 17:05
  New York on the next business day (16:55 on Fridays) for one business day. Direction fixed (risk-off).
- **Counts if** net > 0 with t >= 1.645 on 2007-2025 and > 0 in both 2007-2018 and 2019-2025.

## T1. Slow trend on what a prop firm offers

- Instruments with M1 data: the seven majors, EUR/JPY, gold, silver, US500, NAS100, Japan 225, Brent, US Treasury
  bond and Bund CFDs (each from its first full year).
- Weekly, Friday 20:00 UTC: signal = 0.5 sign(52-week return) + 0.5 sign(EWMA(10 weeks) - EWMA(40 weeks)) of the
  weekly closes; position = signal x 10% / (annualised volatility of the previous 26 weekly returns), capped at 3;
  the portfolio is the average over the instruments available that week.
- Costs: half the quoted relative spread x 1.5 plus 0.35 bp per unit traded; financing per week: FX the rate
  differential minus 0.75% a year on the position, every other CFD 3% a year on the absolute position.
- **Counts if** the weekly portfolio net is > 0 with t >= 1.645 on 2007-2025 and > 0 on 2019-2025. Per asset class
  reported.

## BS1. Rate-expectation carry (BSIC form)

- Two-year yields: US (Treasury), Japan (MoF), euro area (ECB AAA), UK (BoE), Canada (BoC), Switzerland (SNB, to
  July 2025), Australia (RBA, if the parser reads it).
- Weekly, Friday 20:00 UTC: for each currency, D = its 2-year yield minus the US 2-year; long the currency against
  the dollar if D is above its average over the previous 60 trading days, short otherwise; equal weight; costs and
  financing as T1. 20- and 120-day averages reported, not counted.
- **Counts if** net > 0 with t >= 1.645 on 2007-2025 and > 0 in both halves.

## C5E. Euro speculative pressure before its sample

- CFTC legacy futures-only report (1986-2016 history file), euro FX futures: speculators' net = (non-commercial
  long - short) / open interest; pressure = net minus its average over the previous 52 reports, taken from the last
  report published before each month starts; hold the euro future (Yahoo continuous 6E) long if positive, short if
  negative, for the month; 1 bp a change.
- Months: October 2000 (the first futures price) to December 2003 (before C5's 2004 start).
- **Counts if** the mean is > 0 with t >= 1.645 (low power: about 39 months). The other six currencies are reported
  pooled, not counted. Note: the continuous futures series includes the roll jumps.

## P1 completion

Australia's 2-year yield (RBA F2, every sheet searched for the series row) is added to P1 and the P1 rule is
re-evaluated with it.

## Not run here (need data that is paid or not yet collected)

Option volatility selling (O1), FX volatility risk premium and risk reversals (O2), retail positioning (R1), option
expiries (X1), Karnaukh as published (K1), futures volume / CLS (F1): see `docs/testing_list.md`.
