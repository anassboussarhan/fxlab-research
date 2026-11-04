# Tests on the Databento CME data (pre-registration, 2026-11-04)

Written before any of the downloaded data is read. Data: CME Globex (GLBX.MDP3) from June 2010 to December 2025,
bought with the free Databento credit (fed funds futures, every contract of ten currency and metal futures, euro
and yen futures options with definitions and settlements). 2026 is not used.

## F1. V1a / V1b with real futures volume

The pre-registered V1a and V1b rules (`docs/flow_volume_preregistration.md`) are rerun unchanged with the real daily
volume of each product (the sum over all its contracts, CME trade date), in place of Yahoo's broken series. The
series starts in June 2010, so the split is 60/40 by events. Counting as in the V batch; V1a / V1b join no earlier
BH set (they are judged as one batch of their own rows).

## K1. The dollar before FOMC rate changes (Karnaukh, as published)

- Meetings: scheduled FOMC decisions June 2010 - December 2025 (dates from the Federal Reserve's calendar pages).
- Signal, three business days before the meeting: expected target = 100 minus the price of the fed funds contract
  of the month after the meeting (the "next-month" contract); spread = expected target minus the target midpoint
  in force (the front contract's implied rate when the current month has no meeting is used if the target cannot
  be read). Spread >= +12.5 bp: long the dollar; <= -12.5 bp: short the dollar; otherwise no trade.
- Trade: an equal-weight dollar basket against EUR, JPY, GBP, CHF (EUR/USD, USD/JPY, GBP/USD, USD/CHF M1 fills and
  costs as in the flows lab) from 14:00 New York two business days before the announcement to 15 minutes before the
  scheduled statement time.
- **Counts if** the mean net per trade is > 0 with t >= 1.645. Reported also: all meetings with no expected change
  (information), and the post-publication part (2016-2025).

## O1. Selling euro and yen volatility when it is rich

- Implied volatility: from the daily settlement prices of the CME euro and yen futures options (Black-76 on the
  settlement of the underlying future), the call and put nearest to the money of the option expiry closest to 30
  calendar days away (at least 10 days left); ATM implied volatility = the average of the two.
- Forecast: the HAR forecast of the next month's realised volatility of the futures (daily RV from the futures'
  daily returns averaged over the past 1, 5 and 22 days, coefficients fitted on 2010-2017 data only and then fixed).
- Trade, at each monthly expiry cycle: sell one ATM straddle (call + put at the strike nearest to the future) at
  settlement, delta-hedged daily with the future at settlement, held to expiry. P&L in percent of the future's
  notional. Costs: 4% of the straddle premium for the option round trip plus 0.5 tick per future traded.
- **O1a (count):** sell only when implied volatility exceeds the forecast by more than 1 volatility point; counts if
  the mean P&L per cycle is > 0 with t >= 1.645 over 2018-2025 (the check period after the forecast's fit years),
  pooled over EUR and JPY.
- Reported: always selling (O1b), the unhedged straddle held to expiry (O1c), and the worst cycle (tail risk).

These are personal-account strategies (prop CFD accounts do not offer options); they are judged as research only.
