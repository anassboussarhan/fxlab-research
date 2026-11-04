# Free retail-flow and daily speculative-positioning tests (pre-registration, 2026-11-04)

The paid datasets (Nasdaq "Global Retail FX Flow", TraditionData's Gaitame flow, Marex "Daily Speculative
Positioning") are replaced by free equivalents. Written before any of these series is read. 2007-2025 only; fills
and costs as in the flows lab; one-sided; month-clustered t (weekly / daily) or iid t (monthly); directions fixed
here from the published mechanism, nothing fitted. BH at q <= 0.10 over the primary rows.

## FF1. Small speculators in the CFTC report (free retail proxy)

- CFTC Traders in Financial Futures, "non-reportable positions" (traders below the reporting threshold) in the
  seven currency futures: net = (long - short) / open interest; z = net against its previous 52 weeks.
- **Primary (crowding):** when abs(z) >= 1.5, trade against the small traders for the next week (Monday 14:00 ->
  Friday 14:00 UTC after the Friday release), pooled over the currencies.
- Secondary: the same on the weekly change of net (as V3), and gold / silver from the disaggregated report.
- **Counts if** the pooled net is > 0 with t >= 1.645 on 2007-2025 and > 0 in 2007-2018 and 2019-2025.

## FF2. Japanese retail open positions (FFAJ monthly statistics, free)

- Financial Futures Association of Japan, OTC FX margin statistics (`open_position_with_mc.xls` and
  `trading_vol_and_position.xls`): month-end open long and short positions of Japanese retail traders by major
  currency.
- For each currency c with both sides reported (USD, EUR, GBP, AUD, NZD, as available): share n = (long - short) /
  (long + short); signal = the change of n over the month. **Fade it** (retail adding to longs -> short c/JPY).
- The figures are published around the 15th of the next month, so each position runs from the first business day
  on or after the 16th (14:00 UTC) to the same point a month later; pairs USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY,
  NZD/JPY, pooled per month.
- **Counts if** the pooled net is > 0 with t >= 1.645 over all months and > 0 before and after 2019. If the files
  cannot be read, FF2 is reported as data missing.

## FD1. A free daily speculative-positioning estimate (in place of Marex MDSP)

- For EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD, AUD/USD, NZD/USD, gold and silver: daily mid at 16:55 New York.
  Trend-follower position (Macrosynergy-style): for lookbacks of 80, 160 and 260 trading days, mean daily return /
  standard deviation of the last 63 daily returns x sqrt(lookback), capped to [-1, 1]; the position is the average
  of the three (between -1 and +1). Before testing, its weekly values are correlated with the CFTC leveraged-funds
  (currencies) and managed-money (metals) net positions; the correlations are reported, not used.
- **FD1a, the flow:** on days when abs(change of the position) is above the 80th percentile of its 2007-2018 values
  (per instrument), trade in the direction of the change from 17:05 New York (16:55 on Fridays) to the next
  business day's 17:05: the trend followers' own orders arrive the next day.
- **FD1b, crowding:** when abs(position) first reaches 0.95, trade against it for five business days from 17:05 New
  York.
- **Count if** the pooled net is > 0 with t >= 1.645 on 2019-2025 and > 0 on 2007-2018.

## Not run here

OANDA's historical position books (free with a practice account; the `time` parameter of the position-book
endpoint returns past snapshots) need the user's practice token as the `OANDA_TOKEN` secret. Dukascopy's SWFX
sentiment index has no documented history download.
