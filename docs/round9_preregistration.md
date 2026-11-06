# Round 9 tests (pre-registration, 2026-11-06)

Written before any of the data below is read. Sources of the ideas: `docs/testing_list.md` section F (round 9
gaps) and the two lists the user supplied on 2026-11-05 / 06 (AT1-AT6). 2026 is not used anywhere.

## Common rules

- Periods: fit 2007-2018, check 2019-2025 for M1 tests; other splits are stated per test.
- A test's **primary row counts** if (1) its mean net return is > 0 in the check period with month-clustered
  t >= 1.645, (2) its mean is > 0 in the fit period, and (3) its one-sided check-period p-value survives
  Benjamini-Hochberg at q <= 0.10 across all primary rows of this batch (G1 excluded, see there).
- Costs: the flows-lab Book (quoted bid / ask, slippage, commission, swaps) for spot / CFD trades unless stated.
- Every other row is information only. Rows that cannot run for lack of data are reported as "not tested (data)".

## N. Economic releases with consensus forecasts (Forex Factory archive, 2007-01 to 2025-04)

Data: HuggingFace dataset `Tropstan/Forex_Factory_Calendar` (MIT). Before any test: the time zone of the times is
fixed by checking that US Non-Farm Payrolls fall at 08:30 New York; rows without a numeric actual and forecast are
dropped (units K / M / % / B stripped).
- Currencies: USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD. Standardised surprise s = (actual - forecast) / sd of
  (actual - forecast) for the same event title over 2007-2018 (titles with fewer than 12 fit-period values are
  dropped). Sign: good news = currency up, reversed for titles containing "Unemployment", "Jobless", "Claimant".
  Several releases of one currency at the same minute: their signed s are summed.
- Pair: EUR, GBP, AUD, NZD against USD (EURUSD...), CAD, CHF, JPY via USDCAD, USDCHF, USDJPY (sides reversed); a USD
  release trades long USD in both EURUSD (short) and USDJPY (long), averaged.
- **N1 (primary):** high-impact releases with |s| >= 1: enter 15 minutes after the release in the direction of the
  surprise, exit 75 minutes after it (1-hour hold). Pooled over currencies. Info: entry +5 / exit +65 min; entry +15
  / exit +240 min.
- **N2 (primary):** surprise index per currency I = sum over high- and medium-impact releases of the last 90 days of
  s x exp(-age / 30 days). Every Friday 20:00 UTC: long the two currencies with the highest I, short the two with
  the lowest (USD counted as a currency with return 0), hold one week; each leg traded in its USD pair.
- **N3 (primary):** pre-release drift: long the releasing currency from 4 hours to 15 minutes before each
  high-impact release (one trade per currency per release time).

## G1. The Gotobi trade on CME yen futures (futures prop firms)

Futures prop firms (e.g. Topstep: flat by 3:10 PM CT, reopening 5:00 PM CT) allow only the part of the night after
the reopen. Rule: the B0 nights (same calendar), long USD/JPY (short 6J) from 17:05 Chicago time on the evening
before to 00:53 UTC. Cost (primary): 1.5 ticks of 6J (1 tick at the thin reopen, 0.5 at the exit) plus $5 a round
trip per contract (notional 12.5 million yen). Info: the spot Book cost, zero cost, EUR/JPY spot leg.
G1 is a part of the confirmed B0 trade, not a new edge: it is judged on its own (mean > 0, t >= 1.645 on 2019-2025,
> 0 on 2007-2018) and kept out of the BH set. If it holds, a futures-prop simulation follows.

## O / AT6. Signals from the CME euro and yen options (Databento, June 2010 - 2025)

Fit 2010-2017, check 2018-2025 (as O1). Settlements, open interest and definitions as in O1; futures returns from
the F1 bars; option expiries used: the monthly ones in the files.
- **O3 (primary):** 25-delta risk reversal (IV of the 25-delta call minus the 25-delta put, interpolated in delta, on
  the expiry 10-45 days away) every Friday. If it rose over the week, long the currency's future (6E: EUR up; 6J: JPY
  up) to the next Friday settlement; if it fell, short. Cost 1 tick a side on each change. Pooled EUR and JPY.
  Info: contrarian on the 52-week z-score of the level.
- **O4 (primary):** expiry pinning. On each monthly expiry day: the strike with the largest open interest (calls +
  puts, previous day) within 0.5% of spot at 07:00 New York, converted to spot terms with the previous
  settlement's futures-spot basis; if it is at least 10 bp away, trade spot EUR/USD or USD/JPY toward it from 07:00 to
  09:55 New York (the cut is 10:00).
- **O5 (primary):** gamma switch. Daily dollar gamma of all open interest (expiries 1-60 days, previous day's OI,
  each expiry's ATM IV for every strike), ranked against its trailing 250 days: top third -> fade the 08:00-12:00
  New York move from 12:00 to 15:55; bottom third -> follow it; middle -> flat. Spot EUR/USD and USD/JPY, pooled.
- **AT6 (primary):** carry hedged with options: each monthly cycle (as O1), short the 6J future (long USD/JPY with
  its carry) and buy the 6J call (a yen call) with the strike nearest 4% above the future, held to expiry. Costs:
  1 tick round trip on the future, 4% of the option premium. Info: unhedged short 6J, skew, worst cycle.

## F2. Signed order flow in CME currency futures

First a free Databento price check of the `trades` schema for 6E and 6J. Downloaded only if it fits in the remaining
free credit (<= $45); if the full 2010-2025 span does not fit, the longest recent span that does, split 60/40.
Rule (primary): daily signed volume imbalance (aggressor buys minus sells over total, front contract, CME trade
date); above its trailing 250-day 80th percentile -> long the currency for the next day (spot 21:00 -> 21:00 UTC),
below the 20th -> short.

## S. Smaller free tests

- **S1a (primary):** same-calendar-month seasonality (Keloharju et al.): FRED H.10 daily rates from 1971 (JPY, GBP,
  CHF, CAD, AUD, NZD, SEK, NOK, DKK, DEM spliced into EUR in 1999), monthly spot returns against USD. Signal = the
  currency's mean return in the same calendar month over the previous 20 years (at least 10). Long the top 3, short
  the bottom 3; 2 bp a side on changes. Fit 1991-2018, check 2019-2025.
- **S1b (primary):** realised skewness: the 7 currencies with rates (EUR, GBP, JPY, CHF, CAD, AUD, NZD vs USD),
  monthly excess returns (spot + rate differential), signal = skewness of the last 12 months of daily returns; long
  the 3 most negative, short the 3 most positive; 2 bp a side. 2007-2018 / 2019-2025.
- **S2 (primary):** Shanghai gold premium (SGE Au99.99 in USD per ounce minus London gold) above its 20-day mean ->
  long gold the next day (21:00 -> 21:00 UTC), else short. Split 60/40 by days if the history starts after 2007.
- **S3 (primary):** Norges Bank's monthly announcement of its daily foreign-currency purchases for the oil fund:
  if the daily amount (NOK; purchases of foreign currency positive) rises from the previous month, long USD/NOK
  from 07:00 UTC on the first business day of the month to 16:00 UTC on its last business day, else short.
- **S4 (primary):** Bank of Japan JGB purchase operations: on operation days, long USD/JPY from 10:12 to 12:00 Tokyo.
  Split 60/40 if the record starts after 2007.
- **S5 (primary):** coupon conversion: short USD/JPY from 09:05 to 15:00 Tokyo on the first Tokyo business day after
  each mid-quarter US Treasury coupon date (15 February, May, August, November, moved to the next US business day).
  Info: the same on month-end coupon dates.

## AT. The user's lists (2026-11-05 / 06)

- **AT1 (primary):** gold mean reversion with a hard cost gate (XAU/USD M5 from M1 mids): z = (close - SMA20) / SD20;
  at the M5 close, if |z| > 2.2 and SD20 >= 3 x the round-trip cost (spread + slippage + commission at that bar),
  fade at the next minute: target the SMA20 at signal (it is >= 4 x cost by construction), stop 1.5 x SD20 beyond
  the entry, time exit after 60 minutes; one position at a time; not between 20:55 and 22:05 UTC. Primary at 1x
  cost; info at 1.5x and 2x cost.
- **AT2 (primary):** Hurst rule on the gold / silver spread: R = ln(XAU) - ln(XAG) at 21:00 UTC; H = rescaled-range
  Hurst exponent of the last 250 daily changes of R. H < 0.45: fade the 60-day z of R when |z| > 1; H > 0.55: follow
  the 20-day change of R; else flat. Long R = long gold, short silver, equal notional, held one day; spread,
  commission and the flat swap on both legs.
- **AT3 (primary):** carry with a turnover penalty: 7 currencies vs USD, monthly; mu = the rate differential, Sigma =
  36-month covariance of excess returns shrunk 50% to the diagonal; cost-blind mean-variance weights scaled to 10%
  ex-ante volatility; the penalised portfolio maximises mu'w - (gamma/2) w'Sigma w - kappa x |w - w_prev| with the
  same gamma and kappa = 3 bp. Excess returns from FRED month-end rates plus the rate differential; 3 bp a side on
  turnover. 2007-2018 / 2019-2025. Info: the cost-blind portfolio and plain top-3 / bottom-3 carry.
- **AT4 (primary):** USD/JPY London-open momentum: the sign of the 08:00 -> 08:30 London move, traded from 08:30 to
  16:00 London. Info: exit 12:00 London; the same on EUR/USD, GBP/USD, AUD/USD, USD/CHF (the source says they fail).
- **AT5 (primary):** news-tone LASSO: monthly GDELT tone of each country's news (US, euro area as Germany + France,
  Japan, UK, Switzerland, Canada, Australia, New Zealand; GDELT DOC timeline from 2017) and each currency's forward
  discount as features; one LASSO per currency (cross-validated penalty) predicting next month's excess return,
  refitted each month on an expanding window starting with 36 months; long the 3 highest forecasts, short the 3
  lowest. Check period: every forecast month (2020-2025); counts if t >= 1.645 (no fit-period condition).

**Addendum 1 (2026-11-06, after the first M1 run; rules otherwise unchanged).** S5's 09:05 Tokyo entry falls in the
M1 store's hour-0 gap (no bars 00:00-01:00 UTC before 2019, `docs/flows_lab.md`), so no 2007-2018 trade was
priced. S5 is rerun with entry at 10:05 Tokyo for every year. F2: the price check gives $261 (6J) and $456 (6E) for
2010-2025 trades, $88 / $126 for 2019-2025; the span that fits in about $45 of the remaining credit is about 1.4
years of both contracts (or 3.5 years of 6J only), too short for a daily test to have power, so nothing was bought
and F2 is reported as not tested (the user can decide whether 3.5 years of 6J for about $44 is worth it).
