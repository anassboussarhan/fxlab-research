# Forward test: Double Seven on weekly bars (pre-registration, 2026-11-09)

Written and committed before the first forward signal, the weekly close of Friday 2026-11-13. Nothing below is
re-tuned after this commit. A coding or data error found later is fixed and reported next to the first result,
never in its place. Paper trading only: no orders are sent anywhere.

## Why

Round 13 found the rule. Round 14 confirmed it on its pre-registered 2026 check (`docs/trend14_results.md`):
- 33 FX trades;
- +0.78R a trade;
- t 5.4 clustered by month.

It also lost heavily in the trending years 2008-2010 and 2015-2016. The forward test is the only judge left,
because every year through October 2026 has been seen.

## Data

- ECB euro reference rates (14:15 CET), downloaded each week from the ECB data API. They give the 28 crosses of
  EUR, GBP, AUD, NZD, USD, CAD, CHF and JPY.
- Weekly close = the last fixing of a Monday-Friday week.
- Fills are at the next fixing after the signal week, with 3 bp a round trip and no swap. These are the same
  approximations as the 2026 check.
- Gold and the indices are not tracked: there are no free daily fixings for them.

## Tracks

**D7-base (the edge, primary):** the frozen round-13 rule.
- Long when the weekly close is above its 200-week average and is the lowest close of the last 7 weeks.
- Out after a weekly close that is the highest of 7 weeks.
- No stop and no cap.

**D7-safe (a version a prop account could carry, secondary):** D7-base plus three rules.
- **Stop:** exit after a weekly close more than 5 weekly N below the entry, where weekly N is the 20-week EWMA of
  absolute weekly changes at entry.
- **Cap:** at most 4 open trades.
- **Currency cap:** at most 2 open trades that share a currency.
- **Order on a signal week:** exits first, then entries in alphabetical order of the pair until a cap binds.

**How D7-safe was chosen.** 40 combinations were compared on 2003-2018 only: stops of none / 3 / 4 / 5 / 6 N, caps
of none / 8 / 6 / 4, and currency caps of none / 2. The criterion, set before the comparison, was total R /
largest drawdown.
- D7-safe scored 0.523 (18.4R, drawdown 35.2R). A 6N stop with no caps came second at 0.522 (50.7R, drawdown
  97.2R).
- The plain rule scored 0.08 (13.5R, drawdown 160R).
- On 2019-2025, D7-safe made 36.6R (drawdown 23.4R) against the plain rule's 97.5R (drawdown 17.0R): the caps cost
  most of the gain there.
- This choice was made on seen data, so the forward test judges it.

**Units.** Results are in R per trade (return / (2 x weekly N / entry)). Paper sizing is 0.25% of equity per R.

## Period and tests

- **Forward trades:** entries filled from Monday 2026-11-16 onward. Positions opened before then are ignored.
- **First look:** the trades closed by 2027-11-12, about 40 for D7-base. The second look is at 2028-11-10.

| Test | Statistic | Confirmed | Consistent | Not supported | Contradicted |
|---|---|---|---|---|---|
| FD1 (primary): D7-base | mean R per closed trade, t clustered by exit month | > 0 and t >= 1.645 | > 0 | <= 0 | below the 2019-2025 mean (+0.40R) with t of the difference <= -1.645 |
| FD2: D7-safe | the same | > 0 and t >= 1.645 | > 0 | <= 0 | - |
| FD3: crash rule | D7-safe's largest drawdown in R vs D7-base's | reported, no verdict | | | |

## Power, stated before the run

With about 40 trades a year and a per-trade sd of 1.5R (2019-2025), a true mean of +0.40R gives an expected t of
about 1.7 after one year. That is about a 50% chance of "confirmed" if the edge is as large as in 2019-2025. A
crisis year like 2008 would most likely end in "not supported" or "contradicted".

## Procedure

- The job runs weekly (`.github/workflows/forward_d7.yml`, Saturdays) and publishes to the `forward-d7` branch:
  - open positions, closed forward trades and running statistics for both tracks;
  - next Monday's signals.
- The monthly forward report (the 3rd) includes it from 2026-12-03.

## Addendum (2026-11-09, still before the first forward signal): D7-family

Round 15 (`docs/round15_preregistration.md`, addendum 2) tested the whole weekly Connors dip-buying family on 2026.
The seven close-only catalog rules (Double Seven, cumulative RSI, RSI(2), R3, %b, ConnorsRSI, three down closes)
ran at their published defaults, long only. The family was **supported**: 175 trades, +0.38R, t 2.23 clustered
by month. Without Double Seven it was inconclusive (+0.29R, t 1.47).

**D7-family track (secondary):**
- Every trade of the seven rules on the 28 crosses, equal weight, same data and fills as D7-base.
- **FD4:** the FD1 table applied to the family, with a 2019-2025 comparator of +0.24R.
- **FD5 (information only):** the family without Double Seven.
