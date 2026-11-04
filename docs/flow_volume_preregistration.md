# Flow, volume and positioning batch (V1-V9): rules fixed before the data were analysed (2026-11-04)

**Data.** From the `flow-collect` store, downloaded but not yet analysed when this was written:
- CME futures daily volume (Yahoo front contract);
- GLD holdings;
- CFTC Traders in Financial Futures and disaggregated reports;
- SNB sight deposits;
- GDELT news tone (2017-);
- fed funds futures;
- US Treasury, MoF JGB and ECB yields;
- MoF intervention record.

Development data end 2025-12-31. **Nothing from 2026 is used.**

**Common rules.**
- **Markets:** EUR/USD, GBP/USD, USD/JPY, USD/CHF, USD/CAD, AUD/USD, NZD/USD, USD/MXN, gold, silver.
- **Execution:** traded on Dukascopy M1 bid/ask, with fills as in the flows lab. The spread is crossed, plus 25%
  of the spread and 0.1 pip of slippage each side. Commission is 0.35 bp a side. Real swaps (rate gap minus the
  0.75% mark-up).
- **Daily trades** enter at the first M1 bar at or after 17:05 New York on the day after the signal day, and exit
  at 17:05 New York N trading days later.
- **Weekly trades** run Monday 14:00 UTC -> Friday 14:00 UTC.
- **Direction (follow or fade)** comes from the fit period's gross mean.
- **Split:** fit 2007-2018, check 2019-2025. A series that starts after 2008 is split 60/40 by events.
- **Counts if:** check net > 0, t >= 1.645 (clustered by month), and Benjamini-Hochberg q <= 0.10 over every test
  of the batch.

**The tests.**

| ID | Signal | Trade |
|---|---|---|
| V1a | Front-future log volume z (vs the prior 60 days) >= 2 on a day; side = sign of that day's futures close-to-close return | That market, N = 1 and 5 days, follow / fade from fit |
| V1b | Low-volume reversal (CLS paper, futures volume as the proxy): each day, the 3 currency futures with the lowest volume z | Fade that day's return, equal weight, N = 1 |
| V2 | GLD daily holdings change / holdings, z vs the prior 60 days, abs(z) >= 2 | Gold, N = 1 and 5 |
| V3a | CFTC TFF weekly change in leveraged-fund net position / open interest, z vs 52 weeks, abs(z) >= 1.5, each currency | The pair the next week, pooled over the currencies |
| V3b | The same for asset managers | Same |
| V3c | Disaggregated managed-money change for gold and silver | Same |
| V4 | SNB sight deposits weekly change above the fit-period 80th percentile of abs(change); reported weeks, traded from the Monday after the week | EUR/CHF and USD/CHF the next week, in the direction of the change's sign, follow / fade from fit |
| V5 | GDELT news tone, 7-day mean per currency (Filippou-Taylor-Wang reversal); each Friday | Long the 2 lowest-tone currencies, short the 2 highest, against USD, one week, 2017-2025 (60/40) |
| V6 | 5-day change in the 2-year yield gap: US minus Japan (MoF JGB 2y) and US minus euro-area AAA 2y (ECB), z vs 250 days, abs(z) >= 1.5 | USD/JPY and EUR/USD, N = 1 and 5, follow / fade from fit |
| V7 | Daily change in the fed funds front-future implied rate, abs(change) >= 5 bp | USD against EUR, JPY, GBP, AUD equal weight, N = 1 and 5 |
| V8 | MoF intervention days (information only, no count) | USD/JPY over the next 5 and 20 days, in the intervention's direction |
| V9 | GLD holdings and CFTC managed money combined (both z in the same direction) | Gold, N = 5 |

**Data limits stated in advance:**
- Yahoo front-contract volume rolls between contracts, so the z uses log volume against 60 days, which absorbs
  the rolls only partly.
- GDELT has gaps where the API refused requests; a missing week is no trade.
- If a series cannot be read, the test is reported as "data missing", never replaced.

**Addendum (2026-11-04, still before any analysis).** In V7 the continuous front fed funds future rolls to the
next contract at month start, which makes the implied rate jump for reasons other than news. The first trading
day of each month is therefore excluded from V7's signal days.

**Addendum 2 (2026-11-04, after a first run; written before any rerun).** A first run exposed three data
problems, all found from data checks (event counts and file layouts), not from test results:
1. *Friday timestamps.* 17:05 New York on a Friday is after the weekly close, so every trade entering or exiting
   on a Friday was unpriced (roughly a fifth to a half of the daily trades). From now on, a Friday entry or exit
   uses 16:55 New York, the last quoted minutes of the week. Nothing else in the timing changes.
2. *Yahoo futures volume.* The continuous front-future volume drops by about 100 times from 2017 (for example the
   euro future from about 200,000 contracts a day to about 1,000), so it is not the real volume after 2016. V1a and
   V1b keep their rules and are reported as "data unusable after 2016"; their 2019-2025 check is not counted.
3. *GLD holdings.* The SPDR archive address now returns a PDF chart, not a table. V2 and V9 are reported as
   "data missing".
4. *MoF intervention file.* The parser did not match the file's layout (year and month carried down, month as a
   word, amount in 100 million yen, direction in the currency-pair column). The parser is fixed; V8 stays
   information only, USD/JPY rows only.

**New test V10 (added, not a replacement for V1).** Currency and metal ETF share volume from Yahoo (FXE, FXY, FXB,
FXA, FXC, FXF, GLD, SLV), which has no contract rolls. V10a: log ETF volume z (prior 60 days) >= 2; side = sign of
the ETF's close-to-close return that day mapped to the pair (FXE EUR/USD +, FXY USD/JPY -, FXB GBP/USD +, FXA
AUD/USD +, FXC USD/CAD -, FXF USD/CHF -, GLD gold +, SLV silver +); that market, N = 1 and 5 days, follow / fade
from fit. V10b: each day the 3 of the 6 currency ETFs with the lowest volume z, the day's return followed or faded
(from fit), equal weight, N = 1. Same entry/exit times, fills, split (2007-2018 / 2019-2025) and counting rule as
the batch; V10 joins the batch's Benjamini-Hochberg set. No ETF volume result has been looked at.
