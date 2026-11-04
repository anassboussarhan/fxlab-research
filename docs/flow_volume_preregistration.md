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
