# Round 19 pre-registration: the verified gaps (2026-11-14)

Written after checking each candidate against the project's documents (grep over `docs/`), and before any round-19
data was read. Development data ends 2025-12-31. 2026 is used only in the conditional check at the end.

## What was checked and found already covered (not repeated)

- weekend gaps (negative);
- broker swap arbitrage (broker terms, not a market edge);
- cross-market lead-lag (negative);
- Obon, Golden Week, year-end (not significant);
- the 5-minute opening-range breakout at CFD costs (`untested_batch.md` 9.12);
- the first-half-hour → last-half-hour index rule and the opening-range breakout with breakeven and trailing exit
  (`untested_batch.md`, I12a / I12b);
- liquidity sweeps and order blocks on FX (`untested_batch.md` 9.11);
- toshin month-start yen selling (no effect against control days);
- month-end flows;
- trend with a carry filter (`long_results.md`, L12).

**Not testable for free:**
- new-NISA month-start flows (24 months; forward only);
- 0DTE options (no intraday option quotes);
- CME options on four more currencies (USD 60.29, not authorised).

**Left out of this round:** central-bank statement tone (candidate 7, optional; the literature puts the reaction
inside minutes).

## What I already know of the data (stated so the tests can be judged)

- **The index tests (part A) are not clean on 2019-2025:**
  - ORB5 at CFD costs was run on 2019-2025 (NAS100 gross +3.8 bp a day, t 2.2);
  - the C2b VWAP surface was seen on 2012-2025 (round 18).
  - The selection rules below use 2012-2018 only, but the clean test for part A is the 2026 check.
- **The noise-area breakout (A-NB) has never been run here.**
- Parts C to F and the library (part L) have never been run on these instruments or with these rules.
- **The round-18 reply said futures cost "about 0.3 bp".** That holds only for the Nasdaq at recent prices.
  - A tick costs a fixed number of index points, so in basis points it costs more at lower price levels:
    ES 0.25 points is 1.9 bp at 1,300 and 0.5 bp at 5,000.
  - Costs here are therefore set in points and converted at each day's price.

## Part A. Index intraday rules at CME futures costs (futures prop firms trade ES / NQ)

- **Data:** Dukascopy US500 and NAS100 CFD minute mids (they track the futures intraday), 2012-2025, 09:30-16:00
  New York.
- **Costs per side, in index points:**

  | | Ticks | Commission | Total per side |
  |---|---|---|---|
  | ES | 1 tick of 0.25 (half the spread plus slippage) | USD 2.50 / USD 50 a point = 0.05 | 0.30 points |
  | NQ | 1 tick of 0.25 | USD 2.50 / USD 20 = 0.125 | 0.375 points |

  - Each side costs these points divided by the fill price.
  - **Stress case** (reported): twice the cost.
  - CFD-cost versions are also reported.
- **Units:** one unit of notional per trade; the result is the daily return in bp.

| ID | Rule (fixed) |
|---|---|
| A-ORB5-US500, A-ORB5-NAS100 | Zarattini & Aziz (2023), as in `lit6_lab.orb5`, at mid prices. The 09:30-09:35 candle: up → long at the 09:35 open, down → short, none if open = close. Stop at the candle's low (high), target 10R, out at 16:00. Stop checked first within a minute. |
| A-NB-US500, A-NB-NAS100 | Zarattini, Aziz & Barbon (2024), "Beat the Market", base model (VM = 1).<br>- σ(m) = mean over the previous 14 days of \|close at minute m / 09:30 open − 1\|.<br>- UB = max(open, previous 16:00 close) × (1 + σ); LB = min(open, previous 16:00 close) × (1 − σ).<br>- Decisions at HH:00 and HH:30 from 10:00 to 15:30, on the close of the minute before.<br>- Price > UB → long; price < LB → short.<br>- A long exits when price < max(UB, VWAP); a short exits when price > min(LB, VWAP). Both checked only at the decision times.<br>- Fill at the next minute's open. Flat at 16:00.<br>- VWAP from tick counts, as in C2b. |
| A-C2b-US500, A-C2b-NAS100 | The round-18 C2b rule (VWAP trend decided every N minutes; grid N 15 / 30 / 60, start 09:30 / 10:00). The setting with the highest 2012-2018 net t **at futures costs** is chosen per index, then tested on 2019-2025. |

- **Also reported:** the noise-area breakout with the paper's volatility sizing (leverage at most 4, 2% target).

## Part C. Weekly dip-buying only in the carry direction

- **Universe and rules:** the Double Seven forward-test set-up (`family_d7.py`):
  - ECB fixings, 28 crosses of 8 currencies;
  - weekly close-only bars, long only;
  - fill at the next fixing;
  - 3 bp a round trip;
  - R = return / (2 × weekly N).
- **Added to the return:** the carry accrued while held, (base rate − quote rate − 0.75%) × days / 365.
- **Rates:** monthly 3-month interbank rates (OECD MEI through DBnomics, as round 18), lagged one month (the
  previous month's value is the one known).
- **Filter:** take a long entry only if base rate − quote rate > 0 at entry.

| ID | Rule |
|---|---|
| C-D7 | Double Seven with the carry filter |
| C-FAM | The seven-rule family (Double Seven, cumulative RSI, Connors RSI(2), R3, %b, CRSI, three down) with the carry filter, pooled |

- **Periods:** P1 2000-2012, P2 2013-2025 (rates from 1999).
- **t:** from monthly sums of R, as `family_d7.stats`.
- **Extra condition:** in P2 the excluded (negative-carry) trades' mean R must be below the included trades'
  mean R, with Welch t ≥ 1.645. Otherwise the filter adds nothing to the unfiltered rule.

## Part D. Oil around the weekly EIA crude-inventory report

- **Data:**
  - WTI (LIGHTCMDUSD) and Brent (BRENTCMDUSD) Dukascopy minute bid / ask, 2013-2025 (round-15 D2 downloads);
  - EIA weekly U.S. ending stocks of crude oil excluding the SPR (series WCESTUS1, the EIA history file; fallback
    DBnomics).
- **Release time:**
  - Wednesday 10:30 New York, for the week ending the previous Friday.
  - In weeks with a US federal holiday Monday to Wednesday, the release time is detected among Wednesday 10:30,
    Thursday 10:30 and Thursday 11:00, as the candidate with the largest absolute one-minute WTI move. The
    detection uses size only, never direction.
- **Surprise:** S = this week's change in stocks minus the mean change in the same ISO week over the previous
  5 years. No consensus forecast is free.
- **Entry and exit:** enter at release + 5 minutes, exit at 14:30 New York. Bid / ask fills with the project's
  slippage and commission (`flows_lab.Book.trade`).

| ID | Rule |
|---|---|
| D-E1-WTI, D-E1-BRENT | Position −sign(S): a draw (stocks below seasonal) → long |
| D-FADE-WTI, D-FADE-BRENT | Position −sign(the release-minute to release + 5 minutes move) |
| D-FOLLOW-WTI, D-FOLLOW-BRENT | Position +sign(the same move) |

- **Periods:** P1 2013-2018, P2 2019-2025; one value per release.

## Part E. Crypto perpetuals around the funding times

- **Data:** Binance USD-M perpetual BTCUSDT and ETHUSDT 1-minute klines and funding history (data.binance.vision,
  public files), 2020-2025. Funding times from the file.
- **Signal at stamp T:** the rate settled at the previous stamp (known in advance), compared with the 0.01% baseline.

| ID | Rule |
|---|---|
| E-PRE-BTC, E-PRE-ETH | Previous rate > 0.01% → short from T − 60 min to T; previous rate < 0.01% → long; equal → no trade |
| E-POST-BTC, E-POST-ETH | Previous rate > 0.01% → long from T to T + 60 min; < 0.01% → short (the rebound) |

- **Mechanism tested:** crowded longs close before paying and come back after.
- **Cost:** 5 bp per side (taker 0.045% plus slippage). Maker cost (2 bp per side) is reported.
- **Values:** one per day (the sum of that day's stamps).
- **Periods:** P1 2020-2022, P2 2023-2025.

## Part F. Prior-day high / low stop runs on indices and metals

- **Instruments:** US500, NAS100 and JPN225 CFDs; XAUUSD and XAGUSD. Data 2007 (or the first file) to 2025.
- **Trading day:** 17:00 New York roll. PDH and PDL are the prior trading day's mid high and low.
- **Event:** the first minute between 08:00 and 15:30 New York whose mid high > PDH (mid low < PDL for the other
  side).
- **Entry and exit:** enter at the next minute's open, exit at 16:00 New York. Both sides can trigger on one day.
  CFD bid / ask costs.

| ID | Rule |
|---|---|
| F-FADE-IDX, F-FADE-MET | Trade back toward the range (short above PDH, long below PDL); indices and metals averaged by day within the group |
| F-FOLLOW-IDX, F-FOLLOW-MET | Trade in the direction of the break |

- **Periods:** P1 2007-2018, P2 2019-2025.

## Family H and the counting rule

There are 22 tests (A 6, C 2, D 6, E 4, F 4) and one Benjamini-Hochberg correction at q = 0.10, using one-sided
p from the P2 t.

**A test counts if:**
- P1 mean net > 0;
- P2 mean net > 0 with t ≥ 1.645;
- it passes BH;
- for part C only, the extra condition above.

## Part L. The strategy library on instruments it never ran on

- **Instruments:** the 13 round-15 D2 instruments, which only ever got the round-13 trend systems:
  - Brent, WTI, natural gas, copper;
  - DAX, FTSE, CAC, Euro Stoxx, Hang Seng, SMI;
  - Bund, US T-bond, Gilt.
- **Plus** BTCUSDT and ETHUSDT perpetuals. Their minute bid / ask is set to mid ∓ 3.1 bp, so the engine's
  spread, slippage and commission come to about 5 bp per side.
- **Engines, unchanged:** `sweep15.py --daily` (D1 / W1 / MN) and `intraday17.py` (H4 / H1 / M30 / M15), with
  their random-state placebos.
- **Unit:** rule × timeframe × direction × instrument. P&L = the mean over the unit's settings (no selection
  inside a unit).
- **Rules only:** the daily engine's feature-tercile units are left out, as in round 17's PAIR-UNIT tier. They
  are not rules, and their fill-to-fill marks had the 14-hour look-ahead.
- **Periods:**
  - P1 2013-2018 and P2 2019-2025;
  - bonds: P1 2019-2021, P2 2022-2025;
  - crypto: P1 2020-2022, P2 2023-2025.

**A unit counts if:**
- P1 mean > 0;
- P2 mean > 0 with t ≥ 1.645 (daily values);
- its P2 t exceeds the 95th percentile of the P2 t of that instrument's placebo units (same engine);
- it passes BH at q = 0.10 over all non-placebo units of part L (its own family).

## The 2026 check (conditional)

- **When it runs:** only for a test or unit that counts. It is run once on 2026-01-01 to 2026-10-31, with
  unchanged rules.
- **Verdict:**
  - supported: mean > 0 and t ≥ 1.645;
  - contradicted: the difference from the P2 mean has t ≤ −1.645;
  - inconclusive otherwise.
- **Follow-up:** a part-A test that counts and is supported gets a futures-prop simulation, as an addendum.

Not financial advice. Nothing here is an instruction to trade.
