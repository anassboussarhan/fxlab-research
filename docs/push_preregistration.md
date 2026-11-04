# Pushing the leads: pre-registration (P batch, 2026-11-04)

Written before any of the data below was downloaded or read. The aim is to test the leads on data this project
has never looked at, either earlier years or instruments the lead was not found on, because re-tuning a lead on
the 2019-2025 sample it came from would only fit noise. 2026 is not used anywhere.

## Leads and what is done with each

| Lead (where it came from) | Why not simply trade it | What this batch does |
|---|---|---|
| B0, the frozen Gotobi basket (the only edge) | Weaker on 2023-2025 (+3.3 bp, t 1.3) | **P3:** the frozen rule on 2003-2006, before the project's data starts |
| V6 US-euro 2-year gap, EUR/USD 5 days (+12.0 bp, t 1.43) | Not significant; one pair | **P1:** the same rule on five other dollar pairs, never tested |
| Gold before the Chinese New Year (D6: +1.5% a year, 2001-2025, t 2.0) | One event a year; q 0.17 | **P2:** the same rule on 1975-2000 gold (and silver), before the sample |
| Post-fix yen reversal (L3 / E13; an outside study finds it in all yen crosses since 2020) | USD/JPY and EUR/JPY already seen | **P4:** the five other yen crosses on 2019-2025, and USD/JPY and EUR/JPY on 2003-2006 |
| GLD holdings (V2, V9: data missing) | SPDR archive now a PDF | **P5:** try other free addresses; if a table is found, run V2 / V9 exactly as registered |
| USD/JPY H4 high-confidence model (+6.9 bp on 2019-2025, flat 2010-2018) | Its 2019-2025 profit is mostly long trades in 2022 and 2024, the two strongest USD/JPY trend years: a trend / carry ride, not a new signal | Forward watch list only; no new test |
| FS1 66 settings | Lose on 2010-2018 | Forward watch list only |
| Euro COT speculative pressure (C5, t 3.1, 1 of 52 COT tests) | Pre-2004 futures prices give about 45 months: too little power to settle it | Not tested; noted |
| Pre-FOMC dollar drift as published (Karnaukh) | Needs the next-month fed funds contract history; Yahoo only has the rolling front contract, and E26 (economists' forecasts, 2017-2022) was negative | Not tested; noted |

## P1. 2-year yield gaps on five more dollar pairs

- Yields (free, daily): UK 2-year (Bank of England nominal curve), Canada 2-year benchmark (Bank of Canada Valet
  `BD.CDN.2YR.DQ.YLD`), Australia 2-year (RBA F2), Switzerland 2-year (SNB), New Zealand 2-year (RBNZ). US 2-year from
  the Treasury par curve, as in V6. A country whose daily 2-year series cannot be read is reported as data missing.
- Rule, unchanged from V6: gap = US 2y minus the foreign 2y; z = the 5-day change in the gap against its previous
  250 days; on days with abs(z) >= 1.5, trade the pair from 17:05 New York on the next business day (16:55 on
  Fridays) for 5 business days (primary) and 1 day (secondary).
- **Direction is fixed in advance as follow** (gap widening = long the dollar), the direction both V6 rows took in
  their own fit. Nothing is fitted.
- Pairs: GBP/USD, USD/CAD, AUD/USD, USD/CHF, NZD/USD. Full fills and swaps as in the V batch.
- **Primary test:** the 5-day trade pooled over the available pairs (equal weight per event day), 2007-2025.
  **Counts if** net > 0 with month-clustered t >= 1.645 on 2007-2025, and net > 0 in both 2007-2018 and 2019-2025.
  Per-pair rows and the 1-day trade are reported, not counted.

## P2. Gold before the Chinese New Year, 1975-2000

- Gold: LBMA PM price (USD), from 1975 (the first year US residents could hold gold); silver: LBMA silver price.
- Rule, unchanged from D6: long from the close 15 trading days before the Chinese New Year to the last close
  before it; return minus the risk-free rate (Kenneth French's monthly RF, spread evenly over trading days) minus
  2 bp. Years 1975-2000 (26 events), none of them in D6's 2001-2025 sample.
- **Counts if** the gold mean > 0 with t >= 1.645 (iid, one-sided) on 1975-2000. Silver and the combined 1975-2025
  series are reported, not counted.

## P3. The frozen Gotobi basket on 2003-2006

- Data: Dukascopy M1 bid/ask candles for USD/JPY and EUR/JPY, May 2003 (their first month) to December 2006.
- Rule, frozen (B0): long both from 23:00 Tokyo the evening before each Gotobi day to 09:53 Tokyo, 80 bp stop,
  fills at the ask / bid plus slippage, the same Gotobi calendar and Japanese holidays as the panel. Controls: the
  same window on every third other Tuesday-Friday business day.
- Reported: per-night mid gross, fill-based return (spread and slippage, before commission and swap, as in the
  strict-timing audit), the quoted spread at entry, by year, and Gotobi minus control.
- **Confirms** if the fill-based basket mean > 0 with t >= 1.645 (iid, one-sided). This is a confirmation of the
  existing edge, not a new edge; it is reported apart from the BH set. Early Dukascopy spreads may be wide; the mid
  gross is reported to separate the flow from the cost.

## P4. The post-fix yen reversal

- Trade: short the pair from 09:55 to 10:20 Tokyo (00:55 -> 01:20 UTC) on Gotobi days; full fills, commission.
- **P4a:** GBP/JPY, AUD/JPY, CHF/JPY, CAD/JPY, NZD/JPY on 2019-2025 (the project has not looked at this window on
  these pairs). **Primary:** pooled, equal weight per day. Counts if net > 0 with month-clustered t >= 1.645.
- **P4b:** USD/JPY and EUR/JPY on 2003-2006 (Dukascopy, as P3). Primary: pooled. Same counting rule.
- USD/JPY and EUR/JPY on 2019-2025 are shown for reference only (seen before).

## P5. GLD holdings from another free address

- If a daily GLD holdings table (tonnes) can be downloaded, V2 and V9 run exactly as in
  `docs/flow_volume_preregistration.md`, and join the BH set below. Otherwise they stay "data missing".

## Counting

Benjamini-Hochberg at q <= 0.10 over the primary rows of P1, P2, P4a, P4b (and V2 / V9 if P5 finds data). A row
counts only if it meets its own rule and survives BH. P3 is judged on its own rule. Anything that counts goes to the
forward test before any money, and its size is decided on 2007-2025 only.

**Addendum (2026-11-04, after the data probes, before any P1 / P2 / P5 result).**
- LBMA's price files refuse automated requests (HTTP 403) and the Bundesbank series key did not exist. Gold for
  P2 is taken from the Bank of England's daily US-dollar gold series if one of the candidate codes returns it;
  the Bank's notes say it is daily only from late 1979 (Friday averages before), so P2 covers the Chinese New
  Years of 1980-2000 (21 events, still none in D6's sample). Silver: data missing. If no BoE code works, P2 is
  data missing.
- New Zealand: the RBNZ files refuse automated requests (HTTP 403) and DBnomics has no copy; P1 runs on GBP/USD,
  USD/CAD, AUD/USD and USD/CHF (SNB daily 2-year Confederation yield). The 1975 start and NZD are the only changes.
- GLD: the fund page links a holdings archive API (`api.spdrgoldshares.com/api/v1/historical-archive`); if it
  returns a table with tonnes, V2 / V9 run unchanged.
