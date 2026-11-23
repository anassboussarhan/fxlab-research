# Round 24: what was missed and can be tested for free (pre-registered 2026-11-22, before any data was read)

**What this round runs:** the list agreed on 2026-11-22 ("Go ahead"):
- items 1-3 close faults and gaps in round 23;
- items 4-11 are new free data and folklore checks.

Items 5 (insider buying), 11c (Congress trades), 12 (OANDA positioning) and 13 (small caps) are blocked; see
"Blocked".

**Rules for data:**
- development data only (to 2025-12-31);
- 2026 only as stated under "2026";
- no paid data;
- requests carry the project User-Agent only (`fxlab-research/1.0 (research data collection)`), no e-mail.

**What was read before writing this:** the round-24 access probe (`ci/r24_probe.py`, run 2026-11-22). It records
only HTTP status codes and byte counts. No data values from any source below had been read.

## The selection core (every part, as round 23)

- **Walk-forward:** `scripts/wf23.py`, 120 settings:
  - window 12, 24, 36 or 60 months, or expanding;
  - top 1, 3, 5 or 10;
  - re-selection monthly, quarterly or yearly;
  - score t or mean.
- **Two streams per unit, both counted:**
  - **default:** expanding window, top 5, yearly, t;
  - **nested:** each January from the inner start + 3 years, the setting with the best inner t over all earlier
    years.
- **Configurations:** every parameter is free, drawn by scrambled Sobol (128 per unit unless stated). No value is
  fixed by hand except where a part says why (data definitions, costs).
- **Counting rule:**
  - the first out-of-sample period up to 2018 has mean > 0, if it exists;
  - 2019-2025 has mean > 0 with t ≥ 1.645;
  - Benjamini-Hochberg at q = 0.10 over all round-24 real streams.
- **Placebos:** each part has one (below). Placebos go through the same selection and are reported, not counted.
  A real stream that counts while its part's placebos meet their own rule at more than twice the real rate is
  examined for a fault before it is reported.
- **Sanity checks before results are reported** (added after round 23):
  - **Look-ahead:** every new or changed signal function passes the truncation test of `scripts/r23_lookahead.py`
    (signals identical on every bar before 48 cuts) before its part is run.
  - **Implausible streams:** any unit with 2019-2025 t above 8 is examined for a fault before it is reported.

## R1. Carry in every engine (the round-23 USD/TRY fault)

- **Engine change.** `financing="rates"` now works in every engine, not only the vector one: order, passive and
  trade engines (stops, targets, trailing, partial exits).
  - Financing per rollover = (|exposure| × mark-up − exposure × rate differential) / 360.
  - The mark-up is the existing 0.75% a year per side.
  - The rate differential is base minus quote, point-in-time: month m's average is used from the first day of
    month m + 1.
- **Rates:** OECD MEI 3-month interbank (IR3TIB01, else call money IRSTCI01) from DBnomics, monthly, for every
  currency of the FX instruments.
  - CNH uses the CNY series.
  - Metals: XAU and XAG yield nothing, so a long metal position pays the dollar rate plus mark-up.
  - A currency with no OECD series (SGD) has its pairs left out of R1 and reported.
- **What runs:** part LW of round 23, unchanged except for financing, on every FX pair and metal:
  - the 30 round-22 FX and metal instruments and the 14 wave-2 currency pairs;
  - every timeframe M15-W1;
  - the same Sobol configurations (same seeds), runner and placebos.
- **USD/TRY is counted again in R1.** Indices, commodities, bonds and crypto keep the flat swap; their round-23
  results stand and are not rerun.

## R2. The two session rules with tied hour constants

- **Rules:** S.SES.SESSION_MOMENTUM and S.SES.NY_OPEN_REVERSAL.
- **The fix:** every hour constant in their code is replaced by its default plus one shared shift s, an integer in
  [−8, 8], drawn by Sobol. A common shift keeps the reference hour before the trading window for every draw.
- **Everything else as LW:**
  - the rules' own parameters free;
  - continuous execution;
  - carry from R1 on FX;
  - every instrument of round 23;
  - M15-H12.
- **Pre-condition:** both rewritten rules pass the truncation test. If either fails, it is not run and that is
  reported.
- **Placebos:** R1's LW placebos.

## R3. Completeness

- **S.FAMOUS.TTM_SQUEEZE:** in round 23 every draw had `min_periods` above its window. Here the window is drawn
  first and `min_periods` is drawn within [1, window]. Run as LW on every instrument and timeframe, with carry on
  FX.
- **IC-SMT:** its round-22 function returns trades instead of placing orders through the recorder. It is ported to
  the recorder (market entry at the next open and time exit, as round 22) and run as IC on its pairs.
- **Placebos:** R1's LW placebos for TTM; IC's random-entry placebos for SMT.
- **S.RG.HMM below H2:** not rerun. It is stuck in compiled code there at any budget the runners allow.

## R4. Earnings-announcement drift without forecasts (US stocks)

- **Universe:** the current S&P 500 members, with CIK, from Wikipedia's list.
  - Survivorship bias: companies that left the index are missing, which flatters long positions. The long-short
    and market-neutral forms are therefore the primary ones.
- **Earnings dates:** 8-K filings that list Item 2.02 (results of operations), from the SEC submissions API
  (data.sec.gov), 2009-2025.
  - A filing accepted after 16:00 New York time dates to the next trading day.
  - Duplicates within 5 trading days keep the first.
- **Prices:** daily adjusted closes and opens (Yahoo), 2008-2025.
- **Event signal:** the announcement-day abnormal return, the stock's return minus SPY's.
- **Portfolio:**
  - each day, the events of the last D days (D drawn 1-5) are ranked by the signal;
  - long the top fraction q and short the bottom q (q drawn 0.1-0.5), or long-only against SPY;
  - entry at the next open after a delay of 0-5 days;
  - hold H trading days (H drawn 1-60);
  - equal weight over open positions.
- **Free parameters:** D, q, delay, H, direction (drift or reversal), form (long-short / long-only vs SPY), and a
  liquidity filter (all, or above / below the median dollar volume of the last 60 days).
- **Costs:** 5 bp per side (large-cap US stock CFDs at a prop firm).
- **Periods:** P1 2009-2018, P2 2019-2025.
- **Placebo:** the same portfolios with the event dates shifted by a random 20-120 trading days per stock.

## R6. Daily short-sale volume (FINRA)

- **Data:** FINRA consolidated daily short-sale volume files (Reg SHO), 2009-08 to 2025-12, for the R4 universe.
- **Signal:** short-volume ratio = short volume / total volume, as a level averaged over L days (L drawn 1-20), or
  its change against the previous L days.
- **Portfolio:**
  - cross-sectional ranks each day;
  - long the low-ratio fraction q and short the high-ratio fraction q (q drawn 0.1-0.5), or the reverse;
  - hold H days (H drawn 1-20);
  - rebalance each day or each week;
  - equal weight, market-neutral by construction.
- **Costs:** 5 bp per side. **Periods:** P1 2009-2018, P2 2019-2025.
- **Placebo:** the ratio shuffled across stocks within each day.

## R7. Crypto quarterly-futures basis

- **Data (Binance public archive):** hourly closes of the BTC and ETH quarterly contracts, coin-margined (BTCUSD /
  ETHUSD_yymmdd) and USDT-margined (BTCUSDT / ETHUSDT_yymmdd), and of spot BTC / ETH against USDT.
- **Trade:**
  - long spot, short the quarterly, when the annualised basis (F / S − 1) × 365 / days to expiry is at least θ;
  - exit at expiry, when the basis falls below an exit level, or by rolling to the next quarterly;
  - marked to market daily.
- **Free parameters:** θ (0-40% a year), exit level, entry window before expiry (any day, or within N days, N drawn
  7-90), contract (front or next), margin type (coin or USDT).
- **Costs:** 5 bp per leg per side (taker).
- **Periods:** all data from 2020 on, so P2 only.
- **Placebo:** the same trade entered on random days, ignoring θ.

## R8. Perpetual funding across venues

- **Venues** (those that answered the probe):
  - Binance (public archive);
  - Deribit, Kraken Futures, Hyperliquid, dYdX v4, Gate.io (public APIs).
  - OKX serves only recent months and is left out.
- **Data:** funding rates and hourly prices for the BTC and ETH perpetuals.
- **Trade:** short the perpetual on the venue with the highest expected funding and long it on the lowest, when the
  expected spread per 8 hours is at least θ.
  - Expected funding: the last rate or the mean of the last n.
  - Held until the spread falls below an exit level.
  - Price differences between venues are marked hourly.
- **Free parameters:** θ, exit level, n (1-21), venue set (any pair or all).
- **Costs:** 5 bp per leg per side.
- **Periods:** P2 only (most venues start 2019-2023).
- **Placebo:** venues assigned at random at each entry.

## R9. The bitcoin "CME gap"

- **Data:** Binance BTC/USDT minute bars, 2017-08 to 2025-12.
- **Gap:** spot at the CME Sunday 17:00 Chicago open divided by spot at the Friday 16:00 Chicago close, minus 1
  (the weekend move).
- **Trade:** from the Sunday open, fade the gap (or follow it). Exit when the Friday close is reached (a fraction f
  of the way, f drawn 0.25-1), at a stop of k times the gap, or after H hours. Only gaps with |gap| ≥ g.
- **Free parameters:** g, direction, f, k (0.5-5), H (1-120).
- **Costs:** 5 bp per side.
- **Periods:** P1 2017-08 to 2018, P2 2019-2025.
- **Placebo:** the same trade from a random hour of the week.

## R10. Fed "net liquidity"

- **Data:** FRED WALCL (Fed balance sheet), WTREGEN (Treasury General Account) and RRPONTSYD (reverse repo).
  - Net liquidity = WALCL − WTREGEN − RRP, weekly at the Wednesday level.
  - Known after the Thursday 16:30 New York release, and traded from the Friday open.
  - If FRED cannot be reached after retries, R10 is not run and that is reported.
- **Signal:** the change over L weeks (L drawn 1-26), as a sign or as a z-score above a threshold.
- **Trade:** long (or long / short) the Nasdaq CFD (Dukascopy USATECHIDXUSD daily) or BTC/USDT, held to the next
  release.
- **Free parameters:** L, threshold, form, side.
- **Costs:** the round-23 CFD cost for the Nasdaq, 5 bp per side for BTC.
- **Periods:**
  - Nasdaq: P1 2012-2018, P2 2019-2025;
  - BTC: P1 2017-08 to 2018, P2 2019-2025.
- **Placebo:** the signal shifted by a random 26-104 weeks.

## R11. Stock folklore

**a. ETF pairs trading** (distance method):
- **Universe:** the nine original SPDR sector ETFs (1998-) and the iShares single-country ETFs listed by 1996, with
  no survivorship problem.
- **Formation:**
  - normalised prices over F days (F drawn 60-500);
  - pairs with the smallest squared distance;
  - top n traded (n drawn 1-20).
- **Trade:** open when the spread is beyond k standard deviations (k drawn 1-3); close at a fraction of the way to
  zero, or after H days.
- **Re-formation:** monthly.
- **Costs:** 2 bp per side.
- **Periods:** P1 1999-2018, P2 2019-2025.
- **Placebo:** random pairs.

**b. S&P 500 additions:**
- **Data:** Wikipedia's table of changes, effective dates 2008-2025. Announcement dates are not reliably listed, so
  the effective date anchors the window.
- **Trade:** buy (or short) each added stock from e days before to x days after the effective date (e drawn 0-10,
  x drawn −5 to 20), against SPY.
- **Costs:** 5 bp per side.
- **Periods:** P1 2008-2018, P2 2019-2025.
- **Placebo:** random dates for the same stocks.

## Blocked (not run)

| Item | Why |
|---|---|
| 5, insider buying | SEC form 4 details are served only from www.sec.gov/Archives, which refuses requests without a contact e-mail in the User-Agent (HTTP 403); the standing rule bars using Anas's e-mail |
| 11c, Congress trades | the House and Senate Stock Watcher datasets return HTTP 403 |
| 12, OANDA positioning | needs an OANDA practice token as the `OANDA_TOKEN` secret |
| 13, small caps | needs Alpaca keys |

## 2026

Not used for anything above. A stream that counts gets one pre-registered check on 2026-01-01 to 2026-10-31 with
its frozen rule, written before 2026 data is read.

Not financial advice.

## Addendum 1 (2026-11-22, before any round-24 run): how TTM Squeeze is tied

- **What failed in round 23:** checking TTM's code showed the cause was not only `min_periods` above the window.
  Its momentum is a regression over `np.arange(20) - 9.5`, applied to a rolling window of another 20. Once the two
  20s were drawn separately, every configuration failed on the length mismatch.
- **Tied by role instead** (`scripts/r24_tied.py`):
  - L (6-60, log) for every 20-bar length, with (L - 1) / 2 as the regression centre;
  - b (1-3.5) for the Bollinger multiplier;
  - q (2-6) for the falling / rising run;
  - k and m as before;
  - the two midpoint divisors (/ 2) stay: they define an average.
- **The session rules** are tied exactly as written in R2:
  - SESSION_MOMENTUM: h and s;
  - NY_OPEN_REVERSAL: k, s, the ATR length and the √8 scale, the latter two free as in round 23.
- **Checks:** at their defaults all three reproduce the library rules exactly (M15 and H1, synthetic data). The
  truncation test at M15, H1 and H2 (32 configurations each) finds no look-ahead and no failed configuration.

## Addendum 2 (2026-11-22, before any round-24 result was read): rates source and four details

1. **Rates source (R1).** The first R1 run (45027360273) was cancelled before any job finished. Its coverage
   summary showed two problems:
   - the OECD MEI series end in January 2024;
   - the Turkish interbank series ends in April 2008.

   Carried forward, these would have put stale rates on 2024-2025. R1 now uses the BIS central-bank policy rates
   (WS_CBPOL, monthly; China's rate for CNH), still with a one-month lag. SGD has no policy rate and its pair stays
   out. A currency without a series to 2025-11 stops the run.
2. **R11b window.** The announcement precedes the effective date by a few days that the table does not give, so
   an entry up to 10 days early could anticipate an announcement not yet made. The entry is therefore restricted
   to e ∈ {0, 1} trading days before the effective date. The exit x is drawn 1-20 days after it. This tests the
   effective-date move and the post-inclusion drift or reversal, not the run-up after the announcement.
3. **R6 data.** FINRA's Reg SHO files cover volume reported to FINRA facilities (off-exchange), not all US
   volume. The ratio is that share, as the free data define it. Returns run open to open from the day after the
   signal.
4. **Smaller definitions:**
   - **R7:** coin-margined contracts are marked with the same linear approximation as USDT-margined ones (equal
     coin quantity on both legs).
   - **R8:** a venue switch pays 4 legs.
   - **R9:** the placebo enters at a random minute of the surrounding week, with its "gap" measured over the
     previous 49 hours.
   - **R10:** the entry bar is the first daily bar opening after the Thursday 16:30 New York release (CFD FX-day bars
     open at 17:00 New York).

## Addendum 3 (2026-11-22, after the first r24x jobs; no R1, R2, R4, R6, R7, R8, R10 or R11 result had been read)

Already seen at this point, in the job logs: the R3 SMT and R9 CME-gap results (none meets the own rule). Nothing
below changes them.

1. **Rates (R1) checked.** The BIS coverage run (45027533950) has every currency to 2025-12, SGD excepted as
   planned. JPY has 206 months from 2000-08: the Bank of Japan had no policy-rate target in 2001-2006 and
   2013-2016. Months without a value carry the last published rate, as a trader would have known it. The rate
   then was 0.1-0.25% against a market rate near zero, so the error is under 0.25% a year on JPY pairs.
   R1-R3 relaunched on these rates, 16 jobs at a time so the other parts keep runners.
2. **R4, R6, R11 data job fixed.** The Wikipedia page now has a navigation table first, so the tables are taken by
   their columns (Symbol / Security / CIK for members; Added-Ticker and the effective date for changes), not by
   position. The live article no longer has the changes table (only the members and a navigation box), so the
   changes come from the article's last revision before 2026-01-01 that has it (Wikipedia's revision API); the
   members stay the live list, as planned. The revision used is recorded in `data_meta.json`. Nothing else changes;
   the four stock parts rerun (workflow r24xs).
3. **S&P 500 CFD data end in 2024.** The minute store behind round 22 (run 41447353092) holds USA500IDXUSD only
   to part-way through 2024 (2024 has about four months). That has applied to every USA500IDXUSD result since
   round 22 and was not recorded. Its 2019-2025 period is therefore 2019 to early 2024. Round 24 keeps it with
   this note. Round 25 refills 2024-2025 from Dukascopy minute candles (the round-23 downloader, free) and reruns
   the instrument there.

## Addendum 4 (2026-11-22, after the R4, R6 and R11 logs; the R6 result is set aside as a fault)

1. **R6 looked ahead by one day.** The first R6 run (r24xs 45035531557) gave 2019-2025 t of 3.8 (default) and 10.6
   (nested), which the sanity rule sends for examination. The cause is in `sv_run`'s booking: the position formed on
   day t's FINRA file (published after day t's close) earned the open of t to the open of t + 1, which contains
   day t's own session. The pre-registered timing (addendum 2, item 3) is the open of t + 1 to the open of t + 2.
   A synthetic check (a signal equal to day t's open-to-close move) gives t = 194 with the old booking and
   t = -0.6 with the fixed one. That first R6 result is void.
2. **R6 calendar.** The rerun uses the stock trading calendar from the first FINRA day. A day whose FINRA file is
   missing (HTTP 404) has no signal, so a return never spans two days; the first run used FINRA days only.
3. **Rerun:** R6 only (workflow r24xv), on the stock data of r24xs. Nothing else in R4, R6 or R11 changes. The R4
   and R11 timing was checked the same way:
   - R4 enters at the open after the announcement day (or later);
   - R11b holds from the close before the first return day;
   - R11a enters at a close and earns from the next day.
4. **Round 25 note:** the forked workers of round 25 run with one OpenMP / BLAS thread. The HMM rule's k-means
   deadlocked in forked workers once the parent process had started OpenMP threads. That is a likely cause of the
   round-23 HMM stalls below H2, recorded in the round-25 pre-registration.

## Addendum 5 (2026-11-22, after the first R8 and R10 logs)

1. **R8 did not implement three pre-registered definitions.** The first runs (45029171317, rerun 45040695736) gave
   2019-2025 t of 6.5-13.6 (the ETH nested stream above 8). Examining it:
   - **Prices:** the pre-registration marks the price difference of the two venues' perpetuals; the code counted
     funding only. Each venue's daily 00:00 UTC prices (Binance archive; Deribit, Kraken, Hyperliquid, dYdX and
     Gate.io public candles) now mark the legs open to open. Positions change only at day boundaries, so daily
     marks give the same total as hourly ones. A venue without prices is left out and reported.
   - **θ** is now compared with the expected spread per 8 hours, as written; the code used the daily sum.
   - **Costs:** 5 bp per leg per side, as written (2 legs to open, 2 to close; a switch 4). The code charged
     twice that.
   - **Placebo:** venues are drawn at random at each entry or switch of the rule, as written. The code re-drew
     them every day, so it switched (and paid) daily, which is why it showed t of about -24.

   The first R8 results are void. The new run passes the truncation test (returns identical before every cut). A
   synthetic check gives the expected carry when one venue's funding is persistently higher.
2. **R10:** FRED's CSV endpoint timed out on the runner, so R10 was not run, as pre-registered. FRED's own other
   endpoints (the dated CSV, the text file, ALFRED) are tried before giving up. Same data; still not run if none
   answers.
3. **R7 (no change):** its random-entry placebos meet the own rule as often as the real streams (4 of 4 each). The
   basis trade earns the carry whatever the entry timing; θ adds nothing. This is reported as such, not as a
   new edge (round 22's cash-and-carry is the same carry).

## Addendum 6 (2026-11-23, after the combine; written before the benchmark below was computed)

The combine (r24z 45233635208) gives 276 counting streams. 271 are R1 units on USD/TRY (every timeframe H1-W1), 4 are
R1 pooled streams, and 1 is R7 BTC basis (whose placebos pass as often as its real streams). 28% of USD/TRY's real R1
streams meet the own rule, against 7% of its placebos. That pattern is a long-USD/TRY exposure (the lira's fall
2019-2025 outran the rate differential), not timing. Before the results are written, two checks:
1. **Benchmark:** buy-and-hold long USD/TRY with R1's carry (BIS differential) and with that differential x1.5 and x2
   (offshore TRY funding ran above the policy rate). If buy-and-hold meets the own rule, the USD/TRY streams are
   reported as that exposure, not as rule edges.
2. **HAC:** LW units keep no series, so the counting USD/TRY units get their Newey-West t from a deterministic rerun
   with `--save-series`, as round 25 part HAC.

## Addendum 7 (2026-11-23, after the benchmark of addendum 6; no rule stream had been rerun)

**Benchmark result** (r24try 45247565331): buy-and-hold long USD/TRY with R1's carry gives a 2019-2025 t of 1.30
(HAC 1.25), and does not meet the own rule. Without carry the t is 4.71. With the differential x1.5 it is -0.39,
and x2 gives -2.06. So the counting USD/TRY rules beat holding the pair: they are long in the lira's crashes
(2021, 2023) and out of 2024-2025, when the lira held with 45-50% rates.

**Stress test, before anything is reported as a candidate:** every USD/TRY R1 unit is rerun with the same seeds and
the rate differential x1, x1.5 and x2, keeping series (workflow r24trs). Reported for the 271 counting unit streams
at each scale:
- the own rule with and without the Newey-West condition;
- the median t;
- the equal-weight portfolio of the streams.

A USD/TRY candidate goes forward only if, at x1.5, its portfolio still meets the own rule with the Newey-West
condition. Its 2026 check is pre-registered separately, after this test and before 2026 data is read.

## Addendum 8 (2026-11-23, after the stress test of addendum 7)

**Stress result** (r24trs 45258441736). The 271 counting USD/TRY unit streams at carry x1, x1.5 and x2:

| Carry | Meet the own rule | ... with the Newey-West condition |
|---|---|---|
| x1 | 271 | 271 |
| x1.5 | 193 | 191 |
| x2 | 128 | 125 |

Their equal-weight portfolio has a 2019-2025 t of 7.2-7.7 (Newey-West 5.5-5.7) at every scale.

**Examined before anything is reported.** The portfolio's mean is 0.54% a day, about 136% a year at unit exposure.
Holding the pair returned about 35% a year (log) over the same years. That size is implausible for a timing rule,
so the sanity rule applies. Workflow r24tryx reports:
- the pair's largest days and any bad hourly prints;
- the portfolio by year;
- its share of return from its best 10 and 20 days;
- its t without them, and without the pair's 10 largest-move days.

No 2026 data is read until this is settled.
