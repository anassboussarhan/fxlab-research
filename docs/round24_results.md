# Round 24 results (2026-11-23; corrected 2026-11-24)

> **Correction (2026-11-24, addendum 9).** The USD/TRY result below was a trade-engine fault. The engine marked an
> open trade's P&L against its entry price, so positions held through USD/TRY's tenfold rise carried a multiple of
> their exposure, with financing on the entry exposure only.
>
> On the fixed engine (r26try 45361261104), USD/TRY R1 has:
> - 1,614 real streams;
> - 6.4% meet the own rule, against 6.25% of its placebos;
> - a largest t of 4.07;
> - **none counting.**
>
> **Round 24 therefore has no counting stream** except R7 BTC basis, which is the known cash-and-carry (its placebos
> pass as often). The sections below are kept as written, with their USD/TRY conclusions void.

Pre-registration: `round24_preregistration.md`, with addenda 1-8. Combine: r24z 45233635208. Statistics are on
branch `round24` (`r24_summary.json`, `r24_streams.csv.gz`).

## Summary

- **79,814 real streams tested:**
  - 905 meet the own rule (P1 mean > 0 where P1 exists; 2019-2025 mean > 0 with t ≥ 1.645);
  - 340 pass Benjamini-Hochberg at q = 0.10 (first threshold t = 4.71);
  - 276 count.
- **What the 276 are:**
  - **271 are R1 units on USD/TRY**, at every timeframe from H1 to W1;
  - **4 are R1 pooled streams.** They average each rule over every FX pair and metal, USD/TRY included; the same
    effect is shown for the round-25 panels (`round25_results.md`);
  - **1 is R7 BTC quarterly basis** (nested). Its random-entry placebos meet the own rule as often (4 of 4), so it
    is the cash-and-carry of round 22, not a better entry.
- **Outside USD/TRY:**
  - 0.5% of R1's real streams meet the own rule, against 0.3% of its placebos;
  - the largest t is 5.5, below the multiple-testing line once USD/TRY is set aside;
  - nothing else counts.
- **USD/TRY** (addenda 6-8):
  - **Holding the pair does not count:** buy-and-hold long with R1's carry gives a 2019-2025 t of 1.30. The rules
    were long in the lira's crashes (2021, 2023) and out of 2024-2025, when the lira held with 45-50% rates.
  - **The rate stress keeps the portfolio:** at carry x1.5 and x2, 191 and 125 of the 271 streams still meet the own
    rule with the Newey-West condition. Their equal-weight portfolio has a t of 7.2-7.7 (Newey-West 5.5-5.7).
  - **Bad data does not explain it:** 11 hourly moves above 5% in 114,118 hours. Without the portfolio's best 20
    days its t is 5.8; without the pair's 10 largest-move days it is 9.0. Doubling the carry barely moves the mean,
    so the returns are price timing, not carry.
  - **Its size is not tradable as shown.** The mean, 0.54% a day, comes from rule outputs above unit size; the t
    does not depend on that.
  - **It remains one currency in a policy crisis**, with offshore funding at times far above the policy rate.
    Retail spreads and swaps on USD/TRY are much wider than the data's quotes (median 4 bp), and few prop firms
    list the pair.
  - **Status:** a candidate. It gets a pre-registered 2026 check, written before any 2026 data is read (pending).

## By part

| Part | What | Real streams | Own rule | Counts | Placebos meeting the own rule |
|---|---|---|---|---|---|
| R1 | LW of round 23 with rate-differential carry in every engine (FX, metals) | 77,442 | 889 (1.1%) | 275 (271 USD/TRY, 4 pooled) | 24 of 8,089 (0.3%) |
| R2 | the two session rules with tied hour constants | 2,350 | 11 (0.5%) | 0 | (LW placebos) |
| R3 | TTM Squeeze tied (in R2's runs); IC-SMT through the order recorder | 4 (SMT) | 0 | 0 | 0 of 4 |
| R4 | earnings drift (8-K item 2.02), S&P 500 | 2 | 0 | 0 | 0 of 2 |
| R6 | FINRA short-volume ratio | 2 | 0 | 0 | 0 of 2 |
| R7 | crypto quarterly basis | 4 | 4 | 1 | 4 of 4 |
| R8 | perpetual funding across six venues | 4 | 1 | 0 | 0 of 4 |
| R9 | bitcoin "CME gap" | 2 | 0 | 0 | 0 of 2 |
| R10 | Fed net liquidity | not run | | | |
| R11 | ETF pairs, S&P 500 additions | 4 | 0 | 0 | 0 of 4 |

The largest 2019-2025 t per part:

| Part | Largest t |
|---|---|
| R2 | 2.41 |
| R3 SMT | 0.55 |
| R4 | 0.90 |
| R6 | 1.04 |
| R7 | 4.60 |
| R8 | 2.33 (BTC nested, 2025 only) |
| R9 | 0.95 |
| R11 | 0.66 |

## Faults found and fixed before results

| Part | Fault | Effect | Fix |
|---|---|---|---|
| R1 | OECD MEI rates end 2024-01; TRY interbank 2008 | stale carry | BIS policy rates; first run cancelled (addendum 2) |
| R4, R6, R11 | Wikipedia dropped the S&P 500 changes table | stock data job failed | article's last revision before 2026 (addendum 3) |
| R6 | position on day t's FINRA file booked from day t's open | t 10.6 (look-ahead) | open t+1 to open t+2; t now -0.9 / 1.0 (addendum 4) |
| R8 | venue price difference not marked; θ on the daily sum; double costs; placebo re-drawn daily | t 6.5-13.6 | as pre-registered; t now 0.2-2.3 (addendum 5) |
| R10 | FRED times out on the runners (four endpoints) | not run | the Treasury and New York Fed series answer; WALCL needs a FRED API key |
| combine | an empty rows file (R10) | combine failed | skipped |

Also recorded: the S&P 500 CFD minute store stops in May 2024 (refilled in round 25). USD/SGD has no policy rate and
was left out of R1. HMM units at M15 / M30 were lost, as in round 23 (likely an OpenMP fork deadlock; fixed in round
25).

## What this means

Round 24 adds no tradable edge outside USD/TRY.

- **Carry in every engine** removes round 23's USD/TRY fault, but not the USD/TRY effect. With carry, the rules still
  beat holding the pair by being long in its crashes.
- **The folklore and free-data parts** (earnings drift, short volume, basis timing, cross-venue funding, the CME gap,
  ETF pairs, index additions) show nothing once their faults are fixed.

Not financial advice.
