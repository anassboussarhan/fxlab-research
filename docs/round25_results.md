# Round 25 results (2026-11-24)

Pre-registration: `round25_preregistration.md`, with addenda 1-4.
- **Runs:**
  - r25p 45059537567, r25ad 45059537515, r25alt 45059537552;
  - reruns r25re 45130376805 and 45202360420;
  - r25hac 45055876112.
- **Combine:** r25z (branch `round25`: `r25_summary.json`, `r25_streams.csv.gz`). The counts below exclude the void
  rule (addendum 4).

## Summary

- **125,425 real counted streams:**
  - 2,171 meet the own rule (now including the Newey-West condition);
  - 579 pass Benjamini-Hochberg at q = 0.10 (first threshold t = 4.80);
  - **362 count.**
- **What the 362 are:**

  | Group | Streams | What it is |
  |---|---|---|
  | USD/TRY, part P | 326 | the round-24 effect (long in the lira's crashes) |
  | FX panels, part P | 31 | driven by USD/TRY: in every one, the other 41 pairs' streams average about zero and fewer than half have t > 0 |
  | Single streams | 5 | ALT GALA (Double Seven, H1, t 3.90), ALT ADA (Compression, H1, 3.69), ALT WAVES (weekend-gap fade, H8, 4.77); P index CFDs EUS and CHE (range expansion, W1, 3.32 and 3.34) |

  At q = 0.10, about 36 of 362 counts are expected to be false. Five isolated single streams on unrelated rules are
  within that.
- **Void:** S.SEAS.FIX_REVERSAL. It had 100 streams with t of 17-39 on every asset class, crypto included. One of
  its 128 configurations looks ahead, and walk-forward selection picks it (addendum 4). The full-design truncation
  test of every other rule found none:
  - 26,752 configurations at H1 and H4 (r25la3);
  - FIX_REVERSAL's single configuration was the only look-ahead;
  - 9 configurations errored.

## By part

| Part | What | Real own share | ... outside USD/TRY | Placebo own share | Counts (void excluded) |
|---|---|---|---|---|---|
| P | library, every execution choice free, panel selection | 1.42% | 0.99% | 0.56% | 359 (326 USD/TRY, 31 FX panels, 2 index W1) |
| CF | confluence of the top-n rules | 0.26% | | 0.26% | 0 |
| AD | TPE inside each training window, 20 rules | 3.55% | | 0.0% (90 streams) | 0 |
| ALT | 76 altcoin perpetuals, point-in-time top 30, funding as carry | 3.65% | | 2.44% | 3 |

- **CF** equals its placebo.
- **AD** has more own-rule passes than its placebo, but none survives the multiple-testing line; the largest t is 3.18.
- **ALT** runs above its placebo across the board, which fits a common market factor in 2021-2025 altcoins. Three
  single coins count.

## Part HAC (round 23)

- The 100 strongest round-23 LW passes (USD/TRY and the voided rules left out) were rerun with the round-23 code:
  100 of 100 reproduced their published t within 0.01.
- The Newey-West t over the iid t has a median of 1.03, and all 100 stay above 1.645.
- So serial correlation did not inflate the old t. The round-25 counting rule now includes the Newey-West condition.

## Faults found and fixed

| Fault | Effect | Fix |
|---|---|---|
| OpenMP threads in forked workers | HMM k-means deadlock (likely the round-23 HMM stalls) | one thread per process (addendum 1) |
| three rules' windows and min_periods drawn apart (HIGH52, VOL_TARGET, COMPRESSION) | 22-88% of round-23 draws failed | tied by role (addendum 1) |
| regime filter window from whole-sample bar density | look-ahead (found by the gate before any run) | bar count of the first year (addendum 1) |
| altcoin panel pooled future members | selection look-ahead | point-in-time membership (addendum 1) |
| job cost model (1.6x round 23) too low | unfinished rules in 18 of 394 jobs | reruns sized from observed times (addendum 3); no panel left incomplete |
| FIX_REVERSAL hour constants drawn apart | t 17-39 (look-ahead) | void (addendum 4) |

Also: the S&P 500 CFD was refilled for May 2024 to 2025 (median difference 0.0 bp against the store).

## What this means

Round 25 ran every gap of the audit:
- panel selection, higher-timeframe, regime and weekday filters;
- combined entries;
- categorical structure (moving-average type, price source, volatility measure);
- adaptive search, stability and decay scores;
- signal-strength sizing, ATR lengths;
- altcoins, HAC.

None of them produced an edge outside USD/TRY. The USD/TRY effect of round 24 reappears in every engine.

- **Next:** a pre-registered 2026 check for the counting streams:
  - the USD/TRY portfolio, frozen as of 2025-12-31;
  - the five single streams.
- **Still blocked:**
  - R10 net liquidity (needs a FRED API key);
  - stock intraday, OANDA positioning, insider buying (access);
  - order flow, NQ, 0DTE (paid).

Not financial advice.
