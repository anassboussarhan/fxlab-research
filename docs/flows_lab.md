# Flows, calendar and option-implied effects (runs 17 and 19)

Scripts: `scripts/flows_lab.py` (one job per USD pair), `scripts/flows_report.py` (aggregation, BH, random tests).
Tests: `tests/test_flows_lab.py`. Results: branches `results17`, `results19`, folder `sweep_results/flows/`.
Development data only: USD/JPY and the other USD pairs 2007-04 to 2018-06, EUR/USD to 2022-07. External series
(VIX, EVZ, equity indices, FOMC dates) are cut at 2025-12-31. The 2026 holdout is not loaded.

## Why this lab

The literature check (`docs/literature_check.md`) found that the FX effects that survive costs in published work
are mostly driven by flows and calendars (fixing flows, Gotobi settlement days, month-end hedge rebalancing,
announcement days) or by option-implied information, not by price patterns. Runs 1-16 had tested price patterns,
features, models and random entries. This lab tests the flow and calendar effects in the form the papers state
them, with rules fixed before the run, on real bid/ask M1 data with the platform's costs.

## Execution and costs

- Entries and exits at the open of the M1 bar at the stated local time (first bar within 5 minutes, else no trade;
  65 minutes where the data has a gap, see below). Daylight-saving time is handled per time zone.
- Buy at the ask, sell at the bid, plus slippage of 0.25 x spread + 0.1 pip per side and commission 0.35 bp per
  side.
- Run 17 had no swap. Run 19 charges swap on every trade held through 17:00 New York, Wednesday x3. Two versions
  are reported:
  - the platform's flat 0.5 pip per night on either side;
  - the point-in-time 3-month rate differential minus a 0.75% p.a. mark-up (`net_sw`; SEK and NOK have no rate
    data, so they use the flat charge).
  A trade opened or closed at the 17:00 bar itself is treated as flat over that rollover.
- Gross = mid-to-mid, no costs.

**Data gap.** The M1 store has no bars between 00:00 and 01:00 UTC (09:00-10:00 Tokyo time) for 8 of the 9 USD
pairs; only EUR/USD has that hour. The runner counts bars per UTC hour for every pair in this lab. So the Gotobi
exit is the 10:00 JST bar, not the 09:55 fix.

## The 17 registered tests

| Test | Rule | Source |
|---|---|---|
| F1 | Gotobi days: long USD/JPY 03:00 JST to the Tokyo fix | Ito & Yamada (2017); arXiv 2301.13204 |
| F2 | Gotobi days: long USD/JPY 08:00 JST to the fix | same |
| F3 | Gotobi days: short USD/JPY from the fix to 12:00 JST | same |
| M1 | Month-end London 4 p.m. fix, equity-hedging signal, 08:00 to 16:00 London, 9 currencies | Melvin & Prins (2015) |
| M2 | The same, 24 hours into the fix | same |
| L1 | Sell each currency in its home trading hours, 8 currencies | Breedon & Ranaldo (2013) |
| L2 | The same, EUR/USD only | same |
| V1 | Sell one-month EUR/USD variance at EVZ, 0.5 vol point cost | variance risk premium |
| V2 | VRP direction signal, EUR/USD monthly | Della Corte, Ramadorai & Sarno (2016) |
| C1 | Carry only when VIX is calm, 7 pairs | carry crash literature |
| C2 | Carry always, 7 pairs | baseline |
| K1 | Long foreign currencies vs USD in New York hours | Krohn, Mueller & Whelan |
| K2 | Short foreign currencies overnight | same |
| K3 | Both legs, EUR / GBP / CHF | same |
| P1 | FOMC days: long foreign currencies vs USD, 17:00 ET to 17:00 ET | Mueller, Tahbaz-Salehi & Vedolin (2017) |
| P2 | FOMC days: the 3 highest-yielding currencies | same |
| W1 | Fade the 15:30 to 16:00 London move after the fix | Evans (2018) |

The exact definitions are in the docstring of `scripts/flows_lab.py`. One-sided p-values (positive = the effect as
published) and Benjamini-Hochberg over the 17 tests.

## Run 17 results (no swap)

| Test | Events | Net bp / event | t | Sharpe | Years + | Gross bp | Gross t | BH q |
|---|---|---|---|---|---|---|---|---|
| **F1 Gotobi 03:00 to fix** | 700 | **+3.45** | **2.91** | 0.92 | 75% | +5.57 | 4.69 | **0.027** |
| F1 control (other days) | 1,538 | -2.03 | -2.84 | | | +0.09 | 0.13 | |
| F2 Gotobi 08:00 to fix | 800 | -1.47 | -2.04 | | 17% | +0.96 | 1.33 | 1.0 |
| F3 Gotobi reversal | 800 | -0.04 | -0.05 | | 50% | +2.06 | 2.16 | 1.0 |
| M1 month-end 08:00 to fix | 176 | -0.84 | -0.43 | | 44% | +1.88 | 0.95 | 1.0 |
| M2 month-end 24 h | 176 | +3.78 | 1.51 | 0.39 | 63% | +6.36 | 2.53 | 0.25 |
| L1 local hours, 8 currencies | 3,967 | -2.60 | -5.68 | | 13% | +0.80 | 1.74 | 1.0 |
| L2 local hours, EUR/USD | 3,963 | +0.20 | 0.31 | | 50% | +1.83 | 2.76 | 1.0 |
| V1 / V2 options | 0 | | | | | | | not run |
| C1 carry, VIX calm (bp/day) | 3,982 | -0.08 | -0.24 | | 50% | -0.06 | -0.18 | 1.0 |
| C2 carry always (bp/day) | 3,982 | -0.08 | -0.13 | | 50% | -0.25 | -0.43 | 1.0 |
| K1 New York hours | 3,166 | -5.65 | -9.06 | | 0% | -0.14 | -0.22 | 1.0 |
| K2 overnight | 3,164 | -5.41 | -8.44 | | 6% | +0.43 | 0.68 | 1.0 |
| K3 both legs, EUR/GBP/CHF | 3,961 | -4.14 | -5.56 | | 0% | +1.70 | 2.28 | 1.0 |
| P1 FOMC, dollar portfolio | 118 | +10.4 | 1.60 | 0.42 | 69% | +17.9 | 2.70 | 0.25 |
| P2 FOMC, high-yielders | 118 | +15.2 | 1.97 | 0.51 | 69% | +21.7 | 2.77 | 0.19 |
| W1 fix reversal | 3,962 | -2.42 | -21.2 | | 0% | +0.32 | 2.88 | 1.0 |

Splits at the end of each paper's sample:

| Test | Paper sample ends | t inside | t after | Mean after (bp) |
|---|---|---|---|---|
| F1-F3 | 2020 | all events | no data after 2018-06 | |
| L1 | 2007-05 | -2.35 | -5.57 | -2.58 |
| L2 | 2007-05 | -1.37 | 0.39 | +0.26 |
| K1 / K2 / K3 | 2014 | -6.7 / -6.7 / -3.3 | -6.1 / -5.1 / -4.7 | |
| P1 | 2010 | 2.36 | 0.21 | +1.4 |
| P2 | 2010 | 2.96 | 0.15 | +1.1 |

What run 17 shows:

- **Gotobi (F1) is the only survivor.** It is specific to Gotobi days: the same window on other days loses
  2.0 bp after costs and earns nothing before costs. **Caveat:** USD/JPY data ends in mid-2018, so all 700 events
  fall inside the period the published papers studied. This is a replication with real bid/ask costs, not an
  out-of-sample confirmation after publication.
- **FOMC days** look strong in the paper's sample (t 2.4 / 3.0) and disappear after 2010 (t 0.2 / 0.15).
- **Month-end with the equity signal (M2)** is real before costs (t 2.5) and not significant after.
- **Fix reversal, around-the-clock and local hours** are real before costs in some cases (W1 gross t 2.9, K3 2.3,
  L2 2.8), but the edge is a fraction of a basis point to 2 bp, and the two daily round trips cost more.
- **Carry**, filtered by VIX or not, made about zero over 2007-2022 on these 7 pairs.
- **The options tests did not run.** FRED timed out and the Cboe CDN refused the download of EVZ on the runner.
  Run 19 adds fallbacks.

## Run 19

Run 19 keeps the same 17 tests, with nothing re-tuned, and adds four things:

1. **Swap** on every overnight trade (see above).
2. **EVZ fallbacks:** FRED CSV x3 with a 120 s timeout, then the FRED text file, Yahoo `^EVZ` and Cboe.
3. **Random-date test** (`permutation.csv`). Each event rule is compared with the same rule on randomly drawn
   non-event days from the same stratum:
   - year x weekday for Gotobi and FOMC, because Gotobi days are mostly Fridays (weekend dates move back);
   - year x month for the month-end rule, whose placebo applies the same signal (trailing-month equity returns) to
     every other business day.
   There are 10,000 draws, with replacement. p is the share of draws whose mean is at least the actual mean.
4. **Random entry timing inside the event windows** (`random_timing.csv`). On each event, 16 entry minutes are
   drawn uniformly in the window before the event, in the edge's direction:

   | Test | Window | Exit |
   |---|---|---|
   | B1 | Gotobi, 03:00-09:00 JST | at the fix |
   | B2 | Gotobi | random bracket |
   | B3 | Month-end, 24 h before the fix, signal direction | at the fix |
   | B4 | Month-end | random bracket |
   | B5 | FOMC, 17:00 ET to 16:30 ET, long foreign currencies | at 17:00 ET |
   | B6 | FOMC | random bracket |

   - The bracket's take-profit and stop are each drawn from {0.1, 0.2, 0.4} x the previous 20 days' mean range.
   - The stop wins when both are touched in one bar, a gap through the stop exits at the open, and the target
     fills on its own side of the quote.
   - Controls: a coin-flip direction at the same entry minutes.
   - Reported: the seed-average stream (expected value of a random trader), the 16 single traders (median, worst,
     share positive), and BH over B1-B6.

## Run 19 results (run 40207062071, branch `results19`)

### Swap

The registered tests are unchanged from run 17 (same net), and swap changes little except for Gotobi:

| Test | Net, no swap | Net, rate swap (`net_sw`) | Net, flat swap (`net_swf`) |
|---|---|---|---|
| F1 Gotobi | +3.45 bp, t 2.91 | **+3.41 bp, t 2.88** | +2.76 bp, t 2.33 |
| M2 month-end 24 h | +3.78, t 1.51 | +3.53, t 1.41 | +3.22, t 1.29 |
| P1 FOMC | +10.4, t 1.60 | +10.4, t 1.60 | +10.4, t 1.60 |
| P2 FOMC high-yielders | +15.2, t 1.97 | +15.3, t 1.97 | +15.2, t 1.96 |

- F1 is held through one rollover (three on Thursday Gotobi days, since entry is Wednesday 18:00 UTC). Long
  USD/JPY usually earned interest, so the rate-based swap costs only 0.04 bp per trade on average. The flat charge
  costs 0.7 bp.
- FOMC trades enter at 17:00 ET and exit at 17:00 ET the next day, so they almost never hold a rollover. The
  exception is when the 17:00 bar is missing and the exit falls a few minutes later.

### The options tests (V1, V2)

The runner still could not download EVZ:
- the FRED CSV and text file timed out (FRED did not answer the runner at all in this run; VIX came from Yahoo
  `^VIX`);
- Yahoo has no `^EVZ` rows;
- Cboe returned 403.

The month-end closes were therefore read from the FRED series page in the research session and saved as
`data/alt_manual/EVZCLS_monthend.csv`.
- The values were extracted three times with different prompts and agreed on every month.
- The page text the tool returns stops at 2014-10-08, so the file ends at 2014-09-30.
- V1 and V2 need only the last close before each month, so this is enough for them.
- `flows_lab.py` now uses the file when every download fails.

V1 and V2 were run locally, with the same code, on EUR/USD 2007-10 to 2014-12. The monthly rows are in
`docs/flows_v1v2_monthly.csv`. V1 is in vol points:

| Test | Months | Net | t | Sharpe | Years + | Months + | Gross | Worst month |
|---|---|---|---|---|---|---|---|---|
| **V1 sell 1-month variance at EVZ** | 82 (2008-01 to 2014-10) | **+1.31 vol pt** | **5.38** | 2.06 | 7/7 | 83% | +1.81 vol pt (t 7.4) | -6.61 (May 2010) |
| V2 VRP signal for spot EUR/USD | 70 (2009-01 to 2014-10) | +16.3 bp (with swap) | 0.43 | 0.18 | 50% | | +25.7 bp (t 0.68) | |

- Average EVZ at the month start was 11.98; average realised volatility over the next month was 9.78.
- The worst months were May 2010 (euro crisis, -6.61) and September 2008 (-6.49).
- V1 breaks even at about 1.8 vol points of round-trip cost.
- V1 is the variance risk premium: the seller is paid for crash insurance. Its returns are negatively skewed.
  - The sample has 2008 but not 2015-2022 (the March 2020 crash, 2022).
  - It needs options: a variance swap or delta-hedged straddles. A spot-FX account cannot trade it.
  - 0.5 vol points is an assumption about option costs; retail spreads on listed FX options can be wider.

With V1 and V2 included, Benjamini-Hochberg over the 17 tests (rate swap) gives:

| Test | p | q |
|---|---|---|
| V1 | < 1e-6 | < 1e-5 |
| F1 | 0.0020 | 0.017 |
| P2 (next) | 0.024 | 0.14 |

With the flat swap, F1 has q = 0.084: it passes at 10%, not at 5%.

### Random-date test (`permutation.csv`)

Each event is compared with the same rule on random non-event days of the same year and weekday (Gotobi, FOMC),
or year and month (month-end). Values are with the rate swap, bp per event:

| Test | Event days | Random days (mean, 5-95%) | Permutation p | Welch t | Years event > control | Gross: event vs control |
|---|---|---|---|---|---|---|
| **F1 Gotobi** | +3.41 | -1.49 (-3.26 to +0.32) | 0.0001 | **3.97** | 10 / 12 | +5.57 vs +0.09 |
| F2 | -1.47 | -1.53 | 0.46 | 1.16 | 8 / 12 | +0.96 vs -0.02 |
| F3 | -0.04 | +0.33 | 0.72 | 0.22 | 7 / 12 | +2.06 vs +1.83 |
| P1 FOMC | +10.4 | -8.3 | 0.0001 | 2.78 | 14 / 16 | +17.9 vs -0.8 |
| P2 FOMC high-yielders | +15.3 | -8.9 | 0.0001 | 2.92 | 13 / 16 | +21.7 vs -1.2 |
| M1 month-end 08:00 | -0.84 | -2.94 | 0.15 | 1.04 | 11 / 16 | +1.88 vs -0.32 |
| M2 month-end 24 h | +3.53 | -2.24 | 0.018 | 2.21 | 12 / 16 | +6.36 vs +0.62 |

- The permutation treats event and control days as exchangeable. FOMC days are 1.4x as volatile as other days,
  which makes its p-value too small there. The Welch t (unequal variances) is the number to use for FOMC.
- FOMC by period (gross, Welch t against other days):
  - P1: 2007-2010 +46 bp, t 2.85; 2011-2022 +8 bp, t 1.35.
  - P2: 2007-2010 +65 bp, t 3.35; 2011-2022 +7 bp, t 1.04.
  - The FOMC effect is real in the paper's period and much weaker since.
- Gotobi net with rate swap by year (bp per event):

  | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 (to June) |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | -3.0 | -2.4 | -5.1 | +7.2 | +11.2 | +3.0 | +12.7 | +2.6 | +0.8 | +5.9 | +4.0 | +1.6 |

  It lost in 2007-2009 and was positive in every year from 2010 to mid-2018, smaller since 2015.

### Random entry timing (`random_timing.csv`)

16 random traders per test, edge direction, net with rate swap, bp per event:

| Test | Events | Seed-average | t | Gross (t) | Single traders: median / worst / best | Positive | t > 2 | Coin flip (control) |
|---|---|---|---|---|---|---|---|---|
| B1 Gotobi, random time to the fix | 800 | +0.41 | 0.55 | +2.79 (3.72) | +0.34 / -0.33 / +1.36 | 14 / 16 | 0 | -2.28 (t -11.3) |
| B2 Gotobi, random bracket | 797 | -0.69 | -1.70 | +1.86 (4.56) | -0.69 / -1.28 / +0.04 | 1 / 16 | 0 | -2.55 |
| B3 month-end, random time to the fix | 176 | +0.13 | 0.07 | +3.12 (1.64) | +0.01 / -0.92 / +1.53 | 8 / 16 | 0 | -3.57 |
| B4 month-end, random bracket | 176 | -2.45 | -5.99 | +0.72 (1.79) | -2.58 / -3.73 / -1.09 | 0 / 16 | 0 | -3.15 |
| B5 FOMC, random time to 17:00 ET | 118 | +6.88 | 1.28 | +12.4 (2.29) | +6.89 / +4.48 / +8.52 | 16 / 16 | 0 | -5.01 |
| B6 FOMC, random bracket | 118 | -2.01 | -3.60 | +1.47 (2.46) | -1.99 / -3.96 / -0.56 | 0 / 16 | 0 | -3.57 |

- BH over B1-B6: no survivor; the smallest q is 0.60.
- B1 has 800 events against F1's 700 because Monday Gotobi days are included from the weekly open.
- Direction matters: on Gotobi days the edge direction beats a coin flip at the same minutes by 2.7 bp.
- A random entry time keeps only about half of the gross move (+2.8 against +5.6 bp from the fixed 03:00 entry),
  which does not cover costs.
- Take-profit / stop brackets make every version worse. These effects are slow drifts into a fixed time, and the
  brackets cut them short.

## Conclusion

Of the effects tested here, two survive costs and the correction for 17 tests:

1. **Gotobi** on USD/JPY, a spot trade. +3.4 bp per trade, about 62 trades a year, roughly 2% a year on notional
   before leverage, Sharpe 0.9. It is specific to Gotobi days (Welch t 4.0 against other days). It is sensitive to
   swap terms (q 0.017 with rate-based swap, 0.084 with the flat charge). All 700 events fall inside the published
   sample, and the data ends in mid-2018.
2. **Selling one-month EUR/USD variance**, an options trade. +1.3 vol points a month, t 5.4, over 82 months
   (2008-2014). It is the variance risk premium, with crash risk and a short sample. It needs options.

Everything else is either real only before costs (month-end, fix reversal, around-the-clock, local hours, FOMC
since 2011) or has faded since the paper (FOMC, local hours). Random entries do not rescue any of them: random
timing inside the windows keeps part of the gross edge but not enough to pay costs, and brackets destroy it.
