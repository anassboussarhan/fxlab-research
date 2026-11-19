# Round 23 results: every gap from the round-23 audit (2026-11-20, draft: look-ahead audit running)

- **Pre-registration:** `docs/round23_preregistration.md`, with addenda 1-9. **Audit it answers:**
  `docs/round23_audit.md`.
- **Runs used by the combine** (`ci/R23Z`):

  | Run | What |
  |---|---|
  | 44329485816 | r23a: LW wave 1, first run. 13 group-0 jobs and AUDUSD H1-M15 finished; the rest hit the time limit |
  | 44329485826 | r23b: IC, T2 wave 1, event dates |
  | 44329485901 | r23c: A, PF, ETF, N, C30 first attempt |
  | 44340091264 | C30 rerun after its crash (addendum 2) |
  | 44398094254 | r23a2: LW relaunch 1. Three tags finished in full; the rest were replaced by r23a5 |
  | 44513903678 | r23a5: LW relaunch 4, the final runner (addenda 6-8) |
  | 44447569388 | r23w3: wave 2, LW and T2 on 18 new instruments (addenda 3-6) |
  | 44615958412 | r23a6: the seven jobs that had run on part of their data (addendum 9) |

- **Combine:** r23z 44631541269, branch `round23`. 433 result artifacts, all downloaded and checked.
- **Code:** `scripts/wf23.py`, `scripts/r23_*.py`, `scripts/r23_combine.py`. The look-ahead audit added after the
  combine is `scripts/r23_lookahead.py`.
- **2026:** not used. Nothing new counts. The two streams that do are unchanged from round 22, whose 2026 check
  already covers them.

## Summary

**Nothing new counts.** After the two invalid sources below are removed, 2 of 240,953 real streams count:
- crypto cash-and-carry on BTC and ETH;
- the same streams as in round 22 (default selection), with the same figures: BTC +2.1 bp a day, t 21.7 in
  2023-2025; ETH +1.9 bp, t 20.3;
- the 2026 check of round 22 found them supported but shrunk to about 2.4% a year on BTC.

**What the combine first reported:** 1,463 counting streams. On inspection, 1,452 of them came from two faults.
The other 11 (6 LW units, 3 stored-library units and cash-and-carry on BTC and ETH) cleared the BH line only
because those 1,452 had lowered it.
- **A look-ahead in two LW session strategies** (S.SES.SESSION_MOMENTUM, S.SES.NY_OPEN_REVERSAL):
  - LW exposes the numeric constants of each rule as parameters. In these two rules the hours of a reference
    price and of the trading window were separate constants.
  - When they are drawn independently, the trading window can open before the reference price is set. The rule
    then trades on a price later in the same day.
  - Effect: SESSION_MOMENTUM has a median t of 11.3 over all its 813 real units outside USD/TRY. The two
    strategies have 690 counting streams outside USD/TRY. A truncation test confirms the look-ahead (below).
- **USD/TRY without carry:**
  - The LW engine charges a flat swap, not the interest differential. Long USD/TRY therefore collects the lira's
    fall but not the interest it costs, which was tens of percent a year in most of 2019-2025.
  - Effect: 762 counting streams. Random-state placebos meet their own rule on USD/TRY 17.5% of the time,
    against 0.37% on the other instruments.

These 3,010 streams are void. In the Benjamini-Hochberg step they are set to p = 1, with the number of tests
unchanged. That leaves the BH threshold at t ≈ 4.8, and only cash-and-carry clears it.

**The other results:**
- **Own-rule rate:** valid real streams meet their own rule at 0.71%; placebos at 0.50% (0.37% in LW outside
  USD/TRY). Most single passes are noise.
- **Best valid LW unit:** Alligator on the Nasdaq CFD at H2 (nested), t 3.67. Gold dominates the top of the list,
  which is consistent with its 2019-2025 rally and a long bias. No pooled LW strategy meets its own rule (best:
  Heikin-Ashi, t 2.67).
- **D7 (part A):** marking open trades at the window end leaves 2019-2025 unchanged (t 3.2). It removes most of
  the 2026 evidence: D7-base goes from 33 trades, +0.78R, t 5.4 to 47 trades, +0.26R, t 0.6. The family's 2026
  t falls from 2.2 to 0.4.
- **Prop account (PF):**
  - With the 2019-2025 edges, the leverages chosen on 2019-2022 pass both FTMO phases 81% of the time in
    2023-2025, with mean net +$11,566.
  - With the 2026 edges, the pass rate is 50%, the mean net +$3,476 and the median −$585.

## Results by part (development data, out of sample, 2019-2025; void streams excluded)

Counting rule:
- the first out-of-sample period up to 2018 has mean > 0, if it exists;
- 2019-2025 has mean > 0 with t ≥ 1.645;
- Benjamini-Hochberg at q = 0.10 over all real streams.

Default is the round-22 selection (expanding window, top 5, yearly, by t). Nested chooses the walk-forward setting
each year from the 120 settings.

| Part | What | Real streams (default / nested) | Meet own rule | Best 2019-2025 t | Result |
|---|---|---|---|---|---|
| LW | 105 library strategies with every parameter and constant free, continuous entries / exits, M15-W1, 66 instruments | 58,208 / 56,108 | 464 / 469 (0.82%) | 3.67 (Alligator, Nasdaq CFD, H2, nested) | nothing passes BH; placebos 18 / 27 of 12,134 (0.37%) |
| IC | ICT / SMC families (LVL, FVG, OB, OTE, BRK, CBDR, SES) on a causal simulator, timeframe free | 131 / 131 | 1 / 0 | 2.23 (SES, Nasdaq CFD, default) | nothing; every pooled family negative (t −1.1 to −7.5); placebos 1 of 261 |
| T2 | Time windows with half-hours and 21 day filters (66,528 configurations per instrument) | 65 / 63 | 0 / 0 | 2.06 (GBP/USD) | nothing; pooled t −2.3 / −4.1 |
| C30 | Monthly currency signals on 24 currencies, 1999-2025 | 1 / 1 | 1 / 0 | 3.10 (default: P1 +0.17%, P2 +0.44% a month) | meets its own rule, far from BH; nested P1 −0.80% a month |
| ETF | Sector, country, cross-asset rotation, counted as excess over equal weight | 3 / 3 | 0 / 0 | 1.31 (countries, nested) | nothing; raw returns meet the rule (t up to 2.57), which is market beta |
| N-L | Stored library P&L (round-22 L), nested rerun | 62,936 / 60,147 | 332 / 414 | 3.59 | nothing; placebos 34 / 77 of 18,885 |
| N-X | Index breakouts (noise area, ORB, VWAP) | 30 / 27 | 1 / 1 | 1.91 (noise area, NAS100) | nothing |
| N-K | Crypto cross-section; cash-and-carry | 3 / 1 | 2 / 0 | **21.65 (cash-and-carry BTC)** | **cash-and-carry counts on BTC and ETH** (default; the nested stream would start only in 2026, three years after the first out-of-sample year 2023) |
| N-P | Options structures on real quotes | 25 / 25 | 5 / 6 | 2.87 | nothing |
| N-D, E, M, B, C1-C3, S, V | Dip-buying, EIA / funding events, monthly FX, bonds, commodities, S&P anomalies, volatility selling | 1-5 each | 0 | 1.88 (S anomalies, nested) | nothing |

**Notes:**
- **LW:** 197 of 131,134 units ran fewer than 128 configurations because of the 30-minute unit cap (addendum 6):
  entropy 125, HMM 60, volatility forecast 12. S.RG.HMM was lost (no data) at H1, M30 and M15 on most
  long-history instruments (addendum 7).
- **IC-SMT did not run.** Its round-22 function returns trades instead of placing orders, so the round-23 recorder
  saw none and every configuration traded nothing. Its round-22 result (market entries at the next open, time
  exits, so causal) stands: it failed.
- **T2:** the two USD/TRY streams (t 3.46 default) are void with the rest of USD/TRY. They did not meet their own
  rule anyway (2010-2018 negative).
- **C30:**
  - The default stream (carry, momentum, value or composite chosen walk-forward) meets its own rule. At this
    number of tests the BH threshold is t ≈ 4.8.
  - The nested selection loses before 2019.
  - Universe: 24 currencies with an OECD short rate. INR's rate starts in 2011 and JPY's in 2002.

## The void streams

### LW session strategies: a look-ahead created by exposing constants

`fxlab/strategies/rules.py`:
```python
# S.SES.SESSION_MOMENTUM
ref = df["close"].where((hrs >= 8 + h - 1) & (hrs < 8 + h)).groupby(day).transform("last")
active = (hrs >= 8 + h) & (hrs < 16)
# S.SES.NY_OPEN_REVERSAL
lo = df["open"].where((hl >= 8) & (hl < 9)).groupby(day).transform("first")
trigger = (hn >= 8) & (hn < 9) & (move.abs() > k)
```

**Why the defaults are causal:** `transform` spreads a price of one hour over the whole day. At the defaults the
trading window opens after that hour.

**How LW breaks it:**
- LW made each literal (the 8 in `ref`, the 8 in `active`, the 8 and 9 in `lo`) a separate parameter, drawn in
  [x/3, 3x].
- The verification step checks only that the rewritten function reproduces the original at the defaults, which
  it does.
- With other draws the window can open before the reference hour, and the signal then uses a later price of the
  same day.

| Strategy | Real units (outside USD/TRY) | Meet own rule | Median t | Median t by timeframe |
|---|---|---|---|---|
| SESSION_MOMENTUM | 813 | 549 | 11.25 | H2 27.8, H1 15.6, M15 15.2, M30 13.0, H4 11.3, H8 −4.2, H12 −3.3 |
| NY_OPEN_REVERSAL | 685 | 170 | −0.77 | M15 14.6, M30 7.2, H1-H8 −0.5 to −2.7 |

**Truncation test** (`scripts/r23_lookahead.py`):
- **Status:** draft. The test of all 105 LW strategies at M15, H1, H4 and D1 is still running; this section is
  completed when it finishes.
- **Method:** 16 LW configurations per strategy and timeframe, each run on a synthetic random walk and on the same
  series cut at 48 points.
- **So far:** SESSION_MOMENTUM and NY_OPEN_REVERSAL change their signals on earlier bars when the future is cut
  off.

### USD/TRY: no carry in the LW engine

- **What the engine charges:** a flat swap in pips per night, the same for long and short. The rate-differential
  financing (`financing="rates"`) exists in `fxlab/evaluate.py`, but LW does not use it.
- **Why it matters for USD/TRY:** the lira lost most of its value in 2019-2025, while its short rate stood far
  above the dollar's, tens of points in most years. A long USD/TRY position earns the fall and in reality pays
  that differential.
- **The streams:** 1,528 real LW units, of which 940 meet their own rule (median t 3.73). Random-state placebos meet
  it 17.5% of the time (28 of 160), against 0.37% elsewhere.
- **The other pairs:** the same omission biases the other high-yield pairs (MXN, ZAR, HUF, PLN, CZK, CNH) less.
  Their placebo rates are 0-1.2%. Carry trades of the dollar against G10 at intraday holds are barely affected.

## Deviations and fixes during the round (addenda 1-9)

| Addendum | What |
|---|---|
| 1 | New instruments from Dukascopy hour candles (the minute feed throttled); BoE dates dropped (no public source); T2 at whole hours on the new instruments |
| 2 | C30 crashed on missing OECD series (AUD, NZD, RON) on its first run; fixed (quarterly CPI carried forward) and rerun |
| 3 | A fill pass for months missed under throttling. Wave 2 takes instruments with at least 5 calendar years. T2 has 21 filters (66,528 configurations), not the 20 (63,360) first written |
| 4 | The X10 data rule (history from January 2016, price within a factor of 2 of the probe) applied in the combine; USSC2000IDXUSD (from 2019) reported, not counted |
| 5 | XPDUSD and XPTUSD have no data on Dukascopy; their timed-out fill jobs made GitHub skip the wave-2 strategy jobs, which were rerun |
| 6 | LW lost one unit per long job and waited for it until its budget: unit cap of 1,800 s, worker memory limit, dead / stuck worker handling, second relaunch |
| 7 | 20 jobs at a time, longest first; S.RG.HMM is stuck at H1 and below on long histories and is killed |
| 8 | The memory limit of addendum 6 was below what a worker inherits from the M1 data, so no unit ran; fixed and relaunched |
| 9 | HTTP 503 egress errors left seven jobs on part of their data; downloads now retried one by one and verified; the seven rerun |

**After the combine (this document):**
- the LW look-ahead and USD/TRY voids above;
- IC-SMT not run;
- the BH recomputed with the void streams at p = 1.

The combine's own output is kept unchanged on branch `round23`.

## Download coverage (new instruments, hour candles)

| Instrument | First month | Months | Missing months | 2024 level vs probe | Used |
|---|---|---|---|---|---|
| USDSEK, USDNOK, EURSEK | 2007-01 | 227-228 | 0-1 | 1.01 | yes |
| EURNOK, EURPLN, EURHUF, USDHUF, USDPLN, USDMXN | 2007-02 to 2007-04 | 224-226 | 0-2 | 0.995-1.07 | yes |
| USDZAR, USDSGD | 2007-01 | 228 | 0 | 0.97, 1.00 | yes |
| USDTRY | 2007-03 | 226 | 0 | 1.04 | yes (LW and T2 void: no carry) |
| USDCNH | 2012-06 | 162 | 1 | 1.00 | yes |
| USDCZK | 2016-01 | 120 | 0 | 1.00 | yes |
| USA30IDXUSD | 2013-09 | 148 | 0 | 1.02 | yes |
| AUSIDXAUD | 2013-01 | 155 | 1 | 1.02 | yes |
| ESPIDXEUR | 2012-01 | 167 | 1 | 1.08 | yes |
| USSC2000IDXUSD | 2019-03 | 82 | 0 | 1.02 | no (X10: starts after 2016) |
| XPTUSD, XPDUSD | none | 0 | all | | no data |

## What this means

- Round 23 closed every gap in the round-23 audit:
  - constants and every parameter free;
  - continuous entries and exits;
  - the walk-forward settings themselves chosen (nested);
  - half-hour windows with event days;
  - ICT / SMC families on a causal simulator;
  - about 30 currencies, ETF rotation, new instruments and timeframes.
- None of it produced a new edge.
- Two of the round's own faults would have produced a false one at scale: decoupled constants (look-ahead) and a
  cost model without carry. Both were caught by the placebo rates and by results that were too strong to be real.
- **Standing:**
  - Gotobi and the D7 family stay as they were, on forward test.
  - D7's 2026 support is now weak once open trades are marked.
  - Cash-and-carry is real but small.

Not financial advice.
