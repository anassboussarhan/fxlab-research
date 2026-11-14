# Round 19 results: the verified gaps (2026-11-14)

- **Pre-registration:** `docs/round19_preregistration.md`.
- **Code:** `scripts/r19.py`; tests in `tests/test_r19.py`.
- **CI runs:**
  - 43601331590: parts A and F, and the library on the 13 D2 instruments;
  - 43601520485: parts C and D, the crypto data and the crypto library. Its part E timed out.
  - 43607277933: part E, vectorised, and the combine over all three runs.
- **Results:** branch `round19`.
- **Exploratory year-by-year breakdown:** branch `misc-r19nb`.
- **2026:** not used; no test counted, so the conditional check did not run.

## Summary

**0 of 22 hypothesis tests count, and 0 of 21,000 library units count.**

**The one result worth following is the noise-area breakout on the Nasdaq** (Zarattini, Aziz & Barbon 2024):
- It meets every condition of its own rule:
  - 2012-2018: +1.65 bp a day at futures costs, t 1.5;
  - 2019-2025: +3.27 bp a day, t 2.25;
  - positive in 6 of 7 years from 2019.
- It still fails the correction over 22 tests: p 0.012 against a first threshold of 0.0045.
- It keeps +2.8 bp a day at twice the futures cost and +1.85 bp at CFD costs. It trades about 0.8 times a day.
- With the paper's volatility sizing (2019-2025): about 15% a year, Sharpe 1.06, worst drawdown −21%.
- It is the first intraday rule in this project that stays positive after costs in both periods.
- After the paper's sample ends (May 2024 to Dec 2025): +3.1 bp a day, t 0.95. That is too short to judge.

## Family H (22 tests; BH at q = 0.10)

**Counting rule:** P1 mean > 0, P2 mean > 0 with t ≥ 1.645, and passes BH; for part C also the Welch condition.

| Test | P1 net, bp | P2 net, bp (t) | P2 gross, bp (t) | Own rule | Counts |
|---|---|---|---|---|---|
| A-ORB5-US500 | −1.13 | +0.34 (0.24) | +1.96 (1.39) | no | no |
| A-NB-US500 | −1.02 | +2.54 (2.05) | +3.85 (3.12) | no (P1) | no |
| A-C2b-US500 (N60 from 10:00) | −3.40 | −0.12 (−0.05) | +3.85 (1.75) | no | no |
| A-ORB5-NAS100 | +0.62 | +2.65 (1.61) | +3.24 (1.98) | no (t) | no |
| **A-NB-NAS100** | **+1.65** | **+3.27 (2.25)** | +3.76 (2.59) | **yes** | **no (BH)** |
| A-C2b-NAS100 (N60 from 09:30) | −1.45 | +1.03 (0.37) | +2.68 (0.95) | no | no |
| C-D7, carry direction only (mean R) | +0.31R | +0.11R (0.73) | | no | no |
| C-FAM, carry direction only (mean R) | +0.25R | +0.05R (0.51) | | no | no |
| D-E1-WTI (trade the inventory surprise) | −23.9 | −25.4 (−2.73) | −12.3 (−1.33) | no | no |
| D-FADE-WTI | −7.4 | −31.1 (−3.35) | −18.0 (−1.94) | no | no |
| D-FOLLOW-WTI | −21.9 | +4.8 (0.53) | +18.0 (1.94) | no | no |
| D-E1-BRENT | −25.7 | −24.1 (−2.90) | −12.1 (−1.46) | no | no |
| D-FADE-BRENT | −12.6 | −24.3 (−2.92) | −12.3 (−1.47) | no | no |
| D-FOLLOW-BRENT | −14.5 | +0.2 (0.03) | +12.3 (1.47) | no | no |
| E-PRE-BTC | −20.7 | −25.7 (−8.85) | −0.3 (−0.10) | no | no |
| E-POST-BTC | −19.8 | −27.9 (−10.26) | −2.5 (−0.91) | no | no |
| E-PRE-ETH | −20.1 | −27.5 (−7.00) | −1.4 (−0.36) | no | no |
| E-POST-ETH | −9.7 | −26.3 (−6.77) | −0.3 (−0.07) | no | no |
| F-FADE-IDX | −5.0 | −0.3 (−0.15) | +3.1 (1.45) | no | no |
| F-FOLLOW-IDX | −6.2 | −6.5 (−3.04) | −3.1 (−1.45) | no | no |
| F-FADE-MET | −15.3 | −13.9 (−6.20) | −1.2 (−0.52) | no | no |
| F-FOLLOW-MET | −17.7 | −11.6 (−5.01) | +1.2 (0.52) | no | no |

**How to read the table:**
- Part A is in bp a day at ES / NQ futures costs. P1 is 2012-2018 and P2 is 2019-2025.
- Part D is in bp per release (2013-2018 / 2019-2025). Part E is in bp a day (2020-2022 / 2023-2025). Part F is
  in bp a day (2007-2018 / 2019-2025).
- Part C is in R per trade (2000-2012 / 2013-2025).

## What each part shows

### A. Index rules at futures costs

**Futures costs are lower than CFD costs only at high index levels.** One ES tick plus commission is 0.30 points:
- 1.9 bp at 1,300;
- 0.6 bp at 5,000.

So C2b and ORB5 on the S&P stay negative in 2012-2018. Both C2b settings chosen on 2012-2018 fail on 2019-2025.

**The C2b grid (`A_surface.csv`):**
- The best 2019-2025 Nasdaq setting (every 60 minutes from 10:00) earns +5.6 bp a day (t 2.2) at futures costs.
- It was not the setting chosen on 2012-2018 (t −0.76 there).
- This is the pattern of round 18: the best cell moves from sample to sample.

**The noise-area breakout is different:**
- It trades once a day or less, so costs matter less: about 0.5 bp of the gross on the Nasdaq at futures costs.
- On the Nasdaq it is positive at futures costs every year 2014-2025 except 2020. 2013 is negative and 2012 is
  positive. The partial years 2012 and 2018 have few days.
- On the S&P it is negative in 2013-2017 and positive in every year from 2018.

**Data coverage:**
- The S&P minute file ends in early 2024 in the store.
- Some years are thin in the session filter, which needs a full 09:30 start: S&P 2013 has 117 days, Nasdaq 2018
  has 31.
- So the S&P "after the paper" figure rests on 16 days and is not usable.

### C. Dip-buying in the carry direction

- The filter removes as many winners as losers.
- In 2013-2025 the excluded negative-carry trades average +0.13R, against +0.11R for the trades kept (Welch t
  −0.08).
- Carry direction does not help the weekly dip-buying rules. The unfiltered Double Seven with carry accrual earns
  +0.12R in 2013-2025, t 0.96.

### D. Oil around the EIA report

- **Data:** the EIA's weekly history file (671 releases 2013-2025; 93 in holiday weeks, timed from the data).
- **The seasonal surprise points the wrong way:** a draw then a long loses −9 bp gross in 2013-2018 and −12 bp in 2019-2025.
- The first five minutes neither reverse nor follow in a stable way (gross t −1.9 / +1.9 in 2019-2025 with the
  opposite sign before).
- Oil CFD costs (about 13 bp a trade) remove everything.

### E. Crypto around the funding times

- **Data:** complete 2020-2025 (Binance perpetual minute data, 6,576 funding stamps per coin).
- No gross effect before or after the stamps in 2023-2025: −2.5 to −0.3 bp a day.
- Even at maker costs every test is negative (t −2.7 to −4.7).

### F. Prior-day high / low stop runs

- On metals neither fading nor following has a gross edge.
- On indices fading earns +3.1 bp gross in 2019-2025 (t 1.45) but loses net.
- This matches the FX result (round 12, lit5): breaks of obvious levels are not tradable either way after costs.

## Part L. The library on the new instruments

- **Units:** 21,000 (15 instruments × 1,400 rule × timeframe × direction units), plus 3,000 placebo units.
- **Own rule:** 68 meet it (P1 > 0, P2 t ≥ 1.645, above the instrument's placebo 95th percentile). That is 0.3%,
  fewer than chance would give without costs.
- **BH:** none passes (best p 0.0011 against a first threshold of 0.0000048).
- **Median P2 t per instrument:** −1.1 to −2.8. Costs dominate, as on FX.
- **The best units are scattered and small:** ETH 30-minute volatility expansion, a monthly stochastic reversal,
  Euro Stoxx daily volatility compression.

## What this changes

- **The Gotobi trade stays the only candidate with a clean test**, and dip-buying stays in forward test.
- **The noise-area breakout on Nasdaq futures is the best new lead**, but it has not passed the bar. Two honest
  ways to settle it:
  1. a paper forward test from now on, with the rules frozen as above;
  2. the 2026 data once, as a new pre-registered single test (one test, so no multiple-testing penalty). The 2026
     holdout has been used three times already.

  Its natural venue is a futures prop firm (NQ / MNQ, flat by the close, about 0.8 trades a day).
- **Everything else in this round is closed:** dip × carry, EIA oil, crypto funding times, stop runs, and the
  library on commodities, European and Asian indices, bonds and crypto.

Not financial advice.
