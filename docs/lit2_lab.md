# Round-2 literature tests (run 22)

Scripts: `scripts/lit2_lab.py` (per instrument), `scripts/lit2_report.py` (FX factors, statistics, BH). Tests:
`tests/test_lit2_lab.py`. Results: branch `results22` (run 40213570596), folder `sweep_results/lit2/`.
- Data: XAU/USD 2011-05 to 2018-06, XAG/USD 2010-11 to 2018-06, the 9 USD pairs 2007-2018 (EUR/USD to 2022).
- No 2026 data.
- The candidates come from the second literature search (`docs/literature_check.md`, section 4). The rules were
  fixed before the run. Execution and costs follow `docs/flows_lab.md`.
- Metals swap: a long position pays the USD 3-month rate plus 0.75% a year.

## Results (net with swap, bp per event)

| Test | Rule | Events | Net | t | Years + | Gross (t) | BH q |
|---|---|---|---|---|---|---|---|
| S1 | silver fix: short 11:45 -> 12:15 London | 1,882 | -22.0 | -29.8 | 0% | **+3.6 (5.1)** | 1.0 |
| G1 | gold PM fix: short 14:45 -> 15:15 London | 1,832 | -3.8 | -6.4 | 0% | +0.2 (0.4) | 1.0 |
| G2 | gold AM fix: short 10:15 -> 10:45 London | 1,835 | -3.0 | -10.2 | 0% | **+1.0 (3.6)** | 1.0 |
| G3 | gold PM fix: follow 15:00 -> 15:03 until 15:15 | 1,833 | -4.4 | -12.4 | 0% | -0.4 (-1.1) | 1.0 |
| A1 | gold Asian hours: long 18:00 NY -> 08:00 London | 1,828 | -4.8 | -4.4 | 13% | **+2.4 (2.2)** | 1.0 |
| A2 | gold Western hours: short 08:00 London -> 16:55 NY | 1,779 | -2.2 | -1.0 | 25% | +2.6 (1.2) | 1.0 |
| SEAS | gold long in September and November (per month) | 14 | -314 | -2.7 | 14% | -297 | 1.0 |
| DC | dollar carry, 7 currencies (per month) | 130 | -1.4 | -0.07 | 50% | -1.3 | 1.0 |
| FM1 | dollar-factor momentum (per month) | 127 | +25.4 | 1.16 | 67% | +26.4 (1.21) | 1.0 |
| FM2 | carry-factor momentum (per month) | 122 | -7.8 | -0.35 | 50% | -4.7 | 1.0 |
| WM | weekly momentum, 4 / 4 weeks (per 4 weeks) | 145 | -23.6 | -1.09 | 42% | -13.5 | 1.0 |

**0 of 11 survive.**

Fix reforms (net with swap, bp per event):

| Test | Before the auction: gross | Before: net (t) | After: gross | After: net (t) | Reform |
|---|---|---|---|---|---|
| S1 | +6.2 | -17.1 (-13.0) | +1.3 | -26.4 (-36.3) | silver auction from 15 August 2014 |
| G1 | -0.3 | -4.4 (-4.9) | +0.9 | -3.0 (-4.2) | gold auction from 20 March 2015 |
| G2 | +1.7 | -2.4 (-5.2) | +0.2 | -3.7 (-11.1) | gold auction from 20 March 2015 |
| G3 | -0.2 | -4.3 (-7.9) | -0.6 | -4.5 (-10.7) | gold auction from 20 March 2015 |

Diagnostics, outside the BH family:

| Test | Events | Net | t | Gross (t) | By period |
|---|---|---|---|---|---|
| Silver Asian hours (long) | 1,855 | -37.6 | -20.2 | -1.5 | |
| Silver Western hours (short) | 1,807 | -27.1 | -7.1 | +2.3 | |
| Silver September / November | 14 | -586 | -2.5 | | |
| Carry factor, always (per month) | 130 | +10.6 | 0.49 | | |
| Long foreign currencies 2 days before every FOMC | 121 | +11.8 | 1.91 | +15.3 (2.46) | 2007-2010 +32.7 (t 1.76); 2011-2022 +5.0 (t 0.92) |
| Dollar before FOMC rate changes, realised decision (look-ahead) | 15 | +36.7 | 1.24 | +40.8 | |

## What it shows

- **Real before costs, too small after them:**
  - **The silver fix decline is real:** +3.6 bp a day gross, t 5.1. It was +6.2 bp before the 2014 auction and
    +1.3 bp after it. Silver's quoted spread in this data costs about 26 bp per round trip, so it is 4-7 times
    the effect.
  - **The gold AM fix decline** (+1.7 bp before the 2015 auction, +0.2 after) and **gold rising in Asian hours**
    (+2.4 bp, t 2.2) are also real, against about 4-7 bp of gold costs.
  - This matches the CBS thesis ("beat the market before transaction costs … underperform after costs") and
    Caminschi's silver figures.
- **Gold PM fix:** no decline in 2011-2015 here. The Abrantes-Metz & Metz pattern was strongest in 2004-2010.
- **Gold seasonality** went the wrong way: 2011-2017 was a gold bear market, and Septembers and Novembers lost
  3% on average. With 14 months this says little either way.
- **Currency factors:**
  - Dollar carry and carry-factor momentum are about zero.
  - Dollar-factor momentum is the best (+25 bp a month, t 1.16), positive in 2007-2010 and weak since.
  - Weekly momentum is negative.
  - This is the same post-2008 picture as Hutchinson et al. and the 2010-2025 momentum study.
- **The 2-day pre-FOMC dollar weakness** follows the FOMC-day pattern of run 17: real until 2010, small since.
