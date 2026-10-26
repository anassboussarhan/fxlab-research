# Gotobi out of sample, 2019-2025

The Gotobi effect was the only spot-FX edge that survived the 2007-2018 tests: USD/JPY tends to rise into the Tokyo
fix (09:55 JST) on Gotobi settlement days, the 5th, 10th, 15th, 20th, 25th and last day of the month. In runs 17 /
19 (`docs/flows_lab.md`) it earned +3.41 bp per trade net of costs and swap (t 2.88, 700 trades). This is the test
on data nobody here had looked at.

- Script: `scripts/gotobi_oos.py`.
- Workflow `gotobi-oos`, run 40227329900. Outputs are on branch `gotobi-oos`.
- The rules were fixed before the download and are unchanged from run 17: long USD/JPY from 03:00 JST to 09:55 JST
  on Gotobi days (moved back to the previous Tokyo business day), Mondays skipped. The result is net of spread,
  slippage, commission and the rate-differential swap.
- The pre-registered verdict rule: confirmed if the mean is > 0 with one-sided t >= 1.645; strong if t >= 2.
- Development data only: the download ends 2025-12-31 23:59 UTC.

## Data

USD/JPY ticks were downloaded directly from Dukascopy (free), one GitHub job per month, and built into M1 bid/ask
bars:
- 2,339,354 bars, 2019-01-01 to 2025-12-31, median spread 0.5 pip.
- In every month 27-119 hourly files failed to download after six retries (Dukascopy throttling), about 10% of
  hours. The missing hours are spread evenly across the day: 94,000-98,700 bars in every UTC hour. A trade whose
  entry or exit bar is missing is dropped.

## Result: confirmed, smaller than in 2007-2018, and fading

| Test | Trades | Net mean (bp) | t | Sharpe | Years positive | Gross (bp) | Raw ECN cost (bp, t) | IBKR cost (bp, t) |
|---|---|---|---|---|---|---|---|---|
| **F1 Gotobi, 03:00 JST to the fix** | 413 | **+2.13** | **2.05** | 0.80 | 5/7 | +2.81 | +2.46 (2.37) | +2.76 (2.66) |
| F1, same window on other days | 910 | -0.13 | -0.17 | | 3/7 | +0.47 | | |
| F1 with the exit at 10:00 JST | 408 | +0.52 | 0.49 | | 4/7 | +1.26 | | |
| F2 Gotobi, 08:00 JST to the fix | 472 | +0.33 | 0.44 | | 4/7 | +2.24 | | |
| F3 Gotobi reversal, short from the fix to 12:00 JST | 480 | +1.47 | 2.30 | 0.83 | 5/7 | +2.93 | | |
| B1 random entry minute 03:00-09:00 JST, exit at the fix | 490 days | +1.01 | 1.30 | | 5/7 | +2.53 | | |

- **Verdict (pre-registered rule): confirmed, t >= 2.**
- **Random dates:** against the same window on random non-Gotobi days of the same year and weekday, F1 beats the
  draws with p = 0.004 (random-date mean -0.78 bp, 95th percentile +1.00 bp). Welch t against the other days is
  1.75.
- **Smaller:** +2.13 bp is about 60% of the 2007-2018 +3.41 bp.
- **Fading.** By year, in bp per trade:

  | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
  |---|---|---|---|---|---|---|
  | +0.69 | +3.26 | +3.53 | +6.12 | +1.94 | -0.08 | -0.88 |

  The last two years are slightly negative.
- **Exit timing matters.** Exiting at 10:00 JST, five minutes after the fix, keeps only +0.52 bp (t 0.49). The price
  peaks at the fix and reverses after it, which also shows in F3.
- **F3, the reversal after the fix, was flat in 2007-2018** (-0.04 bp) and is +1.47 bp (t 2.30) here. It was a
  secondary test, so it is a hypothesis for later data, not a confirmed edge.

## On an FTMO-type challenge (2019-2025 trades)

The same simulation as `docs/prop_firms.md`:
- FTMO 100k 2-step: EUR 540 fee, +10% then +5%, 5% daily and 10% maximum loss, 80% split.
- The 413 F1 trades in date order, from random starts, at fixed leverage.

Results are in `docs/gotobi_oos_prop.csv`.

| Leverage | Mean trade (% of account) | Pass both phases | Paid at least once | Mean net | Median net | Failed on the daily loss |
|---|---|---|---|---|---|---|
| 3x | 0.064% | 8% | 8% | -$356 | -$585 | 0% |
| **6x** | 0.128% | **62%** | **61%** | **+$4,013** | **+$2,928** | 0% |
| 10x | 0.213% | 34% | 29% | +$4,906 | -$585 | 75% |
| 20x | 0.426% | 48% | 27% | +$3,398 | -$585 | 92% |

- At raw-ECN costs the 6x row is: paid 64%, mean net +$4,920, median +$3,656.
- **3x is now too slow.** At the smaller edge, about 115 trades in the 500-day limit do not reach +10%.
- **10x and above fail on the daily limit:** the worst trade was -70 bp, which at 10x is a 7% day.
- **6x is the only size that fits the rules,** with the worst trade at -4.2% of the account. The leverage grid (3, 6,
  10, 20) was set before this data. Choosing 6x after seeing this table is a selection, and the simulation reuses
  the same seven years, including the negative 2024-2025.

## Conclusion

- The Gotobi effect held up out of sample in 2019-2025 (+2.1 bp a trade after costs, t 2.05, random-date
  p = 0.004).
- It was weaker than before and slightly negative in 2024 and 2025.
- It only works with an exit at the fix itself.
- It remains the only edge in this project that passed a pre-registered out-of-sample test. On a prop account it
  needs about 6x leverage and a precise 09:55 JST exit, and the 2024-2025 fade is the main risk.
- 2026, the locked holdout, has not been used.
