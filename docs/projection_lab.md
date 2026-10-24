# Projection lab: projections where the price is not random

Code: `scripts/projection_lab.py` and `scripts/projection_report.py`.
Cloud runs 4 and 5 (branch `results5`): 34 pairs x 14 timeframes x 20 projections = **9,520 tests**.
Data: 2007-2018 development data (EURUSD to 2022), with 3-year training and 6-month test windows rolled forward.

## Projections tested

| Group | Projections |
|---|---|
| Symbolic dynamics | ordinal patterns (3, 4, 5 and 6 closes); Markov contexts of up/flat/down moves (1, 2, 3 and 5 moves) |
| Delay embedding | Sugihara-May simplex projection (2, 4 and 8 dimensions) |
| Intrinsic time | directional-change events (3x and 10x volatility) |
| Cycle phase | phase of the band-pass-filtered log price |
| Memory | fractional differencing, d = 0.4 |
| Time | hour of the week (day of the week on D1) |
| Visibility graph | horizontal visibility graph degree |
| Path signature | Lévy area over 8 and 32 bars |
| Record times | bars since the last 50-bar high and low |

A test is called **non-random** when it meets both conditions below:
- The out-of-sample forecast correlates with the next move after Benjamini-Hochberg correction across all
  9,520 tests.
- It beats three sign-randomised surrogates by more than 3 standard deviations. The surrogates keep the
  volatility clustering, heavy tails and clock of the series and destroy its direction.

A test is **tradable** when, in addition, its gross edge per bar exceeds the round-trip cost and the
cost-aware rule (trade only when the forecast exceeds the round-trip cost) has a positive net Sharpe.

## Results

| TF | Non-random | Median t | Max t | Median hit (non-random) | Info (bits/bar) | Edge / round-trip cost (median) | Best non-random edge / cost | Tradable |
|---|---|---|---|---|---|---|---|---|
| M1 | 86% | 24.8 | 84.7 | 51.4% | 0.0010 | 0.005 | 0.02 | 0 |
| M5 | 83% | 15.4 | 48.3 | 51.9% | 0.0017 | 0.014 | 0.04 | 0 |
| M10 | 76% | 9.5 | 32.8 | 52.0% | 0.0017 | 0.017 | 0.06 | 0 |
| M15 | 71% | 6.5 | 27.9 | 52.0% | 0.0016 | 0.017 | 0.07 | 0 |
| M30 | 57% | 3.4 | 17.5 | 51.9% | 0.0010 | 0.019 | 0.09 | 0 |
| tick bars | 14% | 0.9 | 9.2 | 51.0% | 0.0000 | 0.010 | 0.10 | 0 |
| volatility bars | 20% | 1.2 | 10.3 | 51.2% | 0.0000 | 0.014 | 0.11 | 0 |
| H1 | 35% | 1.8 | 9.9 | 51.6% | 0.0005 | 0.019 | 0.12 | 0 |
| H2 | 11% | 0.8 | 5.4 | 51.3% | <0 | 0.019 | 0.14 | 0 |
| H4 | 4% | 0.4 | 5.1 | 51.4% | <0 | 0.017 | 0.20 | 0 |
| H6 | 3% | 0.2 | 5.4 | 51.7% | <0 | 0.013 | 0.30 | 0 |
| H8 | 4% | 0.1 | 4.5 | 51.6% | <0 | 0.015 | 0.61 | 0 |
| H12 | 1% | 0.0 | 3.1 | 51.9% | <0 | 0.002 | 0.48 | 0 |
| D1 | 0.3% | 0.0 | 4.1 | 51.7% | <0 | 0.000 | 0.11 | 0 |

Overall, 3,168 of the 9,520 tests are non-random, and **0 are tradable**. The strongest projections are:
- at M1: cycle phase, fractional differencing, path signature, Markov contexts and ordinal patterns
  (median t of 33-44 across all 34 pairs);
- at H1: Markov contexts, ordinal patterns and the visibility graph (t of about 4-5).

## Interpretation

- **The price is not random at short timeframes.** Every symbolic, phase and signature projection detects
  structure at M1-M30 on nearly every pair, far beyond the surrogates. It is mainly short-horizon mean
  reversion: the lag-1 autocorrelation of mid returns is -0.014 to -0.038 at M1-M15 on EURUSD, GBPUSD and
  USDJPY.
- **The structure is tiny in price terms.** The predictable part is 0.02-0.13 bp per bar against a round-trip
  cost of about 5.8 bp: 0.5% of the cost at M1 and about 2% at M30-H1.
- **Longer bars are larger relative to costs, but the structure fades.** Above H2 it is gone. The two curves
  never meet: the best non-random test anywhere reaches 0.61 of its cost (EURCHF H8, clock projection).
- **Information bound.** No projection of the past can carry more information about the next move than the
  past itself (data-processing inequality). The best projections extract about 0.001-0.002 bits per bar. That
  is the ceiling any reshaping of the price can reach on this data.
- **Net-positive rows exist only among non-significant tests.** The best is CADJPY H12, directional change
  3x, with net Sharpe 0.95 on 5 trades a year. With 9,520 tests, such rows are expected by chance.
