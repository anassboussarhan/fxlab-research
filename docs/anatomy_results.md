# Round 12 results: the queued classic systems, and why every failed strategy failed (2026-11-08)

Rules: `docs/anatomy_preregistration.md` (with addendum 1, written before the rerun was read). Code
`scripts/anatomy_systems.py`, `scripts/anatomy_lab.py`, `scripts/anatomy_combine.py` and `scripts/anatomy_deep.py`.
The results are from rerun 42520497617: the combined tables are on the `anatomy` branch and the cross-unit look is
on `misc-anatdeep`. 2026 is not used.

**Nothing counts.**
- **Part A:** none of the 19 rows of the nine queued families counts. All but one have a negative check-period mean.
- **Part B:** no pre-registered fix counts for any of the 196 strategy units.
- **Placebos:** none of the 5 pooled placebo units and none of the 35 per-instrument placebo runs passes.
- **Chance level:** only 3 of 196 chosen fixes reach t 1.645 on 2019-2025 (1.5%). That is below what chance alone
  would give.

## Part A: the nine families as published

| Unit | Fit 2007-2018: n, net bp (t) | Check 2019-2025: n, net bp (t) | Gross bp fit / check | Cost bp | Win rate | Label |
|---|---|---|---|---|---|---|
| MS1 Malaysian SNR H1 | 19,899, -2.95 (-14.4) | 12,352, -2.41 (-10.9) | -1.09 / -0.61 | 2.94 | 17% | no gross edge |
| MS1 Malaysian SNR H4 | 8,422, -3.45 (-5.7) | 5,753, -2.58 (-4.5) | -1.54 / -0.87 | 2.83 | 17% | no gross edge |
| QM1 Quasimodo H1 | 6,868, -3.78 (-8.4) | 4,398, -4.01 (-7.5) | -1.72 / -1.81 | 2.79 | 30% | no gross edge |
| QM1 Quasimodo H4 | 1,380, -1.84 (-0.9) | 1,031, -3.70 (-1.7) | +0.18 / -1.74 | 2.91 | 31% | no gross edge |
| CL1 Dual Thrust H1 | 3,028, -1.96 (-0.5) | 1,810, -3.86 (-0.9) | +2.68 / +0.40 | 2.75 | 34% | no gross edge |
| CL2a R-Breaker H1 | 3,718, -2.46 (-2.8) | 3,258, -4.65 (-5.8) | +2.00 / +0.78 | 2.80 | 42% | cost-bound |
| CL2b Fiali four prices H1 | 14,563, -6.06 (-9.9) | 9,129, -6.06 (-7.5) | -1.52 / -0.11 | 2.83 | 44% | no gross edge |
| CL3a Aberration H4 | 2,578, -2.29 (-0.7) | 1,663, -8.02 (-2.0) | +1.60 / -4.62 | 2.60 | 34% | no gross edge |
| CL3a Aberration D1 | 417, -25.29 (-1.4) | 234, -20.64 (-1.2) | -12.27 / -5.30 | 5.92 | 33% | no gross edge |
| CL3b King Keltner H4 | 3,509, -2.10 (-0.7) | 2,400, -5.32 (-1.7) | +1.70 / -2.01 | 2.61 | 30% | no gross edge |
| CL3b King Keltner D1 | 603, -0.94 (-0.1) | 352, -16.94 (-1.1) | +11.93 / -2.16 | 6.20 | 29% | no gross edge |
| CL4a Thermostat H4 | 4,925, -0.72 (-0.3) | 3,290, -6.92 (-2.9) | +3.06 / -3.65 | 2.61 | 38% | cost-bound |
| CL4a Thermostat D1 | 825, -19.92 (-1.6) | 529, -4.91 (-0.5) | -7.77 / +9.42 | 6.20 | 34% | no gross edge |
| CL4b Bollinger Bandit H4 | 3,578, -1.64 (-0.6) | 2,333, -5.55 (-1.9) | +1.98 / -2.40 | 2.57 | 37% | no gross edge |
| CL4b Bollinger Bandit D1 | 580, -12.55 (-0.9) | 354, -19.49 (-1.4) | -0.71 / -6.38 | 5.93 | 34% | no gross edge |
| CL5 Camarilla H1 | 15,406, -2.63 (-10.1) | 9,689, -2.58 (-10.7) | -0.37 / -0.11 | 2.99 | 45% | no gross edge |
| CL6 Elder triple screen D1 | 2,526, -2.77 (-0.7) | 1,433, +0.93 (0.2) | +1.44 / +6.59 | 3.07 | 31% | no gross edge |
| CL7 Puria H1 | 8,325, -1.80 (-3.2) | 5,266, -4.05 (-5.1) | +1.43 / -0.56 | 2.86 | 40% | cost-bound |
| CL7 Puria H4 | 1,970, +1.52 (0.6) | 1,380, -4.08 (-1.5) | +5.16 / -0.14 | 2.64 | 43% | decay |

Notes:
- The Bill Williams Alligator, the other half of CL7, is already in the catalog (`S.FAMOUS.ALLIGATOR`) and is in
  part B. It loses at every timeframe; on 2019-2025: H1 -2.94 bp (t -8.1), H4 -3.31 bp, D1 -14.3 bp.
- MS1 and Camarilla lose with very high t. These are large samples of trades with no gross edge, so they lose
  about the cost on every trade, with t of -10 or worse.
- The MS1 fade wins only 17% of the time. A fresh level breaks more often than it holds.

## Part B: why each strategy fails

Each of the 197 strategy units (the 19 above plus 178 catalog units at H1 / H4 / D1, pooled over seven instruments)
got one pre-registered label from its 2007-2018 trades:

| Label | Units | What it means here |
|---|---|---|
| No gross edge | 151 | Before costs the trades are a coin toss. Their median gross per trade is -0.40 bp (fit) and -0.24 bp (check), against a median round-trip cost of 2.97 bp. No exit, filter or entry rule can create a gross edge that is not there. |
| Cost-bound | 25 | There is a real gross edge, but it is smaller than the cost. Median gross is +1.41 bp in the fit period against a 2.75 bp cost, and it fell to +0.30 bp in 2019-2025 (smaller in 80% of these units). |
| Decay | 14 | Net positive in 2007-2018, negative in 2019-2025. Most are D1 units with a few hundred trades. |
| Skew trap | 5 | 65-66% winners, but the average loss is 2.1-2.7 times the average win. |
| Weak | 2 | Positive in both periods but not significant. |

### Cost-bound: the closest calls

| Unit | Gross fit bp (t) | Cost bp | Gross check bp | Net check bp (t) |
|---|---|---|---|---|
| Connors RSI(2) H1 | +1.47 (6.7) | 2.88 | +0.45 | -2.28 (-7.8) |
| Intraday seasonality H1 | +0.65 (6.4) | 3.25 | +0.44 | -2.71 (-22.2) |
| Intraday seasonality H4 | +1.47 (4.5) | 2.70 | +0.02 | -2.89 (-6.9) |
| London fix reversal H1 | +0.56 (4.2) | 2.56 | -0.27 | -2.55 (-11.5) |
| Range expansion H1 | +1.70 (4.0) | 2.61 | +0.71 | -2.52 (-5.1) |
| Weekday seasonality H1 | +0.38 (3.5) | 3.24 | +0.40 | -2.76 (-21.7) |
| Williams volatility breakout H4 | +1.39 (3.1) | 2.63 | +0.24 | -3.38 (-4.9) |

To break even, the cost-bound units would need costs about half as large on 2007-2018 trades (median cut 50%). On
2019-2025 they would need costs about 90% lower (median cut 91%).

For scale, the random-entry placebo that holds 24 hours shows a gross of +0.83 bp (t 1.6). A gross t between 1 and 2
is therefore within what chance produces.

### Decay: positive before 2019, negative after

| Unit | Fit n, net bp | Check n, net bp (t) |
|---|---|---|
| TTM squeeze D1 | 279, +24.5 | 132, -26.8 (-2.1) |
| Channel D1 | 287, +20.4 | 166, -29.2 (-1.0) |
| Fibonacci extension D1 | 763, +12.6 | 414, -4.5 (-0.5) |
| Moving-average cross D1 | 237, +7.2 | 124, -56.6 (-1.5) |
| Change-point regime H4 | 745, +5.3 | 581, -4.0 (-1.0) |
| Short-term reversal D1 | 492, +4.2 | 296, -21.9 (-1.6) |

### Skew traps

| Unit | Win rate | Average win / loss bp | Check net bp |
|---|---|---|---|
| Z-score reversion H1 | 65% | +54.9 / -113.5 | -1.24 |
| Z-score reversion H4 | 66% | +107.1 / -222.6 | +3.78 (t 0.5) |
| Wavelet H1 | 66% | +34.2 / -78.3 | -2.35 |
| Wavelet H4 | 66% | +63.2 / -146.6 | -2.84 |
| Wavelet D1 | 66% | +148.8 / -403.6 | +2.67 (t 0.2) |

These are the "high accuracy" systems that the YouTube, forum and EA claims describe: two of three trades win.
None has a gross edge in the fit period (gross t from -1.3 to +0.1). The losses are large enough to cancel the
wins.

### Weak: positive in both periods

- **Range expansion D1:** +19.9 bp on 253 trades (2007-2018) and +12.0 bp on 227 trades (2019-2025), but t 0.62.
- **Regression trend D1:** too few trades (27 and 9).

Only 3 of 197 units have a positive base mean in both periods: these two and Halloween D1 (+1.8 / +26.1 bp, t 0.4).

### "It was in profit before it lost" is not an exit problem

Across units, a median 92% of losing trades had at some point been ahead by more than a round-trip cost. Only 5% of
trades never moved a cost in their favour. That looks like an exit problem, but random entries look the same:

- the 24-hour random placebo has 91% and 5%;
- the 6-hour placebo has 81% and 11%.

A price path that goes up and down passes through profit on the way to a loss. Seeing that in the losing trades
says nothing about a better exit. Changing the exits (below) confirms it.

## Part B: the fix search

For each unit the menu crossed:
- 4 entry methods (E0 market at the bar open, E1 15 minutes later, E2 a limit 0.25 ATR better, E3 a stop 0.25 ATR
  worse);
- 3 exits (X05 half the hold, X1 the strategy's own, X2 double the hold);
- reversing the direction;
- at most one of nine filters.

The combination with the best 2007-2018 trade-level t was the unit's fix. It was judged once on 2019-2025.

| | Result |
|---|---|
| Units with a fix that counts | 0 of 196 |
| Median t of the chosen fix, fit then check | +1.15, then -1.37 |
| Units where even the best combination is negative in the fit period | 29% |
| Units where the best combination is below t 1.645 in the fit period | 63% |
| Fixes at t >= 1.645 on 2019-2025 | 3 (1.5%) |
| Placebo units passing (pooled; per instrument) | 0 of 5; 0 of 35 |

The three fixes at t >= 1.645 rest on 25-38 trades each and miss the Benjamini-Hochberg threshold by a factor of
40 or more (p 0.02-0.04; the threshold for the best of 196 is 0.0005):

| Unit | Fix | Fit n, bp | Check n, bp (t) |
|---|---|---|---|
| Z-score reversion D1 | double hold, Asia-session entries only | 46, +235 | 38, +163 (2.04) |
| Halloween D1 | stop entry, double hold, not near a 00 level | 31, +246 | 25, +144 (1.85) |
| Double Seven D1 | entry 15 minutes later, double hold, low-spread entries | 140, +40 | 35, +33 (1.73) |

The fix types chosen most often were:
- reversing the direction (54 units);
- double holds (78 units in total across entries);
- filters on the 24-hour move (35 units) and the spread at entry (30 units).

None of these choices carried over.

## What does not carry over, and the one thing that does (information only)

This part averages over all units. It is not a test that can count (`scripts/anatomy_deep.py`).

**Filters.** There are 8,870 unit x filter pairs with enough trades in both periods.
- The effect a filter had in 2007-2018 does not predict its effect in 2019-2025: the correlation is -0.008, and the
  sign is the same 54% of the time.
- By feature, the fit-to-check correlation is:

| Feature | Correlation, fit vs check effect | Mean check effect of the top tenth of fit effects |
|---|---|---|
| Distance to a 00 level | +0.37 | +4.3 bp |
| Weekday | +0.23 | +2.1 bp |
| Spread at entry vs the hour's median | +0.11 | +3.5 bp |
| Calendar day (Gotobi / month-end) | +0.03 | -2.7 bp |
| 20-day trend strength | -0.06 | -1.0 bp |
| 24-hour move in the trade's direction | -0.12 | -1.1 bp |
| With or against the 20-day return | -0.16 | -1.2 bp |
| Session | -0.22 | -0.9 bp |
| ATR percentile (volatility regime) | -0.29 | -1.6 bp |

- Three features keep part of their effect: the 00 levels, the weekday and the spread. The spread one is mostly a
  cost cut: skipping wide-spread entries pays less to trade.
- Even for these three, the filtered strategies still lose. Their mean check net is -8.2, -8.6 and -10.0 bp a
  trade, and only 24-27% of these filtered units are positive on 2019-2025.
- The volatility and session filters, the most popular ones in the retail material, do worse than nothing: what
  helped before tends to hurt after.

**Entries.**
- **E2, a limit 0.25 ATR better:** about +5 to +6 bp per filled trade in both periods, better than E0 in 100% of
  units (E2 with the strategy's own exit). This is the price improvement on the trades that fill. 36-68% of trades
  fill (median 59%), and the ones that fill are the ones that first went against the signal. The level stays negative: E2 is
  positive in both periods in 9 of 185 units.
- **E3, a stop entry 0.25 ATR worse:** costs about 4 bp a trade in both periods, in 85-100% of units. Confirmation
  entries pay for confirmation that adds nothing.
- **E1, waiting 15 minutes:** neutral (+0.1 bp).

**Exits and reversal.**
- Half hold: +0.05 to +0.3 bp. Double hold: -0.2 to -0.3 bp. Neither changes a loser into a winner.
- Reversing: median +0.1 bp. The fit-to-check correlation is -0.35, so units where reversing helped before tend to
  be hurt by it after. A strategy with no gross edge, reversed, still has no gross edge and still pays the cost.

**Positive in both periods.** 36 of 2,235 unit x variant pairs are positive in both periods (1.6%). 31 of them are D1
units with a few hundred trades. 23 use the limit entry, with a median fill of 43% of signals.

## By family

| Family | Units | No gross edge | Cost-bound | Decay | Skew trap | Median gross fit bp | Median cost bp | Median win rate | Median check net bp | Median check t of the fix |
|---|---|---|---|---|---|---|---|---|---|---|
| breakout | 15 | 11 | 3 | 0 | 0 | +0.27 | 2.73 | 39% | -3.44 | -1.67 |
| classic (part A) | 19 | 15 | 3 | 1 | 0 | +1.43 | 2.83 | 34% | -4.65 | -1.67 |
| daily | 9 | 8 | 0 | 1 | 0 | +3.72 | 5.66 | 56% | -6.09 | +0.77 |
| event | 1 | 1 | 0 | 0 | 0 | -0.84 | 4.55 | 39% | -4.14 | -1.54 |
| famous | 25 | 17 | 6 | 2 | 0 | +0.69 | 2.88 | 37% | -3.54 | -1.63 |
| forum | 4 | 4 | 0 | 0 | 0 | -0.06 | 2.94 | 22% | -2.79 | -0.37 |
| geometry (Fibonacci, harmonics) | 12 | 10 | 0 | 2 | 0 | -0.24 | 3.09 | 40% | -4.15 | -1.12 |
| market structure | 16 | 13 | 2 | 1 | 0 | -0.50 | 2.97 | 39% | -3.66 | -1.05 |
| mean reversion | 21 | 18 | 0 | 1 | 2 | -0.67 | 2.77 | 59% | -2.61 | -0.61 |
| microstructure | 4 | 4 | 0 | 0 | 0 | -0.04 | 2.85 | 39% | -3.42 | -4.04 |
| regime | 15 | 12 | 1 | 2 | 0 | -0.11 | 3.00 | 42% | -3.30 | -1.02 |
| seasonality | 12 | 5 | 6 | 1 | 0 | +0.61 | 2.97 | 47% | -2.63 | -1.81 |
| session | 5 | 5 | 0 | 0 | 0 | -0.10 | 2.61 | 47% | -2.31 | -3.48 |
| signal processing | 9 | 6 | 0 | 0 | 3 | -1.79 | 2.91 | 58% | -3.68 | -0.72 |
| smart money (SMC / ICT) | 6 | 5 | 1 | 0 | 0 | -1.34 | 3.01 | 33% | -3.72 | -1.59 |
| trend | 15 | 10 | 2 | 2 | 0 | +0.79 | 2.97 | 34% | -4.09 | -1.63 |
| volatility | 9 | 7 | 1 | 1 | 0 | +0.06 | 2.89 | 39% | -3.62 | -0.30 |

Seasonality has the most real gross edges, which fits the known intraday and weekday patterns in FX. Those edges
are 0.4-1.5 bp a trade at H1 / H4, a fifth to a half of the round-trip cost.

## A look-ahead found and removed

In the first run, the half-hold exit used half of the trade's own future duration. That duration is not known at
entry when the trade ends on a later signal.

- It produced 53 false "passes", for example an H1 moving-average cross going from -3 bp to +29 bp a trade at t 13.
- The placebos, whose holds are fixed at entry, showed none.
- It was fixed before any result was used (addendum 1). The rerun is the one reported here.
- A second bug was also fixed before the rerun: reversing a resting limit or stop order had created marketable
  entries.

## What this means

- **Why they fail.** Three quarters of these strategies have no edge before costs. Another eighth have a real one
  that is about half the cost and has shrunk since 2019. The rest decayed, or win often and lose big.
- **Search space.** No entry method, exit, reversal or pre-entry filter chosen on 2007-2018 makes any of them pay
  on 2019-2025. Parameters had already been searched jointly with entries and exits in the catalog (about 762,000
  candidates, `docs/conclusions.md` 8.1c), with the same result.
- **What does carry over is about cost, not direction:**
  - resting limit entries improve the price of the trades that fill;
  - skipping wide-spread moments helps.
  These belong in how any trade is executed, the Gotobi trade included. They do not turn a strategy with no edge
  into one with an edge.
- **Nothing is added to the forward watch list.** Range expansion D1 (t 0.62) is too weak to be worth tracking.
