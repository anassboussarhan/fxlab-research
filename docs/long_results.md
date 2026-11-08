# Round 13 results: long trades that let winners run (2026-11-08)

Rules: `docs/long_preregistration.md`, committed before the run. Code: `scripts/long_lab.py` (engine and systems),
`scripts/long_combine.py` (tests) and `scripts/long_diag.py` (a breakdown, information only). Run 42525289851:
33 instruments, every job finished with 0 errors. The tables are on the `long` branch and in `docs/long/`; the
breakdown is on `misc-longdiag`. 2026 is not used.

**Short answer to the question.**
- Holding for weeks or months does make the cost small: for the Turtle system it is a fifth of the gross move.
- But on FX the gross move itself is negative. From 2019 to 2025 currency moves reversed more often than they
  ran on, and in 2008-2018 trend following broke about even before costs.
- The systems lose with or without costs.
- On gold, silver and the indices, trend systems made money in 2019-2025, but each one made less than simply
  holding those markets long.

**Verdicts:**
- **Part A (12 long-trend systems):** 0 of 23 rows count, and no row reaches t 1.645 on 2019-2025.
- **Part B (catalog with long exits):** by the rule as written, 1 of 109 units counts. It is Connors' "Double
  Seven" on weekly bars, kept with its own exit. It is a weekly mean-reversion trade, not a trend trade, and it
  lost in 2008-2018 (details below).
- None of the four "let winners run" exits produced a passing unit. They made most strategies worse than their
  own exits.
- **Placebos:** none of the coin-flip or random-sign controls passes.

## Part A: the long-trend systems (portfolio at 10% annual volatility, monthly returns, Newey-West t)

| System | FX 2008-2018: mean %/month (t) | FX 2019-2025: mean %/month (t); gross t | Non-FX 2019-2025: mean (t) | Non-FX minus long-only (t) |
|---|---|---|---|---|
| L1 Turtle System 2 (55 / 20, pyramids) | -0.21 (-0.9) | -0.86 (-3.9); -2.8 | +0.30 (0.7) | -0.76 (-2.2) |
| L2 Turtle System 1 (20 / 10, pyramids) | -0.33 (-1.4) | -1.20 (-4.0); -2.6 | +0.10 (0.3) | -0.95 (-2.3) |
| L3 Clenow core trend (3N trailing stop) | -0.18 (-0.8) | -0.66 (-3.2); -2.3 | +0.41 (1.0) | -0.65 (-1.7) |
| L4 Donchian 100 / 50 | -0.10 (-0.4) | -0.47 (-1.7); -0.9 | +0.56 (1.3) | -0.50 (-1.2) |
| L5 Moving average 50 / 200, always in | -0.24 (-0.9) | -0.34 (-0.9); -0.3 | +0.83 (1.5) | -0.23 (-0.6) |
| L6 Time-series momentum 12 months | -0.20 (-0.8) | -0.27 (-0.9); +0.1 | +0.67 (1.4) | -0.39 (-0.9) |
| L7 Time-series momentum 1 / 3 / 12 | +0.01 (0.1) | -0.32 (-1.0); -0.2 | +0.29 (0.6) | -0.76 (-1.5) |
| L8 Faber 10-month average | -0.13 (-0.5) | -0.39 (-1.2); -0.4 | +0.23 (0.5) | -0.83 (-1.9) |
| L9 EWMAC (Carver) | -0.13 (-0.5) | -0.31 (-1.2); -0.4 | +0.56 (1.2) | -0.50 (-1.3) |
| L10 Squeeze release, 3N trailing stop | -0.02 (-0.1) | -0.71 (-3.0); -2.0 | +0.11 (0.3) | -0.95 (-2.1) |
| L11 Weekly breakout 20 / 10 | -0.18 (-0.7) | -0.43 (-1.3); -0.7 | +0.43 (0.9) | -0.63 (-1.6) |
| L12 EWMAC with carry filter (FX only) | -0.50 (-2.5) | -0.08 (-0.2); -0.5 | - | - |
| PL1 control: coin-flip entries, 3N trailing stop | -0.40 (-1.8) | -1.18 (-3.7); -2.2 | -0.39 (-0.8) | -1.45 (-2.8) |
| PL2 control: random-sign momentum | -0.67 (-2.9) | -0.68 (-2.7); -0.8 | -0.50 (-1.0) | -1.56 (-2.6) |
| BH long-only benchmark | -0.21 (-0.9) | +0.05 (0.2); +0.5 | **+1.06 (2.5)** | - |

- **FX** is the 28 crosses of the eight major currencies. **Non-FX** is gold, silver, the S&P 500, the Nasdaq-100
  and the Nikkei.
- At 10% volatility, -0.86% a month is about -10% a year.
- The first year was the best one: Turtle System 2 made +25% on FX in 2008 and has lost in most years since
  (2019-2025: -10, -3, -9, -1, -14, -18, -19%).
- The whole 33-instrument book does no better: every system's 2019-2025 t is between -3.8 and -0.2 (the long-only
  benchmark: +0.8).

### Do longer trades make the cost small? Yes. They still lose (FX, 2019-2025)

| System | Trades | Median hold (days) | Win rate | Average win: pips, R | Average loss: pips, R | Gross R | Costs R | Swap R | Net R |
|---|---|---|---|---|---|---|---|---|---|
| L1 Turtle System 2 | 1,354 | 8 | 16% | +303, +5.2 | -86, -2.0 | -837 | 171 | -152 | -1,159 |
| L2 Turtle System 1 | 2,615 | 9 | 18% | +279, +4.6 | -81, -1.7 | -962 | 346 | -214 | -1,522 |
| L3 Clenow | 739 | 18 | 29% | +231, +0.8 | -151, -0.6 | -111 | 17 | -23 | -151 |
| L4 Donchian 100 / 50 | 456 | 60 | 30% | +376, +2.2 | -271, -1.8 | -180 | 20 | -81 | -281 |
| L5 MA 50 / 200 | 317 | 112 | 26% | +633, +3.5 | -331, -2.3 | -129 | 7 | -104 | -240 |
| L10 Squeeze, trailing stop | 993 | 15 | 28% | +265, +1.4 | -134, -0.9 | -154 | 35 | -46 | -236 |
| L11 Weekly breakout | 313 | 65 | 25% | +554, +1.3 | -301, -0.9 | -68 | 4 | -27 | -100 |
| PL1 coin-flip, trailing stop | 807 | 14 | 29% | +271, +1.4 | -140, -0.9 | -105 | 29 | -47 | -181 |

R = the risk at the initial stop (0.5% of equity a unit). Pyramided Turtle trades can lose about 2R. The median
hold counts trades stopped out early, so most trades are short even in the slow systems.

- **Costs.** On the slow systems, costs are only 5-11% of the size of the gross result (L4, L5, L11), against
  about a fifth for the Turtles. That part of the idea works.
- **Payoff.** The trades have the shape trend followers want: winners 2-4 times the size of losers, hundreds of
  pips each.
- **Win rate.** It is too low for that payoff. At 16% winners, Turtle System 2 needs an average win of about
  10R to break even. It got 5.2R. The big runs that pay for the many small stops did not come often enough.
- **Placebo.** Per trade, the coin-flip control with a 3N trailing stop loses about what the trailing-stop systems
  lose: -0.22R, against -0.20R for L3 and -0.24R for L10. On this data the entry signals add nothing to the
  stop.
- **Swaps.** Swap is the hidden cost of holding. On the slow FX systems it is as large as the costs or larger
  (L4 -81R against 20R of costs; L5 -104R against 7R). At 10% volatility it averages:
  - FX: -0.11 to -0.27% a month, about -1.3% to -3.2% a year (the carry-filtered L12 earns +0.14%);
  - gold and the indices: -0.22 to -0.48% a month, from the financing on long positions.

### Gold, silver and the indices: the trend was the drift

- Every trend system is positive on the non-FX book in 2019-2025. Those markets rose in most years: gold in five
  of the seven, and the S&P 500 in all but 2022.
- Holding them long, scaled the same way, made +1.06% a month (t 2.5), more than any trend system.
- Their excess over holding long is negative for all 11 systems (t -0.6 to -2.3).
- What they earned came from being long when everything went up. Timing in and out cost money.
- 2022 shows it: the indices fell, the trend systems were still long into the fall, and they lost 8-25% that
  year at 10% volatility.

## Part B: the catalog with exits that let winners run

There are 109 strategy units: 61 catalog strategies allowed at D1, run on D1 and on W1 bars (122 units), keeping
the units with at least 30 fit-period trades in at least one exit. Each was tried with its own exit (T0) and
with four exits that let winners run (T1-T4):
- T1: a 3N trailing stop;
- T2: a 20-bar channel exit;
- T3: breakeven at 1R, then the trail;
- T4: half off at 2R, then the trail.

**Long exits make strategies worse, not better.** Median 2019-2025 t across units:

| Bars | T0 own exit | T1 trailing | T2 channel | T3 breakeven + trail | T4 half at 2R + trail |
|---|---|---|---|---|---|
| D1 (median hold in days) | -1.26 (15) | -2.07 (22) | -2.54 (18) | -2.07 (20) | -1.61 (22) |
| W1 | -0.42 (67) | -0.95 (110) | -0.42 (83) | -0.78 (102) | -0.81 (110) |

- Against each strategy's own exit, the trailing stop (T1) is better in 38% of units on 2008-2018 and in 29% on
  2019-2025.
- Breakeven (T3) is better in 38% and 35%, the half-at-2R exit (T4) in 40% and 38%.
- The channel exit (T2) is better in 53% of units on 2008-2018, but only in 33% on 2019-2025.
- The chosen fix, picked on 2008-2018, had a median fit t of -0.41 and a median check t of -1.20. For most units
  even the best of the five exits lost in the fit period.

**Placebos.** Random entries at three rates, on D1 and W1, went through the same five exits and the same choice:
0 of 6 pass.

### The one unit that counts: Double Seven on weekly bars, with its own exit

**Rule** (Connors, long-only):
- When the weekly close is above its 200-week average and is the lowest close of the last 7 weeks, buy at the
  next Monday 08:00 London fill.
- Sell after a weekly close that is the highest of 7 weeks.
- There is no stop.

| | 2008-2018 | 2019-2025 |
|---|---|---|
| Trades, all 33 instruments | 223 | 327 |
| Net, R (Newey-West t of monthly returns) | -6.9R (-0.35) | +64.3R (3.61; BH passes at q 0.10) |
| Gross t | 0.12 | 4.46 |
| FX only | -12.6R (-0.60) | +39.6R (2.37), 247 trades |
| Gold and silver | -2.1R (-0.94) | +10.5R (2.35), 36 trades |
| Indices | +7.8R (1.56) | +14.1R (1.72), 44 trades |
| Instruments positive | 16 of 33 | 22 of 33 |

**Why it is a lead and not an edge:**
- **It lost in 2008-2018.** Part A's rule required a positive fit period. Part B's rule, as I wrote it, only
  asked for the best of the five exits on the fit period. In this unit the best was negative. Under part A's rule
  this unit would not count. That gap is my pre-registration's, and I report the unit as counting because that is
  what was written.
- **It is the mirror image of the trend losses.** It buys weekly dips in up-trending markets and sells into
  strength. 2019-2025 is exactly when FX moves reversed more than they continued, the years that ruined trend.
- **Two similar weekly dip-buying rules** did well in the same years and missed the multiple-testing bar:
  - Connors cumulative RSI: check t 2.79, fit t +1.44;
  - Connors RSI(2): check t 2.29, fit t +0.07.
  - On daily bars the same three rules have check t 0.2-0.6 here and lost in round 12. The edge appears only on
    the slower bars, and only after 2018.
- **Few trades.** About 47 trades a year spread over 33 instruments, about 1.4 per instrument a year. It holds
  about 8 weeks and has no stop, so a single bad trend against a position can be large. The monthly data here do
  not give the worst trade. That needs the trade list before any sizing.

**Status:**
- Forward watch list from today. Nothing changes the rule.
- It could be tested once on the 2026 holdout (January-October). That would give about 40 trades, which is enough
  to contradict it but not to confirm it.

## Bugs found on the way

- The year-by-year columns of part A were scaled twice (about 7 times too small).
  - Found while reading the results, fixed in `scripts/long_combine.py`, and the columns recomputed from the
    monthly table.
  - No mean, t, drawdown or verdict changed. The rerun matches the first run's means and t to 1e-9.
- The engine was checked on synthetic data before the run:
  - Truncating the data leaves every earlier day unchanged, so nothing looks ahead.
  - With zero spread, slippage and commission, net minus gross equals the swap exactly.
  - Every system makes money on a planted trend.
- The long-only benchmark's yearly signs match the known moves of gold, the S&P 500, USD/JPY and EUR/USD.

## Data notes

- Twelve FX crosses start between 2009 and 2010 instead of 2007-2008. CHF/JPY and NZD/JPY end in October 2025.
- Gold and silver start in 2011, the indices in May / June 2012.
- The S&P 500 and the Nikkei data end in 2024 (May and April). The Nasdaq-100 runs to 2025.
- Missing months count as zero in the portfolio sums.

## What this means

- **"Catch longer trades with more pips" fixes the cost problem, but there is no direction edge to collect.**
  - On currencies, 2008-2018 trend following was flat before costs, and 2019-2025 was against it.
  - This matches Kurth, Eisler, Rej & Bouchaud (2026), who find that trend in currencies "has effectively
    vanished" since 2008/9.
  - Trailing stops, channel exits, breakeven stops and partial profits did not change that. On these markets
    they hurt more strategies than they helped.
- **On gold and the indices, a long-trend system is a costly way to be long.** Simply holding them did better.
  It is still drift, which pays only while those markets keep rising.
- **The one survivor goes the other way: weekly dip-buying in up-trends.** It worked only after 2018. It is worth
  watching, but it is not yet an edge.
