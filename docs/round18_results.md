# Round 18 results: what the full review found untested, and the regime question asked two new ways (2026-11-13)

Rules: `docs/round18_preregistration.md` and its addendum 1, each committed before the run it governs. The full
inventory of the project is `docs/inventory.md`.
- **Code:** `scripts/r18_m1.py`, `scripts/r18_regime0.py`, `scripts/r18_long.py`, `scripts/r18_combine.py`,
  `scripts/r18x.py`.
- **Runs:** 43027200926 (round 18) and 43394469570 (addendum 1). Outputs are on branches `round18` and `round18x`;
  tables are in `docs/round18/`.

## The answer

**Nothing counts: 0 of 18 pre-registered tests.** But this round explains why regime switching has not worked, and
it changes how the dip-buying forward test should be read.

1. **There are no performance regimes to detect in the daily, weekly and monthly strategies.** Their returns show
   no persistence beyond what a random shuffle of time gives, at any horizon from 3 to 24 months or 5 to 252 days.
   Round 16's "persistence" (rank correlation 0.05) is fully explained by lasting cost differences between
   strategies: the shuffled data give 0.042.
2. **The intraday strategies do have persistent states, but they are cost states.** Random-signal strategies show
   the same persistence, and even in the best state the intraday strategies lose (-0.26 volatility units a day; no
   strategy is significantly profitable there).
3. **On 50 years of data, FX regimes are eras.**
   - Trend-following and carry paid from 1976 to about 2005 and have paid nothing since.
   - Weekly reversal lost heavily while trends paid.
   - A walk-forward detector did not beat simply holding what had worked (t -0.9 against that benchmark). The
     switch happened once in 50 years, so it cannot be learned in advance.
4. **Weekly dip-buying has no edge over 50 years** (Sharpe -0.15).

   | Period | Sharpe |
   |---|---|
   | 1976-1985 | -0.74 |
   | 1986-1995 | -0.36 |
   | 1996-2005 | +0.62 |
   | 2006-2015 | -0.15 |
   | 2016-2025 | +0.34 |

   Its recent good years look like a favourable stretch, as in 1996-2005. They do not look like a permanent edge.
   The forward test should be read with that in mind.
5. **The published effects re-run on 2019-2025 are still there before costs and still lose after them.** Gold in
   Asian hours: +2.7 bp a night before costs (t 2.5), -4.5 bp after. In 2019-2025 gold rose just as much in Western
   hours, so the "Asian hours" anomaly is gone: it was the gold bull market.
6. **Trading index CFDs with the VWAP trend rule makes money before costs** (NAS100 +12.5 bp a day, t 4.7) **and
   loses after them**, even when it trades only once an hour.

## 1. Part R0: is there anything for a regime detector to find?

**The test.** For each strategy, VR(K) is the variance of K-period sums of its scaled return, divided by K times
the one-period variance. It is about 1 without persistent states and rises with K with them. The null shuffles the
time order (the same shuffle for every strategy), which keeps averages and cross-correlations and removes
persistence.

| Library | Strategies | Median VR at the horizons tested | Null 95th percentile | p |
|---|---|---|---|---|
| Monthly, D1 / W1 / MN units (round 16) | 1,944 | 0.98, 0.87, 0.86, 0.75 at 3, 6, 12, 24 months | 1.04, 1.05, 1.03, 0.96 | 0.60, 0.98, 0.73, 0.70 |
| Monthly placebo units | 300 | 1.01, 0.97, 0.92, 0.87 | 1.01, 1.00, 0.97, 0.88 | 0.02, 0.36, 0.54, 0.11 |
| Monthly, nine styles | 9 | 0.87, 0.67, 0.64, 0.69 | 1.09-1.22 | 0.73-1.00 |
| Daily, all units (round 17) | 2,713 | 0.97, 1.02, 1.00, 0.95, 0.96 at 5, 21, 63, 126, 252 days | 1.01, 1.02, 1.03, 1.03, 1.00 | 1.00, 0.08, 0.32, 0.55, 0.21 |
| Daily placebo units | 380 | 0.97, 0.96, 0.98, 0.99, 0.97 | 1.00, 1.00, 1.00, 0.99, 0.97 | 1.00, 1.00, 0.42, 0.06, 0.03 |
| Daily, nine styles | 9 | 0.98, 0.93, 0.91, 0.87, 0.81 | 1.03-1.18 | 0.76-0.94 |

**Verdict by the pre-registered rule: there are no states to detect.** No real library is above its null at two
horizons.

**Rank persistence** (trailing 12-month mean against the next month): 0.046 in the real library against 0.042 in
the shuffled data (p 0.39). The lasting differences between strategies are their costs.

**Exploratory split by engine (not in the pre-registered text).**

| Daily library | Strategies | Median VR at 21, 63, 126, 252 days | p |
|---|---|---|---|
| D1 / W1 / MN units | 1,954 | 1.00, 0.97, 0.97, 0.91 | 0.43-0.63 |
| Intraday units (H4 to M15) | 814 | **1.23, 1.68, 2.28, 2.96** | 0.003 |
| Intraday placebo units (random signals) | 80 | **1.37, 1.92, 2.64, 3.34** | 0.005 |
| Intraday units minus the placebo units' common daily return | 814 | 1.04, 1.18, 1.29, 1.41 | 0.025-0.005 |

**What the intraday persistence is.**
- **It is the cost regime.** Random-signal strategies have more of it than the real ones. Most of it goes when
  their common return is removed.
- **It never turns into profit.**
  - The intraday units' average scaled return is negative in every year (-0.17 to -0.51) and on 99.98% of days.
  - Sorting days by its trailing 63-day mean, the next day's average runs from -0.45 in the worst fifth to -0.26
    in the best.
  - In the best fifth, 2% of units have a positive mean and none has t >= 1.645.
- **So a detector for intraday strategies can say "they lose less now". It cannot say "they win now".**

## 2. Part L: regimes on a long history (1976-2025)

**Data.** Bank of England daily rates (12,915 days, 28 crosses of 8 currencies) and OECD 3-month rates. Costs are
1.5 bp per side (today's level), so early results are upper bounds.

**L1, the styles by decade (Sharpe ratio).**

| Period | Carry | Trend (12-month) | Weekly reversal | Weekly dip-buying |
|---|---|---|---|---|
| 1976-1985 | 0.27 | **0.89** | -1.85 | -0.74 |
| 1986-1995 | 0.62 | 0.40 | -0.42 | -0.36 |
| 1996-2005 | **1.40** | 0.70 | -0.45 | **0.62** |
| 2006-2015 | -0.13 | 0.01 | 0.16 | -0.15 |
| 2016-2025 | 0.07 | -0.19 | -0.01 | 0.34 |
| 1976-2025 | 0.42 (t 2.8) | 0.41 (t 2.9) | -0.60 (t -4.1) | -0.15 (t -1.0) |

- **Trend and carry were real for 30 years and stopped around 2006.** Round 13's finding that FX trend loses before
  costs after 2019 is the end of that era, not a testing error.
- **Dip-buying is not the mirror image of trend.** Their monthly correlation is +0.09. It had good and bad decades
  of its own.

**L2, the regime detector.** Monthly, choosing among the four styles and cash. Settings were chosen on 1999-2025
and verified once on 1981-1998, which had never been examined.

| Family | Chosen on 1999-2025 | t 1999-2025 | t 1981-1998 | minus static, t | Counts |
|---|---|---|---|---|---|
| B1 static (benchmark) | expanding mean, best 2 | 1.50 | 3.34 | - | - |
| M1 momentum | 60 months, best 1 | 0.73 | 0.70 | -2.48 | no |
| M3 one feature | dispersion of interest rates, 3 regimes | 2.33 | 1.60 | -0.95 | no (also below its placebo's 95th percentile: Sharpe 0.39 against 0.69) |
| C1 literature rule | trend only when the forward discount is positive and volatility and dispersion are low | 0.02 | 1.49 | -2.87 | no (worse than always holding trend, t -2.46; in cash 88% of the time) |

- **In 1981-1998 the styles simply worked,** so "hold what has worked" scored t 3.34. No detector added to it; 3%
  of the 256 single-feature settings beat it.
- **The one switch that mattered,** leaving trend and carry around 2006, came once. A 60-month momentum rule
  adapts to it, but too slowly to help.

**L3, global FX volatility and the dollar** (long the foreign basket after high volatility, short after low).

| Period | t |
|---|---|
| 1981-1998 | 0.96 |
| 1999-2025 | 0.83 |
| Pooled | 1.25 |

It is positive in both halves and not significant. It does not count.

## 3. Part G: the published effects, second sample

Net is after the quoted spread, slippage and commission. The first period is 2007-2018 for FX and 2011-2018 for
metals.

| Effect | First period: gross bp (t) | 2019-2025: gross bp (t) | 2019-2025: net bp (t) | Verdict |
|---|---|---|---|---|
| Gold, Asian hours (18:00 New York to 08:00 London) | +2.39 (2.2) | **+2.70 (2.5)** | -4.45 (-4.1) | cost-bound; no anomaly |
| The same without Sunday-evening entries | +3.56 (3.2) | **+3.71 (3.3)** | -3.46 (-3.0) | cost-bound; no anomaly |
| Gold, Western hours, short | +2.61 (1.2) | -3.11 (-1.5) | -7.18 (-3.4) | reversed |
| Gold AM fix, short | +1.05 (3.6) | +0.78 (2.9) | -2.82 (-10.3) | cost-bound |
| Gold PM fix, short | +0.24 (0.4) | +0.17 (0.3) | -3.49 (-5.4) | nothing |
| Silver fix, short | +3.64 (5.1) | +1.81 (2.8) | -17.65 (-26.6) | cost-bound |
| Sell each currency in its home hours | +0.27 (0.6) | -0.35 (-0.7) | -3.29 (-7.0) | nothing |
| Around the clock, both legs | +0.51 (0.6) | +0.62 (0.7) | -8.86 (-10.1) | nothing |
| Fade the move after the 4 p.m. fix | +0.18 (1.5) | +0.47 (3.5) | -1.74 (-12.9) | cost-bound |

**Gold in Asian hours, in detail (`docs/round18/gold_costs.json`).**
- **The gross edge is steady:** about +2.5 to +3.7 bp a night in both periods.
- **The anomaly is gone.** In 2011-2018 gold rose in Asian hours and fell in Western hours (difference t 2.0). In
  2019-2025 it rose in both: +2.7 bp against +3.1 bp (difference t -0.2). Buy-and-hold made +6.4 bp a day. The
  Asian-hours long is now gold exposure.
- **Costs did not fall with the higher gold price.** The median quoted spread went from 2.2 bp to 1.8 bp. At the
  18:00 reopen it is about 6 bp.
- **Entering later does not rescue it (exploratory).**

  | Entry (New York) | Round-trip cost | Net 2019-2025 |
  |---|---|---|
  | 18:00 | 7.2 bp | -4.5 bp |
  | 19:00 | 4.0 bp | -1.9 bp |
  | 20:00 | 3.7 bp | -2.2 bp |

- **The outside study** reporting +5.5 bp a night after costs assumed a 0.20 USD spread. That is far below the
  Dukascopy quotes used here.

## 4. Part C: the two new published effects, and the VWAP follow-up

| Test | First period: net bp (gross, t) | 2019-2025: net bp (t) | 2019-2025: gross bp (t) | Verdict |
|---|---|---|---|---|
| C1 overnight-to-intraday reversal, 7 currencies | -2.51 (gross +1.25, t 1.9) | -2.71 (-5.3) | +0.18 (0.4) | decayed, and cost-bound before that |
| C2 VWAP trend, US500, any minute | -63.9 (gross +6.3, t 4.0) | -43.4 (-14.8) | +6.4 (2.6) | cost-bound (38 units traded a day) |
| C2 VWAP trend, NAS100, any minute | -95.7 (gross +4.3, t 2.1) | -29.9 (-9.6) | +12.5 (4.7) | cost-bound |
| C2b US500, chosen: every 60 minutes from 10:00 | -5.35 (gross +3.6) | -2.36 (-1.0) | +4.1 (1.8) | fails |
| C2b NAS100, chosen: every 60 minutes from 09:30 | -10.48 (gross +3.9) | -3.50 (-1.2) | +3.0 (1.0) | fails |

- **The intraday index trend is real before costs** in both periods and both indices. It costs 1.6-1.8 bp per unit
  traded on these CFDs, and even hourly decisions trade about 5 units a day.
- **The best setting on 2019-2025** (NAS100, hourly from 10:00) made +1.6 bp a day (t 0.6). It was not the one
  chosen on 2012-2018, and it is not significant.
- **At futures-like costs** (about 0.3 bp per unit) the best hourly NAS100 setting would keep most of its +7 bp a
  day gross. Like the opening-range breakout, this is a futures-account effect; it does not work at CFD costs.

## 5. The family

18 tests with one BH correction: none passes. The smallest p-values are 0.055 (L2, one feature), 0.069 (L2,
literature rule) and 0.105 (L3). The first threshold is 0.0056.

## What this means for a regime detector

**The four ways a detector could have worked, and what the data say.**

| Failure class of the strategy | What a detector would need | What we found |
|---|---|---|
| No gross edge (87% of the round-15 library) | good and bad states that persist | none: returns are as persistent as shuffled data (R0) |
| Cost-bound (intraday rules, fixes, gold hours, index trend) | states where the move beats the cost | cost states exist and persist, but the best state still loses (R0x); costs have not fallen (G) |
| Decayed (trend, carry, overnight reversal) | a signal that the edge is alive again | eras last decades; the switch came once (L1, L2) |
| Period-dependent (dip-buying) | a variable that marks its good periods | none of 33 features did (round 16); 50 years show alternating decades and a negative average |

**What can be implemented, and what it is worth.**
- **A health monitor.** For any live strategy: stop when its record is no longer consistent with its tested edge.
  The forward tests already have such rules. Tested as selection rules (round 16's performance filter, round 17's
  change-point resets, and M1 here), they do not add return; they only limit the damage.
- **A regime switcher that picks the strategy for the current market state.** This was tested in 114
  pre-registered tests in rounds 16 to 18, on 18 and on 50 years, monthly to daily, for the whole book and per
  instrument. It has never beaten
  holding the best past strategy. The reason is now clear: there are no persistent performance states to detect.

**What changes for the forward tests.**
- **Gotobi:** unchanged. It is a forced flow at a known time; it is not a pattern that comes and goes.
- **Weekly dip-buying:** keep the forward test, but treat it as an era bet. Its 2019-2026 record resembles
  1996-2005, which was followed by a losing decade. It should not be sized as if the edge were permanent.

Not financial advice.
