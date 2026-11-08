# Round 13: long trades that let winners run (pre-registration, 2026-11-08)

Written before any data of this round is read. The question (the user's): every strategy so far lost about its cost
on each trade, so would trades that are held for weeks or months, and are allowed to catch big moves, make the cost
small enough not to matter? Earlier rounds capped holds at 48 bars and never let a stop trail behind a winner on
daily or weekly bars across a broad portfolio. 2026 is not used.

**Prior (stated before the run).**
- Kurth, Eisler, Rej & Bouchaud (2026) find that trend following has "effectively vanished" in currencies since
  2008/9.
- Here, 12-month cross-sectional FX momentum earned t 0.14 (run 20), and random entries with trailing stops lost
  before costs (`docs/conclusions.md` 8.1i).
- Gold and equity indices trended up strongly in 2019-2025, so any system that is mostly long them will look good
  there. The rules below separate that drift from a trend edge.
- **Power:** 7 years of monthly returns reach t 1.645 only at a Sharpe ratio of about 0.62. A real but modest trend
  edge (Sharpe 0.3) would most likely not be detected.

## Data, instruments, periods

- M1 bid / ask store (staged run 41645097549), 2007-2025.
- **FX:** the 28 crosses of USD, EUR, GBP, JPY, CHF, CAD, AUD and NZD (point-in-time 3-month rates exist for all
  eight currencies).
- **Non-FX:** XAUUSD, XAGUSD, USA500IDXUSD, USATECHIDXUSD, JPNIDXJPY.
- USDCNH, USDMXN, USDSGD and USDZAR are left out: there are no rate data for them, so the carry their long holds
  would earn or pay cannot be priced.
- **Signal bars:** D1, from M1, with days rolling at 17:00 New York (`fxlab.data.store.resample`). W1 bars are the
  D1 bars of a Monday-Friday week.
- **Fit period:** 2008-2018 (2007 is warm-up). **Check period:** 2019-2025. Daily P&L is assigned to the period of
  its date, so a trade open across the boundary contributes to both.

## Execution and costs (every trade on the M1 path)

- **Market orders.** Orders that follow a daily signal are filled at the first M1 bar at or after 08:00 London on
  the next trading day (within 30 minutes; otherwise the next trading day). That is the most liquid hour that
  follows the close. A buy fills at ask + slippage, a sell at bid - slippage. Slippage is 0.25 x the quoted spread
  + 0.1 pip. Commission is 0.35 bp a side.
- **Resting orders.** Stop entries and stop exits rest from the day's open (17:00 New York) at levels computed from
  completed bars only.
  - A stop fills at the level, or at the bar's open if the price gapped through, plus slippage.
  - A long stop and a long stop-entry are judged on the bid and ask paths respectively; shorts mirror this.
  - A trade entered in an M1 bar is first checked for its stop in the next M1 bar.
  - Limits (targets) fill at the level without slippage, on the bid for a long and on the ask for a short.
  - When a stop and a target fall in the same M1 bar, the stop fills first.
- **Swap**, charged for every 17:00 New York rollover held (Wednesday x3):
  - FX: (direction x (base rate - quote rate) - 0.75% mark-up) / 365 a night (`fxlab.data.rates`).
  - Metals and indices: the quote currency's rate (USD; JPY for JPNIDXJPY) is paid by a long and earned by a short,
    with a 2.5% a year mark-up on either side.
  - Index dividends are not credited. This is conservative for longs.
- **Sizing.**
  - Stop-based systems risk a fixed fraction of equity per unit at the initial stop, in notional = risk /
    (stop distance / entry price). Systems without an initial stop use a nominal distance of 2N.
  - Volatility-target systems hold notional = forecast x target vol / instrument vol.
  - Each instrument gets the same risk budget. A portfolio's daily return is the sum over its instruments, marked
    to the mid at each 17:00 New York close.
  - For the tables, each portfolio is scaled to 10% annual volatility using the volatility of its own fit-period
    returns. The scale is fixed on 2008-2018 and does not change t statistics.
- **N** = the Turtle 20-day average true range: N_t = (19 N_{t-1} + TR_t) / 20.

## Part A. Long-trend systems as published (fixed parameters)

| ID | System | Rules |
|---|---|---|
| L1 | Turtle System 2 | Stop-entry at the prior 55-day high / low. Initial stop 2N. Up to 4 units added every 0.5N in the trade's favour, with all stops moved to 2N from the latest unit. Exit at the prior 20-day low / high (resting stop) |
| L2 | Turtle System 1 (without the skip-after-winner filter) | The same rules with 20-day entry and 10-day exit levels |
| L3 | Clenow core trend | Long only when the 50-day close average is above the 100-day average (short mirror). Enter at market after a close at a 50-day closing high (low). Exit on a 3N trailing stop below the highest close since entry. No pyramiding |
| L4 | Donchian 100 / 50 | Stop-entry at the prior 100-day high / low. Exit at the prior 50-day low / high. No other stop |
| L5 | Moving-average 50 / 200, always in | Long when the 50-day close average is above the 200-day average, short below. Reverse at market after a cross |
| L6 | Time-series momentum, 12 months (Moskowitz, Ooi & Pedersen 2012) | At each month-end close: the sign of the past 252-day return. Notional = sign x 40% / annualised EWMA volatility (centre of mass 60 days). Rebalanced (direction and size) monthly at market |
| L7 | Time-series momentum, 1 / 3 / 12-month blend | Average of the three signs. Otherwise as L6 |
| L8 | Faber 10-month average | At each month-end: long if the close is above the 10-month average of month-end closes, short if below. Sized as L6, monthly |
| L9 | EWMAC (Carver, Systematic Trading) | Four crossovers of exponential averages of the close: 8 / 32, 16 / 64, 32 / 128, 64 / 256 days. Each is divided by the 36-day EWMA of daily price changes and multiplied by its published scalar (5.3, 3.75, 2.65, 1.87). Each is capped at +-20; the average is multiplied by 1.1 and capped at +-20. Notional = forecast / 10 x 20% / instrument vol. Traded daily at market only when the change exceeds 10% of the average absolute notional |
| L10 | Squeeze release with a trailing stop | The Bollinger band (20, 2 sd) inside the Keltner channel (20-day average +- 1.5 x ATR(20)) on the previous day and not today: enter at market in the direction of close minus its 20-day average. Initial stop 2N, trailing stop 3N below the highest close (above the lowest) since entry. No target, no time limit |
| L11 | Weekly breakout | A weekly close above the prior 20-week high (below the low) gives a long (short) at the next market fill. Exit at market after a weekly close below the prior 10-week low (above the high), with a resting catastrophe stop of 2 weekly N |
| L12 | EWMAC with a carry filter (FX only) | The L9 forecast is set to 0 when its sign disagrees with the sign of the point-in-time rate differential |

**Controls** (not candidates):
- **PL1 coin-flip trailing (Tharp / Basso):** random entries at the average rate of L1's entries, random side, one
  position per instrument, initial stop 2N, 3N trailing stop.
- **PL2 random-sign time-series momentum:** L6 with a random sign each month.
- **BH long-only benchmark:** long every instrument, sized and rebalanced monthly as L6. It is the drift a long-biased
  system could collect without timing anything.

**Rows.** Each system gives two primary rows:
- **FX:** the 28 crosses, equal risk.
- **Non-FX:** gold, silver and the three indices, equal risk. This row counts only through its excess over BH.

That is 23 rows (L12 has no non-FX row).

**A row counts** if all of these hold:
- its check-period mean monthly return is > 0 with Newey-West (3 lags) t >= 1.645;
- its fit-period mean monthly return is > 0;
- it survives Benjamini-Hochberg at q 0.10 across the 23 rows;
- for a non-FX row, the monthly return of the system minus BH (scaled alike) is also > 0 with t >= 1.645 in the
  check period.

If PL1 or PL2 passes the same test, no row of Part A is credited.

Reported with every row:
- trades;
- median hold (days);
- win rate;
- average win and loss in pips and in R (initial risk);
- payoff ratio;
- share of gross P&L from the best 10% of trades;
- costs and swap as a share of the gross move;
- Sharpe and maximum drawdown at 10% volatility;
- the yearly returns.

## Part B. The catalog with exits that let winners run

**Units.** Every single-instrument catalog strategy allowed at D1 (the families excluded in round 12 stay
excluded), run on D1 bars and on W1 bars, over the 33 instruments. A strategy's long / short signal on a bar's close
gives a market order at the next fill. Trade-mode strategies keep their own entries.

**Exit variants:**
- **T0:** as the strategy (signal exit; trade-mode strategies keep their own stop, target and hold).
- **T1:** initial stop 2N, then a 3N trailing stop on closes; no target, no time limit; the strategy's own exit
  signal is ignored.
- **T2:** initial stop 2N; exit at the prior 20-bar opposite extreme (resting); no time limit.
- **T3:** initial stop 2N; stop to breakeven once the trade is 1R (2N) ahead; then a 3N trailing stop.
- **T4:** half closed at 2R; the stop then moves to breakeven; the rest runs on a 3N trailing stop.

W1 units use the weekly N, and their trailing stops move only at weekly closes. When flat, a new trade needs a
fresh signal (the side changes to long or short). A position is never added to in part B.

**Selection.** Each unit's variant is the one with the highest fit-period Newey-West t of its 33-instrument portfolio's
monthly returns. It is then judged once on 2019-2025.

**A unit counts** if its check mean monthly return is > 0 with t >= 1.645 and it survives Benjamini-Hochberg at
q 0.10 across all part-B units.

**Placebo calibration.** Random-entry units at three entry rates (one every 10, 20 and 40 trading days; random side),
on D1 and W1, go through the same five exits and the same selection. T0 for these is a fixed hold of 10 bars. If the
placebos pass as often as the strategies, no unit is credited.

## What is not done

- No parameter is tuned in this round.
- No portfolio-level heat limits (the Turtles' unit caps across correlated markets) are applied.
- Prop-firm rules are applied only to a row or unit that counts.
