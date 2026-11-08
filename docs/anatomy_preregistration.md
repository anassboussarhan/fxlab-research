# Round 12: the queued classic systems, and why every failed strategy failed (pre-registration, 2026-11-08)

Written before any data of this round is read. Two parts: (A) the nine untested rule families (MS1, QM1, CL1-CL7)
as published; (B) a failure anatomy of every single-instrument strategy family in the catalog plus (A), with one
pre-registered search for a fix per strategy, chosen on 2007-2018 and judged once on 2019-2025. 2026 is not used.

## Common setting

- Instruments: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, XAU/USD (M1 bid / ask store, 2007-2025; gold
  from 2011). Bars H1, H4, D1 built from M1 (`fxlab.data.store.resample`, D1 at the 17:00 New York rollover).
- A strategy unit = one strategy at one timeframe, pooled over the seven instruments.
- Trades: entry at the open of the bar after the signal bar; state-mode strategies exit at the open of the bar after
  the position changes; trade-mode strategies use their own stop / target / max hold, else the catalog risk default
  (stop k x ATR(14), target a + b x stop, max hold 48 bars; trailing stops are not modelled). Every trade is priced
  on the M1 path with the flows-lab costs (quoted spread, slippage, commission, financing); stop first when both
  levels are touched in one minute.
- Periods: fit 2007-2018, check 2019-2025; t statistics month-clustered.

## A. The nine untested families (rules as published; default parameters only)

| ID | Rule | Timeframes |
|---|---|---|
| MS1 Malaysian SNR | Line-chart (close) turning points of 3 bars each side: A-level = the turning close at a peak, V-level at a valley. Fresh until a later bar's wick touches it. Storyline = sign of the D1 close vs its 20-day mean. Long at a fresh V-level (limit at the level) when the storyline is up, short at a fresh A-level when it is down; stop 0.5 ATR beyond the level, target the nearest fresh opposite level (at least 1R, else 2R), max hold 48 bars | H1, H4 |
| QM1 Quasimodo | Zigzag (3-bar swing points): bearish QM = high H1, low L1, higher high H2 > H1, then a lower low L2 < L1; sell limit at H1 (the left shoulder), stop 0.2 ATR above H2, target 2R; mirror for bullish; expires after 48 bars | H1, H4 |
| CL1 Dual Thrust | Range = max(HH - LC, HC - LL) of the last 4 D1 bars; long when price crosses day open + 0.5 x range, short below open - 0.5 x range, always in, reversing | H1 (D1 levels) |
| CL2a R-Breaker | From yesterday's H, L, C (pivot form): P = (H + L + C) / 3; break-buy H + 2(P - L), break-sell L - 2(H - P); sell-setup P + (H - L), sell-enter 2P - L, buy-enter 2P - H, buy-setup P - (H - L). Close above break-buy -> long, below break-sell -> short; after the day's high exceeded sell-setup, a close below sell-enter -> short; after the day's low fell below buy-setup, a close above buy-enter -> long; flat at the day's end | H1 |
| CL2b Fiali four prices | Long above yesterday's high, short below yesterday's low, first break of the day only, flat at the day's end | H1 |
| CL3a Aberration | Close above the 35-bar Bollinger band (2 sd) -> long, below the lower band -> short, exit at the middle band | H4, D1 |
| CL3b King Keltner | 40-bar average of the typical price +- 40-bar average true range; long above the upper band when the average is rising, short below the lower when falling, exit at the average | H4, D1 |
| CL4a Thermostat | Choppiness = abs(close - close 30 bars ago) / (highest high - lowest low, 30); < 0.2 (range): buy at the 3-bar average low band / sell at the high band, exit at the opposite band; >= 0.2 (trend): 50-bar Bollinger-type breakout (0.5 x 30-bar range from the open), exit at the 50-bar average | H4, D1 |
| CL4b Bollinger Bandit | 50-bar Bollinger band (1.25 sd); long above the upper band when close > close 30 bars ago, short mirror; exit at a moving average shortening from 50 bars by one each bar in the trade (minimum 10) | H4, D1 |
| CL5 Camarilla | From yesterday's H, L, C: H3 = C + 1.1(H - L)/4, H4 = C + 1.1(H - L)/2 (L3 / L4 mirror); fade at H3 / L3 (stop beyond H4 / L4, target C), breakout beyond H4 / L4 (stop at H3 / L3, target 1R), flat at the day's end | H1 |
| CL6 Elder triple screen | Weekly (5 D1 bars) MACD-histogram slope sets the side; daily 2-bar force-index EMA against it (pullback) arms; buy stop one tick above yesterday's high (sell stop below the low), stop at yesterday's low (high), exit after 10 bars or at the stop | D1 |
| CL7 Puria | 75- and 85-period LWMA of the low, 5-period EMA of the close crossing both with MACD on the same side; stop 2 ATR, target 3 ATR | H1, H4 |

Each (family, timeframe) is one primary row. **A row counts** if its pooled mean net per trade is > 0 with t >= 1.645
in 2019-2025, > 0 in 2007-2018, and it survives Benjamini-Hochberg at q 0.10 across the rows of part A.

## B. Failure anatomy and one fix search per strategy

Units: every single-instrument catalog strategy (no panel, ML, grid / martingale sizing or esoteric families) at its
allowed timeframes among H1 / H4 / D1, plus the part-A units, plus five placebo units (random entries at the same
rate as a typical strategy, random side; holds of 1 bar, 6 bars, 24 bars, and two ATR brackets).

**Diagnosis (base version, fit period only):** gross vs net per trade (cost share); win rate, average win, average
loss; MFE and MAE on the mid path (median, in units of the round-trip cost); share of losing trades whose MFE had
exceeded the cost (an exit problem); share of trades that never moved a cost in their favour (an entry problem);
and the difference in mean net between the feature buckets below. Each unit gets one label:
- **no gross edge**: gross t < 1 in the fit period;
- **cost-bound**: gross t >= 1 and gross mean > 0 but net <= 0;
- **decay**: net > 0 in the fit period, <= 0 in the check period;
- **skew trap**: win rate >= 60% and average loss >= 2 x average win;
- otherwise **weak**.

**Fix menu (all pre-entry information):**
- Entry method: E0 market at the bar open; E1 market 15 minutes later; E2 limit 0.25 x ATR(H1) better than the open,
  valid 60 minutes, else no trade; E3 stop entry 0.25 x ATR(H1) in the trade's direction, valid 60 minutes.
- Exit: X0 as the strategy; X1 half the holding time; X2 twice the holding time (same stop / target).
- Direction: as the strategy, or reversed.
- Filter (one at most): keep one bucket, or drop one bucket, of one feature, with tercile cut points from the unit's
  fit-period trades: H1 ATR percentile (250 days), trade direction vs the 20-day return (with / against), 20-day
  trend strength (|return| / volatility), the 24-hour move in the trade's direction in ATR, session (Asia / London /
  New York / late), weekday (Monday / mid-week / Friday), spread at entry vs that hour's median, distance to the
  nearest 00 level in ATR, calendar day (Gotobi or month-end vs other).
- Selection: the combination with the highest fit-period t (trade-level, at least max(30, a quarter of the base
  trades)) is the unit's fix. It is then evaluated once on 2019-2025.

**A fix counts** if its check-period mean net per trade is > 0 with month-clustered t >= 1.645 and it survives BH
at q 0.10 across all non-placebo units of part B. The placebo units measure how often this selection procedure
"passes" by luck; if the placebos pass as often as the strategies, no fix is credited, whatever the BH result.
Parameter changes are not searched here (the catalog's joint optimisation already searched about 762,000
parameter x entry x exit candidates; `docs/conclusions.md` 8.1c).
