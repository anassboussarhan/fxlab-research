# Round 21 pre-registration: the verified gaps across markets (2026-11-15)

Written before any round-21 data was read. Each idea below was checked against the project's documents and found
untested (see the reply of 2026-11-15). Development data ends 2025-12-31.

**Common rules:**
- **Counting:** a test counts if:
  - P1 mean net > 0;
  - P2 mean net > 0 with t ≥ 1.645;
  - it passes Benjamini-Hochberg at q = 0.10 over the 23 tests of this round.
- **2026:** used once, only for a test that counts.
- **Fills:** at mid plus a per-side cost. Intraday CFD cost per side is half the quoted spread, plus the
  project's slippage (`BASE.slip_spread_frac` × spread + `slip_fixed_pips`), plus commission. Limit fills pay the
  same cost (conservative).
- **Futures costs** (indices): ES 0.30 and NQ 0.375 index points per side, as round 19.
- **Same-bar rule:** a stop and a target in the same bar count as the stop.

## Part I. ICT on the indices (NAS100, US500 minute mids; P1 2012-2018, P2 2019-2025; primary cost: futures)

| ID | Rule |
|---|---|
| I1 Silver Bullet | - **Signal:** the first fair value gap formed between 10:00 and 10:59 New York on 1-minute bars. Bullish: low[k] > high[k−2]; bearish: high[k] < low[k−2]. The gap must be at least half the median 1-minute range of 09:30-09:59.<br>- **Entry:** limit at the gap's near edge (low[k] for bullish, high[k] for bearish), filled if touched by 11:00.<br>- **Stop:** low[k−2] (high[k−2] for bearish).<br>- **Target:** 2R.<br>- **Exit:** flat at 12:00. One trade a day. |
| I2 Judas swing | - **Condition:** between 09:30 and 09:59 price trades above both the 09:30 open and the midnight (00:00 New York) open, but the 09:59 close is back below the 09:30 open.<br>- **Entry:** short at the 10:00 open; mirror for longs.<br>- **Stop:** the 09:30-09:59 extreme.<br>- **Exit:** flat at 16:00. |
| I3 Midnight-open bias (power of three) | - **Bias:** the prior day's 09:30 → 16:00 direction.<br>- **Entry:** bullish bias and the 09:30 price below the midnight open → long; bearish bias and the 09:30 price above it → short.<br>- **Exit:** 09:30 to 16:00. |

Each rule runs on US500 and NAS100 separately: 6 tests.

## Part X. Other ICT pieces never coded

| ID | Market | Rule |
|---|---|---|
| X1a SMT (FX) | EUR/USD and GBP/USD, H1, 2007-2025 | - **Bearish signal:** one pair's bar makes a new 24-bar high and the other's does not. Short the pair that made the high from the next bar's open for 12 bars.<br>- **Bullish:** mirror at lows.<br>- **Periods:** P1 2007-2018, P2 2019-2025. |
| X1b SMT (indices) | US500 and NAS100, 5-minute bars, 09:30-15:55 | - **Signal:** a new 12-bar high (low) in one index but not the other. Short (long) the one that made it for 12 bars, or until 16:00.<br>- **Cost:** futures. |
| X2 NDOG fill | US500 and NAS100 | - **Signal:** the 18:00 New York reopen differs from the 16:59 close by ≥ 0.05%.<br>- **Trade:** toward the close, exiting at the fill or at 09:30 the next morning.<br>- **Cost:** futures. Both indices pooled. |
| X3 OTE | EUR/USD, GBP/USD, USD/JPY, AUD/USD, XAU/USD, H1 | - **Displacement:** a bar whose range is ≥ 2 × ATR(20) and whose close lies in the outer 25% of the range.<br>- **Leg:** from the extreme of the 10 bars up to it, to the bar's extreme.<br>- **Entry:** within 10 bars, a limit at the 70.5% retracement (the OTE zone 62-79%).<br>- **Stop:** the leg start. **Target:** the leg end (about 2.4R). Cancelled if the leg end is passed before the fill.<br>- **Exit:** at most 48 bars. Pooled. |
| X4 CBDR | EUR/USD, GBP/USD, AUD/USD, USD/JPY, USD/CAD | - **Range:** the 14:00-20:00 New York range, used only if ≤ 40% of the 20-day average daily range.<br>- **Next 00:00-12:00 New York:** short limit at range high + 2 × range, long limit at range low − 2 × range.<br>- **Stop:** 4 × range beyond. **Target:** the range midpoint.<br>- **Exit:** flat at 12:00. Pooled. |
| X5 Breaker | EUR/USD, GBP/USD, USD/JPY, AUD/USD, XAU/USD, H1 | - **Swings:** 3-bar fractals, confirmed 3 bars later.<br>- **Bearish setup:** a swing high above the previous swing high (a sweep), then a close below the swing low before it (a break of structure).<br>- **Breaker:** that swing-low bar's range.<br>- **Entry:** within 24 bars, a limit short at the breaker's low.<br>- **Stop:** the breaker high + 0.5 ATR(20). **Target:** 2R.<br>- **Exit:** at most 48 bars. Mirror for bullish. Pooled. |

FX periods: P1 2007-2018, P2 2019-2025 (gold 2011-2018 only where the store has no later data).

## Part M. Monthly currency signals (8 currencies, Bank of England spot rates, OECD rates via DBnomics)

- **Return:** the spot change against the other currencies plus the short-rate differential, less 3 bp per side on
  turnover.
- **Book:** each month, long the top 3 and short the bottom 3, equal weight.

| ID | Rule |
|---|---|
| M1 Real-rate carry | Rank by the 3-month rate minus the year-on-year CPI inflation. Rates lagged 1 month, CPI lagged 2 (publication). |
| M2 Curve slope | Rank by the 10-year yield minus the 3-month rate (lagged 1 month). Long the flattest, short the steepest (Lustig, Stathopoulos & Verdelhan 2019: flat-curve currencies earn the carry premium). |

- **Periods:** P1 1985-2004, P2 2005-2025.
- **Also reported:** nominal carry on the same book.

## Part S. Single-stock anomalies on point-in-time S&P 500 stocks

- **Data:** the round-20 panel (run 43624416126, Yahoo daily, membership `fja05680/sp500`).
- **Portfolios:** monthly at the month-end close. Long the top decile, measured against the equal-weight eligible
  universe.
- **Costs:** 5 bp per side on turnover.
- **Periods:** P1 2005-2014, P2 2015-2025; t from monthly excess returns.

| ID | Rule |
|---|---|
| S1 Momentum 12-1 | Top decile of the return from t−252 to t−21 days |
| S2 Short-term reversal | Bottom decile of the past 21-day return |
| S3 52-week high | Top decile of close / 252-day high |
| S4 Low volatility | Bottom decile of 252-day daily volatility |
| S5 Overnight momentum | Top decile of the past 21 days' summed overnight (close → open) returns (Lou, Polk & Skouras 2019) |
| S6 Deletion rebound | - **Entry:** buy at the close of the first day a stock is out of the index.<br>- **Hold:** 60 trading days.<br>- **Measure:** against SPY; t clustered by month. |
| S7 Pairs (Gatev, Goetzmann & Rouwenhorst) | - **Formation:** each month, 12 months of normalised prices; the 20 closest pairs among members.<br>- **Trading:** the next 6 months; open at 2 standard deviations of the spread, close at the cross.<br>- **Return:** monthly return on committed capital, both legs paying costs. Six overlapping books averaged. |

## Part K. Crypto across coins (Binance spot, USDT pairs, daily)

- **Universe:** every USDT pair in Binance's public daily files, including delisted ones. Stablecoins,
  fiat-pegged and leveraged tokens are excluded.
- **Eligibility:** each week, the 50 coins with the highest 30-day quote volume and at least 60 days of history.
- **Rebalancing:** weekly, Sunday 00:00 UTC.
- **Portfolio:** long the top quintile, short the bottom quintile (the shorts through perpetuals), equal weight.
- **Costs:** 10 bp per side on turnover; funding not charged (reported separately as a caveat).
- **Periods:** P1 2019-2021, P2 2022-2025.

| ID | Rule |
|---|---|
| K1 Momentum | Rank by the past 28-day return |
| K2 Reversal | Long the past 7-day losers, short the winners |

**Note (2026-11-15, after the first run):** X2 found no 16:59 and 18:00 minute bars after 2018 in the CFD
files. It is rerun with the last bar from 16:30 to 16:59 and the first bar from 18:00 to 18:29, the same rule
with a tolerance. Nothing else changes.

## Not in this round

- Net foreign assets (annual, 9 currencies; too little data).
- PEAD, commodity curve carry, order flow, 0DTE (paid data).
- Warrior-style small caps (waiting for data).

Not financial advice.
