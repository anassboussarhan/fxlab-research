# Round 20 pre-registration: the famous stock traders' setups (2026-11-14)

Written before any stock data was read.

**Why this round:** the request was to test traders who claim outstanding gains (Warrior Trading / Ross Cameron
and similar). None of their stock setups had been tested here; `literature_check.md` noted Warrior Trading as
"not our markets".

**What the search found about the claims (sources in the results):**
- **Warrior Trading:** the FTC (2022) found that the vast majority of customer accounts lost money. The company
  paid USD 3 million, and its advertised "USD 101,280 in verified profits in 45 days" was Ross Cameron's own
  account.
- **Retail day traders generally:** in the largest study, 97% of Brazilian futures day traders who persisted
  300 days lost money, and 1.1% earned more than the minimum wage.
- **The traders with documented large gains:** Kristjan Kullamägi, Mark Minervini (US Investing Championship) and
  Ross Cameron all trade US stocks with momentum setups. A star's own record is not the setup's record: the
  question here is whether the setup, run mechanically, has an edge.

**Part S** (this document) uses free daily data. **Part W** (Warrior-style intraday small caps) needs minute data
of delisted stocks and waits for the user's choice of data (Databento credit or Alpaca keys). It will be
pre-registered as an addendum before its data are read.

## Part S. Swing setups on S&P 500 stocks (stock CFDs at prop firms)

### Data and universe

- **Universe:** S&P 500 members, point in time. The membership is rebuilt from Wikipedia's current constituents
  and its table of changes, applied backwards.
- **Prices:** daily OHLCV from Yahoo Finance (yfinance), adjusted for splits and dividends, 2003-2025. SPY is the
  benchmark.
- **Survivorship:** companies that left the index and whose price history Yahoo no longer serves are missing.
  The number missing is reported. This bias favours long momentum setups, so a failure is conclusive and a pass
  needs care.

### Costs

- 5 bp per side;
- long financing at 6% a year on the days held (CFD swap).

### Signals

All signals use data up to the close of day t; entries are on day t+1 or later.

| ID | Rule (fixed) |
|---|---|
| S-QB | **Kullamägi breakout.**<br>- **Setup at t:**<br>&nbsp;&nbsp;- 63-day return in the top 10% of eligible members;<br>&nbsp;&nbsp;- the last 10 days' high ≤ 1.10 × their low (tight range);<br>&nbsp;&nbsp;- 5-day average daily range ≤ 20-day average daily range;<br>&nbsp;&nbsp;- close > 10-day SMA > 20-day SMA.<br>- **Entry:** buy stop at the 10-day high on t+1, filled at max(open, level).<br>- **Stop:** the entry day's low, capped at one 20-day average daily range below entry; active from the next day, exit at min(open, stop).<br>- **Exit:** first close below the 10-day SMA from day 3; at most 60 days. |
| S-EP | **Episodic pivot.**<br>- **Setup:** open gap ≥ 10% over the previous close, and the previous 63-day return ≤ +10% (a neglected stock).<br>- **Entry:** buy at the open.<br>- **Stop:** open − 1 × ATR(20), including on the entry day.<br>- **Exit:** first close below the 20-day SMA; at most 60 days. |
| S-MV | **Minervini trend template with a breakout.**<br>- **Setup at t:**<br>&nbsp;&nbsp;- close > SMA150 and SMA200; SMA150 > SMA200; SMA200 above its value 21 days ago;<br>&nbsp;&nbsp;- SMA50 > SMA150 and SMA200; close > SMA50;<br>&nbsp;&nbsp;- close ≥ 1.30 × 252-day low and ≥ 0.75 × 252-day high;<br>&nbsp;&nbsp;- 252-day return in the top 30%;<br>&nbsp;&nbsp;- the last 10 days' high ≤ 1.08 × their low.<br>- **Entry:** buy stop at the 20-day high on t+1.<br>- **Stop:** 7% below entry.<br>- **Exit:** first close below SMA50; at most 120 days. |

- One position per stock at a time. A member that leaves the index is held until its exit rule fires.

### Statistic and counting

- **Statistic:** each trade's net return minus SPY's return over the same entry-to-exit prices (open or close as
  the trade). Mean excess per trade, with t clustered by exit month.
- **Periods:** P1 2005-2014, P2 2015-2025.
- **Counting rule:** a test counts if:
  - P1 mean excess > 0;
  - P2 mean excess > 0 with t ≥ 1.645;
  - it passes BH at q = 0.10 over the three tests.
- **Also reported:**
  - an equal-weight portfolio of open positions: CAGR, Sharpe and worst drawdown against SPY;
  - win rate and average win / loss;
  - the same rules without the setup filters (base rates: every 10-day-high breakout, every 10% gap).
- **2026:** only for a test that counts, once, and reported as for earlier holdout checks.

Not financial advice.

## Addendum W (2026-11-14, before any data): Warrior-style small-cap gap trading

This part runs only after the user approves the data cost.

### Data

- **Daily bars:** Databento consolidated daily bars (EQUS.SUMMARY, unadjusted, every listed US stock, delisted
  ones included), 2024-07-01 to 2025-12-31. This dataset starts in July 2024, so only 18 months are available.
- **Minute bars:** Nasdaq ITCH minute bars (pre-market from 04:00 New York) for the event stock-days only.
- **Spending:** every request is priced first; the total is capped at USD 9.

### Events (known at the open)

- the open is ≥ 10% above the previous close;
- the previous close is between USD 1 and 20;
- previous-day volume is ≥ 100,000 shares.

These approximate Ross Cameron's "five pillars". Float and news are not available.

### Tests

| ID | Rule |
|---|---|
| W1 gap-and-go (long) | - **Condition:** pre-market volume (04:00-09:29) ≥ 50,000 shares.<br>- **Entry:** the first minute from 09:30 to 10:29 whose high exceeds the pre-market high; buy at max(that minute's open, the pre-market high).<br>- **Stop:** the lowest low of the entry minute and the 5 before it, at most 10% and, if no lower low exists, 5% below entry.<br>- **Target:** 2 × the risk.<br>- **Exit:** the stop or target, otherwise out at 11:00. |
| W2 opening momentum (base rate) | Buy at the open, sell at 10:30. |
| W3 gap-up short (Steven Dux style) | - **Condition:** gap ≥ 50% and open ≥ USD 3.<br>- **Entry:** at 10:30, if the price is at least 10% below the high so far, short.<br>- **Stop:** that high.<br>- **Exit:** cover at 15:55 otherwise. |
| W4 fade (base rate) | Gap ≥ 30% and open ≥ USD 2: short at the open, cover at 15:55. |

### Costs

- **Base:** 0.30% per side (small-cap spread plus slippage).
- **Stress:** 0.60% per side.
- **Shorts:** plus 0.5% borrow for the day.

### Periods and counting

- **Periods:** P1 2024-07 to 2025-03, P2 2025-04 to 2025-12. The t is from daily means.
- **Counting rule:** a test counts if:
  - P1 net mean > 0;
  - P2 net mean > 0 with t ≥ 1.645 at base costs;
  - it passes BH at q = 0.10 over the 4 tests.
- **Caveat:** with 18 months this is a first look, not a decision.
