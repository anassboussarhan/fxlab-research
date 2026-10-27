# The untested batch: every remaining MQL5 and tastytrade idea, tested

The question was what the project had not yet tested from the MQL5 review (`docs/mql5_review.md`), the tastytrade
review (`docs/tastytrade.md`) and the free-data list, with no assumption about whether retail methods lose.

**Method:**
- Three labs, each with its rules written in its docstring before it ran; nothing was tuned after a result.
  - Options mechanics: `scripts/untested_options_lab.py` (branch `untested-options`, run 40401542066).
  - Daily data: `scripts/untested_edges_lab.py` (branch `untested-edges`, run 40400396874).
  - Intraday M1 bid/ask: `scripts/untested_intraday_lab.py` (branch `untested-intraday`, pass 3: run 40498739289).
- Development data only, to 2025-12-31. The 2026 holdout stayed locked.
- Costs:
  - options: fills at mid +- 0.75 half-spread, USD 0.65 a contract;
  - daily: 1-10 bp a side by market (in each docstring);
  - intraday: the quoted bid/ask plus 25% of the spread and 0.1 pip slippage a side, 0.35 bp commission a side, and
    the rate-differential swap.
- Every intraday result is also given gross (mid prices, no costs), so a signal that costs kill is told apart from
  no signal.
- Benjamini-Hochberg runs over all primary tests together (100 so far) (`docs/untested_batch/summary.csv`).

## 1. Headline

**Interim: 100 of the pre-registered tests have run; the Dukascopy-dependent ones (section 5) are still
downloading and this page will be updated.**

- **One new result is strong: the "Gotobi Overnight JPY" rule** (long USD/JPY from 23:00 Tokyo the evening before a
  Gotobi day to 09:53, stop at half the 20-day range): +4.45 bp a trade after all costs, t 3.9, 15 of 19 years,
  +4.18 bp in 2019-2025, against -2.37 bp for the same trade on other days. It is a better form of the project's
  confirmed Gotobi edge, not an independent one.
- **The tastytrade management rules do not help** on clean SPY and DIA quotes: 50% profit-taking and the 21-day
  exit lower the mean on every structure; the 21-day exit's apparent benefit in always-on books is February-April
  2020 alone.
- **Known anomalies reappear on daily data:** TLT into month-end, RSI(2) dip-buying in index ETFs, the overnight
  drift in European and Asian indices. Each has a practical catch (section 4).
- **Many retail rules have real gross edges that the spread removes:** night scalping, candle-pattern fades, the
  gold London breakout, the EURJPY Gotobi trade, US index opening-range breakouts.
- 7 of 100 tests survive Benjamini-Hochberg at q <= 0.10 (section 6).

## 2. Data problems found on the way

1. **Option quotes for XLF, XLE and many single stocks hold impossible prices.**
   - The first run showed a 16-delta XLF strangle losing 25.7% of notional on a 2.4% move, and iron condors losing
     more than their width.
   - A leave-one-out smile check now replaces a mid that sits far from the smile fitted on its neighbours. It
     replaced 1,141 of 103,803 checked XLF quotes, 286 of 101,246 XLE, 159 of 109,393 DIA and 45 of 107,052 SPY.
   - It cannot catch an expiry whose quotes are all wrong on one day. After it:
     - 16% of XLE and 7% of XLF iron condors still open for a negative credit (impossible for a short condor);
     - so do 4% of single-stock condors and 3% of DIA condors; SPY has none.
   - Holding to expiry settles at intrinsic value and is unaffected. Early exits read a quote and are not.
   - **So every early-exit comparison below uses SPY and DIA only.** The pre-registered pooled tests (SPY, DIA,
     XLF, XLE) are reported, flagged as data-limited.
2. **Yahoo's FTSE 100 opens are invalid**: 73% equal the previous close. The FTSE overnight test is dropped.
3. **Two calendar features of the earlier sweeps were wrong.** `F.TS.NFP_WINDOW` and `F.TS.DST_TRANSITION` assumed
   nanosecond timestamps; the M1 bars load with millisecond resolution under pandas 3, so the hours to the next
   payrolls release came out about 1,000 times too large (474 instead of 0.5 an hour before a release).
   - Fixed in `fxlab/features/seasonality.py` (commit 179a5ea).
   - No look-ahead was involved, and two features out of 269 cannot change the sweep's overall result.
4. The BTC turn-of-month series has one event labelled January 2026. It holds only the 31 December 2025 return;
   no 2026 data were read.
5. Dukascopy answered most requests with HTTP 503 from GitHub's runners. The 2019-2025 index, gold, EURJPY,
   AUDUSD and cross-pair subsets came down slowly; coverage is in section 5.

## 3. Options: the tastytrade mechanics

### 3.1 Managing winners and losers (50% profit, 21 days left) on the monthly schedule

One 45-day position a month, the same entries for every rule, paired per cycle, 2019-2025:

| Structure | Underlying | Hold, bp a cycle | 50% profit vs hold | t | 21 days vs hold | t | Both vs hold | t |
|---|---|---|---|---|---|---|---|---|
| 16-delta strangle | SPY + DIA | +31.1 | -19.2 | -3.9 | -53.0 | -2.6 | -50.8 | -2.5 |
| Broken-wing strangle | SPY + DIA | +50.6 | -20.9 | -2.4 | -69.4 | -2.9 | -69.7 | -2.9 |
| 30-delta short put | SPY + DIA | +57.6 | -24.2 | -2.7 | -57.7 | -2.8 | -58.4 | -2.8 |
| Iron condor 16/5 | SPY + DIA | +10.3 | -10.8 | -3.6 | -33.3 | -3.0 | -31.3 | -2.8 |
| Jade lizard | SPY + DIA | +32.0 | -11.2 | -1.3 | -57.9 | -2.4 | -56.9 | -2.3 |

(t from monthly means of the paired differences; 141 cycles, 137 for the lizard, over 74 months. SPY and DIA
separately give the same sign in every cell.)

- **Every management rule lowers the mean, on every structure.** The 21-day exit costs a third to two-thirds of a
  percent of notional a cycle.
- It does cut the worst cycle (strangle: -17.1% held, -12.0% with the 21-day exit), which is the insurance
  tastytrade sells it as.

### 3.2 Always-on books with same-day redeployment

The objection to 3.1 is that a managed position frees its capital early. Here the next position opens at the exit
close (30% of capital in buying power, VIX-scaled in the T3 test):

| SPY book | Hold | 21 days left | t of the difference | Difference without Feb-Apr 2020 | t |
|---|---|---|---|---|---|
| 16-delta strangle | +0.04% a month | +0.21% | +0.76 | -0.08 | -0.66 |
| Broken-wing strangle | +0.25% | +0.40% | +0.57 | -0.11 | -0.61 |
| 30-delta short put | +0.27% | +0.48% | +1.21 | +0.05 | +0.40 |
| Jade lizard | +0.05% | +0.30% | +1.03 | -0.00 | -0.01 |

- **Redeploying makes the 21-day books look better than holding on SPY, but the whole difference is February to
  April 2020.** Without those three months it is at or below zero on every SPY and DIA structure (largest
  +0.05% a month, t 0.4).
- The same pattern showed in the 1994-2018 model layer, where the 21-day rule helped only through 2008: it is
  crash insurance, paid for in normal years.
- **None of the books has a significant alpha after beta** (best alpha t 1.2). The 30-delta put book earns
  +0.48% a month with beta 0.30 to SPY.

### 3.3 Rolling the untested side, VIX sizing, earnings

- **Rolling the untested short toward the money when a side is tested does not help.** On SPY and DIA it is flat
  for the held strangle (+0.02% and +0.07% a month, t 0.2 and 0.5) and lowers the other six books (t -1.1 to
  -2.9).
- **Sizing by VIX (25% of capital below VIX 15 up to 50% above 40), scaled to the same average size, adds
  +0.04% to +0.10% a month** on the strangle, put and lizard books (t 1.0-1.9, pooled across the four ETFs, so
  partly on the flawed quotes). Not significant after correction.
- **Selling premium over earnings loses on 72 months of S&P 100 names:**
  - short ATM straddle: -52 bp a trade, 56% winners (the median trade wins +38 bp; the losers are larger);
  - short 16-delta strangle: -28 bp, 63% winners;
  - the iron condor test is void (10% of its entries had negative credits).

## 4. Daily data

Rules and costs as in the docstring of `scripts/untested_edges_lab.py`; Yahoo prices, the CFTC legacy reports,
point-in-time rates.

| Test | Result | t | Detail |
|---|---|---|---|
| D1 RSI(2) dip-buying on SPY, QQQ, DIA, IWM | +38 bp a trade, 873 trades | 4.08 | random-day control +19 bp (difference t 1.97); 1993-2018 +42 bp (t 3.9), 2019-2025 +26 bp (t 1.4); SPY 56, QQQ 39, IWM 36, DIA 19 bp |
| D1b the same, flat at each Friday close | +13 bp | 1.50 | 65% of D1's trades hold a weekend; most of its gain sits there |
| D2 TLT, last 5 days of the month | +0.38% a month | 4.12 | 19 of 24 years; permutation p 0.0025; 2023-2024 +0.22% |
| D3 COT speculative pressure, 7 currencies | -0.01% a month | -0.12 | after the published sample (June 2018 on): -0.07%, t -1.1 |
| D4 volatility-managed SPY | alpha +0.26% a month | 2.26 | beta 0.51; Sharpe 0.71 against SPY's 0.60 |
| D5 overnight, close to next open (1 bp a round trip) | DAX +2.9 bp a night, CAC +2.6, Nikkei +2.1, Hang Seng +3.8, Euro Stoxx +0.6 | 4.06, 3.30, 3.04, 3.74, 1.33 | after CFD financing (T-bill + 2.5% a year): DAX +1.5 bp (t 2.05), Hang Seng +2.4 (t 2.31), CAC +1.2 (t 1.49), Nikkei +0.7 (t 0.95), Euro Stoxx -0.4 |
| D6 BTC calendar | weekend +0.15% a week; Monday +0.37%; turn of the month +1.50% | 1.03, 2.39, 2.24 | turn of the month positive in 6 of 13 years |
| D7 BTC 14/56-day trend | alpha +1.25% a month | 1.65 | beta 0.63; worst month -24% against -44% held |
| D8 crypto carry on Binance and Bybit | not run | - | both venues refuse GitHub's US runners (HTTP 451 and 403) |
| D9 commodity ETF seasonality | -0.39% a month | -0.85 | 8 of 16 years |
| D10 factor ETFs (MTUM, QUAL, USMV, VLUE, SIZE) | alpha -0.01% a month | -0.13 | SIZE -0.23% (t -2.2); none positive with t > 1 |
| D10b Russell reconstitution, IWM minus SPY (two-sided) | +0.74% in the 5 days before, -0.60% in the 5 after | 2.72, -1.74 | 25 years; descriptive |
| D11 currency value, 5-year reversal | +0.08% a month | 0.78 | 12 of 25 years |
| D12 cross-asset signals into FX, next day | oil to USDCAD, SPY to AUDJPY, VIX to AUDJPY, gold to AUDUSD, yield to USDJPY | -1.53, -4.23, -2.91, +1.54, -2.29 | the SPY and VIX rules lose significantly: AUDJPY tends to reverse the day after a risk move |
| D13 earnings-announcement premium, S&P 100 | +0.43% a month over SPY | 1.48 | 2,381 events, 4 of 6 years |
| D14 gold multi-length Donchian | +0.66 bp a day | - | 200 random-entry books with the same exits make +0.34 bp; p 0.28 |

- **Three known anomalies show up with t above 4:** short-term dip-buying in index ETFs, TLT into month-end and
  the overnight drift in European and Asian indices.
- **Each has a catch:**
  - D1: half of it is what random entries with the same holding periods also earn (difference t 1.97); it drops
    to +26 bp after 2018 (t 1.4) and to +13 bp when every position is closed on Friday.
  - D2 is a bond-ETF trade, about 12 trades a year.
  - D5 halves once a CFD account's overnight financing is paid. On index futures the T-bill part is in the futures
    price but the 2.5% mark-up is not.
- D12's two negative results are as large as the positives elsewhere, but they were not the hypothesis; reversing
  them would be a new test.

## 5. Intraday

Rules in the docstring of `scripts/untested_intraday_lab.py`. M1 bid/ask bars: the FX-Data mirror for 2007-2018
(EURUSD to mid-2022; most pairs end May-June 2018) and Dukascopy for 2019-2025. Net includes every cost and the
swap; gross is the same trades on mid prices.

### 5.1 USD/JPY around the Tokyo fix (2007-2025, complete)

Each Gotobi test also ran on a control: the same trade on Tuesday-Friday business days that are not Gotobi days.

| Test | Trades | Net, bp a trade | t | Gross bp | t 2007-2018 | 2019-2025, bp (t) | Control net bp | Gotobi minus control, t |
|---|---|---|---|---|---|---|---|---|
| I1 short 09:55-10:30 Tokyo after the fix | 1,256 | -0.42 | -1.17 | +1.46 | -2.86 | +1.33 (2.90) | -1.45 | 2.47 |
| I2 long 07:30-09:55 Tokyo | 1,250 | -0.34 | -0.61 | +2.02 | -1.27 | +0.74 (0.90) | -2.18 | 2.73 |
| **I3 long 23:00 the evening before to 09:53, stop 0.5 x the 20-day range** | **1,074** | **+4.45** | **3.89** | **+6.24** | **2.95** | **+4.18 (2.72)** | **-2.37** | **5.05** |
| I5 03:00-09:55 on the day after a Japanese holiday | 122 | +1.69 | 0.65 | +3.52 | -0.35 | +7.89 (2.80) | -1.60 | 1.24 |
| I6a long 00:00-07:00 UTC, every weekday | 4,481 | -3.61 | -7.74 | -1.74 | -8.09 | -1.37 (-1.85) | - | - |
| I6d short 13:00-20:00 UTC, every weekday | 4,376 | -2.44 | -4.21 | -0.58 | -3.15 | -2.62 (-2.91) | - | - |

- **I3, the "Gotobi Overnight JPY" rule from the MQL5 review, is the strongest new result of the batch.**
  - Positive in 15 of 19 years, and in both halves (t 2.95 and 2.72).
  - In 2019-2025 it makes +4.18 bp a trade; the confirmed 03:00-09:55 Gotobi rule made +2.13 bp there
    (`docs/gotobi_oos.md`). In 2025 it made +6.78 bp where that rule lost 0.88.
  - The same window on other days loses 2.37 bp, so this is the Gotobi effect, not a time-of-day drift of USD/JPY.
  - It overlaps the confirmed Gotobi trade on the same days. It is a better form of that edge, not an
    independent second edge.
  - Its worst trade is -0.67% in 2019-2025 (-1.29% in 2008, a gap through the stop).
- **On the FTMO rules of `docs/prop_firms.md`** (20,000 simulated traders, block-bootstrapped trades):

  | Stream | Leverage | Pass both phases | Ever paid | Mean net per challenge | Median | Failed on daily loss |
  |---|---|---|---|---|---|---|
  | I3, 2019-2025 | 6x | 72% | 71% | +$10,626 | +$5,731 | 0% |
  | I3, 2019-2025 | 10x | 58% | 56% | +$15,958 | +$9,984 | 54% |
  | I3, 2007-2025 | 6x | 60% | 55% | +$8,094 | +$2,233 | 13% |
  | Confirmed Gotobi rule, 2019-2025 | 6x | 62% | 61% | +$4,013 | +$2,928 | 0% |

  - The simulator books each trade on one day. I3 is open across the FTMO midnight reset, so a trade's loss can
    count against two days; with the stop, one trade at 6x risks about 4% at most.
- **The fix trades without the overnight leg (I1, I2) beat their controls but not their costs.** I1 flipped sign
  between the two halves.
- **I6, home-currency depreciation in home hours (Ranaldo 2009), is reversed on USD/JPY:** the yen rose in Tokyo
  hours and the dollar in New York hours. Trading the reverse would be a new, post-hoc hypothesis.

### 5.2 Other FX pairs and gold (2007-2018; the 2019-2025 files are downloading)

| Test | Trades | Net bp | t | Gross bp | Gross t |
|---|---|---|---|---|---|
| I4 EURJPY Gotobi 03:00-09:55 | 685 | -0.06 | -0.04 | +2.53 | 1.74 |
| I6b short EURUSD 07:00-12:00 UTC (to 2022) | 3,963 | +0.42 | 0.88 | +1.96 | 4.07 |
| I6c long EURUSD 13:00-20:00 UTC (to 2022) | 3,962 | -0.61 | -1.00 | +0.94 | 1.55 |
| I7 night scalping AUDNZD | 4,261 | -7.40 | -40.4 | +1.58 | 8.48 |
| I7 night scalping EURGBP | 4,268 | -3.34 | -23.7 | +1.77 | 12.1 |
| I7 night scalping GBPCAD | 2,890 | -6.13 | -34.9 | +2.00 | 11.9 |
| I8a fading EURUSD H1 pin bars (to 2022) | 2,998 | -0.61 | -1.72 | +0.76 | 2.18 |
| I8b fading EURUSD H1 inside-bar breakouts (to 2022) | 3,109 | -0.25 | -0.47 | +1.49 | 2.85 |
| I9 gold London breakout of the Asian range (2011-2018) | 1,791 | -0.86 | -0.63 | +3.51 | 2.52 |

- **Night scalping works before costs and fails after them.** The Bollinger fade at 21:00-01:00 UTC earns 1.6-2.0
  bp a trade at mid (t 8-12), but the rollover-hour spread costs 5-9 bp a round trip.
- **Retail candle setups lose a little, so fading them gains a little before costs** (+0.8 and +1.5 bp), not
  after.
- **The gold London breakout and the EURJPY Gotobi trade also have gross edges that the spread removes.**
- I10, the weekend-gap fade, counts only on 2019-2025 (its 2007-2018 version was the project's earlier near-miss:
  31 AUDUSD and 33 EURJPY trades, t 0.34 and 1.81). Its files are downloading.

### 5.3 Index CFDs (2019-2025; downloading)

I11 (index-review closing auctions on FTSE, CAC, SMI) and I12 (US500/NAS100 intraday momentum; opening-range
breakout with breakeven and trailing on US500, NAS100, DAX, FTSE) need the Dukascopy index files.
| Test | Trades | Net bp | t | Gross bp | Gross t | Years positive |
|---|---|---|---|---|---|---|
| I12a US500, first half-hour sign held 15:30-16:00 New York | 1,299 | -2.41 | -2.52 | +0.03 | 0.03 | 0/7 |
| I12a NAS100, the same | 1,008 | -3.70 | -2.83 | -1.42 | -1.09 | 0/6 |
| I12b US500 opening-range breakout, breakeven and trailing | 1,111 | +0.10 | 0.06 | +2.11 | 1.35 | 2/7 |
| I12b NAS100 opening-range breakout, breakeven and trailing | 875 | -1.66 | -0.67 | +0.30 | 0.12 | 3/6 |

- **Neither US index rule has an edge.** The intraday-momentum effect is zero before costs on the CFDs in
  2019-2025; the opening-range breakout with trailing stops makes 2 bp before costs and nothing after.
- NAS100 2024-2025 and the DAX, FTSE, CAC and SMI files (I11, the rest of I12) are still downloading.
- About 22% of the requested index minutes are missing (hours Dukascopy refused after every retry); days with
  holes in the session are skipped, which is why US500 has 1,111 breakout days rather than about 1,760.

## 6. All tests together

Interim, 100 primary tests (`docs/untested_batch/summary.csv`; FTSE overnight excluded, D8 not run):

| Test | Effect | t | q |
|---|---|---|---|
| D2 TLT month-end | +0.38% a month | 4.12 | 0.0008 |
| D1 RSI(2) dip-buying, 4 ETFs | +38 bp a trade | 4.08 | 0.0008 |
| D5 overnight DAX (1 bp a round trip) | +2.9 bp a night | 4.06 | 0.0008 |
| I3 USD/JPY Gotobi overnight | +4.45 bp a trade | 3.89 | 0.0013 |
| D5 overnight Hang Seng | +3.8 bp a night | 3.74 | 0.0018 |
| D5 overnight CAC 40 | +2.6 bp a night | 3.30 | 0.0082 |
| D5 overnight Nikkei 225 | +2.1 bp a night | 3.04 | 0.0167 |

- Just outside: BTC Monday (q 0.104), volatility-managed SPY and BTC turn of the month (q 0.126).
- After CFD financing the overnight results fall to t 2.05 (DAX), 2.31 (Hang Seng), 1.49 (CAC) and 0.95 (Nikkei).
- No options test survives.

## 7. What this changes

- **The Gotobi trade has a better form to carry forward.** The next step is the locked 2026 data: freeze I3
  exactly as written, next to the confirmed 03:00-09:55 rule, and test both once - only with explicit
  authorisation to open the holdout.
- **For option selling, hold to expiry.** Management buys crash protection and costs return in normal years; no
  option book earned a significant alpha.
- **The daily anomalies are tradeable in principle but not obviously at a prop firm:** D1 holds over weekends, D2
  is a bond-ETF trade, D5 pays overnight CFD financing. Each would need its own firm-rule check before use.
- **For the many cost-killed rules, the spread is the binding constraint, not the signal.**
