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
- Benjamini-Hochberg runs over all primary tests together, 100 so far (`docs/untested_batch/summary.csv`).

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
- **Second batch (section 8):**
  - the whole strategy catalog on 33 daily stock-index, ETF, bond, commodity and crypto markets: no family
    beats buy-and-hold after correction;
  - boosted three-outcome ML entries with three targets beat random entries but lose 5 bp a trade after costs;
  - cost-aware tuning does not rescue the cost-killed rules, except EURJPY Gotobi in the same overnight form as
    the USD/JPY result: +4.67 bp a trade, t 2.89, on 2019-2025.

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
breakout with breakeven and trailing on US500, NAS100, DAX, FTSE) need the Dukascopy index files. The US500 and NAS100 results so far:

| Test | Trades | Net bp | t | Gross bp | Gross t | Years positive |
|---|---|---|---|---|---|---|
| I12a US500, first half-hour sign held 15:30-16:00 New York | 1,299 | -2.41 | -2.52 | +0.03 | 0.03 | 0/7 |
| I12a NAS100, the same | 1,008 | -3.70 | -2.83 | -1.42 | -1.09 | 0/6 |
| I12b US500 opening-range breakout, breakeven and trailing | 1,111 | +0.10 | 0.06 | +2.11 | 1.35 | 2/7 |
| I12b NAS100 opening-range breakout, breakeven and trailing | 875 | -1.66 | -0.67 | +0.30 | 0.12 | 3/6 |

- **Neither US index rule has an edge.** The intraday-momentum effect is zero before costs on the CFDs in
  2019-2025; the opening-range breakout with trailing stops makes 2 bp before costs and nothing after.
- NAS100 2024-2025 and the DAX, FTSE, CAC and SMI files (I11, the rest of I12) are still downloading.
- 5-13% of the requested index hours could not be fetched (Dukascopy refused them after every retry). Days with
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

## 8. Second batch: the strategy catalog on other markets, boosted ML entries, cost-aware tuning

### 8.1 The full catalog on 33 daily markets (`scripts/cross_market_ta_lab.py`, branch `cross-market`, run 40562840315)

**Setup:**
- **Markets:** 16 US stock, sector and country ETFs; the DAX, CAC 40, Euro Stoxx 50, Nikkei 225 and Hang Seng;
  4 bond ETFs; 6 commodity ETFs; Bitcoin and Ether. Daily bars from 1993 (or listing) to 2025.
- **Prices:** excess-return prices (dividends in, the cash rate out), so a long earns the return over cash, as on
  a CFD.
- **Costs and financing:** each market's one-way cost (1-12 bp), plus a 2.5% a year overnight financing mark-up
  (5% for crypto).
- **Strategies:** 64 families: the 53 single-market families of the FX catalog plus 11 new daily ones
  (`fxlab/strategies/daily.py`: internal bar strength, double 7s, cumulative RSI, n down days, 52-week high,
  Faber timing, absolute momentum, volatility target, Halloween, option-expiration week, VIX spike).
- **Tuning:** the project's own walk-forward. Each 6-month block trades the best of 12 settings on the previous
  3 years.
- **Test:** alpha against buying and holding the same markets.

**Results** (2,112 family-market pairs, no errors):

- **No family beats buy-and-hold after correction** (best q 0.91). The median pair has an out-of-sample Sharpe
  of -0.09, against a median of +0.43 for simply holding the market.
- **1.3% of pairs have an alpha t above 2, fewer than chance (2.3%); 10.6% have an alpha t below -2.**
- **The strongest pairs repeat one idea, short-term dip-buying in US stock ETFs:**

  | Market | Family | Sharpe | Alpha a year | Alpha t | Sharpe at 2x costs |
  |---|---|---|---|---|---|
  | XLF | Connors RSI(2) | 0.56 | +8.6% | 3.63 | 0.50 |
  | SPY | Connors RSI(2) | 0.52 | +3.9% | 2.88 | 0.45 |
  | SPY | cumulative RSI | 0.53 | +3.3% | 2.78 | 0.47 |
  | XLY | Connors RSI(2) | 0.54 | +4.6% | 2.76 | 0.46 |
  | XLF | internal bar strength | 0.60 | +12.4% | 2.74 | 0.46 |
  | BTC | Turtle | 0.88 | +23.8% | 2.72 | 0.84 |

  - Pooled over the 16 stock ETFs, Connors RSI(2) has an alpha t of 3.0 and cumulative RSI 2.9. Across all 33
    markets they fall to 1.3 and 1.8, because the same rules earn nothing on bonds, commodities and crypto.
  - It is the same effect as the daily D1 test in section 4: real on US stock indices, weaker after 2018, and not
    significant once all the families and markets tried are counted.
  - Trend rules (Turtle, Faber, Donchian, moving averages) do best on crypto (alpha t 1.6-2.8) and are negative
    on bonds.
- **The worst families are the pattern and reversal rules:** Gann, false breakouts, stochastic reversal,
  weekend-gap fade, support-resistance reversal (alpha t -3.9 to -5.1).

### 8.2 Boosted three-outcome ML entries with three targets (`scripts/ml_entry_lab.py`, same run)

**Setup:**
- **Model:** one LightGBM model across the 33 markets. For each day it learns three outcomes of a trade opened at
  the next open: +2 ATR touched first, -2 ATR first, or neither within 10 days.
- **Features:** 49, fixed in advance (returns from 1 day to 1 year, RSI, internal bar strength, distance to
  averages and highs/lows, volatility, VIX, the yield-curve slope, S&P 500 moves, the asset class).
- **Training:** retrained every year on earlier years only; no tuning. Test years 2007-2025 (earlier years
  had too few markets to train on).
- **Entry:** its top 10% of signals, long or short.
- **Exits:** three take-profits, a third each at +1, +2 and +3 ATR, with the stop at entry after the first.
- **Check:** compared with 200 sets of random entries with the same markets, years, trade counts and long/short
  mix.

**Results:**

| Exit | Trades | Net bp a trade | t | Hit rate | Random entries |
|---|---|---|---|---|---|
| Three targets (primary) | 9,799 | -5.1 | -0.64 | 61% | -18.9 bp (model better, p < 0.005) |
| One target at 2 ATR (the labelled trade) | 9,454 | -5.5 | -0.54 | 49% | - |
| Time exit after 10 days | 6,941 | -7.8 | -0.53 | 48% | - |

- **The model picks better trades than chance, by about 14 bp a trade, but not good enough to cover costs.**
  - The edge over chance is real direction skill, not just picking volatile times: before costs the model's
    trades make +4.7 bp against -8.9 bp for random entries (re-run 40567306176, p < 0.005 on gross returns).
  - Costs and overnight financing take about 10 bp a trade, which turns the +4.7 bp into -5.1 bp.
  - Its longs make +10.9 bp (t 0.9) and its shorts lose 25.3 bp (t -2.5).
  - It is positive in 8 of 19 years.
- **Three targets raise the hit rate** (61% against 49% with one target) **without raising the mean.**
- **Its most-used inputs are the yield-curve slope, the month and the VIX:** market regime, not price patterns.

### 8.3 Cost-aware tuning of the rules that earn before costs (`scripts/cost_tuning_lab.py`, branch `cost-tuning`, run 40562234862; pass 1)

Each year trades the setting that did best in all earlier years, from fixed grids of 8-54 settings:

| Rule | Tested years | Settings | Net bp a trade, tuned | t | Untuned rule, same years |
|---|---|---|---|---|---|
| Night scalping AUDNZD | 2010-2018 | 54 | -4.5 | -4.3 | -6.4 |
| Night scalping EURGBP | 2010-2018 | 54 | +4.0 (100 trades) | 1.1 | -3.1 |
| Night scalping GBPCAD | 2011-2018 | 54 | -3.3 | -1.7 | -5.9 |
| Fading pin bars, EURUSD | 2010-2022 | 24 | -0.1 | -0.1 | -0.4 |
| Fading inside bars, EURUSD | 2010-2022 | 24 | -0.6 | -1.0 | -0.6 |
| Gold London breakout | 2014-2018 | 36 | +0.2 | 0.1 | -2.0 |
| US500 opening-range breakout | 2022-2025 | 18 | -0.7 | -0.3 | +1.8 |
| NAS100 opening-range breakout | 2022-2025 | 18 | -0.5 | -0.2 | +0.8 |
| **EURJPY Gotobi** | **2010-2019** | **8** | **+5.0** | **2.8** | +2.5 |

- **Tuning narrows the night-scalping losses but does not turn them positive** (EURGBP's +4.0 is not
  significant).
- **The fades, gold and the index breakouts stay at zero.**
- **EURJPY Gotobi is the exception:** the tuning picked the same entry in every year, the overnight form of
  section 5.1 (buy 23:00 Tokyo the evening before, stop at half the 20-day range).
  - It is the USD/JPY result appearing on a second pair.
  - **2019-2025 check (pass 2, run 40590254234, with the missing hours downloaded): +4.67 bp a trade, t 2.89,
    371 trades, positive in 6 of 7 years** (2021: -1.0 bp). Over 2010-2025: +4.90 bp, t 3.93.
  - The walk-forward kept the 23:00 entry throughout; it dropped the stop from 2023. In 2019-2022 the stop's
    20-day range came from files holding only the hours around the trade, which made it tighter than designed;
    the no-stop form is unaffected.
  - EURJPY moves with USD/JPY (both are yen pairs), so this confirms the same Gotobi yen flow rather than adding
    an independent edge.

### 8.4 The strategy catalog on US500 and NAS100 intraday bars (`scripts/index_intraday_ta_lab.py`, branch `index-intraday-ta`, run 40567551158)

- **Setup:** 61-63 families on M5, M15, M30, H1 and H4 bars of the two CFDs (Dukascopy, 2019-2025, cash-session
  hours), tuned walk-forward (3 years of training, 6-month test blocks from 2022), with the quoted spread, slippage,
  commission and CFD financing.
- **Result: 614 family-timeframe pairs, none significant after correction, and none with an alpha t above 2.**

  | Timeframe | Median Sharpe, US500 | Median Sharpe, NAS100 | Share with alpha t below -2 (US500 / NAS100) |
  |---|---|---|---|
  | M5 | -1.27 | -0.85 | 59% / 49% |
  | M15 | -0.68 | -0.47 | 38% / 31% |
  | M30 | -0.61 | -0.44 | 25% / 18% |
  | H1 | -0.39 | -0.26 | 13% / 6% |
  | H4 | -0.24 | -0.11 | 3% / 3% |

- **The shorter the bars, the worse the result**, as in the FX sweep: costs grow relative to each trade's move.
- The best pairs are Toby Crabel's narrow-range breakout on US500 (alpha t 1.5-1.8 on M5-H1) and Williams'
  volatility breakout; both are well inside what 614 tries produce by chance.

### 8.5 Triple-barrier ML entries on every timeframe (`scripts/ml_triple_barrier_lab.py`, branch `ml-triple-barrier`, run 40568194520)

**Setup:**
- **Data:** 11 FX pairs and gold on M1, M5, M15, M30, H1, H4 and D1, 2007-2018 (EURUSD to 2022, USD/JPY to 2025);
  the US500 and NAS100 CFDs on M1-H1, 2019-2025. One model per market and timeframe.
- **The trade is the triple barrier itself:** a take-profit barrier, a stop barrier and a time barrier, in three
  shapes fixed in advance (symmetric 1.5 / 1.5 ATR in 12 bars; trend 2 / 1 ATR in 24 bars; reversion 1 / 2 ATR
  in 12 bars).
- **Model:** LightGBM learns which barrier a trade opened at the next bar touches first. It is retrained every
  year on earlier years only, and the strategy takes its top 10% of calls.
- **Fills:** real bid/ask on the M1 path, which decides the order of the barriers.
- **Check:** random entries with the same counts, mix and barriers, compared before and after costs.

**Results (282 market x timeframe x barrier tests):**

| Timeframe | Tests | Median net bp a trade | Median gross bp | Random entries, gross | Share beating random before costs | Median trades |
|---|---|---|---|---|---|---|
| M1 | 42 | -2.86 | +0.20 | -0.02 | 93% | 205,389 |
| M5 | 42 | -3.10 | +0.33 | -0.05 | 86% | 46,119 |
| M15 | 42 | -2.68 | +0.37 | -0.11 | 79% | 15,990 |
| M30 | 42 | -2.16 | +1.00 | -0.18 | 81% | 4,424 |
| H1 | 42 | -1.95 | +1.10 | -0.22 | 79% | 2,702 |
| H4 | 36 | -2.35 | +0.69 | -0.38 | 19% | 1,021 |
| D1 | 36 | -7.48 | -1.04 | -6.45 | 6% | 124 |

- **No test is positive after costs with t above 2; 209 of 282 are significantly negative.** 4% have a positive
  net mean; none survives Benjamini-Hochberg (best q 1.0).
- **The models do read direction a little.** Before costs they beat random entries in 79-93% of the M1-H1 tests,
  and 79% of all tests have a positive gross mean.
- **But the edge is a fraction of a basis point to 1 bp a trade, against 2-3 bp of spread, slippage and
  commission.** The edge grows with the bar size up to H1, but never reaches the costs.
- **The barrier shape hardly matters** (median net -2.5 to -3.1 bp for all three).
- The best tests are D1 trades on EURGBP and GBPUSD (+9 to +10 bp a trade, t 0.9-1.1, about 130 trades each):
  too few trades to mean anything.

### 8.6 Adaptive triple barriers, everything tuned (`scripts/ml_adaptive_barrier_lab.py`, branch `ml-adaptive-barrier`, run 40657560894)

**Setup:**
- **The model sets every barrier.** One boosted model predicts the net return of a trade for any side,
  take-profit (0.5-4 ATR or none), stop (0.5-4 ATR or none) and time limit (3-48 bars): 490 combinations.
- **Each bar's trade** is the combination with the best prediction, so the three barriers are a learned nonlinear
  function of the market state.
- **Features:** 55 (price and indicator features, long context, calendar including Gotobi days and fix times,
  carry, spread and ticks).
- **Tuning inside the walk-forward:** tree settings, loss (L2 or Huber), linear leaves and the trading threshold
  are tuned by Optuna on the year before each test year, for the policy's realised net return. The model is then
  refitted on all earlier years.
- **Coverage:** 11 FX pairs and gold on M1-H4, US500 and NAS100 on M1-H1. D1 ran separately (branch
  `ml-adaptive-barrier-d1`, run 40775845733) with minimums of 1,000 training and 200 validation bars, because a
  daily year (about 260 bars) fell below the 300-bar validation minimum of the main run.

**Results (81 market x timeframe tests):**

| Timeframe | Tests | Median net bp a trade | Median gross bp | Random entries, gross | Share beating random before costs | Median predicted bp |
|---|---|---|---|---|---|---|
| M1 | 13 | -1.58 | -1.52 | -0.59 | 23% | +3.6 |
| M5 | 14 | -2.41 | -0.32 | -0.65 | 14% | +6.4 |
| M15 | 14 | -2.09 | -0.56 | -0.88 | 29% | +7.7 |
| M30 | 14 | -2.92 | -0.60 | -1.16 | 14% | +9.4 |
| H1 | 14 | -2.15 | -0.89 | -1.37 | 0% | +15.2 |
| H4 | 12 | -3.89 | -1.65 | -2.91 | 8% | +26.3 |
| D1 (separate run) | 11 | -12.79 | -3.47 | -10.02 | 18% | +70.4 |

- **None is positive after costs with t above 2; 33 of 81 are significantly negative; none survives
  Benjamini-Hochberg.**
  - 7 of 81 have a positive mean. The best is USD/JPY H4, +3.1 bp, t 1.4.
  - D1 adds 11 tests with the same picture: none with t above 2, 2 below -2, 2 positive (USD/CHF +44.9 bp on 63
    trades, t 1.5; AUD/NZD +22.1 bp on 38 trades, t 0.5).
- **Letting the model choose the barriers made things worse than fixed barriers (section 8.5).**
  - Only 33% of tests have a positive gross mean, against 79% with the fixed shapes.
  - The model predicts +4 to +26 bp a trade and realises about -2 bp: classic overfitting of a 490-way choice.
- **It almost always picks the same shape:** a small take-profit (0.5-1 ATR), a wide or no stop, and the longest
  time limit. That shape wins 60-85% of the time but loses on average, the pattern of grid and martingale systems.
- **Linear leaves blew up in 3 of 81 tests** (mean predictions of 1,500 to 3.4 million bp), which made their
  thresholds meaningless. Removing that option is a possible re-run, but the other 78 tests show the same picture.

### 8.7 Trading as an optimisation problem (`scripts/policy_opt_lab.py`, branch `policy-opt`, run 40778681452)

Section 8.6 still chose the barriers from a grid: a model scored 490 fixed combinations and the best one was
taken. Here there is no grid to back-test. The entry, the side, the take-profit, the stop and the time limit are
explicit functions of the 55 features, and their parameters are found by maximising the strategy's own net profit.

**Setup:**
- **The function:** for each bar it outputs 8 numbers:
  - two logits for long and short against flat;
  - a continuous take-profit (0.25-16 ATR), stop (0.25-16 ATR) and time limit (1-96 bars) for each side.
- **Three function classes, each tested separately:**
  - linear;
  - a small neural network (1-2 hidden layers, 8-64 units);
  - boosted trees fitted directly to the profit objective (LightGBM with a custom objective).
- **Why no backtest loop is needed:** each training bar's trade is simulated once on the M1 bid/ask path for 13
  take-profits x 13 stops x 13 time limits per side, with the same costs and fill rules as before. Between those
  points the profit is interpolated, so it is a smooth function of the parameters, and the optimiser climbs its
  exact gradient (Adam for linear and the network, gradient boosting for the trees).
- **Objective:** chosen on the validation year from three options:
  - the expected net return per bar of the stochastic policy;
  - its Sharpe ratio;
  - the net return per bar held, which matches one account that is either in a trade or waiting.
- **Tuning on the year before each test year:**
  - learning rate, weight decay, network size, tree settings, the objective, and a trading margin;
  - early stopping on the validation year;
  - selection by the validation year's realised net profit (exact simulation, one position at a time).
- **Validation gate:** a test year is traded only if the chosen policy made money in its validation year.
- **Controls:** random entries with the same counts and barriers, before and after costs.
- **Coverage:** the same markets and timeframes as 8.6, plus D1.
- **Saved output:** the fitted linear coefficients for every year, so the learned function can be read.

**Results (226 market x timeframe x function tests with trades, M1-H4; 273,805 trades):**

| Function | Tests | Median net bp a trade | Median gross bp | Share positive | Significantly negative (t < -2) |
|---|---|---|---|---|---|
| Boosted trees | 73 | -2.36 | +0.62 | 12% | 26 |
| Linear | 77 | -2.79 | -0.06 | 6% | 39 |
| Neural network | 76 | -2.59 | +0.02 | 5% | 37 |

| Timeframe | Tests | Median net | Median gross | Random entries, gross |
|---|---|---|---|---|
| M1 | 36 | -2.48 | +0.18 | +0.04 |
| M5 | 41 | -2.16 | +0.22 | -0.08 |
| M15 | 41 | -2.42 | +0.74 | +0.12 |
| M30 | 42 | -2.65 | +0.35 | -0.17 |
| H1 | 36 | -2.89 | -0.37 | -0.42 |
| H4 | 30 | -4.68 | -1.63 | -1.31 |

- **None is positive after costs with t above 2; 102 of 226 are significantly negative; none survives
  Benjamini-Hochberg.**
- **The best:** USD/JPY M30 with the trees, +0.84 bp over 2,638 trades, t 1.2, positive in 11 of 16 years;
  gold H1 trees +7.8 bp on only 72 trades (t 1.0).
- **The objective is positive on the training years and negative on the test years** in every function class and
  timeframe: the optimiser fits the past and none of it carries over.
- **The learned shapes:** linear and network functions mostly choose very wide targets and stops (10-15 ATR) with
  the longest time limits, which amounts to a time exit; the trees choose about 2-3 ATR and 15-20 bars.
- **The before-cost edge is the same as in 8.5 and 8.6:** the trees beat random entries before costs by under
  2 bp a trade, and costs are 2-3 bp.
- **Not tested in this run:** D1 (a daily year minus the 96-bar purge left fewer than the 200 validation bars) and
  four jobs that stopped in their first minute (AUD/NZD and gold H4, US500 and NAS100 H1: LightGBM dropped every
  feature on the small samples). Both were fixed (`policy-opt-fix`) and re-run.

**Conclusion:** Optimising the trade function directly does no better than choosing barriers from a grid. With
price-based features, flexibility only widens the gap between the training and the test years.

## 9. Third batch: lowering costs, new information, more flows, and the prop-firm arithmetic

### 9.1 CFTC positioning (`scripts/cot_lab.py`, branch `cot`, run 40948258662)

**Setup:**
- **Data:** the CFTC legacy futures-only reports (speculators = non-commercials, hedgers = commercials) for the
  euro, yen, pound, Swiss franc, Canadian, Australian and New Zealand dollars and gold; Yahoo daily prices,
  2004-2025.
- **Timing:** positions are as of Tuesday and published on Friday; a signal is traded from the close of the next
  Monday to the following Monday. Carry, financing mark-up and costs (1-2 bp a side) included.
- **Signals, fixed in advance:**
  - C1 crowding reversal: fade the speculators when their positioning is more than 1.5 standard deviations from
    its 3-year average;
  - C2 positioning momentum: follow the 4-week change in their positioning;
  - C3 cross-sectional: long the two most-shorted currencies, short the two most-crowded;
  - C4 follow the hedgers at their extremes.

**Results (35 tests: 4 signals x 8 markets plus pooled):**

| Signal | Pooled bp a week | t | Years positive |
|---|---|---|---|
| C1 crowding reversal | -1.9 | -1.5 | 12/22 |
| C2 positioning momentum | -1.0 | -0.4 | 7/22 |
| C3 cross-sectional crowding | +0.4 | +0.8 | 17/22 |
| C4 follow the hedgers | -0.7 | -0.6 | 9/22 |

- **None has t above 2; none survives Benjamini-Hochberg.** The best single market is C3 on the pound
  (+2.8 bp a week, t 1.9).
- **Positioning data does not predict weekly FX or gold returns after costs.** It is public a week late, and it
  shows.
