# tastytrade's options rules against our real-quote tests

The question was whether tastytrade (tastylive) holds something the project missed.

**tastytrade's core mechanics:**
- sell premium at about 45 days to expiry, at one standard deviation (16 delta);
- strangles, iron condors, short puts, jade lizards;
- take profit at 50% of the credit, or close at 21 days to expiry;
- sell when IV rank is high;
- trade small and often.

**Data:**
- Their own studies report win rates and "P&L per day", not risk-adjusted returns or comparisons with buy-and-hold.
- Our numbers come from the 92-structure run: real DoltHub quotes, 2019-2025, fills at mid +- 0.75 half-spread,
  USD 0.65 per contract (run 40370344366, branch `options-strategies-all`).
- Full table: `docs/tastytrade/spy_dia_45d.csv`.

## 1. SPY, 45 days, one position a month (66 cycles)

| Structure | Rule | bp per cycle | bp per day held | Sharpe | t | Win rate | Worst cycle | Alpha t vs SPY |
|---|---|---|---|---|---|---|---|---|
| 16-delta strangle | hold | +26.7 | +0.54 | 0.42 | 0.99 | 76% | -14.8% | +0.2 |
| 16-delta strangle | 50% profit | +13.1 | +0.39 | 0.22 | 0.52 | 83% | -14.8% | -0.15 |
| 16-delta strangle | 21 days left | -22.5 | -0.78 | -0.35 | -0.83 | 65% | -10.6% | -1.7 |
| 16-delta strangle | 50% or 21 days | -18.2 | -0.71 | -0.29 | -0.67 | 71% | -10.6% | -1.5 |
| 16-delta strangle | hold, IV above its 1-year median | +3.0 | +0.06 | 0.03 | 0.05 | 70% | -14.8% | -0.15 |
| Iron condor 16/5 | hold | +7.9 | +0.16 | 0.19 | 0.45 | 74% | -7.4% | 0.0 |
| Iron condor 16/5 | 50% profit | -4.8 | -0.13 | -0.13 | -0.29 | 79% | -7.4% | -0.6 |
| 30-delta short put | hold | +57.9 | +1.18 | 0.65 | 1.51 | 86% | -19.2% | -0.5 |
| 30-delta short put | 50% profit | +28.6 | +1.51 | 0.35 | 0.82 | 94% | -19.2% | -0.4 |
| Jade lizard | hold | +29.1 | +0.59 | 0.35 | 0.82 | 62% | -18.6% | -0.5 |
| Jade lizard | 50% profit | +9.2 | +0.23 | 0.12 | 0.28 | 68% | -18.6% | -1.1 |

**What the table shows:**
- **Taking profit at 50% raises the win rate, as tastytrade says** (76% → 83% on the strangle), **but lowers the
  mean and the Sharpe.** It cuts the winners and keeps every loser.
  - Per day held, the short puts do earn more with it (30-delta: +1.51 against +1.18 bp a day), which is
    tastytrade's capital-efficiency argument.
  - That only pays if the freed capital is redeployed at once. Our test waits for the next monthly entry
    (section 3).
- **Closing at 21 days turns the strangle negative**, even per day held (-0.78 bp a day against +0.54).
  - The first half of a 45-day option's life earns less than the last three weeks.
  - Closing early also pays a second spread.
- **Excluding 2020 does not change this.** Across SPY and DIA short-premium structures entered at 45 days, closing at 21 days lost
  13.6% of notional over the other six years against holding, and helped only 18% of structures.
- **Selling only when IV is high made SPY worse** (+27 → +3 bp on the strangle). In 2019-2025 the high-IV
  entries ran into 2020 and 2022.
- **None of these has a significant alpha:**
  - the short puts and the jade lizard earn SPY beta;
  - the strangle and condor earn about zero after spreads.

**One outlier:**
- The 16-delta SPY put with a 50% take-profit shows Sharpe 2.5 (t 5.9).
- That rests on one lucky exit: its February 2020 position hit 50% before the crash, so 2020 was +8.3% instead
  of -3.0%.
- The same rule on DIA lost 9.8% in 2020 (Sharpe 0.27).

## 2. Independent tests

- **Spintwig** (SPY short puts, 2018-2020): holding to expiry beat taking profit at 50%, which lowered premium
  capture and raised commissions.
- **FlashAlpha** (SPY put spreads, minute chains 2019-2026, realistic fills):
  - mid-price fills overstate CAGR by 30-60%;
  - the 30-day, 50% take-profit cell made 3.8% a year (Sharpe 0.23);
  - a 200% stop lost money;
  - no buy-and-hold comparison and no holdout.
- **Option Alpha** (1,546 earnings events): selling premium into earnings won 57% of the time with a negative
  average P&L (`docs/options_retail.md`).

## 3. What tastytrade does that this project has not tested

| Idea | Why not tested | Could it be? |
|---|---|---|
| Rolling the untested side of a strangle when the other side is tested | Our ROLL rule rolls the tested short out in time (the worst rule: -1.5 Sharpe) | Yes, on the same chains |
| Closing at 50% and opening the next 45-day position the same day | Our rules wait for the next monthly entry | Yes; the per-day gain on short puts says it could help, at the cost of more trades and spreads |
| Closing at 21 days and opening the next 45-day position the same day | Our D21 waits for the next monthly entry | Yes; but D21 is negative per day held, so repeating that phase should not help |
| Buying-power allocation by VIX level (25-50%) | Sizing, not an entry rule | Yes |
| Earnings strangles / iron condors (IV crush) | The idea came from our own H8 result, so it needs unused data | Needs 2026 holdout or 2007-2018 quotes |
| 0DTE SPX | End-of-day quotes only | Needs intraday chains (not free) |
| Futures-option strangles (/ES, /CL, /GC, /ZB, /6E) | Not in the free database | Needs paid futures-options history |

## 4. tastytrade as a broker

- **Commissions** (stockbrokers.com, 2026):
  - stock / ETF options $1 per contract to open, $0 to close, capped at $10 per leg;
  - futures options $1.25 each way.
  - Close to the $0.65 per side in our cost model; slightly cheaper for positions closed early.
- **It is a broker, not a capital provider.** It does not fund traders: the money at risk is your own.

## 5. Conclusion

- tastytrade's mechanics are well tested here, and on 2019-2025 SPY and DIA quotes they do not add an edge:
  - holding to expiry beats their 50% and 21-day management in mean and Sharpe;
  - their IV-rank timing did not help;
  - their structures earn either equity beta or the plain index variance premium.
- **Their claims are true on their own terms:** managing raises win rates and lowers variability. But a higher
  win rate is not a higher return.
- **Testable remnants:** rolling the untested side, same-day redeployment after a 50% or 21-day close, and
  VIX-based sizing. They need the
  option chains, whose artifact (run 40227329852) expires around 30 October 2026.
