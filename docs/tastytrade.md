# tastytrade: its people, its options rules, and our tests

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

## 2. The longer history: 1994-2018 model layer

The model layer prices options from the underlying and a smile calibrated on the real quotes (median correlation 0.88
with real 2019-2025 cycles). It covers SPY and DIA with the same rules: 298 SPY cycles at 45 days, 1994-2018.
Table: `docs/options_strategies/synth_pre2019_nofilter.csv`.

| SPY, 45 days, 1994-2018 | bp per cycle | Sharpe | t | Win rate | Worst cycle | 2008 |
|---|---|---|---|---|---|---|
| 16-delta strangle, hold | +26.5 | 0.46 | 2.29 | 78% | -18.1% | -4.7% |
| 16-delta strangle, 50% profit | +13.6 | 0.27 | 1.32 | 87% | -18.1% | -8.1% |
| 16-delta strangle, 21 days left | +13.6 | 0.59 | 2.93 | 71% | -4.1% | +7.0% |
| 30-delta short put, hold | +52.0 | 0.64 | 3.17 | 86% | -22.2% | -29.5% |
| 30-delta short put, 21 days left | +30.8 | 0.90 | 4.50 | 77% | -6.2% | -3.1% |

- **Here the 21-day exit does what tastytrade says:**
  - lower return per trade, but a higher Sharpe;
  - a much smaller worst cycle.
- **Its gain is almost all 2008.** Across the 28 SPY / DIA short-premium structures at 45 days:
  - it changes Sharpe by +0.02 on average and beats holding in 43% of them;
  - it adds +13.8% of notional in 2008;
  - in the other years it beats holding in only 10-21% of years.
- **So the 21-day exit is crash insurance:**
  - It pays in a slow crash spread over weeks (2008).
  - It costs premium in normal years.
  - It hurts in a V-shaped crash inside one cycle (2020, when it locked in the low).
- **The 50% take-profit lowers Sharpe in both periods:**
  - -0.14 on average in 1994-2018, better in 14% of structures;
  - -0.18 in 2019-2025, better in 18%.
- **Caveat:** these are model prices, with marks at 21 days from the fitted smile, not quotes. A real-quote check
  for 2010-2018 needs the OptionsDX SPX files (free download, by hand).

## 3. Independent tests

- **tastylive's own allocation study** (SPY 45-day one-standard-deviation strangles at 25% of capital; period not
  stated):
  - closing at 21 days gave the highest return and lowest volatility;
  - closing at 50% beat holding on return.
  - That fits the model layer's long history with 2008 in it, not 2019-2025.
  - Their studies use mid prices and no costs (EliteTrader critique), and report no buy-and-hold comparison.
- **Spintwig** (SPY short puts, 2018-2020): holding to expiry beat taking profit at 50%, which lowered premium
  capture and raised commissions.
- **FlashAlpha** (SPY put spreads, minute chains 2019-2026, realistic fills):
  - mid-price fills overstate CAGR by 30-60%;
  - the 30-day, 50% take-profit cell made 3.8% a year (Sharpe 0.23);
  - a 200% stop lost money;
  - no buy-and-hold comparison and no holdout.
- **Option Alpha** (1,546 earnings events): selling premium into earnings won 57% of the time with a negative
  average P&L (`docs/options_retail.md`).
- **0DTE:**
  - Retail traders lost about $350,000 a day on SPX 0DTE options (Beckmeyer, Branger & Gayda, via Neudata).
  - FlashAlpha (193 SPY sessions, 2022-2026): 0DTE straddles are mildly overpriced (realised / implied move 0.59
    against 0.67 if fair).
  - One day (9 April 2025, +8.7% against a 3.5% implied move) erased weeks of premium.
  - "Selling it naked is a leverage decision, not an edge decision."
- **tastylive's expected-value formula** (SPY 30-day strangles): keep $31.25 of a $125 credit, about 25%.
  - The 16-delta strangle here kept about 23% of its credit (+0.27% on a 1.15% credit).
  - tastytrade's portfolio target of 0.1-0.5% of net liquidation a day in theta is therefore not income: about
    three quarters of the credit is given back in losses.

## 4. The people tastytrade puts forward

**The founders made their money building brokerages:**
- **Tom Sosnoff:** a CBOE market maker in the 1980s (Sosnoff Sheridan Group, with Scott Sheridan).
  - Co-founded thinkorswim in 1999, sold to TD Ameritrade in 2009 for about $606 million (his share was reported
    at about $84 million).
  - Launched tastytrade in 2011 and the broker tastyworks in 2017.
  - Sold to IG Group in 2021 for $1 billion ($300 million cash plus 61 million IG shares).
  - Left the network in September 2025.
- **No personal trading record of the founders or hosts is published.** Forum threads asking for one (NexusFi,
  EliteTrader) found only anecdotes.
- **The business earns on volume:** tastytrade's trading revenue hit a record $50.9 million in one quarter
  (December 2024 - February 2025), up 30% on the year.

**Featured retail traders** (tastylive's "Rising Stars" and interviews). All results are self-reported; tastylive
says documentation is available "on request" and publishes none.

| Trader | Claim | Method | Note |
|---|---|---|---|
| Khalid (article 2023) | $90,000 → $130,000 in a year (+44%) | Premium selling | "A few setbacks"; no figures |
| Derek (episode 2025) | $25,000 → over $10 million since about 2020 | 20-delta, 45-day strangles and covered calls; 50% capital use; NVIDIA and MicroStrategy | A leveraged bet on 2020-2025's best stocks |
| Ash (article 2023) | +41% on $2 million | Formerly 0DTE | **Lost $1 million in one day in 2021**; broke even in 2022 |
| Nishat (article 2023) | $2,500 → $200,000 | Options, mean reversion | None |
| Bobby Gaines (article 2023) | +208%, including 2020 | Short puts and put spreads, 98% win-rate target | None |

- These are survivors chosen after the fact. The losers are not interviewed.
- The academic data say retail option traders lose in aggregate (`docs/options_retail.md`; 0DTE above).

**The cautionary cases:**
- **Karen Bruton, "Karen the Supertrader".**
  - Interviewed several times on tastytrade, including with Sosnoff.
  - Sold far out-of-the-money SPX / RUT / NDX puts and strangles at about 56 days; her funds reached more than
    $175 million.
  - The SEC (May 2016) alleged she hid more than $50 million of losses by rolling them and timing realisations, so as
    to collect monthly incentive fees she was not owed.
  - A 2018 consent judgment ordered $1.24 million disgorgement, a $250,000 penalty and a permanent injunction.
- **OptionSellers.com** (not tastytrade, same short-premium family).
  - Naked short options on natural gas and oil lost about $150 million in November 2018.
  - Clients lost everything; one with $470,000 invested owed $150,000 more to the clearing firm.

## 5. Every tastytrade method and the evidence

| Method | tastytrade's evidence | Independent / our evidence | Verdict |
|---|---|---|---|
| Sell premium | Win rates above the theoretical probability of profit | Index variance premium real (+1.8 vol pts a month 1990-2025, t 4.7); single-stock premium negative after spreads | Real on indices, crash-exposed |
| 45 days to expiry | The theta curve | 14 / 30 / 45 days similar; none has alpha | No special edge |
| 16 delta (one standard deviation) | Probability of profit | SPY strangle Sharpe 0.42 (2019-25), 0.46 (1994-2018 model) | Fine, no alpha |
| Take profit at 50% | Win rate 82% → 90%, P&L per day +77% vs 25% | Lowers Sharpe in both periods; Spintwig agrees | Higher win rate, lower return |
| Close at 21 days | Highest return, lowest volatility (their allocation study) | Worse in 28 of 28 structures in 2019-25; neutral 1994-2018, carried by 2008 | Crash insurance with a cost |
| IV rank > 50 | Their IVR studies | Hurt SPY in 2019-25; Sinclair: compare implied with forecast realised, not IV with its own past | Not supported |
| Trade small, trade often | Law of large numbers | Converges to the expectancy, about zero after spreads except index puts (beta) | Sound sizing, not an edge |
| Theta 0.1-0.5% of net liq a day; beta-weighted delta ≈ SPY; delta / theta 0.5 | No methodology published | About 77% of the credit is given back in losses (above) | Bookkeeping, not an edge |
| Buying power by VIX level | Allocation studies | Not tested; an arXiv paper on 0-5-day SPX puts reports better drawdown control with VIX-rank sizing | **Testable** |
| Roll the tested side out in time | "Trade defense" studies | Worst rule here (-1.5 Sharpe) | Avoid |
| Roll the untested side in | "Rolling untested sides" study | Not tested | **Testable** |
| Strangle vs iron condor | Return on capital studies | Condors lower after spreads (16/5 Sharpe 0.19; 30/16 negative) | Wings cost more than they save |
| Jade lizard, big lizard, ZEBRA, broken wing, ratios | Named and taught by tastytrade | 92-structure run: equity beta or nothing | No edge |
| Bullish-skewed strangles (30/16) | Higher ROC in a bull market (2024 study) | More delta is more beta | Beta |
| Earnings strangles / condors | "Six years of earnings trades" (no figures published) | Option Alpha: negative mean; our long straddle into earnings -0.85% a trade | Mirror untested |
| 0DTE | Recent shows | Retail loses; premium small and tail-dominated | Not for retail |
| Futures strangles (/ES /CL /GC /ZB /6E) | Diversification | No free data; OptionSellers shows the tail | Untestable here |

## 6. What tastytrade does that this project has not tested

| Idea | Why not tested | Could it be? |
|---|---|---|
| Rolling the untested side of a strangle when the other side is tested | Our ROLL rule rolls the tested short out in time (the worst rule: -1.5 Sharpe) | Yes, on the same chains |
| Closing at 50% and opening the next 45-day position the same day | Our rules wait for the next monthly entry | Yes; the per-day gain on short puts says it could help, at the cost of more trades and spreads |
| Closing at 21 days and opening the next 45-day position the same day | Our D21 waits for the next monthly entry | Yes; but D21 is negative per day held, so repeating that phase should not help |
| Buying-power allocation by VIX level (25-50%) | Sizing, not an entry rule | Yes |
| Earnings strangles / iron condors (IV crush) | The idea came from our own H8 result, so it needs unused data | Needs 2026 holdout or 2007-2018 quotes |
| 0DTE SPX | End-of-day quotes only | Needs intraday chains (not free) |
| Futures-option strangles (/ES, /CL, /GC, /ZB, /6E) | Not in the free database | Needs paid futures-options history |

## 7. tastytrade as a broker

- **Commissions** (stockbrokers.com, 2026):
  - stock / ETF options $1 per contract to open, $0 to close, capped at $10 per leg;
  - futures options $1.25 each way.
  - Close to the $0.65 per side in our cost model; slightly cheaper for positions closed early.
- **It is a broker, not a capital provider.** It does not fund traders: the money at risk is your own.

## 8. Conclusion

- tastytrade's mechanics are well tested here. They do not add an edge beyond the index variance premium and
  equity beta:
  - holding to expiry beats their 50% take-profit in both periods;
  - their IV-rank timing did not help;
  - their structures earn equity beta or the plain index variance premium.
- **The 21-day exit is the one rule with a real case.** On the 1994-2018 model layer it improves Sharpe by cutting
  slow-crash losses (2008). It lost in every structure in 2019-2025. It is insurance, not an edge.
- **Their claims are true on their own terms:** managing raises win rates and lowers variability. But a higher
  win rate is not a higher return.
- **The founders' wealth came from building and selling brokerages** (about $606 million and $1 billion), not from
  a published trading record.
- The traders they feature are self-reported survivors. The best-known one was charged by the SEC for hiding
  more than $50 million of losses.
- **Testable remnants:** rolling the untested side, same-day redeployment after a 50% or 21-day close, and
  VIX-based sizing. They need the
  option chains, whose artifact (run 40227329852) expires around 30 October 2026.
