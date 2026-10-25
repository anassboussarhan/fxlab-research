# Prop trading with options, every options strategy, and what profitable retail traders actually do

Second, deeper search, 2026-10-25. It complements `docs/options_retail.md` (free-data tests) and
`docs/prop_firms.md` (prop challenge simulation).

## 1. Prop firms and options

| Firm type | Options? | Constraint that matters |
|---|---|---|
| CFD prop firms (FTMO and similar) | No: forex, 3 US indices, gold, silver, oil, BTC | Overnight holding allowed (Standard funded accounts close before the weekend; Swing accounts have no restriction) |
| Futures prop firms (Topstep, Apex, BluSky, …) | Topstep's help centre says "futures-only". NexusFi reports some firms allow futures options, mostly defined-risk only; BluSky is described as the most options-friendly. Policies change, so get them in writing. | Most force you flat every day (Topstep: "All positions must be closed by 3:10 PM CT"), with an intraday trailing drawdown |
| Options-specialist prop firms (Maverick Trading and similar) | Stocks and options | $5,000 at-risk deposit plus a membership fee; terms "confidential" |

- Option-premium selling earns its return by holding positions as time passes, while evaluation firms impose
  intraday-flat rules and trailing drawdowns. NexusFi calls the two timelines "in structural conflict".
- The only premium selling that fits a futures prop is intraday 0DTE, on the few firms that allow futures
  options.
- Per Early Retirement Now's 2024 results, his 0DTE puts captured only 32% of the premium sold.

## 2. Options strategies by evidence

| Strategy | Evidence | Verdict for retail |
|---|---|---|
| Sell index puts / put-write | PUT 9.28% a year vs S&P 9.76%, at 9.7% vs 14.1% volatility (Dec 1990-2018, Neuberger Berman); 8.5% vs 11.4%, max drawdown -33% vs -51% (1996-2025, this project); alpha +2.3% a year, t 2.2 | Positive. A lower-volatility way to own the equity premium. |
| Covered call / buy-write | ATM (BXM) 8.14% vs out-of-the-money 30-delta (BXMD) 9.59% vs S&P 9.29% (1990-2018). The short-volatility part has Sharpe about 1.0 and adds about 2% a year; the at-the-money version adds an uncompensated timing bet (Israelov & Nielsen) | Positive if out of the money; at the money gives up return |
| Short index variance / strangles / iron condors (premium selling) | +1.8 vol points a month after 1 vol point cost, t 4.7, 86% of years positive (1990-2025, this project). Sharpe 0.91 when selling only in contango (2006-2025). Worst month -88 vol points (March 2020). | The strongest premium found. Defined-risk, small size and a contango filter are essential. |
| Weekly vs monthly writing | WPUT vs PUT differed by more than 800 bp in 2018 alone; implementation details have large effects | Spread across expiries and strikes |
| The "wheel" on single stocks | Popular. Early Retirement Now criticises it ("chasing losses"). Index options carry a larger premium than single-stock options, because the index premium includes correlation risk. | Prefer index options to single stocks |
| Selling into earnings | 648,861 simulated trades on 1,546 events: 57% win rate, **negative** average P&L from tail moves (Option Alpha) | Avoid |
| Buying options before earnings | Average loss 5-9%, 10-14% when big moves are expected (de Silva, Smith & So) | Avoid |
| Buying short-dated / OTM options, 0DTE | US retail lost $1.14bn plus $4.13bn in execution costs, Nov 2019 - Jun 2021 (Bryzgalova et al.) | Avoid |
| Tail hedging (long OTM puts) | Negative returns in every decade from 1985 to 2020; a trend-following hedge is cheaper in the long run (AQR) | Insurance, not a profit source |
| Short VIX ETNs (XIV / SVXY) | About 31% a year (2004-2013 simulation, before costs) and -90% in one day on 5 February 2018 | Do not hold unhedged. One day can end it. |
| Patient execution | Effective option spreads are about a quarter below conventional estimates, and below 40% of them for traders who time their orders (Muravyev & Pearson) | Always use limit orders near mid |

**A real-world premium seller (Early Retirement Now, 2024):**
- Mostly 1-day-to-expiry SPX puts far out of the money, with stop orders and portfolio margin.
- Net options profit was $95,861 in 2024, with 77% of the premium collected. That added 4.8 points to a
  multi-million-dollar portfolio's return.
- **It is an overlay on large capital, not a way to turn a small account into a large one.**

## 3. What profitable traders actually do

| Evidence | Finding |
|---|---|
| Taiwan, all day trades 1992-2006 (Barber, Lee, Liu & Odean) | Fewer than 1% are persistently profitable. They win with fast, aggressive orders that react to market changes, not with information or liquidity provision. |
| US direct-access day traders, 2000 (Garvey & Murphy) | About 50% profitable over 3 months. "Profitable trading goes through ECNs"; most used limit orders; "the average day trader breaks even". |
| Japan retail FX 2021-2023 (Yoshimi) | More limit orders (range trading) go with better results, more stop orders (trend chasing) with worse. Experience lowers success through overconfidence. |
| Brazil index-futures day traders 2013-2015 | 97% of those who kept going lost money; no learning |
| US Investing Championship, a recent six-month round | 83% of 688 participants lost money while the S&P rose 10.2%. The winners made +350% to +1,656% with futures and options leverage in one six-month period, and consistency is "rare". |
| eToro (popularity and performance) | Popularity amplifies performance into more followers; the study does not show that performance persists |
| Post-earnings drift, a classic "small-player" niche | Disappeared from non-microcap stocks after about 2001-2006 (Martineau). It is significant only when microcaps are included (t 2.18, against 1.43 without; Subrahmanyam 2001-2024). Microcaps are hard to trade. |
| Prop payouts tracked in 2025 | About $325m, 77% of it from three firm brands. That is the total paid to all winners combined, against fees from far more buyers. |

The persistent winners share:
1. **An execution or speed advantage:** reacting first, or providing liquidity with limit orders.
2. **A structural, mechanism-based edge with strict sizing:** risk premia, flows.
3. **Enough capital and time** to let a small edge compound.

The very large retail gains come from leverage plus variance, and are rarely repeated.

## 4. A realistic plan without capital

1. **Prop account plus a verified structural edge.** The only candidate here is Gotobi; its out-of-sample test on
   2019-2025 is in `docs/gotobi_oos.md`. In the challenge simulation it was paid about 50% of the time, against
   about 6% for a no-edge trader.
2. **Small account of your own plus index premium selling.** Defined-risk XSP or SPX put spreads, sold only when
   the VIX term structure is in contango, at small size. Expect about +0.5-0.9% a month on capital, with losses of
   20-45% in a crash month at those sizes. It compounds a small account slowly; it does not make it large quickly.
3. **Avoid:** buying options, earnings trades, 0DTE lottery tickets, short-VIX ETNs, martingale, and prop
   "challenge gambling". They are where retail money goes.
4. **Capital comes faster from earnings outside trading than from leverage.** The evidence says turning a few
   hundred dollars into a fortune is mostly luck. A real edge compounds a real amount of money.

## Sources
- Topstep products and hours: https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade
- NexusFi, options trading at prop firms: https://nexusfi.com/a/prop-firms/options-trading-prop-firm-funded-accounts
- FTMO instruments: https://ftmo.oanda.com/faq/which-simulated-instruments-can-i-trade-and-what-strategies-am-i-allowed-to-use/
- FTMO overnight / weekend rules: https://ftmo.oanda.com/faq/do-i-have-to-close-my-positions-overnight-or-before-weekend
- Maverick Trading: https://tradersunion.com/brokers/prop/view/maverick_trading/
- Neuberger Berman, index option writing benchmarks: https://www.nb.com/-/media/NB/Article-Assets/wp_index_option_writing_strategy_benchmarks.pdf
- Israelov & Nielsen, Covered calls uncovered (summary): https://quantpedia.com/covered-calls-uncovered/
- Option Alpha earnings research: https://optionalpha.com/we-stopped-trading-earnings-after-we-saw-this-new-research-188320.html
- AQR, tail risk hedging: https://www.Aqr.Com/-/media/AQR/Documents/Journal-Articles/Journal-of-Systematic-Investing-Vol-1-Issue-1--Tail-Risk-Hedging-AQR.pdf?sc_lang=en
- Volatility trading strategies (Cooper, via CXO): https://cxoadvisory.com/volatility-effects/volatility-trading-strategies
- Early Retirement Now, 2024 options review: https://earlyretirementnow.com/2025/01/14/options-trading-series-part-13-year-2024-review/
- Muravyev & Pearson: https://ideas.repec.org/a/oup/rfinst/v33y2020i11p4973-5014..html
- Garvey & Murphy (via CXO): https://www.cxoadvisory.com/individual-investing/the-odds-of-winning-as-an-active-trader/
- Barber, Lee, Liu & Odean: https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf
- Yoshimi (2025): https://www.chuo-u.ac.jp/uploads/2025/07/discussno412.pdf
- Brazil day traders: https://quantpedia.com/retail-day-trading-is-an-uphill-battle/
- US Investing Championship, recent half-year results: https://briefglance.com/articles/the-17-solution-elite-traders-struggle-as-market-soars-in-us-championship
- eToro popularity and performance: https://arxiv.org/pdf/1406.7729
- Post-earnings drift (UCLA Anderson Review): https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/
- Prop Firm Match payouts 2025: https://www.financemagnates.com/forex/prop-firm-match-tracked-about-325-million-in-payouts-to-traders-in-2025/
- Bryzgalova, Pavlova & Sikorskaya: https://www.fa-mag.com/news/mom-and-pop-investors-took-a-billion-dollar-bath-trading-options-during-the-pandemic-67571.html?section=43
- de Silva, Smith & So: https://mitsloan.mit.edu/ideas-made-to-matter/retail-investors-lose-big-options-markets-research-shows
