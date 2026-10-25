# Paths to real money in trading and markets, ranked by evidence

Third, deeper search, 2026-10-25. It complements `docs/profitable_traders.md`, `docs/options_retail.md` and
`docs/prop_firms.md`.

**The money in this field goes to:**
1. those who own or work at the firms with the edge (speed, flow, capital);
2. those who sell services to traders;
3. those who run a verified edge on other people's capital.

Retail traders compounding a small account of their own are rarely on the list.

## 1. Where the money is

| Who | Evidence |
|---|---|
| Market makers / proprietary trading firms | 2025 net trading revenue: Jane Street **$39.6bn** (record), HRT about **$12.3bn**, Citadel Securities about **$12.2bn**. JPMorgan's trading revenue was $35.8bn. (Reuters via MarketScreener, April 2026) |
| Their employees | Graduate total pay: Jane Street and Citadel Securities about $400k-700k in year one, HRT $300k-500k, Optiver $250k-400k. Senior $700k-2m+; portfolio managers $1m-10m+. These are self-reported estimates (Quantt 2026), not employer data. |
| Prop "challenge" firms | FTMO 2023: revenue **$203.9m**, operating costs $100.3m, staff $4.7m. Industry benchmark payouts are about 30% of challenge sales. About $325m of payouts were tracked in 2025, 77% of it from three brands. |
| On-chain arbitrage (MEV) | $233.8m extracted by 19 searchers (Aug 2023 - Mar 2025). The top 3 took 73%, rising to about 90% by Q1 2025, and active searchers fell from 23 to 11. Capital, latency and builder deals are the barriers. |
| Retail traders | Fewer than 1% of day traders are persistently profitable (Taiwan); 97% of persistent Brazilian day traders lost; about 7% of prop-challenge buyers are ever paid; 83% of a trading championship's 688 entrants lost in a rising market. |

## 2. The paths, from most to least reliable

### A. Get paid by those with the capital: a quant / trading career

- **Payoff:** the highest expected value in the field for someone with the skills.
- **Skills:** probability, statistics, programming. Firms hire through puzzles, competitions and interviews.
- **Low-cost entry points:**
  - **WorldQuant BRAIN:** free platform; at Gold level (10,000 points) you may be invited to the part-time
    consultant programme. Pay depends on performance: "upwards of $2,000" a quarter at Master and "$8,000 or more"
    at Grandmaster; typical self-reported early earnings are far lower. The 2026 International Quant Championship
    drew more than 156,000 students; the top global prize is $20,000 per team. WorldQuant says "more than 80
    participants have received offers" through IQC and BRAIN.
  - **Numerai:** stake NMR on model predictions. Payouts depend on the token price and on rules that have changed
    over time, and the forum questions their sustainability.
  - **Kaggle / firm-run competitions:** a public record of skill.

### B. Sell picks and shovels: a business serving traders

- **Prop-firm economics are fee-driven.**
  - Start-up costs, per TradeInformer: a white-label platform $15k-40k, website and CRM $10k-20k, a payout reserve
    $50k-150k, then a team at $20k-40k a month and paid marketing from $4k-10k+ a month. Profitability typically
    takes 6-7 months.
  - Risks: payment processors (Deel stopped paying out for some firms), regulators (the CFTC's My Forex Funds case;
    it was dismissed, with the agency sanctioned), and firm collapses (The Funded Trader and Skilled Funded Trader in
    2024).
- **Tools:** trading journals and analytics (TradeZella and similar), data, backtesting.
- **Compliance and risk software** for prop firms and brokers, which the evidence points to as a growing need:
  - "Payment providers expect robust KYC, AML, and payout monitoring";
  - the regulatory scrutiny of prop firms is increasing.
- This path earns from the traders' activity rather than from beating the market.

### C. Run a verified edge on other people's capital (no capital of your own)

| Route | How it pays | Requirement |
|---|---|---|
| Prop challenges | 80-90% of profits | An edge that fits the rules. In the simulation, Gotobi was paid about 50% of the time against about 6% for no-edge traders (`docs/prop_firms.md`); the 2019-2025 check is in `docs/gotobi_oos.md`. |
| Darwinex Zero | A subscription; a rating of 75 or more qualifies for DarwinIA allocations (one trader reports EUR 30,000); the trader earns **15%** of the profit on the allocated capital | A risk-adjusted, audited track record (profit relative to drawdown) |
| A fund of your own | Management and performance fees | A multi-year track record. 82% of emerging managers name raising capital as their top hurdle (IG survey of 507 UK managers). |

### D. Compound your own capital with real premia (slow, needs capital)

| Premium | Evidence | Realistic pace |
|---|---|---|
| Index option selling (defined-risk, contango filter) | +1.8 vol points a month, t 4.7, 1990-2025; worst month -88 (`docs/options_retail.md`) | About 6-12% a year at survivable sizes, with crash years |
| Crypto cash-and-carry (long spot, short perpetual) | BitMEX funding positive in 71.4% of 9,941 periods since 2016. The 2024 mean of 0.0173% per period is about 19% a year (8-hour funding assumed); extreme levels are down about 90% since 2017. Risks: exchange failure, liquidation of the short leg, negative funding. | Low to mid teens a year in calm markets, compressed since the ETFs |
| Gotobi, own account | +3.4 bp a trade, about 2% a year unlevered (2007-2018) | Small |

Even 15% a year turns $5,000 into about $20,000 in 10 years. Compounding a small account does not make anyone rich
quickly; it needs capital.

### E. Lottery paths (avoid)

Leveraged trading competitions, buying options or 0DTE, earnings bets, short-VIX ETNs, martingale, prop-challenge
gambling, and copying popular traders (popularity grows with past performance, but the study does not show that
performance persists).

## 3. Synthesis

- **Fastest real wealth:** skill sold to the firms that already have the edge (A), or a business that serves the
  traders (B).
- **For a trader without capital:** a *verified* edge run on other people's capital (C): a prop firm or Darwinex
  now, with a track record that can later raise money.
- **Own capital:** use real premia (D), sized to survive the worst month in the record, and expect slow
  compounding.
- **E** is where retail money goes.

## Sources
- Jane Street 2025 trading revenue (Reuters via MarketScreener): https://www.marketscreener.com/news/jane-street-s-40-billion-trading-haul-tops-rivals-sources-say-ce7f59dfdf8ff52d
- Quant trader pay 2026 (Quantt): https://www.quantt.co.uk/resources/quant-trader-salary
- WorldQuant BRAIN consultant programme (Quantt guide): https://www.quantt.co.uk/resources/worldquant-brain
- Numerai forum, sustainable payouts: https://forum.numer.ai/t/sustainable-payouts/2561
- Darwinex Zero allocations (trader's blog, promotional): https://www.mql5.com/en/blogs/post/773634
- IG Prime, raising funds for a hedge fund launch: https://www.ig.com/uk/prime/insights/articles/how-to-raise-funds-for-a-hedge-fund-launch-250203
- TradeInformer, the true cost of starting a prop firm: https://www.tradeinformer.com/learn/the-true-cost-of-starting-a-prop-firm-in-2025
- TradeInformer, what FTMO tells us about prop firm finances: https://www.tradeinformer.com/prop-weekly/what-ftmo-tells-us-about-prop-firm-finances
- Prop Firm Match payouts 2025: https://www.financemagnates.com/forex/prop-firm-match-tracked-about-325-million-in-payouts-to-traders-in-2025/
- NexusFi, prop firm fraud and regulatory risk: https://nexusfi.com/a/prop-firms/prop-firm-fraud-regulatory-risk
- CEX-DEX arbitrage / MEV searcher profitability (AFT 2025): https://drops.dagstuhl.de/storage/00lipics/lipics-vol354-aft2025/html/LIPIcs.AFT.2025.26/LIPIcs.AFT.2025.26.html
- BitMEX, 9 years of funding rates: https://blog.bitmex.com/2025q2-derivatives-report
- Funding-rate arbitrage risks (Athenum, promotional): https://athenum.mataroa.blog/blog/funding-rate-arbitrage-how-the-perp-spot-basis-pays-and-when-it-does-not/
- eToro popularity and performance: https://arxiv.org/pdf/1406.7729
