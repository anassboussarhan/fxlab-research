# People make money at prop firms: skill or variance?

Question raised: "people make money on prop etc." This page compares what the prop-firm and trader-level
evidence says with a simulation of the same challenge rules on this platform's data.
- Script: `scripts/prop_challenge.py`, run locally in 19 s.
- Results: `docs/prop_challenge.csv`.

## 1. What the data on real traders says

| Source | Data | Finding |
|---|---|---|
| FPFX Tech (prop-firm technology provider), via Finance Magnates / Coingape | over 300,000 challenge accounts | 14% of traders ever pass a challenge; **7% ever receive a payout**; the average payout is about 4% of the account; the average funded trader lasts about 22 days |
| The Funded Trader (CEO statement, 2025) | firm data, sample not stated | 5-10% pass; about 20% of those are paid; **1-2% of all clients ever receive a payout** |
| FTMO 2023 accounts (via TradeInformer) | company filings | revenue $203.9m; operating costs $100.3m; staff costs $4.7m. Payouts are not disclosed. |
| Barber, Lee, Liu & Odean (Taiwan, 1992-2006) | every day trade on the exchange, about 360,000 day traders a year | 13-15% profitable after fees in a typical year. **Fewer than 1% (about 1,000) outperform consistently.** The top 500 by prior-year Sharpe earn 28 bp a day net the next year. They win by fast, aggressive orders that react to market changes, not by supplying liquidity or inside information. |
| Chague, De-Losso & Giovannetti (Brazil, 2013-2015) | all 19,646 new mini-index futures day traders | Of those who kept going for more than 300 days, **97% lost money**, 1.1% earned more than the minimum wage and 0.4-0.5% more than a bank teller. No evidence of learning. |
| "The Retail FX Trader: Random Trading and the Negative Sum Game" (NTU) | several retail FX data sets | About 20% of retail FX accounts are profitable; as a group, retail traders show **little if any evidence of beating random trading**, and profits are strongly tied to the spread. |
| Yoshimi (2025), SBI FXTRADE, Japan 2021-2023 | over 120,000 retail FX accounts | Profitable on 55% of trading days. More limit orders (buying low and selling high in a range) go with better results, more stop orders with worse. Experience lowers success (overconfidence). |
| Baron, Brogaard, Hagströmer & Kirilenko (JFQA 2019) | high-frequency trading firms | Differences in relative latency (speed rank) account for large differences in HFT firms' performance. |

So some people do make money consistently, but they are a small minority (well under 1% to a few percent), and
their edge is mostly speed, reaction and market structure. A much larger group is profitable for a while.

## 2. Simulation: how often does a trader with NO edge pass and get paid?

Rules: FTMO 100k 2-step as published (luxalgo.com, verified 2026-01-15).
- **Fee and targets:** fee EUR 540 (USD 585), refunded with the first payout; phase 1 target +10%, phase 2 target
  +5%.
- **Loss limits:** daily loss 5% and maximum loss 10% of the initial balance.
- **Other:** at least 4 trading days per phase; up to 500 trading days to pass. Funded for 250 trading days, with
  80% of profits paid every 10 trading days.

**No-edge trader:**
- Random minute between 07:00 and 19:00 UTC and coin-flip direction, on real EUR/USD M1 bid/ask data 2010-2018.
- Stop of 10 or 20 pips, target 0.5-2 x the stop, closed at the end of the day.
- Real costs: spread, slippage and commission.
- 40,000 trades in the pool. Each of 20,000 simulated traders takes consecutive trades from a random start.

Trade pool (average outcome per trade in R, i.e. multiples of the amount risked):

| Bracket | Mean R | Win rate | t |
|---|---|---|---|
| 10-pip stop, 1:1 | -0.185 | 45.2% | -37.6 |
| 20-pip stop, 1:1 | -0.098 | 46.2% | -21.4 |
| 20-pip stop, target 2x | -0.097 | 36.8% | -16.5 |
| 20-pip stop, target 0.5x | -0.092 | 60.2% | -26.8 |

Outcomes per challenge bought (net = payouts + refund - fee), one trade a day:

| No-edge trader | Pass phase 1 | Pass both | Ever paid | Mean net per challenge |
|---|---|---|---|---|
| 20 pips 1:1, risk 0.5% per trade | 0.5% | 0.0% | 0.0% | -$585 |
| 20 pips 1:1, risk 1% | 7.1% | 1.4% | 0.7% | -$558 |
| 20 pips 1:1, risk 2% | 21.8% | 8.7% | 3.7% | -$332 |
| 20 pips 1:1, risk 4.5% ("bold") | 32.7% | 14.4% | 4.8% | +$15 |
| 20 pips 1:2, risk 2% | 32.7% | 16.2% | **7.4%** | +$156 |
| 20 pips 1:2, risk 4.5% | 33.9% | 16.6% | 6.2% | **+$554** |
| 10 pips 1:1, risk 2% | 13.0% | 4.1% | 1.5% | -$499 |
| 20 pips 1:1, risk 2%, 3 trades a day | 13.4% | 2.9% | 0.5% | -$556 |
| Coin flip with **no costs**, risk 1% (reference) | 50.6% | 35.2% | 29.0% | +$2,206 |

What the simulation shows:

1. **The industry's numbers are what no-edge trading produces.** A trader with no edge, paying real EUR/USD
   costs, passes both phases 1-17% of the time and is paid 0.5-7% of the time, depending on how boldly they bet.
   The 5-14% pass rate and 7% payout rate reported for 300,000 real accounts sit in this range. Seeing many funded
   traders paid out does not show that any of them have an edge.
2. **For a trader without an edge, betting big maximises the chance of passing.** Risking 0.25-0.5% per trade
   almost never passes, because costs grind the account down before the +10% target. Risking 2-4.5% per trade
   with a 2:1 target passes about 17% of the time. This is the classic gambler's result: in an unfavourable game,
   bold play maximises the chance of reaching a target.
3. **A challenge is a lottery ticket with a capped loss.**
   - Under the published rules, a bold no-edge trader can even have a slightly positive expected value: +$156 to
     +$554 per $585 challenge in this model, because the loss is capped at the fee and payouts are kept.
   - But about 93% of such traders lose the whole fee.
   - The model is generous to the trader. It does not include open-trade drawdown inside a trade, consistency
     rules, maximum-risk-per-trade rules, news-trading bans or execution on simulated accounts, which real firms
     use and which the Velotrade report says decide many payouts.
   - The firms' economics work because most buyers trade more often, at smaller risk or with worse costs than
     the best no-edge bet, and lose the fee.
4. **Costs are what separates a free coin flip from a losing one.** Without costs, a 1% coin flip is paid 29% of
   the time and makes +$2,206 per challenge. With the real spread and slippage of a 20-pip bracket (-0.1 R per
   trade), the same risk is paid 0.7% of the time.

## 3. With a real edge: the Gotobi trade

The surviving spot edge (run 19): 700 USD/JPY trades 2007-2018, net of costs and swap, +3.4 bp each. Each
simulated trader starts at a random point of the trade sequence, and on each day a trade occurs with the observed
frequency (0.24 a day).

| Leverage | Mean trade, % of account | Pass both | Ever paid | Mean net per challenge | Median | Fail on daily loss |
|---|---|---|---|---|---|---|
| 3x | +0.10% | 50.3% | **50.2%** | **+$3,142** | +$388 | 17% |
| 6x | +0.20% | 42.0% | 40.8% | +$7,365 | -$585 | 60% |
| 10x | +0.34% | 49.3% | 47.2% | +$12,951 | -$585 | 63% |
| 20x | +0.68% | 27.2% | 18.0% | +$2,335 | -$585 | 87% |

- A small but real edge changes the odds completely: paid about **50%** of the time against 0.5-7% for no-edge
  traders.
- The positive expected value is driven by payouts, not by the lottery effect.
- Higher leverage raises the average but makes most traders fail on the 5% daily-loss rule. Gotobi's worst
  trade is -1.74% unlevered, so at 3x it is -5.2%, about one daily limit.
- Caveats:
  - Gotobi is tested only to mid-2018, entirely inside the papers' sample.
  - A prop firm's spreads, commissions and swaps differ from the costs modelled here.
  - The simulation assumes the edge persists.

## 4. Conclusion

- People do make money at prop firms, but **the reported payout rate (about 7% of buyers, 1-2% at one firm) is
  what traders with no edge produce.** A no-edge trader who bets boldly is paid about 6-7% of the time with real
  costs.
- Payouts are therefore not evidence of skill. Persistence is:
  - in the large academic data sets, fewer than 1% (Taiwan) and about 0.5% (Brazil) of day traders are
    consistently profitable;
  - their edge is speed and reaction, not chart patterns.
- **What would make a prop challenge worth taking** is a real, persistent edge like Gotobi, sized so that one bad
  trade cannot breach the daily limit. Without one, the challenge is a lottery ticket: positive or negative
  expected value depending on the firm's hidden rules, with about a 93% chance of losing the fee.

## Sources
- FPFX Tech / Coingape, 93% of funded prop traders never see a payout: https://coingape.com/blog/93-of-funded-prop-traders-never-see-a-payout-a-new-report-explains-why/
- Finance Magnates, only 1 in 20 traders pass (The Funded Trader): https://www.financemagnates.com/forex/only-1-in-20-traders-pass-prop-firm-challenges-reports-the-funded-trader/
- TradeInformer, what FTMO tells us about prop firm finances: https://www.tradeinformer.com/prop-weekly/what-ftmo-tells-us-about-prop-firm-finances
- Track360, prop trading industry statistics 2026 (partly unsourced): https://track360.io/blog/prop-trading-industry-statistics-2026
- luxalgo, FTMO 100k 2-step rules (verified 2026-01-15): https://www.luxalgo.com/mcp/prop-firm-sim/ftmo-100k-2step/
- Barber, Lee, Liu & Odean, The cross-section of speculator skill: https://faculty.haas.berkeley.edu/odean/papers/day%20traders/Day%20Trading%20Skill%20110523.pdf
- Chague, De-Losso & Giovannetti, Day trading for a living? (summary): https://quantpedia.com/retail-day-trading-is-an-uphill-battle/
- The Retail FX Trader: Random Trading and the Negative Sum Game: https://irep.ntu.ac.uk/id/eprint/26837/
- Yoshimi (2025), Japanese retail FX traders: https://www.chuo-u.ac.jp/uploads/2025/07/discussno412.pdf
- Relative latency and HFT performance (JFQA 2019): https://ideas.repec.org/a/cup/jfinqa/v54y2019i03p993-1024_00.html
