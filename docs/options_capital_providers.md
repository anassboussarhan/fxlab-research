# Options capital providers, and the best edge on the best provider

The question is the best edge on the best capital provider, for options. **Update:** the real-quote test of 118,216 option
strategies (`docs/options_strategies.md`) finds the SPY premium after costs about zero in 2019-2025, and the
walk-forward strategies lose on Options Funding and roughly break even on Imperial. The simulation below uses the
variance-swap proxy, which has no bid/ask, and is more favourable than listed options. Rules were checked from public pages on
2026-10-25 and change often, so verify them in writing before buying anything.
- Simulation: `scripts/options_prop_sim.py`.
- Results: `docs/options_prop_sim.csv`.

## 1. Who will fund an options trader

| Provider | Model | Options allowed | Overnight | Drawdown | Cost | Split | Notes / red flags |
|---|---|---|---|---|---|---|---|
| **Options Funding** (WY LLC) | Simulated evaluation, then "funded" | Equity and ETF options (261 whitelisted tickers, incl. SPY / QQQ). Growth plan: spreads, condors, butterflies, calendars, diagonals; naked shorts restricted by delta. Express: long options only. **Index options buy-only since 2026-10-19.** | **Allowed** (Growth) | Growth 6% **end-of-day** trailing; Express 5% intraday trailing | **Monthly** subscription until funded: Express $239-389, Growth $309-499 (discount codes about 50%); $129 activation refunded on first payout | 80% | Payout needs 8 winning days per cycle; claims $1.1m+ paid; Trustpilot 3.2 from 9 reviews (charges after cancellation) |
| **Imperial Trader Funding** | One-step / two-step / instant; sim-funded, then live review | US stocks and options (strike open interest of 500 or more); CME futures add-on | **Allowed on Elite** | Elite: end-of-day trailing, **no daily limit**; Foundation: static + daily | **One-time** fee: $158-$3,457 (Elite $50k about $357) | Elite 90% | Spreads / short options rules **not published**; 33% consistency; sim-funded withdrawals capped at 7% of the balance before live review; unregulated, safety grade C, Trustpilot about 3.2 (48 reviews) |
| Vanquish Trader | Demo accounts | SPX / XSP / VIX long single-leg only; Advanced plan adds spreads and selling to open on other tickers | **Not allowed** (flat by 3:59 pm) | 3-5% trailing | Monthly, $79-1,499 | "100%" | US residents only; demo capital |
| TradeFundrr | Simulated | Stocks, options, futures, crypto | Not allowed | EOD trailing | $199-1,999 + $99/month | 80% | Lifetime payout cap of $15k-25k per account; no refunds |
| Maverick Trading | Interview and training | SPY, QQQ, single stocks | Not stated | Not stated | $1,000 fee + $1,000 bond + $199/month | 60-90% | Terms confidential |
| **T3 Trading Group** | **Real** broker-dealer capital (SEC / FINRA / SIPC member since 2007) | Real listed options, multi-leg with risk approval | Desk rules | Desk rules | First-loss capital contribution (amount on consultation); SIE + Series 57 licence (sponsored) | Not published | US residents / work-authorised only; platform $250 + options module $110 a month |
| The Trading Pit | Broker-written **CFD** options | CFD options | n/a | n/a | n/a | n/a | No expiry / assignment, not listed contracts |
| Futures prop firms (Topstep, Elite Trader Funding, …) | Evaluation | Velotrade found **no** current retail firm with verified options on futures | Mostly flat daily | Intraday trailing | Monthly | 80-100% | n/a |

**The provider must allow what the edge needs:**
- holding for weeks (overnight allowed);
- short premium through spreads;
- an end-of-day, not intraday, trailing drawdown;
- ideally a one-time fee, because the edge is slow.

**Options Funding Growth** is the only provider whose published rules verify all of the strategy needs, but it
charges monthly. **Imperial Elite** has the right structure (one-time fee, end-of-day drawdown, overnight allowed)
but does not publish its options-selling rules.

## 2. The edge

From `docs/options_retail.md`, test O3: sell one-month S&P 500 premium only in months that start with VIX < VIX3M
(contango).
- +1.63 vol points a month after 1 vol point of cost, t 4.0, Sharpe 0.91, 2006-2025.
- It was out of the market in March 2020.
- Worst in-market days: -42.0 vol points (5 Feb 2018), -27.4 (5 Aug 2024), -24.5 (29 Sep 2008).
- In the market 87% of days.

**Proxy:** the daily mark-to-market of short one-month S&P variance at the VIX. On Options Funding the trader
would sell SPY put spreads / iron condors, since index options are buy-only there. Spreads cap the loss and give
up part of the premium, so the proxy is rougher in crashes than a spread book.

## 3. The edge on the providers: every month start 2006-09 to 2023-12 (208 starts, up to 3 years each)

Size = account fraction per vol point of variance notional. A 0.3% size makes about +0.5% a month and lost 12.6%
on the worst in-market day.

| Provider | Size | Pass | Paid at least once | Mean paid | Mean fees | **Mean net** | Median net | 90th pct net | Breached within 3 years |
|---|---|---|---|---|---|---|---|---|---|
| Options Funding Growth $50k | 0.1% | 25% | 23% | $169 | $7,718 | -$7,549 | -$9,000 | -$4,439 | 12% |
| | 0.2% | 48% | 46% | $1,721 | $4,171 | -$2,450 | -$2,625 | +$2,001 | 58% |
| | 0.3% | 59% | 58% | $2,800 | $2,954 | -$155 | -$1,476 | +$6,490 | 72% |
| | 0.5% | 57% | 53% | $3,988 | $1,690 | **+$2,298** | -$500 | +$10,548 | 87% |
| Imperial Elite $50k | 0.1% | 25% | 23% | $190 | $357 | -$167 | -$357 | +$557 | 12% |
| | 0.2% | 49% | 46% | $1,301 | $357 | +$944 | -$357 | +$3,143 | 47% |
| | 0.3% | 60% | 59% | $1,629 | $357 | **+$1,272** | **+$1,036** | +$3,143 | 65% |
| | 0.5% | 61% | 56% | $1,727 | $357 | +$1,370 | +$1,385 | +$3,143 | 58% |

The table rows at 0.15% are omitted; see the CSV.

**Reference: Gotobi on FTMO** (FX, `docs/prop_firms.md`, 3x leverage, EUR 540 fee): paid 50% of the time, mean
net **+$3,142**, median +$388.

What the simulation shows:
1. **The fee model decides it.**
   - The same edge loses money on a monthly subscription at small sizes. Slow premium accrual takes 1-3 years to
     reach +10%, and the fees pile up.
   - On a one-time-fee provider it is positive at every size from 0.15%. At 0.3% it is paid 59% of the time, with a
     median of +$1,036 per $357.
2. **Imperial's 7% sim-funded withdrawal cap** limits the payout to about $3,143 per account before the live
   review, which is where its 90th percentile sits. Beyond that is the unknown live stage.
3. **Bigger size passes faster but breaches more.** At 0.5%, 87% of Options Funding accounts are eventually lost
   within 3 years, yet the payouts before that still beat the fees on average.
4. **The best spot edge on the best FX provider (Gotobi on FTMO)** has a higher mean (+$3,142). The options edge
   on a one-time-fee options provider has a better median. They are uncorrelated, so running both diversifies.

## 4. Recommendation

1. **Options:** run the premium-selling edge on the provider whose written rules allow it.
   - Options Funding Growth is verified: spreads, overnight, 6% end-of-day drawdown.
   - Imperial Elite has the better economics if it confirms in writing that put spreads / iron condors on SPY are
     allowed.
   - Size it so that a Feb-2018-sized day (-42 vol points) costs less than the drawdown, i.e. about 0.15% per vol
     point for a 6% limit. Or accept that a crash ends the account (you lose only the fee) and size at about
     0.3%.
   - Trade only in contango months, with SPY defined-risk spreads, and buy the challenge when the VIX curve is in
     contango.
2. **FX:** Gotobi on an FTMO-type account. The 2019-2025 out-of-sample test confirmed it (+2.1 bp a trade after costs, t 2.05), smaller than before and fading in 2024-2025; it needs about 6x leverage on FTMO rules (`docs/gotobi_oos.md`).
3. **Later, with a track record:** T3 Trading Group (real capital, licensed) or a fund. A verified record is what
   moves you from simulated payouts to real capital.

**Caveats:**
- These are simulated payouts from unregulated firms.
- The rules change: index options became buy-only at Options Funding on 2026-10-19.
- The variance-swap proxy is not the exact P&L of SPY spreads.
- Imperial's $50k drawdown is assumed at 7%.
- No 2026 data was used.

## Sources
- Options Funding: https://optionsfunding.co/
- Velotrade, best prop firms for options trading (rules checked 2026-10-20): https://velotrade.com/blog/best-prop-firm-for-options-trading
- PropFirmReviews, options prop firms (Sept 2026): https://propfirmreviews.net/prop-firms/options/
- Vanquish Trader: https://www.vanquishtrader.com/
- Imperial Trader Funding review (PropFirmMap): https://propfirmmap.com/blog/imperialtraderfunding-review
- TradeFundrr review (PropFirmMap): https://propfirmmap.com/blog/tradefundrr-review
- T3 Trading Group review: https://daytradingz.com/t3-trading-group-review/
- Options Funding reviews (Trustpilot): https://nl-be.trustpilot.com/review/optionsfunding.co
- Trade The Pool (no options): https://tradethepool.com/the-program/
- NexusFi, options at prop firms: https://nexusfi.com/a/prop-firms/options-trading-prop-firm-funded-accounts
