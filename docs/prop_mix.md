# Which edges fit a prop account, and mixing them with Gotobi

The question was whether SVXY short volatility suits prop trading, what else does, at which firm, with what
realistic revenue, and whether it can be mixed with Gotobi.
- Script: `scripts/prop_mix_sim.py`. Workflow `prop-mix`, run 40380396038. Results are on branch `prop-mix`.
- Rules: FTMO 2-step 100k (fee USD 585, refunded with the first payout; +10% then +5%; 5% daily and 10% maximum loss
  of the initial balance; 80% of profits). The 2026 rules were checked again (tradetanto.com).
- Each of 20,000 simulated traders starts on a random day of 2019-2025 and walks forward through the real daily
  outcomes, so the strategies keep their true joint behaviour.

## 1. Where each edge can be traded

| Edge | Retail prop firms | Why |
|---|---|---|
| Gotobi USD/JPY | Yes (FTMO and other FX / CFD firms) | FX pair, about 7 hours a trade, swap included |
| Nasdaq-100 overnight | Yes, on an index CFD (e.g. FTMO's US100) | Overnight holding is allowed. Standard funded accounts must be flat over weekends, so Monday-Thursday nights only |
| SVXY / VIX futures (short volatility) | **No** at the common firms | Futures prop firms (Apex, Topstep, MyFundedFutures, Earn2Trade, Bulenox, Elite) route CME-group contracts only and do not list CFE's VIX futures (pickmytrade.io). FTMO-type CFD firms list FX, indices, commodities, stocks and crypto, not volatility ETFs. It is a personal-brokerage trade |
| Hedged SPY / DIA straddles (index variance premium) | Only at options prop firms that allow the stock hedge | `docs/options_edges.md`: Imperial (one-time fee) positive on average; Options Funding (monthly fee) not |
| BTC / ETH cash-and-carry | No | Needs spot plus a perpetual short at an exchange; prop crypto CFDs charge financing on both sides |

## 2. Inputs, 2019-2025 (per unit of notional)

| | Events | Mean | Sd | Worst | Return a year at 1x |
|---|---|---|---|---|---|
| G Gotobi (net of costs and swap) | 413 trades | +2.1 bp | 21 bp | -70 bp (2020-02-28) | +1.3% |
| O Nasdaq-100 overnight (1 bp spread; financing at T-bill + 2.5% a year, an assumption) | 1,376 nights | +5.4 bp | 89 bp | -674 bp (2020-03-12) | +10.6% |
| S SVXY in contango (hypothetical at a prop) | 1,662 days | +7.3 bp | 211 bp | -2,143 bp (2024-08-05) | +17.3% |

- Correlations on days both trade: G-O 0.06, G-S -0.08, O-S 0.37.
- **The SVXY window excludes February 2018.** At its current half leverage, a repeat (VIX futures doubling) would
  cost about 45-50% in a day.

## 3. Outcomes per USD 585 challenge (FTMO 2-step 100k)

"Net" is payouts plus the refund minus the fee, over the time to pass plus one funded year.

| Mix (leverage on 100k) | Pass both phases | Ever paid | Mean net | Median net | Failed on daily loss / max loss |
|---|---|---|---|---|---|
| Gotobi 6x | 62% | 61% | +$4,123 | +$3,311 | 0% / 7% |
| Nasdaq-100 overnight 1x | 58% | 57% | +$4,615 | +$3,020 | 32% / 61% |
| **Gotobi 6x + overnight 1x** | **78%** | **72%** | **+$7,598** | **+$7,596** | 32% / 47% |
| Gotobi 3x + overnight 0.5x | 59% | 57% | +$4,178 | +$1,331 | 0% / 0% |
| Gotobi 3x + overnight 0.5x + SVXY 0.1 (hypothetical) | 74% | 72% | +$6,897 | +$6,203 | 0% / 0% |
| SVXY 0.25 alone (hypothetical) | 40% | 38% | +$965 | -$585 | 33% / 14% |
| SVXY 0.5 alone (hypothetical) | 42% | 39% | +$4,379 | -$585 | 55% / 25% |
| Gotobi 6x + SVXY 0.25 (hypothetical) | 81% | 79% | +$9,688 | +$11,227 | 15% / 10% |

- **SVXY alone is a poor prop strategy.** Small enough to survive its bad days (0.1x), it never reaches the target
  (0.7% pass). Large enough to pass, the median account loses the fee: it is a lottery ticket.
- **Mixing raises the pass and pay rates** because the strategies are nearly uncorrelated. Gotobi plus the
  Nasdaq-100 overnight trade, both tradable at FTMO, is the best realistic mix: paid 72%, median +$7,596 per
  challenge.
- **"Failed" means the account was breached at some point,** including after it had already been paid. Most paid
  accounts end that way.

## 4. Realistic revenue

- FTMO caps simulated capital at $400,000 per trader or strategy across all accounts, before scaling (FTMO FAQ).
- So the realistic ceiling is four 100k accounts running the same mix.
- **Per account:** a mean of about +$7,600 over roughly two years (time to pass plus one funded year).
- **Four accounts:** about $15,000 a year on average. Some years zero; the best decile about +$16,000 per account.

**What is not in the simulation:**
- **Gotobi has weakened:** slightly negative in 2024 (-0.08 bp) and 2025 (-0.88 bp).
- **The overnight trade's intra-night path:** FTMO counts open-equity drawdown, and a night that drops 5% and
  recovers by the open would breach the daily limit. Only the close and open are modelled.
- **CFD details:** the financing rate and spreads are assumptions, not FTMO's quoted terms.
- **Firm rules change.** Running the same trades on several accounts counts toward the $400k cap.
- **Taxes.**

## 5. Conclusion

- **SVXY is not a prop-firm strategy:**
  - the common firms do not offer it;
  - its tail (-21% on 5 August 2024, about -50% in a February-2018 repeat) is incompatible with 5% / 10% loss
    limits.
  - In a personal account, at about 10% of capital, it is a reasonable diversifier, uncorrelated with Gotobi.
- **The realistic prop combination is Gotobi plus the Nasdaq-100 overnight trade on an FX / CFD firm such as FTMO:**
  - paid 72% of the time in this simulation;
  - median +$7,600 per 100k challenge;
  - capped at four accounts.
- Before any money is committed, both edges should be frozen and checked on the untouched 2026 data.
