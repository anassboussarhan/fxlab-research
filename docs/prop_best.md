# Best prop firm and setup for the Gotobi trade (2026-11-02)

**Question.** Using everything found so far, which prop-firm programme, account size and leverage give the highest
gain, the best chance of being paid, the fastest payout and the lowest cost?

**The trade.** This is the frozen trade with no filter. Go long USD/JPY and EUR/JPY with equal notional. Enter at
23:00 Tokyo (14:00 UTC) the evening before each Gotobi day and exit at 09:53 Tokyo (00:53 UTC), about 5 nights a
month. Use a resting stop 80 bp below each entry. The winning-night filter cuts the number of nights, which slows
progress towards a target, so it is left out here (`docs/final_outcome.md`, Accuracy).

**Method.** Code: `scripts/prop_best.py`, with results in `docs/prop_best/`.
- Each attempt is a simulated trader who starts on a random business day and trades the real 2019-2025 calendar of
  nights: net return after costs, plus the worst open loss inside each trade.
- 4,000 attempts are run per setup. The evaluation is capped at 2 years and the funded stage at 1 year, so payouts
  after that year are not counted.
- Payouts are 80-90% of any profit on each payout date, after which the account restarts at its initial balance.
- The fee is refunded where the firm says it is.
- Three sizes of edge are tested:
  - **2019-2025:** +4.8 bp a night with the stop.
  - **2026:** every night shifted down so that the mean is +2.04 bp, the 2026 holdout mean.
  - **zero:** mean 0.

The 2026 case is the honest central case.

## Programmes (terms as published, checked 2026-11-02)

FTMO fees are in EUR and converted at 1.15 USD per EUR.

| Programme | Fees by size | Targets | Daily / max loss | Other rules that matter here | Payout |
|---|---|---|---|---|---|
| FTMO 2-Step | 10k €89, 25k €250, 50k €345, 100k €540, 200k €1,080 | 10% then 5% | 5% / 10% static; daily loss measured from the **balance** at midnight CE(S)T | 4 trading days a phase; funded Standard accounts may not open or close within 2 min of major news (the Swing account type is exempt) | 80%; fee refunded with the 1st reward; first reward after 14 days, then every 14 |
| FTMO 1-Step | €79 / 199 / 319 / 499 / 999 | 10% | 3% / 10% trailing (end of day) | best day at most 50% of the profit | 90%, no refund |
| FundedNext Stellar 2-Step | 6k $59.99, 15k $119.99, 25k $199.99, 50k $299.99, 100k $549.99, 200k $1,099.99 | 8% then 5% | 5% / 10% static; measured from the start-of-day balance | **combined open risk at the stops at most 3% of the initial balance** (so at most 3.5x with the 80 bp stop); funded trades opened within 5 min of news keep 40% of profit | 80%; refund with the 1st reward; 21 days, then every 14 |
| FundedNext Stellar 1-Step | $65.99 / 129.99 / 219.99 / 329.99 / 569.99 / 1,099.99 | 10% | 3% / 6% static | 3% risk cap | 80%; refund at the 3rd reward; every 5 business days |
| FundedNext Stellar Lite | 5k $32.99, 10k $59.99, 25k $139.99, 50k $229.99, 100k $399.99, 200k $798.99 | 8% then 4% | 4% / 8% static | 3% risk cap | 80%; refund at the 3rd reward; every 10 days |
| The5ers High Stakes | 2.5k $19, 5k $35, 10k $69, 25k $176, 50k $249, 100k $405 | 10% then 5% | 5% / 10% static; daily loss measured from the **higher of balance and equity at MT5 midnight** (17:00 New York) | 3 profitable days (at least 0.5% closed) a step; no new order within 2 min of high-impact news | 80%; refund (modelled with the 1st payout); 14 days, then every 14 |
| The5ers High Stakes Classic | 2.5k $22, 100k $455 (other sizes not found) | 8% then 5% | as High Stakes | as High Stakes | as High Stakes |

**Looked at but not simulated:**
- FXIFY: 30- or 60-day time limits and a 75% split.
- Goat Funded Trader: a -2% floating-loss auto-close.
- FundingPips: a 1% floating-loss rule on funded accounts.
- Blue Guardian: Guardian Shield.
- Alpha Capital Pro 8%: rules found (8% / 5%, 4% daily of the starting balance, 8% max, 1:100), but no reliable prices.
- E8 One: configurable, with a 40% consistency rule on funded accounts.
- Futures firms: must be flat intraday.
- Instant-funding accounts: trailing drawdowns plus consistency rules; terms not verified.

## Leverage caps (both pairs together, as a multiple of the account)

An 80 bp stop-out costs about lev × 0.8%. A gap through the stop costs about lev × 1%. The worst stopped trade in
development was -97 bp in 2007-2018 and -81 bp in 2019-2025.

- **FTMO 2-Step: 5x.** A stop-out costs 4% against the 5% daily limit. At 6x, 2007-2018 has 2 nights that breach.
- **3%-daily programmes (FTMO 1-Step, FundedNext 1-Step): 3x.**
- **FundedNext: 3.5x.** This keeps the open risk at 2.8%, under the 3% rule.
- **The5ers: 4x.** Server midnight falls inside the trade, so an open profit at midnight raises the reference point for
  the daily loss. On 2007-2018 nights (`scripts/prop_checks.py`, `docs/prop_best/checks.json`), The5ers' way of
  measuring gives 2 breaches at 4.5x and at 5x (0 in the simulator's measure) and 0 at 4x.

## Results at the best leverage of each programme (100k, or the nearest size)

Paid = the share of attempts that receive at least one payout. EV/fee = mean net per attempt divided by the fee.

| Programme, leverage | Fee | Paid: 2026 / 2019-25 / zero | Mean net per attempt: 2026 / 2019-25 / zero | EV/fee (2026) | Paid within 6 months (2026) | Median business days to 1st payout (2026) |
|---|---|---|---|---|---|---|
| **FTMO 2-Step, 5x** | $621 | **45% / 80% / 32%** | **+$6,108** / +$12,187 / +$2,792 | **9.8** | 12% | 181 |
| FTMO 1-Step, 3x | $574 | 43% / 79% / 29% | +$3,537 / +$7,839 / +$1,492 | 6.2 | 11% | 178 |
| FundedNext Stellar 2-Step, 3.5x | $550 | 33% / 67% / 23% | +$3,356 / +$7,579 / +$1,206 | 6.1 | 1% | 249 |
| FundedNext Stellar 1-Step, 3x | $570 | **46%** / 82% / 30% | +$3,336 / +$7,608 / +$1,420 | 5.9 | **14%** | 177 |
| FundedNext Stellar Lite, 3.5x | $400 | 42% / 82% / 30% | +$3,532 / +$8,120 / +$1,616 | 8.8 | 7% | 181 |
| The5ers High Stakes, 4x | $405 | 33% / 82% / 23% | +$3,678 / +$8,968 / +$1,528 | 9.1 | 2% | 213 |
| The5ers High Stakes Classic, 4x | $455 | 41% / **84%** / 25% | +$4,328 / +$9,690 / +$1,959 | 9.5 | 9% | 179 |

- **Size.** The chances do not depend on size. The mean net scales with size, and so does the fee. FTMO 200k: fee
  $1,242, +$12,215 at the 2026 edge.
- **The median attempt loses the fee in every programme at the 2026 edge.** Even at the 2019-2025 edge, 16-33% of
  attempts lose it.
- **About 35-45% of the mean is not the edge.** With a zero-edge trade, every programme still shows a positive mean,
  because the loss is capped at the fee while profits are paid out. That depends on the firm honouring its payouts.
- **FundedNext 2-Step pays less often than it passes** (67% vs 83% at the 2019-2025 edge). Funded accounts that fall
  behind early stay under water for the whole simulated funded year.

**Cheapest tickets** (2026 edge; "fees per payout" = fee / P(paid), the expected fees spent until one attempt is
paid, if attempts are independent):

| Ticket | Fee | Paid | Mean net | Fees per payout |
|---|---|---|---|---|
| The5ers High Stakes 2.5k, 4x | $19 | 33% | +$86 | $57 |
| The5ers High Stakes Classic 2.5k, 4x | $22 | 41% | +$102 | $54 |
| FundedNext Stellar Lite 5k, 3.5x | $32.99 | 42% | +$168 | $79 |
| FundedNext Stellar Lite 10k, 3.5x | $59.99 | 42% | +$339 | $144 |
| FTMO 2-Step 10k, 5x | $102 | 45% | +$589 | $228 |

## Answer

- **Highest gain and the best chance of being paid: FTMO 2-Step at 5x.** It has the highest mean per attempt and the
  highest mean per dollar of fee. It is in the top group for the chance of being paid (45% at the 2026 edge, 80% at
  the development edge), and the fee is refunded with the first reward. Its daily loss is measured from the balance,
  and it has no risk-per-trade cap. Choose the **Swing** account type to be exempt from the funded news rule. Size it
  to the budget: 100k (€540) or 200k (€1,080) have the same mean per fee.
- **Lowest cost:** The5ers High Stakes Classic 2.5k ($22) or FundedNext Stellar Lite 5k ($33). Each is a ticket with a
  41-42% chance of a small payout (about $100-200 expected). Payouts from them can fund an FTMO attempt later.
- **Fastest:** no programme is fast with a trade that fires about 5 nights a month. The best are FundedNext Stellar
  1-Step at 3x and FTMO 2-Step at 5x: only 12-14% of attempts are paid within 6 months, and the median for those paid
  is about 8.5 months. More leverage would be faster, but one gap through the stop then breaks the daily limit.
- **"Always a win" is not possible.** The best chance of a payout on one attempt is about 45% at the 2026 edge and
  about 80% at the development edge. Sequential attempts on later nights are roughly independent. If they are, three
  FTMO 10k attempts (€267 at most) give a 1 - 0.55^3 = 83% chance of at least one payout at the 2026 edge (68% at zero
  edge).

## How to run it

1. **Entry.** On the day before each Gotobi day (5th, 10th, 15th, 20th, 25th, month end; the previous Tokyo business
   day if those fall on a weekend or holiday), at **14:06 UTC** buy USD/JPY and EUR/JPY with equal notional. Skip
   Gotobi Mondays.
   - Six minutes after the tested 14:00 keeps the entry clear of the 10:00 New York releases in US summer time, which
     the firms' news rules cover.
   - On 2007-2018 nights a 3-15 minute delay was no worse: +0.2 to +0.8 bp, t 0.5-2.3 (`checks.json`).
2. **Stop and exit.** Place a resting stop 80 bp below each entry. For example, at USD/JPY 150.00 the stop is at
   148.80. Close both pairs at **00:53 UTC** (09:53 Tokyo).
3. **Size** (FTMO 100k at 5x): $500k notional in total.
   - 2.5 lots of USD/JPY, plus EUR/JPY lots worth $250k (about 2.2 lots at EUR/USD 1.15).
   - A stop-out costs about $4,000 (4%).
   - At 1:30 (Swing), the margin is about 17% of the account.

## Caveats

- **Edge.** The edge is small. It was positive but not significant in 2026, and the forward test only starts on
  2026-11-05. The 2026 and zero rows show what happens if it shrinks or vanishes.
- **Model limits.** The model ignores scaling plans, discounts, payout denials, rule changes, and the firm's
  counterparty risk (prop firms have closed and stopped paying before).
- **Sources.** Rules and prices were taken from the firms' pages and third-party summaries on 2026-11-02. Check the
  current agreement before buying.
- Not financial advice.

## Sources

- FTMO: https://ftmo.com/en/trading-objectives/ ; https://proptradingvibes.com/blog/ftmo-account-sizes-pricing ;
  https://tradetanto.com/learn/ftmo-rules-evaluation-process
- FundedNext: https://fundednext.com/package-comparison ; https://proptradingvibes.com/de/blog/fundednext-pricing ;
  https://fundednext.com/api_docs/cfds-glossary/daily-loss-limit ; https://help.fundednext.com/en/articles/14702245 ;
  https://ptv-dev.pages.dev/blog/fundednext-risk-limit-rule
- The5ers: https://tradetanto.com/learn/the-5-ers-rules-explained-a-complete-guide ; https://propfirmmap.com/firms/the5ers ;
  https://fundedtrading.com/the-5-ers-announces-price-updates-for-high-stakes-accounts
- Alpha Capital, E8: https://tradetanto.com/learn/alpha-capital-group-rules-what-traders-need-to-know ;
  https://tradetanto.com/learn/e8-markets-rules
- FXIFY: https://huggingface.co/datasets/propfirmkey/fxify-discount-code-pfk-28-off/blob/main/data/fxify_rules.csv
- Comparisons: https://velotrade.com/blog/prop-firm-comparison ; https://velotrade.com/blog/cheapest-prop-firm ;
  https://www.quantvps.com/blog/forex-prop-firms-that-pay ; https://brokeranalysis.com/prop-trading-firms/compare-challenges/ ;
  https://forextester.com/blog/cheapest-prop-firm-challenge/

## Second search (2026-11-02): Maven, Blueberry and others

**Added to the simulation** (`scripts/prop_best.py`):
- **Maven 2-Step:** 8% / 5% targets; 4% daily loss, measured from the higher of balance and equity at 00:00 UTC (inside the trade), so leverage is capped at 3.5x; 8% static max loss; 80% split; payouts every 10 business days; withdrawals capped at $10k per 30 days. Prices with the 10% code: 5k $19.80, 50k $198, 100k $396.
- **Blueberry Prime 2-Step:** 8% / 6% targets; 4% daily loss; 10% static max loss; 5 trading days a phase; 80% split; 14-day payout cycle; fee refunded with the 1st payout. 10k $90, 100k $650.

| Programme, leverage | Best size by mean per fee | Paid at the 2026 edge | Mean per fee: 2026 / 2019-25 |
|---|---|---|---|
| Maven 2-Step, 3.5x | 50k ($198) | 33% | 8.8 / 19.7 |
| Blueberry Prime 2-Step, 4x | 200k ($1,170) | 33% | 7.0 / 15.3 |

- Neither beats FTMO 2-Step at 5x (9.8 / 19.6).
- Maven 5k ($19.80) is the most account per dollar: 33% paid, mean +$175.
- FundedNext Stellar Lite 5k ($32.99, 42% paid) still has the best chance of being paid among the tickets under $35.

**Read but not simulated:**
- **BrightFunded:** no prices found. Its daily loss is measured from the higher of balance and equity at the 23:30 CET rollover, which falls inside the trade. The first payout comes 30 days after funding.
- **FundingPips, Goat, Sure Leverage, FundYourFX:** floating-loss rules, mandatory stop-loss timing, or consistency rules on funded accounts.
- **Instant-funding accounts:** trailing drawdowns of 3-4% and 2% daily limits, too tight for an 80 bp stop at any useful leverage.

## Practical checks on 2007-2025 and several firms at once (2026-11-03, CI run 41651787391)

Script: `scripts/prop_checks.py`. Results: `docs/prop_best/checks.json`.

- **The5ers midnight rule.**
  - 2019-2025: no night breaches a 5% daily loss at 3x-6x.
  - 2007-2018: 2 nights would at 4.5x-5x, and 7 at 6x. The simulator's own measure gives 0 at 5x.
  - The 4x cap stands.
- **Maven, daily loss from equity at 00:00 UTC.** Testable on 2019-2025 only (419 nights). No breach of the 4%
  limit at 3x, 3.5x or 4x.
- **Later entry.** Gross gain over the 14:00 entry, per night:

  | Delay | 2007-2018 | 2019-2025 |
  |---|---|---|
  | +3 min | +0.8 bp (t 2.3) | +0.4 bp (t 1.4) |
  | +6 min | +0.3 bp (t 0.8) | +0.5 bp (t 1.6) |
  | +10 min | +0.3 bp (t 0.6) | +0.6 bp (t 1.6) |
  | +15 min | | +0.0 bp |

  - No delay costs anything, so the 14:06 entry (which keeps clear of the 2-5 minute news rules) is safe.
- **US release at 14:00 UTC.** 49 of 375 Gotobi nights in 2017-2023 have one. Those nights average +2.9 bp, against
  +5.9 bp on the others.
- **Three accounts on the same nights** (`scripts/prop_joint.py`): FTMO 2-Step 5x, FundedNext Stellar 1-Step 3x,
  The5ers High Stakes Classic 4x. Fees $1,646 in total.

  | Edge | P(at least one paid) | P(all three paid) | Median total net |
  |---|---|---|---|
  | 2019-2025 edge | 87% | 79% | +$15,013 |
  | 2026 edge (+2.0 bp) | 47% | 41% | -$1,646 (all fees lost) |
  | Zero edge | 32% | 24% | -$1,646 |

  - The accounts move together: when one fails, the others usually fail too. Three firms raise the payout per good
    year but do not diversify the risk of a bad one.

## Real costs and the scaling question (2026-11-03, web research; figures change, check the platform)

**Costs found:**
- **Commission per round lot:**
  - FTMO and FundedNext: $5, which is about 0.5 bp of notional.
  - The5ers: $4, about 0.4 bp.
  - The model charges 0.7 bp, so it is slightly too expensive here.
- **Swaps** (retail, long, early November 2026):
  - USD/JPY: OANDA +1.66% a year (about +0.45 bp a night); Afterprime about zero; HFM 0.
  - EUR/JPY: OANDA +0.21% a year; HFM -30 JPY a lot; XM -545 JPY a lot.
  - FTMO, FundedNext and The5ers publish swaps only in the platform. All three charge swaps; The5ers has no
    swap-free account.
  - The model's average swap on 2019-2025 nights is about +0.25 bp. A real swap of 0 to -0.5 bp is plausible, the
    same size as the commission saving.
- **Spreads:** no firm publishes USD/JPY or EUR/JPY figures; they are variable. The model uses Dukascopy's quoted
  spread x 1.25 + 0.1 pip at 14:00 UTC and 00:53 UTC. Measure the firm's live spread at those two times before
  trading.
- **Net effect:** within about ±0.5 bp of the model. The +1 bp stress (2019-2025: +4.0 bp, t 2.8) covers it.

**Scaling plans:**
- **The plans:**
  - FTMO: +25% every 4 months, up to $2M.
  - FundedNext: +40% every 4 months, up to $4M.
  - Both require 10% profit in the 4 months and 2 payouts.
  - The5ers High Stakes: needs a 10% target, and its ceiling is $500k ($4M is Hyper Growth only).
- **What the trade makes in 4 months** (real 2019-2025 nights, rolling windows, compounded):

  | Leverage | Median 4-month gain | P(window >= 10%) | Same, at the 2026 edge |
  |---|---|---|---|
  | 3x | 2.7% | 9% | 2% |
  | 4x | 3.6% | 21% | 12% |
  | 5x | 4.4% | 26% | 20% |

  - Scaling from $100k to $1M at FundedNext takes 7 consecutive qualifying reviews. That is close to impossible
    with this edge.
- **Initial allocation caps:** FTMO $400k per trader or strategy (all accounts combined, before scaling);
  FundedNext $300k. Copying the same strategy across accounts at one firm is policed.
- **Conclusion:**
  - Prop scaling plans are built for 10% per 4 months, about 30% a year. The edge earns about 9-15% a year at
    3-5x on the 2019-2025 edge, and 3.5-6% at the 2026 edge.
  - The realistic route to $1M of buying power is the initial caps at several firms: FTMO $400k + FundedNext
    $300k + others. Those accounts fail together.
  - The route to $1-4M of own capital is a real-money track record and investors. Liquidity is no limit: USD/JPY
    absorbs $4M without slippage worth measuring.

## Third search (2026-11-04): every CFD programme in LuxAlgo's public rules directory

**Data.** LuxAlgo's keyless directory API (the data behind luxalgo.com/prop-firms, fetched by
`ci/misc_propfirms.sh`): 27 firms, 520 programmes; 419 CFD programmes allow overnight holding. FundedNext is not in
the directory (covered above). Code: `scripts/prop_scan.py` (the directory fields mapped as in LuxAlgo's own
adapter, github.com/LuxAlgo/prop-firm-sim); results `docs/prop_best/prop_scan_best.csv`. The frozen basket's real
2019-2025 nights, development edge and 2026 edge (+2.04 bp), leverage 2-8x up to each programme's cap (daily
limit / 1.2% where the daily reference is the end-of-day equity inside the trade, / 1.0% where it is the balance).

**Directory errors and exclusions found on the way:** FTMO 1-Step is listed at $90 for every size (real: about
€79-999), so it is left out; Darwinex Zero is a different model (allocation from investors, performance fee), left
out; instant-funding programmes look fastest in the raw scan (first payout within days) but the ones checked are
unusable for this trade: FundingPips Zero caps the total floating loss at 1% and forbids holding through high-impact
news (the basket is open 14:00 -> 00:53 UTC, through US and Japanese releases); Lark and Funded Trading Plus instant
accounts cost $4,500 for 100k (Lark's price confirmed on three review sites; Lark re-run on its own rules below:
positive on average, but the fee is about a year of the trade's 2026 profit).

**Verified on the firm's terms and re-simulated (6,000 attempts, `docs/prop_best/prop_focus_lark_ftmo.csv`):**
Lark Funding 1-Stage: 10% target, 5% daily loss from the **balance** at 5 PM New York, 6% static max loss, no
minimum days, news and overnight allowed, weekends not (the basket never holds a weekend), FX leverage 15:1, 80%
split, first payout 30 days after funding, then every 14.

| Programme (100k), leverage | Fee | Paid: 2026 / dev edge | Mean net per attempt: 2026 / dev | Mean / fee (2026) | Paid within 6 months: 2026 / dev | Median business days to 1st payout (2026) |
|---|---|---|---|---|---|---|
| **Lark Funding 1-Stage, 5x** | $500 | **46% / 73%** | **+$6,156 / +$11,118** | **12.3** | **20% / 31%** | **141** |
| Lark Funding 2-Stage, 6x | $550 | 41% / 77% | +$7,929 / +$13,614 | 14.4 | 13% / 23% | 165 (16% daily-limit failures) |
| FTMO Swing 2-Step, 5x | €540 | 44% / 85% | +$5,148 / +$12,152 | 8.3 | 10% / 20% | 177 |
| Lark Funding 3-Stage (no daily limit), 6x | $370 | 27% / 39% | +$4,993 / +$9,757 | 13.5 | 13% / 18% | 140 |

**Answer.** For the most money per fee and the fastest first payout with this trade: **Lark Funding 1-Stage 100k at
5x** (about twice FTMO's chance of a payout within six months, the same overall chance of being paid, 50% more
expected profit per dollar of fee). FTMO Swing at 5x stays the most established firm (founded 2015, Trustpilot 4.8,
now owner of OANDA's prop business) against Lark (founded 2022, 4.5). Nothing is fast: with about 5 trading nights
a month, the median first payout is about 6.5 months at the 2026 edge. Running both at once raises the chance of
at least one payout, but the accounts trade the same nights and fail together.

### Instant funding with this trade (2026-11-04)

The instant accounts checked on the firms' own pages share one design: 3% daily loss counting open losses, 5-6%
max loss trailing the high, a 1% cap on the total floating loss (Funding Traders, FundingPips Zero, Blue Guardian
Starter), a 15-20% best-day consistency rule before any payout, 5-7 profitable days of at least +0.5% (For
Traders, Blue Guardian), a 3% non-withdrawable cushion (Funding Traders, FundingPips), and no opening or closing
within 5-10 minutes of red-folder news.

- The 1% floating cap with the 80 bp stop allows at most 1.25x; at 1.25x only 8% of nights (2026 edge) reach the
  +0.5% that counts as a profitable day, so the 5-7 profitable days take about a year.
- Simulated on the 2019-2025 nights with Funding Traders' rules (cushion, 15% consistency, trailing 6% locking at
  +3%, on-demand payouts, $500 fee assumed for 100k): at 1.25x, **27% paid within a year at the 2026 edge, median
  first payout after about 9 months, mean -$229 per attempt** (dev edge: 46%, +$512). Even 2x, which breaks the
  floating cap, gives +$542 at the 2026 edge.
- Instant accounts are built for many small daily profits; a trade that earns on 5 nights a month cannot meet their
  payout conditions quickly. They are slower and less profitable here than Lark's 1-Stage evaluation.

### Lark Funding Instant, on its published rules (2026-11-04)

Checked on thetrustedprop.com, thegodfunded.com (rules and the 2026-03-02 drawdown change), propfirmmap.com (data of
2026-05-18) and luxalgo.com; Lark's own help centre could not be opened from here. Rules: **no profit target, no
minimum days, no consistency rule**, first payout on demand then every 30 days, $100 minimum, 90% split (older
pages 80%), $40 payout fee (Rise, waivable), daily loss 5% of the initial balance reset at 5 PM New York, **max loss
8% trailing the highest closed balance, locking at the initial balance once 8% up "or after the first payout"**,
stop loss required, weekends need an add-on (the basket never holds one: its exits are Tuesday-Friday 00:53 UTC),
news allowed, leverage 1:15 or more. Price: $5k $200, $10k $400, $25k $1,125, $50k $2,250 (one site $2,750),
$100k $4,500. The pages do not say what the floor does after a payout, so three readings were run:
**strict** (after the first payout the floor is the starting balance: the only cushion is the profit left in),
**gentle** (the floor freezes where it was), and **reset** (a fresh 8% trail after each payout: the earlier
directory scan's assumption, the most generous). Code `scripts/lark_instant.py`; one funded year (250 business
days) from a random start in the real 2019-2025 nights; the payout policy (take the first payout at +T, then every
21 business days withdraw everything above K) chosen on the development edge only; results
`docs/prop_best/lark_instant.csv`.

| Lark Instant 100k ($4,500), 4x, one funded year | Paid at least once: 2026 / dev | Mean net: 2026 / dev | Median net (2026) | Chance net > 0 (2026) | Median business days to 1st payout | Mean cash in first 3 months (2026) |
|---|---|---|---|---|---|---|
| gentle (first payout at +0.5%, then withdraw all) | 87% / 98% | **+$2,025 / +$5,603** | -$871 | 46% | 9 | $2,369 |
| strict (first payout at +10%, keep 4%) | 42% / 53% | +$66 / +$2,638 | -$4,500 | 42% | 110 | $334 |
| reset (information) | 71% / 94% | +$2,588 / +$6,535 | -$150 | 50% | 26 | $2,406 |
| *Lark 1-Stage, 5x ($500), one funded year after passing* | *46% / 73%* | *+$6,156 / +$11,118* | | | *141* | |

- **The trade fits Lark Instant's rules** (the earlier "instant does not work" came from other firms' 1% floating
  caps, consistency rules and profitable-day counts, which Lark does not have). It is the fastest cash of any
  option found: under the gentle reading the first payout comes after about 9 business days.
- **It is not the better use of money.** The fee is 4.5% of the account; at 4x the trade earns about 4.9% a year at
  the 2026 edge (2.04 bp x 60 nights x 4), so a year roughly buys the fee back: mean +$2,025 (gentle) or about
  zero (strict) on $4,500 at risk, and in more than half of the attempts the fee is not fully recovered. The
  1-Stage puts $500 at risk for a higher mean (+$6,156) with one funded year after passing.
- Kept longer, the instant account pays more (2026 edge, gentle reading: mean +$5,728 and median +$2,728 over two
  years; +$7,353 / +$3,070 over three), but most accounts are breached by then (alive after three years: 8%).
- 5x (if the daily limit is from the previous day's closed balance, as one page says): gentle +$2,914, strict
  +$970 at the 2026 edge. Smaller accounts are worse per dollar ($40 payout fee, $100 minimum): $25k gentle
  +$394 at the 2026 edge. Below 2x (Lark's "ideally 1.5% risk per trade" guideline; 4x with the 80 bp stop risks
  3.2%) the instant account loses money at the 2026 edge in every reading.
- What decides it, to ask Lark before buying: "On an Instant account, where is the max-loss floor after the first
  payout: the starting balance, or does it stay 8% below the highest closed balance?" Under the strict answer the
  instant account is break-even at the 2026 edge.
