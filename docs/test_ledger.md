# Ledger: everything raised so far and its status (2026-11-03, 16:30 UTC)

Status words: **done** (result in the doc named), **running**, **written** (rules fixed, code not yet built),
**open** (raised in the discussion, not yet written as rules), **untestable** (no data), **user** (needs an action
from the user).

The queue of what is still worth testing is in `docs/testing_list.md`.

## 1. The Gotobi trade

| Item | Status | Where |
|---|---|---|
| Stops, targets, exits, entries, filters, indicators, losing nights | done | `docs/untested_batch.md` 9.14-9.18 |
| Panel, 2007-2018 screen (34 instruments) | done | 9.19 |
| Panel, 2019-2025 check | done: USD/JPY, EUR/JPY (and HKD/JPY, the same trade) confirmed; wider baskets dilute it; B0 stays | `docs/edges_batch_results.md` |
| New York-leg hedges and pre-entry signals (H, F) | done: none counts (bond hedge cuts sd 17% but also the mean); bond data incomplete | `docs/edges_batch_results.md` |
| Sizing by filter probability (E1) | done: P(paid) 60% vs 49%, fails its own rule; forward line FW1 | `docs/edges_batch/e1.json` |
| Cheaper limit entry (E2), short after the fix (E13), Japan's flows (E3) | done, nothing | `docs/edges_batch_results.md` |
| Entry at 14:06 instead of 14:00 checked on 2019-2025 | done: +0.5 bp vs 14:00 (t 1.6), no cost | `docs/prop_best.md` |
| Skip the 25th and month-end | forward-report line FW2 (exploratory) | `scripts/forward_eval.py` |
| USD/SGD, USD/CNH as Asian dollar carriers | done: neither is a carrier (USD/SGD -2.6 bp, USD/CNH -3.3 bp net on 2019-2025) | panel |
| Forward test | running: first nights from 2026-11-05, monthly report on the 3rd | `docs/forward_test_preregistration.md` |

## 2. Prop firms and execution

| Item | Status |
|---|---|
| Seven programmes simulated, plus Maven and Blueberry; leverage caps; The5ers midnight rule | done (`docs/prop_best.md`) |
| Drawdown-scaled leverage (E12); gold calendar next to Gotobi (E17) | done, neither helps |
| Maven's daily loss from equity at 00:00 UTC | done on 2019-2025 (419 nights): no breach at 3-4x |
| Several firms at once | done: all three paid 79% (2019-2025 edge), 41% (2026 edge); the accounts fail together |
| Gotobi nights with a US release at 14:00 UTC | done: 49 of 375 (2017-2023), +2.9 bp vs +5.9 bp |
| FTMO Swing account price and availability; Alpha Capital, BrightFunded prices | open (prices not found; check on the firms' sites) |
| The5ers Hyper Growth ($4M scaling), instant-funding accounts | not simulated (rules too tight for the trade, or not verified) |

## 3. New edges (edges batch)

| Item | Status |
|---|---|
| E2-E4, E6-E11, E13-E16, E18, E19, E23, E24 | done, nothing counts after correction |
| E20 confidence curve, E22 timeframe agreement, E21 catalog on 2019-2025 | done, nothing counts; USD/JPY H4 at the 98% cut-off is a lead (+6.9 bp, t 3.3 on 2019-2025, flat before) |
| E5 FX option expiries | untestable (DTCC history) |
| BoE meetings, ISM surprises, Akshaya Tritiya, TRY carry, gold positioning | untestable (no data in the free sources found) |
| **Payroll and retail-sales surprises** | done but too few events (fewer than 10 with abs(z) >= 1 in 2017-2023) |
| Emerging-market files with gaps | done: re-downloaded; E4 / E16 rerun on full data, neither counts |
| Japanese government bond auctions (E25) | done: MoF history file (659 auctions); both windows negative on 2019-2025 |
| FOMC in the expected direction (E26) | done: 18 decisions, -18.6 bp (t -1.4), information only |
| Retail positioning (OANDA) | **user**: needs a free OANDA practice token as the `OANDA_TOKEN` secret |
| Flow, volume, positioning batch V1-V10 (futures / ETF volume, CFTC TFF and disaggregated, SNB deposits, GDELT tone, 2y yield gaps, fed funds futures, MoF interventions) | done: nothing counts; futures volume and GLD holdings unusable from free sources; V6 US-euro gap is a watch-list lead (`docs/edges_batch_results.md`) |
| Daily forward-only data (DTCC option trades, Saxo option vols / risk reversals, GDELT, GLD) | collecting every weekday (`flow-collect`); testable after about a year |
| Platinum, palladium | skipped (no mechanism) |
| Gold breakout short side | forward only (chosen after seeing it) |

## 4. ML and features

| Item | Status |
|---|---|
| M1 weekly pooled model; M2 cost-aware meta-labels; M4 pooled intraday model with the FS5 large-move filter | done, none counts (`docs/edges_batch_results.md`, addenda 3-5) |
| M3 cross-asset intraday features | done: slightly better, still negative (-1.1 to -1.6 bp) |
| FS1 settings and transforms chosen inside the walk-forward | done: 66 of 58,920 meet the rule on 2019-2025, but they lose on 2010-2018 and cluster in a few GBP / NZD crosses; forward watch list only |
| FS2 session / volatility cells | done: 0 after the non-overlapping fix |
| FS3 new intraday sources | done: 9 significant ICs (cross-asset), none tradeable after costs |
| Feature predictability on 2019-2025 | done in FS1 |
| Volatility or large-move targets (FS5) | done: trading rule no count (M4); 59 significant ICs, regime-like (yen MAs vs volatility) |
| Systematic pairs of features (FS4) | done: 0 of 9,120 |
| Deep sequence models | not planned (Chronos failed; the limit is the information, not the model) |

## 5. Outside forex prop accounts (personal account)

| Item | Status |
|---|---|
| Index option premium selling (short variance) | tested earlier: the strongest premium; not set up as a plan |
| Crypto cash-and-carry | tested earlier (+0.69% a month 2019-2025); not set up |
| Real-money track record (Myfxbook / FX Blue) | **user** |

## 6. Data and infrastructure

| Item | Status |
|---|---|
| Mixed-source M1 files (missing spread columns) | fixed (`fxlab/data/store.py`) |
| API rate limit in parallel jobs | fixed (`m1-stage`, one artifact per symbol); bonds to be staged once downloaded |
| Local test suite: 96 tests fail only because `hmmlearn` is not installed here | open (environment) |
