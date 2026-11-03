# Ledger: everything raised so far and its status (2026-11-03, 14:20 UTC)

Status words: **done** (result in the doc named), **running**, **written** (rules fixed, code not yet built),
**open** (raised in the discussion, not yet written as rules), **untestable** (no data), **user** (needs an action
from the user).

## 1. The Gotobi trade

| Item | Status | Where |
|---|---|---|
| Stops, targets, exits, entries, filters, indicators, losing nights | done | `docs/untested_batch.md` 9.14-9.18 |
| Panel, 2007-2018 screen (34 instruments) | done | 9.19 |
| **Panel, 2019-2025 check, indices / bonds / commodities** | **open**, waiting for the last 4 bond and commodity downloads; the panel CI has not been triggered | `scripts/gotobi_panel_lab.py` |
| **New York-leg hedges and pre-entry signals (H, F)** | **written**, code built (`scripts/panel_hedge_lab.py`), waits for the bond data | `docs/panel_hedge_preregistration.md` |
| Sizing by filter probability (E1) | done: P(paid) 60% vs 49%, fails its own rule; forward line FW1 | `docs/edges_batch/e1.json` |
| Cheaper limit entry (E2), short after the fix (E13), Japan's flows (E3) | done, nothing | `docs/edges_batch_results.md` |
| **Entry at 14:06 instead of 14:00 checked on 2019-2025** | **open**: checked on 2007-2018 only (`scripts/prop_checks.py`) | - |
| **Skip the 25th and month-end** | **open**: can only be judged forward; not yet a forward-report line | - |
| **USD/SGD, USD/CNH, USD/HKD as Asian dollar carriers** | open: in the panel check above (the USD/CNH, USD/SGD files have gaps) | - |
| Forward test | running: first nights from 2026-11-05, monthly report on the 3rd | `docs/forward_test_preregistration.md` |

## 2. Prop firms and execution

| Item | Status |
|---|---|
| Seven programmes simulated, plus Maven and Blueberry; leverage caps; The5ers midnight rule | done (`docs/prop_best.md`) |
| Drawdown-scaled leverage (E12); gold calendar next to Gotobi (E17) | done, neither helps |
| **Maven's daily loss from equity at 00:00 UTC (inside the trade)** | **open**: the 2007-2018 files have no 00:00-01:00 UTC bars; testable on 2019-2025 |
| **Several firms at once: joint outcome of correlated accounts** | **open**: discussed, not simulated |
| **Which Gotobi nights have a high-impact US release at 14:00 UTC** | **open**: the calendar dataset (2017-2023) can list them |
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
| **Emerging-market currency files with gaps (2013-2015, after 2022)** | **open**: re-download would let E4 / E16 run on full data |
| **Japanese government bond auctions** | **open**: calendar not downloaded, rules not written |
| **FOMC in the expected direction (Karnaukh)** | **open**: about 20 decisions, information only, not run |
| Retail positioning (OANDA) | **user**: needs a free OANDA practice token as the `OANDA_TOKEN` secret |
| Platinum, palladium | skipped (no mechanism) |
| Gold breakout short side | forward only (chosen after seeing it) |

## 4. ML and features

| Item | Status |
|---|---|
| **M1 weekly pooled FX model; M2 cost-aware filters on before-cost edges; M3 cross-asset intraday features; M4 pooled intraday model** | **written**, not built |
| **FS1 feature settings and forms chosen inside the walk-forward (H8, D1, weekly); FS2 by session and volatility; FS3 new intraday sources** | **written**, not built |
| **Feature predictability on 2019-2025** | open: part of FS1 |
| **Volatility or large-move targets** (instead of the next return) | open |
| **Systematic pairs of features (interactions)** | open |
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
