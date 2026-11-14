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

## 7. Rounds 9-12 (2026-11-05 to 2026-11-08)

| Item | Status |
|---|---|
| Round 9 gaps (forced flows, premia, news, options, BoJ / Norges flows, CME futures volume and order flow) | done, nothing counts (`docs/round9_results.md`) |
| MS1 Malaysian SNR, QM1 Quasimodo, CL1-CL7 (Dual Thrust, R-Breaker, Fiali, Aberration, King Keltner, Thermostat, Bollinger Bandit, Camarilla, Elder, Puria) | done, 0 of 19 count (`docs/anatomy_results.md`) |
| Why each failed strategy fails, and a fix search (entry, exit, reversal, one filter) per strategy | done: 151 of 197 have no gross edge, 25 are cost-bound, 14 decayed, 5 are skew traps; 0 of 196 fixes count (`docs/anatomy_results.md`) |
| Retail positioning (OANDA order / position books) | **user**: needs a v20 API token as the `OANDA_TOKEN` secret |
| Long trades that let winners run: Turtle 1 / 2, Clenow, Donchian 100 / 50, MA 50 / 200, time-series momentum, Faber, EWMAC, squeeze with trailing stop, weekly breakout, carry-filtered EWMAC; the catalog on D1 / W1 with trailing, channel, breakeven and partial exits | done: 0 of 23 part-A rows count; 1 of 109 part-B units (Double Seven W1) counts by its rule but lost in 2008-2018, forward watch only (`docs/long_results.md`) |
| Round 14: independent re-check of round 13 (second engine, ECB fixings 1999-2025, fill at the close); 23 trend speeds, long-or-flat, currency-strength indices, trend + carry, high-volatility trend; direction and trendiness prediction | done: round 13 reproduced; 0 of 26 rows count; no trendiness predictor works (`docs/trend14_results.md`) |
| Double Seven on weekly bars, 2026 holdout (third use, ECB fixings, FX) | supported: 33 trades, +0.78R, t 5.4 clustered; crash-prone (2008-2010, 2015-2016); forward test next |
| Round 15: every rule (91, incl. 26 new book systems) and feature (315 columns) at D1 / W1 / MN, full grids, both directions, 33 instruments | done: 0 of 1,071 units survive BH; placebos 0 of 150 |
| Weekly Connors dip-buying family on 2026 (7 rules, published defaults) | supported: 175 trades, +0.38R, t 2.2; forward-test track D7-family |
| Free data: Forex Factory surprises (drift 60 m / 1 d, surprise index), CBOE option-selling indices, 2y rate momentum | done: 1-day surprise drift +5.4 bp t 2.2 (2019-2025) fails the correction; the rest fail |
| Free data: commodity / index / bond CFDs with the round-13 trend systems | done: fails (0 of 33 rows; 2019-2025 t at most 0.53); no D row passes BH (`docs/round15_results.md`, D2) |
| Round 16: regime detection and the best strategy per regime (strategy momentum, performance filter, k-means / Gaussian-mixture clusters with 2-8 regimes, single-feature splits, HMM, ridge / boosting), over 4,140 configurations, 2,258 units and 9 FX styles; 33 regime features incl. VIX family and CME euro / yen IV and risk reversals | done: 0 of 18 count; none beats static selection or its placebo; no number of regimes beats none (`docs/round16_results.md`) |
| Options data as regime inputs, timing signals (117 tests) and a dip-buying filter | done: nothing counts; EVZ no longer published; more CME FX options priced at USD 60.29, not bought |
| Round 17: regimes with intraday strategies (H4 / H1 / M30 / M15, 3,648 configurations), daily / weekly / monthly switching, per-instrument regimes (33 instruments), Hedge, CUSUM resets, pooled boosting, intraday-state and daily CME features | done: 0 of 93 count; 0 of 486 long-only timing tests; a 14-hour fill-to-fill look-ahead in the feature-tercile units was found by a lag check and fixed (`docs/round17_results.md`) |
| Round 18: second-sample check (2019-2025) of the cost-bound published effects: gold Asian / Western hours, gold and silver fixes, home hours, around the clock, fix reversal | done: gross edges persist (gold Asian hours +2.7 bp, t 2.5), all lose after costs; the gold anomaly is gone (`docs/round18_results.md`) |
| Round 18: overnight-to-intraday currency reversal; VWAP trend on US500 / NAS100 CFDs (any minute, and every 15 / 30 / 60 minutes) | done: reversal decayed; VWAP trend real before costs (NAS100 +12.5 bp a day, t 4.7), negative after |
| Round 18: feature-free test for performance regimes (variance ratios against a time-shuffle null) | done: none in D1 / W1 / MN strategies; intraday persistence is the cost regime and never profitable |
| Round 18: 1976-2025 styles and regime detector (Bank of England data); global FX volatility and the dollar | done: trend and carry paid until about 2005; weekly dip-buying Sharpe -0.15 over 50 years; detector does not beat static selection; volatility-dollar t 1.25 |
| Round 19: index rules at CME futures costs (ORB5, noise-area breakout, VWAP trend), dip-buying in the carry direction, oil around the EIA report, crypto funding times, prior-day stop runs (22 tests) | done: 0 count; noise-area breakout on NAS100 meets its own rule (+3.3 bp a day 2019-2025, t 2.25; +1.65 in 2012-2018) but fails BH (`docs/round19_results.md`) |
| Round 19: the strategy library on 13 commodity / index / bond CFDs and BTC / ETH perpetuals (21,000 units) | done: 0 pass BH; median 2019-2025 t negative on every instrument |
| Noise-area breakout (NAS100, NQ futures costs), 2026 check (fourth use of the holdout; one pre-registered test, power ~20%) | inconclusive: +2.64 bp a day, t 0.73 over 153 days; consistent with 2019-2025; weekly forward test from 2026-11-16 |
| Round 20 S: Kullamaegi breakout, episodic pivot, Minervini trend template on point-in-time S&P 500 stocks, daily, 2005-2025 | done: 0 of 3 count; breakout and template lose 0.5-1.6% per trade against SPY; base rates no better (`docs/round20_results.md`) |
| Round 20 W: Warrior-style small-cap gap trading (gap-and-go, fade, Dux short) | pre-registered; waits for data (Databento about USD 10-11, or Alpaca keys) |
