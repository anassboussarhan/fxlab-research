# Inventory: everything tested, why each thing failed, and what the last search found (2026-11-11)

This page lists the whole project in one place. Details and exact numbers are in the document named in each row.
The status of every single item is in `docs/test_ledger.md`.

## 1. What holds up

| Result | Evidence | Status |
|---|---|---|
| **Gotobi** (long USD/JPY and EUR/JPY from 23:00 Tokyo before a Gotobi day to the 09:53 fix) | +4.45 bp a trade after costs (t 3.9) over 2007-2025; confirmed on 2019-2025 and on 2003-2006; a forced flow at a known time | forward test since 2026-11-05 |
| **Weekly dip-buying in currencies** (Double Seven and the Connors family, weekly bars, long only) | won in 2019-2025 (t 3.0) and on the 2026 holdout (175 trades, +0.38R, t 2.2); **lost in 2008-2018** (t -1.5) | forward test, three tracks |
| Selling index volatility (the variance premium) | SPY / DIA delta-hedged: t 1.7-2.5; real but crash-prone; needs an options account | not set up |
| Crypto cash-and-carry (long spot, short perpetual) | +0.69% a month, t 4.6, 2019-2025; exchange risk | not set up |
| Equity overnight returns, turn of the month, TLT into month-end, daily dip-buying in US index ETFs | significant on long samples; mostly the equity premium; weaker after 2018 | not FX; not set up |

## 2. Everything tested, by area

**Failure classes used below**
- **N:** no gross edge. Before costs the trades are a coin toss.
- **C:** cost-bound. A real gross edge, smaller than the cost.
- **D:** decayed. It worked in an earlier sample and stopped.
- **R:** a risk premium or market exposure, not a prediction. It pays for bearing a risk.
- **S:** skew trap. Many small wins, rare large losses.
- **M:** fails the correction for the number of things tried. Alone it looks fine; among thousands it is what luck
  gives.
- **P:** period-dependent. The sign flips between long periods.
- **X:** cannot be judged: no data, or too few events.

### 2.1 Technical rules, features and machine learning on FX and metals

| What | Size | Result | Why | Where |
|---|---|---|---|---|
| Strategy catalog, walk-forward: 97 families, 34 instruments, 10 timeframes (M5 to D1, tick and volatility bars) | 40,818 strategy streams | median net Sharpe -0.32; 0 significant | N at H1 and above; C below (M15 -2.4, M5 -5.4) | `conclusions.md` 8.1c |
| Every feature as a trading rule: 269 features | 173,204 streams | 0 significant; real rules barely beat placebo columns | N, M | 8.1c |
| Joint optimisation of parameters, entry and exit per pair | about 762,000 candidates | median Sharpe -0.41; 0 survive | overfitting: training +1.41 against test -0.36 | 8.1c, 8.1h |
| Entry types (market, limit, pullback, stop) and passive entries | every stored trade | loses less, never wins | C | `untested_batch.md` 9.3 |
| Trade management with random entries: 13 exits, pyramiding, martingale, grids, zone recovery, fully optimised | 3,094 cells + 114 portfolio tests | 0 survive; management reshapes the distribution and cannot create an edge | N; S for martingale and grids | `no_prediction.md`, `random_entry_opt.md`, `random_entry_pool.md` |
| Mean-reversion grids before costs | 34 pairs, H1-D1 | gross real (t 3.7); pays only at about one third of our costs | C | `random_entry_pool.md` |
| Mathematical projections (ordinal patterns, Markov, simplex, cycles, signatures...) | 9,520 tests | 86% significant at M1, but the move is 0.5-2% of the cost; 0 tradable | C | `projection_lab.md` |
| Feature information coefficients | 857,202 tests | 2.8% significant; the predicted move is 2-10% of the cost | C | `feature_predictability.md` |
| Direction models on EUR/USD H1: linear, trees, boosting, neural nets, LSTM / TCN / Transformers, RL | 198 runs | best AUC 0.539, worth 0.3 bp an hour against 1.35 bp of cost | C | `conclusions.md` 8.4 |
| ML entries: boosted three-outcome, triple barrier, adaptive barriers, policy optimisation, meta-labels, pooled weekly and intraday models, cross-asset features | all pairs, M5-D1 | beat random entries, lose 1-5 bp a trade after costs | C | `untested_batch.md` 8.2-8.7; `edges_batch_results.md` |
| Feature settings, session cells, feature pairs, volatility targets (FS1-FS6) | 58,920 + 9,120 tests | 0 count; 66 "passes" lose on the earlier sample | M, P | `edges_batch_results.md` |
| Time-series foundation model (Chronos) | 5 series | direction 49-52% | N | `untested_batch.md` 9.13 |
| Linear projections of the currency space (most predictable, most mean-reverting basket) | D1 / H4 | nothing; one "edge" was the triangular identity | N | `projections.md` |
| Famous-trader, forum and esoteric rules (Turtle, Connors, Raschke, round numbers, lunar...) | round 4 | every family's median is negative | N | 8.1c |
| 19 named systems from YouTube, forums and other languages (Malaysian SNR, Quasimodo, Dual Thrust, R-Breaker, Aberration, King Keltner, Thermostat, Camarilla, Elder, Puria...) | round 12 | 0 of 19 | N, C | `anatomy_results.md` |
| Order blocks, liquidity sweeps, fair-value gaps, volume profile, kill zones, session-range rules | lit5, lit6 | all lose; sweeps continue, they do not reverse | N | `untested_batch.md` 9.11-9.12 |
| Long trades that let winners run: Turtle 1 / 2, Clenow, Donchian, MA 50 / 200, time-series momentum, Faber, EWMAC, squeeze, weekly breakout; trailing, channel, breakeven and partial exits | round 13: 23 + 109 units, 33 instruments | 0 of 23; FX trend lost before costs in 2019-2025 | N (FX); R (non-FX: less than buy-and-hold) | `long_results.md` |
| Trend speeds, long-or-flat, currency-strength indices, trend + carry, trend prediction | round 14 | 0 of 26; no trendiness predictor works | N | `trend14_results.md` |
| Every rule (91, incl. 26 book systems) and every feature at D1 / W1 / MN, full grids, both directions | round 15: 1,071 units, 4,140 configurations | 0 survive | see section 3 | `round15_results.md` |
| The catalog on 33 daily stock, bond, commodity and crypto markets | 2,112 family-market pairs | none beats buy-and-hold after correction | R, M | `untested_batch.md` 8.1 |
| The catalog on US500 / NAS100 intraday; opening-range breakout; index intraday momentum | index CFDs 2019-2025 | Nasdaq 5-minute ORB +3.8 bp gross (t 2.2), +1.7 net | C | `untested_batch.md` 5.3, 9.12 |

### 2.2 Flows, calendars, events and macro

| What | Result | Why | Where |
|---|---|---|---|
| 17 published flow and calendar effects (month-end fix, local hours, around the clock, FOMC days, fix reversal, VIX-filtered carry...) on 2007-2018 | only Gotobi and FX variance selling survive | C (local hours, around the clock, fix reversal); D (FOMC days: real until 2010) | `flows_lab.md` |
| Gold and silver fixes, gold Asian / Western hours, gold seasonality, dollar carry, factor momentum, weekly momentum, on 2011-2018 | 0 of 11; silver fix, gold AM fix and gold Asian hours are real before costs | C; D after the 2014-2015 fix reforms | `lit2_lab.md` |
| Macro fundamentals (economic momentum, output gap, Taylor rule, elastic net) | none; the elastic net never trades | N; D (momentum worked to 2012) | `macro_lab.md` |
| Tokyo-fix extensions, other yen pairs, post-fix reversal, month-end London fix, CFTC positioning, rebalancing | only the overnight Gotobi form | N, X | `untested_batch.md` 9.1-9.8 |
| Edges batch E1-E26: option expiries, Treasury and JGB auctions, central-bank drifts, MSCI reviews, CPI / payroll surprises, positioning, search interest, festivals, EM carry, round numbers, night fade, currency strength | nothing counts; ECB drift and CPI surprise met their own rule on 19 and 11 events | N; X (too few events) | `edges_batch_results.md` |
| Flow, volume and positioning V1-V10 (futures / ETF volume, CFTC, SNB deposits, news tone, yield gaps, fed funds futures, interventions) | nothing counts | N, X | same |
| Round 9: economic surprises with real consensus, Gotobi on futures, CME risk reversals, pinning, dealer gamma, signed futures flow, seasonality, skewness, Shanghai premium, Norges, BoJ, coupons | 0 of 22 | N, C, X | `round9_results.md` |
| Free versions of paid data: calendar archive (surprise drift, surprise index), CBOE option-selling indices, 2-year rate momentum, commodity / index / bond CFDs | 0 of 48; the 1-day surprise drift is closest (t 2.2) | M, R | `round15_results.md` |

### 2.3 Options

| What | Size | Result | Why | Where |
|---|---|---|---|---|
| 92 option structures on 106 underlyings, 3 maturities, 6 management rules, 6 filters, walk-forward | 745,383 combinations | no SPY alpha; single stocks negative; every management rule lowers Sharpe | R (index), C (stocks) | `options_strategies_all.md` |
| 22 published options edges (variance premium, skew, dispersion, earnings, events, calendars, weekends) | real quotes 2019-2025 | index variance premium t 1.7-2.5; nothing survives BH | R, M | `options_edges.md`, `options_edges2.md` |
| tastytrade mechanics (50% profit, 21 days, rolling, VIX sizing) | SPY, DIA | lower the mean on every structure | N | `untested_batch.md` 3 |
| FX options on CME (euro, yen): selling volatility, risk-reversal momentum, pinning, gamma switch, hedged carry, implied Hurst | 2010-2025 | premium eaten by costs; signals fail | C, N | `round9_results.md`, `edges_batch_results.md` O1 |
| Options funding programmes | 2 providers | the median account loses the fee | - | `options_capital_providers.md` |

### 2.4 Regimes

| What | Size | Result | Where |
|---|---|---|---|
| Regime models as return predictors (HMM, Markov switching, regime-gated ensembles) | catalog models | no better than linear | `catalog_models.md` |
| Round 16: six regime-method families, monthly, three library tiers, 33 features incl. options | 18 tests | 0 count; none beats static selection or its placebo | `round16_results.md` |
| Round 17: intraday libraries, daily / weekly / monthly switching, per-instrument regimes, Hedge, CUSUM, pooled boosting | 93 + 486 tests | 0 count; faster switching is worse; a 14-hour look-ahead found and fixed | `round17_results.md` |
| Round 18: a feature-free test for performance regimes; a regime detector on 1976-2025 | permutation tests; 3 tests | no persistent states in D1 / W1 / MN strategies; intraday states are cost states; on 50 years no detector beats static selection | `round18_results.md` |

### 2.5 Prop firms and execution

| What | Result | Where |
|---|---|---|
| Nine programmes simulated with and without an edge; leverage; several firms at once; drawdown-scaled leverage; Gotobi as a lottery | a no-edge trader is paid 0.5-7% of the time; Gotobi at 5-6x: paid about 45-60% | `prop_firms.md`, `prop_best.md`, `prop_mix.md` |
| Break-even costs at four venues (ours, raw ECN, IBKR, institutional) | cheaper execution helps Gotobi; nothing else becomes significant | `breakeven.md` |

### 2.6 Searches (13 rounds before today)

Papers and working papers; quant blogs; MQL5 articles, blogs, Market, forum and signals; Forex Factory and other
forums; YouTube methods; prop-firm pages and trader stories; Trustpilot; tastytrade; high-win-rate and small-account
claims in other languages; who actually made money in FX. See `literature_check.md` (sections 1-13),
`mql5_review.md`, `tastytrade.md` and `profitable_traders.md`.

## 3. Why strategies fail, in numbers

**Round 12 anatomy (197 strategy units, trade by trade).**

| Class | Units | Meaning |
|---|---|---|
| No gross edge | 151 | median gross -0.40 bp a trade against 2.97 bp of cost |
| Cost-bound | 25 | median gross +1.41 bp against 2.75 bp of cost; the gross fell to +0.30 bp in 2019-2025 |
| Decay | 14 | net positive in 2007-2018, negative in 2019-2025 |
| Skew trap | 5 | 65-66% winners, average loss 2.1-2.7 times the average win |
| Weak | 2 | positive in both periods, not significant |

A fix search (entry, exit, reversal, one filter per strategy) rescued 0 of 196.

**Round 15 library (2,070 configurations in their published direction, D1 / W1 / MN, FX book; classified today from
the saved statistics, `docs/round18/failure_classes_round15.csv`).**

| Class | Configurations | Share |
|---|---|---|
| No gross edge in 2019-2025 (gross abs(t) below 1.645) | 1,796 | 86.8% |
| Wrong way before costs; the reverse is cost-bound | 116 | 5.6% |
| Cost-bound (gross t >= 1.645, net below) | 75 | 3.6% |
| Wrong way: the reverse made money net in one period | 43 | 2.1% |
| Worked only in 2019-2025 | 32 | 1.5% |
| Worked in 2008-2018, not after | 8 | 0.4% |
| Worked in both periods | 0 | 0% |

Luck alone would put about 5% of configurations in each "worked in one period" row. There are 1.5% and 0.4%, so
the library is slightly worse than chance, because costs push everything down.

**What the classes mean for a regime detector**
- **No gross edge (most strategies).** A detector helps only if the zero average hides good and bad states that
  persist. Round 18's part R0 tests this directly.
- **Cost-bound.** The matching "regime" is a cost regime: trade only when the expected move is large against the
  spread. This was tested: the cost-to-volatility gate (VS4), cost-aware meta-labels (M2) and the large-move filter
  (FS5 / M4). None counts. Round 18's part G asks the remaining question: has the cost itself fallen since 2018
  (gold)?
- **Decay.** A structural change (fix reforms, crowding). A detector can say "stop", and cannot bring the edge back.
- **Skew trap.** The detector would have to predict crashes. VIX-filtered carry and the crash filters tested do not.
- **Period-dependent (dip-buying).** This is the one true regime case. 2008-2025 holds one flip, too few to learn
  from. Round 18's part L uses 1975-2025.

## 4. What today's search found (2026-11-11)

| Find | Source | Status |
|---|---|---|
| Gold's Asian-hours rise became much stronger after 2020 (+5.5 bp a night after costs, t 3.6, mid-2020 to 2026) | MQL5 blog, a 49,000-backtest study | **new: round 18, G-A1.** Our test covered 2011-2018 only (gross +2.4 bp, cost-bound) |
| The other 2007-2018-only tests with a gross edge were never re-run on 2019-2025 | found by the inventory | **new: round 18, part G** |
| Overnight-to-intraday reversal across currencies (Sharpe 1.26 before costs in currency futures, 1982-2014) | Della Corte, Kosowski and Wang | **new: round 18, C1** |
| Global currency volatility predicts dollar returns over 3-12 months | Journal of Empirical Finance, 2026 | **new: round 18, L3** |
| Timing currency factors with state variables (forward discount, volatility, dispersion) | "Conditional currency momentum portfolios" (2025); "Factor timing in currency markets" (2026) | **new: round 18, L2**, on 50 years |
| VWAP trend following on QQQ | Zarattini and Aziz | **new: round 18, C2** |
| An LLM that reads economic releases (Sharpe 0.5-0.6 before costs, 36-60-month lookbacks) | arXiv 2608.00761 | not run: needs a paid model, and has look-ahead risk; close to the macro-momentum and surprise-index tests already run |
| 15-minute reversal in crypto | arXiv 2608.21888 | not run: the authors find +1.3 bp gross against 5 bp of cost |
| ICT "Power of Three" / Judas swing on 18 years of gold | MQL5 blog | agrees with our tests: a coin flip |
| FX seasonality "that still works" (home hours, month-end, January, August in USD/JPY, fiscal year-end) | FXEmpire | covered: local hours, month-end fix, same-month seasonality (round 9) |
| Carry, value and momentum over 200 years | Quantpedia | supports part L: momentum and value worked only in some eras |
| Opening-range breakout, intraday momentum, crypto trend, industry trend | Zarattini's papers | covered: ORB5, intraday momentum, crypto trend (daily) |

**Outcome of the new items (`docs/round18_results.md`).** None counts. Gold's Asian-hours rise is real before
costs and loses after them, and in 2019-2025 it is no longer different from Western hours. The index VWAP trend is
real before costs and loses after them. The long-history regime test and the volatility-dollar test fail.

**Three corrections to the previous summary.** Daily dip-buying on index ETFs, daily crypto trend, and the Nasdaq
opening-range breakout at futures costs had all been tested (`untested_batch.md`). Only their weekly / CFD forms
had not.

## 5. What cannot be tested with free data

| Item | What is missing |
|---|---|
| Retail positioning, contrarian | the OANDA order and position books need the user's API token |
| Japanese retail FX flow | TraditionData or Nasdaq GRFF: paid, trial on request |
| FX option expiries at the New York cut | DTCC history: collecting forward since 2026-11-04 |
| FX implied-volatility surfaces beyond euro and yen | CME options for four more currencies: USD 60.29 on Databento |
| Order-book and footprint methods | no free depth data |
| BoE meeting drifts, ISM surprises, TRY carry | no usable free history |
