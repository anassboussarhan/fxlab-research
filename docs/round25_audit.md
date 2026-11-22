# Audit after round 24: every test, every dimension, and what was never varied (2026-11-22)

**What this answers:** "enumerate every test you did, what method you missed, what parameter range or combination
wasn't tested". It updates `round23_audit.md` with rounds 23-24 and checks each claim of "not tested" against the
project documents.

**Where the detail is:**
- `inventory.md`: everything tested, with failure classes;
- `test_ledger.md`: the status of every item;
- the round documents: exact numbers.

Round 24 is running while this is written (`round24_preregistration.md`). Its parts are listed as "running", not
as results.

## Summary

- **Size of the search** (`inventory.md`):
  - rounds 22 and 23: 64,499 and 240,953 walk-forward streams;
  - before them: 40,818 catalog streams, 173,204 feature streams, about 762,000 optimisation candidates,
    857,202 feature-coefficient tests and 745,383 option combinations;
  - plus about 300 targeted tests of flows, events, macro and named systems.
- **What came out:**
  - only Gotobi, crypto cash-and-carry and (weakly) D7 hold up;
  - every larger search since round 15 found nothing new;
  - two of the round's own faults (round 22's FVG simulator, round 23's decoupled constants and missing carry)
    produced the only "passes".
- **What was never varied** falls into three groups:
  1. **How the search is done:**
     - 128 random points per unit in 25-45 dimensions (sparse);
     - one setting chosen per instrument, never across instruments;
     - scores limited to t or mean, with no stability or plateau criterion;
     - no decay-weighted training.
  2. **Structural choices inside rules and their combinations:**
     - moving-average types, price sources and normalisations are fixed;
     - only one entry type at a time;
     - no rule combined with a second rule or a higher-timeframe filter in the free-parameter library;
     - no regime or day filters jointly with the parameters;
     - stops always on ATR(14);
     - sizing never varied in LW.
  3. **A few markets and data:**
     - altcoins intraday;
     - stock intraday;
     - futures and options intraday;
     - order flow;
     - retail positioning;
     - insiders.
- **Priors:**
  - Group 1 changes how robust a selection is. It cannot create a gross edge where the median gross edge is
    negative (`inventory.md` section 3).
  - Group 2 widens a space whose tail matched its placebos in rounds 15, 22 and 23.
  - Group 3 is mostly blocked by data.
- **The ranked list is in section 9.** All of its items are free.

## 1. Every test, by area

| Area | What | Size | Result | Where |
|---|---|---|---|---|
| Technical rules on FX / metals | 97-134 catalog families, 34 instruments, M5-D1, tick / volatility / range / Renko bars | 40,818 streams | median net Sharpe -0.32, 0 significant | `conclusions.md` 8.1c |
| Features as rules | 269 features | 173,204 streams | 0 significant | 8.1c |
| Joint optimisation of rule, entry and exit per pair (Optuna / TPE) | about 762,000 candidates | | 0 survive; train +1.41, test -0.36 | 8.1c, 8.1h |
| Entry types | market, limit, pullback, stop, passive | every stored trade | lose less, never win | `untested_batch.md` 9.3 |
| Trade management on random entries | 13 exits, pyramiding, martingale, grids, zone recovery | 3,094 cells + 114 portfolios | 0 survive | `random_entry_opt.md`, `random_entry_pool.md` |
| Mathematical projections | ordinal patterns, Markov, simplex (nearest neighbours), cycles, signatures | 9,520 tests | real at M1, 0.5-2% of cost | `projection_lab.md` |
| Feature information coefficients | | 857,202 | 2.8% significant, 2-10% of cost | `feature_predictability.md` |
| Machine learning | linear, trees, boosting, nets, LSTM / TCN / Transformer, RL, genetic programming, grammatical evolution, differential evolution, foundation model | 198+ runs | best AUC 0.539 | `conclusions.md` 8.4, `catalog_models.md` |
| ML entries | three-outcome, triple barrier, meta-labels, pooled models, cross-asset features | all pairs M5-D1 | lose 1-5 bp a trade | `edges_batch_results.md` |
| Named systems | 19 YouTube / forum / foreign-language systems, famous traders, Turtle, Connors, Raschke, lunar | rounds 4, 12 | 0 | `anatomy_results.md` |
| ICT / SMC | order blocks, sweeps, FVG, OTE, breakers, Silver Bullet, Judas, midnight open, SMT, CBDR, kill zones | lit5-6, rounds 21-23 | no gross edge; the one pass was a simulator look-ahead | `round21_results.md`, `fvg_audit_results.md` |
| Trend following | Turtle, Clenow, Donchian, MA 50/200, TSMOM, Faber, EWMAC, squeezes; trailing, channel, partial exits | rounds 13-14 | 0 of 49 | `long_results.md`, `trend14_results.md` |
| Daily / weekly / monthly library | 91 rules incl. 26 book systems, both directions | round 15: 4,140 configurations | 0 | `round15_results.md` |
| Other markets with the catalog | 33 daily stock / bond / commodity / crypto markets; index CFDs intraday | 2,112 pairs | none beats buy-and-hold | `untested_batch.md` 8.1 |
| Flows and calendars | 17 published effects, Gotobi forms, fixes, month / quarter / fiscal-year ends, gold / silver fixes, festivals | | only Gotobi | `flows_lab.md`, `lit2_lab.md` |
| Events | option expiries, Treasury / JGB auctions, central-bank drifts, MSCI, CPI / payroll surprises with consensus, EIA, interventions | edges E1-E26, V1-V10, round 9 | nothing counts | `edges_batch_results.md`, `round9_results.md` |
| Positioning and attention | CFTC (all trader classes), small speculators, search interest, news tone (GDELT), SNB deposits | | nothing | same |
| Macro | economic momentum, output gap, Taylor rule, real-rate carry, curve slope | | worked before 2005 | `macro_lab.md`, `round21_results.md` |
| Long histories | carry, trend, weekly reversal, weekly dip-buying on 1976-2025 | round 18 | carry / trend paid until about 2005 | `round18_results.md` |
| Options | 92 structures × 106 underlyings × 3 maturities × 6 management × 6 filters; 22 published edges; tastytrade mechanics; CME FX options | 745,383 combinations | index variance premium only (R) | `options_strategies_all.md` |
| Crypto | variance premium (DVOL), cross-section, funding times, cash-and-carry | | cash-and-carry counts (small) | `crypto_vrp.md`, `round22_results.md` |
| Stocks | S&P 500 anomalies, Minervini / Kullamaegi setups, TLT month-end, overnight, turn of month | | equity premium only | `round20_results.md`, `round21_results.md` |
| Regimes | HMM, Markov switching, six method families, intraday switching, CUSUM, Hedge, 50-year detector | rounds 16-18 | none beats static selection | `round16-18_results.md` |
| Every parameter walk-forward | all parts, yearly re-selection | round 22: 64,499 streams | only cash-and-carry | `round22_results.md` |
| Every constant free, continuous entries / exits, nested selection, 30 currencies, ETF, causal ICT, half-hour windows, 18 new instruments | | round 23: 240,953 streams | nothing new | `round23_results.md` |
| Prop firms | 9 programmes, leverage, several firms, drawdown scaling, live candidates together | | no-edge trader paid 0.5-7%; Gotobi 45-60% | `prop_*.md`, `round23_results.md` |
| Selection statistics | White's Reality Check, Hansen SPA, deflated Sharpe, BH, placebos, look-ahead truncation test | | | `conclusions.md`, `round23_results.md` |
| **Round 24 (running)** | carry in every engine, tied session rules, TTM, SMT, earnings drift, short volume, crypto basis and cross-venue funding, CME gap, net liquidity, ETF pairs, S&P additions | | | `round24_preregistration.md` |

## 2. Signal families

**Tested:**
- trend, breakout, mean reversion, momentum, carry, value, seasonality and calendar;
- sessions and time windows, volatility, volume and tick imbalance;
- chart patterns (head and shoulders, harmonic), Heikin-Ashi, Ichimoku, ICT / SMC;
- cointegration and baskets, lead-lag, entropy / Hurst / fractal, Kalman, wavelets;
- regime models, ML on all features, genetic programming;
- flows, events, positioning, attention, macro.

**Not tested as signals:**
- **Rule confluence:** k of n library rules agreeing, as a rule of its own. ML combined features, not the
  rules' signals.
- **Rule plus higher-timeframe filter:** e.g. an H1 entry only in the D1 trend direction. It exists only as a
  feature (`catalog_features.md`).
- **Rule plus regime filter in the free-parameter search:** round 12 tried one filter per fixed rule (0 of 196),
  round 16 regime gating on fixed libraries.

## 3. Parameters and their ranges

| Part | Parameters | Range drawn | What was never drawn |
|---|---|---|---|
| LW (round 23) | the rules' own parameters | half the smallest to twice the largest value of the published grid | values outside ×0.5-×2 of the grid |
| LW | numeric constants in the rules' code | ×1/3 to ×3 of the literal, log for integers ≥ 2 | beyond ×1/3-×3; constants tied by role (only the 3 rules of round 24) |
| LW | categorical structure | never exposed | moving-average type (SMA / EMA / WMA / Hull), price source (close / HL2 / HLC3 / typical), ATR vs standard deviation, simple vs log returns |
| LW | sampling | 128 Sobol points per unit in about 25-45 dimensions | adaptive search inside the walk-forward (TPE / CMA-ES); more points; local refinement |
| T2 | entry at each half-hour; hold 0.5-24 h (11 values); 6 signs; 21 day filters | full grid, 66,528 per instrument | minute-level entry times; two-window combinations; stops / targets on windows; holds above 24 h |
| IC | round-22 Sobol ranges per family; timeframe M5-H1 (indices), M15-H4 (FX) | 128 points per family | D1 for ICT; families combined (e.g. sweep + FVG + OTE); wider ranges than round 22 |
| C30 | carry, real carry, momentum 1-12 (skip 0-1), value 36-84, composite; n 1-5; rebalance 1-3; VIX 15-40 | 256 points | weekly frequency; risk-parity or rank weights; signal blends other than equal-rank |
| ETF | momentum 1-12 (skip 0-1), raw or vol-scaled, top 1-5, filters, weights, rebalance 1-3 | 256 per universe | carry / value / low-vol for ETFs; daily or weekly rebalance |
| Round 24 | as pre-registered, e.g. R4 H 1-60, R7 θ 0-40%, R9 H 1-120 h | 128 per unit | |

## 4. Entries

- **Tested:**
  - market, limit at touch, pullback 0.05-1.5 ATR with a 1-24 bar life, stop breakout 0.05-1.5 ATR;
  - confirmation over 1-5 bars;
  - a London-time session window (start 0-23, length 4-24 h);
  - passive maker entries;
  - scale-in (half at market, half on a pullback).
- **Never combined:** LW draws one entry type per configuration. It never combines a session window with a
  pullback, or a confirmation with a stop entry.
- **Session windows only in London time.** Tokyo and New York windows exist only in T2 and the session rules.
- **No day-of-week filter in LW.** Day filters exist only in T2.

## 5. Exits and management

- **Tested in LW:**
  - stops: ATR, realised volatility or structural (2-10 bars, with a buffer), × 0.5-6;
  - target 0.5-6 R;
  - trailing 0.5-6 ATR;
  - breakeven 0.3-3 R;
  - partial exit at 0.5-4 R (20-80%);
  - time stop 2-500 bars;
  - the rule's own exit.
- **Tested elsewhere:** pyramiding, martingale, grids and zone recovery (random-entry labs).
- **Never varied:**
  - the ATR length behind stops and targets (always 14);
  - volatility-adjusted time stops;
  - exits on an opposite signal of a different rule.

## 6. Sizing and portfolios

- **Tested:**
  - fixed fractional;
  - volatility targeting;
  - Kelly fraction;
  - drawdown scaling;
  - correlation-aware and currency-exposure caps (engine);
  - prop leverage 1-6×.
- **Never varied in LW:** sizing is "none" (unit exposure).
- **Never tested:**
  - sizing by signal strength (a continuous position from a rule's distance to its threshold);
  - portfolios of many weak units chosen walk-forward across the whole library, beyond the per-strategy pooled
    streams and the round 16-17 library tiers.

## 7. Timeframes, bars, instruments, periods

| Dimension | Tested | Not tested |
|---|---|---|
| Timeframes | M1 (projections), M5-MN, W1, D2; tick, volatility, range and Renko bars | — |
| FX | 28 G10 pairs, 14 Nordic / EM pairs (USD/SGD has no rate series in round 24) | other EM pairs (INR, KRW, BRL, IDR: not on Dukascopy minute data) |
| Metals, commodities, bonds, indices | XAU, XAG; Brent, WTI, gas, copper; Bund, Gilt, T-bond; 13 index CFDs | platinum and palladium (no data) |
| Crypto | BTC and ETH intraday; the Binance spot universe daily (round 21) | **altcoins intraday** |
| Stocks | S&P 500 daily (current members), swing setups | **stock intraday**, small caps (no free minute data) |
| Futures and options | CME daily, FX options, US options quotes 2019-2025 | intraday futures (NQ paid), 0DTE |
| Periods | 2007-2025 (P1 to 2018, P2 2019-2025); Gotobi 2003-2006; daily FX 1976-2025; 2026 five one-time checks | intraday before 2007 for anything but Gotobi |

## 8. Selection and statistics

- **Selection (rounds 22-23):**
  - window 12, 24, 36 or 60 months, or expanding;
  - top 1, 3, 5 or 10, equal weight;
  - re-selection monthly, quarterly or yearly;
  - score t or mean;
  - nested choice among these 120.
- **Never used:**
  - **panel selection:** one setting chosen on all instruments' pooled training performance, then applied to
    each (fewer free choices, more data per choice);
  - **stability scores:** the plateau or neighbourhood of a configuration, or its rank stability across years;
  - **decay-weighted training:** recent years weighted more;
  - **other scores:** Sharpe with a turnover penalty, deflated Sharpe or PBO used as the selection score (they
    were used only to judge results);
  - **other weights:** weekly re-selection, inverse-variance or minimum-variance weights among the top k.
- **Statistics:**
  - iid t on daily streams;
  - BH at q = 0.10;
  - placebos;
  - Reality Check and SPA (the early catalog campaign);
  - Newey-West in the long and holdout tests.
- **Never done:** HAC t for the round 22-24 counting rule. It could only lower t; it matters for passes, and
  there are almost none.

## 9. The gap list, ranked

**Ranking rule:** plausibility against cost. Everything here is free. Development data only; 2026 only for a
pre-registered check of something that counts.

| # | Gap | Test | Prior | Effort |
|---|---|---|---|---|
| 1 | Panel selection | LW at H1-W1 with one configuration chosen per strategy × timeframe on all FX pairs together (and per asset class), applied out of sample to each | low-moderate: it attacks the overfitting the per-unit walk-forward shows | large (a job loads every instrument; H1+ only) |
| 2 | Rule plus higher-timeframe filter | each LW rule at H1 / H4, taken only in the direction of a D1 / W1 trend (MA slope or Donchian state, lengths free) | low-moderate (a common practitioner fix; in round 12 one filter per rule rescued 0 of 196) | medium |
| 3 | Confluence | k of n library rules agreeing (n from the top rules of the training window, k free) as a new rule, walk-forward | low (ML on features already combines information) | medium |
| 4 | Joint filters | LW with one regime filter (ADX, ATR percentile, trend strength, spread percentile; thresholds free) and a day-of-week mask drawn jointly with the rule | low | large |
| 5 | Combined entries | session window × entry type × confirmation drawn together; session in Tokyo, London or New York time | very low (entries lose less but never win) | medium |
| 6 | Categorical structure | MA type, price source, ATR vs standard deviation as draws for the rules that use them | very low | medium |
| 7 | Adaptive search | TPE or CMA-ES inside each training window (512 evaluations) for the 20 rules with the best round-23 training scores | low (round 8.1h: train +1.41, test -0.36) | large |
| 8 | Stability selection | score = median t over the configuration's 8 nearest neighbours, or the share of years positive; and decay-weighted training | low-moderate as a robustness check of near-misses | small (needs the stored matrices: LW saves none) |
| 9 | Signal-strength sizing | continuous positions for the rules with a threshold (z-scores, RSI distance), vol-targeted | very low (sizing cannot create an edge) | medium |
| 10 | Altcoins intraday | the library at H1-D1 on the 30 most liquid Binance perpetuals, with funding as the carry | low (costs and funding) | medium (public archive) |
| 11 | Stops on other ATR lengths | ATR 5-50 for stop / target / trail | very low | small, inside 4 |
| 12 | HAC t | Newey-West t for every own-rule pass of rounds 22-24 | none (a check) | minutes |

**Still blocked:**
- insider buying (SEC archive needs a contact e-mail);
- Congress trades (datasets gone);
- OANDA positioning (token);
- small caps and stock intraday (Alpaca keys);
- NQ futures and 0DTE (paid);
- order flow (paid).

Not financial advice.
