# Round 16 results: market regimes, the best strategy for each regime, and what options data adds (2026-11-09)

Rules: `docs/round16_preregistration.md` (addenda 1-2), committed before the engine ran.
- Code: `scripts/r16_data.py` (data), `scripts/regime16.py` (engine; leakage tests in `tests/test_regime16.py`),
  `scripts/r16_explore.py` (follow-ups, not counted).
- Runs: data 42669526702 (reusing the library and CME artifacts of 42662594180), engine 42670029876, follow-ups
  42674038581. Outputs are on branches `round16`, `round16-data` and `misc-r16x`.

## The answer

**No regime method works.** In 18 tests, the method, number of regimes and setting were chosen on 2011-2018. The
families are strategy momentum, a past-performance filter, clustering, single-feature splits, hidden Markov models
and ridge / boosting meta-models, each in three library tiers.
- None was profitable on 2019-2025 after the correction.
- None beat plain "pick the best past strategy".
- None beat its time-shifted placebo.
- Strategies do win in some periods and lose in others, but the market state at the time (volatility, correlation,
  trend strength, carry, risk appetite, implied volatility, skew, rates) does not tell you which strategy to run
  next month.

**The number of regimes.**
- **Information criterion (BIC) on 2008-2018:** 2 regimes for the price features and for the options features;
  1 (no regimes at all) for the full feature set.
- **Out of sample:** no number of regimes, from 2 to 8, beat having none.

**Options data.**
- The CME euro and yen implied volatility and risk reversals bought in round 9, and the free CBOE indices (VIX,
  VIX3M, VVIX, SKEW, gold and oil volatility), were tested three ways: as regime inputs, as timing signals, and as a
  filter for the dip-buying trade. They added nothing that counts.
- **Market-wide implied volatility.** This is the measure of fear in the market: CBOE's VIX and its relatives, plus
  CME FX implied volatility. Its effect on the FX strategies is the same as realised volatility's: none that can be
  traded.

**The weekly dip-buying trade (the forward test).** No regime filter improves it. Every filter that looked best on
2008-2018 made 2019-2025 worse or no better.

**One descriptive finding (exploratory, not counted).** How strategies deviate from their average return in
high- versus low-yen-volatility months repeats from 2008-2018 to 2019-2025.
- The correlation across the library is 0.27, beyond all 2,000 time-shifted placebos.
- The month's VIX change behaves the same way for the nine styles.
- This repetition is real, but too small to trade: the split method on those same features lost money on
  2019-2025 (median t -0.7 to -1.6 in the two larger tiers).

## 1. Data

**The strategy library**

| Tier | What it holds | Size |
|---|---|---|
| CFG | every round-15 configuration: rule or feature, timeframe, setting, direction | 4,140 |
| UNIT | each (unit, direction), averaged over its settings | 2,258 |
| FACTOR | nine classic FX styles: carry, 12- and 3-month trend, EWMAC, weekly and daily reversal, weekly dip-buying (the forward-tested family), Donchian 20, the dollar | 9 |

**The placebo library.** The round-15 random-signal units: 660 configurations (300 unit-directions).

**Regime features.** 33 of the 34 pre-registered features.
- **Dropped:** EVZ. CBOE no longer serves it and Yahoo has none.
- **Sources:**
  - FX: the M1 panels (16 crosses from 2007, 28 from 2010);
  - S&P 500 and gold: the index and the GLD ETF (addendum 2);
  - VIX family: CBOE, with Yahoo for VIX3M / GVZ / OVX before CBOE's files begin;
  - CME: euro and yen at-the-money IV and 25-delta risk reversals, 187 month ends from 2010-06;
  - rates and conditions: Treasury yields, the Chicago Fed NFCI and the daily EPU index.
- **FRED** refuses connections from the runners, so the original publishers were used instead.

**Options data: what exists and at what cost**

| Source | What | Cost / status |
|---|---|---|
| CME via Databento (bought in round 9) | euro and yen futures options, daily settlements and open interest, 2010-06 to 2025 | already paid from the free credit |
| CME via Databento (sterling, AUD, CAD, CHF) | the same for four more currencies | priced at USD 60.29; not bought (above the USD 10 limit set inside the remaining free credit) |
| CBOE (free) | VIX, VIX3M, VVIX, SKEW, GVZ, OVX; option-strategy indices (round 15) | used |
| CBOE EVZ (euro volatility index) | FX implied volatility | no longer published (403); no free history found |
| DoltHub (free) | US equity options end-of-day, 2019 on | used in the options rounds (`docs/options_strategies_all.md`) |
| Deribit (free) | BTC / ETH implied volatility (DVOL) | used (`docs/crypto_vrp.md`) |
| Saxo FX option vols and DTCC FX option trades | daily forward collection | collecting since round 9 (`flow-collect`); no history |
| CME CVOL, Bloomberg / Refinitiv FX vol surfaces | full FX implied-vol history | paid only |

## 2. The regime methods (part B): no test counts

Each row is the setting with the best Newey-West t on 2011-2018 in its family and tier, judged once on 2019-2025.
- Returns are in units of each strategy's volatility; a switching cost is charged.
- "Minus B1" is the difference from static selection (the best past strategies, no regimes), chosen the same way.

| Tier | Family | Chosen on 2011-2018 | t 2011-18 | t 2019-25 | Sharpe 2019-25 | minus B1, t | Placebo 95th pct Sharpe | Counts |
|---|---|---|---|---|---|---|---|---|
| FACTOR | B1 static (benchmark) | trailing 60 months, best 1 | -0.47 | 1.42 | 0.51 | 2.63 vs equal weight | - | - |
| FACTOR | M1 momentum | 36 months, best 1 | -0.99 | -0.12 | -0.04 | -2.01 | 0.87 | no |
| FACTOR | M1b filter | 6 months | -2.30 | -1.15 | -0.42 | -3.36 | 0.00 | no |
| FACTOR | M2 clusters | k-means, 8 regimes, ALL, 60-month window | 0.85 | 0.48 | 0.17 | -1.08 | 0.66 | no |
| FACTOR | M3 one feature | 10y-2y slope, 2 regimes | 1.87 | -1.05 | -0.37 | -2.70 | 0.74 | no |
| FACTOR | M4 HMM | 2 states, ALL | -0.10 | 1.87 | 0.64 | 1.04 | 0.66 | no |
| FACTOR | M5 predict | ridge, ALL, alpha 10 | 0.94 | 0.42 | 0.13 | -1.12 | 0.36 | no |
| UNIT | B1 static (benchmark) | expanding, best 5 | 0.83 | -2.10 | -0.65 | 0.27 vs equal weight | - | - |
| UNIT | M1 momentum | 12 months, best 1 | 0.01 | -0.61 | -0.22 | 0.40 | 0.24 | no |
| UNIT | M1b filter | 12 months | -4.78 | -4.43 | -1.61 | -0.03 | -1.22 | no |
| UNIT | M2 clusters | k-means, 6 regimes, OPT | 1.28 | -1.57 | -0.58 | -0.62 | 0.10 | no |
| UNIT | M3 one feature | S&P variance premium, 2 regimes, best 5 | 2.33 | -2.55 | -0.78 | -0.44 | -0.25 | no |
| UNIT | M4 HMM | 2 states on the library's own returns, best 20 | 0.92 | -1.53 | -0.57 | 0.90 | -0.29 | no |
| UNIT | M5 predict | ridge, PRICE, alpha 100 | 0.90 | 0.03 | 0.01 | 0.95 | -0.07 | no |
| CFG | B1 static (benchmark) | expanding, best 1 | 0.13 | 0.72 | 0.25 | 1.61 vs equal weight | - | - |
| CFG | M1 momentum | 12 months, best 1 | 0.24 | 0.28 | 0.09 | -0.12 | 0.20 | no |
| CFG | M1b filter | 12 months | -4.93 | -3.60 | -1.32 | -1.71 | -0.90 | no |
| CFG | M2 clusters | k-means, 6 regimes, OPT | 1.47 | 1.36 | 0.45 | 0.87 | 0.47 | no |
| CFG | M3 one feature | VIX / VIX3M, 2 regimes | 1.71 | -0.98 | -0.37 | -1.91 | 0.30 | no |
| CFG | M4 HMM | 3 states, OPT | 0.53 | -0.36 | -0.13 | -1.88 | 0.41 | no |
| CFG | M5 predict | ridge, PRICE, alpha 100, best 10 | -0.07 | -0.77 | -0.31 | -0.99 | -0.22 | no |

**Reading the table**
- **No test passes.** The closest is the two-state HMM on the nine styles (t 1.87, Sharpe 0.64). It does not pass
  the correction over the 18 tests (it would need p <= 0.0056; it has 0.031). It does not beat static selection
  (t 1.04), and its Sharpe is below its placebo's 95th percentile (0.66).
- **The best setting on 2011-2018 failed afterwards.** In every tier, the family with the best 2011-2018 result was
  the single-feature split, and it lost on 2019-2025 (t -1.05, -2.55, -0.98).
- **The whole surface looks like chance.** Of 5,590 settings across tiers and families, the share with t >= 1.645
  on 2019-2025 is 0-8% per family. That is near or below the 5% that luck gives one-sided.
- **The machinery itself can fool you.** On the placebo library of random strategies, the single-feature split
  chose a setting with t 2.36 on 2019-2025, purely from selection. That is why P4 exists, and why one good-looking
  number proves nothing.

**The number of regimes (best 2011-2018 setting for each K).** No K is consistently better; most rows are below
zero in the larger tiers.

| Tier | Method | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 |
|---|---|---|---|---|---|---|---|
| FACTOR | clusters, t 2019-25 | 0.92 | 0.82 | 0.93 | 1.08 | 0.33 | 0.48 |
| UNIT | clusters | -2.37 | -2.35 | -2.22 | -1.41 | -1.57 | -2.70 |
| CFG | clusters | -0.42 | 1.40 | -1.06 | 0.45 | 1.36 | -0.87 |
| FACTOR | HMM | 1.87 | 1.33 | -0.07 | | | |
| UNIT | HMM | -1.53 | -1.84 | -1.01 | | | |
| CFG | HMM | -0.10 | -0.36 | -0.06 | | | |

**What the regimes look like** (the clusters on the nine styles at the 2025 refit, in-sample; `regime_maps.json`).
The clusters are sensible, but they do not carry forward.
- **Calm, low-volatility months, with high SKEW and VVIX:** dip-buying (F7) had the best average.
- **High-volatility sell-offs:** weekly reversal (F5).
- **High-VIX months with high variance ratios (trending moves) and compressed carry:** trend (EWMAC, TSMOM
  3-month, Donchian).

## 3. Diagnostics (part A)

**A1, does a strategy's recent performance predict its next month?** Yes in rank, but not usefully.

| Tier | Trailing window | Mean rank correlation with next month | t |
|---|---|---|---|
| UNIT | 12 months | 0.050 | 3.55 |
| UNIT | 36 months | 0.055 | 5.23 |
| FACTOR | 12 months | 0.028 | 0.97 |
| FACTOR | 36 months | 0.061 | 2.05 |

The persistence does not reach the top of the ranking. Static selection of the best 5 units lost on 2019-2025
(t -2.10). The likely reason, not tested separately, is that the lasting differences are costs: high-turnover
rules keep losing.

**A2, do regime effects repeat across decades?**
- **Pre-registered (200 shifts):** none passes BH.
- **A design flaw:** with 200 shifts the smallest possible p is 0.005, above the first BH threshold (0.10 / 33 =
  0.003), so a single feature could never pass.
- **Exploratory follow-up with 2,000 shifts (not counted):**
  - yen implied volatility (UNIT tier): correlation 0.27, p = 0.0005;
  - the month's VIX change (FACTOR tier): correlation 0.66, p = 0.0005.
  - Both are below that threshold. Yen risk reversals (p = 0.007) and the rest are not.
- **What it means:** the pattern of which strategies do relatively better in high- versus low-yen-volatility months
  repeated. But levels shifted between decades (the dip-buying family flipped sign), and the M3 splits on those
  features did not make money (median 2019-2025 t: JPY_IV -1.09 / -1.60 / +0.69 and DVIX -0.65 / -1.42 / +1.03 for
  CFG / UNIT / FACTOR).

## 4. What options data adds (part O)

**O2, options in the regime methods (27 comparisons).**
- The feature set of the chosen M2 / M4 / M5 settings was swapped, with everything else fixed. No comparison
  passes BH; the largest difference has t 1.72.
- **ALL against price only, in the two-state HMM on the styles:** Sharpe 0.64 against 0.25. But adding the CME
  volatility to it lowers the Sharpe again, to 0.26 (t -1.98).

**O3, options features as timing signals for the nine styles (117 tests).**
- 0 count; the best has p = 0.004 against a first BH threshold of 0.00085.
- The share with t >= 1.645 was 3% for the CBOE features and 7% for the CME ones.

**A note on the signed tilt (exploratory).** The pre-registered tilt can short a style whose return is already net
of costs, and then earns those costs as if they were profit.
- This inflated some price-feature results, which were reported only for comparison. Example: realised volatility
  timing daily reversal had t 3.54.
- With long-only tilts (follow-up X2), no feature-style pair passes BH in any group.
- **Long-only rules that made money (t >= 1.645, 11 of them):** all are on dip-buying or the dollar, which made
  money on their own in 2019-2025. None beat simply holding that style at the same average weight (difference
  t <= 1.66).
- **The largest gain over holding:** realised volatility timing daily reversal (difference t 2.93). Holding daily
  reversal mostly in high-volatility months loses less than always holding it. The rule itself is not significant
  (t 1.39).

**Earlier options results, for completeness (`docs/test_ledger.md`)**
- **Selling US index options (the variance premium):** the one real options premium; it is crash-prone.
- **92 option structures on 106 underlyings, optimised walk-forward:** not significant.
- **FX option selling on CME:** the premium is eaten by costs.
- **Bitcoin variance premium:** t 1.76.
- **Options-implied FX signals (risk-reversal momentum, strike pinning, dealer gamma, implied Hurst):** failed.

## 5. A filter for the weekly dip-buying trade (part D)

For each of the 33 features, the dip-buying family F7 was held only in the tercile where it did best in 2008-2018.
- **Result:** no filter improved 2019-2025; the best has difference t 0.44, and 30 of 33 were negative.
- **The clearest warning:** in 2008-2018, F7 made its money in the top VIX tercile (mean +1.19 volatility units a
  month, against -0.17 and -0.09 in the others). On 2019-2025 that filter kept 35% of months and earned nothing in
  them (-0.001); all of F7's 2019-2025 gains (+0.17 a month) came in the months it would have skipped.
- **So the D7 forward test stays unfiltered**, as registered.

## What this means

**Across sixteen rounds the answer is consistent.**
- The FX strategies that exist in books, forums, EAs and the literature do not have edges that can be timed with
  market-state information. That information includes options-implied volatility and skew.
- Their good and bad periods are real, but they look like slow drifts in average returns. They do not look like
  switches that repeat with an identifiable state.

**What remains**
- **The Gotobi trade:** forward test running.
- **Weekly dip-buying (Double Seven and its family):** forward test running, unfiltered.
- **Data not yet tried:** the OANDA order and position books (needs your token), and the sterling / AUD / CAD / CHF
  CME options. Those options would cost USD 60 of Databento credit. The results here (euro and yen options added
  nothing) make them unlikely to change the answer.

Not financial advice.
