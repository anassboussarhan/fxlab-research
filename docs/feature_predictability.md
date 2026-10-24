# Feature predictability: do the 269 features carry information about the next move?

Code: `scripts/feature_ic_lab.py`, `scripts/feature_ic_report.py`, `scripts/feature_ic_page.py`.
Cloud run 6 computed the tests; run 7 re-aggregated them (branch `results7`, `sweep_results/feature_ic/`).

## Coverage

- 34 pairs and metals, 2007-2018 development data (EURUSD to 2022).
- 10 timeframes: M5, M15, M30, H1, H2, H4, H8, D1, TICK60 and VOLA60.
- Every output column of all 269 features, in raw form and as the deviation from its own 20-bar EMA.
- Horizons of 1, 6 and 24 bars.
- **857,202 tests** in total.

## Method

**Information coefficient (IC).** Walk-forward with 3-year training and 6-month test windows, first test in
2010.
- The training window learns the sign of the feature's relation with the next-h-bar return.
- The test window measures the signed, uncentred correlation. The feature is standardised with training
  statistics only.
- The result equals the normalised gross P&L of holding a position proportional to the feature.

**Null.** The target is circularly shifted 20 times, which keeps the feature's own persistence. The p-values use
a t(19) reference, because the null spread comes from only 20 draws. Benjamini-Hochberg runs over all tests.

**Economic size.** Half the training top-minus-bottom decile spread of the next move, measured out of sample,
divided by a retail round-trip cost.

**Two corrections to earlier work:**
- The "lift over placebo" in the feature scan was not a valid predictability measure. The noise and shuffled
  placebo columns mostly stayed flat because of the cost filter, and they had no persistence.
- The first version of this lab centred the test window on its own mean. That biased level-like features
  (drawdown reached IC 0.26). It is fixed.

## Results

| TF | Tests significant | Median edge / cost of significant tests | Significant tests covering cost |
|---|---|---|---|
| M5 | 17.6% | 0.02 | 8 |
| M15 | 4.6% | 0.03 | 4 |
| M30 | 2.1% | 0.07 | 23 |
| H1 | 1.0% | 0.10 | 37 |
| H2 | 0.7% | 0.24 | 106 |
| H4 | 0.55% | 0.38 | 167 |
| H8 | 0.52% | 1.06 | 235 |
| D1 | 0.44% | 0.98 | 189 |
| TICK60 | 0.69% | 0.12 | 60 |
| VOLA60 | 0.63% | 0.11 | 53 |

**Verdicts across the 269 features:**

| Verdict | Share of the feature's tests significant | Features |
|---|---|---|
| Predictive | 5% or more | 68 |
| Weak | 1-5% | 58 |
| Rare | under 1% | 142 |
| None | 0% | 1 (HMM regime) |

**By category.**
- Most predictive: trend (22 of 27 predictive), momentum and oscillators (12 of 22), and price-shape features
  (10 of 16).
  - The top features are close location value, Connors RSI, intrabar return, fractional differencing,
    Lee-Mykland jumps, KAMA, Ultimate oscillator, Heikin-Ashi and streak counts.
  - All of them are predictive almost only at M5 (70-94% of their M5 tests significant).
- Least predictive: information theory, learned representations, fractal/chaos, cross-asset and esoteric.

**Economics.**
- At M5-H1, the predicted move of significant tests is 2-10% of the round-trip cost.
- Above H8 the few significant tests (about 0.5%) are about as large as costs. They come mostly from level- and
  trend-like features (fractional differencing, drawdown regime, MA ribbons, session clock, month-end fix) at
  6-24-bar horizons.
  - They are concentrated in EURSEK and EURNOK: 242 of the 591 significant, cost-covering tests at H4-D1.
- As trading rules after costs, the same features had no survivors after program-wide correction (feature scan,
  section 8.1c).

Full list: `fx_feature_predictability.csv`, plus the published page "FX Feature Predictability".
