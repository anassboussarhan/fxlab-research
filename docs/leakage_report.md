# Leakage validation report

Every catalog entry is run through an automated causality test on real data. A value or signal at bar t that changes when bars after t are deleted is reported as lookahead.

## Features - prefix-invariance of feature values (real EURUSD H1, 2 cut points)

passed 266 / failed 0 / not run 0 / blocked (missing data) 0


## Strategys - prefix-invariance of trading signals (real EURUSD H1 + panel)

passed 66 / failed 0 / not run 0 / blocked (missing data) 1

* blocked: S.XA.CARRY_FILTER

## Models - prediction causality (prefix of test block reproduces outputs) on real EURUSD H1 features

passed 69 / failed 0 / not run 0 / blocked (missing data) 0


## Additional structural controls

* Labels expose `label_end`; walk-forward and purged k-fold splits purge overlapping labels and embargo the test boundary (`tests/test_validation.py`).
* Every loader refuses holdout timestamps (>= 2026-01-01); the store holds no 2026 data.
* Model-driven strategies train inside a purged walk-forward; signals before the first test block are flat.
* Issues found during development and fixed: test-block normalisation in time-series model probabilities and in the generic signal mapper (caught by the model causality test), contemporaneous exogenous regressors in SARIMAX (found in code review; now lagged one bar), and an incomplete final walk-forward block in model-driven strategies (caught by the strategy causality test).
