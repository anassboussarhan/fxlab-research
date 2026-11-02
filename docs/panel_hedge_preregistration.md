# Pre-registration: hedging the New York leg and pre-entry signals from the panel data (2026-11-02)

**Status when this was written.** The 2019-2025 index, bond and commodity M1 data (panel run 41447353092) were still
downloading. None of it had been read, and no file from it existed locally.

**Purpose.** The panel lab (`scripts/gotobi_panel_lab.py`) only lets a hedge count if it helps on 2007-2018. Bond and
index CFDs have no history that far back, so they could never qualify. This document adds a separate test for them.

**Why.** Losing nights are driven by the New York session: falling US yields and falling stocks account for 79% of
the night's variance (`docs/untested_batch.md` 9.16). The Gotobi gain itself arrives steadily from 01:00 to 07:00
Tokyo (`docs/gotobi_accuracy/conditional.csv`). So a hedge that removes the US-rates and equity shock could cut the
variance without touching the gain.

## Data

- B0 = the frozen basket with the 80 bp stop, net of costs.
- Instruments: USTBONDTRUSD, BUNDTREUR, USA500IDXUSD, USATECHIDXUSD, JPNIDXJPY, XAUUSD.
- Fills and costs are those of the panel lab.
- Development nights only: **fit on 2019-2022, check on 2023-2025.** Nothing from 2026 is used.

## H: hedges (6 instruments × 2 windows = 12 tests)

**Windows.**
- (a) The whole night, 14:00 UTC to 00:53 UTC.
- (b) The New York leg only, 14:00 UTC to 21:00 UTC.

**Construction.**
- The hedge position is -beta × X.
- beta is the OLS slope of B0's net return on X's mid return over that window, on 2019-2022 Gotobi nights.
- The hedge leg pays its own spread, commission and swap.

**A hedge counts only if all three hold on 2023-2025:**
1. Hedged t ≥ B0 t + 0.3.
2. Standard deviation at least 10% lower.
3. Mean at most 0.5 bp lower.

It must also raise t on 2019-2022. Holm correction is applied over the 12 tests at 10%.

**If one counts.** It is run through the FTMO 2-Step 5x simulation in `scripts/prop_best.py` and reported next to B0.
It is then frozen as a secondary forward rule, starting from the night after it is frozen.

## F: pre-entry signals (6 instruments × 2 lookbacks = 12 tests)

**Signals.** Each instrument's mid return over two lookbacks before entry:
- 12:00 to 14:00 UTC.
- 08:00 to 14:00 UTC.

**Test.** Spearman correlation with B0's net return.

**A signal counts only if:**
1. It is significant on 2019-2022 (p < 0.05).
2. It has the same sign on 2023-2025 with one-sided p < 0.05.
3. It survives Benjamini-Hochberg at 10% over the 12 tests.

**If one counts.** The rule "skip the nights in the worst train tercile" is reported (kept vs skipped, win rate, mean,
prop simulation). It may then be frozen as forward-v3, starting from the night after it is frozen, and tested next
to forward-v1 and forward-v2, never replacing them.

## Expected size of the effect, stated before the run

- **H:** if a hedge removes 20-30% of the variance, the per-night t rises 10-20% and the win rate by about 2-4 points.
- **F:** features of this kind have given AUC 0.58-0.61 so far.
- With about 170 check nights, a "counts" verdict needs a large effect. The likeliest outcome is that nothing counts.
