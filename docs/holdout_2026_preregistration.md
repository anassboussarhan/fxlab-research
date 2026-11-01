# 2026 holdout: pre-registration

Written and committed on 2026-11-01, **before any 2026 price, positioning or rate data was downloaded or read**,
on Anas's explicit instruction ("Unlock 2026"). Every rule below is frozen exactly as it stood in the development
work (data to 2025-12-31). The holdout is the period 2026-01-01 to 2026-10-31 (the last complete month). It is
used once: the results are reported as they come out, nothing is re-tuned afterwards. If a coding error is found
after the run, the corrected run is reported next to the first, never in place of it.

Machine-readable copy: `results/frozen_candidates.json` (its `unlock_token_sha256` opens the M1 loader; the
phrase is set only in `.github/workflows/holdout_2026.yml`).

## Frozen rules

| ID | Rule (as in the development code) | Development result (net, after costs) | Source |
|---|---|---|---|
| **H1 (primary)** | **I3 USD/JPY Gotobi overnight:** on Gotobi days (5th, 10th, 15th, 20th, 25th, month end; the previous Tokyo business day when that is a weekend or a Japanese holiday) long from 23:00 Tokyo the evening before to 09:53 Tokyo; stop 0.5 x the mean daily range of the previous 20 FX trading days; no target. Entries need a quote within 10 minutes (so Monday Gotobi days, whose entry falls on Sunday, are skipped) | +4.45 bp a trade, t 3.89, 1,074 trades (2007-2025) | `scripts/untested_intraday_lab.py` (`i3`) |
| H2 | F1 USD/JPY: long 03:00 -> 09:55 Tokyo on Gotobi days | +2.13 bp, t 2.05, 413 trades (2019-2025) | `scripts/gotobi_oos.py` |
| H3 | USD/JPY 23:00 -> 09:53 Tokyo, no stop | +4.91 bp, t 4.08, 1,077 trades (2007-2025) | `scripts/gotobi_opt_lab.py` (BASE) |
| H4 | EUR/JPY 23:00 -> 09:53 Tokyo, no stop | +4.40 bp, t 3.00, 1,037 trades (2007-2025) | same |
| H5 | Yen basket: the average of H3 and H4 on each Gotobi day | (H3, H4 above; walk-forward basket +5.06 bp, t 4.51) | `scripts/flow_edges_lab.py` (T3 form) |
| H6 | Nasdaq-100 overnight: QQQ close -> next open, Monday-Thursday nights only (consecutive trading days), minus 1 bp and CFD financing of (T-bill + 2.5%) / 360 a night | +5.4 bp a night, sd 89 bp, 1,376 nights (2019-2025) | `scripts/prop_mix_sim.py` (O) |
| H7 | R1 60/40 rebalancing front-run, calendar signal: SPY - IEF spread, -clip(drift / 1.5%) in the last 4 trading days of the month, the month-end drift reversal on the first day; 1 bp a unit of change a leg | +2.9 bp a day, Sharpe 0.92, t 4.29 (2003-2025); after the paper Sharpe 0.62 | `scripts/rebalance_lab.py` (R1) |
| H7b | E1: the same position on the S&P 500 leg only (index CFD), 2 bp a change, CFD financing | +2.3 bp a day, Sharpe 0.82 | same (E1) |
| H8 | C5 EUR speculative pressure (Fan et al.): sign of the speculators' net position (CFTC, non-commercial, share of open interest) minus its average over the previous 52 reports, set at the start of each month and held for the month; weekly returns as in the COT lab (spot + carry - 0.75% a year mark-up, 1 bp a side) | +12.0 bp a week, t 3.1 (2004-2025) | `scripts/cot_lab.py` (C5, EUR) |

Costs, swaps and execution are exactly as in the development code (M1 bid/ask fills; crossing the quoted spread
plus 25% of it and 0.1 pip; 0.35 bp commission a side; swap from the repo's rate table, whose last month,
December 2025, carries forward through 2026). Data: Dukascopy M1 for USD/JPY and EUR/JPY (2026-01 to 2026-10,
downloaded in CI; November-December 2025 development bars for the 20-day range warm-up); Yahoo daily closes for
QQQ, SPY, IEF and ^IRX; CFTC legacy reports for 2026 and Yahoo EURUSD=X.

## Verdicts, decided now

For each test, on the 2026 trades only:
- **Confirmed:** mean > 0 and one-sided t >= 1.645.
- **Consistent:** mean > 0, t < 1.645.
- **Not supported:** mean <= 0.
- **Contradicted:** the 2026 mean is below the development mean with t <= -1.645 (t of the difference, using
  the 2026 standard deviation).

**Primary test:** H1 alone. **Overall test:** Stouffer's Z over four roughly independent strategies, H5 (yen
basket), H6, H7 and H8, each contributing its t; the set is confirmed if Z >= 1.645.

## Power, stated before the run

Ten months is short. The expected t if each rule keeps its development edge exactly:

| Test | 2026 trades (expected) | Expected t | Chance of t >= 1.645 |
|---|---|---|---|
| H1 I3 | 53 (60 Gotobi days, 7 on Mondays) | 0.86 | about 22% |
| H2 F1 | 53 | 0.73 | about 18% |
| H3 / H4 | 53 | 0.90 / 0.68 | about 23% / 17% |
| H5 basket | 53 | about 1.1 | about 29% |
| H6 Nasdaq overnight | about 170 nights | 0.79 | about 20% |
| H7 R1 | about 210 days | 0.6-0.8 | about 15-20% |
| H8 C5 EUR | about 43 weeks | 0.60 | about 15% |
| Stouffer Z (H5, H6, H7, H8) | - | about 1.65 | about 50% |

So "consistent" is the most likely outcome even if every edge is real, and a failure to confirm is weak evidence
against an edge. A **contradiction** (a 2026 mean significantly below the development mean) is the result that
would count against a rule.

## Addendum (2026-11-01, still before any 2026 data was downloaded)

- **t-statistics for the verdicts:** per-trade tests (H1-H6) the ordinary t; the daily R1 / E1 series a
  Newey-West t with 5 lags and the weekly C5 series 4 lags, as in their labs, with the variance never below the
  iid variance.
- **Code check:** `scripts/holdout_2026.py --dry 2017` runs the Gotobi rules on a development year. It reproduces
  the development numbers for 2017 (USD/JPY no stop +3.60 bp on 63 trades, EUR/JPY +5.06 bp on 63, as in
  `gotobi-opt` diagnostics), so the copied code is the frozen code.
- Mondays: as in development, a Gotobi Monday has no trade when its entry time falls before the weekly open.
