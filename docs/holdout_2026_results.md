# 2026 holdout: results (run 41347184098, one run, as pre-registered)

Period 2026-01-01 to 2026-10-31. Rules, verdicts and power: `docs/holdout_2026_preregistration.md` (committed
before any 2026 data). Data: Dukascopy M1 USD/JPY and EUR/JPY (23-67 hours a month failed to download, 5-9%, as in
the 2019-2025 data; trades whose entry or exit fell in a gap are missing), Yahoo closes, CFTC reports to 2026-10-27.

| ID | Rule | Trades | Mean (bp) | t | Development mean | t vs development | Verdict |
|---|---|---|---|---|---|---|---|
| **H1 (primary)** | I3 USD/JPY Gotobi overnight with the 0.5-range stop | 46 | **-2.55** | -0.78 | +4.45 | **-2.11** | **contradicted** |
| H2 | F1 USD/JPY 03:00-09:55 | 49 | +2.61 | 0.75 | +2.13 | +0.14 | consistent |
| H3 | USD/JPY 23:00-09:53, no stop | 46 | +2.41 | 0.75 | +4.91 | -0.77 | consistent |
| H4 | EUR/JPY 23:00-09:53, no stop | 51 | +2.01 | 0.79 | +4.40 | -0.93 | consistent |
| H5 | Yen basket (H3, H4) | 53 | +2.04 | 0.87 | +4.66 | -1.10 | consistent |
| H6 | Nasdaq-100 overnight (a night) | 165 | -3.00 | -0.44 | +5.4 | -1.22 | not supported |
| H7 | R1 rebalancing front-run, SPY - IEF (a day) | 209 | -0.15 | -0.10 | +2.9 | -1.88 | contradicted |
| H7b | E1, equity leg only (a day) | 209 | +0.55 | 0.34 | +2.3 | -1.07 | consistent |
| H8 | C5 EUR speculative pressure (a week) | 42 | +1.84 | 0.14 | +12.0 | -0.82 | consistent |

**Overall (Stouffer Z over H5, H6, H7, H8): 0.24, not confirmed** (pre-registered threshold 1.645).

- **The primary rule, I3 with its stop, failed:** -2.6 bp a trade, significantly below its development mean.
  The same entry and exit without the stop (H3) earned +2.4 bp; the stop (half the 20-day range) cut trades that
  recovered by 09:53.
- **The Gotobi flow itself is still there, at about half its development size:** every no-stop Gotobi variant
  is positive (+2.0 to +2.6 bp, 59-64% winners, 7 of 10 months positive) but none is significant on ~50 trades,
  as the power table predicted.
- **The Nasdaq overnight and the rebalancing calendar did not work in 2026.**
- Gold over the 15 days before the Chinese New Year (information only, one event): +0.99%.
- After this run 2026 is no longer an untouched holdout; later months (November 2026 on) are the next clean test.
