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

## Second use: the stop chosen on 2007-2018 (run 41423405774, pre-registered in the addendum)

The exit rule `scripts/gotobi_risk_lab.py` chose on 2007-2018 (stop 0.75 W; runner-up stop 40 bp) against the
no-stop rule, on the same 2026 Gotobi days (`scripts/holdout_2026_stop.py`). Open loss = the worst mark-to-market
loss while the trade is open; CVaR5 = the average of the worst 5% of those.

| Series | Exit | Trades | Mean (bp) | t | Winners | Worst trade (bp) | Worst open loss (bp) | CVaR5 open loss (bp) |
|---|---|---|---|---|---|---|---|---|
| Basket | no stop | 53 | **+2.04** | 0.86 | 64% | -40.9 | -110.4 | -79.3 |
| Basket | stop 0.75 W | 53 | **-1.07** | -0.48 | 57% | -30.9 | -30.9 | -29.7 |
| Basket | stop 40 bp | 53 | -0.83 | -0.33 | 60% | -40.9 | -43.4 | -42.7 |
| USD/JPY | no stop / 0.75 W / 40 bp | 46 | +2.41 / -1.21 / -0.76 | | 59% / 52% / 54% | -67.7 / -32.5 / -40.9 | -112.3 / -32.5 / -43.2 | -82.5 / -31.5 / -42.4 |
| EUR/JPY | no stop / 0.75 W / 40 bp | 51 | +2.01 / +0.17 / -0.50 | | 61% / 59% / 59% | -47.9 / -26.8 / -40.9 | -108.4 / -30.4 / -43.6 | -83.8 / -28.9 / -42.8 |

**Verdict (as pre-registered): both stops cut the losses (open-loss CVaR5 -79 -> -30 / -43 bp) and neither keeps
the gains** (the mean falls from +2.0 bp to below zero; the threshold was 75% of the no-stop mean). The 2026 trades
fell overnight and recovered into the 09:55 fix: the no-stop basket's deepest open loss was -110 bp while its worst
closed trade was -41 bp. The same thing sank H1 (the 0.5-range stop). Wider stops (1.5-2 W, 100 bp) were not run on
2026 and will not be: 2026 has now been used twice.
