# Edges batch: results (interim, 2026-11-03)

The rules are in `docs/edges_batch_preregistration.md`, fixed before the data was read. Code:
`scripts/edges_lab2.py` (CI run 41636450530, branch `edges-batch`), `scripts/edges_e1.py`, `scripts/edges_e12.py`.
The full table is `docs/edges_batch/edges_batch.csv`. Benjamini-Hochberg runs over every test with a p-value.

**Nothing counts after correction.** Two tests met their own rule before correction, on very few events:
- the ECB pre-decision drift: 19 check meetings, +19.6 bp, t 2.0;
- the CPI surprise trade in USD/JPY: 11 check releases, t 1.8.

Both have q = 1.0 over the batch.

Net is after costs, in bp a trade; "fit" and "check" are the two halves of the split.

| Test | Fit: n, net (t) | Check: n, net (t) | Verdict |
|---|---|---|---|
| E1 sizing by the filter's probability (Gotobi) | - | FTMO 5x P(paid) at the 2026 edge 60% vs 49% (all nights) and 44% (skip a third); t 3.6 vs 4.0 | does not meet its rule (t); added to the forward test as exploratory FW1 |
| E2 cheaper Gotobi entry (limit) | 700, -0.04 (-0.3) | 437, +0.07 (0.6) | no gain |
| E3a Japan's weekly bond purchases | rho -0.01 | rho +0.01 | nothing |
| E3b Japan's trade balance | rho -0.03 | rho -0.12 | not in the fit period |
| E4 yuan fixing, USD/CNH | 263, -4.4 (-3.8) | 85, -3.5 (-0.8) | loses |
| E6a gold options expiry day | 100, -3.7 | 67, -5.7 | loses |
| E6b gold, day before expiry | 97, +15.8 (1.5) | 65, -17.2 (-1.9) | reversed |
| E7 Treasury auctions, USD/JPY and gold, before and after | -3 to +0.7 | -8.4 to +7.0, all |t| < 2 | nothing |
| E8 BoJ pre-decision drift | 95, +4.4 (0.8) | 64, -12.4 (-2.3) | reversed |
| E8 ECB pre-decision drift | 28, +5.6 (0.8) | 19, +19.6 (2.0) | meets its rule; q 1.0; 19 meetings |
| E9 MSCI review days, USD/JPY and EUR/JPY | +1.7 to +5.5 | -12.7 to +7.8, all |t| < 2 | nothing |
| E10 surprises: CPI, USD/JPY | 5 | 11, +29.5 (1.8) | meets its rule; q 1.0; 11 releases |
| E10 surprises: payrolls, retail sales | - | - | fewer than 10 releases with |z| >= 1 |
| E11a hedge-fund positioning (TFF), yen / euro | +9.5 / +8.3 (t 1.0) | +1.9 / -3.9 | nothing |
| E11b search-interest spikes | 9-14 months | 4-24 months, reversed | nothing |
| E12 lower leverage after a 4% drawdown | - | P(paid) 43% vs 45% (fixed 5x), 23% (3x); accounts lost 32% vs 81% | does not count |
| E13 short after the fix, after a large rise | 175, -0.2 | 76, +1.5 (1.1) | nothing |
| E14 end-of-day momentum after large moves, gold / FX | +0.5 / -2.6 | -3.7 (-3.8) / -1.3 (-4.4) | loses |
| E15 gold before Dhanteras; silver before the Chinese New Year | 18 years each | 7 years each | too few years to judge |
| E16 emerging-market carry | 32 months, +30 | 22 months, -11 | nothing (data gaps; no TRY) |
| E18 round-number limit orders | -2.1 to -6.6 (t -6 to -21) | -1.6 to -3.9 | loses |
| E19 night fade after the rollover | -3.0 to -5.3 | -2.5 to -3.9 (t -13 to -21) | loses |
| E23 currency strength, 5 / 20 days, both directions | -5.5 to -10.1 | -10.3 to -13.9 | loses |
| E24 G10 carry 2019-2025 (and with a VIX filter) | -4.0 / -15.1 a month | -8.9 / -2.6 a month | nothing |

**Untestable:**
- the BoE (its pages are rendered by script);
- ISM (not in the forecast dataset);
- Akshaya Tritiya (no dates);
- TRY carry (no USD/TRY bars);
- gold positioning (not in the TFF report);
- option expiries (DTCC only has recent months);
- retail positioning (no OANDA token).

**Data limits:**
- The USD/CNH, USD/MXN, USD/ZAR and USD/SGD files have gaps (2013-2015) and stop in 2022-2023, because Dukascopy throttled the downloads.
- The forecast dataset covers 2017-2023 only.

**Still running:**
- E20 / E22 (the confidence curve and timeframe agreement): run `ml-confidence`.
- E21 (the catalog on 2019-2025): run `catalog-rep`.
- E17 (the gold calendar next to Gotobi): run `edges-extra`.
- The panel test period, and the hedges and pre-entry signals: they wait for the bond and index downloads.
