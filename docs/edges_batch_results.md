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
| E17 gold calendar trades (turn of month, Chinese New Year) at 2x next to Gotobi at 5x | - | P(paid) 35% vs 45% (Gotobi alone) at the 2026 edge; 44% vs 80% at the 2019-2025 edge | hurts: the gold days add losses that end accounts |
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


## E20 / E22 / E21 (CI runs 41637369890 + 41647033603, 41647033503)

- **E20, the confidence curve: the edge barely grows with confidence.**
  - Median gross per trade by cut-off (top 10% / 5% / 2% / 1% of calls):
    - H1: +1.00 / +0.87 / +1.04 / +1.02 bp;
    - M30: +0.73 / +1.09 / +1.37 / +1.20 bp.
  - Median net is about -1.9 to -3.1 bp everywhere.
  - On 2019-2025, 0-12% of the tests are positive.
  - Two series meet the rule's own conditions (net rising with confidence, t >= 1.645 at 98% or 99%), but both
    fail the correction over all tests (q = 1.0):
    - EUR/USD M30 at 99%: +0.74 bp, t 1.75;
    - USD/JPY H4 at 98%: +3.4 bp overall, t 1.96. That is -0.1 bp on 2010-2018 and +6.9 bp (t 3.3, 248 trades) on
      2019-2025.
  - The second is a lead to watch, not an edge.
  - **Quantile abstention** (trade only when the 25% quantile beats the cost) is no better.
- **E22, agreement across timeframes:** median -2.2 bp on H1, no better than without it.
- **E21, the catalog on 2019-2025** (10 families x 12 markets x H1/H4, plus the night fade and gold breakout): none
  after correction.
  - The best are cross-asset statistical arbitrage on USD/CHF H4 (+1.2 bp a day, t 2.1) and the change-point rule
    on gold H4 (t 1.8), both q = 1.0.
  - Median Sharpe by family is -1.0 to +0.03.
- **E5, FX option expiries:** untestable. The DTCC files exist only for recent months; 3 of the 4 sampled days
  were refused.

**A data bug found and fixed:** the first E20 and E21 runs failed on the instruments whose 2019-2025 files come only from
Dukascopy candles (AUD/NZD, EUR/GBP, GBP/CAD, gold) and on every cross-asset family. Those files lack the spread
columns of the older ones. The loader (`fxlab/data/store.py`) now aligns the columns, and both have been rerun.

**Still running:**
- The panel test period, and the hedges and pre-entry signals: they wait for the bond and index downloads.


## Addenda 3-5: ML aimed at costs, JGB auctions, Fed direction (2026-11-03, CI runs 41666606308, 41667582897, 41670214634, 41672048522)

All on 2007-2025 data; 2026 untouched. None of these counts.

- **M1, weekly model pooled over the 28 pairs** (`scripts/ml_weekly_lab.py`, branch `misc-weekly`).
  - Features: momentum over 1/4/12/26 weeks, carry, its 3-month change, volatility and its ratio, the 52-week range,
    VIX, S&P 500, TFF, and the calendar.
  - Trades: top 5 pairs above the fit-year cut, held a week, real swaps. 1,549 trades.
  - 2011-2018: -3.8 bp a week (t -1.2). 2019-2025: -2.5 bp a week (t -1.2), -7.1 bp per trade, 46% winners.
  - Weekly rank IC of the prediction: +0.009 and -0.014. The model carries no information about next week.
  - Timing note: 21:00 UTC Friday is the summer market close, so fills are at 20:45 UTC. The S&P 500 is the
    Dukascopy CFD, which starts in 2013.
- **M2, cost-aware meta-labels** (`scripts/ml_meta_lab.py`, branch `misc-last`).
  - Method: predict each trade's gross from its entry features; keep the trade if the prediction exceeds 1.2 times
    its cost.
  - Net per trade on 2019-2025, all trades vs kept trades:

    | Rule | All trades | Kept trades | t (kept) |
    |---|---|---|---|
    | I7 night fade | -8.5 bp | -4.1 bp | -12.2 |
    | I9 gold breakout | -0.9 bp | -1.0 bp | -0.5 |
    | E14 end-of-day momentum | -1.6 bp | -1.7 bp | -2.3 |
    | E18 round numbers | -2.2 bp | -1.4 bp | -2.0 |

  - The filter raises the kept trades' gross only slightly. The prediction's correlation with the gross is 0.00 to
    0.18, which is far too little to clear costs.
- **M4, one pooled intraday model across 12 markets, and the FS5 large-move filter** (`scripts/ml_pooled_lab.py`,
  branch `ml-pooled`).
  - H1: none of 60 market x rule rows is net > 0 with t >= 1.645 on 2019-2025.
  - H4: 2 of 60 are: USD/JPY at the 99% cut (+6.2 bp, t 2.2, 166 trades) and USD/JPY with the large-move filter
    (+4.7 bp, t 1.95). Neither survives correction over the 120 rows.
  - It is the same USD/JPY H4 lead as in E20, seen again by a different model. It goes to the forward watch list,
    not into trading.
  - Gold H4 at 98% shows +9.5 bp, but t is only 1.4.
- **E25, 10- and 30-year JGB auctions.**
  - Dates: 659 auctions from the MoF history spreadsheet, now saved as `data/jgb_auctions.csv`; 402 were priced for
    2007-2025.
  - Before the result (00:00 -> 03:00 UTC): direction from the fit period is short; 2019-2025 gives -3.2 bp, t -2.1.
  - After the result (03:40 -> 08:00 UTC): direction is long; -2.0 bp, t -1.2.
  - Note: the 2007-2018 files have no 00:00-01:00 UTC bars, so the 00:00 entry takes the first bar up to 01:05.
- **E26, the dollar before Fed decisions in the expected direction** (information only).
  - Events: 18 decisions in 2017-2022 with an expected change. The 2023 rows have no forecast.
  - Result: -18.6 bp per event (t -1.4), 39% winners. Before the 15 expected hikes the dollar fell on average
    (-24 bp).

**Still running:**
- FS1, FS2, FS4, FS5 information (`fs-lab`, 138 jobs).
- E4 / E16 on the re-downloaded files.
- FS3, M3, the panel test period and the hedges: they start when the bond and Brent downloads finish.


## Hedges (H / F), FS3 and M3: the bond, index and Brent data (2026-11-03, CI runs 41676706418, 41679528174)

**Data caveat.** The Dukascopy candle files are incomplete. US Treasury bond: 2018-01 to 2024-04. Bund: 2018-05 to
2024-08. US500, NAS100, JPN225: 2012 to 2024-2025, with gaps. So the 2023-2025 check of the hedges rests on 62-69
nights for bonds and the Nikkei, instead of about 165.

- **H, hedging the Gotobi night with a New York-session leg** (`scripts/panel_hedge_lab.py`, branch `panel-hedge`):
  none counts.
  - Best case, short US bonds over the whole night, on 2023-2025:
    - the night's sd falls 17% (28.6 -> 23.6 bp);
    - the mean falls from 4.4 to 2.8 bp, and t falls from 1.27 to 0.98.
  - The rule needs t to rise, so it fails. Indices and gold fail too.
- **F, pre-entry signals** (12 tests): none.
  - On 2019-2022 a rise in US stocks over 08:00 -> 14:00 UTC went with a weaker night (rho about -0.2,
    p 0.001). On 2023-2025 it is -0.21 (one-sided p 0.03) for US500 but fails the BH correction (q 0.32).
  - Weak lead, not a rule.
- **M3, cross-asset features in the 8.5 model, H1** (2012-, 60/40 split; check from 2020): the extra features raise
  net slightly, but every version loses on the check:

  | Market | Without cross-asset | With cross-asset |
  |---|---|---|
  | USD/JPY | -1.8 bp | -1.6 bp |
  | EUR/USD | -2.0 bp | -1.4 bp |
  | Gold | -2.1 bp | -1.1 bp |

  Does not count.
- **FS3, new intraday sources** (528 checks on USD/JPY, EUR/USD and gold, H1-D1): nothing counts. It is the first
  test where the information is clearly real but too small to trade.
  - Information: 9 of the 528 IC tests survive BH at 10%. All but one are cross-asset returns. Examples:
    - the Bund's 24-bar return predicts gold 24 H4 bars ahead (z 4.3);
    - US500's last H4 bar predicts gold (z 3.7);
    - the 3-month rate-gap change predicts USD/JPY 24 days ahead (z 3.2, +62 bp per trade, t 1.58).
  - Trading: none of them clears costs with t >= 1.645. The decile trades with net > 0 and t >= 1.645 (15) are not
    the ones with a significant IC.
  - Leads for the forward watch list:
    - the rate-gap change on USD/JPY D1 and EUR/USD H4 (t 2.9 net, but IC z 0.4);
    - the Bund -> gold link.


## Gotobi panel, 2019-2025 check (CI run 41676706514, branch `gotobi-panel`)

The 2007-2018 screen of 54 instruments picked six carriers of the Gotobi night: CHF/JPY, EUR/JPY, HKD/JPY,
MXN/JPY, USD/JPY and gold. Each was then checked on 2019-2025.

- **Confirmed on 2019-2025: USD/JPY, EUR/JPY and HKD/JPY.**

  | Pair | Night net, in its direction | t |
  |---|---|---|
  | USD/JPY | +5.0 bp | 3.3 |
  | EUR/JPY | +4.8 bp | 3.0 |
  | HKD/JPY | +4.0 bp | 2.1 (170 nights) |

  HKD is pegged to the dollar, so HKD/JPY is USD/JPY under another name. It adds nothing new.
- **Not confirmed:**
  - CHF/JPY: +1.7 bp, t 1.4.
  - MXN/JPY: no 2019-2025 data.
  - Gold: -8.1 bp.
- **Baskets on 2019-2025:**

  | Basket | Mean per night | t |
  |---|---|---|
  | B0 = USD/JPY + EUR/JPY (the frozen trade) | +4.7 bp | 3.35 |
  | All six carriers, equal weight | +1.3 bp | 1.0 |
  | All six carriers, inverse variance | +2.3 bp | 1.9 |
  | B0 + gold | +1.0 bp | 0.7 |

  Widening the basket only dilutes the trade, so **B0 stays as it is**.
- **Hedge legs** (17 tested, from TRY/JPY to the indices): none raises B0's t on 2019-2025. The yen crosses cut the sd
  but also remove the edge, because they carry the same yen move.
- **Data notes:**
  - Bund failed to load (no date column in its pieces).
  - The US bond file has only 12 nights before 2019.
  - Most index and commodity series start in 2012-2013.


## FS1, FS2, FS4, FS5: feature settings, cells, pairs, volatility targets (CI runs 41670196230 and recheck 41687280097)

**Correction before reading the numbers.** The first run counted every signal bar as a trade, even while an h-bar
position from an earlier bar was still open. That inflates t: a D1 signal held 24 days was counted on each of the
days. The rule says to hold the position for h bars, so the recheck takes one position at a time and needs 20
trades for a t.

The recheck reran the 70 FS1 jobs that have a BH-significant IC, and every FS2 job. The IC side (z, BH) does not
change. FS1's correction is still taken over all 58,920 tests.

- **FS4, pairs of features:** 9,120 tests, none significant after BH. Does not count.
- **FS5, volatility targets** (information only): 59 of 1,920 are significant. Nearly all are moving-average levels
  on yen pairs predicting the size of the next moves. That is most likely the 2022-2024 regime (weak yen, larger
  swings), not a tradable signal.
- **FS2, session and volatility cells:** 4 rows passed in the first run. **After the fix: 0 of 8,448.**
- **FS1, settings and transforms chosen inside the walk-forward** (58,920 tests on 2019-2025 folds):
  - 318 ICs survive BH at 10%.
  - Of those, 66 also trade net > 0 with t >= 1.645 after the fix (90 before it). Rows: `docs/fs_lab/fs1_passing_rows.csv`.
  - **By the letter of the rule, these 66 count. I do not treat them as edges, for four reasons:**
    1. **They fail on 2010-2018.** The same procedure (same feature, same selection inside each fold) scored on
       the 2010-2018 folds gives a median IC z of 0.21 and net < 0 in 46 of the 66. A real effect should show in
       both periods. These appear only after 2019.
    2. **They are few trades:** a median of 72 positions in seven years (22-327). Weekly rows have 22-80.
    3. **They are not independent.** 29 involve the pound and 19 the New Zealand dollar, many on the same pairs
       (GBP/NZD, GBP/CHF, AUD/CAD, EUR/GBP), so they are a handful of moves counted many times.
    4. **The expected number of false ICs among the 318 is about 32** (10% FDR). The economic filter keeps the
       ones that happened to trend in 2019-2025.
  - **Status:** a forward watch list (FS1-W), not a trading plan. Promoting any of them would need its own
    pre-registered forward test on data after 2026-11-03; the 2026 holdout stays locked.


## External verification report (2026-11-03) and the timing audit of the frozen basket (CI misc-strict)

The report audits the trade-log archive and **F1**, the older USD/JPY-only 03:00 -> 09:55 Tokyo variant. Its
findings on the archive (nothing survives costs or correction) agree with this repository's. For the frozen basket
B0 (USD/JPY + EUR/JPY, 23:00 -> 09:53 Tokyo, 80 bp stop):

- **Exit timing** (`scripts/gotobi_strict_timing.py`):
  - **2019-2025:** 835 of 838 leg exits are exact (within 2 minutes); 3 are late, at most 7 minutes.
  - With strict endpoints (416 nights) the gross is the same, +5.41 bp, t 3.80.
  - **2007-2018:** every exit is 7-8 minutes late, because those files have no 00:00-01:00 UTC bars. The 2007-2018
    figures are for an exit at about 01:00 UTC, not 00:53.
- **Extra cost** (net per night, t), from `docs/gotobi_losses/trades_basket_stop_80_bp.csv`:

  | Period | Nights | Recorded | +1 bp | +2 bp |
  |---|---|---|---|---|
  | 2007-2018 | 690 | +4.40 (2.91) | +3.40 (2.25) | +2.40 (1.59) |
  | 2019-2025 | 423 | +4.96 (3.48) | +3.96 (2.77) | +2.96 (2.07) |
  | 2023-2025 | 182 | +3.05 (1.21) | +2.05 (0.82) | +1.05 (0.42) |
  | 2024-2025 | 121 | +3.02 (0.95) | +2.02 (0.64) | +1.02 (0.32) |

- **As the report says, the recent years alone do not establish the edge.** 2023-2025 is positive but not
  significant, and an extra 1-2 bp of real cost removes most of it.
- **The 2026 holdout has been opened once** (`docs/holdout_2026_preregistration.md`), so it is no longer untouched.
  The genuinely new sample is the forward test from 2026-11-05.


## FS6 cross-asset composite, and the cost audit of the frozen basket (CI runs 41723718676, 41724161522)

**FS6 does not count** (12 tests, 0 after BH).
- On the 2012-2019 fit period, only 0-3 candidates per market cleared |t| >= 2, so 7 of the 12 tests selected
  nothing.
- Where something was selected, the check-period (2020-2025) net is:

  | Market, horizon | Signal kept | Net | t |
  |---|---|---|---|
  | USD/JPY D1, 6 bars | Brent | +0.2 bp | 0.03 |
  | USD/JPY D1, 24 bars | JPN225 | +16 bp | 0.40 (40 trades) |
  | Gold D1, 24 bars | JPN225 | +47 bp | 0.53 (31 trades) |
  | Gold H4, 6 bars | USTBOND | -5.4 bp | -1.3 |

- Limit: the US bond and Bund files start in 2018, so the 2012-2019 fit period has almost no bond history. The
  Bund -> gold and rate-gap leads from FS3 could not be selected honestly.

**Cost audit** (`scripts/gotobi_costs.py`, `docs/prop_best/costs_audit.json`).
- **Dukascopy spreads at the basket's entry (14:00 UTC) and exit (00:53 UTC), median / 90th percentile, pips:**

  | Pair | 2007-2018, entry | 2007-2018, exit | 2019-2025, entry | 2019-2025, exit | 2023-2025, entry | 2023-2025, exit |
  |---|---|---|---|---|---|---|
  | USD/JPY | 0.7 / 1.9 | 0.6 / 1.5 | 0.4 / 0.9 | 0.5 / 0.9 | 0.6 / 1.4 | 0.6 / 0.9 |
  | EUR/JPY | 1.4 / 3.2 | 1.4 / 2.5 | 0.8 / 1.7 | 0.8 / 1.5 | 1.1 / 2.9 | 1.15 / 1.55 |

- **The model charges 1.5 x those spreads plus 0.2 pip and 0.7 bp commission.** That is a round trip of 1.7 bp on
  2019-2025 and 2.0 bp on 2023-2025.
- **Published prop figures** are cheaper or similar:
  - FTMO (undated): USD/JPY about 0.1 pip and EUR/JPY about 1.3 pips in normal hours.
  - Commission: $5 per lot ($3 in older FTMO material).
  - Spreads are not overstated by the model.
- **The swap is the optimistic part.** The modelled rate swap adds +0.4 bp a night on 2019-2025 and +1.0 bp on
  2023-2025, because USD rates were high. Retail long swaps now are about +1.7% a year on USD/JPY (OANDA) and 0 or
  negative on EUR/JPY. Prop firms do not publish theirs.
- **Basket net per night, t** (fixed spread at both ends + 0.1 pip slippage each side):

  | Scenario | 2019-2025 | 2023-2025 |
  |---|---|---|
  | Model (Dukascopy, rate swap) | +5.08 (3.57) | +3.63 (1.48) |
  | 0.5 pip, $5 commission, no swap | +5.37 (3.79) | +3.58 (1.47) |
  | 1.0 pip, $5 commission, no swap | +5.00 (3.52) | +3.26 (1.34) |
  | 1.0 pip, $7 commission, swap -0.5 bp | +4.30 (3.03) | +2.56 (1.05) |
  | 2.0 pip, $7 commission, swap -0.5 bp | +3.54 (2.50) | +1.91 (0.78) |

- **Reading:** the costs were not overstated. At realistic prop costs (about 1 pip and $5, no swap) the 2019-2025
  edge is +5.0 bp (t 3.5). 2023-2025 stays positive but not significant in every scenario.
