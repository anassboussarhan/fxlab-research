# Pre-registration: the edges batch (2026-11-02)

This document fixes the rules of the batch before any of its data is downloaded or read.

**Status when written.**
- The data scripts are `ci/edges_data.py` and `.github/workflows/edges_data.yml`. They had not run yet.
- The panel data (run 41447353092) was still downloading.

**Rules that apply to every test.**
- Development data only, to 2025-12-31. Nothing from 2026 is used.
- M1 bid/ask fills and costs as in the other labs.
- **Split:** 2007-2018 for train and 2019-2025 for test. A series that starts later is split at its first 60% of
  events (fit) and the last 40% (check).
- **Direction of each trade:** the sign on train, unless a published paper fixes it, as noted below.
- **A test counts only if all of these hold on the check period:**
  1. the net mean after costs is above 0;
  2. the one-sided t is at least 1.645;
  3. it survives Benjamini-Hochberg at q = 0.10 over every test in this batch together with
     `docs/panel_hedge_preregistration.md`.
- **What happens to a test that counts:** it goes through the FTMO 2-Step simulation in `scripts/prop_best.py`. It
  is then frozen as a forward rule starting the night after it is frozen, next to the existing frozen rules and
  never replacing them.
- **Missing data:** a source that cannot be downloaded is reported as "untestable". It is never replaced by a
  guess.

## Improving the Gotobi trade

- **E1. Sizing by the filter's probability.**
  - The forward-v1 walk-forward probabilities p (refitted yearly, as in `scripts/gotobi_filter_wf.py`) set the
    size: weight = p / mean(p on the training years), capped at 2.
  - This is compared with no filter and with the skip-the-lowest-third rule, on 2012-2025 walk-forward nights.
  - **Counts if:** FTMO 2-Step 5x P(paid) at the 2026-shifted edge is at least 3 points above both comparisons on
    2019-2025, and the t of the weighted night beats both.
- **E2. Cheaper entry.**
  - Post a buy limit at the 14:00 UTC bid + 25% of the spread. If it is not filled by 14:10 UTC, buy at market.
  - The exit is unchanged.
  - **Counts if:** the paired difference against the 14:00 market entry passes the test rules on 2019-2025.
- **E3. Japan's portfolio flows.** Two features, each tested with Spearman correlation against the Gotobi night's
  net, under the same rules as F in the panel pre-registration:
  - (a) the latest Ministry of Finance weekly figure for net foreign bond purchases by Japanese residents,
    published Thursdays, z-scored over 52 weeks;
  - (b) Japan's trade balance (FRED), lagged one month.

## New edges

- **E4. Yuan fixing.**
  - Signal: the official USD/CNY central parity minus the USD/CNH price at 01:14 UTC, in pips.
  - Trade USD/CNH in the direction of the signal from 01:16 to 08:00 UTC, on days with |signal| above the
    fit-period 75th percentile.
  - Also run as AUD/USD in the opposite direction.
- **E5. FX option expiry at the 10:00 New York cut.**
  - For EUR/USD and USD/JPY, take the strike with the largest notional expiring that day (DTCC public reports).
  - If it lies within 30 pips of the 12:00 UTC mid, trade towards it from 12:00 UTC to the cut.
  - Untestable if the DTCC files cannot be had.
- **E6. Gold options expiry.**
  - The COMEX gold option expiry is taken as the 4th-last business day of each month (the exchange's rule).
  - Gold's return 13:30-17:30 UTC on expiry day, and from 13:30 UTC the day before, compared with all other days.
  - The trade goes in the train direction.
- **E7. US Treasury auctions** (10-year and 30-year nominal, from fiscaldata.treasury.gov). Two windows, each traded
  in USD/JPY and gold:
  - (a) pre-auction: 13:30 UTC to 5 minutes before the auction close;
  - (b) post-auction: the close to 14:00 UTC the next day.

  The direction is from Lou, Yan & Zhang (2013): yields rise before an auction and ease after it, so the dollar
  goes up before and down after.
- **E8. Central bank meetings: BoJ, ECB, BoE** (dates from their own pages; the Fed's from federalreserve.gov,
  which also repairs the FOMC list that stopped in 2022).
  - Pre-announcement drift: from 24 hours to 15 minutes before the decision.
  - Pairs: USD/JPY for the BoJ, EUR/USD for the ECB, GBP/USD for the BoE.
  - The direction is from train.
- **E9. MSCI quarterly index review days** (last business day of February, May, August and November).
  - USD/JPY and EUR/JPY from 14:00 to 21:00 UTC on the day, and over the next Tokyo morning, 00:00-03:00 UTC.
- **E10. Economic surprises** (actual minus forecast from the public calendar dataset).
  - Releases: US non-farm payrolls, CPI, retail sales, ISM manufacturing.
  - The surprise is z-scored over the fit period.
  - Trade USD/JPY and gold from 5 minutes to 4 hours after the release, in the direction of the surprise; only
    when |z| is at least 1.
- **E11. Sentiment.**
  - (a) CFTC Traders in Financial Futures: leveraged funds' net position in JPY, EUR and gold futures, z-scored
    over 156 weeks. Trade the following week's return, with the sign (contrarian or trend) from train.
  - (b) Google Trends attention for "yen", "gold price", "forex" and "dollar": the month after an attention spike
    (above the fit-period 90th percentile).
  - (c) Retail position ratios (OANDA position book) only if a free OANDA practice-account token is added as
    the `OANDA_TOKEN` secret; otherwise untestable.

## Expected outcome, stated before the run

- Each of these has perhaps a 5-10% chance of counting after costs. The likeliest result is zero or one.
- A test that counts on fewer than about 150 check events is treated as "promising", not "confirmed", until the
  forward test.

## Addendum (2026-11-03): E12-E19, from the inventory in `docs/untested_variations.md`

These rules were fixed before any of their data was read. For E13, E14, E16, E18 and E19, the 2019-2025 M1 files
are part of the panel data run, which had not finished. The same counting rules apply (net > 0, one-sided
t >= 1.645, BH over the whole batch), and the same 2007-2018 / 2019-2025 split.

- **E12. Prop leverage that falls after a drawdown.**
  - FTMO 2-Step simulation as in `scripts/prop_best.py`.
  - Trade at 5x. Drop to 2.5x while equity is more than 4% below the phase or payout start. Return to 5x once it
    is within 2%.
  - Compared with a fixed 5x and a fixed 3x.
  - **Counts if:** P(paid) at the 2026-shifted edge is at least 3 points above both, on 2019-2025.
  - This is a sizing rule on nights already studied, so a "count" means "use it", not "a new edge".
- **E13. Short USD/JPY after the fix, only after a large rise into it.**
  - The trade: short 09:55 -> 10:30 Tokyo on Gotobi days.
  - Condition: the 03:00 -> 09:50 mid move is above its 2007-2018 75th percentile.
- **E14. End-of-day momentum, only after a large earlier move.**
  - Gold: follow the sign of the 08:00 London -> 13:00 New York move from 13:00 to 13:30 New York.
  - FX: the same for the 7 majors pooled, with the 08:00 London -> 14:30 New York move traded from 14:30 to
    15:00 New York.
  - Only when |move| is in the top 20% of the 2007-2018 days.
- **E15. Gold before Indian festival demand.**
  - Long gold over the 15 business days before Dhanteras (two days before Diwali; dates from the `holidays`
    package's India calendar) and before Akshaya Tritiya, if that package has it.
  - Silver over the 15 days before the Chinese New Year.
  - Daily data from 2000, as in the Chinese New Year test (lit3 D6).
- **E16. Emerging-market carry with a crash filter.**
  - Monthly. Hold the two highest-yielding of MXN, ZAR, TRY, CNH and SGD against USD, using three-month
    interbank rates from FRED (OECD series).
  - Flat in a month that follows a month-end VIX above 25.
  - Costs: spread + 0.35 bp commission each way, plus the rate differential as swap.
  - The split is 60/40 if the M1 history starts after 2008.
- **E17. Gold calendar trades added to Gotobi in a prop account.**
  - E6b (if it counts), the gold turn of the month and gold before the Chinese New Year, at 2x on their own
    days, next to the Gotobi trade at 5x.
  - FTMO 2-Step simulation on 2019-2025.
  - **Counts if:** P(paid) at the 2026-shifted edge is at least 3 points above Gotobi alone.
  - This is a portfolio evaluation, exploratory.
- **E18. Round numbers with resting limit orders.**
  - EUR/USD, GBP/USD, USD/JPY and gold.
  - Buy limit 2 pips above a 100-pip level (gold: 2 dollars above a 50-dollar level) when price approaches from
    above within the day, and the mirror image for sells.
  - Target 10 pips, stop 10 pips, out by the end of the day.
  - Only fills that the bid/ask path reaches count.
- **E19. Night-scalping fade after the rollover hour.**
  - The Bollinger fade of `docs/untested_batch.md` I7 with entries only from 23:00 to 02:00 UTC, so that the
    21:00-23:00 UTC rollover spreads are avoided.
  - Exits as in I7.

Not added:
- **Japanese government bond auctions:** the MoF auction calendar has not been downloaded yet. It will be
  pre-registered once its format is known.
- **Platinum and palladium:** no mechanism found.
- **FOMC in the expected direction:** about 20 decisions with a change, too few for this batch's bar. It is
  reported as information only.

## Addendum 2 (2026-11-03): E20-E24, levers across all tests

These come from `docs/untested_variations.md` section 4. They were fixed before any 2019-2025 FX or gold M1 file
was read, and the same counting rules apply. Tick-count features are dropped everywhere: the 2019-2025 candle files
have none, and dropping them keeps both periods alike.

- **E20. The confidence curve.**
  - Rerun of the fixed-barrier ML lab (8.5) with its symmetric barrier (1.5 / 1.5 ATR, 12 bars), on M15, M30, H1
    and H4.
  - Markets: 11 FX pairs and gold, 2007-2025, with the 2019-2025 panel files.
  - The score of every bar is saved.
  - Long and short cut-offs at the 90th, 95th, 98th and 99th percentiles of the calibration-year scores, set as
    in 8.5.
  - **Counts if:** a cut-off at 98% or 99% is net > 0 with t >= 1.645 on the test years (2019-2025 reported
    separately), and the net mean rises from 90% -> 95% -> 98%.
  - Also: abstain unless a LightGBM quantile model's 25th-percentile prediction of the 12-bar return is above
    the round-trip cost (long; mirror image for short).
- **E21. Replication on 2019-2025.**
  - The catalog's walk-forward (each 6-month block trades the best of the family's settings on the previous 3
    years), H1 and H4, 11 FX pairs and gold.
  - Only 2019-2025 blocks are scored.
  - The 10 families with the highest 2007-2018 gross Sharpe in `docs/research_report.md` section 4, plus the
    gold London breakout and the night fade.
  - **Counts if:** net > 0, t >= 1.645, BH over E21.
- **E22. Agreement across timeframes.** An H1 trade from E20 at its 90% cut-off is taken only when the M15 and H4
  models' latest scores point the same way.
- **E23. Currency strength.**
  - Each day at 21:00 UTC, rank the 8 currencies by their average 5-day return across the 28 pairs.
  - Hold the strongest against the weakest (the one pair) until the next day at 21:00 UTC.
  - Also run with a 20-day ranking, and with the trade reversed (weakest against strongest).
- **E24. G10 carry on 2019-2025.**
  - The round-2 carry factor (long the top 2, short the bottom 2 of the 8 currencies by three-month rate,
    monthly), with real swaps.
  - Run without a filter and flat after a month-end VIX above 25.

## Addendum 3 (2026-11-03): ML rules aimed at the cost problem (M1-M4)

These were written before any of their data was assembled. The same counting rules and split apply (2007-2018 fit,
2019-2025 check; walk-forward refits yearly on earlier years only). Costs, fills and swaps are those of the flows
lab. Models: LightGBM with the 8.5 settings, nothing tuned.

- **M1. Weekly FX model, 28 pairs pooled.**
  - Each Friday at 21:00 UTC, predict the 5-day return of each pair, in units of its 20-day volatility.
  - Features:
    - momentum over 1, 4, 12 and 26 weeks;
    - carry (the 3-month rate difference, point-in-time);
    - the 3-month change in that difference;
    - the 20-day volatility and its ratio to the 120-day volatility;
    - the distance to the 52-week high and low;
    - VIX and its 1-week change;
    - S&P 500 4-week return;
    - TFF leveraged-fund z-scores for the pair's currencies (JPY, EUR, GBP, AUD, CAD, CHF, MXN where reported), published Friday;
    - month and week-of-month.
  - Trades: long or short the 5 pairs with the largest |prediction| in the top 10% of the fit-year distribution.
    Each is held 5 days, with real swaps.
  - **Counts if:** weekly portfolio net > 0, t >= 1.645 on 2019-2025.
- **M2. Cost-aware meta-labels on the rules that earn before costs.** The rules:
  - night fade I7: AUD/NZD, EUR/GBP, GBP/CAD;
  - end-of-day momentum E14: gold, 7 majors;
  - gold London breakout I9;
  - round numbers E18.

  For each, a model predicts the trade's gross (mid) return from features known at entry:
  - the live spread, and the spread over its 100-bar median;
  - band width or move size, in ATR;
  - the hour, the weekday;
  - 20-bar volatility;
  - the distance to the day's high and low.

  A trade is taken only if the prediction exceeds 1.2 times its own round-trip cost at entry.
  - **Counts if:** net > 0, t >= 1.645 on 2019-2025.
- **M3. Cross-asset intraday features.** The 8.5 symmetric-barrier model on H1 for USD/JPY, EUR/USD and gold, with
  extra features:
  - the last 1-hour and 4-hour returns of USTBOND, BUND, US500, NAS100, JPN225 and Brent;
  - each relative to its 20-bar volatility.

  Bars from 2012 (the bond / index history), 60/40 split; reported next to the same model without those features.
  - **Counts if:** the C90 rule is net > 0 with t >= 1.645 on the check period, and above the version without the
    extra features.
- **M4. One pooled intraday model across the 12 markets** (11 FX pairs and gold), H1 and H4, with the market as a
  feature, under the C90 / C95 / C98 / C99 rules of E20.
  - **Counts as in E20.**

M3 and M4 run after E20 reports. If E20's net does not rise with confidence, M4 is reported as the pooled variant
only.

## Addendum 4 (2026-11-03): feature parameters, transforms, conditions and new sources (FS1-FS3)

**Gap.** The feature predictability lab (`scripts/feature_ic_lab.py`, 857,202 tests) used every feature at its
**default parameters only**, in two forms (raw and minus its 20-bar EMA). It used horizons of 1, 6 and 24 bars on
2007-2018. Strategies sampled 6 settings per fold, and the ML labs used fixed lookbacks. These rules were written
before any new computation.

- **FS1. Parameters and transforms chosen inside the walk-forward.**
  - Scope: every feature with a search space, on H8, D1 and weekly bars, for 28 pairs plus gold and silver,
    2007-2025.
  - In each fold (3-year training, 6-month test), choose the feature's setting and transform on the training
    window by its training IC. Transforms: raw, minus its 20-bar EMA, percentile rank over 250 bars, z over 100
    bars, 1-bar change.
  - Measure the out-of-sample IC on the test window. Horizons: 6 and 24 bars (weekly: 1 and 4).
  - The null repeats the same selection on 20 circularly shifted targets, so choosing the best of many settings
    is charged against the feature.
  - Economic test: hold the sign of the chosen feature's top / bottom decile for h bars, after costs.
  - **Counts if:** net > 0, t >= 1.645 on the 2019-2025 folds, and the IC z beats the null after BH over
    feature x timeframe x horizon x instrument.
- **FS2. Conditional predictability.**
  - The 68 features rated "predictive", on M30-H4.
  - IC by session (Tokyo 00:00-07:00, London 07:00-13:00, New York 13:00-21:00 UTC) and by volatility tercile.
    The cell is chosen on 2007-2018, and only that cell is traded on 2019-2025, after costs.
- **FS3. Feature sources never used intraday, for USD/JPY, EUR/USD and gold on H1-D1.**
  - Cross-asset returns over 1, 4 and 24 bars: US and German bonds, US500, NAS100, JPN225, Brent.
  - Event clock: minutes to and from the next scheduled US release, central-bank decision, Treasury auction,
    Gotobi day and fix (2017-2023 where the calendar exists).
  - TFF positioning z.
  - The 3-month rate-gap change.
  - IC and decile trading as in FS1. A series starting after 2008 is split 60/40.

Order: FS1 on the weekly and daily bars first (where costs are smallest relative to the move), then FS3, then FS2.

## Addendum 5 (2026-11-03): the last open items of `docs/test_ledger.md`

These rules were written before their data was computed or downloaded. The batch counting rules and split apply.

- **FS4. Pairs of features.**
  - The 20 features with the highest median IC in `docs/fx_feature_predictability.csv`, at defaults; all 190 products
    of two of them, each standardised with training statistics.
  - IC on H4 and D1 for 12 markets (11 FX pairs and gold), horizons 6 and 24 bars, 2007-2025.
  - The same null with 20 circular shifts, BH over all.
  - Decile trading after costs on 2019-2025, as in FS1.
- **FS5. Volatility and large-move targets.**
  - For the same 20 features: the IC with the next-h-bar absolute return, and with the indicator |return| > 2x
    the round-trip cost. These measure information only.
  - A trading rule: the direction model of M4 is traded only when the FS5 model's predicted P(|return| > 2x cost)
    is in its top tercile (fit-year cut).
  - **Counts as in E20.**
- **E25. Japanese government bond auctions.**
  - 10-year and 30-year JGB auction days, from the Ministry of Finance auction calendar if it can be downloaded;
    untestable otherwise.
  - USD/JPY in two windows: 00:00 -> 03:00 UTC before the 03:35 UTC result, and 03:40 -> 08:00 UTC after it.
  - Direction from the fit period.
- **E26. The dollar before Fed decisions, in the expected direction** (Karnaukh).
  - Fed decisions 2017-2023 from the calendar dataset. The expected move is the sign of forecast minus previous
    rate; no trade when they are equal.
  - Long the dollar against EUR, JPY, GBP and AUD (equal weight) for an expected hike, short for a cut.
  - Window: 48 hours to 15 minutes before the decision.
  - **Information only:** about 20 decisions with an expected change.
- **E4 / E16 rerun** on the re-downloaded emerging-market files (run `em-redownload`), with unchanged rules. Both
  versions are reported.

## Addendum 6 (2026-11-03, 21:45 UTC): FS6, a cross-asset composite chosen on the fit period only

Written before any composite was computed. FS3 found real but small cross-asset information; this asks whether
the signals add up when combined, with the selection kept out of the check period.

- **Markets:** USD/JPY, EUR/USD, gold, on H4 and D1; h = 6 and 24 bars (12 tests).
- **Candidates:**
  - the FS3 cross-asset returns over 1, 4 and 24 bars of USTBOND, BUND, US500, NAS100, JPN225 and Brent;
  - the 3-month rate-gap change and the TFF leveraged-fund z (USD/JPY and EUR/USD).
  - The event clock is left out: it only covers 2017-2023, which straddles the split.
- **Fixed split:** fit 2012-01-01 to 2019-12-31, check 2020-01-01 to 2025-12-31. 2020 is about 60% of the way
  through the 2012-2025 cross-asset history.
- **Selection, fit period only:** for each candidate, the correlation of its value with the next-h-bar mid log
  return. It is taken on non-overlapping bars (every h-th bar), and t = corr x sqrt(n). A candidate is kept if
  |t| >= 2, with sign(corr).
- **Composite:**
  - Each kept candidate is standardised with fit mean and sd, and clipped at ±4.
  - The score is the mean of sign x z over the kept candidates available at the bar.
  - The score needs at least half of them to be present, else it is missing.
- **Trading on the check period:**
  - Long when the score is at or above its fit-period 90th percentile, short at or below its 10th.
  - One position at a time, held h bars.
  - Net = signed next-h-bar mid return minus the round-trip cost at the bar, as in FS1-FS5. t is clustered by
    month.
- **Counts if:** net > 0, t >= 1.645, and BH q <= 0.10 over the 12 tests. If no candidate is kept, that test is
  reported as "nothing selected".
