# Final outcome (2026-11-02)

**The one edge left:** Gotobi overnight, long USD/JPY and EUR/JPY from 23:00 Tokyo the evening before to 09:53
Tokyo on Gotobi days, no stop (about 59 trades a year). Development 2007-2025: +4.6 bp a basket trade (sd 39 bp).
2026 out of sample (Jan-Oct): +2.0 bp, positive but not significant. CHF/JPY also carries it in development.
Everything else tested (seven literature rounds, ~35 labs, YouTube / forum / commercial / AI methods) failed after
costs or failed in 2026 (Nasdaq overnight, rebalancing).

Simulations: `scripts/final_sims.py` (results in `docs/final_sims/`). Each basket trade's worst open loss counts
against the loss limits. Three edge sizes: development (+4.6 bp), 2026 (+2.0 bp), zero.

## Prop firms (100k; mean and median net per attempt, after the fee)

*Superseded by `docs/prop_best.md` (below): this table is without a stop, and FundedNext at 5x breaks its 3% open-risk
rule.*

| Goal | Firm, leverage (passing / funded) | Pass | Paid | Mean net, 2026 edge | Mean net, zero edge | Paid within 3 / 6 months | Median days to 1st payout |
|---|---|---|---|---|---|---|---|
| Highest | FundedNext Stellar 2-step, 5x / 5x | 33% | 30% | +$3,843 | +$1,969 | 2.8% / 14% | 131 |
| Both (best mean with >= 15% paid in 6 months) | The5ers High Stakes, 5x / 5x | 33% | 31% | +$3,743 | +$1,831 | 3.8% / 16% | 120 |
| Fastest | FundedNext Stellar 1-step, 5x / 3x | 16% | 16% | +$577 | +$24 | 8.0% / 12% | 63 |
| (with the development edge) | FundedNext 2-step, 3x / 5x | 47% | 41% | +$6,560 (dev) | +$1,572 | 0.1% / 6% | 219 |

- **The median attempt loses the fee** in every setup: two in three attempts fail, mostly on the daily loss limit
  hit by an open loss overnight.
- **About half the expected value is not the edge:** with a zero-edge trade the same setups still show +$1.6-2.0k,
  because the loss is capped at the fee while funded profits are paid out. That depends on the firm paying and on
  its conduct rules; it is luck, not skill.
- Leverage above 8x fails the daily limit in most attempts; the 1-step programmes (3% daily) suit this trade worse.
- Operational: on funded FTMO / FundedNext / The5ers accounts, 23:00 Tokyo (14:00 UTC) can coincide with a 10:00
  New York release in summer; the entry must be moved a few minutes on those days (not tested).

## Own account, compounding (lottery)

| Leverage | 3 years, 2026 edge: median / P(wipe-out) / P(10x) / P(100x) | 3 years, development edge: same |
|---|---|---|
| 10x | 1.26x / 0% / 0% / 0% | 1.98x / 0% / 0.1% / 0% |
| 15x | 1.27x / 0.3% / 0.4% / 0% | 2.51x / 0% / 3.6% / 0% |
| 20x | 1.20x / 3.3% / 1.9% / 0% | 2.96x / 0.6% / 12% / 0% |
| 30x | 0.70x / 33% / 6.0% / 0.1% | 3.06x / 16% / 24% / 1.3% |
| 40x | 0.05x / 69% / 5.6% / 0.4% | 0.00x / 60% / 18% / 3.4% |

With the 2026 edge size, the median best is about 15x (+27% in 3 years); a lottery ticket at 30-40x gives about a
6% chance of 10x and a one-in-a-hundred to one-in-three-hundred chance of 100x, with most accounts wiped out.

## Risk control (added 2026-11-02; `docs/untested_batch.md` 9.14, `docs/holdout_2026_results.md`)

- **Stop: use a wide disaster stop, about 100 bp (or 2 W, twice the recent window range) below entry, not a tight
  one.** In development it keeps the whole gain (+4.68 / +4.71 bp vs +4.37 / +4.77 without, 2007-2018 / 2019-2025), caps
  the worst trade near -110 bp (vs -268 / -141) and is touched on 1-5% of trades. Its 2026 effect is not measured. Tight
  stops (0.75 W, 40 bp) halve the tail but turned 2026's +2.0 bp into -1.1 / -0.8 bp.
- With the 100 bp stop the prop numbers above improve: FundedNext Stellar 2-step 5x / 5x, pass 39%, +$4,908 an
  attempt at the 2026 edge (+$3,759 without a stop in the same run); The5ers 5x / 5x +$4,800. Own account, 3 years,
  2026 edge: 15x median 1.36x; 30x median 1.04x with 15% wiped out (33% without the stop).
- **No hedge or filter raised accuracy without cutting the gain.** Shorting other yen crosses removes the edge
  (they rise on the same nights); VIX, FOMC and prior-move filters do not hold out of sample; a profit target
  lifts winners to 62% but halves the mean. The only diversification that helps is the trade itself on two pairs
  (USD/JPY + EUR/JPY).
- **Size against the stop:** with a 100 bp stop, leverage x 1% is what a stop hit costs. At 5x that is about 5%,
  the whole daily limit at FundedNext, The5ers and FTMO, so a hit still ends the challenge; at 3x it costs 3%
  (daily-limit failures 10% instead of 33%) but the target takes longer and the mean per attempt is lower
  (+$2,248 at 3x / 3x vs +$4,908 at 5x / 5x, 2026 edge). Risking 1% a trade means 1x.

## Accuracy (added 2026-11-02; `docs/untested_batch.md` 9.15, `docs/gotobi_accuracy/`)

- **About 60 more exits and entries tested; none beats holding to the fix.** Targets, half-profits and tight stops
  raise the share of winners (to 60-66%) only by cutting the mean; re-entries, scale-ins and dip entries lower it.
- **The disaster stop is confirmed in development at 70-100 bp** (80 bp was the 2007-2018 pick for the worst
  case: worst trade -97 / -81 bp, mean 4.40 / 4.96 bp vs 4.37 / 4.77 without, 2007-2018 / 2019-2025).
- **Why:** trades that fall below about -0.75 W end as losers 84-91% of the time, but held they lose about as much
  as the stop: from there the rest of the night is a coin flip. The gain comes from nights that stay near flat.
- **The one way found to raise accuracy without cutting the gain is to skip some nights:** a logistic model of
  winning nights (fixed before the run; 24 features known at entry, led by the Nikkei's previous-day return)
  skipping the third it rates lowest gave, on 2019-2025, 63% winners and +7.5 bp a trade (vs 58%, +4.8 bp), the
  same total per Gotobi day on two-thirds of the nights; refitted yearly, kept nights beat skipped nights in 13 of
  14 years (2012-2025). It helps the own account (smaller drawdowns) more than a prop challenge (fewer nights reach
  the target more slowly). Not yet tested on any month after 2025 that was not used before.

## Losing nights and the forward test (added 2026-11-02; `docs/untested_batch.md` 9.16, `docs/forward_test_preregistration.md`)

- **What goes with losing nights:** US yields having risen the day or week before (about half those nights lose,
  vs a third after falling yields, in both periods) and a strong Tokyo session the day before. During the night,
  losses come from the New York session (79% of the night's variance): falling US yields, falling stocks, a lower
  Nikkei open. The worst nights are shocks (2008, Brexit, BoJ surprises, Japan's 2024 intervention) that nothing
  known at entry flagged. Calendar items (months, holidays, Obon, year-end, FOMC nights) do not matter.
- **Skipping the nights most likely to lose big does not work:** they are also the best nights.
- **Frozen for the forward test from 2026-11-05:** the trade, the 80 bp stop (100 bp secondary), the winning-night
  filter forward-v1 (primary) and its 80-feature version forward-v2 (secondary), with verdicts and power stated in
  advance. First verdict after October 2027.

## Technical indicators (added 2026-11-02; `docs/untested_batch.md` 9.17)

- 64 standard indicator readings at entry (RSI, MACD, stochastics, Bollinger, ATR, ADX, CCI, Williams %R, moving
  averages, Ichimoku, SAR, pivots, candles) add nothing the filter does not already have. The few that hold in both
  periods say the night goes better **after a dip within an up-trend** (oversold Williams %R, CCI, stochastics, a
  close below the lower Bollinger band, a red Heikin-Ashi day; Tenkan above Kijun); overbought readings are not a
  warning. No indicator warns of big losses, and adding them to the filter made it worse. Nothing new is frozen.
- **The monthly forward-test job** (`.github/workflows/forward_test.yml`, 07:23 UTC on the 3rd of each month)
  downloads the new month, scores every Gotobi night from 2026-11-05 with both frozen filters and the stops, and
  publishes the tests to the `forward-test` branch (`forward_test/report.md`).

## Best prop setup (added 2026-11-02; `docs/prop_best.md`, `scripts/prop_best.py`)

- **FTMO 2-Step at 5x with the 80 bp stop** (Swing account type, entry moved to 14:06 UTC) has the highest mean per
  attempt and per dollar of fee and is in the top group for being paid: 45% of attempts paid at the 2026 edge (80%
  at the 2019-2025 edge), +$6.1k mean net per attempt on 100k (fee EUR 540, refunded with the first reward).
- Cheapest tickets: The5ers High Stakes Classic 2.5k ($22, 4x) and FundedNext Stellar Lite 5k ($33, 3.5x), 41-42%
  paid at the 2026 edge, small payouts.
- No setup is fast (12-14% paid within 6 months at best) and none is a sure win: at the 2026 edge the median
  attempt loses the fee everywhere.
- Leverage caps: 5x for 5% daily limits measured from the balance, 3x for 3% daily limits, 3.5x at FundedNext (3%
  combined open risk), 4x at The5ers (its daily loss starts from the open profit at server midnight, which falls
  inside the trade).

Not financial advice: research results on historical data with an edge that is small and not confirmed out of
sample.
