# Final outcome (2026-11-02)

**The one edge left:** Gotobi overnight, long USD/JPY and EUR/JPY from 23:00 Tokyo the evening before to 09:53
Tokyo on Gotobi days, no stop (about 59 trades a year). Development 2007-2025: +4.6 bp a basket trade (sd 39 bp).
2026 out of sample (Jan-Oct): +2.0 bp, positive but not significant. CHF/JPY also carries it in development.
Everything else tested (seven literature rounds, ~35 labs, YouTube / forum / commercial / AI methods) failed after
costs or failed in 2026 (Nasdaq overnight, rebalancing).

Simulations: `scripts/final_sims.py` (results in `docs/final_sims/`). Each basket trade's worst open loss counts
against the loss limits. Three edge sizes: development (+4.6 bp), 2026 (+2.0 bp), zero.

## Prop firms (100k; mean and median net per attempt, after the fee)

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

Not financial advice: research results on historical data with an edge that is small and not confirmed out of
sample.
