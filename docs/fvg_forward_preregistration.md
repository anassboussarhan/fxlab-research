# Fair-value-gap trade on Nasdaq futures: paper forward test (pre-registered 2026-11-15)

**Origin:**
- Round 22 part I (`docs/round22_results.md`): I-FVG on NAS100 met the counting rule on 2010-2025 (2019-2025
  +7.9 bp a day, t 6.7).
- It held under harsher execution (t 3.2-5.4).
- It was supported in the one-time 2026 check (2026-01 to 2026-10, +10.0 bp a day, t 3.2, 189 days).

## The rule (frozen)

- **Configurations:** the five the walk-forward chose for 2026 on data up to 2025, Sobol indices 64, 56, 8, 40
  and 88 of `scripts/r22_I.py` (FVG family, seed 25, 128 points).
- **What they have in common:** each enters on a pullback into a fresh three-candle gap in the direction of the
  gap, with a stop just beyond the first candle. The parameters are fixed in the code.
- **Bars and sizing:** NAS100 session bars, 5 or 15 minutes (per configuration), 09:30-16:00 New York. Equal
  weight, one unit of notional per trade.
- **Cost:** NQ futures, 0.375 index points per side.

## Forward test

- **Schedule:** every Saturday the workflow `forward_fvg.yml` downloads Dukascopy NAS100 minute candles from
  2026-11-02 (warm-up) to the last Friday.
- **Ledger:** it writes the daily ledger from 2026-11-16 to branch `forward-fvg`.
- **Review:** after 125 and 250 trading days. A one-sided t ≥ 1.645 on forward days alone counts as support.
- **Weekly rebuild:** each run rebuilds the ledger from the warm-up. A trade still open at the end of the data is
  marked at the last bar and settled in the next week's run. Holds reach up to about three sessions, so the last
  three trading days of each week can be revised once.
- **Warm-up check** (`scripts/fvg_forward_check.py`, development data, July-December 2025):
  - a two-week warm-up gives the same daily results as the full 2010-2025 history on 106 of 110 days;
  - the 4 days that differ are each within the last 3 days of a test window: open trades marked at the end of
    the data.
- **Rules:** no rule changes; a change starts a new track.
- **Prop simulation:** a futures-prop simulation (NQ / MNQ) follows only if the forward test supports it.

Not financial advice.
