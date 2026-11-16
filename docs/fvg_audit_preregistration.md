# Fair-value-gap result: second audit (pre-registered 2026-11-16, before any real data was run)

The round-22 fair-value-gap result (I-FVG on NAS100) is re-checked for look-ahead and fill assumptions.
- **Code:** `scripts/fvg_audit.py`.
- **Data:** development data only; 2026 is not touched unless the condition at the end is met.

## What was found before running real data

**1. The round-22 simulator had a look-ahead in trade sequencing.**
- **How it worked:** `fam_FVG` took each gap in turn and resolved its limit order to the end (filled or expired)
  before looking at the next gap.
- **What went wrong:** if an earlier gap's order filled later, it was taken, and gaps formed in the meantime were
  skipped. If it never filled, the later gaps were used. So which trade was taken depended on future prices.
- **Dropped fills:** a fill at an open beyond the stop was dropped instead of being closed at once.
- **On random walks** (no edge, no costs; sampled every 5 seconds, so bar extremes are real path extremes):
  - the round-22 logic earns **+0.85 bp gross a trade** (+1.77 bp for the with-gap configurations, which are the
    ones that were chosen);
  - every seed is positive.

**2. A causal simulator removes it.**
- **How it works:** every gap places a resting order, one position is held at a time, the other orders are
  cancelled on a fill, and gap-through fills are kept.
- **On the same random walks:**
  - +0.18 bp (with-gap +0.47, seeds −0.4 to +1.75);
  - flat daily: +0.14 (+0.23);
  - with one tick of stop slippage: −0.00 (+0.03).
- The small remainder is stops filled at the stop price between samples; the tick of slippage covers it.
- Saved in `results/fvg_audit/fvg_audit_null.json`.

**3. Same picture on an end-to-end run.** A local test of the full pipeline on a synthetic random walk
(2016-2020, minute bars, no edge):
- the round-22 logic finds 2019-2020 t 3.5;
- it chooses configurations 56, 8, 64 and 40, four of the five chosen for 2026 on NAS100;
- the causal variants find nothing (t 0.2, −2.4, −1.9).

**4. An "exact limit price" variant is not used.** A limit placed through the market fills at the market,
so filling it at the limit is wrong. On random walks that variant loses 4 to 11 bp a trade.

## Variants run on real data

All four use the same 128 configurations and the same yearly walk-forward as round 22.

| Variant | What it is |
|---|---|
| A0 | Round-22 logic. Must reproduce round 22 to 1e-12 on the CFD data. |
| A1 | Causal. Orders and stops work in the cash session; positions may be held overnight. |
| **A2** | **Causal and flat every day.** Same-day gaps, orders expire at 16:00 New York, positions closed at the last bar's close. **Primary.** |
| A3 | A2 with harsh execution: fills only if price trades through by a tick, one tick of slippage on every stop, twice the futures cost. |

### Data
- the Dukascopy NAS100 CFD store, 2012-2025;
- NQ futures minute bars built from trades, Databento GLBX.MDP3, front contract by volume, 2010-06 to 2025:
  - each contract is simulated on its own, so no trade spans a roll;
  - the cost is checked first and the download is aborted above USD 2.50 (estimated free credit left about
    USD 5.50).

## Rules

**The result stands only if A2 meets the counting rule on both the CFD and the NQ data:**
- 2010/15-2018 mean > 0;
- 2019-2025 mean > 0 with t ≥ 1.645;
- A3 has a 2019-2025 mean > 0 on both sources.

**If it stands:**
- **New forward configurations:** the forward test switches to the A2 simulator and to the five
  configurations A2 chooses for 2026 from the CFD data up to 2025 (the forward data are CFD candles). This is a
  new freeze, made before any forward data exist (first forward day 2026-11-16, first weekly run 2026-11-21).
- **2026 re-check:** a one-time check of A2 on 2026-01 to 2026-10 (the sixth use of the holdout). The earlier
  2026 check ran on the round-22 simulator and is void.
  - **supported:** mean > 0 and t ≥ 1.645;
  - **contradicted:** t against 2019-2025 ≤ −1.645.

**If it does not stand:** the result is withdrawn, the forward test is stopped, and the inventory and ledger say
so.

Not financial advice.
