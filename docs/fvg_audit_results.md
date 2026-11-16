# Fair-value-gap result: second audit — withdrawn (2026-11-16)

- **Pre-registration:** `docs/fvg_audit_preregistration.md`.
- **Run:** fvg-audit 43865176623 (branch `fvg-audit`).
- **Code:** `scripts/fvg_audit.py`.
- **Data:** development data only. 2026 was not used.

## Verdict

**The round-22 fair-value-gap result on NAS100 came from a look-ahead in the simulator.**
- With a causal simulator, the same 128 configurations and the same walk-forward find nothing.
- **The result is withdrawn:** NAS100, pooled, and the USD/JPY and USD/CHF own-rule passes.
- **The 2026 "supported" verdict is void:** it ran on the same simulator.
- **The paper forward test is stopped** before its first forward day.
- The pre-registered 2026 re-check was not run, because its condition (A2 passes on development data) failed.

## The bug

**How round 22 simulated trades:**
- `fam_FVG` looked at each gap in turn and resolved its limit order to the end, filled or expired, before
  moving on.
- If the order filled, even many bars later, the gaps formed in between were skipped. If it never filled, those
  later gaps were traded.
- So which trade was taken depended on prices that came after the decision.
- A fill at an open beyond the stop was dropped instead of being closed at once.

**The causal simulator:**
- every gap places a resting order, one position is held at a time;
- a fill cancels the other orders;
- a fill beyond the stop is closed at once.

## Results

### Random walks (no edge, no costs; 6 seeds)

| Gross bp a trade | All configurations | With-gap configurations |
|---|---|---|
| A0 round-22 logic | **+0.85** (every seed positive) | **+1.77** |
| A1 causal | +0.18 | +0.47 |
| A2 causal, flat daily | +0.14 | +0.23 |
| A3 A2 with harsh execution | −0.00 | +0.03 |

A full pipeline run on a synthetic random walk:
- the round-22 logic finds t 3.5;
- it picks four of the five configurations it chose for NAS100 in 2026.

### NAS100 CFD, 2012-2025, futures costs, walk-forward as in round 22

| Variant | 2015-2018 bp a day | 2019-2025 bp a day | t | Counts | Best training t at the end of 2025 |
|---|---|---|---|---|---|
| A0 round-22 logic (reproduces round 22 exactly: max difference 0.0) | +2.26 | **+7.88** | **6.68** | yes | 4.78 |
| A1 causal, overnight holds | −0.63 | +0.92 | 1.09 | no | 1.52 |
| **A2 causal, flat daily (primary)** | +0.36 | **−0.16** | **−0.27** | **no** | 0.92 |
| A3 A2 with harsh execution | −0.06 | −0.31 | −1.00 | no | −0.11 |

- **By year:** A0 is positive in 10 of 11 years. A2 swings around zero (−2.8 to +4.5 bp).
- **The five configurations chosen for 2026 under the causal simulator:**
  - A1 +0.5 bp a day (t 0.4) over 2019-2025;
  - A2 +1.8 bp (t 1.9);
  - A3 +0.5 bp (t 0.5).
- These figures are in-sample, because the configurations were chosen on data that includes these years.

### NQ futures cross-check

Not run.
- **Cost:** Databento priced NQ minute bars at USD 19.22 (2010-2025), 13.96 (2015-2025) and 9.01 (2019-2025).
- **Nothing downloaded:** all three are above the USD 2.50 cap. Nothing was spent.
- **Not needed:** the result already fails on the CFD data, and both sources had to pass.

## What else this touches

- **Round-22 part I:** the order-block, OTE and breaker families use the same sequencing. They were already
  negative, and the bias only flattered them, so their verdict stands.
- **Not affected:** no other result that counts uses limit orders: Gotobi, the D7 family, the noise-area
  breakout, and crypto cash-and-carry are market orders on lagged signals.
- **Round 22 now counts only crypto cash-and-carry:** BTC and ETH, 2026 supported but at about 2.4% a year on
  BTC.

Not financial advice.
