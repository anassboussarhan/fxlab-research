# Round 22 results: every parameter optimised, walk-forward, across all markets (2026-11-15)

- **Pre-registration:** `docs/round22_preregistration.md` (addenda 1-3). Audit of what had and had not been
  optimised: `docs/round22_audit.md`.
- **Runs:**
  - r22a 43740046103 (L, T);
  - r22b 43740446671 (I, X);
  - r22c 43741219550 (D-P);
  - r22e 43745777671 (C2, V, E-FUND reruns; E-FUND ran for 1 h 50 min);
  - r22g 43756162309 (a duplicate E-FUND run with progress output, started while r22e looked stuck; not used);
  - r22f 43748823690 (P, S, C3, V after the minimum-history fix);
  - r22h 43745632116 (2026 check of the fair-value gap);
  - final combine: r22d 43756260351, branch `round22`.
- **Code:** `scripts/wf22.py` (walk-forward core), `scripts/r22_*.py`, `scripts/r22_combine.py`,
  `scripts/r22_fvg_check.py`, `scripts/r22_h26.py`.
- **2026:** used once, as pre-registered, for the four streams that counted (the fifth use of the holdout).

## Summary

**4 of 64,503 real streams count** (the counting rule plus the multiple-testing correction). They are two
results, each on two streams.

**Two things count, and both held on 2026:**

1. **The fair-value-gap pullback on Nasdaq futures (I-FVG, NAS100).**
   - Development data: 2019-2025 +7.9 bp a day after futures costs, t 6.7; 2015-2018 +2.3 bp.
   - It stays significant under every harsher execution tried (t 3.2 to 5.4).
   - 2026 (January-October, configurations chosen on data up to 2025): +10.0 bp a day, t 3.2, 189 days.
     **Supported.**
2. **Crypto cash-and-carry** (long spot, short perpetual when funding is high; BTC, ETH).
   - Development data: 2023-2025 +2.1 bp a day, t 21.7 (BTC); +1.9 bp, t 20.3 (ETH).
   - 2026: still positive (BTC +0.65 bp a day, t 7.1; ETH +0.25 bp, t 2.9), so **supported**, but the yield has
     shrunk to a third (BTC) and an eighth (ETH) of 2023-2025: roughly 2.4% a year on BTC, below a cash rate.

**Everything else fails** once each method's parameters are chosen walk-forward, with no fixed settings:
- the strategy libraries (64,292 units), every time window, the other ICT / SMC families;
- index breakouts, dip-buying, events, monthly currency and bond signals, commodities;
- stock anomalies, volatility selling and options structures.

The real streams met their own rule (345 of 64,503) at about the placebo rate (0.53% against 0.35%), so most
single passes are noise. The two results above are far outside that noise and survived the correction.

## The protocol

- **Selection:** each family's parameter space is covered by an exhaustive grid where cheap (L, T) and by 128 or
  256 scrambled Sobol points elsewhere. At the start of each year the five configurations with the best training t-statistic on all earlier data
  (expanding window) are chosen and traded in that year with equal weight. Nothing is fixed by hand.
- **Counting rule:** out-of-sample 2010-2018 mean > 0 (if traded), 2019-2025 mean > 0 with t ≥ 1.645, and a
  Benjamini-Hochberg correction at q = 0.10 over all real streams.
- **Placebos:** the engines' placebo units go through the same walk-forward, as a null.

## Results by part (development data, out of sample, 2019-2025)

| Part | What | Streams | Meet own rule | Best 2019-2025 t | Result |
|---|---|---|---|---|---|
| L | The strategy libraries, re-selected every year, per engine / unit / direction | 64,292 | 332 (0.52%) | 3.56 (GBP/JPY weekly cumulative RSI; 2010-2018 negative) | nothing passes BH |
| T | Every time window: 14,256 entry / exit / weekday / calendar configurations per instrument | 49 | 1 | 2.09 (GBP/USD; 2010-2018 negative) | nothing |
| I | ICT / SMC families with all parameters free: levels, FVG, order blocks, OTE, breakers, CBDR, sessions, SMT | 85 | 4 | **6.68 (FVG, NAS100)** | **FVG NAS100 and FVG pooled count**; all other families negative pooled (t −0.6 to −8.7) |
| X | Index breakouts: noise area, ORB, VWAP trend on 9 indices | 30 | 1 | 1.91 (noise-area, NAS100) | nothing passes BH |
| D | Dip-buying and the D7 family, every parameter (ECB fixings) | 1 | 0 | −0.04 | nothing |
| E | EIA crude and natural-gas reports; trading around crypto funding stamps (BTC, ETH) | 5 | 0 | −0.26 (funding, ETH) | nothing; EIA t −2.1 to −3.1 |
| M | Monthly currency signals, 8 currencies, 1976-2025 | 1 | 0 | −1.31 | nothing |
| B | Bonds across 8 countries, 1976-2025 | 1 | 0 | 1.59 | nothing (just short of the rule) |
| C | Commodity spreads (C1), CFTC managed-money positioning (C2), commodity cross-section (C3) | 5 | 0 | 1.19 (Brent-WTI spread) | nothing; C2 t −1.51, C3 t −1.49 |
| S | S&P 500 anomalies, index deletions, pairs (point-in-time members) | 3 | 0 | 0.65 | nothing |
| K | Crypto cross-section; cash-and-carry | 3 | 2 | **21.65 (BTC)** | **cash-and-carry counts on BTC and ETH**; cross-section t 0.81 |
| V | SPX variance premium, SVXY, BTC implied-volatility selling, all filters optimised | 3 | 0 | 1.41 (SPX) | nothing; SVXY −0.01, BTC 0.54 |
| P | Options structures on real quotes (DoltHub 2019-2025): puts, strangles, condors, put spreads, covered calls, hedged straddles / strangles, every strike / delta / tenor / exit | 25 | 5 | 2.24 (CSCO) | nothing passes BH; pooled t 1.02 |

Notes:
- **P (options):** after the minimum-history fix, these meet their own rule: MSFT t 1.85, GOOGL 1.69, NVDA 1.74,
  DIA 2.06 and CSCO 2.24. SPY is at t 1.58 (t 3.33 before the fix, when it traded only 2024-2025). None is near
  the BH threshold (t of about 4.4 to 4.7 at this number of tests). Options do not give a counting edge on this
  data.
- **V and S:** rerun after the same fix; the numbers above are the reruns.
- **L:** the best units are weekly rules on crosses whose 2010-2018 half lost; the share passing is at the
  placebo rate.

## The fair-value-gap result in detail

### Per instrument (2019-2025, bp a day after costs)

| Instrument | 2010/15-2018 | 2019-2025 | t | Own rule | BH | Average of all 128 configurations |
|---|---|---|---|---|---|---|
| **NAS100 (NQ costs)** | +2.26 | **+7.88** | **6.68** | yes | **yes** | −0.09 |
| US500 (ES costs) | −0.32 | +2.18 | 1.68 | no | no | −0.80 |
| USD/JPY | +0.71 | +0.93 | 3.17 | yes | no | −0.32 |
| USD/CHF | +0.69 | +0.50 | 2.65 | yes | no | −0.80 |
| EUR/USD | +0.46 | +0.25 | 1.03 | no | no | −0.30 |
| Gold | +0.06 | +0.82 | 1.43 | no | no | −0.98 |
| Silver | −2.17 | −0.84 | −2.05 | no | no | −6.50 |
| GBP/USD, AUD/USD, NZD/USD, USD/CAD | | −0.29 to +0.06 | −0.99 to 0.33 | no | no | |
| **Pooled (11)** | +0.25 | **+0.90** | **4.93** | yes | **yes** | |

**What the walk-forward picks:**
- The average configuration is flat (NAS100 −0.09 bp); only about 30% of the 128 are positive in 2019-2025.
- The edge is in one region of the parameter space, and the yearly selection keeps finding it.
- That region: **trade in the direction of the gap**, enter on a pullback into it, stop just beyond the first
  candle, target 1.2 to 3.3 times the risk, 5- or 15-minute bars of the New York cash session.

### Harsher execution (NAS100, development data only)

| Variant | 2019-2025 bp a day | t |
|---|---|---|
| V0 as tested | +7.88 | 6.68 |
| V1 CFD bid / ask costs instead of futures | +3.70 | 3.16 |
| V2 a limit fills only if price trades through it by a tick | +6.32 | 5.44 |
| V3 one tick of slippage on every stop and twice the futures cost | +4.59 | 3.99 |
| V4 V2 and V3 together | +4.23 | 3.94 |

**The trades** (V0, chosen configurations, 2015-2025):
- 16,397 trades, about 1,500 a year over the five configurations;
- 44% winners, +4.6 bp net a trade (+5.5 gross), median 11 bars;
- long +6.5 bp, short +2.6 bp: both sides earn, so it is not just the Nasdaq's rise;
- positive in every year except 2016 (−0.9 bp a day);
- the best 20 days hold 28% of the total.

On US500 the same rules are weaker (t 1.7) and fail under V3 / V4 (t 1.0 and 0.2). On currencies the edge is too small to matter (under
1 bp a day).

### 2026 check (pre-registered, configurations chosen on data up to 2025)

| Stream | Days with data | 2026 bp a day | t | t against 2019-2025 | Verdict |
|---|---|---|---|---|---|
| **I-FVG NAS100** | 189 | **+10.0** | **3.23** | +0.65 | **supported** |
| **I-FVG pooled** | 260 | **+1.95** | **3.38** | +1.76 | **supported** |
| K cash-and-carry BTC | 304 | +0.65 | 7.07 | −11 | supported, much smaller |
| K cash-and-carry ETH | 304 | +0.25 | 2.85 | −13 | supported, much smaller |

- Per instrument in 2026: gold +6.7 bp (t 1.85), silver +4.1 (t 1.45), US500 +2.0 (t 0.77); the currencies
  between −0.3 and +0.8 bp.
- **The five configurations chosen for 2026 on NAS100** (all trade with the gap):

| Sobol index | Minimum gap (ATR) | Entry in the gap | Stop beyond candle 1 (ATR) | Target (R) | Window | Limit valid (bars) | Max hold (bars) | Bars |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.23 | far edge | 0.10 | 2.26 | any | 25 | 79 | 15 min |
| 56 | 0.01 | near edge | 0.73 | 1.15 | any | 33 | 15 | 5 min |
| 8 | 0.15 | middle | 0.59 | 2.53 | any | 46 | 44 | 15 min |
| 40 | 0.12 | near edge | 0.24 | 2.12 | 10-11 New York | 22 | 68 | 5 min |
| 88 | 0.09 | near edge | 0.79 | 3.26 | New York morning | 38 | 87 | 5 min |

## Bugs found and fixed during the round

- `r22_T`: JSON keys were numpy integers; now strings.
- `r22_I`: the CBDR / session families searched each day's bars from scratch; positions are now precomputed (same
  results, faster).
- C2: the 2006-2009 CFTC yearly files do not exist; the 2006-2016 history file is used.
- V-btc: implied-volatility months were misaligned with returns; now aligned by month.
- E-FUND: a date comparison added a string to a timestamp; fixed.
- **The minimum-history rule** (addendum 3): 50 valid training periods blocked monthly families for their first
  years (P 2021-2023, S 2008-2009, V-btc always). It is now a parameter (12 for monthly families) and P, S, C3 and
  V were rerun. The reruns changed no counting verdict.
- 2026 cash-and-carry: Binance files mix millisecond and microsecond timestamps; converted per row.

None of these changed a rule or was found by looking at 2026 results.

## What follows

- **Fair-value gap on NQ:** a frozen paper forward test from 2026-11-16, weekly
  (`docs/fvg_forward_preregistration.md`, workflow `forward_fvg.yml`, branch `forward-fvg`). A futures
  prop-firm simulation follows only if the forward test supports it.
- **Cash-and-carry:** the 2026 yield (about 2.4% a year on BTC) is below a cash rate after exchange risk; not
  worth setting up at present.

Not financial advice.
