# Round 23: every gap from the round-23 audit (pre-registered 2026-11-18, before any data was run)

**What this round runs:** everything ranked in `docs/round23_audit.md` section 9.
**Rules for data:**
- development data only (to 2025-12-31);
- 2026 is used only as stated under "2026";
- no paid data.

**Code:**
- `scripts/wf23.py`: the general and nested walk-forward;
- `scripts/r23_*.py`: one script per part.

## The selection core (all parts)

**General walk-forward:** `wf23.walk_forward(..., window, top, reselect, score)`, on per-month aggregates of each
configuration's period returns.
- **window:** 12, 24, 36 or 60 months, or expanding;
- **top:** 1, 3, 5 or 10 configurations, equal weight;
- **reselect:** monthly, quarterly or yearly;
- **score:** training t-statistic or training mean.

That gives 120 settings. Eligibility and the first out-of-sample year follow round 22 for each part:
- at least 3 training years;
- the same minimum active and valid periods.

**Nested walk-forward (part N, applied everywhere):**
- every setting gives an inner out-of-sample stream;
- each January from the inner start + 3 years, the setting whose inner stream has the best t over all earlier
  years is chosen and traded that year.

**Two streams per unit, both counted:**
- **default:** expanding window, top 5, yearly, t (the round-22 setting);
- **nested:** as above.

**Counting rule (as round 22):**
- the first out-of-sample period up to 2018 has mean > 0, if it exists;
- 2019-2025 has mean > 0 with t ≥ 1.645;
- Benjamini-Hochberg at q = 0.10 over all round-23 real streams (default and nested, every part).

**Placebos:** where a part has placebo units, they go through the same selection. They are reported, not counted.

## Parts

### A. D7 statistics with open trades marked (a correction, not a test)

- **What changes:** `scripts/family_d7.py` and the 2026 D7 figures are recomputed. Trades still open at the end of
  a window are now closed at that window's last fixing; until now they were dropped.
- **Windows:** 2019-2025 and the same 2026 window as before.
- **Reported:** old and corrected figures. No selection is made, so this is not a new use of 2026.

### C30. Monthly currency signals on about 30 currencies

**Data:**
- **Spot:** ECB reference rates from 1999, converted to USD per unit (monthly, last fixing).
- **Short rates and CPI:** OECD MEI via DBnomics (3-month interbank, else call money; CPI index and change).
- **VIX:** FRED VIXCLS.
- **Universe:** every currency with a short-rate series, except those pegged or managed against the euro or the
  dollar (DKK, BGN, HKD).

**Signals and book** (Sobol, 256 points):
- **Signals:** carry, real carry, momentum 1-12 months (skip 0-1), value 36-84 months, composite.
- **Book:** long the top n, short the bottom n, n 1-5; rebalance 1-3 months; lag 1-2 months.
- **Crash filter:** none, or flat when VIX > a threshold of 15-40.
- **Universe option:** all currencies, or G10 only.
- **Cost:** 5 bp per unit of turnover for G10, 15 bp for the rest.

**Walk-forward:** first out-of-sample year 2005; minimum 24 active and 12 valid months.

### T2. Time windows with half-hours and event days

**Grid per instrument: 66,528 configurations.**
- **Entry:** the first minute of each half-hour, 00:00-23:30 UTC (48).
- **Hold:** 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16 or 24 hours (11).
- **Sign:** long, short, fade or follow the previous 1 h or 4 h (6).
- **Day filter (21):**
  - the 11 of round 22: all days, each weekday, Gotobi, last and first business day, FOMC, payroll Friday;
  - **new:**
    - US, UK and Japanese pre-holiday days (`holidays` package);
    - monthly options expiry (third Friday);
    - quad witching, also the FTSE / CAC / SMI index-review effective day (third Friday of March, June,
      September, December);
    - quarter-end (last business day);
    - MSCI review day (last business day of February, May, August, November);
    - BoJ, ECB and BoE meeting days;
    - US CPI days.

**Event dates:**
- BoJ meeting dates from its statement links (as edges E8).
- ECB and BoE meeting dates and US CPI days from the public calendar dataset (2017-2023), and the BLS release
  archive where it can be fetched.
- Outside their coverage these filters are false. A filter with too few dates is ineligible under the usual
  minimum of 30 active days.

**Index-review closing auctions (the unfinished I11):** covered by the quad-witching filter with half-hour windows
around the London, Paris and Zurich closes.

**Instruments and costs:**
- every instrument of round 22 plus the new downloads;
- entry and exit at the first-minute mid of the half-hour, costs from that minute's spread (as round 22).

### IC. The ICT / SMC families on a causal simulator

**What changes:**
- The 8 families of round-22 part I (LVL, FVG, OB, OTE, BRK, CBDR, SES, SMT) now generate their orders without
  resolving them.
- A causal event-driven engine (`scripts/fvg_audit.py`'s rules) then executes them:
  - resting limit orders, market orders, cancel levels, time exits;
  - one position at a time, and the other orders are cancelled on a fill;
  - a fill at an open beyond the stop is closed at that open;
  - flat at the session end for index families as in round 22.

**Timeframe dimension** (now a choice among more options, so the Sobol points map to new configurations):
- index CFDs: M5, M15, M30 or H1;
- FX and metals: M15, H1 or H4.

**Other settings:**
- **Instruments:** every round-22 part-I instrument plus all index CFDs (round-15 D2 and the new downloads).
- **Sobol:** 128 points per family, the round-22 ranges.
- **Placebos:** a random-direction version of each family (same orders, side drawn at random), through the same
  engine and selection.

### LW. The strategy library with every parameter free, and continuous entries and exits

**Rules:** the 105 single-asset catalog strategies the library engines ran (the machine-learning and
money-management families excluded, as in rounds 15 and 17).

**Rule parameters:**
- each strategy's grid parameters, its keyword defaults, and the numeric constants inside its function are made
  parameters;
- the constants are exposed by rewriting the function so that each numeric literal (other than 0, 1, −1,
  exponents, indices and slices) becomes a keyword argument with the literal as its default;
- a rewritten function must reproduce the original output exactly at its defaults, else the original is used and
  the constants stay fixed (listed in the results).

**Ranges:**

| Parameter | Range |
|---|---|
| Grid parameter (numeric) | [min / 2, max × 2], log-uniform for integers ≥ 2 |
| Constant or keyword default x, integer ≥ 2 | [max(2, x / 3), 3x], log-uniform |
| Constant or keyword default x, float | [x / 3, 3x] |
| Booleans, strings, 0 / 1 flags | categorical |

**Execution and exits, drawn with the rule parameters** (Sobol, 128 points per strategy × instrument × timeframe):

| Dimension | Options |
|---|---|
| Direction | as the rule, or reversed |
| Exit mode | the rule's own exits (state engine), or the trade engine with the settings below |
| Entry | market; limit pullback k 0.05-1.5 ATR, time to live 1-24 bars; stop entry k 0.05-1.5 ATR, time to live 1-24; confirmation by 1-5 closes; session window, start 0-23 h London, length 4-24 h |
| Stop | ATR × 0.5-6, realised volatility × 0.5-6, structural (swing 2-10, buffer 0-1 ATR), or none (100 ATR) |
| Target | none, or 0.5-6 × the stop |
| Trailing | none, or ATR × 0.5-6 |
| Breakeven | none, or at 0.3-3 × the stop |
| Partial exit | none, or at 0.5-4 × the stop, fraction 0.2-0.8 |
| Maximum hold | none, or 2-500 bars |
| Size | one unit of notional |

**Coverage:**
- **Timeframes:** M15, M30, H1, H2, H4, H8, H12, D1, D2, W1.
- **Instruments:** the 48 of round 22 plus the new downloads.
- **Placebos:** 10 random-state placebo rules per instrument and timeframe, persistence 0.9-0.99, through the same
  execution Sobol.
- **Units and selection:** each strategy × instrument × timeframe is a unit, with daily net P&L and the default and
  nested walk-forwards (minimum 30 active, 50 valid days).
- **Pooled streams:** each strategy pooled across instruments and timeframes.
- **Time budget:** a job that would exceed its budget skips the remaining units and lists them.

### N. Round-22 parts rerun with the nested selection

**Rerun with the default and nested streams:**
- L (the stored library P&L);
- X;
- D;
- E (EIA, funding);
- M;
- B;
- C1, C2, C3;
- S;
- K;
- V;
- P.

**Replaced:**
- **T** by T2;
- **I** by IC.

**Unchanged:** spaces, data and code otherwise as round 22 (after its minimum-history fix).

### ETF. Sector, country and cross-asset rotation

**Data:** yfinance adjusted closes, monthly.

**Universes:**

| Universe | Members |
|---|---|
| Sectors | the nine sector SPDRs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY), from 1999 |
| Countries | 17 country iShares (EWA, EWC, EWD, EWG, EWH, EWI, EWJ, EWK, EWL, EWM, EWN, EWO, EWP, EWQ, EWS, EWU, EWW), from 1996 |
| Cross-asset | SPY, IWM, EFA, EEM, TLT, IEF, LQD, AGG, GLD, DBC, VNQ, each from its start |

**Signals and book** (Sobol, 256 points per universe):
- momentum, lookback 1-12 months, skip 0-1, raw or divided by volatility;
- top n 1-5, at most half the universe;
- absolute-momentum filter: none, above 0, or above the T-bill (^IRX);
- weights equal or inverse-volatility;
- rebalance 1-3 months;
- cost 5 bp per unit of turnover.

**What is counted:** the book's return minus the equal-weight return of the same universe, so that market beta is
not counted as an edge. The raw return is reported.

**Walk-forward:** first out-of-sample year 2003; minimum 12 active and 12 valid months.

### PF. The live candidates in one prop account

**Daily 2019-2025 outcomes:**
- **Gotobi B0:** test F1 of `gotobi_oos`, net of costs and swap.
- **D7-base:** ECB fixings marked daily, open trades included.
- **Noise-area breakout:** the frozen NAS100 rule of the forward test, development data.

**Prop rules:** FTMO 2-step 100k as in `scripts/prop_challenge.py`, with traders starting on random days, walking
the real calendar.

**Leverage choice:**
- each strategy gets a leverage of 0, 1, 2, 3, 4 or 6, chosen on 2019-2022 by expected value per attempt;
- the choice is evaluated on 2023-2025;
- also shown with each strategy's 2026 mean in place of its 2019-2025 mean, as in `prop_best.md`.

**Reported, not counted.**

### X10 and timeframe gaps

**Covered inside LW, T2 and IC:**
- **New instruments:** the Dukascopy downloads of `r23dl.yml` (probe first; candidates in
  `ci/r23_candidates.json`). A downloaded instrument is used only if its median price is consistent with the
  probe and its history has at least 3 years before 2019.
- **New timeframes:** H2, H8, H12 and D2 in LW; M30 and H1 for index ICT families, and H4 for FX, in IC.

## 2026

**A one-time 2026 check** (2026-01-01 to 2026-10-31) is run only for a stream that counts:
- its configurations are chosen on data up to 2025;
- it is **supported** if its 2026 mean > 0 and t ≥ 1.645;
- it is **contradicted** if its t against 2019-2025 is ≤ −1.645.

That would be the sixth use of the holdout. Part A recomputes earlier figures and is not a use.

Not financial advice.

## Addendum 1 (2026-11-18, before any real data was run): implementation details

1. **New instruments come from hour candles.**
   - The Dukascopy minute feed throttled the probe to 5 instruments in 30 minutes (run 44323400795, cancelled).
   - Hour candles (one file per month and side, `ci/r23_hours.py`) are used instead. They are checked against the
     M1 store on EURUSD 2024 (`r23dl2.yml`).
   - The new instruments therefore enter:
     - LW at H1, H2, H4, H8, H12, D1, D2 and W1 only;
     - T2 at whole-hour entries only (half-hour slots have no price and are ineligible);
     - not IC.
   - **Instruments:** USDSEK, USDNOK, EURSEK, EURNOK, USDMXN, USDZAR, USDTRY, USDPLN, EURPLN, USDHUF, EURHUF, USDCZK,
     USDSGD, USDCNH, USA30IDXUSD, USSC2000IDXUSD, AUSIDXAUD, ESPIDXEUR, XPTUSD, XPDUSD, whichever exist.
   - **Price scale:** rescaled by the power of ten that matches `ci/r23_expected_2024.json`.
2. **T2 has no BoE filter (20 filters, 63,360 configurations).** There is no free machine-readable list of past MPC
   dates. The Bank Rate page lists only the days the rate changed, and filtering on those would be a look-ahead.
3. **IC sessions and exits.**
   - **Sessions:** only US index CFDs (US500, NAS100, and the Dow and Russell when present) use the 09:30-16:00
     New York session. Every other instrument, including the European and Asian index CFDs, uses full-day bars, as
     the FX instruments do.
   - **Exits:** as round 22. Positions are not forced flat at the session end, except by the families' own time
     exits.
4. **LW details.**
   - **Session-window entries** end at midnight London at the latest.
   - **Excluded from the rewrite:** the 3 strategies whose rewritten functions did not reproduce the original
     output (S.RG.VOL_FORECAST, S.WEIRD.LUNAR, S.WEIRD.MERCURY_RETRO) keep their constants fixed. So do the 29
     that have no constants in their own body.
   - **Configurations that fail:** one that raises or exceeds 300 s is treated as no data.
