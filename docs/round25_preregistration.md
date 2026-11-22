# Round 25: the twelve free gaps of the audit (pre-registered 2026-11-22, before any round-25 data was read)

**What this round runs:** every item of the ranked gap list in `round25_audit.md` section 9 ("Run all",
2026-11-22), plus the repair of the S&P 500 CFD data found in round 24 (addendum 3 of round 24).

| Audit item | Where it runs |
|---|---|
| 1 panel selection | part P: per asset class, every rule × timeframe |
| 2 higher-timeframe filter | part P: an execution dimension |
| 3 confluence | part CF |
| 4 joint regime and day filters | part P: execution dimensions |
| 5 combined entries | part P: entry type, confirmation and session window drawn together |
| 6 categorical structure | part P: price source, moving-average type, volatility measure |
| 7 adaptive search | part AD |
| 8 stability and decay selection | part P: selection scores |
| 9 signal-strength sizing | part P: an execution dimension |
| 10 altcoins intraday | part ALT |
| 11 ATR lengths for stops | part P: an execution dimension |
| 12 HAC t | every round-25 stream, and part HAC for rounds 22-24 |

**Rules for data:**
- development data only (to 2025-12-31);
- 2026 only as stated under "2026";
- no paid data (GitHub Actions on this public repository, Dukascopy, the Binance public archive, Wikipedia);
- requests carry the project User-Agent only (`fxlab-research/1.0 (research data collection)`), no e-mail.

**What was read before writing this:**
- round-23 published statistics (`round23/r23_streams.csv.gz`): the pre-2019 (P1) t of every LW unit, used
  only to choose the 20 rules of part AD (below);
- round-24 logs: the R3 SMT and R9 CME-gap results (round 24 addendum 3);
- round-23 job timings (`round23/*_LW_meta_*.json`), used for the job layout.

No round-25 return of any kind had been computed.

## Data

- **Instruments (parts P and CF):** the 66 of round 24 R2, with USSC2000IDXUSD left out by the round-23 X10 rule:
  - FX: the 28 G10 pairs and the 14 Nordic / EM pairs (42);
  - metals: XAU, XAG;
  - indices: USA500, USATECH, JPN, DEU, GBR, FRA, EUS, HKG, CHE, USA30, AUS, ESP (12);
  - commodities: Brent, WTI, natural gas, copper;
  - bonds: Bund, T-bond, Gilt;
  - crypto: BTC, ETH.
- **Bars:** the H1, H2, H4, H8, H12, D1, D2 and W1 bars that the round-24 R2 jobs build and keep (artifacts
  `r25bars-SYM`), regrouped by class and timeframe.
- **USA500IDXUSD repair:** Dukascopy minute candles for 2024-2025 (the round-23 downloader). The store's 2024
  months are compared with the download; the download replaces the store from the first month the store lacks.
  If the two disagree by more than the round-23 format check allows (median absolute difference of hourly mid
  closes above 1 bp), the instrument keeps the round-24 note and runs on the store only.
- **Financing:** FX with the BIS policy rates of round 24 (rate differential minus mark-up, every engine); other
  instruments the flat swap, as in rounds 22-24.
- **Not in round 25:** M15 and M30. They would cost about 2.5 times H1-W1 per instrument; round 23 ran them per
  unit.

## The selection core and the counting rule

- **Selection settings** as round 23 (`scripts/wf23.py`):
  - window 12, 24, 36 or 60 months, or expanding;
  - top 1, 3, 5 or 10;
  - re-selection monthly, quarterly or yearly;
  - score t or mean.
- **New scores (item 8)** (`scripts/wf25.py`):
  - **nbr:** the median training t of the configuration and its 8 nearest neighbours in the unit's Sobol space;
  - **share:** the share of training years with a positive sum (ties broken by t);
  - **decay-h:** t with monthly weights halving every h months, h ∈ {12, 24, 36, 60}, expanding window only.
- **The extended grid:** the 120 round-23 settings plus nbr and share with every window, top and re-selection
  (60 each), plus decay with every h, top and re-selection (48). That makes 288 settings.
- **Counted streams per unit** (rule × instrument × timeframe):
  - **default:** expanding window, top 5, yearly, t;
  - **nested-plus:** each January from the inner start + 3 years, the setting of the 288 with the best inner t over
    all earlier years.
- **Diagnostics (reported, not counted):**
  - the round-23 nested stream (120 settings);
  - the fixed nbr, share and decay-24 streams (expanding, top 5, yearly);
  - the per-instrument panel streams.
- **Counting rule:**
  - the first out-of-sample period up to 2018 has mean > 0, if it exists;
  - 2019-2025 has mean > 0 with t ≥ 1.645;
  - **new (item 12):** the Newey-West t of 2019-2025 is also ≥ 1.645, with lag floor(4 (n / 100)^(2/9));
  - Benjamini-Hochberg at q = 0.10 over all round-25 real counted streams, on the iid t.
- **Placebos:** each part has one (below). Placebos go through the same selection and are reported, not counted.
  A real stream that counts while its part's placebos meet their own rule at more than twice the real rate is
  examined for a fault before it is reported.
- **Sanity checks before results are reported:**
  - **Look-ahead:** the complete signal path of parts P, CF and ALT passes the truncation test (targets identical
    on every bar before 48 cuts of a synthetic random walk), with 64 configurations at H1, H4 and D1. That path
    covers the rule, the categorical swaps, the neighbourhood sizing, the confirmation and the gate (session, day,
    regime, higher timeframe). A part whose path fails is not run until it passes.
  - **Engine:** with the new dimensions off, the engine changes reproduce the round-24 engine bit for bit.
  - **Implausible streams:** any stream with 2019-2025 t above 8 is examined for a fault before it is reported.

## Part P: the library with every execution choice free, and panel selection (items 1, 2, 4, 5, 6, 8, 9, 11)

- **Rules:** the 105 single-asset library rules of round 23, with these changes:
  - the two session rules and TTM Squeeze are replaced by their tied versions of round 24 (`scripts/r24_tied.py`);
  - S.RG.HMM is left out below H2, where it gets stuck (round 23).
- **Parameters:**
  - the rules' own parameters and exposed constants as round 23;
  - the execution dimensions below, all drawn jointly by scrambled Sobol, 128 configurations per unit;
  - seeds depend on rule and timeframe only, so configuration i is the same on every instrument.
- **Timeframes:** H1, H2, H4, H8, H12, D1, D2, W1.

**Execution dimensions** (an "on" switch is u ≥ 0.5, probability 1/2, as round 23):
| Group | Dimension | Range |
|---|---|---|
| as round 23 | direction | follow or reverse |
| | mode | native (the rule's own execution) or custom (the engine below) |
| | stop | ATR × 0.5-6, realised vol × 0.5-6, structural (2-10 bars, buffer 0-1 ATR) or none |
| | target, trailing, breakeven, partial, time stop | as round 23, each on or off |
| entries (item 5) | order type | market, pullback limit or stop entry (k 0.05-1.5 ATR, life 1-24 bars) |
| | confirmation | on or off; 1-5 bars |
| | session window | on or off; Tokyo, London or New York time; start 0-23 h; length 2-22 h (wraps midnight) |
| | all three combine | |
| day filter (item 4) | entries only on chosen weekdays | on or off; any non-empty proper subset of Mon-Fri (Mon-Sun for crypto) |
| regime filter (item 4) | entries only when the regime's percentile rank lies in a band | on or off; ADX(n), ATR(n) / price, trend strength \|c - c[n]\| / (ATR(n) √n) or spread (n-bar mean) / price; n 5-200 (log); rank window 0.25-5 years (log); band low 0-0.8, width 0.2-1 |
| higher timeframe (item 2) | entries only with (or against) the higher-timeframe state | on or off; D1 or W1 above the bar size (off for W1 bars); MA slope, close against MA, or Donchian state; length 3-200 higher bars (log); with or against |
| stops (item 11) | ATR length for stops, targets, trailing and order levels | 5-50 (log) |
| | realised-vol stop length and horizon (fixed at 50 and 10 so far) | 10-200 and 2-40 (log) |
| | target intercept (fixed at 0 so far) | 0-2 ATR |
| sizing (item 9) | neighbourhood strength | on or off; the base configuration and K neighbours (K 2-4, each rule parameter jittered by ±w, w 0.02-0.3 of its range) vote; size = max(1/(K+1), (agree - oppose) / (K+1)), fixed at entry |
| | volatility targeting | on or off; size × min(1, c × median σ / σ), σ the n-bar std of log returns (n 20-500, log), median over all earlier bars, c 0.3-1, fixed at entry |
| categorical (item 6) | price source of the rule's signal | close, HL2, HLC3 or OHLC4 (execution always on real prices) |
| | moving-average type | native, SMA, EMA, WMA or Hull, replacing the rule's own `sma` / `ema` calls |
| | volatility measure | native ATR, or k × std of close changes (k 0.8-2.5), replacing the rule's own `atr` calls |

**Definitions:**
- **The gate:** all entry filters act together, as `session_entries` of round 23. A new entry or an addition is
  taken only when every active filter allows it; exits and reductions are always allowed. A reversal on a blocked
  bar exits and waits.
- **Higher-timeframe state:** computed only on completed higher bars, so a bar uses the state as of the previous
  trading day (D1) or week (W1). The days and weeks are those of the bars (17:00 New York).
- **Rules that only signal entries** (trade-mode rules): a neighbour's vote is the side of its last entry signal.

**Panel selection (item 1):**
- Panels are the asset classes above (FX 42, metals 2, indices 12, commodities 4, bonds 3, crypto 2).
- At each re-selection, each configuration's training statistics are pooled over the class's instruments (their
  monthly sums, squares and day counts added). The top configurations are then applied to every instrument of the
  class.
- **Counted:** the class's pooled stream (equal weight over the instruments trading that day), two forms:
  - **panel default:** expanding, top 5, yearly, pooled t;
  - **panel nested:** each January the setting of the 120 whose pooled inner stream has the best t over all
    earlier years.
- **Placebos:**
  - 10 random-state units per instrument and timeframe through the same execution space, as round 23;
  - their panel streams per class.
- **Jobs:** one per class group (FX, or the other 24 instruments), timeframe and rule chunk. Rule chunks are sized
  from round-23 unit timings.
  - Unit cap 3600 s; a unit past it keeps the configurations it finished.
  - A rule × timeframe whose instruments are not all finished is reported, and its panel streams are not counted.

## Part CF: confluence (item 3)

- **Members:** the 105 rules at their library defaults (tied defaults for the three of round 24). On every
  instrument and timeframe of part P, each member's side and its daily returns (market entry, its own mode) are
  computed once.
- **The rule:** each January, members are ranked by their training score over the window before it. The top n
  with a positive score are taken, and the side is the sign of their summed sides when |sum| ≥ k × (members
  taken), else flat.
- **Free:**
  - n 3-30 (log);
  - k 0.5-1;
  - ranking window 12, 24, 36 or 60 months, or expanding;
  - score t or mean;
  - for trade-mode members, their entry signal held 1-48 bars (log);
  - every execution dimension of part P.
- **Sampling:** 128 configurations per instrument × timeframe; streams and counting as part P (default and
  nested-plus).
- **Placebo:** the same procedure with the members ranked by random numbers.

## Part AD: adaptive search (item 7)

- **Rules:** the 20 with the highest median round-23 P1 t over their real, counted LW units (the three voided
  rules excluded). They are written to `ci/r25_ad_rules.json` with this document:
  - S.MR.ZSCORE, S.RG.CHANGE_POINT, S.BOOK.CONNORS_PCTB, S.DAILY.VOL_TARGET, S.DAILY.HIGH52;
  - S.BOOK.CONNORS_R3, S.DAILY.OPEX, S.DAILY.CUM_RSI, S.TREND.REGRESSION, S.BOOK.BW_FRACTAL;
  - S.BOOK.HEIKIN_ASHI, S.DAILY.ABS_MOM, S.MR.BOLLINGER, S.DAILY.HALLOWEEN, S.BOOK.HEAD_SHOULDERS;
  - S.BOOK.CONNORS_CRSI, S.TREND.DONCHIAN, S.BOOK.PNF_DOUBLE, S.SEAS.WEEKEND_GAP_FADE, S.GEO.HARMONIC.

  Their medians are all negative (-0.49 to -0.65).
- **Universe:** the 28 G10 pairs, XAU and XAG at H1, H4 and D1.
- **Search space:** part P's (rule parameters and every execution dimension).
- **Procedure:**
  - 64 Sobol points to start;
  - each January from the data start + 3 years, a TPE sampler (Optuna, multivariate) is refitted on every
    earlier trial, scored by its t over all earlier years, and proposes 32 new trials;
  - the top 5 trials by that score are traded that year, equal weight.

  That is about 512 evaluations per unit.
- **Counted:** that stream, on 2019-2025 only (P1 chose the rules, so it is not used again). The HAC condition
  and BH apply as above.
- **Placebo:** the same procedure on one random-state unit per instrument and timeframe.

## Part ALT: altcoins intraday (item 10)

- **Universe, point in time:**
  - all USDT-margined perpetuals in the Binance public archive (`data.binance.vision`), delisted ones included;
  - left out: BTC, ETH, stablecoins and dated contracts;
  - in year Y, the 30 with the highest mean daily quote volume over Y - 1, among those with 90 or more days in
    Y - 1 (fewer than 30 if fewer qualify).

  A coin trades only in its member years. Its whole earlier history is used for training.
- **Data:**
  - 1-minute klines (bid and ask from a half-spread of 3.1 bp that includes fees, as BTC and ETH since round 19);
  - funding from the archive's funding-rate files, charged as carry at each funding time (longs pay a positive
    rate);
  - no flat swap.
- **What runs:** part P's rules and execution space at H1, H2, H4, H8, H12 and D1, with the per-coin streams
  (default, nested-plus) and the ALT panel (pooled over the members of each year; panel default and panel
  nested), all counted.
- **Placebos:** 10 random-state units per coin and timeframe, and their panel streams.

## Part HAC: rounds 22-24 (item 12)

- **Newey-West t of 2019-2025** for:
  - every round-24 stream meeting the own rule, from the stored series;
  - the 100 round-23 LW own-rule passes with the highest 2019-2025 t, rerun deterministically with the
    round-23 code (`scripts/r23_LW.py`, now able to save unit series). A rerun stream counts as reproduced when its
    2019-2025 t matches the published one within 0.01.
- **What it changes:** nothing is re-counted. It shows how far serial correlation inflated the old t.

## 2026

Not used for anything above. A stream that counts gets one pre-registered check on 2026-01-01 to 2026-10-31 with
its frozen rule, written before 2026 data is read.

## Order and cost

- **Order of the CI runs:**
  1. engine changes and regression check;
  2. truncation test;
  3. data (USA500 repair, bar regrouping, altcoin archive);
  4. part P;
  5. CF;
  6. AD;
  7. ALT;
  8. HAC;
  9. combine.
- **Cost:** GitHub Actions minutes on a public repository and public data only, so no charge. The estimate is
  about 2.2 million core-seconds for P, 0.4 for AD, 0.5 for ALT and 0.1 for CF. That is roughly 12 hours of the
  20-runner pool, shared with the rest of round 24.

## Blocked (unchanged)

- insider buying (SEC archive needs a contact e-mail);
- Congress trades (datasets gone);
- OANDA positioning (token);
- small caps and stock intraday (Alpaca keys);
- NQ futures and 0DTE (paid);
- order flow (paid).

Not financial advice.

## Addendum 1 (2026-11-22, before any round-25 data was read)

Changes found while building and testing the code on synthetic data:

1. **Three more rules tied by role** (`scripts/r25_tied.py`). Round 23's unit statistics show that drawing every
   literal separately broke three rules besides TTM Squeeze, each because a rolling window and its min_periods
   were drawn apart:
   - S.DAILY.HIGH52 failed on 88% of round-23 draws;
   - S.DAILY.VOL_TARGET on 57%;
   - S.VOL.COMPRESSION on 22%.

   The audit (`round25_audit.md`) named only TTM. They are now tied:
   - HIGH52: L for the 52-week window;
   - VOL_TARGET: n for the window, ann for the annualisation;
   - COMPRESSION: W for the rank window, f for its min_periods as a share of W, a for the arming window.

   At their defaults each reproduces the library rule exactly (H4 and D1, synthetic). The part-AD list stays as
   written: it was chosen from the round-23 draws that ran.
2. **One thread per process.** In forked workers the HMM rule's k-means deadlocked in OpenMP once the parent had
   started OpenMP threads (seen in a local part-CF test, stack dumps). Every round-25 job sets OMP / BLAS / numba
   to one thread per process. This is a likely cause of the round-23 HMM stalls. HMM is still left out at H1, as
   written.
3. **Look-ahead found and fixed before any run.** The first truncation test (synthetic H4) flagged the regime
   filter. Its rank window, in bars, came from the bar density of the whole sample. It now comes from the bar
   count of the instrument's first year, known once that year is over. The session and weekday gates take the bar
   step from the first 200 bars. After the fix every tested configuration is identical before every cut. The full
   gate runs on CI (H1, H4, D1 and part CF) before part P starts.
4. **Part ALT panel, point in time.** Only the members' bars are downloaded, so pooling every coin's history in
   training would use future membership. In year Y the panel plan is made on the pooled history of Y's members
   only (each with its whole earlier history) and applied to them during Y. Per-coin streams train on the coin's
   own history.
5. **Part ALT bars** come from the archive's 1-hour klines (H1-D1 need nothing finer). Bid and ask are 3.1 bp
   either side, as written. A test with a constant position confirms that funding is charged at each stamp (longs
   pay a positive rate) in place of the flat swap.
6. **Part AD timing, made explicit.** The 32 trials proposed each January are scored on the days before that
   January and compete with the earlier trials for the year's top 5. A trial's returns on a day depend only on
   data up to that day.
7. **Job layout** (`ci/r25_matrix.py`): 62 part-P jobs, about 580 core-hours. HMM and ENTROPY at H1 / H2 on the
   42 FX pairs are too large for one job and are split by instrument. Their panel streams are made in the combine
   from the saved unit aggregates.

## Addendum 2 (2026-11-22, gate and data results; no round-25 return had been computed)

- **Look-ahead gate passed** (r25data 45055180321), 48 cuts of a synthetic random walk:
  - H1: 208 configurations, 0 with look-ahead, 0 errors;
  - H4: 210, 0, 0;
  - D1: 210, 0, 0;
  - part CF (H4): 16 configurations, 0, 0.

  Two (H1) and three (H4) configurations differ on the last bar of a cut only (rules acting on a bar's own close),
  as allowed in round 23.
- **USA500IDXUSD refill accepted.** The store ends 2024-05-23. 2,084 hours of 2024 were compared, with a median
  absolute difference of 0.0 bp. The Dukascopy hour candles replace the store from 2024-05-01. The regrouped bars
  have no missing instrument or timeframe.
- **Altcoin universe:** 919 USDT perpetuals in the archive after the exclusions. 2020 has no member: no altcoin had
  90 days of 2019 data in the archive's files, so ALT starts in 2021. 2021-2025 have 30 members each (60-326
  eligible), 76 coins in all.
