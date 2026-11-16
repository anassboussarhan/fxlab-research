# Round 23 audit: every strategy, entry, exit and parameter tested, and what is still missing (2026-11-16)

**How this was checked:**
- **Strategy catalog:** dumped from `fxlab/registry.py` with each strategy's grid, fixed defaults and keyword
  defaults (Appendix A).
- **Engines:** every entry and exit option read from `fxlab/execution` (simulator, entries, risk).
- **Round-22 spaces:** read from `docs/round22_preregistration.md` and the scripts.
- **Earlier batches:** taken from `docs/round22_audit.md`, `docs/test_ledger.md`, `docs/cloud_sweep.md`,
  `docs/methodology.md` and `docs/random_entry_opt.md`.
- **Timing:** every simulator that resolves orders was read for look-ahead.
- **Random-walk nulls:** run on the code of the three strategies still in forward tests
  (`results/fvg_audit/null_live_strategies.json`).

## Summary

**The round-22 claim "every parameter optimised" was not true for most of the library.**
- The 134 catalog strategies were only ever searched on small grids: a median of 3 configurations, 776 in all.
  - 11 have no grid at all; 66 vary a single parameter.
  - 61 keyword values, and the constants inside the functions, were never varied. Example: Connors R3 has
    RSI(2) thresholds 60 / 10 / 70 and a 200-bar average fixed in the code.
- Round 22 part L re-selected every year among those small grids; it did not widen them.
- Entries and exits were searched as **menus of fixed values** (8 entries, 14 exits), and only on 2007-2018 at
  H1 / H4 / D1. They were never searched as continuous ranges together with the rules' own parameters.
- Apart from the random-entry study, only the round-22 parts T, I, X, D, E, M, B, C, S, K, V and P used wide,
  declared ranges.

**The walk-forward's own settings were never optimised either:** top-5, expanding window, yearly re-selection,
t-statistic score, 30 active periods. Those are hard-coded values too.

**Simulator check:**
- **Look-ahead in the ICT / SMC simulators.** The FVG look-ahead (resolving each limit order before the next) is
  also in I-LVL (limit retest), I-OB, I-OTE, I-BRK and the round-21 OTE and breaker loops (X3, X5).
  - All of them failed even with that upward bias, so the verdicts stand.
  - They were never rerun with a causal simulator.
- **The three live forward tests are clean:**
  - noise-area breakout null: +0.03 bp a trade (t 0.1);
  - D7 family null: about zero, once the walk is an arithmetic martingale and open trades are marked;
  - Gotobi: fixed clock times, so a look-ahead is not possible.
- **One small bias in D7:** its reported statistics drop trades still open at the end of a window, which flatters
  the 2019-2025 and 2026 figures a little.

**Never tested in any form:**
- **Unfinished test:** the index-review closing auctions (FTSE, CAC, SMI). Started in the untested batch (I11),
  never finished.
- **Markets:** monthly carry / momentum / value on the ECB's ~30 currencies since 1999 (emerging-market and
  Nordic currencies were only in a short, fixed test); sector, country and cross-asset ETF rotation.
- **Event days:** pre-holiday, options-expiry, quad-witching, quarter-end and central-bank-meeting days as filters
  inside the time-window grid.
- **Timeframes:** half-hour time windows (needed for the last-half-hour intraday-momentum trade); the
  out-of-sample library at H2, H8, H12 and D2 (H2 and H8 ran only in round 8, on 2007-2018).
- **Portfolio:** the surviving candidates (Gotobi, D7, noise-area breakout) combined in one account.
- **Rules:** the limit-order ICT families on a causal simulator.

## 1. What was tested, by market

Classes:
- **O-wide:** walk-forward over declared wide ranges (Sobol or a full grid).
- **O-grid:** walk-forward over a small grid.
- **M:** a menu of fixed values.
- **F:** fixed or published values.

### FX and metals
(33 M1 instruments 2007-2025; ECB fixings of 28 crosses; OECD monthly data from 1976.)

| Family | Parameters and values searched | Class | Result |
|---|---|---|---|
| Strategy library: 134 rules (Appendix A) | each rule's grid; M15, M30, H1, H4 (round 17); D1, W1, MN (round 15); M5-D1 plus tick and volatility bars (round 8); both directions | O-grid | 0 after correction (rounds 8, 15, 17); round-22 L at the placebo rate |
| Joint optimisation: 96 families × 34 pairs | the family grid × 8 entries × 14 exits; H1 / H4 / D1; 2007-2018 | M | 0 survivors |
| Random-entry trade management, 7 families | stops, targets, trailing, adding, averaging, sizing; continuous, TPE; 13 timeframes; 2007-2018 | O-wide | none profitable after costs |
| ML / RL / meta-labels / triple barrier | 69 model families, 269 features; train 3,000 / 6,000 bars, test 500 / 1,000; thresholds 0.5-0.6 | O-grid | 0 |
| Time windows (T) | entry hour 0-23 UTC × hold 1, 2, 3, 4, 6, 8, 12, 16, 24 h × 6 signs × 11 day filters (all, weekdays, Gotobi, first / last business day, FOMC, payrolls): 14,256 per instrument | O-wide | best t 2.1, nothing |
| ICT / SMC (I) | 8 families, 128 Sobol points each, M5 / M15 / H1; ranges in `round22_preregistration.md` part I | O-wide | nothing (FVG withdrawn) |
| Gotobi | stops, targets, exits, entries, filters, indicators; 34-instrument panel; walk-forward (9.7) | O-grid + F | the frozen B0 rule leads; forward test running |
| Dip-buying (D) | N-low 3-20, trend SMA 50-300 or none, exit N-high 2-20 / RSI / time, RSI(2) 2-30, carry filter, time stop 2-26 weeks; daily or weekly | O-wide | nothing passes the walk-forward; the frozen D7 tracks are in a forward test |
| Monthly currency signals (M) | carry, real carry, slope, momentum 1-12 months (skip 0-1), value 3-7 years, composite; 1-3 long / short; rebalance 1-3 months; lag 1-2 | O-wide | nothing (8 currencies only) |
| Flows, fixes, calendars (17 + 11 + Tokyo / London / month-end) | published windows | F → T | covered by T |
| Events E1-E26, V1-V10: auctions, expiries, central banks, MSCI reviews, surprises, sentiment, festivals | published windows | F | nothing counts; only FOMC and payrolls entered T |
| CFTC positioning (FX, gold) | fixed z rules | F → C2 | nothing |
| Emerging-market carry (E16) | top 2 of MXN / ZAR / CNH, VIX 25 filter, 2019+ | F | nothing (gaps; no TRY) |

### Indices
(US500, NAS100, Nikkei; DAX, FTSE, CAC, Euro Stoxx, SMI, Hang Seng.)

| Family | Parameters | Class | Result |
|---|---|---|---|
| Library, time windows, ICT (US500 / NAS100) | as above | O-grid / O-wide | nothing |
| Noise-area breakout, ORB, VWAP trend (X) | band 0.5-2.5, lookback 5-30 days, step 5 / 15 / 30 / 60 min, first decision 30-120 min, trail VWAP / band / none, gap adjustment; ORB 5-60 min, stop opposite side / 0.5-2 ATR, target 1-10 R or the close, exit 12:00-16:00; VWAP step 1-60 min, start 0-120 min, band 0-0.3% | O-wide | best t 1.9 (NB, NAS100); the frozen NB rule goes to a forward test |
| Intraday momentum: the first half-hour's sign held 15:30-16:00 (I12a); end-of-day momentum after a large move (E14) | published | F | gross +0.03 bp (US500), −1.4 bp (NAS100); net negative |
| Opening-range breakout with breakeven and trailing (I12b) | published | F | +2 bp gross, about 0 net |
| Overnight returns, turn of the month, pre-holiday (M6) | published | F | positive on long samples, mostly the equity premium |
| Index-review closing auctions (I11: FTSE, CAC, SMI) | - | **never finished** (the data were still downloading) | - |

### Commodities, bonds, stocks, crypto, volatility, options

| Market | Family | Parameters | Class | Result |
|---|---|---|---|---|
| Commodities | C1 spreads / ratios (Brent-WTI, gold / silver, copper / gold) | z lookback 10-250, entry 1-3, exit 0-1, stop 3-5 z; D1 / H4 | O-wide | best t 1.2 |
| | C2 managed-money positioning | percentile 52-260 weeks, extreme 5-30%, contrarian / trend, hold 1-12 weeks | O-wide | t −1.5 |
| | C3 cross-commodity momentum / reversal / seasonality (ETFs) | lookback 1-12, skip 0-1, n 1-5, seasonality 3-10 years | O-wide | t −1.5 |
| | E-EIA crude, E-NG gas storage | entry 1-30 min after, exit 15 min-16:00, surprise sign / first-move follow / fade, threshold 0-2 sd | O-wide | t −2.1 to −3.1 |
| Bonds | B momentum / carry / value (8 countries, 1976-2025) | momentum 1-12; n 1-3; rebalance 1-3 | O-wide | t 1.6 |
| | Treasury auctions (E7), JGB auctions (E25) | published | F | nothing (all \|t\| < 2) |
| | TLT into month-end | published | F | significant on long samples, weaker after 2018 |
| Stocks | S-ANOM momentum, reversal, 52-week high, low vol, overnight momentum | lookback 5-252, skip 0-21, quantile 5-30%, hold 1-6 months | O-wide | t 0.65 |
| | S-PAIRS, S-DEL | 64 configurations; 24 holding periods | O-wide (reduced) | t −0.2, 0.6 |
| | Swing setups (Kullamägi, episodic pivot, Minervini) | published | F (S-SWING not run in round 22) | lose against SPY |
| Crypto | K-XS cross-section | lookback 3-90, skip 0-7, quantile 10-30%, universe 20-100, rebalance 1-4 weeks | O-wide | t 0.8 |
| | K-CC cash-and-carry | funding window 1-30 days, entry 0-0.03%, exit 0-0.02% | O-wide | **counts**; 2026 shrunk to about 2.4% a year |
| | E-FUND funding-time windows | 15-240 min before / after, threshold 0-0.05% | O-wide | nothing |
| Volatility | V-SPX, V-SVXY, V-BTC | VIX / VIX3M 0.85-1.05, VIX − RV −5 to +10, level band; DVOL − RV −10 to +20 | O-wide | t 1.4, 0.0, 0.5 |
| Options | P: 7 structures on DoltHub quotes (2019-2025) | delta 0.05-0.50, wings 0.05-0.20, maturity 7-60 days, take-profit 20-90% or none, stop 1-4 × credit or none, exit 0-30 days before expiry, hedge band 0.05-0.5 | O-wide | 5 own-rule passes, nothing near the correction |
| | 92 structures; 22 published edges; tastytrade mechanics; CME FX options (round 9) | menus / published | M / F | nothing; the round-9 data have expired |

## 2. Entry types

| Entry type | Parameters | Where it was used | Values searched |
|---|---|---|---|
| Market at the next bar's open | latency 0 bars (1, 2, 5 as stress) | every engine | - |
| At a fixed minute, at that minute's mid | entry time | T (whole hours 0-23 UTC), Gotobi (14:00 and 14:06 UTC), flows and fixes | whole hours only in T |
| At a fixing | which fixing | D7 (the ECB 14:15 CET reference rate after the signal week), WM/R 16:00 London, Tokyo 09:55 | fixed |
| Resting limit at the touch, re-quoted every bar | offset k × spread, maximum wait | passive engine; joint optimisation ("limit at the signal close, 3 bars") | single values |
| Limit pullback | k × ATR from the decision price, time to live | joint optimisation | **0.25 ATR for 3 bars and 0.5 ATR for 6 bars only** |
| Stop (breakout) entry | k × ATR, time to live | joint optimisation | **0.25 ATR, 3 bars only** |
| Confirmation | number of confirming closes | joint optimisation | **1 only** |
| Session window | start and end hour | joint optimisation (London / New York); T | one window in the joint optimisation |
| Scale-in | split | joint optimisation | half at market, half on a pullback only |
| Limit retest of a level, gap or zone | retest 0-1 ATR, expiry 3-100 bars | round-22 I families, round 21 | Sobol ranges, but with the sequencing look-ahead (section 8) |
| Band / range breakout at a decision step | band, step, first decision time | X, round 19 | Sobol ranges |
| Options at a delta and maturity | delta, maturity, calendar | P | monthly (weekly below 21 days) |

**Never tested as entry types:**
- **Distance and timing:**
  - a limit at a moving average or at VWAP (pullback to a level that moves);
  - continuous limit and stop offsets and lifetimes;
  - more than one confirming close;
  - any other session window in the joint optimisation;
  - the time of day at which a daily signal is entered (fills are at the next open, or at 08:00 London in the
    round-13 engine).
- **Order types:**
  - entering in slices over a window (time-weighted);
  - stop-and-reverse brackets outside the news straddle;
  - pyramiding outside the retail money-management family, which the library engines excluded.

## 3. Exits and trade management

| Exit | Options in the engine | Values searched |
|---|---|---|
| Signal exit (flat / reverse) | every engine | - |
| A rule's own stop / target | per rule | fixed in the rule's code |
| Stop | fixed pips; ATR × k; realised volatility; structural (swing k, buffer); quantile | joint optimisation: 20 pips, ATR 1.5 / 2 / 3 and one each of the others; random-entry TPE: continuous |
| Target | affine, polynomial, log, power, piecewise, none; R multiples | joint optimisation: 1R, 2R, none and one of each function; I 0.5-5 R; X 1-10 R |
| Trailing stop | ATR multiple | joint optimisation: one; random-entry TPE: continuous |
| Breakeven, partial exit | trigger r, fraction | joint optimisation menu; random-entry TPE |
| Time stop | bars | joint optimisation: 24 bars; I 2-96 bars; D 2-26 weeks |
| Session-end or fixed-time exit | time | T, X, Gotobi, I-SES 11:00-16:00 |
| Same-bar stop and target | stop first; M1 replay in the trade engine (run 3) | - |

**Never tested:** continuous stop, target, trailing, breakeven and partial settings searched together with the
library rules' own parameters, walk-forward on 2019-2025. The random-entry study shows management alone cannot
create an edge on a martingale, so the prior is low.

## 4. Sizing and portfolios

- **In the engine:** fixed-fractional risk (0.5% by default), volatility targeting, a Kelly fraction, a
  drawdown scaler (E12), correlation-aware weights, currency-exposure caps.
- **Prop simulations:** leverage 1-5×.
- **Combinations simulated:** Gotobi with Nasdaq overnight and SVXY (`prop_mix.md`); Gotobi with gold calendar
  trades (E17); several firms at once.
- **Never simulated:** the three live candidates (Gotobi, D7, noise-area breakout) in one account.

## 5. Timeframes and bars

| Engine | Timeframes |
|---|---|
| Round 8 cloud sweep (2007-2018) | M5, M15, M30, H1, H2, H4, H8, D1, 60-tick bars, volatility bars |
| Round 15 / 17 library | D1, W1, MN / H4, H1, M30, M15 |
| Joint optimisation | H1, H4, D1 |
| Random-entry study | 13 timeframes, M1 to D1, plus tick and volatility bars |
| Round-22 I | indices M5 / M15, FX M15 / H1 (each a parameter) |
| T | whole hours |

**Never tested:**
- the library at H12 or D2, and at H2 / H8 outside round 8 (2007-2018);
- the I families at M1, M30 or H4;
- half-hour or quarter-hour windows in T;
- range and Renko bars in the round-17 and round-22 library.

## 6. Instruments

| Set | Instruments |
|---|---|
| M1 store | 28 G10 crosses, gold, silver, US500, NAS100, Nikkei (2007-2025) |
| D2 CFDs | Brent, WTI, gas, copper, DAX, FTSE, CAC, Euro Stoxx, Hang Seng, SMI, Bund, T-bond, Gilt |
| Crypto | BTC, ETH perpetuals (Binance); all USDT spot coins in K-XS |
| Daily / monthly | ECB 28 crosses, OECD 8 currencies and 8 bond markets, S&P 500 point-in-time, commodity ETFs, SPY / DIA / XLF / XLE and 20 stocks with options |

**Never tested in the library or the round-22 parts:**
- **Currencies:** emerging-market and Nordic currencies (MXN, ZAR, TRY, PLN, HUF, CZK, SEK, NOK, SGD, CNH and the
  rest of the ECB's list). They were only in the Gotobi panel and the short E16 test.
- **Indices and commodities:** the Dow, Russell 2000, ASX 200 and IBEX; platinum, palladium, grains and softs as
  CFDs.

## 7. Selection settings (the walk-forward itself)

| Setting | Value used | Ever chosen by the data? |
|---|---|---|
| Training window | expanding (round 22); rolling 3 years / 6-month test (round-8 engine); one split (round 15) | no |
| Re-selection | yearly (round 22) | no |
| How many configurations are traded | top-5 at equal weight (top-1 as a secondary) | no |
| Score | t-statistic (round 22); daily Sharpe with at least 10 trades (joint optimisation) | no |
| Eligibility | 30 (I: 20) active periods; 50 (monthly: 12) valid periods | no |
| Abstaining when nothing beats luck | diagnostic only (joint optimisation) | no |

**Why this matters:** D7 lost in 2008-2018 and won in 2019-2025. A shorter rolling window or monthly
re-selection might follow such regimes, or might just add noise. This can only be judged with a nested
walk-forward that picks these settings on earlier data.

## 8. Simulator checks

| Simulator | Used by | Check | Verdict |
|---|---|---|---|
| `fxlab.execution` vector, trade, order and passive engines | library, joint optimisation, random entry, ML | decision at the close of bar t, fill at the open of t+1; pending orders re-checked every bar; strict trade-through for limits; prefix-invariance tests | causal |
| `r22_I.trade` loop | I-LVL (limit retest), I-FVG, I-OB, I-OTE, I-BRK | resolves each limit order to the end before the next signal | **look-ahead** (+0.85 bp a trade gross on random walks for FVG); FVG withdrawn; the others failed anyway |
| `r22_I` CBDR, `r21_intraday` Silver Bullet and CBDR | I-CBDR, I1, X4 | one order per side per day, taken or not regardless of other orders | causal |
| `r21_intraday.sim_limit` loops | X3 OTE, X5 breaker | the same pattern as `r22_I.trade` | look-ahead; failed anyway |
| `r19.nb_day` | NB (forward test), X-NB | decides at a minute's close, fills at the next open; band from the 14 prior days | causal; random-walk null +0.03 bp a trade (t 0.06, 2,262 trades) |
| `family_d7` / `forward_d7` | D7 forward test | signal at the week's last fixing, fill at the next fixing | causal; see the note below |
| Gotobi forward test | B0 | fixed clock times and a calendar known in advance | causal by construction |
| K-CC | 2026 check | funding filter lagged one day | causal |
| lit6, untested intraday, ML labs, cost tuning, round-21 SMT | | market entries; skipping only while a position is open, which is known at the time | causal |
| `r20_swing` | round-20 setups | not re-read in this audit | failed anyway |

**D7 note:**
- On 60 geometric random walks, the D7 code earns +6 to +35 bp a trade on all seven rules.
- **Where it comes from:**
  - part is the upward arithmetic drift of a geometric walk (σ²/2 a period, real for any long-only book);
  - part is dropping the trades still open at the end of a window;
  - the rest is noise.
- With an arithmetic martingale and open trades marked, 400 paths give +1.3 to +4.9 bp (t 0.3-1.2).
- The real 2019-2025 and 2026 statistics drop open trades in the same way. A rerun with those trades marked is
  listed below.

## 9. The gap list, ranked

**Ranking rule:** how plausible an edge is, against what the test costs. All tests use free data and
development data only (to 2025); 2026 is used only for a pre-registered check of something that counts.

| # | Gap | Test | Prior | Effort |
|---|---|---|---|---|
| 1 | D7 statistics drop open trades | rerun the D7 2019-2025 and 2026 figures with open trades marked at the window end | it decides how much of the D7 evidence is real | minutes |
| 2 | Selection settings fixed | nested walk-forward on the stored P&L matrices of L, T, I, X, D, M, C, K, V, P. Choose, on earlier years: window 1 / 2 / 3 / 5 years or expanding × top 1 / 3 / 5 / 10 × re-selection monthly / quarterly / yearly × score t / mean / Sharpe | low-moderate (helps regime-dependent rules such as D7) | hours, no new simulation |
| 3 | Monthly currency signals on ~30 currencies | M on ECB fixings 1999-2025 plus OECD / FRED short rates: carry, momentum, value, composite, crash filter (VIX threshold optimised) | moderate for carry (a known, crash-prone premium) | hours |
| 3b | Index-review closing auctions (I11) never finished | FTSE / CAC / SMI review dates; into the close of the effective day; entry time, hold and side optimised | low-moderate (a documented flow), but CFDs do not trade the closing auction | hours |
| 4 | Event days missing from T | add day filters: pre-holiday (US, UK, Japan), monthly options expiry, quad witching, quarter-end, MSCI review days, BoJ / ECB / BoE meeting days, CPI days; and half-hour windows (needed for the last-half-hour trade) | low-moderate | hours |
| 5 | Library parameters never varied | Sobol 128 over wide ranges of every rule's own parameters, including the constants in its code, × timeframe M15-W1 × both directions, yearly walk-forward | low (the library's tail matched its placebos) | large (CI, many jobs) |
| 6 | Entries and exits as menus | continuous limit / stop offsets and lifetimes, confirmation 1-5, session windows, stop / target / trail / breakeven / partial, jointly with the rule parameters, walk-forward on 2019-2025 | low (management cannot create an edge on a martingale) | large |
| 7 | Limit-order ICT families never run causally | rerun I-LVL (retest) / OB / OTE / BRK and round-21 X3 / X5 with the causal simulator of `fvg_audit.py` | very low (all negative even with the bias) | hours |
| 8 | ETF rotation | sector SPDRs (1998+, no survivorship problem), country ETFs (1996+), cross-asset dual momentum; lookback 1-12, top n 1-5, absolute-momentum filter | low (mostly beta), not prop-tradable | hours |
| 9 | Surviving candidates together | Gotobi + D7 + noise-area breakout in one prop account (2019-2025 outcomes, 2026-shifted edges) | portfolio effect only | hour |
| 10 | Other instruments | the library and T on Nordic / emerging-market currencies, Dow, Russell 2000, ASX, platinum, palladium (Dukascopy) | low (higher costs) | large (downloads) |
| 11 | Timeframes | library at H2 / H12 / D2; I families at M30 / H4 | very low | medium |

**Still blocked by data:**
- PEAD and small-cap minute data (Warrior-style; needs Alpaca keys or paid data);
- index and commodity options, 0DTE;
- order flow;
- OANDA positioning (needs your token);
- Google Trends (rate-limited);
- OPEC and crop-report date archives;
- commodity curves (for carry);
- quarterly crypto basis (coin-margined files: next).
- **NQ futures minute data:** priced at USD 9-19 against the free credit left.

Not financial advice.

## Appendix A. The 134 catalog strategies and their parameter grids

Generated from `fxlab/registry.py`: the grid is each entry's `space`, the fixed values are its `params` and
keyword defaults outside the grid. These grids are what the library engines ran (rounds 8, 15, 17) and what
round-22 part L re-selected from every year.

| Family | Strategy | Bars | Grid searched | Fixed (never searched) |
|---|---|---|---|---|
| books | `S.BOOK.BW_AO` | M1..D1 | **none** | - |
| books | `S.BOOK.BW_FRACTAL` | M1..D1 | **none** | - |
| books | `S.BOOK.CONNORS_3DAY` | M1..D1 | short ∈ [0, 1] | - |
| books | `S.BOOK.CONNORS_CRSI` | M1..D1 | lo ∈ [5, 10, 15]; short ∈ [0, 1] | - |
| books | `S.BOOK.CONNORS_PCTB` | M1..D1 | short ∈ [0, 1] | - |
| books | `S.BOOK.CONNORS_R3` | M1..D1 | short ∈ [0, 1] | - |
| books | `S.BOOK.DOUBLE_TOP_BOTTOM` | M1..D1 | k ∈ [3, 5] | - |
| books | `S.BOOK.ELDER_IMPULSE` | M1..D1 | **none** | - |
| books | `S.BOOK.GANN_HILO` | M1..D1 | n ∈ [3, 10] | - |
| books | `S.BOOK.HEAD_SHOULDERS` | M1..D1 | k ∈ [3, 5] | - |
| books | `S.BOOK.HEIKIN_ASHI` | M1..D1 | **none** | - |
| books | `S.BOOK.ICHIMOKU_KUMO` | M1..D1 | **none** | - |
| books | `S.BOOK.LIEN_CHANNEL` | M1..D1 | thr ∈ [20, 25] | - |
| books | `S.BOOK.LIEN_DBB` | M1..D1 | n ∈ [20, 50] | - |
| books | `S.BOOK.LIEN_FADER` | M1..D1 | **none** | - |
| books | `S.BOOK.LIEN_INSIDE2` | M1..D1 | **none** | - |
| books | `S.BOOK.LIEN_PERFECT_ORDER` | M1..D1 | thr ∈ [20, 25] | - |
| books | `S.BOOK.PNF_DOUBLE` | M1..D1 | rev ∈ [2, 3] | - |
| books | `S.BOOK.PRIOR_PERIOD_BO` | M1..D1 | m ∈ [5, 20] | - |
| books | `S.BOOK.RASCHKE_ANTI` | M1..D1 | k ∈ [7]; d ∈ [10, 14] | - |
| books | `S.BOOK.TURTLE_SOUP` | M1..D1 | n ∈ [20, 55] | - |
| books | `S.BOOK.WILDER_DMI` | M1..D1 | thr ∈ [20, 25, 30] | - |
| books | `S.BOOK.WILDER_VOLSYS` | M1..D1 | k ∈ [2.0, 3.0, 4.0] | - |
| books | `S.BOOK.WILLIAMS_OOPS` | M1..D1 | k ∈ [1.0, 2.0] | - |
| books | `S.BOOK.WILLIAMS_R` | M1..D1 | n ∈ [10, 14]; m ∈ [50, 100] | - |
| books | `S.BOOK.WILLIAMS_SMASH` | M1..D1 | n ∈ [1, 3] | - |
| breakout | `S.BO.FALSE_BREAKOUT` | M1..D1 | n ∈ [20, 55]; m ∈ [1, 3] | - |
| breakout | `S.BO.OPENING_RANGE` | M5..H1 | max_hold ∈ [4, 8, 12] | - |
| breakout | `S.BO.RANGE_EXPANSION` | M1..D1 | k ∈ [1.5, 2.0, 3.0] | n=20 |
| breakout | `S.BO.RETEST` | M1..D1 | n ∈ [20, 55]; m ∈ [5, 10]; tol ∈ [0.2, 0.5] | - |
| breakout | `S.BO.VOL_BREAKOUT` | M5..H4 | k ∈ [0.4, 0.6, 1.0] | - |
| breakout | `S.TREND.DONCHIAN` | M1..D1 | n ∈ [20, 55, 100]; m ∈ [10, 20] | - |
| carry | `S.XA.CARRY` | M1..D1 | thr ∈ [0.0, 0.005, 0.01, 0.02] | panel=None; symbol=EURUSD |
| carry | `S.XA.CARRY_FILTER` | M1..D1 | thr ∈ [0.0, 0.005, 0.01] | panel=None; symbol=EURUSD |
| carry | `S.XA.CARRY_MOM` | M1..D1 | thr ∈ [0.0, 0.01]; n ∈ [21, 63, 126, 252] | panel=None; symbol=EURUSD |
| carry | `S.XA.CARRY_VOLGATE` | M1..D1 | thr ∈ [0.0, 0.01]; q ∈ [0.7, 0.8, 0.9] | panel=None; symbol=EURUSD |
| carry | `S.XA.RATE_MOMENTUM` | M1..D1 | n ∈ [1, 3, 6, 12]; thr ∈ [0.0, 0.0025] | panel=None; symbol=EURUSD |
| cross_asset | `S.XA.COINTEGRATION` | M1..D1 | k ∈ [1.5, 2.0]; crit ∈ [-3.4, -3.0] | ref=GBPUSD; n=500 |
| cross_asset | `S.XA.CURRENCY_STRENGTH` | M1..D1 | n ∈ [12, 24, 120] | thr=1.0 |
| cross_asset | `S.XA.GOLD_FX` | M1..D1 | k ∈ [1.5, 2.0] | n=1000 |
| cross_asset | `S.XA.LEAD_LAG` | M1..D1 | c ∈ [0.05, 0.1, 0.2] | ref=EURUSD |
| cross_asset | `S.XA.PAIRS` | M1..D1 | ref ∈ ['GBPUSD', 'EURGBP', 'USDCHF']; k ∈ [1.5, 2.0, 2.5] | n=500 |
| cross_asset | `S.XA.RELATIVE_VALUE` | M1..D1 | n ∈ [1000, 2000]; k ∈ [2.0, 2.5] | ref=XAGUSD |
| cross_asset | `S.XA.STAT_ARB` | M1..D1 | k ∈ [1.25, 1.5, 2.0] | n=500; refit=100 |
| cross_asset | `S.XA.TRIANGULAR` | M1..D1 | k ∈ [2.0, 2.5, 3.0] | n=200 |
| daily | `S.DAILY.ABS_MOM` | D1..D1 | L ∈ [63, 126, 252]; short ∈ [0, 1] | - |
| daily | `S.DAILY.CUM_RSI` | D1..D1 | k ∈ [2, 3]; lo ∈ [10, 35, 50]; x ∈ [55, 65]; short ∈ [0, 1] | - |
| daily | `S.DAILY.DOUBLE_SEVEN` | D1..D1 | n ∈ [5, 7, 10]; short ∈ [0, 1] | - |
| daily | `S.DAILY.FABER` | D1..D1 | n ∈ [100, 150, 200, 250]; short ∈ [0, 1] | - |
| daily | `S.DAILY.HALLOWEEN` | D1..D1 | short ∈ [0, 1] | - |
| daily | `S.DAILY.HIGH52` | D1..D1 | tol ∈ [0.0, 0.02, 0.05]; m ∈ [50, 100]; short ∈ [0, 1] | - |
| daily | `S.DAILY.IBS` | D1..D1 | lo ∈ [0.1, 0.2, 0.3]; x ∈ [0.5, 0.7]; short ∈ [0, 1] | - |
| daily | `S.DAILY.OPEX` | D1..D1 | **none** | - |
| daily | `S.DAILY.THREE_DOWN` | D1..D1 | n ∈ [2, 3, 4]; f ∈ [0, 1]; short ∈ [0, 1] | - |
| daily | `S.DAILY.VIX_SPIKE` | D1..D1 | q ∈ [0.8, 0.9, 0.95]; k ∈ [5, 10, 20] | panel=None; symbol=None |
| daily | `S.DAILY.VOL_TARGET` | D1..D1 | tv ∈ [0.1, 0.15, 0.2, 0.3] | - |
| esoteric | `S.WEIRD.LUNAR` | M1..D1 | phase ∈ ['new', 'full']; w ∈ [2, 4]; dir ∈ [-1, 1] | - |
| esoteric | `S.WEIRD.MERCURY_RETRO` | M1..D1 | mode ∈ ['normal', 'inverse'] | - |
| event | `S.EVT.FOMC_USD` | M5..H4 | start ∈ [3, 8, 12]; end ∈ [15, 17] | panel=None; symbol=EURUSD |
| event | `S.EVT.GOLD_FIX` | M5..H1 | fix ∈ ['am', 'pm']; dir ∈ [-1, 1] | panel=None; symbol=XAUUSD |
| famous | `S.FAMOUS.ALLIGATOR` | M1..D1 | scale ∈ [1, 2, 3] | - |
| famous | `S.FAMOUS.CONNORS_RSI2` | M1..D1 | lo ∈ [5, 10, 20] | - |
| famous | `S.FAMOUS.CRABEL_NR` | M5..H4 | k ∈ [4, 7]; x ∈ [0.2, 0.3, 0.5] | - |
| famous | `S.FAMOUS.HOLY_GRAIL` | M1..D1 | thr ∈ [25, 30, 35] | - |
| famous | `S.FAMOUS.RASCHKE_8020` | M1..D1 | q ∈ [0.2, 0.25] | - |
| famous | `S.FAMOUS.TD_SEQUENTIAL` | M1..D1 | n ∈ [9, 13] | - |
| famous | `S.FAMOUS.TTM_SQUEEZE` | M1..D1 | k ∈ [1.5, 2.0]; m ∈ [3, 6, 12] | - |
| famous | `S.FAMOUS.TURTLE` | H1..D1 | system ∈ [1, 2] | - |
| famous | `S.FAMOUS.WILLIAMS_VBO` | M5..H4 | k ∈ [0.3, 0.5, 0.7, 1.0] | - |
| forum | `S.FORUM.NEWS_STRADDLE` | M5..H1 | k ∈ [0.3, 0.5, 1.0]; hold ∈ [1, 2, 4] | - |
| forum | `S.FORUM.THREE_DUCKS` | M1..D1 | mult ∈ [3, 4] | - |
| geometry | `S.GEO.FIB_EXTENSION` | M1..D1 | zz ∈ [0.003, 0.005] | - |
| geometry | `S.GEO.FIB_RETRACE` | M1..D1 | zz ∈ [0.003, 0.005, 0.01] | - |
| geometry | `S.GEO.GANN` | M1..D1 | zz ∈ [0.003, 0.005, 0.01] | - |
| geometry | `S.GEO.HARMONIC` | M1..D1 | thr ∈ [-0.5, -0.3, -0.1] | zz=0.005 |
| machine_learning | `S.ML.META_LABEL` | M1..D1 | q ∈ [0.5, 0.55, 0.6]; h ∈ [24, 48]; train ∈ [3000, 6000]; test ∈ [500, 1000] | model=M.TREE.LGBM; panel=None; symbol=X |
| machine_learning | `S.ML.REGIME_GATED_ENSEMBLE` | M1..D1 | w ∈ [250, 500, 1000] | panel=None; symbol=X |
| machine_learning | `S.ML.RETURN_REGRESSION` | M1..D1 | model ∈ ['M.LIN.RIDGE', 'M.TREE.LGBM', 'M.LIN.HUBER']; k ∈ [0.5, 1.0, 2.0]; train ∈ [3000, 6000]; test ∈ [500, 1000] | h=5; panel=None; symbol=X |
| machine_learning | `S.ML.RL_POLICY` | M1..D1 | model ∈ ['M.RL.DQN', 'M.RL.PPO', 'M.RL.SAC', 'M.RL.LINUCB']; train ∈ [3000, 6000]; test ∈ [500, 1000] | panel=None; symbol=X |
| machine_learning | `S.ML.SUPERVISED_DIRECTION` | M1..D1 | model ∈ ['M.LIN.LOGISTIC', 'M.TREE.LGBM', 'M.NN.MLP']; h ∈ [1, 5, 20]; m ∈ [0.02, 0.05]; train ∈ [3000, 6000]; test ∈ [500, 1000] | panel=None; symbol=X |
| machine_learning | `S.ML.TRIPLE_BARRIER` | M1..D1 | k ∈ [1.0, 1.5, 2.0]; q ∈ [0.5, 0.55, 0.6]; train ∈ [3000, 6000]; test ∈ [500, 1000] | model=M.TREE.LGBM; h=24; panel=None; symbol=X |
| market_structure | `S.ST.CANDLESTICK` | M1..D1 | n ∈ [5, 10, 20] | - |
| market_structure | `S.ST.CHANNEL` | M1..D1 | n ∈ [50, 100]; k ∈ [1.0, 1.5, 2.0] | - |
| market_structure | `S.ST.PIVOT_POINTS` | M5..H1 | **none** | - |
| market_structure | `S.ST.SR_REVERSAL` | M1..D1 | t ∈ [2, 3]; k ∈ [3, 5] | - |
| market_structure | `S.ST.SWING_STRUCTURE` | M1..D1 | k ∈ [2, 3, 5] | - |
| market_structure | `S.ST.TRENDLINE_BREAK` | M1..D1 | k ∈ [3, 5, 8] | - |
| mean_reversion | `S.MR.BOLLINGER` | M1..D1 | n ∈ [20, 50]; k ∈ [2.0, 2.5] | - |
| mean_reversion | `S.MR.OU` | M1..D1 | n ∈ [100, 200]; k ∈ [1.0, 1.5, 2.0] | hl_min=5; hl_max=100 |
| mean_reversion | `S.MR.RSI_REVERSAL` | M1..D1 | n ∈ [2, 7, 14]; lo ∈ [10, 20, 30] | hi=70 |
| mean_reversion | `S.MR.SHORT_REVERSAL` | M1..D1 | k ∈ [1, 3, 6]; thr ∈ [1.5, 2.0, 2.5]; h ∈ [3, 6, 12] | n=500 |
| mean_reversion | `S.MR.STOCH_REVERSAL` | M1..D1 | n ∈ [5, 14]; lo ∈ [10, 20] | hi=80 |
| mean_reversion | `S.MR.TWAP_REVERSION` | M1..D1 | n ∈ [12, 24, 48]; k ∈ [1.0, 1.5, 2.0] | - |
| mean_reversion | `S.MR.ZSCORE` | M1..D1 | n ∈ [50, 100, 200]; k ∈ [1.5, 2.0, 2.5] | - |
| microstructure | `S.MS.LIQUIDITY_FILTERED` | M1..D1 | m ∈ [10, 20, 40] | - |
| microstructure | `S.MS.ORDER_FLOW` | M1..D1 | n ∈ [6, 12, 24]; thr ∈ [0.05, 0.1, 0.2] | - |
| microstructure | `S.MS.ROUND_NUMBER` | M5..H1 | mode ∈ ['break', 'fade']; h ∈ [1, 2, 4]; level ∈ ['00', '50'] | panel=None; symbol=EURUSD |
| microstructure | `S.MS.TICK_IMBALANCE` | M1..D1 | n ∈ [1, 3, 6]; thr ∈ [0.05, 0.1, 0.2] | - |
| regime | `S.RG.CHANGE_POINT` | M1..D1 | p ∈ [0.3, 0.5] | - |
| regime | `S.RG.ENTROPY` | M1..D1 | q ∈ [0.2, 0.3, 0.5] | - |
| regime | `S.RG.FRACTAL` | M1..D1 | h_lo ∈ [0.4, 0.45]; h_hi ∈ [0.55, 0.6] | - |
| regime | `S.RG.HMM` | M1..D1 | p ∈ [0.6, 0.7, 0.8] | - |
| regime | `S.VOL.REGIME_SWITCH` | M1..D1 | lo ∈ [0.2, 0.3]; hi ∈ [0.7, 0.8] | - |
| retail_mm | `S.RETAIL.GRID` | M1..D1 | n ∈ [50, 200]; step ∈ [0.5, 1.0, 2.0]; N ∈ [3, 5] | - |
| retail_mm | `S.RETAIL.MARTINGALE` | M1..D1 | z_in ∈ [1.5, 2.0, 2.5]; K ∈ [3, 5] | - |
| retail_mm | `S.RETAIL.PYRAMID` | M1..D1 | n ∈ [20, 55, 100]; N ∈ [2, 4] | - |
| seasonality | `S.SEAS.FIX_REVERSAL` | M5..H1 | k ∈ [0.3, 0.5, 1.0] | - |
| seasonality | `S.SEAS.GOTOBI` | M5..H1 | start ∈ [3, 5, 7]; short_leg ∈ [0, 1] | panel=None; symbol=USDJPY; placebo_offset=0 |
| seasonality | `S.SEAS.INTRADAY` | M5..H4 | D ∈ [20, 60, 120]; thr ∈ [1.5, 2.0] | - |
| seasonality | `S.SEAS.MONTH_END_FIX` | M5..H1 | start ∈ [8, 12, 14] | - |
| seasonality | `S.SEAS.TURN_OF_MONTH` | M1..D1 | Y ∈ [12, 24, 48] | - |
| seasonality | `S.SEAS.WEEKDAY` | H1..H4 | W ∈ [26, 52, 104] | thr=2.0 |
| seasonality | `S.SEAS.WEEKEND_GAP_FADE` | M1..D1 | k ∈ [0.3, 0.5, 1.0] | - |
| session | `S.SES.ASIAN_RANGE` | M5..H1 | **none** | - |
| session | `S.SES.LONDON_BREAKOUT` | M5..H1 | **none** | - |
| session | `S.SES.NY_OPEN_REVERSAL` | M5..H1 | k ∈ [0.5, 1.0, 1.5] | - |
| session | `S.SES.OVERLAP_MOMENTUM` | M5..H1 | n ∈ [1, 3, 6] | - |
| session | `S.SES.SESSION_MOMENTUM` | M5..H1 | h ∈ [1, 2, 3] | - |
| signal_processing | `S.SP.SPECTRAL_CYCLE` | M1..D1 | s ∈ [0.1, 0.15, 0.25] | - |
| signal_processing | `S.SP.WAVELET` | M1..D1 | k ∈ [1.0, 1.5, 2.0] | w=128 |
| signal_processing | `S.TREND.KALMAN` | M1..D1 | thr ∈ [0.02, 0.05, 0.1]; q_slope ∈ [1e-05, 0.0001] | - |
| smart_money | `S.SMC.FVG_RETRACE` | M1..D1 | tol ∈ [0.1, 0.25, 0.5]; max_age ∈ [10, 30] | - |
| smart_money | `S.SMC.SILVER_BULLET` | M1..M15 | window ∈ ['ny', 'london'] | - |
| smart_money | `S.SMC.SWEEP_REVERSAL` | M1..D1 | k ∈ [3, 5, 8]; filt ∈ [0, 1] | - |
| trend | `S.TREND.ADAPTIVE` | M1..D1 | n ∈ [10, 20]; e_min ∈ [0.2, 0.3, 0.4] | - |
| trend | `S.TREND.INDICATOR_SIGNAL` | M1..D1 | kind ∈ ['macd', 'supertrend', 'psar', 'ichimoku', 'rsi50', 'aroon', 'vortex'] | - |
| trend | `S.TREND.MA_CROSS` | M1..D1 | f ∈ [10, 20, 50]; s ∈ [50, 100, 200] | - |
| trend | `S.TREND.REGRESSION` | M1..D1 | n ∈ [50, 100, 200]; thr ∈ [1.5, 2.0, 3.0] | - |
| trend | `S.TREND.TSMOM` | M1..D1 | n ∈ [24, 120, 480] | vol_n=100 |
| trend | `S.TREND.XS_MOMENTUM` | M1..D1 | n ∈ [24, 120, 480] | - |
| value | `S.XA.VALUE_PPP` | H1..D1 | W ∈ [24, 36, 60]; thr ∈ [0.5, 1.0, 1.5] | panel=None; symbol=EURUSD |
| volatility | `S.RG.VOL_FORECAST` | M1..D1 | m ∈ [0.1, 0.15, 0.3] | - |
| volatility | `S.VOL.COMPRESSION` | M1..D1 | q ∈ [0.05, 0.1, 0.2] | n=20 |
| volatility | `S.VOL.EXPANSION` | M1..D1 | k ∈ [1.5, 1.8, 2.5] | s=3 |
