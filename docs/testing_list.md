# Testing list (queue of what is still worth testing)

Updated 2026-11-04. Everything done is in `docs/test_ledger.md`; this page is only what is left, ranked by
(prior chance it works) x (value if it works) / (cost to test). Every item gets written rules in a
pre-registration before its data is read; 2026 stays untouched until a frozen rule is evaluated on it.

## Running now

| ID | Test | Status |
|---|---|---|
| P3 | Frozen Gotobi basket B0 on 2003-2006 (Dukascopy, before the project's data) | **done: confirms** (+8.4 bp, t 3.85, 152 nights, every year positive; control -1.1 bp) |
| P4 | Post-fix yen reversal | done: fails on the other crosses and on 2003-2005 |
| Q | Queue batch: VS1-VS5, T1, BS1, C5E, P1 with Australia | **done: nothing new counts** (BS1 closest: Sharpe 0.31, t 1.5); see `docs/edges_batch_results.md` |
| Forward test | B0 live-data nights from 2026-11-05, monthly report on the 3rd | scheduled |

## A. Using the volatility forecast in spot FX (the one strongly predictable part of price)

Why these: volatility is predictable (HAR out-of-sample R^2 0.3-0.5, rank correlation 0.66 at 20 hours) but a
spot position pays only on direction. A delta-hedged option earns roughly 1/2 x gamma x (realised variance -
implied variance), so the natural way to sell a volatility forecast is an option, which needs option data
(section C). In spot, a volatility forecast can only help in four indirect ways, one test each.

| ID | Idea | Why it might work | Data | Prior |
|---|---|---|---|---|
| **VS1** | **Size the Gotobi trade so the chance of a prop daily-loss breach stays constant** (HAR forecast of the 14:00 -> 00:53 UTC window's range; leverage = the level at which P(loss > daily limit) equals a fixed target), judged by prop-sim pass rate and breach count, not by mean | The earlier volatility sizing was judged on mean / CVaR and was not robust; the prop objective is different: a breach ends the account, so equalising breach risk can allow a higher average leverage at the same failure rate | M1 2007-2025 in hand | Medium-high |
| **VS2** | **Choose the gamma, not the direction:** run a mean-reversion grid (short gamma) when the forecast variance ratio VR(k) = Var(k-bar move) / (k x Var(1-bar move)) is below 1, and a trend rule (long gamma) when it is above 1 | A grid is a spot short straddle, trend following a spot long straddle (Fung-Hsieh). Their P&L depends on the variance ratio, not the volatility level. Grids were slightly positive before costs because short-term VR < 1; if VR itself is forecastable, switching could clear costs | M1 in hand | Low-medium |
| **VS3** | **Implied minus realised volatility as a directional signal for gold** (GVZ minus the trailing 21-day realised gold volatility, monthly; direction from the fit period) | Della Corte, Ramadorai & Sarno (2016): currencies with expensive volatility insurance subsequently depreciate; the City University straddle study finds implied minus realised predicts spot. GVZ is in the flow store | GVZ 2008-2025 + XAUUSD | Medium |
| **VS4** | **Cost-to-volatility gate on the intraday effects that are real before costs** (ECB-fix W, gold and FX end-of-day momentum, the post-fix reversal if P4 shows a gross edge): trade only when the forecast move of the window is large relative to the quoted spread | Their gross edge is 1-3 bp below cost. If the gross edge scales with volatility while the spread does not, high-forecast days should have a better edge / cost ratio. E14 conditioned on the realised earlier move and lost, so the prior is low | M1 in hand | Low |
| VS5 | Volatility-forecast jumps and carry-pair direction (forecast rises sharply -> short AUD/JPY, NZD/JPY next day) | Risk-off moves come with volatility spikes; whether a predictable spike carries direction is untested | M1 + VIX in hand | Low |

## B. Slow trend on what a prop firm offers

| ID | Idea | Why | Data | Prior |
|---|---|---|---|---|
| **T1** | Slow time-series trend (EWMA 50-200 and 12-month sign, volatility-scaled) across index CFDs, gold, silver, oil, bond CFDs and the FX majors, real CFD costs and swaps, 2007-2025 | The one trend form with a positive record (22 ETFs: +0.51% a month, t 1.65, Sharpe 0.38; Kurth et al.: slow trend Sharpe 0.40 after 2008). Expected: a small edge that suits a personal account; prop-challenge fit judged separately | M1 / D1 in hand (bonds partial) | Medium for "small edge", low for "prop fit" |

## C. Data still needed (free first, paid only after an itemised estimate is approved)

| ID | Idea | Data and how to get it | Cost |
|---|---|---|---|
Databento cost check (2026-11-04, the $125 free credit; nothing downloaded yet): K1 fed funds futures $13.08; F1 futures
volume $4.58; O1 euro and yen options (bars, definitions, settlements, underlyings) $54.18; gold options $177.08; all
$250.34. K1 + F1 + FX options = $71.84, inside the free credit: approved and downloaded; F1, K1, O1 done (none counts).

| O1 | Sell FX / gold volatility when implied is above the HAR forecast (options on 6E, 6J, GC, or OTC through a broker) | Historical option prices: CME DataMine or a vendor (paid); free path is the Saxo daily snapshots already collecting (about a year needed) | Quote first |
| O2 | FX volatility risk premium and risk reversals across currencies (Della Corte et al.; Jurek) | Saxo snapshots (collecting daily since 2026-11-04) | Free, forward |
| R2 | Japanese retail FX flow and positioning, hourly (Gaitame, 29 pairs incl. USD/JPY, EUR/JPY), as a contrarian signal
and as a Gotobi filter | TraditionData "Japanese Retail Foreign Exchange Flow" (from February 2015; trial / sample on request) or
Nasdaq Data Link "Global Retail FX Flow" (GRFF, from 2012, about 6% of global retail spot volume) | No public price: ask for a
trial / sample first |
| M2 | Daily speculative positioning in commodity futures (gold / silver if covered) | Nasdaq Data Link "Marex Daily
Speculative Positioning" (MDSP, from 1992); our weekly CFTC versions failed (V3), so low priority | No public price |
| R1 | Retail positioning, contrarian (2025 study: against retail flow over 4-20 hours) | OANDA order / position books: needs a free OANDA practice token as the `OANDA_TOKEN` secret | Free, needs the user |
| X1 | Option-expiry pinning at the 10:00 New York cut | DTCC public option trades (collecting daily) | Free, forward |
| K1 | Pre-FOMC dollar drift as published (Karnaukh) | Next-month fed funds futures history (paid) | Quote first |
| F1 | Real futures volume / CLS flow | CME DataMine / CLS (paid) | Quote first |

## F. Round 9 (2026-11-05): what nine rounds of search still had not tested

**Done 2026-11-08: nothing counts** (`docs/round9_results.md`); F2, S3, S4 not tested for lack of data.

Found by listing every tested item (`docs/test_ledger.md`, all pre-registrations) against what is known about where
FX money comes from (forced flows at known times, risk premia, information in flows) and a new search
(`docs/literature_check.md` section 11). Ranked by prior x value / cost.

| ID | Idea | Why it might work | Data | Cost | Prior |
|---|---|---|---|---|---|
| **N1-N3** | **Economic surprises with real consensus forecasts.** N1: after a high-impact release with a large standardised surprise (actual - forecast), trade in its direction from release + 5 min for 1-4 hours. N2: per-currency surprise index (sum of the last 1-3 months' standardised surprises, our own "Citi index"), weekly / monthly G10 cross-section. N3: drift in the hours before high-impact releases (dates and times only) | The largest category never tested: earlier tests had fewer than 10 surprises (no consensus archive). Literature: most of the reaction is within minutes (low prior for N1); surprise momentum and pre-release positioning are less studied | Forex Factory calendar 2007-01 to 2025-04 with actual / forecast / previous / impact (HuggingFace `Tropstan/Forex_Factory_Calendar`, MIT licence); quality check first (time zone, forecast coverage) | Free | Low-medium |
| **G1** | **The Gotobi trade on CME yen futures in a futures prop firm.** Futures firms force you flat each day (Topstep: by 3:10 PM CT, trading reopens 5:00 PM CT) but 5 PM CT is 22:00 / 23:00 UTC, so the fix leg (22:00 or 23:00 UTC -> 00:53 UTC) fits inside one of their trading days. Short 6J = long USD/JPY | The fix leg is 36% of B0's mean on 2019-2025 (t 2.7) with 24% of its variance, i.e. a better ratio than the whole night; futures firms have no swap, cheaper evaluations and allow several accounts. Risk: 6J costs (1 tick = 0.76 bp + commission, wide spreads at the 5 PM reopen) may take most of a 2-3 bp gross leg; payout rules (winning-day counts, 50% consistency) suit frequent traders | M1 USD/JPY in hand; the firms' rules (Topstep, Apex, Tradeify, MyFundedFutures, Lucid) to verify | Free | Medium (value high) |
| **O3-O5** | **Signals from the CME option data already bought.** O3: 25-delta risk reversals (level and change) -> next 1-5 days of EUR/USD, USD/JPY. O4: pinning to the largest open-interest strike before the 10:00 New York cut on CME expiry days (makes E5 testable). O5: open interest x gamma near spot -> next day's intraday reversal or momentum (the Baltussen et al. switch) | Uses data paid for with the free credit (artifact kept to early February 2027). CME is a small part of FX options, so the signals are noisy proxies | Databento 6E / 6J option settlements, definitions, open interest 2010-2025 (in hand) | Free | Low |
| **F2** | **Signed order flow from CME currency futures** (aggressor-side volume per hour / day) -> next hours' and days' returns (Evans-Lyons order flow, which was "untestable" because interbank flow is private) | Order flow explains most same-period FX moves and has some forecasting power at days to a month (Evans & Lyons 2005); CME flow is a public, partial proxy | Databento `trades` for 6E / 6J; price check first, download only inside the remaining free credit (about $50) | Free if it fits | Low-medium |
| S1 | Same-calendar-month return seasonality and realised-skewness sorts across currencies | Keloharju et al. (2016) find it in commodities (t 1.9) but did not test currencies; currency skewness premia (Cambridge FX initiative) | FRED H.10 daily FX from 1971 | Free | Low |
| S2 | Shanghai gold premium (SGE price in dollars minus London) -> gold over the next days | Chinese physical demand; the premium is reported to lead in Asian hours | SGE daily benchmark history (access to check) | Free if reachable | Low-medium |
| S3 | Norges Bank's monthly announcement of its daily NOK purchases / sales -> EUR/NOK after the announcement | A pre-announced official flow | Norges Bank announcement history; NOK costs are high on prop accounts | Free | Low |
| S4 | Bank of Japan JGB purchase operations (rinban, announced 10:10 Tokyo, right after the fix) -> USD/JPY | A known-time Tokyo event inside the post-fix window (the post-fix reversal lead) | BoJ operation results (scrape) | Free | Low |
| S5 | Coupon conversion: US Treasury coupon dates (15th of Feb / May / Aug / Nov, month-ends) and Japanese holders' yen buying | A calendar flow of the Gotobi kind | Dates by rule, M1 | Free | Low |
| S6 | New NISA month-start buying of foreign funds (from 2024) | Large, regular yen selling; too short a history (24 months) | M1 | Free | Forward only |

Not tested on purpose: hedging opposite positions across prop firms (against every firm's terms), trading on slow
prop price feeds (prohibited), the BoJ's ETF purchases on falling mornings (ended in 2024), lunar cycles.

## L. Round 17 (2026-11-10): regimes, harder

**Done 2026-11-10** (`docs/round17_results.md`).
- **What was added over round 16:**
  - intraday rules at H4 / H1 / M30 / M15;
  - daily, weekly and monthly switching;
  - regimes per instrument for 33 instruments;
  - Hedge with forgetting, CUSUM resets and pooled boosting;
  - intraday-state features (5-minute volatility, jumps, autocorrelation, Asian-session share) and daily CME
    implied volatility / risk reversals.
- **Result:** 0 of 93 tests count, and 0 of 486 timing tests.
- **A flaw found and fixed:** a 14-hour look-ahead in the feature-tercile units (fill-to-fill marks), caught by a
  one-day-lag check and fixed before the final run.

## K. Round 16 (2026-11-09): market regimes, the best strategy per regime, options data

**Done 2026-11-09** (`docs/round16_results.md`).
- **Methods tested:** six regime-method families (strategy momentum, performance filter, clusters with 2-8 regimes,
  single-feature splits, HMM, ridge / boosting) in three library tiers. Each family's method, number of regimes
  and setting was chosen on 2011-2018.
- **Result on 2019-2025:** 0 of 18 count. None beats static selection or its time-shifted placebo.
- **Options data:** as regime inputs, timing signals and a dip-buying filter, it adds nothing.
- **Number of regimes:** BIC prefers 1-2, and none out of sample.
- **Exploratory:** regime effects tied to yen implied volatility repeat across decades (p = 0.0005 with 2,000
  shifts), but they cannot be traded.
- **Not bought:** sterling / AUD / CAD / CHF CME options (USD 60.29).

## J. Round 15 (2026-11-09): every rule at D1 / W1 / MN, the book systems, every feature, free data

**Done 2026-11-09** (`docs/round15_results.md`). 233 rule units (91 rules incl. 26 new book systems) and 838 feature
units, full grids and both directions, on 33 instruments: 0 survive BH. The weekly dip-buying family was supported
on 2026 (175 trades, +0.38R, t 2.2) and is the forward test's D7-family track. Free data:
- the Forex Factory archive: the one-day surprise drift is +5.4 bp, t 2.2 on 2019-2025, not significant after the
  correction;
- CBOE strategy indices, two-year yields and the surprise index: fail;
- commodities, indices and bonds: see the results.
Still missing: retail sentiment history, Google Trends, TFX history, and the OANDA books (needs a token).

## I. Round 14 (2026-11-09): re-check, missed trend variants, trend prediction, Double Seven on 2026

**Done 2026-11-09** (`docs/trend14_results.md`). Round 13 reproduced by an independent engine and by ECB fixings;
0 of 18 trend variants and 0 of 8 prediction rows count; no trendiness predictor works. Double Seven on weekly
bars passed its pre-registered 2026 check (33 trades, +0.78R, t 5.4 clustered) but lost heavily in 2008-2010 and
2015-2016; next step is a live forward test at small size with a written crash rule.

## H. Round 13 (2026-11-08): long trades that let winners run

**Done 2026-11-08: nothing in part A counts** (`docs/long_results.md`): 12 long-trend systems on 28 FX crosses and
on gold, silver and three indices; FX loses before costs, non-FX earns less than holding long. In part B, one weekly
mean-reversion unit counts by its rule but lost in 2008-2018 (forward watch list).

## G. Round 10 (2026-11-08): two rule families from the multilingual sweep

**Done 2026-11-08 (round 12): nothing counts** (`docs/anatomy_results.md`). MS1, QM1 and CL1-CL7 were tested as
published (0 of 19 rows count), together with a failure anatomy and a fix search over every catalog strategy (0 of
196 count). R1 still needs the user's OANDA v20 token.

| ID | Idea | Rule sketch (to be pre-registered) | Data | Prior |
|---|---|---|---|---|
| MS1 | Malaysian SNR | H4 line-chart turns -> A-levels (resistance) / V-levels (support) at the body price; fresh until a wick touches; fade the first revisit of a fresh level in the direction of the D1 storyline, stop beyond the level, target the next fresh level | M1 in hand | Low (support / resistance family) |
| QM1 | Quasimodo (QM) pattern | Zigzag high - low - higher high - lower low (bearish QM; mirror for bullish); limit at the left shoulder level, stop beyond the head, target 2R | M1 in hand | Low |
| CL1 | Dual Thrust (N 4, k 0.5) on FX majors, gold, indices H1 | As published; daily open triggers | M1 in hand | Low |
| CL2 | R-Breaker and the Fiali four-price breakout, intraday, flat at the close | As published | M1 in hand | Low |
| CL3 | Aberration (35-bar Bollinger, 2 sd) and King Keltner, D1 / H4 | As published | M1 in hand | Low |
| CL4 | Thermostat (choppiness switch) and Bollinger Bandit | Pruitt & Hill rules | M1 in hand | Low |
| CL5 | Camarilla pivots: fade H3 / L3, breakout H4 / L4 | As published | M1 in hand | Low |
| CL6 | Elder triple screen | Weekly MACD-histogram slope + daily force index / stochastic + intraday stop entry | M1 in hand | Low |
| CL7 | Bill Williams Alligator + fractals; Puria | As published | M1 in hand | Low |
| R1 | OANDA position / order books | Needs a v20 (not MT4 / MT5) OANDA account's API token | User | Medium |

## D. Forward watch list (no new test; judged on live data only)

USD/JPY H4 high-confidence model (mostly a 2022 / 2024 trend ride); FS1 66 settings; the 60/40 rebalancing
calendar (Sharpe 0.6 after publication); euro COT speculative pressure (1 of 52); ECB pre-decision drift (19
meetings); USD/JPY CPI surprise (11 releases); gold breakout short; skipping the 25th and month-end (FW2); the
forward-v1 / v2 win models; the post-fix USD/JPY / EUR/JPY reversal as a second Gotobi leg (short 09:55 -> 10:20
Tokyo; +1.6 / +0.8 bp on 2019-2025, absent in the other crosses and in 2003-2005); Double Seven on weekly bars
(round 13: buy a 7-week closing low above the 200-week average, sell a 7-week closing high; +64R, t 3.61 on
2019-2025 over 33 instruments, but t -0.35 on 2008-2018; round 14: supported on 2026, 33 FX trades, +0.78R,
t 5.4 clustered, with heavy losses in 2008-2010 and 2015-2016; `docs/trend14_results.md`).

## E. Dropped (tested on unseen data and failed; kept here so they are not re-run)

Databento options: selling euro / yen volatility, hedged or not, when rich or always (O1: -13.2 bp a cycle when rich,
-5.7 bp always, 2018-2025; implied volatility is not rich against a forecast and costs take the small premium).


Databento: real futures volume (F1: V1b -11.1 bp, V1a never fires); Karnaukh pre-FOMC drift as published (K1:
-13.9 bp, 31 trades). Lead from K1, information only: the dollar weakens into FOMC decisions (-10.9 bp before
93 no-change meetings, t -2.0); testable only on new meetings.


Free retail / positioning: CFTC small traders (FF1), FFAJ Japanese retail positions monthly (FF2), daily
trend-follower flow and crowding estimated from prices (FD1a, FD1b). The paid feeds (R2, M2) stay listed only for
their higher frequency (hourly retail flow), not for a different idea.


Q batch: volatility sizing of the Gotobi trade (VS1), variance-ratio switching (VS2), gold volatility premium as
a direction signal (VS3), the cost-to-volatility gate on E2 / E3 / K3 (VS4), volatility jumps (VS5), slow trend on
prop CFDs (T1), euro COT before 2004 (C5E). BS1 rate-expectation carry: positive but not significant (forward watch).


US-euro 2-year gap and its extension to GBP, CAD, CHF (P1); gold before the Chinese New Year 1980-2000 (P2); GLD
holdings flows (P5); futures and ETF volume (V1, V10); CFTC positioning changes (V3); SNB deposits (V4); news tone
(V5); fed funds futures moves (V7).

## Low priority (tested in a close form, or weak evidence)

BSIC rate-expectation carry exactly as written (its authors find no edge after 2010; close to V6 / P1); Australia
and New Zealand 2-year gaps for P1 (parser / access); euro COT C5 before 2004 (too little data).
