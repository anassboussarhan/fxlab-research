# Testing list (queue of what is still worth testing)

Updated 2026-11-04. Everything done is in `docs/test_ledger.md`; this page is only what is left, ranked by
(prior chance it works) x (value if it works) / (cost to test). Every item gets written rules in a
pre-registration before its data is read; 2026 stays untouched until a frozen rule is evaluated on it.

## Running now

| ID | Test | Status |
|---|---|---|
| P3 | Frozen Gotobi basket B0 on 2003-2006 (Dukascopy, before the project's data) | first run partial (USD/JPY 2003-2005: +4.2 bp, t 1.3); full download running (`early-gotobi`) |
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
| O1 | Sell FX / gold volatility when implied is above the HAR forecast (options on 6E, 6J, GC, or OTC through a broker) | Historical option prices: CME DataMine or a vendor (paid); free path is the Saxo daily snapshots already collecting (about a year needed) | Quote first |
| O2 | FX volatility risk premium and risk reversals across currencies (Della Corte et al.; Jurek) | Saxo snapshots (collecting daily since 2026-11-04) | Free, forward |
| R1 | Retail positioning, contrarian (2025 study: against retail flow over 4-20 hours) | OANDA order / position books: needs a free OANDA practice token as the `OANDA_TOKEN` secret | Free, needs the user |
| X1 | Option-expiry pinning at the 10:00 New York cut | DTCC public option trades (collecting daily) | Free, forward |
| K1 | Pre-FOMC dollar drift as published (Karnaukh) | Next-month fed funds futures history (paid) | Quote first |
| F1 | Real futures volume / CLS flow | CME DataMine / CLS (paid) | Quote first |

## D. Forward watch list (no new test; judged on live data only)

USD/JPY H4 high-confidence model (mostly a 2022 / 2024 trend ride); FS1 66 settings; the 60/40 rebalancing
calendar (Sharpe 0.6 after publication); euro COT speculative pressure (1 of 52); ECB pre-decision drift (19
meetings); USD/JPY CPI surprise (11 releases); gold breakout short; skipping the 25th and month-end (FW2); the
forward-v1 / v2 win models; the post-fix USD/JPY / EUR/JPY reversal as a second Gotobi leg (short 09:55 -> 10:20
Tokyo; +1.6 / +0.8 bp on 2019-2025, absent in the other crosses and in 2003-2005).

## E. Dropped (tested on unseen data and failed; kept here so they are not re-run)

Q batch: volatility sizing of the Gotobi trade (VS1), variance-ratio switching (VS2), gold volatility premium as
a direction signal (VS3), the cost-to-volatility gate on E2 / E3 / K3 (VS4), volatility jumps (VS5), slow trend on
prop CFDs (T1), euro COT before 2004 (C5E). BS1 rate-expectation carry: positive but not significant (forward watch).


US-euro 2-year gap and its extension to GBP, CAD, CHF (P1); gold before the Chinese New Year 1980-2000 (P2); GLD
holdings flows (P5); futures and ETF volume (V1, V10); CFTC positioning changes (V3); SNB deposits (V4); news tone
(V5); fed funds futures moves (V7).

## Low priority (tested in a close form, or weak evidence)

BSIC rate-expectation carry exactly as written (its authors find no edge after 2010; close to V6 / P1); Australia
and New Zealand 2-year gaps for P1 (parser / access); euro COT C5 before 2004 (too little data).
