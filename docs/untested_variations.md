# What is still untested in FX and metals (2026-11-03)

This is an inventory of every FX and metals effect the project has met that showed any signal, before or after
costs. For each one it lists what was tested and which variations were not. A second list covers ideas that
were never tested at all.

Sources: `docs/untested_batch.md` (sections 4-9.19), `docs/flows_lab.md`, `docs/lit2_lab.md`,
`docs/literature_check.md` (rounds 3-7), `docs/breakeven.md`, `docs/gotobi_*`. Tests already in the edges batch
(`docs/edges_batch_preregistration.md`, E1-E11) are marked "batch".

## 1. Effects with a signal: what was tested, what was not

| Effect | Best result so far | Variations tested | Not tested |
|---|---|---|---|
| **Gotobi overnight, USD/JPY + EUR/JPY** | +4.6 bp net, t 3.9 (2007-2025); +2.0 bp in 2026 | 72 entry/exit/stop settings; 7 yen pairs; about 100 stops; targets, scale-in, dip and re-entry; filters v1/v2; 64 indicators; losing-night features; volatility sizing; limit entry at the bid; subsets (quarter-end, 25th/month-end, day before); sizing by filter probability (E1) | **Leverage that drops after a drawdown** (prop survival); USD/SGD, USD/CNH, USD/HKD as Asian importer-dollar carriers (in the panel run, pending); New York-leg hedges (pending); cheaper limit entry (E2, pending) |
| Gotobi after the fix (short 09:55-10:30 Tokyo) | Gross +1.46 bp (t 4.1); net -0.42 overall, +1.33 (t 2.9) in 2019-2025 | Unconditional, USD/JPY and EUR/JPY | **Only after a large rise into the fix** (flow exhaustion) |
| 25th and month-end Gotobi days | +1.13 bp (t 0.7), against +4.45 on all days | Tested as a subset | Dropping them (5th/10th/15th/20th only): decided after seeing the data, so it can only be judged on forward nights |
| Dollar on FOMC days (P1/P2, two days before) | +10 to +15 bp, t 1.6-2.0; faded after 2010 | Unconditional | **In the direction of the expected decision** (Karnaukh), now possible for 2017-2023 using the consensus forecasts in the calendar dataset (about 20 decisions with a change) |
| London month-end fix, 24 hours | +3.8 bp, t 1.5; faded after 2010 | Equity-hedging signal, 8-hour and 24-hour windows | Quarter-ends and year-ends only (the largest rebalancing); weak prior |
| Fix "W" (Tokyo / ECB / London) | Gross +0.6 to +0.9 bp, t 3-4; net negative | Each fix separately | Nothing that beats about 2 bp of costs |
| End-of-day momentum, gold (13:00-13:30 NY) and FX (14:30-15:00 NY) | Gross +1.27 bp (t 3.4) and +0.27 bp (t 2.9); net negative | Every day traded | **Only on days with a large move earlier in the session** (the published effect is concentrated there) |
| Gold London breakout | Gross +3.8 bp (t 2.1); net -0.4 bp; short side +1.4 bp | One rule, both sides | Short-only: chosen after seeing it, so forward only |
| Gold in Asian hours, gold AM fix, silver fix | Gross +1.0 to +3.6 bp; costs 4-26 bp | Published forms | Nothing a retail spread allows |
| **Gold before the Chinese New Year** | +1.5% a year, t 2.0, 16/25 years | One 15-day window | **Indian festival demand (Dhanteras/Diwali, Akshaya Tritiya)**; silver before the Chinese New Year |
| Gold turn of the month | +0.23% a window, t 1.6 | Last day + first three days | **Gold calendar trades added to Gotobi in a prop account** (more trading days, uncorrelated) |
| Gold variance premium (GVZ) | +0.74 vol points a month, t 1.5 | Variance swap proxy | Needs gold options; not tradable at forex prop firms |
| CFTC positioning | One market of 52 (euro speculative pressure) | Legacy report, six forms | Traders in Financial Futures: leveraged funds (batch E11a) |
| Round numbers (Osler) | Gross edge, net negative | Market entries | **Resting limit orders at the level** (earn the spread instead of paying it) |
| Night scalping (Bollinger fade 21:00-01:00 UTC) | Gross +1.6-2.0 bp, t 8-12; the rollover-hour spread costs 5-9 bp | Window including the rollover | **The same fade after the rollover hour only (23:00-02:00 UTC)** |
| Mean reversion of random entries | Gross Sharpe about 1.0; costs 3-8 times the edge | Every timeframe and management | Nothing at retail cost |
| Carry (G10) | About zero | Plain, VIX-filtered, momentum- and volatility-gated, dollar carry | **Emerging-market carry** (MXN, ZAR, TRY, CNH, SGD; panel data now includes these pairs) |
| Toshin start-of-month yen selling | Gross +1.35 bp, t 1.2 | One window | Weak; not worth a test |

## 2. Never tested

| Idea | Mechanism | Data | Status |
|---|---|---|---|
| US Treasury auctions | Yields rise into an auction and ease after (Lou, Yan & Zhang) | Treasury auction dates, M1 | batch E7 |
| Yuan fixing | The central bank sets the rate away from the market | CFETS fixing history, USD/CNH M1 | batch E4 |
| BoJ / ECB meeting drift | Positioning into announcements | BoJ pages, calendar dataset | batch E8 (no BoE: its pages did not load) |
| Economic surprises (payrolls, CPI, retail sales) | Reaction to the forecast miss | Calendar dataset, 2017-2023 only | batch E10 (ISM missing from the dataset) |
| Japan's weekly portfolio flows, trade balance | The size of the yen flows behind Gotobi | MoF weekly CSV, FRED | batch E3 |
| Gold options expiry, MSCI review days | Hedging and index flows | Calendar rules, M1 | batch E6, E9 |
| Google search interest | Retail attention | Google Trends monthly | batch E11b |
| Japanese government bond auctions | Same mechanism as Treasury auctions, in yen | MoF auction calendar (not yet downloaded) | new |
| Leverage that falls after a drawdown (prop) | Fewer accounts lost on a bad run | Gotobi nights | new |
| Platinum, palladium | No specific mechanism found | Not downloaded; few prop firms list them | skipped |

## 3. Not testable with free data

- Bank order flow (CLS, interbank).
- FX and gold option flows: open interest by strike, risk reversals. DTCC only gives recent months.
- Retail positioning (needs an OANDA token).
- The Shanghai gold premium.
- Fed funds futures, beyond the consensus forecasts above.
- News text.
- Consensus forecasts before 2017.
