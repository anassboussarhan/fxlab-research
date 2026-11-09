# Round 15: every strategy at every slow timeframe, setting and direction; the book systems; every feature; free versions of paid data (pre-registration, 2026-11-09)

Written before any data of this round is read. The user asked to:
- check again every possible strategy (discussions, books, proprietary methods, EAs, features);
- find what was missed;
- get the paid data for free.

Development data end on 2025-12-31. 2026 is read only in part E, a pre-registered use of the holdout.

## Why Double Seven was missed (the coverage audit)

Double Seven was in the catalog from round 4. It failed every test it was given because those tests were too
narrow:

| Dimension | Covered before this round | Never covered |
|---|---|---|
| Timeframe | M5-D1 on 2007-2018 (the cloud sweep); H1 / H4 / D1 on 2019-2025 (round 12, 7 instruments); D1 / W1 on 2019-2025 for D1-allowed rules (round 13, default settings) | W1 for rules outside the D1 list; any monthly bars |
| Settings | full grids on 2007-2018, walk-forward | full grids judged on 2019-2025 |
| Direction | as published, plus the mirrored short where a rule has one | the reversed (contrarian) version of every rule at D1 / W1 / MN |
| Rules | catalog families except the excluded ones | esoteric rules after round 4; bar rules filed as intraday only (Crabel NR4 / NR7, Williams volatility breakout, volatility breakout, weekday); the round-12 classic state systems at W1 / MN |
| Features | 269 features at M5-D1 on 2007-2018 | every feature at W1 / MN, and at D1 judged on 2019-2025 |
| Named systems | about 40 in the catalog | the book systems listed in part B |
| Data | free price, positioning, rates and options data in hand | the items in part D |

## Common setting

- 33 instruments: the 28 crosses of eight currencies, gold, silver, S&P 500, Nasdaq-100, Nikkei. Data are the M1
  bid / ask store, 2007-2025.
- Bars: D1 at 17:00 New York; W1 = Monday-Friday weeks; MN = calendar months by trading date.
- Fills, costs and swaps as in round 13 (`docs/long_preregistration.md`): market orders at 08:00 London after
  the signal bar, stops on the M1 path.
- Sizing is equal risk per instrument. Results are monthly portfolio returns; fit 2008-2018, check 2019-2025;
  Newey-West t (3 lags).
- **Primary portfolio:** the 28 FX crosses, where a long bias earns no drift. The 5 non-FX instruments are
  reported separately, with the excess over holding them long.

## A. Every rule, every slow timeframe, every setting, both directions

**Units:** (rule, timeframe) for timeframe in {D1, W1, MN}. The rules:
- every single-instrument catalog strategy, except:
  - machine-learning rules (tested in the ML campaigns);
  - grid / martingale / pyramid sizing schemes (they are money management, not signals);
  - time-of-day rules (session, opening range, fixes, Gotobi, intraday seasonality, news straddle, silver
    bullet, intraday pivots);
- plus four bar rules filed as intraday only: Crabel NR, Williams volatility breakout, volatility breakout, and
  weekday (D1 only);
- the esoteric rules;
- the round-12 state systems: Aberration, King Keltner, Thermostat, Bollinger Bandit, Elder;
- the book systems of part B.

**Configurations per unit:** the rule's full parameter grid x {as published, reversed}. Exits are the rule's own:
- signal exits for state rules;
- for trade rules, their own stop / target / hold, else the catalog default of a 1.5 or 2 ATR stop, a 2R target
  and 48 bars.

**Selection:** per unit, the configuration with the highest 2008-2018 FX t (at least 30 fit trades). It is judged
once on 2019-2025.

**Two verdict tracks:**
- **Counts (strict):** fit mean > 0; check mean > 0 with t >= 1.645; Benjamini-Hochberg at q 0.10 across all
  units of parts A-C.
- **Regime candidates:** check mean > 0 with t >= 1.645 and BH pass, whatever the fit period. This is the
  Double Seven pattern: a rule that works only after 2018. Regime candidates go to the part-E check on 2026 and
  are never called edges on development data alone.

**Placebos:** per timeframe, 20 random-signal rules:
- random long / short / flat states with persistence in {0.8, 0.9, 0.95, 0.98}, both directions;
- 8 configurations, against a median catalog grid of about 12;
- chosen the same way.
- If the placebos pass either track as often as the rules do, that track is not credited.

## B. Book and proprietary systems not yet in the catalog (`fxlab/strategies/books.py`)

| Author / source | Systems |
|---|---|
| Larry Williams | Oops! gap reversal; Smash Day reversal; %R with trend filter |
| Linda Raschke | Turtle Soup; Anti |
| Larry Connors | ConnorsRSI; 3-day high / low; R3; %b |
| Bill Williams | Awesome Oscillator zero cross; fractal breakout above the Alligator |
| Welles Wilder | DMI system (+DI / -DI with ADX > 25); Volatility System (3 x ATR(7) stop and reverse) |
| Kathy Lien (FX books) | Double Bollinger Bands; Perfect Order (10 / 20 / 50 / 100 / 200 averages stacked, ADX > 20); double inside-day breakout; the Fader; ADX range channel |
| Ichimoku | cloud (kumo) breakout |
| Charting | Heikin-Ashi trend; point and figure double-top / double-bottom (3-box reversal, ATR box); double top / bottom with neckline break; head and shoulders |
| Others | Gann HiLo activator; Elder Impulse system; prior-week high / low breakout |

Each is coded from its published rule with a small grid. Each is checked for causality (prefix invariance) before
any data is read.

## C. Every feature as a rule

- **Columns:** every output column of every single-instrument, non-learned catalog feature, at default
  parameters, on D1, W1 and MN.
- **Rule:** long when the feature is above its 2008-2018 upper tercile (cut points per instrument, from fit data
  only); short below the lower tercile; flat between. Also the reversed rule.
- **Trading:** positions are held to the next bar close, sized 0.20 / volatility, and traded through the
  continuous engine of round 13 with costs and swap.
- **Selection:** the direction is picked on 2008-2018 FX t; the same two tracks and BH family as part A.
- **Placebos:** 30 noise columns per timeframe (random walks and white noise).

## D. Free versions of the paid data

Each dataset gets its own addendum, written after the download succeeds and before its values are read:
- **D1. Forex Factory calendar archive** (HuggingFace, MIT licence; actuals and consensus forecasts, 2007-2025).
  It allows real economic surprises: (i) drift after a surprise; (ii) a weekly surprise index per currency, as
  a W1 direction signal. That index is a free version of the paid Citi Economic Surprise Index.
- **D2. Dukascopy instruments not yet in the store:**
  - commodities: Brent, WTI, natural gas, copper;
  - government bonds: Bund, US T-bond, Gilt;
  - more indices;
  - Scandinavian and emerging-market FX.
  They get the round-13 trend systems and the part-A sweep: trend following is known to work best in
  commodities and bonds, which were never in the data.
- **D3. Tokyo Financial Exchange "Click 365"** daily volume and open interest by currency, a public retail
  positioning series (the free side of the paid Gaitame / TraditionData flow).
- **D4. CBOE strategy benchmark indices** (BXM, BXMD, PUT, CNDR, BFLY, CLL, CMBO, WPUT, PPUT), for the
  option-selling test O5 that Yahoo could not supply.
- **D5. Two-year government yields** from the central banks and treasuries, for rate-differential momentum.
- **D6. OANDA order and position books.** These need the user's own API token, as before.
- **D7. Dukascopy SWFX sentiment history**, if a public history exists.
- **D8. Google Trends** search interest in currencies and gold.

Only legitimate free access is used: public files and APIs, free tiers, and the user's own accounts. Nothing
behind a paywall or a login that is not the user's, and nothing that breaks a site's terms. A dataset that
cannot be had that way is reported as not available.

## E. 2026 check (fourth use of the holdout)

- **Who is checked:** every unit that counts (strict) or is a regime candidate in A-C, plus any pre-registered
  D test that passes on development data.
- **What is checked:** the frozen configuration on January-October 2026, on Dukascopy 2026 prices for the same
  instruments.
- **Verdicts:** as in the holdout protocol, on the 2026 trades or months.
  - Supported: 2026 mean > 0 with t >= 1.645 (clustered by month).
  - Contradicted: below the 2019-2025 mean with t <= -1.645.
  - Otherwise inconclusive.
- **Use:** only "supported" moves a rule to the forward test.

## Addendum 1 (2026-11-09, before any data of this round is read)

- **Three of the four "intraday-filed bar rules" are left out:** Crabel NR, the Williams volatility breakout and
  the volatility breakout. On reading their code, each trades a breakout from the day's own open inside the same
  day. A daily bar cannot express that: it needs intraday bars, where these rules were tested (the M5-H4 sweep and
  round 12). The weekday rule stays, at D1 only.
- **The book module has 26 systems.** The Raschke Anti uses %K(7) and %D(10 or 14). All 26 passed the causality
  test (`tests/test_books.py`) before any market data was read.

## Addendum 2 (2026-11-09, after the part A-C sweep and before any part-E computation): the weekly dip-buying family on 2026

**Why.**
- The sweep found no unit that survives Benjamini-Hochberg. So, by the rule above, part E has nothing to check.
- But the published-direction weekly versions of the Connors dip-buying rules lost in 2008-2018 and won in
  2019-2025, nearly all of them. Double Seven passed its own 2026 check.
- A single family test answers whether weekly dip-buying is a family effect or one lucky rule. Picking the
  best-looking rules from the sweep instead would be a search.

**The family, fixed now.** Every close-only Connors rule in the catalog, at its published default and long only,
on W1 bars:

| Rule | Published default |
|---|---|
| Double Seven | n = 7 |
| Cumulative RSI | k = 2, lo = 35, x = 65 |
| RSI(2) | lo = 10 |
| R3 | - |
| %b | - |
| ConnorsRSI | lo = 10 |
| Three down closes | the catalog default |

**Data and rules.**
- The same data and fills as the 2026 Double Seven check: ECB fixings for the 28 crosses, fills at the next
  fixing after the signal week, 3 bp a round trip, no swap.
- Results in R per trade (2 x the close-based weekly N).
- One book: every trade of every rule, equal weight.

**Verdicts on 2026** (entries from 2026-01-01, exits by 2026-10-31), as in round 14 part D:
- **Supported:** mean R > 0 with t >= 1.645, clustered by exit month.
- **Contradicted:** below the 2019-2025 mean with t <= -1.645.
- **Otherwise inconclusive.**
- The 2019-2025 comparator must be positive. Double Seven is reported inside and outside the family; the family
  result without it is the new information.

## Addendum 3 (2026-11-09, after the availability probe and before any value of these datasets is read): the part-D tests

**What the probe found.**
- Available:
  - the Forex Factory calendar archive (83,427 releases, 2007-01 to 2025-04, actual / forecast / previous; MIT);
  - all CBOE strategy indices;
  - two-year yields for USD, EUR, CAD, AUD, CHF and JPY;
  - Dukascopy commodities and indices from about 2013, and bonds from 2019.
- Not available: the Bank of England and RBNZ files, Dukascopy sentiment (403), Google Trends (429), and
  Scandinavian / EM FX candles. TFX shows only a current file.
- D3 and D6-D8 are reported as data missing.

**D1. Economic surprises (Forex Factory archive).**
- **Releases used:** High-impact releases of the eight currencies where both actual and forecast parse as
  numbers. The units (%, K, M, B, T) are stripped.
- **Surprise z:** (actual - forecast) divided by the standard deviation of that event's earlier surprises (at
  least 8 earlier releases, expanding). It is multiplied by -1 when the event name contains "Unemployment",
  "Jobless" or "Claimant", where a higher number is bad news.
- **D1a, drift after the release** (|z| >= 1):
  - Buy the currency in the surprise direction against USD. For USD releases, trade EUR/USD, USD/JPY and
    GBP/USD in the USD direction, equal weight.
  - Enter at the first M1 bar 5 minutes after the release time, at the store's bid / ask.
  - Two rows: exit after 60 minutes, and after 1 day (the same time the next trading day).
- **D1b, weekly surprise index:**
  - For each currency, the sum over the last 13 weeks of capped z (+-3) of High- and Medium-impact releases.
  - Each Friday close, cross XY holds (I_X - I_Y) / the cross-sectional standard deviation of I, capped at +-2,
    times 0.10 / volatility, for one week. It runs on the 28 crosses through the round-14 continuous engine.
- **Periods:** fit 2008-2018, check 2019-2025 (the archive ends in April 2025).

**D2. New asset classes on Dukascopy (commodities, indices, bonds).**
- **Instruments:**
  - commodities: Brent, WTI, natural gas, copper;
  - indices: DAX, FTSE 100, CAC 40, Euro Stoxx 50, Hang Seng, SMI;
  - government bonds: Bund, US T-bond, Gilt.
  - Each is used from its first full year. The price scale is set by matching one date against Yahoo.
- **Systems:** the fixed round-13 systems L1-L11 and the long-only benchmark. They need no fitting, so they can be
  judged on 2019-2025 directly, with 2013-2018 reported.
- **A row counts** (commodity, index and bond books) if all of these hold:
  - 2019-2025 mean > 0 with t >= 1.645;
  - excess over long-only with t >= 1.645;
  - BH across the D rows.
- Swaps use the quote currency's rate plus a 2.5% mark-up, as for metals.

**D4. CBOE option-selling indices (the O5 comparison).**
- **Data:** monthly returns of BXM, BXMD, PUT, CNDR, BFLY, CLL, CMBO, WPUT, PPUT and CLLZ, regressed on the
  S&P 500 total return.
- **Counts** if alpha > 0 on 2008-2018 and alpha t >= 1.645 (Newey-West) on 2019-2025, with BH across the indices.
- Index returns carry no trading costs, so a pass is an upper bound on what a trader could keep.

**D5. Two-year rate-differential momentum** (USD, EUR, CAD, AUD, CHF, JPY; 15 crosses).
- Each Friday close, cross XY holds sign(change in (y2_X - y2_Y) over 4 weeks) x 0.10 / volatility, for one
  week. A second row uses 13 weeks.
- Fit 2008-2018, check 2019-2025; count rule as in part A, with BH across the D rows.

**The D family.** BH runs over the rows D1a (2), D1b, D2 (3 books x 11 systems), D4 (10) and D5 (2).
