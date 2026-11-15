# Round 22 pre-registration: every parameter optimised, walk-forward (2026-11-15)

Written before any round-22 data was read. The scope is `docs/round22_audit.md`: every batch that used fixed,
averaged or menu-only parameters, plus the untested ideas that free data allow.

**The user's rule:** "no hardcoded values, always optimise". Every parameter of every rule below is chosen by
the data. Wide ranges are declared in advance, and the choices are made only on data before the year traded.

## The walk-forward protocol (all parts)

**Configurations:**
- Each family declares a parameter space.
- Exhaustive grids are used where cheap (T, L).
- Elsewhere a scrambled Sobol sample of the space: 128 configurations per instrument, or 256 where cheap.
- Every configuration's net P&L series (daily, weekly or monthly, by family) is computed once.

**Selection:**
- At the start of each calendar year Y, every configuration is scored on all earlier data (expanding window).
- The score is the t-statistic of its net returns.
- The 5 best are traded in year Y with equal weight ("top-5").
- A configuration needs at least 30 active periods in training to be eligible.
- The top-1 choice is also reported (secondary).

**Out-of-sample stream:**
- The concatenation of the traded years. It starts after at least 3 years of training, and never before 2010
  for FX / metals (the data start in 2007).
- Every number reported for a test is from this stream only.

**Test unit:** a family × instrument stream, plus each family pooled across its instruments (equal weight per
period).

**Counting rule.** A stream counts if:
- its out-of-sample mean before 2019 is > 0 (where the stream has years before 2019);
- its 2019-2025 out-of-sample mean is > 0 with t ≥ 1.645;
- it passes Benjamini-Hochberg at q = 0.10 over all round-22 streams.

**2026:** only for a stream that counts. One check with the parameters selected on data up to 2025, reported as
supported / inconclusive / contradicted as before.

**Costs:** as in the earlier rounds of each market:
- FX / CFDs: bid / ask plus slippage and commission;
- index futures: ES 0.30 / NQ 0.375 points per side;
- stocks: 5 bp per side plus 6% a year long financing for CFD positions;
- crypto: 10 bp per side on turnover;
- options: mid ± 0.75 × half-spread plus USD 0.65 a contract.

## Part L. The strategy libraries, re-selected every year

- **Inputs:** the stored per-configuration daily P&L of:
  - the daily engine (`sweep15 --daily`) and intraday engine (`intraday17`), for the 33 instruments of round 17
    and the 15 of round 19;
  - rules only, no feature-tercile units.
- **Families:** each rule × timeframe × direction is a family whose grid is its configurations.
- **ALL:** one more family per instrument, the whole library (every real configuration) as one grid.
- **Placebos:** the engines' placebo units get the same walk-forward as a null.

## Part T. Every time window (flows, fixes, sessions, calendars)

- **Instruments:** every M1 instrument in the store (33 + the 13 D2 CFDs + BTC / ETH perpetuals).
- **Grid:**
  - entry hour 0-23 UTC;
  - holding 1, 2, 3, 4, 6, 8, 12, 16 or 24 hours;
  - sign: long, short, or fade / follow the previous 1 h / 4 h move;
  - day filter: all days, each weekday, Gotobi days, the last and the first business day of the month, FOMC days
    and US payroll Fridays.
  - That is 24 × 9 × 6 × 11 = 14,256 configurations per instrument.
- **Fills:** entry and exit at the hour's first-minute mid, with the bid / ask cost of that minute.

## Part I. ICT / SMC and level rules, parameterised

- **Instruments:**
  - the 7 USD majors (EUR/USD, GBP/USD, USD/JPY, AUD/USD, NZD/USD, USD/CAD, USD/CHF);
  - XAU/USD and XAG/USD;
  - US500 and NAS100 (futures costs).
- **Timeframes:** M5, M15 and H1 (each a parameter).

| Family | Parameters (ranges) |
|---|---|
| I-LVL level sweep / break (sweeps, Turtle Soup, Asia sweep, prior-day / week stop runs, session breakouts) | level: prior day, prior week, Asian session, opening range (15-120 min), N-bar fractal (N 2-10); trigger wick or close; action fade or follow; entry at once or limit retest (0-1 ATR); stop 0.5-3 ATR; target 0.5-5 R; window start / end (hour); max hold 2-96 bars |
| I-FVG fair value gaps (Silver Bullet, IFVG) | min gap 0-1.5 ATR; direction with / against; entry near edge / middle / far edge; stop beyond candle 1 (+0-1 ATR); target 1-5 R; time window (any, London, New York AM, 10-11 New York); expiry 3-48 bars |
| I-OB order blocks | displacement 1-4 ATR; zone body / wick; entry at zone edge or middle; stop beyond zone + 0-1 ATR; target 1-5 R; expiry 5-100 bars |
| I-OTE | displacement 1.5-4 ATR; leg lookback 3-30 bars; retracement 0.5-0.85; stop at leg start + 0-1 ATR; target 0-1 extension; expiry 3-48 bars |
| I-BRK breakers | fractal N 2-8; BOS window 6-96 bars; entry at breaker edge or middle; stop 0-2 ATR beyond; target 1-5 R |
| I-CBDR | range start 12-16 h and end 18-21 h New York; max range 0.2-1 × ADR; entry 1-4 × range; stop 2-6 × range; target mid / opposite edge / 1 × range; flat at 08-14 h New York |
| I-SMT | pair (EUR / GBP; US500 / NAS100); lookback 6-72 bars; hold 3-48 bars; fade the leader or follow the laggard |
| I-SES Judas / midnight open / power of three (indices; FX in London) | session open 02:00 / 09:30; probe window 15-90 min; reference midnight or session open; action fade / follow; stop at probe extreme or 1-3 ATR; exit 11:00-16:00 |

## Part X. Index rules (NAS100 and US500 at futures costs; DAX, FTSE, CAC, Euro Stoxx, SMI, Hang Seng at CFD costs)

| Family | Parameters |
|---|---|
| X-NB noise-area breakout | band multiplier 0.5-2.5; lookback 5-30 days; decision step 5 / 15 / 30 / 60 min; first decision 30-120 min after the open; trailing stop VWAP / band / none; gap adjustment on / off |
| X-ORB opening range | range 5-60 min; direction from the candle / breakout of either side; stop opposite side / 0.5-2 ATR; target 1-10 R or the close; exit time 12:00-16:00 |
| X-VWAP trend | decision step 1-60 min; start 0-120 min after the open; band 0-0.3% around VWAP |

## Part D. Dip-buying and the D7 family, every parameter

- **Setting:** ECB weekly fixings, 28 crosses (as `family_d7.py`).
- **Parameters:**
  - N-period low 3-20;
  - trend filter SMA 50-300 or none;
  - exit N-period high 2-20, or RSI / time exits;
  - RSI(2) thresholds 2-30;
  - carry filter on / off;
  - time stop 2-26 weeks.
- **Also:** the same on daily bars of the 33 instruments.

## Part E. Scheduled events

| Family | Parameters |
|---|---|
| E-EIA crude (WTI, Brent) | entry 1-30 min after the release; exit 15 min-16:00; signal: seasonal surprise sign, follow / fade the first 1-15 min move; surprise threshold 0-2 sd |
| E-NG natural-gas storage (gas CFD; Thursday 10:30 New York, holiday-shifted) | the same, with a 5-year same-week surprise |
| E-FUND crypto funding times (BTC, ETH) | window before 15-240 min; window after 15-240 min; funding threshold 0-0.05%; side |

## Part M. Monthly currency signals (8 currencies, 1976-2025)

- **Signals:** carry, real carry, curve slope, momentum (lookback 1-12 months, skip 0-1), value (5-year real
  reversal, lookback 3-7 years), and their equal-weight composite.
- **Book parameters:**
  - number long / short 1-3;
  - rebalance every 1-3 months;
  - lag 1-2 months.

## Part B. Bonds across countries (8 markets, OECD 10-year yields, 1976-2025)

- **Return proxy:** yield / 12 − 7 × the change in yield.
- **Signals:**
  - momentum (1-12 months);
  - carry (10-year minus 3-month);
  - value (10-year minus inflation);
  - their composite.
- **Book:** n long / short 1-3; rebalance 1-3 months.

## Part C. Commodities

| Family | Data | Parameters |
|---|---|---|
| C1 spreads and ratios: Brent-WTI, gold / silver, copper / gold | CFDs, D1 and H4 | z lookback 10-250 bars; entry 1-3 z; exit 0-1 z; stop 3-5 z; max hold |
| C2 speculator positioning (managed money, CFTC) | weekly; crude, gas, gold, silver, copper, corn, wheat, soybeans, sugar, coffee, cocoa, cotton → their ETFs | percentile lookback 52-260 weeks; extreme 5-30%; contrarian / trend; hold 1-12 weeks |
| C3 cross-commodity momentum, reversal, seasonality | ETFs USO, BNO, UNG, UGA, GLD, SLV, PPLT, CPER, DBB, DBA, CORN, WEAT, SOYB, CANE, JO, NIB, BAL, DBC, ... | lookback 1-12 months; skip 0-1; n long 1-5 (long-only and long-short); seasonality: same-month mean over 3-10 past years |

## Part S. Stocks (point-in-time S&P 500, 2005-2025)

| Family | Parameters |
|---|---|
| S-ANOM momentum, reversal, 52-week high, low volatility, overnight momentum | lookback 5-252 days; skip 0-21; quantile 5-30%; holding 1-6 months; long-only vs the equal-weight universe |
| S-PAIRS | formation 6-24 months; entry 1.5-3 sd; exit 0-1 sd; pairs 5-50; trading 3-12 months |
| S-SWING breakout / episodic pivot / trend template | breakout length 5-60; tightness 5-20%; stop 3-12% or day low; trailing SMA 5-50; gap 3-15%; max hold 20-120 days |
| S-DEL deletion rebound | holding 5-120 days |

## Part K. Crypto

| Family | Parameters |
|---|---|
| K-XS cross-section (Binance spot, all USDT coins incl. delisted) | lookback 3-90 days; skip 0-7; quantile 10-30%; universe 20-100 by volume; rebalance 1-4 weeks; long-short / long-only |
| K-CC cash-and-carry (BTC, ETH: long spot, short perpetual) | enter when the trailing funding (1-30 days) > 0-0.03% per 8 h; exit below 0-0.02%; cost 10 bp per side per leg |

## Part V. Volatility selling with optimised filters

| Family | Parameters |
|---|---|
| V-SPX (variance at the VIX, monthly, 1990-2025) | VIX / VIX3M below 0.85-1.05; VIX − realised volatility above −5 to +10; VIX level band |
| V-SVXY | the same filters on the daily decision |
| V-BTC (DVOL) | DVOL − realised volatility above −10 to +20 |

## Part P. Options on real quotes (DoltHub, SPY, DIA, XLF, XLE and the 20 most liquid stocks, 2019-2025)

- **Structures:** short put, short strangle, iron condor, put spread, covered call, delta-hedged short straddle /
  strangle.
- **Parameters:**
  - delta 0.05-0.50;
  - wing width 0.05-0.20 delta;
  - maturity 7-60 days (nearest listed);
  - take-profit 20-90% of credit or none;
  - stop 1-4 × credit or none;
  - exit 0-30 days before expiry;
  - for hedged structures, a hedge band 0.05-0.5 delta and daily / weekly hedging.
- **Calendar:** monthly entries. Training starts in 2019, so the out-of-sample years are 2021-2025.

## Addendum 1 (2026-11-15, before the data of parts D-P were read)

- **Compute limits:** fewer configurations where each costs minutes:
  - S-pairs: 64 configurations;
  - V-BTC: a 32-point grid on its single filter;
  - S-deletions: a 24-point grid of holding periods.
- **S-SWING is not run in this round:** each configuration re-simulates every stock, and the round-20 rules
  failed against their own base rates by 0.5-1.6% a trade.
- **D runs on the ECB-fixing book only** (28 crosses, daily or weekly as a parameter). The 33-instrument daily
  bars are covered by part L's daily engine.
- **P:** the database lists about three expiries per date, so the maturity parameter picks the nearest listed
  expiry. Entries follow the engine's calendar for that maturity (monthly from 21 days, weekly below).

## Notes

- **Frozen forward tests stay frozen:** Gotobi B0, Double Seven, the D7 family, the noise-area breakout. An
  optimised version that counts would become a separate forward track.
- **Selection risk:** selecting among thousands of configurations per instrument is honest only out of sample.
  That is why only the traded years are reported.
- **Not run:** OPEC, crop reports, commodity curve carry, quarterly crypto basis, sector rotation, net foreign
  assets (see the audit).

Not financial advice.

## Addendum 2 (2026-11-15): the 2026 check, as pre-registered

**What counted:** four streams met the counting rule on development data:
- I-FVG on NAS100;
- I-FVG pooled over the 11 instruments of part I;
- K cash-and-carry on BTC;
- K cash-and-carry on ETH.

**The check:**
- The pre-registered one-time 2026 check runs for these four, with the configurations the walk-forward selects
  on data up to 2025.
- This is the fifth use of the 2026 holdout.

**Data:**
- Dukascopy M1 candles 2026-01 to 2026-10 for the 11 instruments;
- Binance public daily klines and funding for 2026.

**Before running, a robustness check on development data only** (`scripts/r22_fvg_check.py`, branch
`misc-r22fvg`):
- I-FVG on NAS100 stays positive and meets the rule under CFD costs, limits that must trade through by one tick,
  one tick of stop slippage and twice the futures cost: 2019-2025 t 3.2 to 5.4.
- It is positive in every year except 2016.

The verdict rules are unchanged.
