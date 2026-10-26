# Options strategies on real quotes: many structures, many underlyings, optimised and tested out of sample

The question was the best options edge across many strategies and products, and whether it survives on the
best capital provider.
- Scripts: `scripts/options_strategy_lab.py` (engine, statistics, walk-forward), `scripts/options_synth.py`
  (long-history model), `scripts/options_strategy_prop.py` (provider rules), `scripts/options_strategy_report.py`
  (tables).
- Workflow `options-strategies`, final run 40244024631. Full outputs are on branch `options-strategies`; the
  tables are in `docs/options_strategies/`.
- Development data only: every quote after 2025-12-31 is dropped and every position must expire by then.

## 1. Data

**Source:** DoltHub `post-no-preference/options`, free and cloned on the GitHub runner (8.3 GB, about 3 minutes).
It has end-of-day bid, ask, implied volatility and greeks from February 2019.

**Coverage:**
- 108 underlyings were exported: SPY, DIA, XLF, XLE and the current S&P 100 stocks.
- QQQ, IWM, GLD, SLV, TLT and EEM are not in the database.
- 106 ran: BK had no Yahoo closes and BKNG failed.
- FDX, HON and CMCSA quotes do not match their closes on most dates. Over 500 entries each were skipped, so their
  results rest on few cycles.

**What the data look like, and what that forces:**
- **Snapshots:** weekly in 2019, about three a week (Mon / Wed / Fri) in 2020-2023, daily from 2024. Management
  rules are checked on those closes.
- **About three expiries per date,** roughly 2, 4 and 6-8 weeks out and often not the standard monthlies. Entries
  therefore use whichever listed expiry is nearest the target: 14 days (weekly entries), 30 and 45 days (monthly
  entries).
- **10-27 strikes per expiry,** and a different set on each snapshot.
  - Legs are matched to the nearest listed delta within 0.08.
  - On dates when a held contract is not listed, it is valued from that day's own smile (Black-76 on the
    put-call-parity forward, vol and half-spread interpolated in log-moneyness). That was 26% of all path points.
  - Accuracy, from hiding each listed quote and re-pricing it from the rest:
    - median error 0.074% of the underlying (median across underlyings; SPY 0.012%);
    - 90th percentile 0.35%;
    - FDX is the outlier at 12%.
- **Corporate actions:**
  - Positions are skipped if they span a day on which the traded and total-return prices diverge by more than 3%
    (splits, special dividends, adjusted spin-offs).
  - They are also skipped if put-call parity at the leg's own strike breaks on two or more listed dates; 104
    positions were removed this way.
  - Run 1 settled a position held through AVGO's 10-for-1 split against the post-split price, which produced a fake
    -71% cycle. That is why these checks exist.
- **Survivorship:** the stocks are today's S&P 100 members, a list that favours names that did well.

**Costs:**
- Options fill at mid +- 0.75 x half-spread (effective spreads about 75% of quoted, Muravyev & Pearson 2020). The
  sensitivities are 1.0 (at the quote) and 0.4 (patient limit orders).
- USD 0.65 per contract per leg; 1 bp each way on the underlying.
- Expiring options settle at intrinsic value. Early assignment is ignored; the underlying earns its dividends.

**Unit:** P&L per share of underlying divided by its price at entry, i.e. the return on the underlying notional.

## 2. What was tested

| Dimension | Values |
|---|---|
| Structures (19) | short puts 16 / 30 / 50 delta; put credit spreads 30/16 and 20/5; call credit spread 30/16; short strangle 16/16; short straddle; iron condors 16/5 and 30/16; iron butterfly with 16-delta wings; put ratio 1x40 / 2x20 delta; covered call 30 delta; buy-write at the money; collar 30/30; put calendar; long 30-delta put and long straddle (premium-buying controls); the underlying alone |
| Days to expiry | 14 (entered every week), 30 and 45 (entered every month) |
| Management (5) | hold to expiry; take profit at 50%; close at 21 days left; both; take profit at 50% with a stop at 2x the credit |
| Entry filters (5) | none; VIX below VIX3M (contango); the underlying's IV above its one-year median; IV above 21-day realised vol; no earnings before expiry (stocks) |
| Fills (3) | 0.75 (primary), 1.0, 0.4 |
| Products | 106 single underlyings, plus stock portfolios built from them: all stocks at equal weight (EW); the fifth of stocks with the largest IV minus realised vol (XSQ5) and the smallest (XSQ1); and their Goyal-Saretto long-short (short straddles on XSQ5, long straddles on XSQ1) |

That makes **118,216 combinations** at the primary fill.

**Statistics:**
- Per combination:
  - Newey-West t of the mean per cycle;
  - long-run Sharpe;
  - worst cycle and CVaR;
  - alpha and beta against the underlying's own return over the cycle;
  - year-by-year results.
- Benjamini-Hochberg across all combinations.
- An in-sample / out-of-sample split (2019-2022 vs 2023-2025).
- A **walk-forward**: each year from 2021, pick the best combination using only cycles that ended earlier (by
  Sharpe, or by t), then trade it that year.

## 3. Results

### 3.1 Optimising everything does not survive out of sample

The best in-sample combinations look spectacular. Out of sample, the same selection gives a fraction of that,
mostly not significant.

- **In sample:** the best had Sharpes of 7-20 with no losing cycle. Examples:
  - JPM short 16-delta put, 45 days, take profit 50%, IV > RV filter: 56 cycles, worst cycle +0.14%;
  - V, the same put without a filter: 81 cycles.
- **Benjamini-Hochberg** leaves 2,603 combinations significant on the mean and 1,025 on alpha.

Walk-forward, choosing on earlier years and trading 2021-2025:

| Family (selection criterion) | Out-of-sample months | Mean per month | t | Sharpe | Worst cycle |
|---|---|---|---|---|---|
| Everything (Sharpe) | 28 | +0.58% | 1.21 | 0.79 | -9.3% |
| Everything (t) | 37 | +0.26% | 0.39 | 0.22 | -20.2% |
| Defined risk only (either) | 29 | +0.04% | 0.10 | 0.06 | -9.3% |
| Index ETFs only (either) | 30 | +0.71% | 2.12 | 1.34 | -4.6% |
| Stock portfolios (Sharpe) | 57 | +0.52% | 1.43 | 0.66 | -6.8% |

Notes on the table:
- Months are months with a position. The mean is in units of underlying notional.
- **How much the picks overstated themselves:** their in-sample Sharpes were 5-22, e.g. AMD 20/5 put spreads
  (2021-2022, about 17-22), then AVGO short puts (2023-2025). Out of sample in 2023-2025, filters such as "IV > RV"
  left only 3-8 cycles a year.
- **The index-ETF family** picked:
  - an SPY covered call (2021);
  - a DIA short put (2022);
  - SPY short 16-delta puts (2023-2025).

  These are equity-exposed positions in mostly rising years.
- **Rank persistence:** across 23,613 combinations, the in-sample (2019-2022) and out-of-sample (2023-2025) Sharpes
  have a rank correlation of 0.60. The top in-sample decile averaged 0.58 out of sample, against -0.24 for all.
  What persists is the *kind* of structure (below), not the specific "best" one.

### 3.2 On SPY the option premium after costs is about zero: the returns are equity beta

SPY, held to expiry, no filter, 2019-2025. Alpha is per cycle against SPY's own return over the cycle.

| Structure | Days | Cycles | Mean / cycle | Sharpe | Worst cycle | Beta | Alpha (t) |
|---|---|---|---|---|---|---|---|
| Short put 16 delta | 45 | 66 | +0.39% | 0.67 | -15.1% | 0.19 | +0.02% (0.1) |
| Short put 30 delta | 45 | 66 | +0.58% | 0.65 | -19.2% | 0.37 | -0.12% (-0.5) |
| Put spread 30/16 | 45 | 66 | +0.18% | 0.43 | -4.1% | 0.17 | -0.16% (-1.8) |
| Put spread 20/5 | 45 | 66 | +0.33% | 0.75 | -7.7% | 0.16 | +0.01% (0.1) |
| Iron condor 16/5 | 30 | 76 | +0.19% | 0.64 | -5.2% | 0.06 | +0.11% (1.0) |
| Strangle 16/16 | 30 | 77 | +0.43% | 1.24 | -5.0% | 0.05 | +0.34% (2.6) |
| Covered call 30 delta | 30 | 78 | +1.17% | 0.95 | -28.7% | 0.85 | +0.08% (0.6) |
| Collar 30/30 | 30 | 77 | +0.93% | 1.63 | -3.9% | 0.30 | +0.54% (3.8) |
| Short straddle | 45 | 66 | -0.09% | -0.08 | -18.7% | 0.15 | -0.38% (-0.8) |

- Of about 27 SPY hold-to-expiry rows, only the 30-day strangle and the 30-day collar have alpha t > 2. Their 45-day
  versions do not (t 0.2 and 0.7). Two out of 27 is what chance gives.
- **The IV > RV filter on 45-day SPY short puts** gives a Sharpe of 8.9 (42 cycles, no losing cycle), but the same
  filter at 30 days gives 1.1 with alpha t 0.6. An effect that disappears between neighbouring maturities is a
  sample artefact.
- **Comparison with the variance-swap test** (`docs/options_retail.md`): short S&P variance at the VIX earned +1.26
  vol points a month in 2019-2025 (t 0.99), against +1.95 (t 5.2) over 1990-2018.
  - With the contango filter it earned +2.67 a month (t 6.2, 75 months), but listed options with spreads did not
    capture it: VIX < VIX3M was true for 78 of 82 (30-day) and 62 of 66 (45-day) SPY entries, so the filter barely
    changed the option results.
  - The variance swap trades no strikes and pays no bid/ask; listed SPY options at 0.75 of the half-spread give most
    of it back.

### 3.3 Single stocks: selling premium loses after costs; the IV-RV sort loses too

Equal-weight portfolio of all stocks (EW), held to expiry, no filter:

| Structure | Days | Mean / cycle | Sharpe | Alpha t |
|---|---|---|---|---|
| Iron condor 16/5 | 30 | -0.22% | -0.95 | -4.2 |
| Iron butterfly | 30 | -0.45% | -1.62 | -4.4 |
| Short straddle | 30 | -0.34% | -0.35 | -2.6 |
| Strangle 16/16 | 30 | -0.16% | -0.19 | -2.4 |
| Put spread 30/16 | 30 | -0.02% | -0.08 | -3.8 |
| Short put 30 delta | 45 | +0.27% | 0.25 | -2.8 |
| Covered call 30 delta | 45 | +1.10% | 0.66 | -2.3 |
| Goyal-Saretto long-short | 30 | -0.60% | -0.79 | -2.5 |
| Long 30-delta put | 30 | -0.32% | -0.36 | +2.2 |

- **Every short-premium structure on stocks has negative alpha:**
  - t -1.7 to -15;
  - the 14-day versions are worst, at -2 to -15;
  - restricting to the fifth of stocks with the richest volatility (XSQ5) does not help.
- **Why:** this matches the literature. The variance premium is large in indices and small in single stocks (Carr &
  Wu 2009: significant for 21 of 35 stocks, in level terms for only 3), and single-stock option spreads are wide.
- **The positive raw returns** of covered calls and short puts on stocks are the 2019-2025 equity market (betas
  0.3-0.8), not an options edge.

### 3.4 Management and filters

Change in Sharpe against holding to expiry, for the same combination:

| Rule | ETFs / portfolios: mean | Stocks: mean | Share of combinations that improved |
|---|---|---|---|
| Take profit 50% | -0.08 | -0.04 | 25-31% |
| Close at 21 days | -0.59 | -0.42 | 22-26% |
| Both | -0.61 | -0.44 | 20-25% |
| Take profit 50% + 2x stop | -0.27 | -0.17 | 20-29% |

The popular rules (50% take-profit, 21-day exit, 2x stop) lower risk-adjusted returns on average. Holding to expiry
is the best default here.

Change in Sharpe from each entry filter against no filter:

| Filter | Mean | Share that improved |
|---|---|---|
| IV > RV | +0.06 | 51% |
| Contango | +0.03 | 55% |
| IV above its median | +0.03 | 47% |
| No earnings before expiry | -0.00 | 45% |

**Fills** (median mean P&L per cycle across products, at fills 0.4 / 0.75 / 1.0, in bp of notional):
- short put 30 delta: 20 / 16 / 12;
- iron condor 16/5: -20 / -30 / -39;
- straddle: -15 / -30 / -37.

Execution matters, but no fill assumption turns the delta-neutral structures positive.

### 3.5 Long history (a model, not quotes), 1994-2018

The same structures were priced from the VIX (SPY) and VXD (DIA), with the smile and spreads calibrated on the real
2019-2025 chains.
- **Validation on 2019-2025:**
  - model and real cycle P&L have a median correlation of 0.76 (13,585 pairs; above 0.9 for most 30-day SPY
    structures);
  - real minus model averages +0.04% per cycle;
  - the means have the same sign in 73% of combinations.
- **SPY, held to expiry, 1994-2018 (model):**

  | Structure | Days | Sharpe | Alpha t | 2008 |
  |---|---|---|---|---|
  | Short put 16 delta | 30 | 0.72 | 2.3 | |
  | Strangle 16/16 | 30 | 0.57 | 2.2 | |
  | Short put 30 delta | 45 | 0.64 | | -29% |
  | Covered call 30 delta | 45 | 0.65 | | -58% |
  | Iron condor 16/5 | 45 | 0.13 | | |
  | Put spread 30/16 | 45 | 0.33 | -1.7 | |

- **Reading:** over a long history, selling index puts and strangles carries a small positive premium beyond beta,
  in line with the variance-premium literature. It is small after spreads, and 2008-type years take years of it.

### 3.6 On the capital providers

The walk-forward strategies themselves (what "optimise everything" would really have traded) were run on the
published rules from `docs/options_capital_providers.md`:
- 29 monthly starts in 2021-2023;
- daily marks from the smile valuation;
- size set by the daily volatility, capped at 4x the account in notional.

| Strategy | Options Funding Growth (monthly fee): mean net per account | Imperial Elite (USD 357 once): mean / median net | Imperial pass rate |
|---|---|---|---|
| Index ETFs walk-forward | -$1,183 to -$7,724 | -$357 to +$464 / -$357 | 0-48% |
| Everything walk-forward (Sharpe) | -$985 to -$3,079 | +$55 to +$729 / -$357 | 38-66% |
| Defined-risk walk-forward | -$1,342 to -$3,690 | -$315 to -$357 / -$357 | 0-38% |

- **On the monthly-fee provider** every walk-forward strategy loses at every size.
- **On the one-time-fee provider** the mean is near break-even to slightly positive, but the median account just
  loses the fee.
- **The in-sample best combinations** pass 100% of the time on these rules (`prop_in_sample_best.csv`). That is
  what selection on the same data shows, and section 3.1 shows it does not carry forward.

## 4. Conclusions

1. **No options edge survives costs and out-of-sample selection on 2019-2025 real quotes,** beyond what equity
   exposure pays.
   - Optimising 118,216 combinations finds Sharpe 7-20 in sample. Picking that way, year by year, gave out-of-sample
     Sharpes of 0.06-0.79 (not significant).
   - The only significant family was index ETFs (1.34, t 2.1), whose picks were equity-exposed short puts and
     covered calls in rising years.
2. **Index (SPY) premium is the only place with a premium at all.**
   - Real quotes 2019-2025: SPY alphas are about zero.
   - The 1994-2018 model and the variance-swap tests show a small, crash-exposed positive premium.
   - Practical form: SPY short puts or put spreads at 16-30 delta, 30-45 days, held to expiry, small size.
3. **Single-stock premium selling loses after costs,** including the IV-minus-RV sort and the Goyal-Saretto
   long-short.
4. **The usual rules of thumb do not help:**
   - taking profit at 50% and exiting at 21 days lower Sharpe;
   - filters add at most a few hundredths;
   - buying options (long puts, long straddles) loses before considering their hedge value.
5. **On a prop provider,** the edge that survives is too small to beat a monthly fee and roughly pays a one-time
   fee. The best FX candidate remains Gotobi, which passed its 2019-2025 out-of-sample test (`docs/gotobi_oos.md`).

**Limitations:**
- Seven years of quotes, with sparse snapshots and strikes. Unlisted contracts are valued from the day's smile
  (median error 0.07% of the underlying). Management decisions are taken on snapshot closes only.
- Strike-by-delta matching is coarse (within 0.08 delta).
- The stock list has survivorship bias.
- Early assignment and pin risk are ignored.
- The 1994-2018 layer is a model calibrated on 2019-2025.
- The provider rules are from public pages and change often.
