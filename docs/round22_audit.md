# Round 22 audit: every test batch, fixed or optimised, and what was never tested (2026-11-15)

**How this was checked:**
- every document in `docs/` (119 files);
- every row of `test_ledger.md`;
- every section of `untested_batch.md` (sections 3-9.19);
- each batch's own statement of how its parameters were set ("as published", "fixed before the run", "nothing
  tuned", grids, walk-forward, TPE).

**Classes:**
- **O:** optimised walk-forward (parameters chosen on earlier data only, then traded on later data).
- **G:** a grid searched once on a training split, then checked.
- **F:** fixed, published or chosen parameters.
- **A:** averaged over a grid, no selection.

## 1. Every batch

| Batch | Where | Class | Round 22 |
|---|---|---|---|
| Strategy catalog, 97 families × 34 instruments × 10 timeframes | `conclusions.md` 8.1c | O (small grids) | covered by L (wider: yearly re-selection per instrument) |
| Joint optimisation of parameters, entry, exit (762,000) | 8.1c, 8.1h | O | done |
| Trade management, random entries, every parameter (TPE) | `random_entry_opt.md` | O | done |
| ML / RL / meta-labels / triple barrier / policy optimisation | `untested_batch.md` 8.2-8.7 | O | done |
| Cost-aware tuning of gross-positive rules | 8.3 | O | done |
| Feature settings inside the walk-forward (FS1) | `edges_batch_results.md` | O | done |
| Gotobi on yen pairs, optimised walk-forward | 9.7 | O | done; the frozen B0 forward test stays frozen |
| Trend speeds (round 14) | `trend14_results.md` | G | covered by L |
| Round 15: 91 rules × full grids at D1 / W1 / MN | `round15_results.md` | G (one split) | **L: yearly walk-forward re-selection** |
| Round 17 intraday library (H4-M15) on 33 instruments | `round17_results.md` | A / G | **L** |
| Round 19 library on 13 commodity / index / bond CFDs and crypto | `round19_results.md` | **A** | **L** |
| Regimes, rounds 16-18 | | O | done |
| Options: 92 structures | `options_strategies_all.md` | G (menus only: 14 / 30 / 45 days, set deltas, 50% / 21 days / 2x stop) | **P: continuous delta, maturity, profit-take, stop, exit day, hedge band** |
| Options: 22 published edges (hedged straddles, strangles, strips, skew, dispersion, events, calendars, weekends) | `options_edges.md`, `options_edges2.md` | F ("nothing tuned") | **P** |
| tastytrade mechanics | `untested_batch.md` 3 | F menu | **P** |
| S&P variance at the VIX, VIX filters; SVXY | `options_retail.md`, `missed_edges.md` | F | **V: filter thresholds optimised** |
| Crypto variance premium (DVOL) | `crypto_vrp.md` | F | **V** |
| CME euro / yen options (selling, risk reversals, pinning, gamma) | `round9_results.md`, `edges_batch_results.md` O1 | F | not rerun: the bought data expired with its artifact (round 9); signals tested as features in round 16 |
| Flows and calendars (17 effects) | `flows_lab.md` | F | **T: every time window** |
| Gold / silver fixes and session hours (11) | `lit2_lab.md` | F | **T** |
| Tokyo-fix extensions, month-end London fix, rebalancing | 9.4, 9.5, 9.8 | F | **T** |
| Round 18: gold hours, fixes, home hours, around the clock, fix reversal, overnight reversal | `round18_results.md` | F | **T** |
| Edges batch E1-E26 and V1-V10 (auctions, expiries, central-bank drifts, surprises, positioning, festivals) | `edges_batch_results.md` | F | **T** (event-day filters FOMC, payrolls, month turns, Gotobi, weekdays); events with fewer than about 30 dates stay as they are (too few) |
| Round 9 (22) and round 15 free-data versions (48) | | F | **T** / **E** where the data allow; surprise data not refetched |
| Daily tests D1-D12 (composite, Hau-Rey, gold variance, cross-asset) | `untested_batch.md` 4 | F | **M** (currency signals), **T** (cross-asset next day) |
| Macro fundamentals | `macro_lab.md` | F (elastic net fitted) | **M** |
| CFTC positioning (FX, gold) | 9.1, 9.8 | F | **C2** (with commodities) |
| ICT / SMC: order blocks, sweeps, FVG, volume profile, kill zones, Asia sweep, Turtle Soup | 9.11, 9.12 | F | **I: every ICT family parameterised** |
| YouTube / forum / book systems (19 named, round 12; lit3, lit4) | `anatomy_results.md`, 9.9, 9.10 | F | covered where they are catalog rules (L); the named systems by **I** where they are level / range rules |
| Long-trend systems as published (round 13 part A) | `long_results.md` | F | **L** (the same systems are in the round-15 grids) |
| Round 19 A (ORB5, noise-area breakout), C (dip × carry), D (EIA oil), E (crypto funding), F (stop runs) | `round19_results.md` | F | **X**, **D**, **E**, **I** |
| Round 20 swing setups (Kullamägi, episodic pivot, Minervini) | `round20_results.md` | F | **S** |
| Round 21 everything (ICT pieces, SMT, real carry, slope, stock anomalies, pairs, crypto cross-section) | `round21_results.md` | F | **I**, **M**, **S**, **K** |
| Dip-buying family (Double Seven and six others) | `forward_d7_preregistration.md` | F (published defaults) | **D**; the frozen forward tracks stay frozen |
| Projections, feature information coefficients | `projection_lab.md`, `feature_predictability.md` | exhaustive | done (cost-bound by orders of magnitude) |
| Prop-firm simulations | `prop_*.md` | n/a | n/a |

## 2. Never tested (verified with a search of `docs/`)

| Market | Idea | Round 22 | Data |
|---|---|---|---|
| Commodities | Natural-gas storage report (Thursday 10:30 New York) | **E** | EIA weekly storage history + gas CFD |
| | Brent-WTI spread mean reversion | **C1** | CFDs |
| | Gold / silver and copper / gold ratio trades | **C1** | CFDs |
| | Momentum, reversal and seasonality across commodities | **C3** | commodity ETFs (real total returns; Yahoo front-month series jump at every roll) |
| | Speculator positioning (CFTC disaggregated "managed money") | **C2** | CFTC annual files |
| | OPEC meetings, USDA crop reports | not run | no reliable free date archive found; noted |
| | Commodity carry / roll yield | not run | needs second-month prices |
| Bonds | Momentum, carry and value across countries | **B** | OECD 10-year yields 1975-2025 |
| | Bonds around FOMC | **T** (FOMC filter on Bund / T-bond / Gilt CFDs, 2019-2025) | CFDs |
| Crypto | Cash-and-carry (funding harvest) with an entry threshold | **K** | Binance perpetuals and funding (round 19 files) |
| | Basis of quarterly futures | not run this round | coin-margined delivery files; next |
| Stocks | Sector rotation | not run | only current sectors are free (survivorship) |
| Options | Continuous optimisation, hedge bands | **P** | DoltHub |
| FX | Net foreign assets | not run | annual, 9 currencies |

**Blocked by data:**
- PEAD;
- small-cap minute data;
- SPX / QQQ / IWM / ES / commodity options;
- 0DTE;
- order flow;
- OANDA positioning (needs your token).
