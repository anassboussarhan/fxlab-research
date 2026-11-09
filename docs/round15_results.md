# Round 15 results: every rule at every slow timeframe, the book systems, every feature, free data (2026-11-09)

Rules: `docs/round15_preregistration.md` (addenda 1-3), committed before each part was run. Code:
- `fxlab/strategies/books.py` (26 book systems; `tests/test_books.py`);
- `scripts/sweep15.py` and `scripts/sweep15_combine.py` (parts A-C);
- `scripts/family_d7.py` (addendum 2);
- `scripts/data15_tests.py` (part D).

Runs: sweep 42585859892, data 42589646502 and misc `data15b`. Tables are in `docs/round15/`.

**Results:**
- **Nothing survives the full multiple-testing correction:**
  - 233 rule units (91 rules: catalog, book and round-12 systems) at D1 / W1 / MN, over 2,442 setting-and-direction
    configurations;
  - 838 feature units (315 feature columns) over 1,698 configurations.
  - Every unit had its setting picked on 2008-2018 and was judged once on 2019-2025: 0 strict passes, 0 regime
    candidates.
- **The only consistent pattern is weekly dip-buying** (Double Seven's family). As a pre-registered family it was
  also **supported on 2026** (175 trades, +0.38R, t 2.2). It now runs as a third forward-test track.
- **Free data:**
  - The paid-data gaps that could be filled for free were filled: the economic-calendar archive, CBOE strategy
    indices, two-year yields, and commodity / index / bond CFDs (the last still downloading).
  - Only the one-day drift after big economic surprises came close (+5.4 bp a trade, t 2.2 on 2019-2025), and it
    does not survive the correction across the data tests.
  - Option-selling indices, yield momentum and a surprise index fail. The option-selling indices' 2019-2025 Sharpe
    ratios (-0.17 to 1.08) are no better than the S&P 500's (1.05) once their market exposure is taken out.

## The coverage audit: why Double Seven was missed

Double Seven was in the catalog from round 4. It was tested at M5-D1 on 2007-2018, at D1 on 2019-2025 with default
settings, and at W1 only in round 13. The audit found these gaps, and round 15 filled them:
- **Monthly bars:** never tested before.
- **Weekly bars:** never tested for rules outside the D1 list.
- **Full parameter grids:** never judged on 2019-2025.
- **Reversed rules:** the contrarian version of a rule was never tested on slow bars.
- **Families and systems:** the esoteric rules after round 4, the round-12 systems at W1 / MN, and features on slow
  bars. Classic book systems were missing outright: 26 are now coded.

**Why selection on 2008-2018 cannot find a rule like Double Seven.** On FX, all six of its published-direction
settings lost in 2008-2018 (t -0.5 to -1.3) and won in 2019-2025 (t 1.4 to 2.4). So the fit period always prefers
the reversed rule, which then loses (t -2.8). A rule whose sign flips between decades is invisible to any "choose
on the past, test on the future" procedure. Only a regime view, checked on new data, can find it.

## A-C. The sweep

| | Units | Median fit t | Median check t | Selected units with check t >= 1.645 | Random-signal placebos with check t >= 1.645 |
|---|---|---|---|---|---|
| Rules, D1 | 91 | -0.16 | -1.35 | 0.0% | 0.0% |
| Rules, W1 | 84 | -0.02 | -0.51 | **6.0%** | 0.0% |
| Rules, MN | 58 | +0.34 | -1.04 | 0.0% | 0.0% |
| Features, D1 | 312 | | -1.98 | 0.0% | |
| Features, W1 | 296 | | -0.97 | 0.3% | |
| Features, MN | 230 | | -1.08 | 0.4% | |
| All placebos | 150 | | | | 0.0% |

**Best selected units on 2019-2025** (none passes BH; with 1,071 units, the first threshold is p <= 0.0001):

| Unit | Setting (direction) | Fit t | Check t |
|---|---|---|---|
| Channel breakout, W1 | k = 2, n = 50 (as published) | -0.25 | 2.72 |
| Cumulative RSI, W1 | k = 3, lo = 50, x = 65, long only | 1.23 | 2.65 |
| TWAP reversion, W1 | k = 2, n = 12 | -0.09 | 2.05 |
| Adaptive (KAMA) trend, W1 | e_min 0.4, n 20 (reversed) | 0.05 | 2.05 |
| Connors R3, W1 (book) | long only | 0.76 | 2.00 |
| Lunar-phase feature, W1 | cos, terciles | -1.11 | 1.86 |
| Moving-average ribbon dispersion, MN | terciles | 1.36 | 1.84 |

**Overall:**
- The rule surface passes t 1.645 at about the rate random signals do: 1.3-2.2% of rule settings against 0-1.9%
  of placebo settings. Weekly bars are the exception: 6% of the selected weekly rule units, mostly dip-buying
  rules.
- **Book systems:** the 26 new systems (68 units) had a median check t of -1.15. Their best was Connors R3 weekly
  (t 2.00).
- **The other book systems failed:** Williams, Raschke, Bill Williams, Wilder, Kathy Lien, Ichimoku, Heikin-Ashi,
  point and figure, double tops, head and shoulders, Gann HiLo and Elder Impulse.
- **Esoteric rules:** they were run, and none passes.

## Addendum 2: the weekly dip-buying family on 2026 (pre-registered)

The seven close-only Connors rules ran at their published defaults, long only, on weekly ECB fixings of the 28
crosses, as one book.

| | Trades | Mean R | t (by month) | Win rate |
|---|---|---|---|---|
| 2008-2018 | 1,559 | -0.20 | -1.49 | 61% |
| 2019-2025 (comparator) | 1,079 | +0.24 | 3.03 | 67% |
| **2026 (holdout)** | **175** | **+0.38** | **2.23** | **73%** |
| 2026 without Double Seven | 142 | +0.29 | 1.47 | 70% |

| Rule (2026) | Trades | Mean R | t |
|---|---|---|---|
| Double Seven | 33 | +0.78 | 5.38 |
| Cumulative RSI | 39 | +0.47 | 2.26 |
| RSI(2) | 35 | +0.33 | 2.04 |
| R3 | 21 | +0.19 | 0.78 |
| %b | 4 | +1.94 | - |
| ConnorsRSI | 5 | +0.20 | 1.11 |
| Three down closes | 38 | -0.04 | -0.16 |

**Verdicts:**
- **The family: supported.** Without Double Seven: inconclusive (positive, t 1.47).
- **What this means:** weekly dip-buying in currencies is not one lucky rule. Six of the seven rules made money in
  2026, as they did in 2019-2025.
- **The caveat is unchanged:** every rule in the family lost in 2008-2018, the trending decade.

**What happened next:** the family is now the forward test's third track, "D7-family", added before the first
forward signal (`docs/forward_d7_preregistration.md`).

## D. Free versions of paid data

| Dataset (what it replaces) | Got it? | Test | 2008-2018 | 2019-2025 | Verdict |
|---|---|---|---|---|---|
| Forex Factory calendar archive, 83,427 releases with consensus (Bloomberg / Reuters consensus feeds) | yes (HuggingFace, MIT) | D1a: drift after a high-impact surprise, 60 minutes | -1.18 bp (t -2.91) | +0.72 bp (t 1.21) | fails |
| | | D1a: drift after a high-impact surprise, 1 day | +1.93 bp (t 1.20), 1,902 trades | **+5.38 bp (t 2.21)**, 646 trades | meets its rule; fails the correction over the D rows |
| | | D1b: weekly surprise index (a free Citi Economic Surprise Index) on 28 crosses | t 0.84 | t 0.54 | fails |
| CBOE option-selling indices (dealer option data) | yes (cdn.cboe.com) | D4: alpha against the S&P 500 total return (BXM, BXMD, PUT, CNDR, BFLY, CLL, CMBO, WPUT, PPUT, CLLZ) | alpha t -1.3 to +0.1 | alpha t -2.1 to +0.5 | fails (all 10) |
| Two-year yields (Bloomberg curves) | yes: US Treasury, ECB, Bank of Canada, RBA (from 2013), SNB, Japan MoF | D5: 4-week rate-differential momentum, 15 crosses | t -1.63 | t 0.23 | fails |
| | | D5: 13-week | t -1.32 | t -0.22 | fails |
| Commodity, index and bond CFDs | yes (Dukascopy, mostly from 2013; bonds from 2019), but the feed throttles | D2: round-13 trend systems | running | running | running (see D2) |
| Retail sentiment (Dukascopy SWFX, IG, Myfxbook) | no: SWFX blocked (403), no public history elsewhere | - | - | - | data missing |
| Google Trends | no (rate-limited, 429) | - | - | - | data missing |
| Japanese retail futures (TFX Click 365) | only a current file, no history | - | - | - | data missing |
| OANDA order and position books | needs your OANDA API token | - | - | - | waiting |
| Scandinavian / EM FX | not served by Dukascopy's free candle feed | - | - | - | data missing |

**The one-day surprise drift** trades the currency in the direction of a high-impact surprise, from 5 minutes
after the release to the same time the next day. It met its own rule: positive in both periods, t 2.21 in
2019-2025. The 60-minute version lost in 2008-2018.

## What is still not coded, and why

| Method | Why not |
|---|---|
| Elliott wave counts, Wyckoff phases, discretionary price action | No unique published rule. Their codable parts are in the catalog: swing structure, the trend-line break, the sweep reversal (Wyckoff spring) and the round-12 Quasimodo. |
| Order-flow absorption, footprint and depth-of-market methods | They need full order books. Free FX data have none, and CME depth would cost money. |
| Martingale, grid and pyramid EAs | These are position-sizing schemes, not signals. They were tested earlier (`retail_mm`) and lose with any signal that has no edge. |
| Copy-trading and signal services | Their track records are survivor-selected. They were reviewed in the MQL5 and Trustpilot rounds. |
| OANDA order and position books | They need your API token. |

## D2. Commodities, indices and bonds (running)

**First attempt (run 42589646502):** Dukascopy refused or timed out on most requests under 13 parallel jobs.
- Commodities and indices came back only for 2011-2015, bonds not at all, so it has no 2019-2025 result.
- On those early years the fixed trend systems were mixed: commodities t -1.1 to +0.7, indices -1.6 to +1.1.

**Second attempt (run 42610347008, workflow `d2dl`):** one instrument-year per job with 3 connections, as the
earlier EM downloads did. The results will be added here when it finishes. The rules are unchanged (addendum 3).
