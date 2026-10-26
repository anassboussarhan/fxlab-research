# Options edges, second batch: skew, crash puts, dispersion, weekends, FOMC / CPI, calendars, variance strips

This is the follow-up to `docs/options_edges.md` (tests H1-H8). It covers the hedged and event trades from the
research that were not tested there. As before, each test's parameters were fixed before the run and nothing was
tuned.
- Script: `scripts/options_edges2_lab.py`.
- Workflow `options-edges2`, final run 40329232105. Outputs are on branch `options-edges2`; tables are in
  `docs/options_edges2/`.
- Data: real DoltHub end-of-day quotes, 2019-2025, 106 underlyings. Every position ends by 2025-12-31.
- Costs: options at mid +- 0.75 x half-spread plus USD 0.65 per contract; hedges 1 bp of traded notional on SPY /
  DIA and 3 bp on stocks; expiring options settle at intrinsic.
- Unit: P&L per unit of underlying notional.

**Data fix in this run.** On some snapshots the database's implied-volatility field is wrong for a whole expiry
(for example SPY in mid-November 2021: 0.83 for options priced at about 11% volatility, or -1).
- Where the database IV disagrees with the quotes' own at-the-money volatility by more than 1.5x, every IV of that
  expiry is now recomputed from the mid prices.
- That happened on 902 expiry-days across the 106 underlyings (SPY 16, DIA 15).
- It mattered for two tests that use marks of unlisted expiries:
  - the SPY weekend trade (a fake -11% weekend in November 2021 is gone; t went from 0.17 to 1.78);
  - the SPY calendar straddle (a fake +16% in 2021 is gone; t went from 1.13 to 0.41).
- The first two runs of this batch (40323724545, 40325437704) are superseded.
- No hedge delta fell back to a crude 0 / +-1 value (326,967 deltas).

## 1. Results

One-sided tests. The Benjamini-Hochberg q is across the 14 tests.

| Test | n | Mean | t | Sharpe | Win | Worst | Years + | 2019-22 mean (t) | 2023-25 mean (t) | BH q |
|---|---|---|---|---|---|---|---|---|---|---|
| **E13 SPY short variance strip, delta-hedged** | 82 | **+0.06%** | **2.02** | 0.74 | 80% | -1.6% | 6/7 | +0.08% (2.39) | +0.03% (0.53) | 0.15 |
| **E14 DIA short variance strip, delta-hedged** | 78 | **+0.07%** | **2.50** | 0.92 | 78% | -1.4% | 6/7 | +0.10% (2.35) | +0.03% (1.03) | 0.09 |
| E6 Weekend effect SPY (sell hedged straddle Friday, buy back next day, net) | 262 weekends | +0.05% | 1.78 | 0.80 | 71% | -5.1% | 5/6 | +0.05% (0.87) | +0.05% (3.42) | 0.17 |
| E8 Weekend minus weekday one-day P&L, SPY (gross) | 56 weeks | +0.08% | 1.53 | 1.49 | 63% | -0.7% | 2/2 | | +0.08% (1.53) | 0.22 |
| E3 SPY delta-hedged short 10-delta put | 82 | +0.05% | 0.85 | 0.32 | 87% | -3.7% | 6/7 | +0.10% (1.50) | -0.01% (-0.08) | 0.47 |
| E4 DIA delta-hedged short 10-delta put | 82 | +0.05% | 0.83 | 0.29 | 79% | -3.4% | 5/7 | +0.05% (0.64) | +0.04% (0.53) | 0.47 |
| E11 SPY delta-hedged calendar straddle | 82 | +0.03% | 0.41 | 0.16 | 55% | -3.0% | 5/7 | -0.02% (-0.15) | +0.09% (1.13) | 0.68 |
| E10 CPI: sell SPY hedged straddle before, buy back after | 81 | -0.03% | -0.47 | -0.17 | 60% | -2.1% | 4/7 | -0.06% (-0.70) | +0.01% (0.23) | 1.00 |
| E9 FOMC: the same around scheduled statements | 52 | -0.15% | -2.05 | -0.77 | 38% | -2.2% | 2/7 | -0.10% (-1.13) | -0.21% (-1.79) | 1.00 |
| E5 Dispersion (short SPY hedged straddle, long hedged stock straddles) | 82 | -0.19% | -2.22 | -0.73 | 35% | -4.7% | 1/7 | -0.22% (-1.85) | -0.14% (-1.24) | 1.00 |
| E1 SPY delta-hedged risk reversal (sell 25-delta put, buy 25-delta call) | 74 | -0.20% | -2.73 | -1.08 | 30% | -3.2% | 1/7 | -0.09% (-1.11) | -0.37% (-2.97) | 1.00 |
| E12 DIA delta-hedged calendar straddle | 82 | -0.28% | -3.61 | -1.32 | 33% | -3.9% | 1/7 | -0.31% (-2.92) | -0.25% (-2.19) | 1.00 |
| E2 DIA delta-hedged risk reversal | 70 | -0.27% | -4.94 | -1.87 | 29% | -1.8% | 0/7 | -0.23% (-2.91) | -0.33% (-4.81) | 1.00 |
| E7 Weekend effect, stocks (equal weight, net) | 267 weekends | -0.64% | -8.95 | -4.96 | 4% | -11.0% | 0/6 | -0.86% (-6.43) | -0.41% (-13.59) | 1.00 |

Notes on the table:
- Monthly tests: mean per position. E6 / E7: per weekend; E8: per week. Weekends need snapshots on consecutive
  trading days, so they start in 2020; E8 needs daily snapshots, so 2024-2025 only.
- **Variance strips (E13, E14).**
  - They sell about 19 listed out-of-the-money options with weights 2 dK / K^2.
  - In volatility points: SPY +1.28 a month (t 1.28), DIA +1.50 (t 1.60). The SPY figure is in line with the
    VIX-based variance-swap estimate for 2019-2025 (+1.26, `docs/options_strategies.md`).
- **Weekend effect.**
  - Gross, the SPY hedged straddle earns +0.095% over a weekend and -0.003% over an ordinary trading day.
  - After costs the SPY trade keeps +0.05% per weekend (about 2.7% of notional a year): 5 of 6 years positive,
    t 3.4 in 2023-2025.
  - On stocks the bid / ask on options bought back after one day costs far more than the weekend decay.
- **Dispersion.** It loses at notional weights. Scaled so that the stock straddles' premium equals the index
  premium, it is flat (-0.01% a month, t -0.10).

**Box-spread check (diagnostic).** Boxes on strikes at or below the forward, at least 2% wide, against the 13-week
T-bill:
- SPY, 20,720 boxes: none could be bought below the T-bill value at the asks after commissions (0.005% of boxes,
  0.1% of days).
- Selling at the bids looked better than the T-bill for 8.5% of SPY boxes (35% of days) and 0.4% of DIA boxes.
- Mid prices imply rates 1.5% (SPY) and 3.4% (DIA) below the T-bill. That is what American-style calls on
  dividend-paying ETFs are worth (early exercise before ex-dividend dates), plus closing quotes that are not
  simultaneous.
- It is not an executable arbitrage. Box arbitrage needs European options (SPX) and live quotes.

## 2. What it adds

- **The index variance premium is the only thing that keeps showing up:**
  - delta-hedged ATM straddles and 25-delta strangles (H1, H3);
  - now the full variance strip (E13, E14), and the weekend version of the same premium (E6).
  - All are positive in 5-6 of 7 years.
  - None is significant after correcting for the number of tests (best q 0.09).
  - All are about half as large in 2023-2025 as in 2019-2022, except the weekend trade.
- **Selling skew on its own loses:** the delta-hedged risk reversals E1 and E2 lose in 6-7 of 7 years. The premium is
  in the level of index volatility, not in puts relative to calls.
- **Deep out-of-the-money puts (E3, E4)** earn a little, not significantly, and lost in 2025.
- **Events lose or are flat:**
  - FOMC: volatility is not overpriced enough before the statement to pay two round trips of spread;
  - CPI: the same;
  - single-stock weekends: the same.
- **Dispersion and DIA calendars lose.** The SPY calendar is flat once the bad marks are fixed.

**Not tested here, and why:**
- **Selling straddles into earnings** (the mirror of H8). The idea came from these data, so it needs data not used
  so far.
- **0DTE and intraday trades.** End-of-day quotes only.
- **SPX, QQQ, VIX options and futures options.** Not in the free database. OptionsDX free files would add SPX, QQQ
  and VIX for 2010-2023.
