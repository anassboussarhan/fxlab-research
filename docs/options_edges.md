# Options edges from the research, tested on real quotes

The question was whether there is an edge in options. This is the follow-up to `docs/options_strategies.md`, which
tested unhedged structures. Here, eight edges from the published research were tested with their parameters fixed
in advance (nothing tuned).
- Script: `scripts/options_edges_lab.py`.
- Workflow `options-edges`, final run 40301779747. Outputs are on branch `options-edges`; tables are in
  `docs/options_edges/`.
- Data: real DoltHub end-of-day quotes, 2019-2025, 106 underlyings. The data and cost conventions are those of
  `docs/options_strategies.md`. Every position ends by 2025-12-31.

**Valuation fix behind this run:**
- The database lists different expiries on each snapshot; a held expiry was quoted on only about 28% of later
  dates.
- On the other dates it is now valued by interpolating total implied variance across that day's listed expiries,
  at the same log-moneyness.
- Accuracy, from hiding whole listed expiries and re-pricing them from the others: median error 0.06% of the
  underlying on SPY (90th percentile 0.31%); 0.17% median across underlyings.
- With that, none of the 637,398 hedge deltas fell back to a crude 0 / +-1 delta.
- The first two runs of this lab (40300348392, 40300593069) used those fallbacks and are superseded.

## 1. Results

Units: P&L per unit of underlying notional, per position (monthly tests) or per week (H6, H7). The tests are
one-sided; Benjamini-Hochberg q is across the eight.

| Test | n | Mean | t | Sharpe | Win | Worst | Years + | 2019-22 mean (t) | 2023-25 mean (t) | BH q |
|---|---|---|---|---|---|---|---|---|---|---|
| **H1 SPY delta-hedged short ATM straddle, 30 days** | 82 | **+0.27%** | **1.74** | 0.71 | 67% | -5.6% | 5/7 | +0.34% (1.59) | +0.18% (0.80) | 0.16 |
| H2 H1 only when VIX < VIX3M | 78 | +0.19% | 1.24 | 0.52 | 65% | -5.6% | 5/7 | +0.24% (1.15) | +0.13% (0.55) | 0.29 |
| **H3 SPY delta-hedged short 25-delta strangle** | 74 | **+0.23%** | **1.76** | 0.74 | 66% | -4.9% | 5/7 | +0.34% (2.27) | +0.05% (0.22) | 0.16 |
| H4 Vasquez IV term slope, straddles (stocks, long-short) | 82 | -0.01% | -0.03 | -0.01 | 52% | -6.2% | 3/7 | | | 0.82 |
| H5 Cao-Han idiosyncratic vol, delta-hedged (stocks) | 82 | -0.54% | -3.97 | -1.57 | 32% | -4.0% | 0/7 | | | 1.00 |
| H6 Cremers-Weinbaum call-put IV spread (stocks, weekly) | 352 | -0.10% | -1.48 | -0.59 | 43% | -2.9% | 1/7 | | | 1.00 |
| H7 Xing-Zhang-Zhao smirk (stocks, weekly) | 352 | +0.11% | 1.04 | 0.41 | 51% | -7.5% | 4/7 | | | 0.30 |
| H8 Gao-Xing-Zhang long straddles into earnings | 72 months (2,285 trades) | -1.36% / month | -5.52 | -2.20 | 22% | -8.3% | 1/6 | | | 1.00 |

- **The index volatility premium is there when the option is delta-hedged.**
  - H1 and H3 are positive in 5 of 7 years, significant on their own (p = 0.04), but not after correcting for the
    eight tests (q = 0.16).
  - SPY by year: +4.5%, +9.2%, +4.2%, -2.1%, +3.5%, +3.2% and -0.5% of notional for 2019-2025.
  - Through the March 2020 crash, the 19 Feb position lost 2.7% hedged, against 26.4% unhedged.
  - The premium collected averaged 3.8% of notional, and hedging cost 0.03%.
- **DIA, not a registered test, shows the same thing** (n = 82, plain t without autocorrelation adjustment):
  - delta-hedged short straddle +0.35% a month, t about 2.5;
  - 25-delta strangle +0.40% a month, t about 3.8 (n = 71).
- **On the sector ETFs it is absent:** XLF hedged short straddle +0.20%, XLE -0.45%.
- **It is smaller recently:** 2023-2025 shows about half the 2019-2022 mean, and 2022 and 2025 were down years.
- **The single-stock anomalies do not survive the bid/ask:**
  - H5 loses 0.54% a month: both books pay single-stock option spreads.
  - H6 is negative after 12 bp a week of stock costs.
  - H4 and H7 are not significant.
- **Buying straddles into earnings loses heavily:**
  - -0.85% of notional a trade, -11% of the premium on average (median -26%);
  - 28% of trades won, and the mean per trade was negative in every year 2020-2025.
  - Implied volatility is overpriced before earnings: the "IV crush".

## 2. A new hypothesis from H8, not a result

The mirror trade of H8 is to **sell** the at-the-money straddle 2-10 days before earnings and buy it back 1-7 days
after. Before its own costs it would have earned about what H8 lost. But it was suggested by these same data, so it
is not tested. It needs data this project has not used: the 2026 holdout (locked), or 2007-2018 option quotes,
e.g. ORATS history (USD 599, not bought).

## 3. On the capital providers

The delta-hedged SPY book (one new position a month, daily marks from the day's smiles) was run on the published
rules from `docs/options_capital_providers.md`. Results are in `docs/options_edges/prop_hedged_spy.csv`.

| Book | Size (daily sd) | Options Funding Growth: mean / median net | Imperial Elite: mean / median net | Imperial paid at least once |
|---|---|---|---|---|
| SPY hedged straddle | 0.6% | -$3,478 / -$2,257 | +$415 / -$357 | 37% |
| SPY hedged straddle | 0.8% | -$2,314 / -$1,908 | +$961 / +$464 | 61% |
| SPY hedged straddle, contango only | 1.0% | -$1,114 / -$1,106 | **+$1,478 / +$1,657** | 73% |
| DIA hedged straddle | 0.8% | -$0 / -$1,000 | +$635 / -$357 | 45% |

- **The monthly-fee provider loses money in every case.**
- **The one-time-fee provider is positive on average,** with a positive median at the larger sizes. Its 7%
  sim-funded payout cap limits the upside to $3,143 per account.
- **The risk is mark-to-market.**
  - At 1.0% daily sd the worst day is about -20% of the account (the March 2020 vega spike).
  - 63-98% of accounts are eventually breached.
- **Practical constraint:** the hedge trades the underlying, so the provider must allow stock / ETF trading next to
  the options:
  - Imperial lists stocks and options;
  - Options Funding lists options only;
  - neither publishes its rules on short straddles.

## 4. Conclusion

- **Where the edge is:** the one options edge that shows up on real quotes is the index variance premium, earned
  by selling at-the-money SPY / DIA straddles or 25-delta strangles monthly and delta-hedging them.
- **How strong it is:** +0.2-0.4% of notional a month, about Sharpe 0.7 on SPY, positive in 5 of 7 years, worst
  month about -5.6%.
- **It agrees with the 35-year variance-swap evidence** (`docs/options_retail.md`: +1.8 vol points a month,
  t 4.7).
- **What weakens it:**
  - in 2019-2025 it is not significant after correcting for the eight tests;
  - it was weaker in 2023-2025;
  - its marks fall hard in volatility spikes.
- **On a prop account** it only makes sense on a one-time-fee provider that allows the hedge, at a size that
  survives a March-2020 day.
- **The single-stock option anomalies and earnings straddles lose after costs.** The earnings IV crush (selling) is
  a hypothesis for new data.
