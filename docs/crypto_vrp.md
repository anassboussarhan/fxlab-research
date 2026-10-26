# Crypto variance premium (free Deribit data)

Deribit publishes DVOL, the 30-day implied volatility index of its BTC and ETH options, through a public API with
no account.
- DVOL starts 2021-03-24.
- This tests selling 30-day variance at DVOL each month against the realised volatility that followed. It uses the
  same proxy as the S&P test in `docs/options_retail.md`: P&L in vol points = (K^2 - RV^2) / 2K, with 365-day
  annualisation, minus 2 vol points of cost.
- Script `scripts/crypto_vrp_lab.py`; workflow `free-options-data`, run 40305837180. Data to 2025-12-31.

| Test | Months | Mean (vol pts / month) | t | Sharpe | Months + | Worst | Gross mean (t) | BH q |
|---|---|---|---|---|---|---|---|---|
| **C1 BTC** | 57 | **+3.5** | **1.76** | 0.78 | 67% | -50 (May 2021) | +5.5 (2.76) | 0.16 |
| C3 BTC, only when DVOL > past 30-day RV | 42 | +2.3 | 0.96 | 0.47 | 62% | -50 | +4.3 (1.79) | 0.34 |
| C2 ETH | 57 | -2.0 | -0.51 | -0.23 | 60% | -164 (May 2021) | +0.0 (0.00) | 0.70 |
| C4 ETH, only when DVOL > past 30-day RV | 34 | -2.1 | -0.40 | -0.22 | 59% | -164 | -0.1 (-0.02) | 0.70 |

- **BTC:**
  - DVOL averaged 61.2 against 53.3 realised.
  - The premium is positive in every year 2021-2025: +86, +50, +31, +23 and +10 vol points.
  - It shrinks each year, and it is not significant after correcting for the four tests (q = 0.16).
- **ETH:** no premium before costs.
- **Caveats:**
  - A variance swap at DVOL is a proxy with no bid/ask beyond the 2 vol-point cost.
  - Listed Deribit options would be sold as delta-hedged straddles or strangles, as in `docs/options_edges.md`.
  - Four and a half years is a short sample with one extreme month (May 2021).
