# Edges the project had not tested

This tests published edges outside the earlier FX and options work. Each test's parameters were fixed before the
run, and nothing was tuned.
- Script: `scripts/missed_edges_lab.py`.
- Workflow `missed-edges`, final run 40374910046. Outputs are on branch `missed-edges`; tables are in
  `docs/missed_edges/`.
- Data, all free:
  - Yahoo daily prices for 22 ETFs, SPY / QQQ / IWM, SVXY, VIX, VIX3M, the 13-week T-bill and 19 FX pairs;
  - the S&P 100 closes and DoltHub earnings dates of the options lab;
  - Deribit's hourly perpetual funding.
- Everything is cut at 2025-12-31.

**Two corrections before the final run:**
- The overnight trades now pay the day's T-bill interest. Cash held in the account earns it, so holding stock over
  the close forgoes it.
- The SVXY contango decision now uses the VIX closes from two days back. VIX closes 15 minutes after SVXY, and
  using the same day's VIX had added several points of Sharpe. Run 40373798461 is superseded.

## 1. Results

One-sided tests. The Benjamini-Hochberg q is across the 15 tests that ran. EVZ could not be fetched: Yahoo no longer
has it and Cboe refused the runner.

| Test | Period | n | Mean | t | Sharpe | Years + | Worst | Before 2019 t | 2019-2025 t | BH q |
|---|---|---|---|---|---|---|---|---|---|---|
| **M14 BTC cash-and-carry (long spot, short Deribit perpetual)** | 2019-25 | 81 months | **+0.69% / month** | **4.63** | 2.08 | 6/7 | -3.3% | | 4.63 | 0.000 |
| **M15 ETH cash-and-carry** | 2019-25 | 81 | **+0.51%** | **2.65** | 1.22 | 6/7 | -4.2% | | 2.65 | 0.012 |
| **M3 QQQ overnight (buy close, sell open)** | 1999-25 | 6,745 days | **+9.5% / year** | **3.76** | 0.67 | 21/27 | -9.5% | 3.34 | 1.44 | 0.001 |
| **M4 IWM overnight** | 2000-25 | 6,437 | **+9.1% / year** | **3.66** | 0.69 | 19/26 | -9.1% | 2.94 | 2.07 | 0.001 |
| **M2 SPY overnight** | 1993-25 | 8,287 | +5.1% / year | 3.01 | 0.48 | 26/33 | -10.5% | 2.78 | 1.07 | 0.005 |
| **M8 Short volatility (SVXY) only in VIX contango** | 2012-25 | 168 months | +2.3% / month | 2.38 | 0.62 | 9/14 | -40.3% (Feb 2018) | 2.05 | 1.26 | 0.021 |
| M9 SVXY held always | 2012-25 | 168 | +2.5% | 2.22 | 0.57 | 9/14 | -89.7% (Feb 2018) | 1.79 | 1.38 | 0.025 |
| M5 Turn of the month, SPY | 1993-25 | 394 months | +0.24% per window | 2.30 | 0.40 | 21/33 | -7.9% | 2.01 | 1.13 | 0.023 |
| M1 Cross-asset time-series momentum, 22 ETFs | 2008-25 | 215 months | +0.51% / month | 1.65 | 0.38 | 14/18 | -12.6% | 1.48 | 0.82 | 0.082 |
| M6 Pre-holiday, SPY | 1993-25 | 296 | +0.08% | 1.53 | 0.25 | 18/33 | -3.3% | 0.49 | 3.59 | 0.095 |
| M11 Earnings-announcement drift, S&P 100 | 2020-25 | 47 months | +0.63% | 1.10 | 0.61 | 4/6 | -6.4% | | 1.10 | 0.18 |
| M7 Pre-FOMC day, SPY | 2019-25 | 55 | +0.17% | 0.99 | 0.38 | 6/7 | -3.0% | | 0.99 | 0.20 |
| M16 FX daily Bollinger mean reversion, 19 pairs | 2019-25 | 84 months | +0.05% | 0.53 | 0.19 | 4/7 | -3.3% | | 0.53 | 0.35 |
| M12 Momentum 12-1, S&P 100 (long-short) | 2008-25 | 204 | -0.31% | -0.70 | -0.18 | 9/18 | -36.9% (Mar 2009) | -0.68 | -0.25 | 0.76 |
| M13 Low volatility, beta-neutral, S&P 100 | 2008-25 | 216 | -0.31% | -0.72 | -0.18 | 9/19 | -21.4% | -0.51 | -0.50 | 0.76 |

## 2. What each one is, and what holds up

- **Crypto cash-and-carry (M14, M15) is the strongest result by Sharpe.**
  - Leveraged crypto longs pay funding to the shorts. Holding spot against a short perpetual collects it with no
    price exposure.
  - BTC: +8.2% a year, positive in 6 of 7 years (2022, the bear market, -2.3%); worst month -3.3% (March 2020).
  - It shrank in 2025: +0.0-0.9% a month.
  - Its risks are not in the backtest: the exchange can fail (FTX, 2022), the short leg can be liquidated in a
    spike if under-collateralised, and fees and funding differ by venue.
  - It is a carry premium, not a prediction.
- **Equity overnight returns (M2-M4) are real.**
  - Since 1993, US index returns came almost entirely overnight. Intraday averaged +0.7 bp a day on SPY, -0.1 bp on
    QQQ and -1.0 bp on IWM.
  - Holding only overnight is still equity exposure.
  - Against buy-and-hold (excess Sharpe SPY 0.51, QQQ 0.43, IWM 0.37):
    - on SPY it is no better (0.48);
    - on QQQ and IWM it is clearly better (0.67-0.69), because it skips intraday losses.
  - Weaker since 2019 (t 1.1-2.1).
  - The 1 bp per round trip assumes market-on-close / market-on-open orders.
- **Short volatility through SVXY (M8, M9) is the VIX futures roll premium:** +27-29% a year, with Feb 2018 (-40% to
  -90% in a month).
  - With the timing corrected, the contango filter did not avoid February 2018. It mainly lowers the worst month.
  - SVXY has carried half the exposure since 2018.
  - This is the same index variance premium as in `docs/options_edges.md`, in its most crash-exposed form.
- **Calendar effects (M5-M7) are mostly the equity premium.**
  - The turn-of-the-month window earned +0.24%, against +0.14% for any four days. The extra is about 0.1% a month.
  - Pre-holiday days earned +0.08% against +0.03% on any day.
  - The pre-FOMC day earned +0.17% against +0.05% (not significant).
- **Cross-asset trend (M1) is positive as expected, modest, and not significant.**
  - It made money in 2008 and 2022, the crisis years, and lost in 2009, 2016, 2018 and 2023.
- **Not there on the S&P 100:** momentum and low volatility lose, and earnings drift is not significant. The sample
  is today's members only.
- **The project's best FX near-miss before 2019, daily Bollinger mean reversion, is flat on 2019-2025 data**
  (t 0.5). It was noise.

## 3. Limits

- Yahoo daily data with modelled costs, not executable quotes.
- Survivorship in the S&P 100.
- The SVXY leverage change in 2018.
- Crypto exchange and counterparty risk.
- Seven years for the crypto and earnings tests.
- These tests are separate from the options and FX programmes. Taken across the whole project, a q-value of 0.02
  is weaker than it looks.
