# Macro-fundamental currency signals (runs 18, 20, 21)

Scripts: `scripts/macro_lab.py` (one job per USD pair: monthly FX series and macro data) and
`scripts/macro_report.py` (strategies, costs, BH). Results: branches `results20` and `results21`, folder
`sweep_results/macro/`.
- FX: 9 currencies vs USD, monthly, 2007-04 to 2018-06. EUR/USD runs to 2022, but the cross-section needs at
  least 6 currencies, so the tests end in 2018-06.
- Macro data are cut at 2025-12-31. The 2026 holdout is not loaded.

## Why

The literature check (`docs/literature_check.md`) found macro fundamentals to be the category of published FX
predictors that runs 1-16 had not tested. The four tests below take the published designs as stated and were fixed
before the run.

## Design

The portfolio is rebalanced at each month-end and held for the next month:
- long the 3 currencies with the highest signal and short the 3 lowest, weights +-1/3;
- excess return = spot return + carry from the point-in-time 3-month rate differential, where rates exist (SEK and
  NOK have none);
- costs = (half spread + 0.25 x spread slippage) x turnover + 0.35 bp commission per unit traded.

| Test | Signal | Source |
|---|---|---|
| EM | Economic momentum: 12/24/36-month changes in industrial production, retail sales, unemployment (sign reversed), CPI and PPI, relative to the US, z-scored across currencies, averaged over at least 2 series | Dahlquist & Hasseltoft (2020) |
| OG | Output gap: log industrial production minus a one-sided HP trend (lambda 129,600, data to the decision date only), relative to the US | Colacito, Riddiough & Sarno (2020) |
| TR | Taylor-rule fundamentals: 1.5 x 12-month inflation + 0.5 x output gap, relative to the US | after Molodtsova & Papell (2009) |
| ML | Elastic net: panel regression of next month's excess return on the signals above plus carry, 3- and 12-month FX momentum and 5-year value; expanding window from 36 months; rank by forecast | after Filippou, Rapach, Taylor & Zhou |

- One-sided p-values, with Benjamini-Hochberg over the four.
- Diagnostics outside the BH family: carry, 12-month FX momentum, 5-year value, and the output gap as a
  time-series signal.
- **Point in time.** The data are revised figures, not real-time vintages. A monthly figure is used from 2 months
  after its month; a quarterly figure from 5 months after the start of its quarter, about 2 months after the
  quarter ends.

## Data

- **Run 18 fetched nothing.** Every request to FRED timed out on the GitHub runner: about 400 requests at 60 s
  each, 51 minutes for the EUR/USD job alone. EM, OG and TR had no data, and ML ran on the FX-only features.
  Run 18 is a data failure, not a result.
- **A probe** (`ci/probe.py`, branch `probe`) found that DBnomics (api.db.nomics.world), which mirrors the OECD
  Main Economic Indicators and the IMF International Financial Statistics, answers from the runner. FRED and Cboe
  do not.
- **Run 20 fetched all 50 series** (10 economies x 5 indicators).
  - The rule prefers an OECD monthly series starting by 2004-01, then OECD quarterly, then IMF.
  - Quarterly series: Australia (CPI, IP, PPI, retail), New Zealand (all five), Switzerland (IP, unemployment).
  - IMF series: UK and Japan PPI.
  - The euro area is EA19 for IP, retail and PPI, and EA20 for CPI and unemployment.
  - Every series starts before 2004 except Swiss unemployment (1999-04) and Swiss PPI (2002-06), which still
    leave enough history before 2007.
  - `sweep_results/macro/coverage.csv` lists the series id, first and last date of each.

## Results: EM, OG, TR (run 20; unchanged in run 21)

| Test | Months | Net bp / month | t | Sharpe | Years + | Gross bp | BH q |
|---|---|---|---|---|---|---|---|
| EM economic momentum | 135 | +25.1 | 1.32 | 0.39 | 58% | +27.4 | 0.38 |
| OG output gap | 135 | -0.5 | -0.03 | -0.01 | 50% | +3.0 | 0.67 |
| TR Taylor rule | 135 | +3.9 | 0.22 | 0.07 | 50% | +6.4 | 0.67 |
| carry (diagnostic) | 135 | -1.3 | -0.06 | -0.02 | 50% | -1.1 | |
| 12-month FX momentum (diagnostic) | 123 | +2.9 | 0.14 | 0.04 | 55% | +5.8 | |
| 5-year value (diagnostic) | 99 | -4.0 | -0.23 | -0.08 | 33% | -2.5 | |
| output gap, time series (diagnostic) | 135 | -2.8 | -0.18 | -0.05 | 42% | -2.0 | |

Year by year, net in bp (2018 to June):

| | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EM | +1138 | -199 | +1300 | +623 | +17 | +717 | -210 | +512 | -467 | -701 | +891 | -237 |
| OG | +97 | -783 | +727 | +646 | -196 | +249 | -58 | +340 | -262 | -1076 | +407 | -157 |
| TR | +185 | -1797 | +1837 | +289 | -34 | +651 | -259 | +116 | -742 | -8 | +538 | -250 |

- **Economic momentum** is the strongest of the three and is not significant.
  - 2007-2012: +52 bp per month, t 1.69. 2013-2018: -3 bp, t -0.15.
  - Worst month -7.3%; maximum drawdown -18.7%.
  - Its monthly returns have a correlation of 0.71 with carry, so it overlaps the interest-rate trade.
- **Output gap and Taylor rule** are about zero, with large swings (the Taylor rule lost 18% in 2008 and made 18% in
  2009).
- **Costs are not the problem.** Monthly rebalancing costs about 2-3 bp a month; the gross returns are just as
  weak.

## ML (elastic net): the run 18/20 result was an artefact

ML gave exactly the same 99 monthly returns in run 18 (no macro data) and run 20 (all macro data): -5.2 bp, t -0.43.

The cause:
- ElasticNetCV usually shrinks every coefficient to zero, so the forecast is the same for every currency.
- The portfolio code then ranked the ties in a fixed arbitrary order.

Run 21 holds such months flat (paying the cost of closing), records every fit in `ml_fits.csv`, and publishes the
inputs (`inputs/`).

**Run 21 result (run 40212676021, branch `results21`).** In all 149 monthly fits (2010-04 to 2022-07), the
cross-validated elastic net shrank **every** coefficient to zero:
- l1 ratio 0.1;
- alpha between 0.006 and 0.021;
- no feature was ever selected: not EM, OG, TR, carry, 3- or 12-month momentum, or value.

Its forecast was therefore the same for every currency in every month, and the strategy never trades. On this
panel, cross-validation found no relation between the seven signals and next month's currency excess returns that
beats predicting the average. EM, OG and TR are unchanged in run 21.

## Conclusion

Macro fundamentals do not predict these 9 currencies against USD over 2007-2018 at a monthly horizon:
- **Economic momentum** is the best of the three published signals, at t 1.3. It is positive only in the first
  half of the sample, and it overlaps carry.
- **Output gap and Taylor-rule fundamentals** are about zero.
- **The elastic net** finds nothing in any month.

This agrees with the post-publication decay reported for currency anomalies (Bartram et al., JFQA 2025) and the
weak post-2008 performance of currency factors. The caveats:
- the data are revised figures, not real-time vintages, which if anything favours the signals;
- the sample is 135 months.
