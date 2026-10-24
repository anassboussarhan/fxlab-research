# Is there a projection of the price that is not random?

Script: `scripts/predictable_projections.py`. Results are in `results/reports/projections_usd/`, plus
`projections_all/` for the version that includes the crosses.

Setup:
- D1 and H4 bars, 2007-2018 development data.
- Walk-forward with 3-year training and 6-month test windows.
- Costs from the platform's vector engine.
- 20 placebos per configuration. A placebo fits the same pipeline after breaking the predictor-to-target link
  inside the training window, then trades the real test data.

## Projections searched

| Projection | Idea | Result (out of sample) |
|---|---|---|
| **Maximally predictable portfolio** (Lo & MacKinlay 1997): rank-1 canonical correlation between all lagged 1/5/20-bar returns and the next h-bar returns of a currency basket, with shrinkage chosen on an inner hold-out | The basket whose direction the whole market's recent history predicts best | Best: D1, h = 5 bars, gross Sharpe 0.46 vs placebo 0.08 +- 0.24 (1 of 20 placebos higher), net -0.05. Median OOS information coefficient 0.03, i.e. R^2 ~ 0.001. The other 5 horizons are at or below placebo. Not significant after testing 6 horizons. |
| **Most mean-reverting basket of price levels** (Box & Tiao 1977, d'Aspremont 2011): the VAR(1) combination with the smallest predictability ratio | A combination of currencies that is stationary | **With the crosses in the universe: gross Sharpe 1.56 (D1) and 1.17 (H4), 0 of 20 random-basket placebos as high, net -11.6 and -13.2.** The basket found is the triangular identity log EURJPY - log EURUSD - log USDJPY ~ 0, and its predictability ratio is 0.000. The "mean reversion" is mid-quote asynchrony inside the spread: about +0.17%/yr gross, about -3.5%/yr after costs. It is real but untradable. **With USD pairs only (no identities):** the best basket has a predictability ratio of 0.77 to 0.89 (far from mean-reverting) and a gross Sharpe of -0.02 / -0.21, below the placebos. |
| **Volatility projection** \|r\| (HAR model of log realised variance) | How much price will move, not where | **Clearly not random:** OOS R^2 is 0.29-0.48 across the 7 USD pairs (EURCHF 0.74 over the 2011-2015 floor period). It carries no direction. |
| **Calendar projection** (gotobi, Tokyo fixing days; see `results/preregistration/gotobi.md`) | Price conditioned on known Japanese settlement days | The only directional projection found that beats its placebos: USDJPY H1 Sharpe 1.05 vs placebo calendars -0.48 to -1.17, with 8 of 9 years positive. Not yet confirmed on post-2018 data. |

## What this means

- The level and direction of the price behave as a martingale in every linear projection of the currency
  space tested. The best maximally predictable basket explains about 0.1% of the variance of its future
  return. That is a gross Sharpe near 0.5 at best, which retail costs consume.
- A projection that looks spectacularly non-random (Sharpe 1.56, beating every placebo) can be an accounting
  identity of the quotes. Any cross-rate search has to remove those identities first.
- Two projections are genuinely non-random:
  - **Volatility.** It is about 40% predictable out of sample, but monetising it needs options (straddles,
    variance swaps), which this platform has no data or costs for. Alternatively, it can be used as risk
    sizing on top of a directional edge.
  - **The calendar** (gotobi). It is directional, small, and still unconfirmed on new data.
