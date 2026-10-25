# What the literature says works in FX, and how it compares with these results

The question: which published FX effects work, whether they agree with the findings here (2007-2018 data,
EURUSD to 2022; retail ECN costs; walk-forward; multiple-testing control), and which ones were not tested.

## 1. Agreement

| Literature | What it found | Results here |
|---|---|---|
| Meese & Rogoff; Rossi (2013) survey | The random walk is the hardest benchmark for exchange-rate forecasts. Predictability is conditional, unstable across countries and samples, and found mainly with Taylor-rule and net-foreign-asset fundamentals. | Directional predictability exists but is tiny: hourly AUC 0.539; projection-lab moves are 0.5-2% of the cost. |
| Neely, Weller & Ulrich (2009) | Technical-rule profits in FX were genuine in the 1970s-80s and vanished by the early 1990s. Example: a 1% filter rule made about 8.5% a year in 1976-1990 and -1.1% in 1991-2005. | 97 strategy families on 2007-2018: median net Sharpe -0.32, 0 BH survivors. |
| Hsu, Taylor & Wang (2016): 21,195 rules, 1971-2015, data-snooping-robust | No reliable rules in developed currencies after the 1992-96 sub-period. Emerging currencies stay more predictable, at about 4.5x the cost. | Same for developed currencies and crosses. Emerging currencies were not tested (no data). |
| Hutchinson et al. (2022) | After publication, currency carry, momentum and value strategies stopped working: average out-of-sample Sharpe fell from +0.39 to -0.32. | The carry test here was negative over 2007-2018. |
| Fan et al. (2022) | Carry trade performance declined after the 2008 crisis. | The period tested here is the weak, post-crisis one. |
| Kaminski & Lo (2014) | Under a random walk, a stop-loss "will always reduce the portfolio's expected return". Stop-losses add value only with momentum and hurt with mean reversion. | The optional-stopping argument used for random entries, and its empirical pattern: trailing stops and pyramids lose even before costs, grids and averaging are slightly positive before costs, because FX swings slightly mean-revert. |
| Tharp / Basso "coin-flip" study (1990s): random entries, 3x ATR trailing stop, 10 trend-prone futures | Profitable. | It harvested the strong futures trends of that era, which is the momentum case in Kaminski-Lo. A 2014-16 FX retest was 2.5 years, 20 manual runs and had no cost analysis. On 2010-2018 FX, trailing exits on random entries lose even before costs. |
| Moreira & Muir; Cederburg et al. (2020): 103 strategies | Volatility-managed portfolios mostly fail out of sample. | Volatility and spread filters did not help: the smart family -0.37 vs bracket -0.39. |
| Harvey, Liu & Zhu (2016); Bailey et al. (backtest overfitting) | New effects should clear t > 3; "most claimed research findings in financial economics are likely false". | The best result here has t 0.9. Several apparent wins were simulator errors. |
| Meng & Khushi (2019) review of reinforcement learning in FX and stocks | "All reviewed articles had some unrealistic assumptions such as no transaction costs"; costs significantly hurt profitability. | ML / RL predicts slightly but does not pay after costs. |
| ESMA (2018); NFA / broker disclosures | 74-89% of retail CFD accounts lose money; 72% of retail forex accounts are unprofitable; profitable accounts rarely exceed 30%. | Retail-cost expectancy is negative. |

## 2. Where the literature qualifies the conclusion

**Historical periods.** Carry and momentum earned real returns in earlier periods, net of costs:
- Accominotti & Chambers, 1920-39: carry about 0.38 net Sharpe, momentum about 0.47.
- Menkhoff et al.: momentum spreads up to about 10% a year, with strong limits to arbitrage.
- Moskowitz, Ooi & Pedersen: time-series momentum at 1-12 months across 58 futures.

So "nothing works" is a statement about developed-currency FX in 2007-2018 at retail costs, not about all FX.

**Emerging-market currencies.** These keep more predictability (Hsu, Taylor & Wang) and are not in this data
(USDMXN, USDZAR, USDTRY, USDCNH unavailable).

**Options-based signals.** Currency volatility risk premia predict currency returns (Della Corte, Ramadorai &
Sarno 2016). This needs option implied volatilities, which this platform does not have.

## 3. Documented effects not yet tested in their published form

These are effects driven by flows rather than price patterns. They have a mechanism, and so are the most credible
candidates:
- **Gotobi / Tokyo fixing.** Ito & Yamada (2017) find predictable order imbalances and biased fixing prices around
  the 9:55 JST Tokyo fix, with calendar effects. A 2023 study finds USD/JPY rising into 9:55 on Gotobi days,
  profitable net of spreads in 2018-2020 (in sample, small).
  - This supports the strongest lead here: Gotobi USDJPY, H1 Sharpe about 1.0 over 2010-2018. That lead was tested
    as a generic calendar rule, not in the published 9:55 JST window.
- **Month-end hedging flows at the London 4 p.m. fix.** Melvin & Prins (2015): equity gains over the month predict
  currency depreciation before the month-end fix (international equity hedging).
- **Local-hours depreciation.** Breedon & Ranaldo: currencies tend to depreciate during their local trading
  hours, and a simple EUR/USD strategy is reported profitable.
- **Canonical cross-sectional currency factors.** Monthly carry, momentum, value and dollar portfolios across
  currencies. Carry and cross-asset rules were tested, but not as a replication of the published monthly
  portfolios. The literature expects weak post-2008 results.

## Sources
- Neely, Weller & Ulrich, The Adaptive Markets Hypothesis: Evidence from the FX Market: https://files.stlouisfed.org/files/htdocs/wp/2006/2006-046.pdf
- Hsu, Taylor & Wang, Technical trading: is it still beating the FX market? https://profiles.wustl.edu/en/publications/technical-trading-is-it-still-beating-the-foreign-exchange-market/ (summary: https://cxoadvisory.com/technical-trading/updated-comprehensive-long-term-test-of-technical-currency-trading)
- Menkhoff, Sarno, Schmeling & Schrimpf, Currency momentum strategies: https://www.bis.org/publ/work366.pdf
- Accominotti & Chambers, Out-of-sample evidence on the returns to currency trading: https://repec.cepr.org/repec/cpr/ceprdp/DP9852.pdf
- Hutchinson et al., Are carry, momentum and value still there in currencies? https://repository.adu.ac.ae/items/49b11e04-d4e2-4397-96aa-dac0c017647b/full
- Fan et al., Currency carry trade: the decline in performance after the 2008 GFC: https://ideas.repec.org/a/eee/intfin/v76y2022ics1042443121001670.html
- Moskowitz, Ooi & Pedersen, Time series momentum: https://papers.ssrn.com/abstract=2089463
- Kaminski & Lo, When do stop-loss rules stop losses? https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf
- Tom Basso / Van Tharp random-entry study, FX retest: https://www.nasdaq.com/articles/revisiting-tom-basso-how-important-your-entry-forex-trading-part-1-2016-11-05
- Cederburg et al., On the performance of volatility-managed portfolios (summary): https://alphaarchitect.com/does-portfolio-timing-based-on-volatility-signals-outperform-buy-and-hold/
- Rossi, Exchange rate predictability: https://econ-papers.upf.edu/papers/1369.pdf
- Harvey, Liu & Zhu, ... and the cross-section of expected returns: https://papers.ssrn.com/abstract=2513152
- Bailey, Borwein, Lopez de Prado & Zhu, The probability of backtest overfitting: https://papers.ssrn.com/abstract=2326253
- Meng & Khushi, Reinforcement learning in financial markets: https://ideas.repec.org/a/gam/jdataj/v4y2019i3p110-d252469.html
- ESMA product intervention (74-89% of retail accounts lose): https://www.esma.europa.eu/node/84933
- NFA / broker disclosure figures (Zweig): https://jasonzweig.com/?p=2530
- Ito & Yamada, Puzzles in the Forex Tokyo fixing: https://www.nber.org/papers/w22820
- Gotobi anomaly strategy (2023): https://arxiv.org/abs/2301.13204
- Melvin & Prins, Equity hedging and exchange rates at the London 4 p.m. fix: https://ideas.repec.org/a/eee/finmar/v22y2015icp50-72.html
- Breedon & Ranaldo, Intraday patterns in FX returns and order flow: https://www.snb.ch/en/publications/research/working-papers/2011/working_paper_2011_04
- Della Corte, Ramadorai & Sarno, Volatility risk premia and exchange rate predictability: https://openaccess.city.ac.uk/id/eprint/13156/
