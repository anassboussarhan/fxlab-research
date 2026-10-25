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

## 4. Round 2 (2026-10-25): a deeper search, and what runs 17-21 found

The question this time: what has been shown to work in FX and precious metals out of sample, after costs and
after publication? Each finding is set against the results here.

### 4.1 What the newer evidence says has stopped working

| Study | Finding | Results here |
|---|---|---|
| Kurth, Eisler, Rej & Bouchaud (2026), "Is Trend Still Your Friend?", ~101 futures 1995-2025 | Trend following broke around 2008/9. For currencies and equity indices "trend has effectively vanished". Full-universe Sharpe of a fast EWM-5-20 trend fell from 0.84 to 0.12; the slowest (50-200) from 0.70 to 0.40. Mechanism offered: after 2008, high-frequency market makers withdrew liquidity in front of predictable trend-follower flow on small-tick contracts. | Every trend family here was negative after costs on 2007-2018; random entries with trailing stops lose even before costs. |
| Aalto thesis (2025), 32 currencies 2010-2025 | Cross-sectional currency momentum: none of 25 formation/holding combinations is significant after the GFC. | Run 20: 12-month FX momentum +2.9 bp a month, t 0.14. |
| Bartram, Djuranovik, Garratt & Xu (JFQA 2025) | Currency anomalies decline after publication, more so where in-sample profits were high and arbitrage cheap; "signal ranks and alphas decay quickly". | FOMC days (t 3.0 until 2010, 0.15 after) and local trading hours fit this pattern. |
| FCA Occasional Paper 46 | Price reversals around the WM/Reuters 4 p.m. fix were significant in 2012-2014, and "disappear from 2015 onwards" after the fix window was extended to 5 minutes. | W1 fix reversal: real before costs (gross t 2.9) on data that ends mostly in 2018 for the crosses and in 2022 for EUR/USD; far below costs. |
| Krohn, Mueller & Whelan (2021), 1999-2019 | The dollar rises before the Tokyo, ECB and London fixes and falls after: about 2.5 bp around each fix. Tokyo reversal 8-12% a year before costs; **negative with full quoted spreads**, positive only at half spreads. | K1-K3: positive gross, -4 to -6 bp a day net. Same conclusion. |
| Evidence Investor / HFRX carry indices | Currency carry has not been attractive since 2010; G10 carry drew down 29% in 2008. | C1 / C2 carry about zero over 2007-2022. |

### 4.2 What the newer evidence says still works, and whether it was tested here

| Effect | Evidence | Status here |
|---|---|---|
| **Gotobi / Tokyo fix** | Ito & Yamada (2017); the 2023 study; practitioner reports. | **Tested (run 19): +3.4 bp per trade, t 2.9, survives BH.** |
| **Selling FX volatility / variance** | Implied above realised volatility; Della Corte, Ramadorai & Sarno (2016). Cross-section of FX straddles (City University, 1999-2016): low-minus-high implied-vol straddles earn 38.9 bp a month on notional after quoted costs, Sharpe 1.03. Implied volatility *level* predicts straddle returns; implied minus realised does not, but it does predict spot. | **Tested (V1): +1.31 vol points a month, t 5.4, 2008-2014.** The straddle cross-section needs option prices for several currencies; not available here. |
| **Dollar before FOMC rate changes** | Karnaukh (1994-2015): the dollar moves in the direction of the coming Fed decision in the 2 days before. When fed funds futures signal a change: 25 bp per trade, Sharpe 0.93; about 15 bp and Sharpe 0.57 after costs. Before no-change meetings the move is about zero. | **Not testable as published:** it needs fed funds futures prices, which are not available here. The P1 / P2 test (announcement day, all meetings) is a different test. |
| **Dollar carry** (Lustig, Roussanov & Verdelhan) | Long the foreign basket when the average foreign rate is above the US rate, else short; 5.6% a year, Sharpe 0.66, 1983-2010. | **Not yet tested:** run 22. |
| **Factor momentum in currencies** (Zhang 2021, 1976-2020) | Carry and dollar factor returns are autocorrelated: they pay after gains and not after losses. | **Not yet tested:** run 22. |
| **Short-term (1-4 week) currency momentum** (Raza, Marshall & Visaltanachoti) | 63 currencies 1997-2011, gross only: 1.9-9.1% a year, Sharpe up to 1.05, weaker after 2004. | **Not yet tested as weekly cross-sectional momentum:** run 22. |
| **Silver fix (noon London)** | Caminschi (2014), 2000-2013: the fix return averages -8.3 bp (daily +3.8 bp); 2.3 falling fixes per rising one; 15 minutes before and after the fix -4.8 and -4.3 bp after 2007. | **Not yet tested:** run 22, XAG/USD 2010-2018 (the silver fix became an auction in August 2014). |
| **Gold PM fix (15:00 London)** | Abrantes-Metz & Metz (2014): large moves during the PM fix were mostly down (two-thirds of the time 2004-2013, 92% in 2010). Caminschi & Heaney (2014): the opening minutes of the fix predicted its direction. The LBMA's rebuttal disputes the manipulation reading, not the statistics. | **Not yet tested:** run 22, XAU/USD 2011-2018 (an electronic auction since March 2015). |
| **Gold rises in Asian hours, falls in Western hours** | Gold returns accrue overnight, with negative trading-day returns (J. Economics & Finance 2018, "economically important even with transaction costs"). CBS thesis (2019): "hat-shaped" intraday pattern; Sharpe up to 1.61 before costs, weaker after. | **Not yet tested:** run 22. |
| **Gold seasonality** (Baur 2012, 1981-2010) | September (+2.2%) and November (+1.8%) are significantly positive. | **Not yet tested:** run 22 (only 7 Septembers in the data, so a diagnostic). |
| **News classified by a language model** (AIFX, 1996-2024) | Gross Sharpe about 0.5-0.6 for a monthly G10 cross-section. | Not testable: needs a news archive. |
| **Retail order flow (OANDA)** | Informative for short-term forecasts. | Not testable: no order-flow data. |
| **CFTC positioning (contrarian)** | Currency futures: "marginally positive returns without statistical significance". | Not tested; weak evidence. |
| **Skewness risk premium** (Li, Sarno & Zinna 2025) | Mostly a carry-like premium with crash risk, strongest in emerging markets. | Needs option skews; not available. |

### 4.3 What this means

- The newer literature agrees with the results here. **Price-based rules (trend, momentum, technical, intraday
  reversal) stopped paying in developed FX around 2008-2010.** The fix reversals that remained were removed by
  the 2015 fix reform, and most published anomalies decay after publication.
- **What survives in the literature is mostly:**
  - flows tied to the calendar or a benchmark: Gotobi, metals fixes, possibly month-end;
  - risk premia for bearing crash risk: variance selling, skew;
  - macro-expectation trades that need data this platform lacks: fed funds futures, option prices, news.
- **Run 22 tests the remaining candidates that the data here can test:**
  - dollar carry;
  - carry and dollar factor momentum;
  - weekly cross-sectional momentum;
  - the silver and gold fixes;
  - gold's Asia / West split;
  - gold seasonality.

## Sources
- Neely, Weller & Ulrich, The Adaptive Markets Hypothesis: Evidence from the FX Market: https://files.stlouisfed.org/files/htdocs/wp/2006/2006-046.pdf
- Hsu, Taylor & Wang, Technical trading: is it still beating the FX market? https://profiles.wustl.edu/en/publications/technical-trading-is-it-still-beating-the-foreign-exchange-market/ (summary: https://cxoadvisory.com/technical-trading/updated-comprehensive-long-term-test-of-technical-currency-trading)
- Menkhoff, Sarno, Schmeling & Schrimpf, Currency momentum strategies: https://www.bis.org/publ/work366.pdf
- Accominotti & Chambers, Out-of-sample evidence on the returns to currency trading: https://repec.cepr.org/repec/cpr/ceprdp/DP9852.pdf
- Hutchinson et al., Are carry, momentum and value still there in currencies? https://repository.adu.ac.ae/items/49b11e04-d4e2-4397-96aa-dac0c017647b/full
- Fan et al., Currency carry trade: the decline in performance after the 2008 GFC: https://ideas.repec.org/a/eee/intfin/v76y2022ics1042443121001670.html
- Moskowitz, Ooi & Pedersen, Time series momentum: https://papers.ssrn.com/abstract=2089463
- Kurth, Eisler, Rej & Bouchaud, Is Trend Still Your Friend? (2026): https://arxiv.org/pdf/2607.01550
- Momentary loss: the disappearance of cross-sectional currency momentum (Aalto, 2025): https://aaltodoc.aalto.fi/items/0db57fa9-e5c6-4b7d-9dd8-1ae15a8940c6
- Bartram, Djuranovik, Garratt & Xu, Mispricing and Risk Premia in Currency Markets (JFQA 2025): https://ideas.repec.org/a/cup/jfinqa/v60y2025i2p695-733_5.html
- FCA Occasional Paper 46 (WM/Reuters fix): https://www.fca.org.uk/publication/occasional-papers/occasional-paper-46.pdf
- Krohn, Mueller & Whelan, Foreign Exchange Fixings and Returns Around the Clock: https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf
- Karnaukh, The Dollar Ahead of FOMC Target Rate Changes: https://acfr.aut.ac.nz/__data/assets/pdf_file/0019/190315/DollarAheadFOMC_NinaKarnaukh.pdf
- Lustig, Roussanov & Verdelhan, Countercyclical Currency Risk Premia (dollar carry; summary): https://quantpedia.com/strategies/dollar-carry-trade
- Zhang, Dissecting Currency Momentum (summary): https://alphaarchitect.com/is-currency-momentum-factor-momentum/
- Raza, Marshall & Visaltanachoti, weekly currency momentum (summary): https://www.cxoadvisory.com/momentum-investing/short-term-currency-exchange-rate-momentum
- Cross section of FX options returns and volatility (City University): https://openaccess.city.ac.uk/24959/1/Cross%20section%20of%20FX%20options%20returns%20and%20volatility_Final.pdf
- Caminschi, Any Silver Linings? (silver fix, 2014): https://www.stern.nyu.edu/sites/default/files/assets/documents/Any%20Silver%20Linings%20-%20WORKING%20PAPER%20-%2020140707%20-%20Merged.pdf
- Caminschi & Heaney, gold fix (J. Futures Markets 2014): https://ideas.repec.org/a/wly/jfutmk/v34y2014i11p1003-1039.html
- LBMA Alchemist, PM gold fix study and rebuttal: https://www.lbma.org.uk/alchemist/issue-73/has-there-been-a-decade-of-london-pm-gold-fixing-manipulation
- Gold overnight returns (J. Economics & Finance 2018): https://ideas.repec.org/a/spr/jecfin/v42y2018i3d10.1007_s12197-017-9403-0.html
- Gold price dynamics around the clock (CBS thesis 2019): https://research.cbs.dk/en/studentProjects/gold-price-dynamics-around-the-clock/
- Baur, gold seasonality (summary): https://cxoadvisory.com/calendar-effects/gold-seasonality-drivers
- AI and Exchange Rate Predictability (2026): https://arxiv.org/pdf/2608.00761
- Retail order flow (OANDA): https://eprints.lancs.ac.uk/id/eprint/71649
- CFTC positioning thesis (Aalto): https://aaltodoc.aalto.fi/items/b84b07d6-3bd2-4040-bbcf-052b938b1f74
- Skewness risk premia (Cambridge summary): https://www.jbs.cam.ac.uk/centres/ceam/research/the-cambridge-foreign-exchange-initiative/fx-initiative-research/skewness-risk-premia-and-the-cross-section-of-currency-returns/
- Carry since 2010 (Evidence Investor): https://www.evidenceinvestor.com/post/does-carry-trading-work
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
