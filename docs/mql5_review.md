# MQL5 review: articles, blogs, forum, Market and code

The question was whether MQL5 (articles, blogs, forum, Market EAs, signals) holds a strategy or edge the project
missed. Sources read are in `docs/mql5_review/sources.tsv` (77 rows: source, author, family, claim, evidence quality,
our status). The newest 155 article titles, classified, are in `docs/mql5_review/articles_newest_155.tsv`.

## 1. Scope and access

- **Size:** the English article section has 90 listing pages of 40 articles (about 3,600 articles). The forum,
  blogs, Market and Code Base are larger.
- **Access limits:**
  - mql5.com's robots.txt disallows automated crawling of `/en/articles`, so the indexer
    (`ci/mql5_articles_index.py`, run 40383328907) stopped without fetching.
  - The research fetch tool refused to page through the listing beyond a few pages.
  - Every page was reached through a search result instead.
- **Coverage method:**
  - Searched every strategy family against articles, blogs, the forum, the Market, signals, and Algo Forge
    (MQL5's git host).
  - Read every source found that reported a testable rule with numbers, plus the most-cited method articles.
  - Read about 100 distinct sources across the four reading rounds, plus the 155-title index.
- **Not read:** most of the roughly 3,450 older articles one by one, and the long forum threads in full (the
  machine-learning thread alone has more than 3,600 pages; one recent page was read).

## 2. What the articles are

Classification of the 155 newest articles (Sept-Oct 2026, by title):

| Type | Articles | Share |
|---|---|---|
| Neural networks, machine learning, LLM agents | 51 | 33% |
| Platform, UI and programming tutorials | 37 | 24% |
| Statistics and validation methods | 15 | 10% |
| Risk, portfolio and prop-firm tools | 13 | 8% |
| Signal processing and maths indicators | 11 | 7% |
| Named strategies / trading systems | 10 | 6% |
| Options, microstructure, market data | 7 | 5% |
| Optimisation algorithms | 7 | 5% |
| Other | 4 | 3% |

**Evidence quality in the strategy articles read** (about 35):
- Almost all report one backtest period, usually without costs and without an untouched out-of-sample period.
- Several reuse the "out-of-sample" period for model selection:
  - Dmitrievsky's regression meta-learner;
  - the ensemble mean-reversion article, which refits on the test years.
- Some test periods are very short:
  - the ~150-part "Neural Networks in Trading" series tests a few months (13 trades in one recent part);
  - the MQL5 Wizard series uses one year of optimisation and one year forward.
- **Authors with honest negative results** (each with nulls or holdouts):
  - Timon-pascal Krueger;
  - the Warrior_EA research kit;
  - Hammad Dilber (shapelets, VMD);
  - Roberto Riccio (block permutation);
  - Patrick Njoroge (meta-labelling).

## 3. Findings by family

| Family | Best MQL5 evidence | Our result | Verdict |
|---|---|---|---|
| Gotobi / Tokyo fix | 4 Market EAs (Nakane, Time ZeroShot, Gotobi Overnight JPY, Tokyo Fix Anomaly); Krueger: month-end / Gotobi / holiday rules +4.5-5.6 bp before commission (Apr 2023 - Jun 2026), failed his correction | Gotobi confirmed out of sample (+2.13 bp, t 2.05, 2019-2025) | Same edge; month-end and holiday rules untested |
| Night scalping, rollover | Vendor guides, no results | Round-4 night / rollover and Asian-range fade: negative | Nothing new |
| Breakouts (ORB, London, Asian box, gold) | ORB on USTEC: Sharpe 2.8-5.9 with no costs (readers could not reproduce); Krueger's replication on 5 indices: net zero; 288 + 400 gold variants: nothing robust | ORB, London breakout, Donchian: negative | Agrees |
| Exits on index intraday breakouts | Krueger: trailing stop +0.082 R per trade, t 16.4, against +0.010 R for a counter-candle exit (66,511 trades on DAX / FTSE / Dow / Nasdaq, 2015-2026, costs at entry) | Trailing exits add nothing on random FX entries | Fold into the intraday index momentum test |
| Mean reversion (Bollinger, z-score, Kalman, RSI) | M15 FX Bollinger: PF 0.8-1.2, no costs; z-score: one 2026 period | D1 Bollinger / z-score about +0.5, not significant; M15 negative after costs | Agrees |
| RSI(2) on US indices | PropDip: +48 bp per trade in the 2019-2026 holdout (t 3.6), 2 of 65 rules passed; Backtrex: PF 1.43, 1.7% a year | Tested only on FX (negative) | **Untested on indices: candidate** |
| Index-review closing auctions | PropFlow: FTSE +8.3 bp per event (2015-22), +11.5 bp holdout (2023-26, 22 of 27 positive); NYSE: auctions swell around rebalances | Never tested | **Candidate** |
| Fading retail candle setups | Warrior_EA: pin-bar / inside-bar fade +0.07-0.11 R gross against 0.14 R cost; 8 of 48 cells clear the spread (all EUR/USD), 6 of 8 hold walk-forward; commission not modelled | Candlestick patterns traded with the pattern: negative; the fade was not tested | Weak candidate |
| Order flow, tick imbalance | Warrior_EA: M5 order flow anti-predictive by ~1 pp, spread costs ~13 pp | Negative | Agrees |
| Machine learning | Short or reused test periods; Warrior_EA purged walk-forward GBM on SP500 / EURUSD / USDJPY: no edge; meta-labelling: edge 0.095 ATR against 0.099 ATR spread | No edge net of costs | Agrees |
| Statistical arbitrage, cointegration | Cointegrated-stocks series: zero-delay tests, author calls it not a real-world strategy | Negative | Agrees |
| Triangular / graph arbitrage | 2-month backtest, -48% drawdown, 80 ms emulated delay | Triangular parity: net -11.6 | Agrees |
| Seasonality (hour, weekday, month) | Boxplots, 75%-hit months, Monday PF; none tested out of sample or for multiple comparisons | Weekday-hour, turn-of-month, intraday seasonality: negative in FX | Agrees |
| Month-end FX flows | Brent Donnelly (FXStreet, May 2026): his month-end model stopped working in mid-2024 | t 1.4 after costs, about zero since 2011 | Agrees |
| Grid, martingale, averaging | Waka Waka (>40% drawdown 2024); 43 gold EAs: ~19% blew up; 10 best-selling scalpers: 80% of profit in the first week, 6 of 10 with blow-up reviews | Random-entry grids: negative at retail costs | Agrees |
| Swap arbitrage across brokers | Forum: ~8% a month on demo; brokers end swap-free status | Not an edge: broker terms | Excluded |
| Cross-account hedging at props | Banned at virtually every firm; accounts terminated | n/a | Excluded |
| Options, implied distributions | Monitoring tools only; one options article hit the same bad-IV data we repaired | 92 structures: no edge beyond beta / variance premium | Agrees |
| Commodity seasonality | (outside MQL5) 15 futures, OOS 2016-2024 net of 8.6 bp: no Holm-significant gain | Not yet tested | Deprioritise |

## 4. New candidates (rules as published, nothing tuned, 2026 locked)

| # | Test | Source claim | Data (free) | Why plausible |
|---|---|---|---|---|
| 1 | RSI(2) < 10 above the 200-day average, buy at the close, exit at the first close above the 5-day average or 20 days, on SPY / QQQ / DIA / IWM and US index CFDs | +48 bp holdout, t 3.6 | Yahoo daily 1993-2025; Dukascopy CFDs 2019-2025 | Most of its profit comes from weekend holds, the overnight / weekend equity premium we already found (t 3.0-3.8) |
| 2 | FTSE 100 quarterly review days (third Friday of Mar / Jun / Sep / Dec): short late session, cover after the closing auction | +8-11 bp per event | Dukascopy UK100 minute data | Index funds must trade the close: forced flow |
| 3 | TLT month-end: long the last 5 trading days | Sharpe ~1 2002-2025, flat 2023-24 | Yahoo daily | Pension and index-extension rebalancing |
| 4 | Tokyo month-end and Japanese-holiday rules on USD/JPY, plus the post-fix short leg (09:55-10:30 JST) | +4.5-5.6 bp before commission | Dukascopy USDJPY M1 (have) | Same flow as Gotobi |
| 5 | Index intraday momentum: first-half-hour → last-half-hour, the noise-band breakout, and the breakout with breakeven + trailing exit, on US500 / NAS100 / DAX / FTSE | Trailing +0.082 R, t 16 (exit comparison); last half-hour R² 0.014 post-sample | Dukascopy index minute data | Hedging-flow intraday momentum (Gao et al.; Baltussen et al.) |
| 6 | Fade pin-bar / inside-bar setups on EUR/USD at raw-ECN costs | Clears spread only | Dukascopy (have) | Short-horizon reversion is real but small (projection lab) |

These would join the earlier proposed batch (Gotobi EA variants, night scalping at ECN costs, gold London breakout,
gold multi-length channel, BTC 14/56, USDJPY home vs away, and the free batch) under one multiple-testing correction.

## 5. Conclusion

- MQL5 contains no published strategy with an untouched, cost-inclusive out-of-sample record beyond the effects this
  project already has:
  - Gotobi;
  - the overnight / weekend equity premium;
  - the variance premium.
- The two strongest new leads come from one careful vendor and are forced-flow or calendar effects, not
  indicator patterns:
  - RSI(2) dip-buying on US indices;
  - FTSE index-review auctions.
- The rigorous MQL5 researchers reach the project's conclusions independently:
  - technical-indicator, pattern and machine-learning rules have no edge after costs;
  - costs and pseudo-replication explain most published edges.
