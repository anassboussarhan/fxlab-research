# Can a retail trader make money with options?

Question raised: profitable options trading for a retail trader, not only FX prop trading.
- Script: `scripts/options_lab.py`; workflow `options-lab`; results on branch `options-lab`.
- Free Yahoo data fetched on the runner and cut at 2025-12-31.

## 1. What the research says

| Study | Finding |
|---|---|
| Bryzgalova, Pavlova & Sikorskaya (J. Finance 2023), Nov 2019 - Jun 2021 | US retail option traders lost **$1.14 billion** (10-day holding assumption) plus **$4.13 billion** in execution costs; very wide spreads in the options they prefer |
| de Silva, Smith & So, "Losing is optional" (32,791 earnings announcements, 2010-2021) | Retail buys options before earnings: overpays for volatility, pays spreads of about 9-10% of the investment, holds too long. Average loss 5-9%, 10-14% around high-volatility announcements |
| Beckmeyer, Branger & Gayda, "Retail traders love 0DTE options… but should they?" (SPX options, 2021-2023) | Studies whether retail 0DTE traders should expect to make money; the summary used here does not report the figures (paywalled) |
| Option Alpha (1,546 earnings events, 40 underlyings, 10 years, 648,861 simulated trades) | Selling premium into earnings won 57% of the time but had a **negative average P&L** (tail moves); the author stopped trading earnings |
| Bondarenko, "Why are put options so expensive?" (S&P futures options 1987-2000) | Selling one-month puts earned very large average returns (ATM about 39% a month on premium); no reasonable risk model explains it; buyers transferred about $18bn to sellers |
| Israelov & Nielsen, "Covered calls uncovered" (FAJ 2015) | A covered call = equity + short volatility + an uncompensated market-timing bet. The short-volatility part has a Sharpe ratio near 1.0 and adds about 2% a year |
| Cboe-commissioned study, Black & Szado (1986-2015) | Index options-selling benchmarks (BXM, PUT, CNDR, BFLY, …) had returns similar to the S&P 500 with lower volatility and drawdowns |
| Cboe PUT index factsheet (2007 - Sep 2026) | Put-writing **7.2%** a year, volatility 10.7%, max drawdown -32.7%, Sharpe 0.52; S&P 500 TR **11.0%**, 15.4%, -50.9%, 0.61 |
| Muravyev & Pearson (RFS 2020) | Effective option spreads are about a quarter below conventional estimates, and below 40% of them for traders who time their executions. Patient execution matters. |
| Volmageddon (FAJ 2021) | On 5 February 2018, short-volatility ETPs (XIV, SVXY) lost **more than 90% in a day** through a feedback loop of forced VIX-futures buying |

In short:
- **Buying options** is where retail loses most: overpriced volatility, wide spreads and bad timing.
- **Selling options** earns a real, well-documented premium, the variance risk premium.
- But selling is short crash insurance. Gains are steady, the occasional loss is enormous, and leveraged sellers can be wiped out in a day.

## 2. Test on free data (run 40217982809, branch `options-lab`)

**Data:**
- Yahoo: S&P 500 (1986-2025), S&P 500 total return (1988-), VIX (1990-), VIX3M (2006-) and the Cboe PUT
  put-write index (1996-).
- Yahoo has no data for BXM, BXMD, CNDR, BFLY, WPUT, PPUT, CLL or CMBO, so the iron-condor test O5 could not run.

**Short S&P variance at the VIX:** a proxy for selling one-month S&P options and delta-hedging them.
- Monthly P&L in vol points of variance notional: (K^2 - RV^2) / 2K.
- K is the VIX at the previous month's close; RV is the month's realised volatility of daily S&P returns.
- 1 vol point of cost a month.

| Test | Months | Mean, vol pt / month | t | Sharpe | Months + | Years + | Worst month | 2008 | 2020 | BH q |
|---|---|---|---|---|---|---|---|---|---|---|
| **O1** always, 1990-2025 | 431 | **+1.82** | **4.68** | 0.78 | 82% | 86% | -87.7 (Mar 2020) | -122 | -33 | < 0.001 |
| O1 at 0.5 vol pt cost | 431 | +2.32 | 5.96 | 1.00 | 83% | 92% | -87.2 | -116 | -27 | |
| O1 before costs | 431 | +2.82 | 7.25 | 1.21 | 84% | 92% | -86.7 | -110 | -21 | |
| **O2** only when VIX > past 21-day realised vol | 431 | +1.73 | 4.94 | 0.83 | 74% | 78% | -87.7 (Mar 2020) | -62 | -51 | < 0.001 |
| **O3** only in contango (VIX < VIX3M), 2006-2025 | 233 | +1.63 | 4.02 | **0.91** | 71% | 80% | -62.1 (Sep 2008) | -59 | **+40** | < 0.001 |
| O1 on the same 2006-2025 sample (reference) | 233 | +1.04 | 1.56 | 0.35 | 80% | 80% | -87.7 | -122 | -33 | |

Worst days of the short-variance position, in vol points:

| Day | Loss |
|---|---|
| 16 Mar 2020 | -43.1 |
| 5 Feb 2018 (Volmageddon) | -42.0 |
| 12 Mar 2020 | -33.2 |
| 4 Apr 2025 | -30.2 |
| 8 Aug 2011 | -28.2 |
| 5 Aug 2024 | -27.3 |
| 29 Sep 2008 | -24.4 |

Year by year (vol points): positive in 31 of 36 years. The losing years are:

| Year | 2008 | 2018 | 2020 | 2025 | 2002 |
|---|---|---|---|---|---|
| Vol points | -122 | -31 | -33 | -6 | -0.4 |

**Cboe PUT index** (cash-secured monthly ATM put writing, 1996-2025), against the S&P 500 total return
(1988-2025):

| | Return a year | Volatility | Sharpe over cash | Max drawdown | 2008 | 2020 | 2022 |
|---|---|---|---|---|---|---|---|
| PUT | 8.5% | 10.6% | 0.70 | -32.7% | -27% | +2% | -8% |
| S&P 500 TR | 11.4% | 14.6% | 0.75 | -50.9% | -37% | +18% | -18% |

- **O4:** PUT's alpha over its S&P beta (0.59) is **+2.3% a year, t 2.16**. The premium for selling puts is
  significant even after the equity exposure, but its Sharpe ratio is no better than simply owning the index.

**A small account selling variance**, every rolling 5-year window 1990-2025 (372 windows). Size is the account
fraction per vol point of variance notional:

| Size | Mean month | Worst month | Median 5-year multiple | 10th percentile | 90th percentile | Windows that lost half |
|---|---|---|---|---|---|---|
| 0.1% | +0.2% | -8.8% | 1.12x | 1.00x | 1.20x | 0% |
| 0.25% | +0.5% | -21.9% | 1.33x | 0.98x | 1.58x | 0% |
| 0.5% | +0.9% | -43.8% | 1.75x | 0.88x | 2.48x | 7% |
| 1% | +1.8% | **-87.7%** | **2.92x** | **0.40x** | 5.92x | **32%** |

## 3. What this means for a retail trader

1. **Selling S&P options premium is a real edge.** It earns about +1.8 vol points a month after realistic costs
   (t 4.7), in 86% of years since 1990. This is the strongest, most persistent premium found in this whole
   project.
2. **It pays for taking crash risk, and the crashes are huge.**
   - March 2020 took 88 vol points in a month, 2008 took 122 in a year, and Volmageddon took 42 in one day.
   - Sized to "gain a lot" (1% per vol point, about +1.8% a month), a third of all 5-year periods would have halved
     the account at some point, and one month cost 88%.
3. **Simple, pre-defined filters cut the crash damage.** Selling only when the VIX term structure is in contango
   (O3) skipped March 2020 (+40 vol points that year instead of -33) and raised the Sharpe ratio from 0.35 to 0.91
   over 2006-2025. It did not avoid September 2008.
4. **Cash-secured put writing** (the PUT index) earned 8.5% a year at two-thirds of the S&P's volatility. Since 1996
   it has been a lower-risk way to own the equity premium, not a way to get rich.
5. **Retail implementation:**
   - Defined-risk positions (put credit spreads, iron condors on SPX / XSP / SPY) cap the crash loss, at the cost
     of part of the premium.
   - Patient limit orders matter: effective option spreads can be under 40% of quoted ones (Muravyev & Pearson).
   - Avoid buying options and avoid earnings trades. Those are the documented retail losers.
6. **Capital:** none of the retail evaluation prop firms checked here allow it (no options, or flat every night). A
   defined-risk spread on XSP (one tenth of SPX) risks a few hundred dollars per contract, so it is possible with a
   small account of your own. With a small account, only small sizes survive the crashes, so the returns in dollars
   are small.

**Caveats:**
- A variance swap is a proxy: retail traders sell strangles or spreads, not variance swaps.
- The 1 vol point cost is an estimate.
- The Cboe indices assume their own execution convention.
- None of this was tested on 2026 data.

## Sources
- Bryzgalova, Pavlova & Sikorskaya (summary): https://www.fa-mag.com/news/mom-and-pop-investors-took-a-billion-dollar-bath-trading-options-during-the-pandemic-67571.html?section=43
- de Silva, Smith & So, "Losing is optional" (MIT summary): https://mitsloan.mit.edu/ideas-made-to-matter/retail-investors-lose-big-options-markets-research-shows
- Beckmeyer, Branger & Gayda, 0DTE: https://cxoadvisory.com/individual-investing/retail-0dte-option-trader-performance
- Option Alpha earnings research: https://optionalpha.com/we-stopped-trading-earnings-after-we-saw-this-new-research-188320.html
- Bondarenko, Why are put options so expensive? (summary): https://www.cxoadvisory.com/equity-options/is-40-per-month-shorting-index-puts-a-fair-return/
- Israelov & Nielsen, Covered calls uncovered (summary): https://quantpedia.com/covered-calls-uncovered/
- Black & Szado, Cboe options-selling indexes: https://www.hedgeweek.com/new-study-analyses-performance-cboe-sp-500-options-selling-indexes/
- Cboe PUT index factsheet: https://cdn.cboe.com/resources/indices/factsheet/CboeGlobalIndices_PUT-Index.pdf
- Muravyev & Pearson, Options trading costs are lower than you think: https://ideas.repec.org/a/oup/rfinst/v33y2020i11p4973-5014..html
- Volmageddon (FAJ 2021): https://rpc.cfainstitute.org/research/financial-analysts-journal/2021/volmageddon-failure-short-volatility-products
