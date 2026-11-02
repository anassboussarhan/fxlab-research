# Pre-registration: the edges batch (2026-11-02)

This document fixes the rules of the batch before any of its data is downloaded or read.

**Status when written.**
- The data scripts are `ci/edges_data.py` and `.github/workflows/edges_data.yml`. They had not run yet.
- The panel data (run 41447353092) was still downloading.

**Rules that apply to every test.**
- Development data only, to 2025-12-31. Nothing from 2026 is used.
- M1 bid/ask fills and costs as in the other labs.
- **Split:** 2007-2018 for train and 2019-2025 for test. A series that starts later is split at its first 60% of
  events (fit) and the last 40% (check).
- **Direction of each trade:** the sign on train, unless a published paper fixes it, as noted below.
- **A test counts only if all of these hold on the check period:**
  1. the net mean after costs is above 0;
  2. the one-sided t is at least 1.645;
  3. it survives Benjamini-Hochberg at q = 0.10 over every test in this batch together with
     `docs/panel_hedge_preregistration.md`.
- **What happens to a test that counts:** it goes through the FTMO 2-Step simulation in `scripts/prop_best.py`. It
  is then frozen as a forward rule starting the night after it is frozen, next to the existing frozen rules and
  never replacing them.
- **Missing data:** a source that cannot be downloaded is reported as "untestable". It is never replaced by a
  guess.

## Improving the Gotobi trade

- **E1. Sizing by the filter's probability.**
  - The forward-v1 walk-forward probabilities p (refitted yearly, as in `scripts/gotobi_filter_wf.py`) set the
    size: weight = p / mean(p on the training years), capped at 2.
  - This is compared with no filter and with the skip-the-lowest-third rule, on 2012-2025 walk-forward nights.
  - **Counts if:** FTMO 2-Step 5x P(paid) at the 2026-shifted edge is at least 3 points above both comparisons on
    2019-2025, and the t of the weighted night beats both.
- **E2. Cheaper entry.**
  - Post a buy limit at the 14:00 UTC bid + 25% of the spread. If it is not filled by 14:10 UTC, buy at market.
  - The exit is unchanged.
  - **Counts if:** the paired difference against the 14:00 market entry passes the test rules on 2019-2025.
- **E3. Japan's portfolio flows.** Two features, each tested with Spearman correlation against the Gotobi night's
  net, under the same rules as F in the panel pre-registration:
  - (a) the latest Ministry of Finance weekly figure for net foreign bond purchases by Japanese residents,
    published Thursdays, z-scored over 52 weeks;
  - (b) Japan's trade balance (FRED), lagged one month.

## New edges

- **E4. Yuan fixing.**
  - Signal: the official USD/CNY central parity minus the USD/CNH price at 01:14 UTC, in pips.
  - Trade USD/CNH in the direction of the signal from 01:16 to 08:00 UTC, on days with |signal| above the
    fit-period 75th percentile.
  - Also run as AUD/USD in the opposite direction.
- **E5. FX option expiry at the 10:00 New York cut.**
  - For EUR/USD and USD/JPY, take the strike with the largest notional expiring that day (DTCC public reports).
  - If it lies within 30 pips of the 12:00 UTC mid, trade towards it from 12:00 UTC to the cut.
  - Untestable if the DTCC files cannot be had.
- **E6. Gold options expiry.**
  - The COMEX gold option expiry is taken as the 4th-last business day of each month (the exchange's rule).
  - Gold's return 13:30-17:30 UTC on expiry day, and from 13:30 UTC the day before, compared with all other days.
  - The trade goes in the train direction.
- **E7. US Treasury auctions** (10-year and 30-year nominal, from fiscaldata.treasury.gov). Two windows, each traded
  in USD/JPY and gold:
  - (a) pre-auction: 13:30 UTC to 5 minutes before the auction close;
  - (b) post-auction: the close to 14:00 UTC the next day.

  The direction is from Lou, Yan & Zhang (2013): yields rise before an auction and ease after it, so the dollar
  goes up before and down after.
- **E8. Central bank meetings: BoJ, ECB, BoE** (dates from their own pages; the Fed's from federalreserve.gov,
  which also repairs the FOMC list that stopped in 2022).
  - Pre-announcement drift: from 24 hours to 15 minutes before the decision.
  - Pairs: USD/JPY for the BoJ, EUR/USD for the ECB, GBP/USD for the BoE.
  - The direction is from train.
- **E9. MSCI quarterly index review days** (last business day of February, May, August and November).
  - USD/JPY and EUR/JPY from 14:00 to 21:00 UTC on the day, and over the next Tokyo morning, 00:00-03:00 UTC.
- **E10. Economic surprises** (actual minus forecast from the public calendar dataset).
  - Releases: US non-farm payrolls, CPI, retail sales, ISM manufacturing.
  - The surprise is z-scored over the fit period.
  - Trade USD/JPY and gold from 5 minutes to 4 hours after the release, in the direction of the surprise; only
    when |z| is at least 1.
- **E11. Sentiment.**
  - (a) CFTC Traders in Financial Futures: leveraged funds' net position in JPY, EUR and gold futures, z-scored
    over 156 weeks. Trade the following week's return, with the sign (contrarian or trend) from train.
  - (b) Google Trends attention for "yen", "gold price", "forex" and "dollar": the month after an attention spike
    (above the fit-period 90th percentile).
  - (c) Retail position ratios (OANDA position book) only if a free OANDA practice-account token is added as
    the `OANDA_TOKEN` secret; otherwise untestable.

## Expected outcome, stated before the run

- Each of these has perhaps a 5-10% chance of counting after costs. The likeliest result is zero or one.
- A test that counts on fewer than about 150 check events is treated as "promising", not "confirmed", until the
  forward test.
