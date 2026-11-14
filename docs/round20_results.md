# Round 20 results: the famous traders' setups, and the 2026 check of the noise-area breakout (2026-11-14)

- **Pre-registrations:** `docs/round20_preregistration.md` and `docs/nb2026_preregistration.md`.
- **Runs:**
  - nb26: 43620848334, branch `nb26`;
  - part S: 43624416126, branch `round20s`.
- **Code:** `scripts/nb_forward.py`, `scripts/r20_swing.py`, `scripts/r20_w.py`.

## 1. The noise-area breakout on Nasdaq futures, 2026 (one pre-registered test)

| | 2019-2025 (development) | 2026-01 to 2026-10 |
|---|---|---|
| Days | 1,652 | 153 |
| Net at NQ futures costs, bp a day | +3.27 (t 2.25) | +2.64 (t 0.73) |
| Gross / twice the cost / CFD costs, bp | +3.76 / +2.78 / +1.85 | +2.84 / +2.45 / +1.75 |
| Volatility-sized return / worst drawdown | | +6.8% / −6.7% |

- **Verdict: inconclusive, as the stated power predicted** (about a 20% chance of "supported" even if the edge is
  real). The 2026 mean is close to the development mean (t of the difference −0.16), so 2026 does not contradict
  it.
- **Months:** 6 of 10 positive (January −6.3, March +16.1, July −13.6, August +15.0, September +9.9, October
  +9.1 bp a day).
- **Coverage:** 153 of about 208 sessions. The tick download misses the 09:30 minute or too many minutes on about a
  quarter of the days, and the rule needs a full session. The dropped days are not chosen by outcome.
- **Forward test:** the weekly paper forward test (`forward_nb.yml`, from 2026-11-16) is the deciding test. At
  this effect size it needs roughly 3 to 4 years for t ≈ 1.645.

## 2. What the search found about the claims

- **Warrior Trading (Ross Cameron):**
  - The FTC (2022) found that the "vast majority" of customer accounts lost money, and the company paid USD 3
    million.
  - Its advertised "USD 101,280.47 in verified profits in under 45 days" was Cameron's own account, not the
    customers'.
  - A star's account is one survivor among many traders, so it says nothing about the setup.
- **Retail day traders:** in the Brazilian study (Chague, De-Losso & Giovannetti), 97% of those who day-traded
  futures for more than 300 days lost money. Only 1.1% earned more than the minimum wage.
- **Small-cap gappers (SmallCapLab statistics, not audited):** about 68% of large gap-ups close below their open.
  Professional small-cap traders such as Steven Dux mostly short them.
- **Kullamägi's episodic pivot, modelled by an outside researcher (stonkscapital):** the raw signal over 50,000
  earnings days had "0 edge". The claimed 30% CAGR needs earnings-surprise filters, and costs, survivorship and
  universe are not stated.

## 3. Part S: the swing setups on point-in-time S&P 500 stocks (daily, 2005-2025)

- **Data:**
  - membership from the maintained `fja05680/sp500` history (1,209 tickers ever);
  - Yahoo prices for 732 of them, covering 71% of member-days.
- **Survivorship:** the missing ones are mostly companies that left. This favours long setups, so the failures
  below are conclusive.

| Test | P1 2005-2014: trades, excess vs SPY per trade (t) | P2 2015-2025: trades, excess (t) | P2 win rate, avg win / loss | Portfolio CAGR 2005-2025 (SPY 10.7%) | Counts |
|---|---|---|---|---|---|
| S-QB Kullamägi breakout | 3,081, −0.52% (−5.96) | 5,101, −0.54% (−5.82) | 21%, +4.2% / −1.5% | −15.3% | no |
| S-EP episodic pivot | 596, −1.36% (−1.40) | 583, −0.20% (−0.15) | 29%, +12.4% / −4.4% | −36.0% | no |
| S-MV Minervini template | 3,180, −0.67% (−2.24) | 4,958, −1.60% (−5.17) | 29%, +8.5% / −4.1% | −0.1% | no |
| Base: every 10-day-high breakout | 46,660, −0.52% (−7.03) | 70,750, −0.45% (−5.82) | 25% | | |
| Base: every 10% gap-up | 761, −2.02% (−2.71) | 761, −0.28% (−0.28) | 27% | | |
| Base: every 20-day-high breakout | 17,747, −0.67% (−2.95) | 28,198, −0.94% (−4.81) | 28% | | |

**Costs:** 5 bp per side and 6% a year financing (stock-CFD terms).

**None counts.** The setup filters do not improve on their base rates: breakout and template trades do worse than
SPY over the same days by 0.5-1.6% per trade.

**Why the setups fail here:**
- On large caps, a breakout above a recent high is followed by about the market's return, minus costs.
- The tight stops (the entry-day low, 7%) cut most trades early: about 75% of trades lose.
- The large average wins (+4% to +12%) are real but too rare.

**Limits:**
- The traders' own markets are mostly small and mid-cap growth stocks with large daily ranges, entered intraday.
  Daily bars on the S&P 500 test the setups where prop-firm stock CFDs are available, not their whole universe.
- The episodic pivot without the earnings filter is what the outside model also found to have no edge.

## 4. Part W (Warrior-style small-cap intraday): data purchase lost to a bug, not run

- **What happened (run 43629956478):**
  - The approved fetch bought the consolidated daily bars for USD 6.75, as priced.
  - My code then lost them. A request for ALL_SYMBOLS carries no ticker mappings, so the symbol column was empty,
    and my `dropna()` deleted every row before anything was saved.
  - The guard stopped the minute purchase, so no events were tested.
  - The fix (in `scripts/r20_w.py`) keeps the raw file (`--reuse`), resolves the tickers with the free symbology
    call, and aborts if the symbols are not resolved.
  - Streaming requests are billed again when repeated, so the USD 6.75 is spent.
- **Lesson:** save the raw purchase to disk before any processing, and test the full fetch path on a tiny paid
  request first.

### Original plan


- Pre-registered as addendum W, with the code in `scripts/r20_w.py`. Not run.
- It needs minute bars of small caps, including delisted ones. On Databento this costs about USD 10-11 for July
  2024 to December 2025, close to the remaining free credit.
- The alternative is free Alpaca keys, which give data from 2016.

Not financial advice.
