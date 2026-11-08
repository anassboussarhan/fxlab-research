# Round 9 results (2026-11-08)

Rules: `docs/round9_preregistration.md` (with addenda 1-5, all written before the rows they change were read).
Code `scripts/round9_lab.py`; raw outputs on the `misc-r9m1`, `misc-r9m1b`, `misc-r9m1c`, `misc-r9opt`,
`misc-r9daily`, `misc-r9at5` branches; the combined table `docs/round9/round9_combined.csv`. 2026 not used.

**Nothing counts.** 18 primary rows, none passes Benjamini-Hochberg at q 0.10, and only one (AT8) reaches t 1.645 in
its check period, with a negative fit period.

| Test | Check period: n, mean net, t | Fit period mean | Verdict |
|---|---|---|---|
| AT8 turn of the month, long gold | 67 months, +40.5 bp, t 1.80 | -2.5 bp | fails (fit negative; BH) |
| O4 option-expiry pinning (CME OI) | 148, +4.4 bp, t 1.62 | -5.7 bp (39 events) | fails |
| AT6 USD/JPY carry hedged with yen calls | 96 cycles, +28.0 bp, t 1.21 | +7.4 bp | fails; the hedge halves the worst cycle (-464 vs -915 bp) and lowers the mean |
| AT5 news-tone LASSO (GDELT) | 69 months, +16.8 bp, t 1.03 | - | fails (first run had a look-ahead, see addendum 5) |
| N1 surprise drift, +15 -> +75 min | 663, +1.0 bp, t 1.01 | -2.2 bp (t -4.9) | fails |
| AT3 carry with a turnover penalty | 84 months, +27.4 bp, t 0.77 (Sharpe 0.29) | +24.5 bp | fails; turnover 0.03 vs 0.17 a month, Sharpe 0.29 vs 0.22 cost-blind |
| S1b currency skewness | 84, +8.3 bp, t 0.47 | -0.7 bp | fails |
| AT2 Hurst rule on gold / silver | 517 days, +2.7 bp, t 0.44 | -9.1 bp | fails |
| S2 Shanghai gold premium | 221, +2.5 bp, t 0.40 | -7.8 bp | fails |
| N2 surprise-index cross-section | 191 weeks, +0.8 bp, t 0.26 | -4.6 bp | fails |
| S5 coupon conversion | 27, -0.3 bp | -0.4 bp | fails |
| S1a same-month seasonality | 84 months, -2.7 bp | -24.9 bp | fails |
| O3 risk-reversal momentum | 417 weeks, -6.9 bp | +6.5 bp | fails |
| AT4 USD/JPY London-open momentum | 1,776 days, -1.5 bp, t -1.7 | -0.7 bp | fails; gross -0.1 bp (no momentum before costs); EUR, GBP, AUD, CHF also negative |
| AT7 gold session momentum, long only | 842 days, -3.5 bp, t -2.6 | -9.6 bp | fails; gross +0.1 bp; silver -17.4 bp |
| O5 gamma switch | 2,095, -1.9 bp, t -4.4 | -0.7 bp | fails |
| N3 pre-release drift | 3,001, -2.3 bp, t -4.6 | -2.4 bp | fails |
| AT1 gold cost-gated z-score | 3,522, -3.7 bp, t -9.4 | -3.3 bp | fails; gross -0.4 bp, so the gate removes nothing that was there; -5.4 / -7.1 bp at 1.5x / 2x cost |

**G1, the Gotobi window on CME yen futures (judged on its own):** the leg from the 5 PM Chicago reopen to 00:53 UTC
earns +2.4 bp before costs on 2019-2025 (t 3.3; +1.8 bp, t 2.0 on 2007-2018), confirming that this part of B0 is
real, but 1.5 ticks plus commission (about 1.7 bp) leave +0.9 bp (t 1.2) and +0.6 bp (t 0.7): **fails**. The full
B0 night on the same nights: +4.9 bp (t 3.2). A futures prop firm cannot hold the part that pays most.

**Not tested:** F2 (CME signed order flow: the affordable span is 1.4 years, nothing bought); S3 (only 20 Norges Bank
announcements found); S4 (the BoJ monthly files list short-term operations only, not the outright JGB purchase
days). Gold "18:00 New York + 9 hours" was A1 in run 22 (+2.4 bp gross, -4.4 bp net).

What this round adds: the claims in the user's three lists that could be checked here (USD/JPY London-open momentum,
gold session momentum, cost-gated gold mean reversion, Hurst on the gold / silver spread, cost-aware carry,
news-drift after 15 minutes, hedged carry, turn of the month) all fail on this data and these costs; the only
before-cost effect found is the one already known (the Gotobi fix leg).
