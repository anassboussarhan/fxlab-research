# Round 21 results: the verified gaps across markets (2026-11-15)

- **Pre-registration:** `docs/round21_preregistration.md`.
- **Runs:** 43736162764; X2 rerun and combine in 43736915049. Branch `round21`.
- **Code:** `scripts/r21_intraday.py`, `scripts/r21_rest.py`.
- **2026:** not used.

## Summary

**0 of 23 tests count; none even meets its own rule** (P1 > 0 and P2 t ≥ 1.645).
- The ICT pieces never coded before all lose after costs on the markets ICT traders use.
- Most have no gross edge at all.
- The stock anomalies, the two macro currency signals and crypto cross-sectional momentum / reversal are flat or
  negative in 2015-2025 / 2005-2025 / 2022-2025.

## Results

**Units:**
- Parts I and X: net in bp per day with trades, at futures costs for the indices and CFD costs for FX.
- Parts M, S and K: per month, or per week for K.

| Test | P1 net (t) | P2 net (t) | P2 gross (t) | Note |
|---|---|---|---|---|
| I1 Silver Bullet, US500 | −3.0 bp (−4.1) | −1.3 bp (−1.9) | +0.3 bp (0.4) | 1,354 trades |
| I1 Silver Bullet, NAS100 | −1.8 bp (−1.5) | −1.1 bp (−1.4) | −0.5 bp (−0.7) | 1,542 trades |
| I2 Judas swing, US500 | −4.5 bp (−1.7) | +0.1 bp (0.0) | +1.7 bp (0.7) | |
| I2 Judas swing, NAS100 | −5.9 bp (−1.7) | +2.5 bp (1.0) | +3.1 bp (1.2) | P1 negative |
| I3 Midnight-open bias, US500 | +2.0 bp (0.3) | −6.5 bp (−1.4) | −4.9 bp (−1.0) | |
| I3 Midnight-open bias, NAS100 | −2.2 bp (−0.3) | −8.8 bp (−1.7) | −8.2 bp (−1.6) | |
| X1b SMT, US500 vs NAS100 (5 min) | −3.1 bp (−10.6) | −1.8 bp (−3.8) | −0.7 bp (−1.4) | 13,714 trades |
| X2 NDOG fill | −1.9 bp (−1.0) | no trades | | not testable after 2018: the stored index files lack the 18:00 New York reopen |
| X1a SMT, EUR/USD vs GBP/USD (H1) | −2.2 bp (−3.0) | −2.5 bp (−3.1) | −0.7 bp (−0.9) | 7,054 trades |
| X3 OTE (62-79% after displacement) | −2.6 bp (−2.5) | −1.9 bp (−1.7) | +0.4 bp (0.3) | 2,222 trades |
| X4 CBDR projections | −1.6 bp (−2.9) | −1.2 bp (−1.9) | +0.7 bp (1.1) | 7,409 trades |
| X5 Breaker blocks | −3.9 bp (−4.0) | −6.2 bp (−6.1) | −3.8 bp (−3.8) | 6,242 trades |
| M1 Real-rate carry | +0.38% / month (2.7) | −0.26% (−2.1) | | 1985-2004 / 2005-2021 (OECD CPI ends) |
| M2 Curve slope (long flat, short steep) | +0.61% (4.0) | +0.01% (0.1) | | worked before 2005, gone since |
| (M0 nominal carry, reported) | +0.46% | +0.07% (0.4) | | |
| S1 Momentum 12-1, vs equal weight | −0.29% (−0.9) | +0.05% (0.2) | | |
| S2 1-month reversal | −0.14% (−0.4) | −0.01% (0.0) | | |
| S3 52-week high | −0.42% (−1.4) | −0.32% (−1.4) | | |
| S4 Low volatility | −0.20% (−0.6) | −0.17% (−0.6) | | |
| S5 Overnight momentum | −0.17% (−0.6) | −0.35% (−1.7) | | |
| S6 Index-deletion rebound, vs SPY over 60 days | +7.4% (1.4) | +0.9% (0.2) | | 137 deletions with prices |
| S7 Pairs (Gatev) | +0.05% (1.0) | −0.06% (−1.2) | | |
| K1 Crypto momentum, top-50 coins | +1.44% / week (1.3) | −0.61% (−1.1) | | 567 coins incl. delisted |
| K2 Crypto reversal | −2.53% (−2.8) | −0.21% (−0.4) | | |

## What this shows

**ICT on indices and FX:**
- Before costs every rule is within ±1 bp of zero (Silver Bullet, OTE, CBDR, SMT) or negative (breakers −3.8 bp,
  t −3.8).
- That matches the published 25,000-variant Silver Bullet study (1.7% of variants survive on paper fills) and the
  2026 SSRN test of ICT as taught.
- What the YouTube results show is discretion plus selection, not a mechanical edge.

**Macro currency signals:**
- Real-rate carry and the curve-slope signal both worked from 1985 to 2004 (t 2.7 and 4.0).
- Both are flat or negative since 2005, the same break as nominal carry and trend (round 18).

**Stocks:**
- On large caps the classic anomalies are flat against an equal-weight benchmark in 2015-2025.
- Momentum and low volatility are known to be weak in large caps recently. The survivorship bias in the Yahoo
  data would push them up, not down.

**Crypto:**
- Weekly cross-sectional momentum was positive in 2019-2021 (+1.4% a week, t 1.3) and negative since.

## Data limits

- The US500 minute file ends in early 2024.
- The 2012-2018 index coverage is partial: 260-395 days for the Silver Bullet.
- The OECD CPI series end in 2021, so M1's P2 is 2005-2021.
- Yahoo lacks about 40% of former S&P 500 members.

Not financial advice.
