# The noise-area breakout on Nasdaq futures: 2026 check and forward test (pre-registered 2026-11-14)

Written before any 2026 Nasdaq data was downloaded or read. Authorised by Anas on 2026-11-14 ("Yes continue
testing", in reply to the offer of a single pre-registered 2026 test and a forward test).

## The rule (frozen, round 19 A-NB-NAS100)

- **Code:** `scripts/r19.py`, functions `session_days`, `nb_day` and `costs_of`; driver `scripts/nb_forward.py`.
- **Instrument:** NAS100 CFD minute mids (Dukascopy USATECHIDXUSD), New York session 09:30-15:59.
- **Bands:**
  - σ(m) = the mean over the previous 14 days of |close at minute m / 09:30 open − 1|;
  - UB = max(open, previous close) × (1 + σ);
  - LB = min(open, previous close) × (1 − σ).
- **Decisions:**
  - at HH:00 and HH:30 from 10:00 to 15:30, on the close of the minute before;
  - price above UB → long, below LB → short;
  - a long exits when price falls below max(UB, VWAP), a short when price rises above min(LB, VWAP);
  - fill at the next minute's open; flat at 16:00.
- **Size:** one unit of notional per trade.
- **Cost:** NQ, 0.375 index points per side (one tick plus USD 2.50 commission).

## A. The 2026 check (one test; the fourth use of the 2026 holdout)

- **Data:**
  - Dukascopy NAS100 minute bars, 2026-01-01 to 2026-10-31, 13:00-21:59 UTC on weekdays (covers 09:30-16:00
    New York in both summer and winter time);
  - the warm-up (14 days) and the previous close from December 2025.
- **Verdict on the daily net return at futures costs:**
  - supported: mean > 0 and t ≥ 1.645;
  - contradicted: the difference from the 2019-2025 mean (+3.27 bp, t 2.25, 1,652 days) has t ≤ −1.645;
  - inconclusive otherwise.
- **Power, stated in advance:** at the 2019-2025 mean and its daily spread (about 59 bp), ten months
  (about 208 days) give an expected t of about 0.8. That means about a 20% chance of "supported" even if the edge
  is real. "Inconclusive" is the likely outcome either way. The forward test is what settles it.
- **Also reported:** twice the cost, CFD costs, the volatility-sized return, and the months.

## B. The paper forward test (from Monday 2026-11-16)

- **Schedule:** weekly on Saturday, the same code downloads NAS100 minute bars from 2026-10-15 (warm-up only) to
  the last Friday.
- **Ledger:** it writes the daily ledger from 2026-11-16 to branch `forward-nb`.
- **Rules:** no rule changes. Any change starts a new track.
- **Review:** after 250 trading days and after 500. A one-sided t ≥ 1.645 on the forward days alone counts as
  support.
- **Futures prop simulation:** only if A or B supports the rule.

Not financial advice.
