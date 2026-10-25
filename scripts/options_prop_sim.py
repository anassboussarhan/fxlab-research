"""The best options edge found (selling one-month S&P 500 premium, only when the VIX term structure is in contango)
run on the options capital providers whose published rules allow it (overnight holding, short premium /
spreads, end-of-day trailing drawdown). Historical scenario analysis, every month start from 2006-08 to 2023-12.

Strategy proxy: the daily mark-to-market of a short one-month S&P 500 variance position at the VIX
(scripts/options_lab.py, short_variance_daily.csv), 1 vol point of cost a month, traded only in months that start
in contango (VIX < VIX3M; test O3: +1.63 vol points a month, t 4.0, 2006-2025). Size lam = account fraction per
vol point of variance notional. A variance swap is a proxy: a retail trader would sell put spreads / iron condors
on SPY (index options are buy-only at Options Funding since 2026-10-19), which cap the loss but give up part of
the premium.

Providers modelled from their published rules (see docs/options_capital_providers.md):
  OF  Options Funding, Growth, $50k: target +$5,000; end-of-day trailing drawdown 6% of the starting balance
      from the high-water mark; overnight allowed; monthly fee (USD 250 assumed) until funded plus a 129
      activation fee refunded with the first payout; funded: 80% of profit paid each 20-trading-day cycle with at
      least 8 winning days, account reset to the starting balance after a payout.
  IMP Imperial Trader Funding, Elite, $50k: one-time fee USD 357; target +$5,000; end-of-day trailing drawdown 7%
      (assumed; not published for $50k), no daily limit; consistency: best day <= 33% of the profit at the target;
      sim-funded payouts: 90% of profit, total capped at 7% of the starting balance (then a live-account review,
      not modelled).
Usage: python scripts/options_prop_sim.py --daily FILE --monthly FILE [--out FILE]"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

INIT = 50_000.0


def load(daily, monthly):
    D = pd.read_csv(daily, parse_dates=["day", "month"])
    M = pd.read_csv(monthly, parse_dates=["month"]).set_index("month")
    M = M[M.vix3m.notna()]
    inmkt = (M.K < M.vix3m)
    D = D[D.month.isin(M.index)].copy()
    D["in"] = D.month.map(inmkt).astype(bool)
    # 1 vol point of cost a month, spread over the month's days
    ndays = D.groupby("month").day.transform("count")
    D["pnl"] = np.where(D["in"], D.pnl_vp - 1.0 / ndays, 0.0)
    return D.set_index("day")["pnl"], M


def run(pnl, lam, provider, start_idx, max_days=750):
    x = pnl.to_numpy()[start_idx:start_idx + max_days] * lam
    if provider == "OF":
        fee_month, dd, fee_once, split, cap = 250.0, 0.06, 0.0, 0.80, np.inf
    else:
        fee_month, dd, fee_once, split, cap = 0.0, 0.07, 357.0, 0.90, 0.07
    eq, hwm, day, fees, passed, paid, best_day, n_win, cyc = 0.0, 0.0, 0, fee_once, False, 0.0, 0.0, 0, 0
    activation_refunded = False
    for i, r in enumerate(x):
        if not passed and i % 21 == 0:
            fees += fee_month                                   # subscription until funded
        eq += r
        if eq - hwm < -dd:                                     # end-of-day trailing breach
            return {"passed": passed, "paid": paid, "fees": fees, "days": i + 1, "breach": True}
        hwm = max(hwm, eq)
        if not passed:
            best_day = max(best_day, r)
            if eq >= 0.10 and (provider == "OF" or best_day <= 0.33 * eq):
                passed, eq, hwm, cyc, n_win = True, 0.0, 0.0, 0, 0
                if provider == "OF":
                    fees += 129.0
            continue
        cyc += 1
        n_win += r > 0
        if cyc >= 20:
            if eq > 0 and n_win >= (8 if provider == "OF" else 0) and paid < cap * INIT:
                amt = min(split * eq * INIT, cap * INIT - paid) if np.isfinite(cap) else split * eq * INIT
                paid += amt
                if provider == "OF" and not activation_refunded:
                    fees -= 129.0
                    activation_refunded = True
                eq, hwm = 0.0, 0.0
            cyc, n_win = 0, 0
        if paid >= cap * INIT:
            return {"passed": passed, "paid": paid, "fees": fees, "days": i + 1, "breach": False}
    return {"passed": passed, "paid": paid, "fees": fees, "days": len(x), "breach": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--monthly", required=True)
    ap.add_argument("--out", default="docs/options_prop_sim.csv")
    a = ap.parse_args()
    pnl, M = load(a.daily, a.monthly)
    starts = [pnl.index.get_indexer([d], method="bfill")[0] for d in pd.date_range("2006-09-01", "2023-12-01", freq="MS")]
    rows = []
    for prov in ("OF", "IMP"):
        for lam in (0.001, 0.0015, 0.002, 0.003, 0.005):
            R = pd.DataFrame([run(pnl, lam, prov, s) for s in starts if s >= 0])
            net = R.paid - R.fees
            rows.append({"provider": prov, "lam_per_vol_point": lam, "starts": int(len(R)),
                         "pass_rate": float(R.passed.mean()), "paid_rate": float((R.paid > 0).mean()),
                         "mean_paid_usd": float(R.paid.mean()), "mean_fees_usd": float(R.fees.mean()),
                         "mean_net_usd": float(net.mean()), "median_net_usd": float(net.median()),
                         "p90_net_usd": float(net.quantile(0.9)), "breach_rate": float(R.breach.mean()),
                         "median_days": float(R.days.median())})
    T = pd.DataFrame(rows)
    T.to_csv(a.out, index=False)
    worst_in = pnl[pnl != 0].nsmallest(5)
    print("worst in-market days (vol pts):", {str(k.date()): round(v, 1) for k, v in worst_in.items()})
    print("mean in-market vol pts per day:", round(pnl[pnl != 0].mean(), 3), "days in market share:",
          round((pnl != 0).mean(), 3))
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
