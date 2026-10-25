"""The best option structures from the strategy lab, run on the options capital providers' published rules
(Options Funding Growth, Imperial Elite; scripts/options_prop_sim.py), from their daily mark-to-market P&L on
real quotes (2019-2025, no 2026).

Size: the daily P&L series (per unit of underlying notional) is scaled so that its daily standard deviation is
0.4%, 0.6%, 0.8% or 1.0% of the account, but never above 4x the account in underlying notional (about what Reg-T
margin of 20% allows for naked short options; the providers' own buying-power rules are not published). Starts: every month start from the first full year of data to 30 months
before the end, horizon up to 750 trading days (fewer near the end of the data, so late starts are censored).
Usage: python scripts/options_strategy_prop.py --daily FILE --out FILE"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from options_prop_sim import run  # noqa: E402

TARGETS = (0.004, 0.006, 0.008, 0.010)
MAX_NOTIONAL = 4.0     # underlying notional per unit of account: about Reg-T margin (20%) on naked short options
NAKED = {"SP30", "SP16", "SP50", "STRG16", "STRD", "PRAT", "CC30", "BW50", "STK"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = pd.read_csv(a.daily, index_col=0, parse_dates=True).sort_index()
    rows = []
    for col in D.columns:
        x = D[col].fillna(0.0)
        first = x.ne(0).idxmax()
        x = x.loc[first:]
        sd = x.std()
        if not sd > 0:
            continue
        starts = pd.date_range(x.index[0] + pd.offsets.MonthBegin(1), x.index[-1] - pd.DateOffset(months=30), freq="MS")
        idx = [x.index.get_indexer([s], method="bfill")[0] for s in starts]
        idx = [i for i in idx if i >= 0]
        sym, struct = col.split("|")[:2]
        for tgt in TARGETS:
            lam = min(tgt / sd, MAX_NOTIONAL)
            for prov in ("OF", "IMP"):
                R = pd.DataFrame([run(x, lam, prov, i) for i in idx])
                if not len(R):
                    continue
                net = R.paid - R.fees
                rows.append({"combo": col, "symbol": sym, "struct": struct, "provider": prov,
                             "daily_sd_target": tgt, "notional_per_account": float(lam),
                             "capped_by_margin": bool(tgt / sd > MAX_NOTIONAL),
                             "worst_day_account": float(x.min() * lam), "starts": int(len(R)),
                             "pass_rate": float(R.passed.mean()), "paid_rate": float((R.paid > 0).mean()),
                             "mean_paid": float(R.paid.mean()), "mean_fees": float(R.fees.mean()),
                             "mean_net": float(net.mean()), "median_net": float(net.median()),
                             "p10_net": float(net.quantile(0.1)), "p90_net": float(net.quantile(0.9)),
                             "breach_rate": float(R.breach.mean()), "median_days": float(R.days.median()),
                             "naked_short": struct in NAKED,
                             "allowed_options_funding": sym not in ("SPX", "XSP", "NDX", "RUT")})
    T = pd.DataFrame(rows).sort_values("mean_net", ascending=False)
    T.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(T.head(40).round(3).to_string())


if __name__ == "__main__":
    main()
