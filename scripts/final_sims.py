"""Final simulations: the one edge that survived (Gotobi overnight, USD/JPY + EUR/JPY, 23:00 -> 09:53 Tokyo, no
stop; the frozen H5 basket), at the size it showed in development and at the size it showed in 2026, and with no
edge at all, through five 100k prop programmes (rules re-checked 2026-11-02) and an own account played for a
lottery-like outcome.

EDGE SCENARIOS (per basket trade, net of costs and swap; the open loss of each trade applied to the loss limits):
  dev   the development trades, 2007-2025 (flow-edges T3 references: +4.7 bp a trade);
  2026  the same trades shifted so the mean is +2.04 bp, the basket's 2026 out-of-sample mean (H5);
  zero  shifted to a mean of 0 (what the fee buys with no edge: pure luck).
FIRMS (100k; fees in USD; third-party summaries of the published terms, luxalgo dataset 2026-01-15, tradetanto,
  thetrustedprop, proptradingvibes 2026-10; confirm on the firm's own page before paying):
  FTMO 2-step $585 refunded with the 1st payout, +10% / +5%, daily 5% / max 10% static, 4 days, 80%, payouts / 10 d
  FTMO 1-step $540 no refund, +10%, daily 3%, max 10% trailing, best day <= 50%, 90%, payouts / 10 d
  FundedNext Stellar 2-step $549.99 refunded with the 1st, +8% / +5%, 5% / 10%, first payout 21 days (15 trading)
    then 14 (10), 80%
  FundedNext Stellar 1-step $569.99 refunded with the 3rd payout, +10%, daily 3%, max 6%, 2 days, first payout
    after 5 days then every 5, 80%
  The5ers High Stakes $495 no refund, +8% / +5%, 5% / 10%, 3 days of +0.5% a phase, 80%, payouts / 10 d
Usage: python scripts/final_sims.py --trades DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gotobi_opt_sims as GS  # noqa: E402
import prop_firms_ev as PF  # noqa: E402

FIRMS = dict(PF.FIRMS)
FIRMS["FundedNext Stellar 2-step"] = dict(FIRMS["FundedNext Stellar 2-step"], fee=549.99)
FIRMS["FundedNext Stellar 1-step"] = dict(fee=569.99, targets=(0.10,), daily=0.03, maxloss=0.06, trailing=False,
                                          min_days=2, profit_day=0.0, min_profit_days=0, best_day=None, split=0.80,
                                          refund=True, refund_after=3, first_pay=5, pay_every=5)
LEV_PASS = (3, 5, 8, 10, 15, 20, 30)
LEV_FUND = (3, 5, 8, 10, 15)
TARGET_2026 = 2.04e-4


def basket(d):
    fr = {}
    for p in ("USDJPY", "EURJPY"):
        f = glob.glob(os.path.join(d, "**", f"trades_T3_reference__{p}_Gotobi_overnight__no_stop.csv"), recursive=True)
        x = pd.read_csv(f[0], index_col=0, parse_dates=True)
        x.index = pd.DatetimeIndex(x.index).normalize()
        fr[p] = x
    net = pd.concat([fr[p].net.rename(p) for p in fr], axis=1).mean(axis=1)
    mae = pd.concat([fr[p].mae.rename(p) for p in fr], axis=1).mean(axis=1)
    B = pd.DataFrame({"net": net, "mae": np.minimum(mae, np.minimum(net, 0.0))}).dropna().sort_index()
    return B


def shifted(B, mean):
    d = B.net.mean() - mean
    out = B.copy()
    out["net"] = B.net - d
    out["mae"] = np.minimum(B.mae - d, np.minimum(out.net, 0.0))
    return out


def daily(B):
    days = pd.bdate_range(B.index.min(), B.index.max())
    X = B.net.groupby(B.index).sum().reindex(days).fillna(0.0).to_numpy()[:, None]
    M = B.mae.groupby(B.index).sum().reindex(days).fillna(0.0).to_numpy()[:, None]
    return X, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B0 = basket(a.trades)
    per_year = len(B0) / ((B0.index.max() - B0.index.min()).days / 365.25)
    scen = {"dev": B0, "2026": shifted(B0, TARGET_2026), "zero": shifted(B0, 0.0)}
    info = {k: {"trades": len(v), "mean_bp": v.net.mean() * 1e4, "sd_bp": v.net.std() * 1e4,
                "worst_open_loss_bp": v.mae.min() * 1e4} for k, v in scen.items()}
    print(info, "trades a year", round(per_year, 1), flush=True)
    rows = []
    for s, B in scen.items():
        X, M = daily(B)
        for firm, rules in FIRMS.items():
            for lp, lf in itertools.product(LEV_PASS, LEV_FUND):
                r = PF.simulate(X, rules, n=a.n, seed=1, M=M, scale_eval=lp, scale_funded=lf)
                rows.append({"scenario": s, "firm": firm, "lev_pass": lp, "lev_funded": lf, **r})
        print(s, "prop done", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(a.out, "prop.csv"), index=False)
    acc = []
    for s, B in scen.items():
        for L in (5, 10, 15, 20, 30, 40, 60):
            for yrs in (1, 3):
                acc.append({"scenario": s, **GS.account(B.net.to_numpy(), B.mae.to_numpy(), L, yrs, per_year)})
    A = pd.DataFrame(acc)
    A.to_csv(os.path.join(a.out, "account.csv"), index=False)
    pd.set_option("display.width", 260)
    cols = ["firm", "lev_pass", "lev_funded", "pass", "paid", "mean_net", "median_net", "median_days_to_first_pay",
            "paid_within_3_months", "paid_within_6_months"]
    for s in scen:
        Q = P[P.scenario == s]
        print(f"\n=== {s}: HIGHEST mean ===\n", Q.sort_values("mean_net", ascending=False)[cols].head(5).round(3).to_string())
        print(f"=== {s}: FASTEST (paid within 3 months) ===\n", Q.sort_values(["paid_within_3_months", "mean_net"], ascending=False)[cols].head(5).round(3).to_string())
        print(f"=== {s}: BOTH (median net) ===\n", Q.sort_values("median_net", ascending=False)[cols].head(5).round(3).to_string())
    print(A.round(3).to_string())


if __name__ == "__main__":
    main()
