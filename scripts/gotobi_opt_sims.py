"""Risk simulations of the walk-forward Gotobi trades (scripts/gotobi_opt_lab.py output, out of sample 2011-2025):
  A. Own account compounding at constant leverage (lottery-style: from safe to reckless), with each night's open
     loss applied (a night whose open loss takes the account below 10% of its start is a margin-call wipe-out).
     Nights are drawn at random with replacement from the out-of-sample nights.
  B. Prop challenges under four 100k programmes (rules in scripts/prop_firms_ev.py), the daily and maximum loss
     limits checked against the open losses, separate leverage while passing and once funded, with the speed of
     the first payout. Mixes: USD/JPY alone, the walk-forward yen basket, and each plus the Nasdaq-100 overnight CFD
     (QQQ proxy, closing losses only).
Development data only. Usage: python scripts/gotobi_opt_sims.py --oos DIR --out DIR"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_firms_ev as PF  # noqa: E402


def account(net, mae, L, years, per_year, n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    k = int(per_year * years)
    ix = rng.integers(0, len(net), size=(n, k))
    eq = np.ones(n)
    peak = np.ones(n)
    mdd = np.zeros(n)
    dead = np.zeros(n, bool)
    for t in range(k):
        low = eq * (1 + L * mae[ix[:, t]])
        new = ~dead & (low <= 0.10)
        mdd = np.where(dead, mdd, np.maximum(mdd, 1 - low / peak))
        eq = np.where(new, np.maximum(low, 0.0), np.where(dead, eq, eq * (1 + L * net[ix[:, t]])))
        dead |= new
        peak = np.maximum(peak, eq)
    return {"L": L, "years": years, "median_x": float(np.median(eq)), "mean_x": float(eq.mean()),
            "p_wipeout": float(dead.mean()), "p_lose_half": float((eq < 0.5).mean()), "p_2x": float((eq >= 2).mean()),
            "p_10x": float((eq >= 10).mean()), "p_100x": float((eq >= 100).mean()),
            "p99_x": float(np.percentile(eq, 99)), "median_max_dd": float(np.median(mdd))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traders", type=int, default=20_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ser = {}
    for name, f in (("UJ", "oos_USDJPY.csv"), ("EJ", "oos_EURJPY.csv"), ("BASKET", "oos_basket.csv")):
        fp = os.path.join(a.oos, f)
        if os.path.exists(fp):
            d = pd.read_csv(fp, index_col=0, parse_dates=True)
            ser[name] = d
    # A. own account
    A = []
    for name, d in ser.items():
        yrs = d.index.year.nunique()
        per_year = len(d) / max(yrs, 1)
        net, mae = d.net.to_numpy(), np.minimum(d.mae.fillna(d.net.clip(upper=0)).to_numpy(), 0)
        for L in (10, 15, 20, 30, 40, 60):
            for years in (1, 3):
                A.append({"series": name, "trades_per_year": round(per_year, 1), **account(net, mae, L, years, per_year)})
    A = pd.DataFrame(A)
    A.to_csv(os.path.join(a.out, "account.csv"), index=False)
    # B. prop (2011-2025 business days)
    days = pd.bdate_range("2011-01-01", "2025-12-31")
    cols, maes = {}, {}
    for name, d in ser.items():
        g = d.groupby(d.index.normalize())
        cols[name] = g.net.sum().reindex(days).fillna(0.0)
        maes[name] = g.mae.sum().reindex(days).fillna(0.0)
    try:
        import prop_mix_sim as PM
        q, irx = PM.yahoo("QQQ"), PM.yahoo("^IRX")
        fq = q["Adj Close"] / q["Close"]
        o, c = q["Open"] * fq, q["Adj Close"]
        rf = (irx["Close"] / 100).reindex(q.index).ffill().fillna(0)
        wk = q.index.to_series().diff().dt.days == 1
        O = ((o / c.shift(1) - 1) - 1e-4 - (rf.shift(1) + 0.025) / 360)[wk].dropna()
        cols["NQ"] = O.reindex(days).fillna(0.0)
        maes["NQ"] = np.minimum(cols["NQ"], 0.0)
    except Exception as ex:                                               # noqa: BLE001
        print("no QQQ:", repr(ex)[:200])
    X, M = pd.DataFrame(cols), pd.DataFrame(maes)
    M = np.minimum(M, np.minimum(X, 0.0))
    mixes = [(k,) for k in ("UJ", "BASKET") if k in X] + [(k, "NQ") for k in ("UJ", "BASKET") if k in X and "NQ" in X]
    if "UJ" in X and "EJ" in X:
        mixes += [("UJ", "EJ"), ("UJ", "EJ", "NQ")] if "NQ" in X else [("UJ", "EJ")]
    rows = []
    for mix in mixes:
        base = {"UJ": 1.0, "EJ": 1.0, "BASKET": 1.0, "NQ": 0.5}
        lev = np.array([base[k] for k in mix])
        Au, Mu = X[list(mix)].to_numpy() * lev, M[list(mix)].to_numpy() * lev
        for firm, rules in PF.FIRMS.items():
            for se, sf in itertools.product((2, 3, 4, 6, 8), (2, 3, 4, 6)):
                r = PF.simulate(Au, rules, n=a.traders, M=Mu, scale_eval=se, scale_funded=sf)
                rows.append({"mix": "+".join(mix), "firm": firm, "yen_lev_pass": se, "yen_lev_funded": sf,
                             "nq_lev_pass": 0.5 * se if "NQ" in mix else 0, **r})
    B = pd.DataFrame(rows)
    B["net_per_month"] = B.mean_net / (B.median_days_to_first_pay.fillna(250) / 21 + 12)
    B.to_csv(os.path.join(a.out, "prop.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    print(A.round(3).to_string())
    cols_ = ["mix", "firm", "yen_lev_pass", "yen_lev_funded", "pass", "paid", "mean_net", "median_net", "p10_net",
             "median_days_to_first_pay", "paid_within_3_months", "paid_within_6_months", "fail_daily"]
    print("BEST MEAN", B.sort_values("mean_net", ascending=False).head(15)[cols_].round(3).to_string())
    print("FASTEST (paid within 3 months)", B.sort_values("paid_within_3_months", ascending=False).head(10)[cols_].round(3).to_string())
    print("SAFEST (p10)", B.sort_values("p10_net", ascending=False).head(10)[cols_].round(3).to_string())


if __name__ == "__main__":
    main()
