"""E17 of docs/edges_batch_preregistration.md (exploratory): gold calendar trades next to Gotobi in an FTMO 2-Step
account, 2019-2025. Gold legs at 2x on their own days (GLD daily closes, Yahoo; 2 bp a window): the turn of the
month (close before the last business day -> close of the third business day of the next month, lit4 G1) and the
15 trading days before the Chinese New Year (lit3 D6). E6b did not count, so it is not added. Gotobi at 5x as in
prop_best (80 bp stop basket). Worst open loss of a gold day = min(0, its close-to-close return) (no intraday path).
Counts if P(paid) at the 2026-shifted Gotobi edge is at least 3 points above Gotobi alone.
Usage: python scripts/edges_e17.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_best as PB  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gold_days():
    import yfinance as yf
    from lunardate import LunarDate
    g = yf.download("GLD", start="2018-06-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    g = (g.iloc[:, 0] if isinstance(g, pd.DataFrame) else g)
    g.index = pd.DatetimeIndex(g.index).tz_localize(None)
    r = g.pct_change().dropna()
    w = pd.Series(0.0, index=r.index)
    cost = pd.Series(0.0, index=r.index)
    idx = r.index
    months = pd.Series(idx, index=idx).groupby([idx.year, idx.month])
    firsts = {k: v.iloc[:3].tolist() for k, v in months}
    lasts = {k: v.iloc[-1] for k, v in months}
    keys = sorted(firsts)
    for k0, k1 in zip(keys[:-1], keys[1:]):
        days = [lasts[k0]] + firsts[k1]
        w[days] = 1.0
        cost[days[0]] += 2e-4
    for y in range(2019, 2026):
        cny = pd.Timestamp(LunarDate(y, 1, 1).toSolarDate())
        b = idx[idx < cny][-15:]
        if len(b):
            w[b] = 1.0
            cost[b[0]] += 2e-4
    ret = r * w - cost
    return ret[(ret.index >= "2019-01-01") & (ret.index <= "2025-12-31")], w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = pd.read_csv(os.path.join(ROOT, "docs/gotobi_losses/trades_basket_stop_80_bp.csv"), index_col=0, parse_dates=True)
    B.index = pd.DatetimeIndex(B.index).normalize()
    B = B[(B.index >= "2019-01-01") & (B.index <= "2025-12-31")][["net", "mae"]].dropna()
    G, w = gold_days()
    res = {"gold_days": int((w[w.index >= "2019-01-01"] > 0).sum()), "gold_mean_bp_active_day":
           float(G[G != 0].mean() * 1e4)}
    for scen, target in (("2019-2025", None), ("2026", 2.04e-4)):
        g = B.copy()
        if target is not None:
            d = g.net.mean() - target
            g["net"] = g.net - d
            g["mae"] = np.minimum(g.mae - d, np.minimum(g.net, 0.0))
        X, M = PB.daily(g)
        days = pd.bdate_range(g.index.min(), g.index.max())
        Gd = G.reindex(days).fillna(0.0).to_numpy() * (2 / 5)              # 2x inside a sim that scales by 5
        for name, XX, MM in (("Gotobi alone", X, M), ("Gotobi + gold calendar", X + Gd, np.minimum(M + np.minimum(Gd, 0), 0))):
            paid, npay, passed, first, fail = PB.simulate(XX, MM, PB.P["FTMO 2-Step"], a.n, 1, 5, 5)
            f = first.astype(float)
            f[f < 0] = np.inf
            res[f"{scen}|{name}"] = {"pass": float(passed.mean()), "paid": float((npay > 0).mean()),
                                     "mean_net_100k": float((paid * 1e5 + 621 * (npay >= 1) - 621).mean()),
                                     "paid_within_12m": float((f <= 252).mean())}
    res["counts"] = bool(res["2026|Gotobi + gold calendar"]["paid"] >= res["2026|Gotobi alone"]["paid"] + 0.03)
    json.dump(res, open(os.path.join(a.out, "e17.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
