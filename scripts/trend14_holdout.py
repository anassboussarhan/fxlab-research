"""Round 14 part D (docs/trend14_preregistration.md): Double Seven on weekly bars, 2026, third use of the holdout.

ECB reference rates (the 28 crosses of EUR, GBP, AUD, NZD, USD, CAD, CHF, JPY). Weekly close = the last fixing of a
Monday-Friday week. Long when the weekly close is above its 200-week average and is the lowest close of 7 weeks;
out after a 7-week highest close. Fill at the next fixing after the signal week; 3 bp a round trip; no swap.
Per-trade results in R (return / (2 x weekly N / price), weekly N = 20-week EWMA of absolute weekly changes) and in %.
Development comparator: trades entered 2019-01-01..2025-12-31 and closed by 2025-12-31. Holdout: entered from
2026-01-01 and closed by 2026-10-31. Usage: python scripts/trend14_holdout.py --ecb exr.csv --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

CUR8 = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")
FX = sorted({a + b for i, a in enumerate(CUR8) for b in CUR8[i + 1:]})
COST = 3e-4
HOLDOUT_END = pd.Timestamp("2026-10-31")


def trades_for(px: pd.Series):
    iso = px.index.isocalendar()
    key = (iso.year * 100 + iso.week).to_numpy()
    wk_last = np.r_[key[1:] != key[:-1], True]
    W = px[wk_last]
    sma200 = W.rolling(200).mean()
    lo7, hi7 = W.rolling(7).min(), W.rolling(7).max()
    N = W.diff().abs().ewm(alpha=1 / 20, adjust=False, min_periods=20).mean()
    days = px.index
    out = []
    state, entry = 0, None
    for i, t in enumerate(W.index):
        if i + 1 >= len(W):
            break
        nxt = days[days > t]
        if not len(nxt):
            break
        fill_t = nxt[0]
        if state == 0 and np.isfinite(sma200.iloc[i]) and W.iloc[i] > sma200.iloc[i] and W.iloc[i] <= lo7.iloc[i]:
            state, entry = 1, (fill_t, px.loc[fill_t], N.iloc[i] / W.iloc[i])
        elif state == 1 and W.iloc[i] >= hi7.iloc[i]:
            r = px.loc[fill_t] / entry[1] - 1 - COST
            out.append({"entry": entry[0], "exit": fill_t, "ret": r, "R": r / (2 * entry[2]) if entry[2] > 0 else np.nan})
            state, entry = 0, None
    return out


def summ(T):
    if not len(T):
        return {"trades": 0}
    x = T.R.dropna()
    return {"trades": int(len(T)), "mean_R": float(x.mean()), "sd_R": float(x.std()),
            "t": float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan,
            "mean_pct": float(T.ret.mean() * 100), "win_rate": float((T.ret > 0).mean()),
            "median_hold_days": float((T.exit - T.entry).dt.days.median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    E = pd.read_csv(a.ecb, usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    E["TIME_PERIOD"] = pd.to_datetime(E.TIME_PERIOD)
    E = E[E.TIME_PERIOD <= HOLDOUT_END]
    X = E.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
    X["EUR"] = 1.0
    rows = []
    for s in FX:
        px = (X[s[3:]] / X[s[:3]]).dropna()
        for tr in trades_for(px):
            tr["symbol"] = s
            rows.append(tr)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "D_trades.csv"), index=False)
    dev = T[(T.entry >= "2019-01-01") & (T.exit <= "2025-12-31")]
    fit = T[(T.entry >= "2008-01-01") & (T.exit <= "2018-12-31")]
    h26 = T[(T.entry >= "2026-01-01") & (T.exit <= HOLDOUT_END)]
    open26 = T[(T.entry >= "2026-01-01") & (T.exit > HOLDOUT_END)]
    r = {"fit_2008_2018": summ(fit), "dev_2019_2025": summ(dev), "holdout_2026": summ(h26),
         "open_at_holdout_end": int(len(open26))}
    d, h = r["dev_2019_2025"], r["holdout_2026"]
    if h.get("trades", 0) >= 2 and d.get("trades", 0) >= 2:
        diff_t = (h["mean_R"] - d["mean_R"]) / (h["sd_R"] / math.sqrt(h["trades"]))
        r["diff_t_vs_dev"] = float(diff_t)
        if d["mean_R"] <= 0:
            r["verdict"] = "moot (the ECB development comparator is not positive)"
        elif h["mean_R"] > 0 and h["t"] >= 1.645:
            r["verdict"] = "supported"
        elif diff_t <= -1.645:
            r["verdict"] = "contradicted"
        else:
            r["verdict"] = "inconclusive"
    json.dump(r, open(os.path.join(a.out, "D_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(r, indent=1, default=str))
    if len(h26):
        print(h26.sort_values("entry").to_string(index=False))


if __name__ == "__main__":
    main()
