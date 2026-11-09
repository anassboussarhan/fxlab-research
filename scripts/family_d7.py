"""Round 15 addendum 2 (docs/round15_preregistration.md): the weekly Connors dip-buying family on 2026.

ECB fixings for the 28 crosses; weekly close = last fixing of a Monday-Friday week; the catalog rules at their
published defaults on close-only weekly bars, long only; fill at the next fixing after the signal week; 3 bp a
round trip; R = return / (2 x weekly N / entry), weekly N = 20-week EWMA of absolute weekly changes.
Usage: python scripts/family_d7.py --ecb exr.csv --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUR8 = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")
FX = sorted({a + b for i, a in enumerate(CUR8) for b in CUR8[i + 1:]})
RULES = {"S.DAILY.DOUBLE_SEVEN": {}, "S.DAILY.CUM_RSI": {}, "S.FAMOUS.CONNORS_RSI2": {}, "S.BOOK.CONNORS_R3": {},
         "S.BOOK.CONNORS_PCTB": {}, "S.BOOK.CONNORS_CRSI": {}, "S.DAILY.THREE_DOWN": {}}
COST = 3e-4
H_END = pd.Timestamp("2026-10-31")
DEV_LO, DEV_HI = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-31")


def stats(T):
    if not len(T):
        return {"trades": 0}
    x = T.R.astype(float)
    n, m = len(x), x.mean()
    g = (x - m).groupby(pd.to_datetime(T.exit).dt.to_period("M")).sum()
    se = math.sqrt((g ** 2).sum()) / n
    return {"trades": int(n), "mean_R": float(m), "t_month": float(m / se) if se > 0 else np.nan,
            "win_rate": float((x > 0).mean()), "sum_R": float(x.sum()), "worst_R": float(x.min()), "sd_R": float(x.std())}


def main():
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    E = pd.read_csv(a.ecb, usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    E["TIME_PERIOD"] = pd.to_datetime(E.TIME_PERIOD)
    E = E[E.TIME_PERIOD <= H_END]
    X = E.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
    X["EUR"] = 1.0
    rows = []
    for s in FX:
        px = (X[s[3:]] / X[s[:3]]).dropna()
        iso = px.index.isocalendar()
        key = (iso.year * 100 + iso.week).to_numpy()
        last = np.r_[key[1:] != key[:-1], True]
        w = px[last]
        nxt = pd.Series(px.index, index=px.index).shift(-1)[last]
        fill_t = nxt.to_numpy()
        fill_px = px.reindex(nxt.to_numpy()).to_numpy()
        N = w.diff().abs().ewm(alpha=1 / 20, adjust=False, min_periods=20).mean().to_numpy()
        wv = w.to_numpy(float)
        bars = pd.DataFrame({"open": np.r_[wv[0], wv[:-1]], "high": wv, "low": wv, "close": wv},
                            index=pd.DatetimeIndex(w.index).tz_localize("UTC"))
        for rid, extra in RULES.items():
            e = CATALOG[rid]
            side = e.fn(bars.copy(), **(e.params | extra))["side"].clip(lower=0).to_numpy()
            state, ent = 0, None
            for i in range(len(side)):
                if pd.isna(fill_px[i]):
                    break
                if state == 0 and side[i] > 0 and np.isfinite(N[i]):
                    state, ent = 1, (fill_t[i], fill_px[i], N[i] / w.iloc[i])
                elif state == 1 and side[i] <= 0:
                    r = fill_px[i] / ent[1] - 1 - COST
                    rows.append({"rule": rid, "symbol": s, "entry": ent[0], "exit": fill_t[i], "ret": r, "R": r / (2 * ent[2])})
                    state, ent = 0, None
    T = pd.DataFrame(rows)
    T["entry"], T["exit"] = pd.to_datetime(T.entry), pd.to_datetime(T.exit)
    T.to_csv(os.path.join(a.out, "family_trades.csv"), index=False)
    per = {"fit_2008_2018": (pd.Timestamp("2008-01-01"), pd.Timestamp("2018-12-31")), "dev_2019_2025": (DEV_LO, DEV_HI),
           "holdout_2026": (pd.Timestamp("2026-01-01"), H_END)}
    out = {}
    for label, sub in (("family", T), ("family_without_double_seven", T[T.rule != "S.DAILY.DOUBLE_SEVEN"])):
        out[label] = {k: stats(sub[(sub.entry >= lo) & (sub.exit <= hi)]) for k, (lo, hi) in per.items()}
    out["by_rule"] = {rid: {k: stats(T[(T.rule == rid) & (T.entry >= lo) & (T.exit <= hi)]) for k, (lo, hi) in per.items()}
                      for rid in RULES}
    for label in ("family", "family_without_double_seven"):
        d, h = out[label]["dev_2019_2025"], out[label]["holdout_2026"]
        if h.get("trades", 0) > 1 and d.get("trades", 0) > 1:
            diff_t = (h["mean_R"] - d["mean_R"]) / (h["sd_R"] / math.sqrt(h["trades"]))
            if d["mean_R"] <= 0:
                v = "moot (comparator not positive)"
            elif h["mean_R"] > 0 and h["t_month"] >= 1.645:
                v = "supported"
            elif diff_t <= -1.645:
                v = "contradicted"
            else:
                v = "inconclusive"
            out[label]["verdict"] = v
            out[label]["diff_t_vs_dev"] = float(diff_t)
    json.dump(out, open(os.path.join(a.out, "family_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("family", "family_without_double_seven")}, indent=1, default=str))
    for rid, v in out["by_rule"].items():
        print(rid, {k: (vv.get("trades"), round(vv.get("mean_R", np.nan), 3), round(vv.get("t_month", np.nan), 2)) for k, vv in v.items()})


if __name__ == "__main__":
    main()
