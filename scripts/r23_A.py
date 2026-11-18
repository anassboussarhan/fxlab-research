"""Round 23 part A (docs/round23_preregistration.md): D7 statistics with trades still open at a window's end marked at
that window's last fixing (they were dropped). A correction of earlier figures, no selection.

Windows: 2008-2018, 2019-2025, 2026-01-01 .. 2026-10-31 (the earlier 2026 window). Tracks: D7-base and D7-safe
(scripts/forward_d7.py) and the seven-rule D7 family (scripts/family_d7.py). For each window two conventions:
  old        trades with entry >= start and exit <= end (as reported before)
  corrected  trades with entry in the window; those exiting after the end, or never, closed at the end's fixing
Usage: python scripts/r23_A.py --out DIR [--ecb FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forward_d7 as F  # noqa: E402

WINDOWS = {"2008_2018": ("2008-01-01", "2018-12-31"), "2019_2025": ("2019-01-01", "2025-12-31"),
           "2026": ("2026-01-01", "2026-10-31")}
FAM = ["S.DAILY.DOUBLE_SEVEN", "S.DAILY.CUM_RSI", "S.FAMOUS.CONNORS_RSI2", "S.BOOK.CONNORS_R3", "S.BOOK.CONNORS_PCTB",
       "S.BOOK.CONNORS_CRSI", "S.DAILY.THREE_DOWN"]


def stats(T):
    if not len(T):
        return {"trades": 0}
    x = T.R.astype(float)
    n, m = len(x), float(x.mean())
    g = (x - m).groupby(pd.to_datetime(T.exit).dt.to_period("M")).sum()
    se = math.sqrt((g ** 2).sum()) / n
    return {"trades": int(n), "mean_R": m, "t_month": float(m / se) if se > 0 else None, "sum_R": float(x.sum()),
            "win_rate": float((x > 0).mean()), "marked": int(T.get("marked", pd.Series(False, index=T.index)).sum())}


def windows(T, daily):
    """T: entry, exit (NaT if still open), symbol, entry_px, N (price units). daily: symbol -> daily fixings."""
    out = {}
    for k, (lo, hi) in WINDOWS.items():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        old = T[(T.entry >= lo) & T.exit.notna() & (T.exit <= hi)]
        sub = T[(T.entry >= lo) & (T.entry <= hi)].copy()
        late = sub.exit.isna() | (sub.exit > hi)
        for i in sub.index[late]:
            px = daily[sub.at[i, "symbol"]]
            px = px[px.index <= hi]
            if not len(px):
                continue
            r = px.iloc[-1] / sub.at[i, "entry_px"] - 1 - F.COST
            sub.at[i, "R"] = r / (2 * sub.at[i, "N"] / sub.at[i, "entry_px"])
            sub.at[i, "exit"] = px.index[-1]
        sub["marked"] = late
        out[k] = {"old": stats(old), "corrected": stats(sub)}
    return out


def base_trades(W, k_stop=0.0, cap=0, cur_cap=0):
    """forward_d7.sim with entry prices kept and still-open trades returned (exit NaT)."""
    weeks = sorted(set().union(*[set(d.index) for d in W.values()]))
    open_, rows = {}, []
    for t in weeks:
        for s in list(open_):
            d = W[s]
            if t not in d.index:
                continue
            r, e = d.loc[t], open_[s]
            stop_hit = k_stop > 0 and r.c < e["px"] - k_stop * e["N"]
            if (r.c >= r.hi7 or stop_hit) and not pd.isna(r.fill_px):
                ret = r.fill_px / e["px"] - 1 - F.COST
                rows.append({"entry": e["t"], "exit": r.fill_t, "symbol": s, "entry_px": e["px"], "N": e["N"],
                             "R": ret / (2 * e["N"] / e["px"])})
                del open_[s]
        for s, d in W.items():
            if s in open_ or t not in d.index:
                continue
            r = d.loc[t]
            if not (np.isfinite(r.sma) and r.c > r.sma and r.c <= r.lo7 and np.isfinite(r.N)):
                continue
            if cap and len(open_) >= cap:
                break
            if cur_cap:
                b, q = s[:3], s[3:]
                if sum(1 for o in open_ if b in (o[:3], o[3:])) >= cur_cap or sum(1 for o in open_ if q in (o[:3], o[3:])) >= cur_cap:
                    continue
            if pd.isna(r.fill_px):
                continue
            open_[s] = {"t": r.fill_t, "px": r.fill_px, "N": r.N}
    for s, e in open_.items():
        rows.append({"entry": e["t"], "exit": pd.NaT, "symbol": s, "entry_px": e["px"], "N": e["N"], "R": np.nan})
    T = pd.DataFrame(rows)
    T["entry"], T["exit"] = pd.to_datetime(T.entry), pd.to_datetime(T.exit)
    return T


def family_trades(W):
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    rows = []
    for s, d in W.items():
        wv = d.c.to_numpy(float)
        bars = pd.DataFrame({"open": np.r_[wv[0], wv[:-1]], "high": wv, "low": wv, "close": wv},
                            index=pd.DatetimeIndex(d.index).tz_localize("UTC"))
        for rid in FAM:
            e = CATALOG[rid]
            side = e.fn(bars.copy(), **e.params)["side"].clip(lower=0).to_numpy()
            state, ent = 0, None
            for i in range(len(side)):
                if pd.isna(d.fill_px.iloc[i]):
                    break
                if state == 0 and side[i] > 0 and np.isfinite(d.N.iloc[i]):
                    state, ent = 1, (d.fill_t.iloc[i], d.fill_px.iloc[i], d.N.iloc[i])
                elif state == 1 and side[i] <= 0:
                    r = d.fill_px.iloc[i] / ent[1] - 1 - F.COST
                    rows.append({"rule": rid, "symbol": s, "entry": ent[0], "exit": d.fill_t.iloc[i], "entry_px": ent[1],
                                 "N": ent[2], "R": r / (2 * ent[2] / ent[1])})
                    state, ent = 0, None
            if state == 1:
                rows.append({"rule": rid, "symbol": s, "entry": ent[0], "exit": pd.NaT, "entry_px": ent[1], "N": ent[2],
                             "R": np.nan})
    T = pd.DataFrame(rows)
    T["entry"], T["exit"] = pd.to_datetime(T.entry), pd.to_datetime(T.exit)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ecb", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = F.load(a.ecb)
    X = X[X.index <= pd.Timestamp("2026-10-31")]
    W = F.weekly(X)
    daily = {s: (X[s[3:]] / X[s[:3]]).dropna() for s in F.FX}
    rep = {"data_last_fixing": str(X.index.max().date())}
    for name, kw in (("D7-base", {}), ("D7-safe", {"k_stop": 5.0, "cap": 4, "cur_cap": 2})):
        rep[name] = windows(base_trades(W, **kw), daily)
    T = family_trades(W)
    rep["D7-family"] = windows(T, daily)
    rep["D7-family_without_double_seven"] = windows(T[T.rule != "S.DAILY.DOUBLE_SEVEN"], daily)
    rep["by_rule"] = {r: windows(T[T.rule == r], daily) for r in FAM}
    json.dump(rep, open(os.path.join(a.out, "r23_A.json"), "w"), indent=1, default=str)
    for k in ("D7-base", "D7-safe", "D7-family", "D7-family_without_double_seven"):
        print(k, {w: {c: (v[c].get("trades"), round(v[c].get("mean_R", float("nan")), 3),
                          round(v[c].get("t_month") or float("nan"), 2)) for c in ("old", "corrected")}
                  for w, v in rep[k].items()}, flush=True)


if __name__ == "__main__":
    main()
