"""Paper forward test of the round-22 fair-value-gap trade on Nasdaq futures (docs/fvg_forward_preregistration.md).

Frozen: the five configurations the walk-forward chose for 2026 on data up to 2025 (round 22, branch round22h):
Sobol indices 64, 56, 8, 40, 88 of the I-FVG family (scripts/r22_I.py, seed 25, 128 points), traded with equal
weight on NAS100 session bars (09:30-16:00 New York), NQ futures costs (0.375 points per side).
Usage: python scripts/fvg_forward.py --extra DIR --from 2026-11-16 --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r22_I as I  # noqa: E402

SYM = "USATECHIDXUSD"
FROZEN = [64, 56, 8, 40, 88]
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]


def load_extra(extra):
    parts = []
    for f in sorted(glob.glob(os.path.join(extra, "**", "*.parquet"), recursive=True)):
        x = pd.read_parquet(f)
        if "ts" in x.columns:
            x = x.set_index("ts")
        x.index = pd.DatetimeIndex(x.index)
        if x.index.tz is None:
            x.index = x.index.tz_localize("UTC")
        parts.append(x[COLS])
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    mid = pd.DataFrame({k: 0.5 * (m[f"bid_{k}"] + m[f"ask_{k}"]) for k in "ohlc"})
    mid["spr"] = (m.ask_o - m.bid_o).clip(lower=0)
    return mid[np.isfinite(mid.o) & (mid.o > 0)]


def book(mid, start):
    """Daily ledger (one column per frozen configuration, plus the equal-weight book) and trades from `start`."""
    pip = float(1e-4 * np.nanmedian(mid.c))                               # used only by the CFD cost path, not here
    BB = {tf: I.Bars(I.bars(mid, tf, True), SYM, pip) for tf in ("5min", "15min")}
    cal = pd.DatetimeIndex(sorted(set(BB["15min"].day)))
    U = qmc.Sobol(I.FAMS["FVG"] + 1, scramble=True, seed=22 + len("FVG")).random(128)
    cols, trades = {}, []
    for i in FROZEN:
        u = U[i]
        tf = I.pick(u[-1], ["5min", "15min"])
        tr = I.fam_FVG(BB[tf], u)
        cols[f"cfg{i}"] = I.daily(tr, BB[tf], cal)
        for t in tr:
            trades.append({"cfg": i, "tf": tf, "day": BB[tf].day[t[0]], "entry": BB[tf].idx[t[0]],
                           "exit": BB[tf].idx[t[3]], "net": t[1], "gross": t[2]})
    D = pd.DataFrame(cols)
    D["book"] = D.mean(axis=1)
    D = D[D.index >= pd.Timestamp(start)]
    T = pd.DataFrame(trades)
    if len(T):
        T = T[T.day >= pd.Timestamp(start)]
    return D, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", required=True)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D, T = book(load_extra(a.extra), a.start)
    D.to_csv(os.path.join(a.out, "fvg_forward_days.csv"))
    T.to_csv(os.path.join(a.out, "fvg_forward_trades.csv"), index=False)
    x = D.book.to_numpy(float)
    n = len(x)
    summ = {"from": a.start, "days": n, "first": str(D.index.min()) if n else None, "last": str(D.index.max()) if n else None,
            "trades": int(len(T)), "mean_bp_per_day": float(x.mean() * 1e4) if n else None,
            "t": float(x.mean() / (x.std(ddof=1) / math.sqrt(n))) if n > 2 and x.std() > 0 else None,
            "sum_bp": float(x.sum() * 1e4) if n else None,
            "dev_2019_2025_mean_bp": 7.85, "h2026_mean_bp": 10.0,
            "by_month_bp": {str(k): float(v * 1e4) for k, v in D.book.groupby(D.index.to_period("M")).mean().items()}}
    json.dump(summ, open(os.path.join(a.out, "fvg_forward_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
