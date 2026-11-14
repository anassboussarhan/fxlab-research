"""The noise-area breakout on the Nasdaq (round 19, A-NB-NAS100), unchanged, on data after 2025.

Used for the one-time 2026 check (docs/nb2026_preregistration.md) and the weekly paper forward test.
The rule, costs and code are those of scripts/r19.py (session_days, nb_day, costs_of); only the data differ.

Usage: python scripts/nb_forward.py --extra DIR --from 2026-01-01 --to 2026-10-31 --out DIR [--label h26]
  --extra: Dukascopy minute files for the new period (any *.parquet below DIR, store schema).
  The 14-day warm-up and the previous close come from the development store (December 2025) when --from is in
  early 2026, otherwise from the extra files themselves.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r19  # noqa: E402

SYM = "USATECHIDXUSD"
DEV_MEAN_BP, DEV_T, DEV_N = 3.268, 2.2524, 1652           # round 19, 2019-2025, at NQ futures costs


def load(extra, start):
    parts = []
    if pd.Timestamp(start) <= pd.Timestamp("2026-02-15"):
        from fxlab.data.store import load_m1
        try:
            parts.append(load_m1(SYM, "2025-11-01", r19.DEV_END, columns=r19.COLS + ["n_ticks"]))
        except Exception:                                                  # noqa: BLE001
            parts.append(load_m1(SYM, "2025-11-01", r19.DEV_END, columns=r19.COLS))
    for f in sorted(glob.glob(os.path.join(extra, "**", "*.parquet"), recursive=True)):
        x = pd.read_parquet(f)
        if "ts" in x.columns:
            x = x.set_index("ts")
        x.index = pd.DatetimeIndex(x.index).tz_convert("UTC") if x.index.tz is not None else pd.DatetimeIndex(x.index).tz_localize("UTC")
        parts.append(x[[c for c in r19.COLS + ["n_ticks"] if c in x.columns]])
    m = pd.concat(parts).sort_index()
    return m[~m.index.duplicated(keep="last")]


def run(m1):
    pip = float(1e-4 * np.nanmedian(0.5 * (m1.bid_c + m1.ask_c)))
    D = r19.session_days(m1)
    rows, moves, closes = [], [], []
    for day in sorted(D):
        A = D[day]
        if len(moves) >= 14 and closes:
            sigma = np.mean(np.vstack(moves[-14:]), axis=0)
            t = r19.nb_day(A, sigma, closes[-1])
            g, fu, cf = r19.costs_of(t, A, r19.FUT_PTS[SYM], pip) if t else (0.0, 0.0, 0.0)
            dr = np.diff(np.log(np.r_[closes[-15:]])) if len(closes) >= 15 else None
            lev = min(4.0, 0.02 / float(np.std(dr, ddof=1))) if dr is not None and np.std(dr) > 0 else np.nan
            rows.append({"day": day, "trades": len(t), "sides": ",".join("L" if x[0] > 0 else "S" for x in t),
                         "gross": g, "net_fut": g - fu, "net_fut2": g - 2 * fu, "net_cfd": g - cf, "lev": lev})
        moves.append(np.abs(A["c"] / A["o"][0] - 1.0))
        closes.append(A["c"][-1])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", required=True)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="h26")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = run(load(a.extra, a.start))
    R = R[(R.day >= pd.Timestamp(a.start)) & (R.day <= pd.Timestamp(a.end))].reset_index(drop=True)
    R.to_csv(os.path.join(a.out, f"nb_{a.label}_days.csv"), index=False)
    x = R.net_fut.to_numpy(float)
    n = len(x)
    m = float(x.mean()) if n else float("nan")
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    t = m / se if se and se > 0 else float("nan")
    se_dev = DEV_MEAN_BP / DEV_T * 1e-4
    t_diff = (m - DEV_MEAN_BP * 1e-4) / math.sqrt(se ** 2 + se_dev ** 2) if n > 1 else float("nan")
    verdict = "supported" if (m > 0 and t >= 1.645) else "contradicted" if t_diff <= -1.645 else "inconclusive"
    v = (R.net_fut * R.lev).dropna()
    summ = {"label": a.label, "from": a.start, "to": a.end, "days": n, "first_day": str(R.day.min()) if n else None,
            "last_day": str(R.day.max()) if n else None, "days_traded": int((R.trades > 0).sum()),
            "net_fut_bp": m * 1e4, "t": t, "t_vs_dev_mean": t_diff, "verdict": verdict,
            "gross_bp": float(R.gross.mean() * 1e4) if n else None, "net_fut2_bp": float(R.net_fut2.mean() * 1e4) if n else None,
            "net_cfd_bp": float(R.net_cfd.mean() * 1e4) if n else None,
            "volsized_return_pct": float(v.sum() * 100), "volsized_maxdd_pct": float((v.cumsum() - v.cumsum().cummax()).min() * 100) if len(v) else None,
            "by_month_bp": {str(k): float(g.net_fut.mean() * 1e4) for k, g in R.groupby(R.day.dt.to_period("M"))},
            "dev": {"mean_bp": DEV_MEAN_BP, "t": DEV_T, "n": DEV_N},
            "power_note": "expected t at the development mean is about mean / (sd / sqrt(n))"}
    json.dump(summ, open(os.path.join(a.out, f"nb_{a.label}_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
