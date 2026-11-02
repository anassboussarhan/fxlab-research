"""Two practical checks for running the Gotobi trade on a prop account, on 2007-2018 nights only (the M1 store held
locally; no 2019+ or 2026 data is read).
1. MIDNIGHT RULE: The5ers measures the daily loss from the higher of balance and equity at MT5 server midnight
   (17:00 New York = 21:00 UTC in US summer time, 22:00 UTC otherwise), which falls inside the 14:00 -> 00:53 UTC trade.
   For each night: the basket's open P&L path with the 80 bp stop; "old" = the worst open loss (what the prop
   simulator counts), "fivers" = the larger of the loss before server midnight and, after it, the drop from
   max(0, open P&L at midnight) to the lowest point. Breaches counted for leverage 3-6x against a 5% daily limit.
   FTMO (midnight balance) and FundedNext (start-of-day balance) are covered by "old".
2. LATER ENTRY: FTMO Standard funded and The5ers forbid opening a trade within 2 minutes of a high-impact release,
   and FundedNext keeps 40% of the profit of a funded trade opened within 5 minutes of one; 10:00 New York releases
   fall at 14:00 UTC in US summer time. Entry 0, 3, 6, 10 and 15 minutes after 14:00 UTC (the bar at or after),
   the same exit, fills (ask + slippage in, bid - slippage out) and 80 bp stop; gross of commission and swap (the
   same for every delay); paired difference vs 14:00.
Usage: python scripts/prop_checks.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

STOP = 0.008
DELAYS = (0, 3, 6, 10, 15)


def leg(bk, i, j):
    """Open P&L path (bid marks) of a long from bar i to bar j with the 80 bp stop, and the final net (gross)."""
    E = bk.ao[i] + R.slip(bk, i)
    S = E * (1 - STOP)
    path = bk.bl[i:j + 1] / E - 1.0
    q = np.flatnonzero(bk.bl[i:j] <= S)
    if len(q):
        q = int(q[0]) + i
        X = (min(S, bk.bo[q]) if q > i else S) - R.slip(bk, q)
        path[q - i:] = X / E - 1.0
    else:
        X = bk.bo[j] - R.slip(bk, j)
        path[-1] = X / E - 1.0
    return pd.Series(path, index=bk.idx[i:j + 1]), X / E - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths, delay = {}, {}
    for p in ("USDJPY", "EURJPY"):
        bk = U.book(p)
        bk._hl()
        G, _, _ = F.calendar(bk)
        P_, D_ = {}, {}
        for d in G:
            if pd.Timestamp(d) > pd.Timestamp("2018-12-31"):
                continue
            i0, j = F.window(bk, d)
            if i0 < 0 or not np.isfinite([bk.bo[i0], bk.ao[i0], bk.bo[j]]).all():
                continue
            P_[d] = leg(bk, i0, j)[0]
            r = {}
            for k in DELAYS:
                i = int(bk.idx.searchsorted(bk.idx[i0] + pd.Timedelta(minutes=k)))
                r[k] = leg(bk, i, j)[1] if i < j and np.isfinite(bk.ao[i]) else np.nan
            D_[d] = r
        paths[p], delay[p] = P_, pd.DataFrame.from_dict(D_, orient="index")
        print(p, len(P_), flush=True)
    rows = []
    for d in sorted(set(paths["USDJPY"]) & set(paths["EURJPY"])):
        A, B = paths["USDJPY"][d], paths["EURJPY"][d]
        ix = A.index.union(B.index)
        pth = (A.reindex(ix).ffill().fillna(0) + B.reindex(ix).ffill().fillna(0)) / 2
        t0 = ix[0]
        mid = (t0.tz_convert("America/New_York").normalize() + pd.Timedelta(hours=17)).tz_convert("UTC")
        if mid <= t0:
            mid += pd.Timedelta(days=1)
        before, after = pth[pth.index < mid], pth[pth.index >= mid]
        at_mid = float(before.iloc[-1]) if len(before) else 0.0
        ref = max(at_mid, 0.0)
        d1 = -min(before.min() if len(before) else 0.0, 0.0)
        d2 = ref - min(after.min() if len(after) else ref, ref)
        rows.append({"date": d, "open_pnl_at_midnight": at_mid, "old": -min(pth.min(), 0.0), "fivers": max(d1, d2),
                     "midnight_utc_hour": mid.hour})
    M = pd.DataFrame(rows).set_index("date")
    M.to_csv(os.path.join(a.out, "midnight_2007_2018.csv"))
    out = {"nights": int(len(M)), "midnight": {}, "delay": {}}
    for L in (3, 3.5, 4, 4.5, 5, 6):
        lim = 0.05 / L
        out["midnight"][f"{L}x"] = {"breaches_old": int((M.old >= lim).sum()), "breaches_fivers": int((M.fivers >= lim).sum())}
    Bk = ((delay["USDJPY"] + delay["EURJPY"].reindex(delay["USDJPY"].index)) / 2).dropna()
    for k in DELAYS:
        dd = Bk[k] - Bk[0]
        t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else 0.0
        out["delay"][f"+{k} min"] = {"mean_gross_bp": float(Bk[k].mean() * 1e4), "diff_bp": float(dd.mean() * 1e4), "t": t}
    out["delay_nights"] = int(len(Bk))
    json.dump(out, open(os.path.join(a.out, "checks.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
