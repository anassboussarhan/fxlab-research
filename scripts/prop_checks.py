"""Practical checks for running the Gotobi trade on a prop account, on 2007-2025 development nights (2007-2018 and
2019-2025 reported apart; nothing from 2026). Added 2026-11-03: Maven's daily loss from the higher of balance and
equity at 00:00 UTC (only the 2019-2025 files have 00:00-01:00 UTC bars), and the nights whose 14:00 UTC entry falls
on a scheduled US release (calendar dataset, 2017-2023; --raw DIR).
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
    ap.add_argument("--raw", default="raw")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths, delay = {}, {}
    for p in ("USDJPY", "EURJPY"):
        bk = U.book(p)
        bk._hl()
        G, _, _ = F.calendar(bk)
        P_, D_ = {}, {}
        for d in G:
            if pd.Timestamp(d) > pd.Timestamp("2025-12-31"):
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
        mz = pd.Timestamp(pd.Timestamp(d).date()).tz_localize("UTC")          # Maven: 00:00 UTC on the Gotobi day
        bz, az = pth[pth.index < mz], pth[pth.index >= mz]
        has0 = bool(len(az) and (az.index[0] - mz) <= pd.Timedelta(minutes=5))
        atz = float(bz.iloc[-1]) if len(bz) else 0.0
        refz = max(atz, 0.0)
        mav = max(-min(bz.min() if len(bz) else 0.0, 0.0), refz - min(az.min() if len(az) else refz, refz)) if has0 else np.nan
        rows.append({"date": d, "open_pnl_at_midnight": at_mid, "old": -min(pth.min(), 0.0), "fivers": max(d1, d2),
                     "midnight_utc_hour": mid.hour, "maven": mav, "net_path_end": float(pth.iloc[-1])})
    M = pd.DataFrame(rows).set_index("date")
    M.index = pd.DatetimeIndex(M.index)
    M.to_csv(os.path.join(a.out, "midnight_2007_2025.csv"))
    out = {"nights": int(len(M)), "midnight": {}, "maven_0000utc_4pct_daily": {}, "delay": {}}
    for per, m in (("2007-2018", M.index <= "2018-12-31"), ("2019-2025", M.index > "2018-12-31")):
        X = M[m]
        out["midnight"][per] = {f"{L}x": {"breaches_old": int((X.old >= 0.05 / L).sum()),
                                          "breaches_fivers": int((X.fivers >= 0.05 / L).sum())} for L in (3, 3.5, 4, 4.5, 5, 6)}
        X2 = X[X.maven.notna()]
        out["maven_0000utc_4pct_daily"][per] = {"nights_with_00utc_bars": int(len(X2)), **{
            f"{L}x": {"breaches_old": int((X2.old >= 0.04 / L).sum()), "breaches_maven": int((X2.maven >= 0.04 / L).sum())}
            for L in (3, 3.5, 4)}}
    Bk = ((delay["USDJPY"] + delay["EURJPY"].reindex(delay["USDJPY"].index)) / 2).dropna()
    Bk.index = pd.DatetimeIndex(Bk.index)
    for per, m in (("2007-2018", Bk.index <= "2018-12-31"), ("2019-2025", Bk.index > "2018-12-31")):
        X = Bk[m]
        out["delay"][per] = {}
        for k in DELAYS:
            dd = X[k] - X[0]
            t = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else 0.0
            out["delay"][per][f"+{k} min"] = {"n": int(len(X)), "mean_gross_bp": float(X[k].mean() * 1e4),
                                              "diff_bp": float(dd.mean() * 1e4), "t": t}
    # nights whose 14:00 UTC entry falls on a scheduled US release (calendar dataset 2017-2023)
    import glob as _g
    rel = []
    for f in _g.glob(os.path.join(a.raw, "**", "calendar_repo", "news_data", "1*_US_*.csv"), recursive=True) + \
            _g.glob(os.path.join(a.raw, "**", "calendar_repo", "news_data", "102*_*.csv"), recursive=True):
        try:
            d = pd.read_csv(f, usecols=["Timestamp"])
            ts = pd.to_datetime(d.Timestamp)
            rel += [(t, os.path.basename(f)[:-4]) for t in ts if t.hour == 14 and t.minute <= 5]
        except Exception:                                                   # noqa: BLE001
            pass
    R_ = pd.DataFrame(rel, columns=["ts", "release"]).drop_duplicates()
    entry = pd.Series(M.index - pd.Timedelta(days=1), index=M.index)
    clash = {d: ", ".join(sorted(set(R_[R_.ts.dt.normalize() == e].release))) for d, e in entry.items()
             if (R_.ts.dt.normalize() == e).any()}
    span = (M.index >= "2017-01-01") & (M.index <= "2023-12-31")
    cl = pd.Series(clash)
    out["news_at_entry"] = {"gotobi_nights_2017_2023": int(span.sum()), "nights_with_a_14utc_release": int(len(cl)),
                            "examples": {str(k.date()): v for k, v in list(cl.items())[:12]},
                            "net_path_end_bp_on_clash_nights": float(M.net_path_end.reindex(cl.index).mean() * 1e4) if len(cl) else None,
                            "net_path_end_bp_other_nights_2017_2023": float(M.net_path_end[span & ~M.index.isin(cl.index)].mean() * 1e4)}
    out["delay_nights"] = int(len(Bk))
    json.dump(out, open(os.path.join(a.out, "checks.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
