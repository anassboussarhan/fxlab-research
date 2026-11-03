"""Exit / entry timing audit of the frozen Gotobi basket (B0: long USD/JPY + EUR/JPY, 23:00 Tokyo -> 09:53 Tokyo,
80 bp stop), 2007-2025 only. The development code takes the first M1 bar within 10 minutes of the entry and within
65 minutes of the exit (the 2007-2018 files have no 00:00-01:00 UTC bars, and 09:53 Tokyo = 00:53 UTC). Here: the
delay distribution by period, and the basket redone with strict endpoints (bar within 2 minutes of both; nights
without such bars dropped), paired against the lenient version on the same nights. Gross of commission and swap
(both identical between versions). Usage: python scripts/gotobi_strict_timing.py --out DIR"""
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
import prop_checks as P  # noqa: E402
import untested_intraday_lab as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    legs = {}
    for p in ("USDJPY", "EURJPY"):
        bk = U.book(p)
        bk._hl()
        G, _, _ = F.calendar(bk)
        rows = {}
        for d in G:
            if pd.Timestamp(d) > pd.Timestamp("2025-12-31"):
                continue
            t0, t1 = U.jst(d - F.DAY, 23, 0), U.jst(d, 9, 53)
            i, j = bk.at(t0, 10), bk.at(t1, 65)
            if i < 0 or j <= i:
                continue
            r = {"delay_in": (bk.idx[i] - t0).total_seconds() / 60, "delay_out": (bk.idx[j] - t1).total_seconds() / 60,
                 "lenient": P.leg(bk, i, j)[1]}
            i2, j2 = bk.at(t0, 2), bk.at(t1, 2)
            r["strict"] = P.leg(bk, i2, j2)[1] if (i2 >= 0 and j2 > i2) else np.nan
            rows[pd.Timestamp(d)] = r
        legs[p] = pd.DataFrame.from_dict(rows, orient="index")
        print(p, len(rows), flush=True)
    A, B = legs["USDJPY"], legs["EURJPY"]
    ix = A.index.intersection(B.index)
    out = {}
    for per, lo, hi in (("2007-2018", "2007", "2018"), ("2019-2025", "2019", "2025"), ("2023-2025", "2023", "2025")):
        m = (ix >= lo) & (ix <= hi + "-12-31")
        a_, b_ = A.loc[ix[m]], B.loc[ix[m]]
        dly = pd.concat([a_.delay_out, b_.delay_out])
        len_ = (a_.lenient + b_.lenient) / 2
        st = (a_.strict + b_.strict) / 2
        both = st.notna()
        d = (st - len_)[both]
        out[per] = {"nights": int(m.sum()),
                    "exit_delay_min": {"exact(<=2)": int((dly <= 2).sum()), "3-10": int(((dly > 2) & (dly <= 10)).sum()),
                                       ">10": int((dly > 10).sum()), "max": float(dly.max())},
                    "entry_delay_gt2": int((pd.concat([a_.delay_in, b_.delay_in]) > 2).sum()),
                    "nights_with_strict_endpoints": int(both.sum()),
                    "lenient_gross_bp_same_nights": float(len_[both].mean() * 1e4),
                    "strict_gross_bp": float(st[both].mean() * 1e4),
                    "strict_minus_lenient_bp": float(d.mean() * 1e4),
                    "t_diff": float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else None,
                    "strict_t": float(st[both].mean() / st[both].std() * np.sqrt(both.sum())) if both.sum() > 2 else None}
    json.dump(out, open(os.path.join(a.out, "strict_timing.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
