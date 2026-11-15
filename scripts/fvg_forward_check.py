"""Code check for scripts/fvg_forward.py on development data only (no 2026 data).

The forward ledger is rebuilt each week from a short warm-up (two weeks of candles before the start). This checks
that a short warm-up gives the same trades as the full history: the five frozen configurations are run (a) on the
whole 2010-2025 NAS100 store and (b) on candles from two weeks before each test month only, for the last six months
of 2025; the daily values in each test month are compared.
Usage: python scripts/fvg_forward_check.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fvg_forward as F  # noqa: E402
import r22_I as I  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mid, _ = I.load(F.SYM)
    full, _ = F.book(mid, "2025-07-01")
    rows = []
    for m in pd.period_range("2025-07", "2025-12", freq="M"):
        s, e = m.start_time, m.end_time
        warm = mid[(mid.index >= pd.Timestamp(s - pd.Timedelta(days=14), tz="UTC")) & (mid.index <= pd.Timestamp(e, tz="UTC"))]
        short, _ = F.book(warm, str(s.date()))
        f = full[(full.index >= s) & (full.index <= e)]
        sh = short.reindex(f.index)
        d = (f - sh).abs()
        rows.append({"month": str(m), "days": int(len(f)), "max_abs_diff_bp": float(d.max().max() * 1e4),
                     "days_identical_book": int((d.book < 1e-12).sum()),
                     "full_mean_bp": float(f.book.mean() * 1e4), "short_mean_bp": float(sh.book.mean() * 1e4)})
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(a.out, "fvg_forward_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
