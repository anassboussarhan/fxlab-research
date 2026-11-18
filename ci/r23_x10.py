"""Round 23 X10 data rule (docs/round23_preregistration.md, X10 and addendum 4) for the wave-2 instruments.

An instrument is used only if
- its history starts at least 3 years before 2019 (first month with data January 2016 or earlier), and
- its 2024 median close, after the power-of-ten rescale of ci/r23_rescale.py, is within a factor of 2 of the level
  in ci/r23_expected_2024.json.
Reads the hour-candle artifacts at DIR/r23h-SYM/*.parquet. Writes JSON {symbol: {...}} with the coverage per instrument.
Usage: python ci/r23_x10.py DIR OUT.json
"""
import glob
import json
import math
import os
import sys

import pandas as pd

d, out = sys.argv[1], sys.argv[2]
exp = json.load(open("ci/r23_expected_2024.json"))
res = {}
for p in sorted(glob.glob(os.path.join(d, "r23h-*"))):
    s = os.path.basename(p)[5:]
    fs = sorted(glob.glob(os.path.join(p, "*.parquet")))
    if not fs:
        res[s] = {"used": False, "why": "no files"}
        continue
    ts = pd.concat([pd.read_parquet(f, columns=["ts"]).ts for f in fs])
    months = sorted(set(zip(ts.dt.year, ts.dt.month)))
    first, last = months[0], months[-1]
    span = (last[0] - first[0]) * 12 + last[1] - first[1] + 1
    missing = [f"{y}-{m:02d}" for y in range(first[0], last[0] + 1) for m in range(1, 13)
               if (y, m) not in set(months) and (y, m) >= first and (y, m) <= last]
    ref = [f for f in fs if "2024" in os.path.basename(f)] or fs[-1:]
    med = float(pd.read_parquet(ref[0], columns=["bid_c"]).bid_c.median())
    e = exp.get(s)
    k = round(math.log10(e / med)) if (e and med > 0) else 0
    ratio = med * 10.0 ** k / e if e else None
    hist_ok = first <= (2016, 1)
    price_ok = ratio is not None and 0.5 <= ratio <= 2.0
    res[s] = {"first_month": f"{first[0]}-{first[1]:02d}", "last_month": f"{last[0]}-{last[1]:02d}", "months": len(months),
              "span_months": span, "missing_months": missing, "hours": int(len(ts)),
              "hours_by_year": {str(y): int(n) for y, n in ts.dt.year.value_counts().sort_index().items()},
              "median_2024_rescaled": med * 10.0 ** k, "expected_2024": e, "ratio": ratio,
              "history_ok": hist_ok, "price_ok": price_ok, "used": bool(hist_ok and price_ok)}
json.dump(res, open(out, "w"), indent=1)
for s, v in res.items():
    print(s, {k: v.get(k) for k in ("first_month", "months", "span_months", "ratio", "used")}, "missing", len(v.get("missing_months", [])))
