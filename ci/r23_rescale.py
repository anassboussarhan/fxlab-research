"""Rescale a new instrument's M1 files by the power of ten that brings its 2024 median close to the expected level."""
import glob
import json
import math
import sys

import pandas as pd

S = sys.argv[1]
exp = json.load(open("ci/r23_expected_2024.json")).get(S)
fs = sorted(glob.glob(f"data/m1/{S}/*.parquet"))
ref = [f for f in fs if "2024" in f] or fs[-1:]
if not fs or exp is None:
    print(S, "no files or no expected level"); sys.exit(0)
med = float(pd.read_parquet(ref[0], columns=["bid_c"]).bid_c.median())
k = round(math.log10(exp / med)) if med > 0 else 0
print(S, "median", med, "expected", exp, "scale 10 **", k)
if k:
    cols = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "spread_mean"]
    for f in fs:
        x = pd.read_parquet(f)
        for c in cols:
            if c in x:
                x[c] = x[c] * 10.0 ** k
        x.to_parquet(f, index=False)
