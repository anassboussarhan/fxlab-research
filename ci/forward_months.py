"""Months of USD/JPY and EUR/JPY M1 the forward test needs (2025-07 to the last complete month) that the stored
parts (ARG 1, a folder of {PAIR}_{YYYY}_{MM}.parquet files from the previous run) do not hold yet. Only complete
months are fetched. Prints a GitHub matrix as JSON."""
import datetime as dt
import glob
import json
import os
import sys

store = sys.argv[1] if len(sys.argv) > 1 else "store"
have = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(store, "**", "*.parquet"), recursive=True)}
today = dt.date.today()
last = (today.replace(day=1) - dt.timedelta(days=1))                    # last day of the previous month
months, y, m = [], 2025, 7
while (y, m) <= (last.year, last.month):
    months.append((y, m))
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
need = []
for pair in ("USDJPY", "EURJPY"):
    for ym in months:
        if f"{pair}_{ym[0]}_{ym[1]:02d}" not in have:
            need.append({"pair": pair, "ym": f"{ym[0]}-{ym[1]:02d}"})
print(json.dumps({"include": need}))
