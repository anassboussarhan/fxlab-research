"""Join downloaded M1 parquet parts (mirror years, Dukascopy months/quarters/hour subsets) into the store layout
data/m1/SYM/SYM_YEAR.parquet; also gathers the acquisition manifests into one coverage file."""
import glob
import json
import os
import sys

import polars as pl

src, out_dir = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/m1"
groups = {}
for f in glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True):
    parts = os.path.basename(f)[:-8].split("_")
    if len(parts) < 2 or not parts[1].isdigit():
        continue
    groups.setdefault((parts[0], int(parts[1])), []).append(f)
for (sym, y), fs in sorted(groups.items()):
    df = pl.concat([pl.read_parquet(f).with_columns(pl.col("ts").dt.cast_time_unit("ms")) for f in fs],
                   how="diagonal_relaxed")
    df = df.unique(subset=["ts"], keep="first").sort("ts")
    os.makedirs(os.path.join(out_dir, sym), exist_ok=True)
    df.write_parquet(os.path.join(out_dir, sym, f"{sym}_{y}.parquet"), compression="zstd")
    print(sym, y, len(fs), "parts", df.height, "minutes", flush=True)
recs = []
for f in glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True):
    for line in open(f):
        if line.strip():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            recs.append({k: r.get(k) for k in ("pair", "year", "status", "source", "n_minutes", "first_ts", "last_ts",
                                               "parquet")} | {"hours_failed": (r.get("drops") or {}).get(
                                                   "hours_failed_download")})
os.makedirs("results", exist_ok=True)
json.dump(recs, open("results/m1_coverage.json", "w"), indent=0)
print(len(recs), "manifest records;", sum(1 for r in recs if r["status"] != "ok"), "not ok")
