"""Round 25 data: refill USA500IDXUSD after its minute store stops in 2024 (round 24 addendum 3).

Inputs:
- chk/store/*.parquet: the minute store (as placed by ci/r23_place.sh);
- data/m1/USA500IDXUSD/*.parquet: Dukascopy hour candles for 2024-2025 (ci/r23_hours.py, M1 schema, one row per hour).

The store's 2024 hours are compared with the download (mid closes). If the median absolute difference is at most 1 bp
(the round-23 format check), the download replaces the store from the first month the store lacks, and the H1-W1
bars are rebuilt as round 24 builds them (scripts/r23_LW.py: resample, D2 / W1 from D1). Otherwise the round-24
bars are kept (exit code 2) and that is reported. Only counts and differences are printed, no prices.
Usage: python ci/r25_usa500.py --out bars_fix
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SYM = "USA500IDXUSD"
DEV_END = pd.Timestamp("2025-12-31 23:59", tz="UTC")


def read(files):
    x = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "ts" in x:
        x = x.set_index(pd.DatetimeIndex(x["ts"])).drop(columns="ts")
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    return x[~x.index.duplicated()].sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    store = read(sorted(glob.glob("chk/store/*.parquet")))
    hours = read(sorted(glob.glob(f"data/m1/{SYM}/*.parquet")))
    hours = hours[hours.index <= DEV_END]
    smon = store.index.to_period("M").unique()
    hmon = hours.index.to_period("M").unique()
    mid = lambda x: (x.bid_c + x.ask_c) / 2                                # noqa: E731
    hc = mid(store).resample("1h").last().dropna()
    hh = mid(hours)
    j = pd.concat([hc.rename("store"), hh.rename("hour")], axis=1).dropna()
    j = j[j.index.year == 2024]
    d = (j.hour / j.store - 1).abs()
    rep = {"store_last": str(store.index.max()), "store_months_2024": [str(m) for m in smon if m.year == 2024],
           "hour_months": len(hmon), "hour_first": str(hours.index.min()), "hour_last": str(hours.index.max()),
           "hours_compared": int(len(j)), "median_abs_diff_bp": float(d.median() * 1e4) if len(d) else None,
           "p99_abs_diff_bp": float(d.quantile(0.99) * 1e4) if len(d) else None}
    ok = len(j) >= 200 and rep["median_abs_diff_bp"] is not None and rep["median_abs_diff_bp"] <= 1.0
    rep["accepted"] = bool(ok)
    if not ok:
        print(json.dumps(rep, indent=1))
        json.dump(rep, open(os.path.join(a.out, "usa500_fix.json"), "w"), indent=1)
        sys.exit(2)
    # the first month the store lacks: after its last complete month
    last_full = store.index.max().to_period("M")
    if store.index.max() < (last_full.to_timestamp(how="end").tz_localize("UTC") - pd.Timedelta(days=3)):
        last_full = last_full - 1                                          # the store's last month is partial
    cut = (last_full + 1).to_timestamp().tz_localize("UTC")
    px = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
    h2 = hours.loc[hours.index >= cut, [c for c in hours.columns if c in store.columns or c in px]].copy()
    if "n_ticks" in store.columns and "n_ticks" not in h2:
        h2["n_ticks"] = 1.0                                                # candle rows, as resample() fills them
    if "spread_mean" in store.columns and "spread_mean" not in h2:
        h2["spread_mean"] = h2["ask_c"] - h2["bid_c"]
    m = pd.concat([store.loc[store.index < cut], h2]).sort_index()
    rep["splice_from"] = str(cut)
    rep["store_columns_missing_in_hours"] = [c for c in store.columns if c not in hours.columns]
    from fxlab.data.store import resample
    from r23_LW import higher
    d1 = None
    for tf in ("H1", "H2", "H4", "H8", "H12", "D1", "D2", "W1"):
        if tf in ("D2", "W1"):
            b = higher(d1, tf)
        else:
            b = resample(m, tf)
            if tf == "D1":
                d1 = b
        b = b[b.index <= DEV_END]
        b.to_parquet(os.path.join(a.out, f"{SYM}_{tf}.parquet"), compression="zstd")
        rep[f"bars_{tf}"] = int(len(b))
        rep[f"years_{tf}"] = sorted({int(y) for y in b.index.year})[-3:]
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(a.out, "usa500_fix.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
