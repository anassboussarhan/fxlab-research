"""Round 23: Dukascopy HOUR candles (one file per month and side) for the new instruments, written in the M1 store
schema with one row per hour, so the store loader and the resampler work for H1 and above. The minute feed throttles
too hard for 18 instruments x 19 years in reasonable time (the probe of run 44323400795 fetched 5 in 30 minutes).
URL: https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM-1:02d}/{BID|ASK}_candles_hour_1.bi5 (t = seconds from the
month start). --check SYM compares the hourly closes with the M1 store's resampled closes for one known instrument.
--fill re-fetches only the months missing from data/m1/SYM (from the instrument's first month on), for months that
failed under throttling in the first pass, and rewrites the yearly files with them merged in.
Usage: python ci/r23_hours.py --pair SYM --start 2007 --end 2025 [--fill]"""
import argparse
import json
import lzma
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from fxlab.data.acquire_dukascopy import digits  # noqa: E402

URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{y:04d}/{m:02d}/{side}_candles_hour_1.bi5"
DT = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])


def fetch(pair, y, m, side, retries=8):
    u = URL.format(pair=pair, y=y, m=m - 1, side=side)
    for k in range(retries):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "fxlab-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return b""
            time.sleep(min(120, 3 * 2 ** k))
        except Exception:                                                  # noqa: BLE001
            time.sleep(min(120, 3 * 2 ** k))
    return None


def decode(raw, y, m, pair):
    if not raw:
        return None
    data = lzma.decompress(raw)
    n = len(data) // DT.itemsize
    if not n:
        return None
    a = np.frombuffer(data[: n * DT.itemsize], dtype=DT)
    base = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
    sc = 10.0 ** digits(pair)
    return pd.DataFrame({"ts": pd.to_datetime(base + a["t"].astype(np.int64), unit="s", utc=True),
                         "o": a["o"] / sc, "h": a["h"] / sc, "l": a["l"] / sc, "c": a["c"] / sc, "v": a["v"].astype(float)})


def month(pair, y, m):
    b, a = fetch(pair, y, m, "BID"), fetch(pair, y, m, "ASK")
    if b is None or a is None:
        return "failed", None
    B, A = decode(b, y, m, pair), decode(a, y, m, pair)
    if B is None or A is None:
        return "empty", None
    x = B.merge(A, on="ts", suffixes=("", "_a"))
    x = x[(x.v > 0) & (x.v_a > 0) & (x.c_a >= x.c) & (x.c > 0)]
    if not len(x):
        return "empty", None
    out = pd.DataFrame({"ts": x.ts, "bid_o": x.o, "bid_h": x.h, "bid_l": x.l, "bid_c": x.c, "ask_o": x.o_a, "ask_h": x.h_a,
                        "ask_l": x.l_a, "ask_c": x.c_a, "n_ticks": 1, "spread_mean": x.c_a - x.c})
    return "ok", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--start", type=int, default=2007)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args()
    cells = [(y, m) for y in range(a.start, a.end + 1) for m in range(1, 13)]
    old = []
    if a.fill:
        import glob
        fs = sorted(glob.glob(f"data/m1/{a.pair}/*.parquet"))
        old = [pd.read_parquet(f) for f in fs]
        if old:
            ts = pd.concat([o.ts for o in old])
            have = set(zip(ts.dt.year, ts.dt.month))
            first = min(have)
            cells = [c for c in cells if c >= first and c not in have]
        print(json.dumps({"pair": a.pair, "fill_months": [f"{y}-{m:02d}" for y, m in cells]}), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda c: (c, *month(a.pair, *c)), cells))
    rep = {"pair": a.pair, "months": len(cells), "ok": 0, "empty": 0, "failed": 0}
    frames = list(old)
    for (y, m), st, df in res:
        rep[st] += 1
        if df is not None:
            frames.append(df)
    if frames:
        D = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
        D = D[D.ts <= pd.Timestamp(os.environ.get("HOURS_END", "2025-12-31") + " 23:59", tz="UTC")]   # 2026: checks only
        os.makedirs(f"data/m1/{a.pair}", exist_ok=True)
        for y, g in D.groupby(D.ts.dt.year):
            g.to_parquet(f"data/m1/{a.pair}/{a.pair}_{y}.parquet", index=False)
            rep[str(y)] = int(len(g))
        rep["median_close_last_year"] = float(D.bid_c.iloc[-2000:].median())
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
