"""Direct Dukascopy M1 candle downloader: two files a day (BID and ASK minute candles) instead of 24 hourly tick
files, so a year of an instrument is about 620 requests. Writes the same M1 bid/ask parquet layout as the tick
downloaders (ts, bid_o..bid_c, ask_o..ask_c; n_ticks is 1 for a minute with volume, other tick statistics null).
Minutes without volume on either side are dropped (as in the tick-built bars, minutes without ticks are absent).

Candle bi5 format: LZMA-compressed 24-byte big-endian records
    uint32 seconds_since_day_start, uint32 open, uint32 close, uint32 low, uint32 high, float32 volume
(order checked on a live file: low <= open, close <= high); price = points / 10**digits (acquire_dukascopy.digits).
URL: https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM-1:02d}/{DD:02d}/{BID|ASK}_candles_min_1.bi5
Usage: python -m fxlab.data.acquire_dukascopy_candles --pairs JPNIDXJPY --start 2012-01-01 --end 2025-12-31
Respect Dukascopy's terms of use; data is for research use and must not be redistributed."""
from __future__ import annotations

import argparse
import lzma
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

from fxlab.data.acquire_dukascopy import digits

URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
DT = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])


class FetchFailed(Exception):
    pass


def fetch_day(pair: str, day: date, side: str, retries: int = 7) -> bytes:
    import urllib.error
    import urllib.request
    url = URL.format(pair=pair, y=day.year, m=day.month - 1, d=day.day, side=side)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fxlab-research/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return b""
            time.sleep(min(90.0, 2.0 ** k))
        except Exception:                                                    # noqa: BLE001
            time.sleep(min(90.0, 2.0 ** k))
    raise FetchFailed(url)


def decode_day(raw: bytes, day: date, pair: str) -> pl.DataFrame | None:
    if not raw:
        return None
    data = lzma.decompress(raw)
    n = len(data) // DT.itemsize
    if n == 0:
        return None
    a = np.frombuffer(data[: n * DT.itemsize], dtype=DT)
    base = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    sc = 10.0 ** digits(pair)
    return pl.DataFrame({"ts": pl.Series(base + a["t"].astype(np.int64) * 1000).cast(pl.Datetime("ms", "UTC")),
                         "o": a["o"] / sc, "h": a["h"] / sc, "l": a["l"] / sc, "c": a["c"] / sc,
                         "v": a["v"].astype(np.float64)})


def day_frame(pair: str, day: date) -> pl.DataFrame | None:
    b, a = fetch_day(pair, day, "BID"), fetch_day(pair, day, "ASK")
    B, A = decode_day(b, day, pair), decode_day(a, day, pair)
    if B is None or A is None:
        return None
    df = B.join(A, on="ts", suffix="_a")
    df = df.filter((pl.col("v") > 0) & (pl.col("v_a") > 0) & (pl.col("c_a") >= pl.col("c")) & (pl.col("c") > 0))
    if df.height == 0:
        return None
    return df.select(pl.col("ts"), pl.col("o").alias("bid_o"), pl.col("h").alias("bid_h"), pl.col("l").alias("bid_l"),
                     pl.col("c").alias("bid_c"), pl.col("o_a").alias("ask_o"), pl.col("h_a").alias("ask_h"),
                     pl.col("l_a").alias("ask_l"), pl.col("c_a").alias("ask_c"),
                     pl.lit(1, dtype=pl.Int32).alias("n_ticks"), (pl.col("c_a") - pl.col("c")).alias("spread_mean"))


def download(pair: str, start: date, end: date, dates: set | None = None, workers: int = 8, suffix: str = "",
             deadline: float | None = None, out_dir: str = "data/m1") -> dict:
    import os
    days, d = [], start
    while d < end:
        if d.weekday() != 5 and (dates is None or d in dates):              # Saturdays carry no FX quotes
            days.append(d)
        d += timedelta(days=1)
    failed = []

    def get(day):
        if deadline is not None and time.time() > deadline:
            failed.append(day)
            return None
        try:
            return day_frame(pair, day)
        except FetchFailed:
            failed.append(day)
            return None
    with ThreadPoolExecutor(workers) as ex:
        frames = [f for f in ex.map(get, days) if f is not None]
    report = {"pair": pair, "days_requested": len(days), "days_failed": len(failed), "days_with_data": len(frames)}
    if not frames:
        return report
    df = pl.concat(frames).unique(subset=["ts"]).sort("ts")
    os.makedirs(os.path.join(out_dir, pair), exist_ok=True)
    for (y,), g in df.group_by(pl.col("ts").dt.year()):
        p = os.path.join(out_dir, pair, f"{pair}_{y}{suffix}.parquet")
        g.sort("ts").write_parquet(p, compression="zstd")
        report[str(y)] = g.height
    return report


def main(argv=None):
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dates-file", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deadline-min", type=float, default=0)
    a = ap.parse_args(argv)
    dates = {date.fromisoformat(x.strip()) for x in open(a.dates_file) if x.strip()} if a.dates_file else None
    dl = time.time() + 60 * a.deadline_min if a.deadline_min else None
    for p in a.pairs:
        r = download(p, date.fromisoformat(a.start[:10]), date.fromisoformat(a.end[:10]), dates, a.workers, a.suffix, dl)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
