"""Direct Dukascopy tick downloader (for running on a machine with normal internet access).

The reachable mirror in the build environment stops in 2018 (2022 for EURUSD); this module
fetches the original hourly ``.bi5`` files for any period, decodes them and writes the SAME
M1 bid/ask parquet layout as acquire_fxdata_git (so everything downstream is unchanged).

bi5 format: LZMA-compressed sequence of 20-byte big-endian records
    uint32 ms_since_hour, uint32 ask_points, uint32 bid_points, float32 ask_volume, float32 bid_volume
price = points / 10**digits (digits = 3 for JPY quotes and metals, 5 otherwise).
URL: https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5

Usage (own machine):  python -m fxlab.data.acquire_dukascopy --pairs EURUSD XAUUSD --start 2019-01-01 --end 2025-12-31
Respect Dukascopy's terms of use; data is for research use and must not be redistributed.
"""
from __future__ import annotations

import argparse
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
REC = struct.Struct(">IIIff")


def digits(pair: str) -> int:
    """Price decimals of Dukascopy's integer points: 3 for JPY quotes, metals and index CFDs (names containing
    "IDX", e.g. USA500IDXUSD), 5 otherwise. Index scale verified by ci/probe_dukascopy2.py."""
    if "IDX" in pair:
        return 3
    return 3 if pair.endswith("JPY") or pair.startswith(("XAU", "XAG")) else 5


def decode_bi5(raw: bytes, hour_start: datetime, pair: str) -> pl.DataFrame:
    """Decode one hourly bi5 blob into ticks (UTC ms, bid, ask, bid_vol, ask_vol)."""
    if not raw:
        return pl.DataFrame(schema={"ts": pl.Datetime("ms", "UTC"), "bid": pl.Float64, "ask": pl.Float64,
                                    "bid_vol": pl.Float64, "ask_vol": pl.Float64})
    data = lzma.decompress(raw)
    n = len(data) // REC.size
    arr = np.frombuffer(data[: n * REC.size], dtype=np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                                                                  ("av", ">f4"), ("bv", ">f4")]))
    scale = 10.0 ** digits(pair)
    base = int(hour_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return pl.DataFrame({"ts": pl.Series((base + arr["ms"].astype(np.int64))).cast(pl.Datetime("ms", "UTC")),
                         "bid": arr["bid"].astype(np.float64) / scale, "ask": arr["ask"].astype(np.float64) / scale,
                         "bid_vol": arr["bv"].astype(np.float64), "ask_vol": arr["av"].astype(np.float64)})


class FetchFailed(Exception):
    pass


def fetch_hour(pair: str, hour: datetime, retries: int = 6) -> bytes:
    """One hourly file. 404 = no ticks in that hour (normal at weekends/holidays) -> b"". Other errors are
    retried with exponential back-off; if every retry fails FetchFailed is raised (counted by the caller)."""
    import urllib.error
    import urllib.request
    url = URL.format(pair=pair, y=hour.year, m=hour.month - 1, d=hour.day, h=hour.hour)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fxlab-research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return b""
            time.sleep(min(60.0, 2.0 ** k))
        except Exception:
            time.sleep(min(60.0, 2.0 ** k))
    raise FetchFailed(url)


def download_year(pair: str, year: int, start: datetime | None = None, end: datetime | None = None, workers: int = 8,
                  hours_utc: set | None = None, weekdays: set | None = None, dates: set | None = None,
                  suffix: str = "", deadline: float | None = None):
    """deadline: time.time() after which no new hour is requested (the rest count as failed), so a job that the
    feed throttles still writes what it fetched before its time limit."""
    from fxlab.data.acquire_fxdata_git import clean_ticks, ticks_to_m1, parquet_path, sha256_file, append_manifest
    s = start or datetime(year, 1, 1)
    e = end or datetime(year + 1, 1, 1)
    hours = []
    h = s
    while h < e:
        ok = h.weekday() != 5  # Saturdays carry no FX ticks
        if ok and hours_utc is not None and h.hour not in hours_utc:
            ok = False
        if ok and weekdays is not None and h.weekday() not in weekdays:
            ok = False
        if ok and dates is not None and h.date() not in dates:
            ok = False
        if ok:
            hours.append(h)
        h += timedelta(hours=1)
    failed = []

    def get(hh):
        if deadline is not None and time.time() > deadline:
            failed.append(hh)
            return hh, b""
        try:
            return hh, fetch_hour(pair, hh)
        except FetchFailed:
            failed.append(hh)
            return hh, b""
    with ThreadPoolExecutor(workers) as ex:
        blobs = list(ex.map(get, hours))
    frames = [decode_bi5(b, hh, pair) for hh, b in blobs if b]
    if not frames:
        return None                      # instrument not quoted in this year (e.g. before its Dukascopy start)
    ticks = pl.concat(frames)
    ticks, drops = clean_ticks(ticks)
    drops["hours_failed_download"] = len(failed)
    m1 = ticks_to_m1(ticks.filter(pl.col("ts").dt.year() == year))
    out = parquet_path(pair, year)
    if suffix:
        out = out.with_name(out.stem + suffix + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    m1.write_parquet(out, compression="zstd")
    status = "ok" if len(failed) <= 0.002 * len(hours) else "partial"
    append_manifest({"pair": pair, "year": year, "status": status, "source": "dukascopy-direct", "n_ticks_raw": ticks.height,
                     "drops": drops, "n_minutes": m1.height, "first_ts": str(m1["ts"].min()), "last_ts": str(m1["ts"].max()),
                     "sha256": sha256_file(out), "parquet": str(out)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--hours", default="", help="UTC hours to fetch, e.g. 13-20 or 0,1,14-23 (default: all)")
    ap.add_argument("--weekdays", default="", help="weekdays to fetch, 0=Mon..6=Sun, e.g. 4 or 0-4 (default: all)")
    ap.add_argument("--dates-file", default="", help="file with one YYYY-MM-DD per line: fetch only those dates")
    ap.add_argument("--suffix", default="", help="suffix of the output parquet name (partial downloads)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deadline-min", type=float, default=0, help="stop requesting new hours after this many minutes")
    a = ap.parse_args(argv)

    def rng(spec):
        if not spec:
            return None
        out = set()
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                out |= set(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        return out
    hours, wds = rng(a.hours), rng(a.weekdays)
    dates = None
    if a.dates_file:
        dates = {datetime.fromisoformat(x.strip()).date() for x in open(a.dates_file) if x.strip()}
    s, e = datetime.fromisoformat(a.start), datetime.fromisoformat(a.end)
    dl = time.time() + 60 * a.deadline_min if a.deadline_min else None
    if e >= datetime(2026, 1, 1):
        print("NOTE: data from 2026 onward is the locked holdout; it is stored but every loader refuses it "
              "until candidates are frozen and the holdout is explicitly unlocked.")
    for p in a.pairs:
        for y in range(s.year, e.year + 1):
            print(p, y, download_year(p, y, max(s, datetime(y, 1, 1)), min(e, datetime(y + 1, 1, 1)), workers=a.workers,
                                      hours_utc=hours, weekdays=wds, dates=dates, suffix=a.suffix, deadline=dl))


if __name__ == "__main__":
    main()
