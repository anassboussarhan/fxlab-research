"""Measure Dukascopy download speed and error behaviour from a GitHub runner (results as annotations)."""
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

TICK = "https://datafeed.dukascopy.com/datafeed/{p}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
CAND = "https://datafeed.dukascopy.com/datafeed/{p}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"


def get(url):
    t = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fxlab-research/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            b = r.read()
            return 200, len(b), time.time() - t
    except urllib.error.HTTPError as ex:
        return ex.code, 0, time.time() - t
    except Exception as ex:
        return type(ex).__name__, 0, time.time() - t


def run(urls, threads):
    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(get, urls))
    dt = time.time() - t
    codes = Counter(r[0] for r in res)
    lat = sorted(r[2] for r in res)
    return dt, codes, sum(r[1] for r in res), lat[len(lat) // 2]


start = datetime(2024, 3, 4)
hours = [start + timedelta(hours=i) for i in range(24 * 5)]
for k, threads in enumerate((4, 8, 16, 32)):
    hs = [h + timedelta(days=7 * (k + 1)) for h in hours]          # different files each time (no caching)
    urls = [TICK.format(p="EURUSD", y=h.year, m=h.month - 1, d=h.day, h=h.hour) for h in hs]
    dt, codes, nb, med = run(urls, threads)
    print(f"::notice title=ticks threads={threads}::{len(urls)} files in {dt:.1f}s = {len(urls) / dt:.1f} files/s, "
          f"codes {dict(codes)}, {nb / 1e6:.1f} MB, median latency {med:.2f}s")
days = [datetime(2023, 5, 1) + timedelta(days=i) for i in range(30)]
urls = [CAND.format(p="EURUSD", y=d.year, m=d.month - 1, d=d.day, side=s) for d in days for s in ("BID", "ASK")]
dt, codes, nb, med = run(urls, 8)
print(f"::notice title=minute candles threads=8::{len(urls)} files in {dt:.1f}s = {len(urls) / dt:.1f} files/s, "
      f"codes {dict(codes)}, {nb / 1e6:.2f} MB, median latency {med:.2f}s")
