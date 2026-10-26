"""Probe 2: (1) whether Dukascopy's daily minute-candle files are served now; (2) the price scale of index CFDs and
gold in tick files (decoded with 1, 2, 3 and 5 digits); (3) first hour with data per instrument (history start)."""
import lzma
import struct
import time
import urllib.error
import urllib.request
from datetime import datetime

TICK = "https://datafeed.dukascopy.com/datafeed/{p}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
CAND = "https://datafeed.dukascopy.com/datafeed/{p}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fxlab-research/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return 200, r.read()
        except urllib.error.HTTPError as ex:
            if ex.code == 404:
                return 404, b""
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    return "fail", b""


INSTR = ["USA500IDXUSD", "USATECHIDXUSD", "DEUIDXEUR", "GBRIDXGBP", "FRAIDXEUR", "CHEIDXCHF", "XAUUSD", "EURUSD",
         "EURJPY", "AUDNZD", "EURGBP", "GBPCAD"]
for p in INSTR:
    code, b = get(TICK.format(p=p, y=2024, m=2, d=13, h=15))          # 13 March 2024 15:00 UTC (index cash hours)
    if code != 200 or not b:
        print(f"::notice title=ticks {p}::code {code}")
        continue
    raw = lzma.decompress(b)
    n = len(raw) // 20
    ms, ask, bid, av, bv = struct.unpack(">IIIff", raw[:20])
    scales = {d: (bid / 10 ** d, ask / 10 ** d) for d in (1, 2, 3, 5)}
    print(f"::notice title=ticks {p}::{n} ticks; first bid/ask points {bid}/{ask}; as digits {scales}")
for p in ["EURUSD", "USA500IDXUSD"]:
    for side in ("BID", "ASK"):
        code, b = get(CAND.format(p=p, y=2024, m=2, d=13, side=side))
        info = ""
        if code == 200 and b:
            raw = lzma.decompress(b)
            info = f"{len(raw)} bytes = {len(raw) / 24:.1f} x 24-byte records; first {struct.unpack('>IIIIIf', raw[:24])}"
        print(f"::notice title=candles {p} {side}::code {code} {info}")
for p in ["USA500IDXUSD", "USATECHIDXUSD", "DEUIDXEUR", "GBRIDXGBP", "FRAIDXEUR", "CHEIDXCHF"]:
    first = None
    for y in range(2008, 2020):
        code, b = get(TICK.format(p=p, y=y, m=2, d=12 if y != 2012 else 13, h=15))
        if code == 200 and b:
            first = y
            break
    print(f"::notice title=history {p}::first year with a mid-March 15:00 UTC tick file: {first}")
