"""Round 9 data collection (docs/round9_preregistration.md). Free public sources only; raw files are saved as
fetched so that parsers can be written after the formats are inspected. Usage: python ci/r9_data.py OUTDIR"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OUT = sys.argv[1]
LOG = {}


def get(url, **kw):
    for k in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=90, **kw)
            return r
        except Exception as ex:                                          # noqa: BLE001
            err = repr(ex)[:200]
            time.sleep(5 * (k + 1))
    class R:                                                             # noqa: D401
        status_code, content, text = -1, b"", err
    return R()


def save(rel, content):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(content if isinstance(content, bytes) else content.encode())
    return len(content)


def ff_calendar():
    api = get("https://huggingface.co/api/datasets/Tropstan/Forex_Factory_Calendar")
    LOG["ff_api"] = api.status_code
    if api.status_code != 200:
        return
    files = [s["rfilename"] for s in api.json().get("siblings", [])]
    LOG["ff_files"] = files
    for f in files:
        if f.lower().endswith((".csv", ".parquet", ".json", ".md")):
            r = get(f"https://huggingface.co/datasets/Tropstan/Forex_Factory_Calendar/resolve/main/{f}")
            LOG.setdefault("ff_get", {})[f] = [r.status_code, save(f"ff/{f}", r.content) if r.status_code == 200 else 0]


FRED = ["DEXJPUS", "DEXUSUK", "DEXSZUS", "DEXCAUS", "DEXUSAL", "DEXUSNZ", "DEXSDUS", "DEXNOUS", "DEXDNUS",
        "DEXGEUS", "DEXUSEU", "DEXCHUS"]


def fred():
    for s in FRED:
        r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": s})
        LOG.setdefault("fred", {})[s] = [r.status_code, save(f"fred/{s}.csv", r.content) if r.status_code == 200 else 0]


GDELT = {"US": "US", "DE": "GM", "FR": "FR", "JP": "JA", "UK": "UK", "CH": "SZ", "CA": "CA", "AU": "AS", "NZ": "NZ"}


def gdelt():
    for name, code in GDELT.items():
        q = f'(economy OR inflation OR "interest rates" OR "central bank") sourcecountry:{code}'
        r = get("https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": q, "mode": "timelinetone", "format": "csv", "startdatetime": "20170101000000",
                        "enddatetime": "20251231235959"})
        LOG.setdefault("gdelt", {})[name] = [r.status_code, save(f"gdelt/{name}.csv", r.content)]
        time.sleep(8)


def norges():
    urls = {"mql5_%d" % p: f"https://www.mql5.com/en/economic-calendar/norway/norges-bank-foreign-exchange-daily-purchase"
            + (f"?page={p}" if p > 1 else "") for p in range(1, 9)}
    urls["nb_topic"] = "https://www.norges-bank.no/en/topics/Markets/foreign-exchange-and-krone-liquidity/"
    urls["nb_gpfg"] = "https://www.norges-bank.no/en/topics/Markets/Foreign-exchange-transactions-for-the-Government-Pension-Fund-Global/"
    urls["nb_search"] = "https://www.norges-bank.no/en/news-events/news-publications/Press-releases/"
    for k, u in urls.items():
        r = get(u)
        LOG.setdefault("norges", {})[k] = [r.status_code, save(f"norges/{k}.html", r.content)]
        time.sleep(2)


def sge():
    try:
        import akshare as ak
        os.makedirs(os.path.join(OUT, "sge"), exist_ok=True)
        for sym in ("Au99.99", "Au(T+D)"):
            try:
                df = ak.spot_hist_sge(symbol=sym)
                LOG.setdefault("sge", {})[sym] = [len(df), str(df.head(2).to_dict())[:300], str(df.tail(2).to_dict())[:300]]
                name = sym.replace("(", "_").replace(")", "").replace("+", "")
                df.to_csv(os.path.join(OUT, "sge", f"{name}.csv"), index=False)
            except Exception as ex:                                      # noqa: BLE001
                LOG.setdefault("sge", {})[sym] = repr(ex)[:300]
        try:
            df = ak.spot_golden_benchmark_sge()
            os.makedirs(os.path.join(OUT, "sge"), exist_ok=True)
            df.to_csv(os.path.join(OUT, "sge", "benchmark.csv"), index=False)
            LOG.setdefault("sge", {})["benchmark"] = [len(df), str(df.head(2).to_dict())[:300]]
        except Exception as ex:                                          # noqa: BLE001
            LOG.setdefault("sge", {})["benchmark"] = repr(ex)[:300]
    except Exception as ex:                                              # noqa: BLE001
        LOG["sge"] = "akshare: " + repr(ex)[:300]


def boj():
    for y in range(2007, 2026):
        for m in range(1, 13):
            u = f"https://www.boj.or.jp/en/statistics/boj/fm/ope/m_release/{y}/ope{y % 100:02d}{m:02d}.htm"
            r = get(u)
            LOG.setdefault("boj", {})[f"{y}-{m:02d}"] = [r.status_code, save(f"boj/ope{y}{m:02d}.htm", r.content)
                                                          if r.status_code == 200 else 0]
            time.sleep(0.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (ff_calendar, fred, gdelt, norges, sge, boj):
        try:
            f()
        except Exception as ex:                                          # noqa: BLE001
            LOG[f.__name__ + "_error"] = repr(ex)[:300]
        print(f.__name__, json.dumps({k: v for k, v in LOG.items() if k.startswith(f.__name__[:3])})[:1500], flush=True)
    json.dump(LOG, open(os.path.join(OUT, "r9_data_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
