"""Download (or probe) the free data for the edges batch (docs/edges_batch_preregistration.md). Every source is tried
on its own; failures are recorded, never fatal. Web pages are fetched only where robots.txt allows it. Raw files go
to OUT/raw (uploaded as an artifact, not committed); OUT/report.json says what worked.
Usage: python ci/edges_data.py OUT"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.robotparser as rp
import zipfile
from urllib.parse import urlparse

import requests

UA = {"User-Agent": "fxlab-research/1.0 (research; contact via github.com/anassboussarhan)"}
OUT = sys.argv[1] if len(sys.argv) > 1 else "out_edges"
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)
REPORT = {}
_robots = {}


def allowed(url):
    host = "{0.scheme}://{0.netloc}".format(urlparse(url))
    if host not in _robots:
        r = rp.RobotFileParser()
        try:
            txt = requests.get(host + "/robots.txt", headers=UA, timeout=30).text
            r.parse(txt.splitlines())
        except Exception:                                                    # noqa: BLE001
            r.parse([])
        _robots[host] = r
    return _robots[host].can_fetch(UA["User-Agent"], url)


def get(url, page=False, **kw):
    if page and not allowed(url):
        raise PermissionError(f"robots.txt disallows {url}")
    method = kw.pop("method", "GET")
    for k in range(4):
        try:
            r = requests.request(method, url, headers=UA, timeout=60, **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as ex:                                              # noqa: BLE001
            last = repr(ex)[:200]
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"{url}: {last}")


def save(name, data, mode="wb"):
    p = os.path.join(RAW, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as f:
        f.write(data)
    return p


def source(name):
    def deco(fn):
        def run():
            t = time.time()
            try:
                info = fn() or {}
                REPORT[name] = {"ok": True, **info}
            except Exception as ex:                                          # noqa: BLE001
                REPORT[name] = {"ok": False, "error": repr(ex)[:400]}
            REPORT[name]["seconds"] = round(time.time() - t, 1)
            print(name, json.dumps(REPORT[name]), flush=True)
        run.__name__ = fn.__name__
        return run
    return deco


@source("treasury_auctions")
def treasury():
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
    rows, page = [], 1
    while True:
        r = get(base, params={"filter": "auction_date:gte:2007-01-01", "page[size]": 5000, "page[number]": page}).json()
        rows += r["data"]
        if page >= r["meta"]["total-pages"]:
            break
        page += 1
    save("treasury_auctions.json", json.dumps(rows).encode())
    return {"rows": len(rows)}


@source("cny_fixing")
def cny():
    url = "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew"
    out = []
    for y in range(2007, 2026):
        for half in (("01-01", "06-30"), ("07-01", "12-31")):
            r = get(url, method="POST", data={"startDate": f"{y}-{half[0]}", "endDate": f"{y}-{half[1]}",
                                              "currency": "USD/CNY", "pageNum": 1, "pageSize": 400})
            j = r.json()
            out.append({"year": y, "half": half[0], "payload": j})
            time.sleep(1)
    save("cny_fixing.json", json.dumps(out).encode())
    n = sum(len((o["payload"].get("records") or o["payload"].get("data", {}).get("records") or [])) for o in out)
    return {"chunks": len(out), "records_guess": n}


@source("mof_weekly_flows")
def mof():
    got = {}
    for u in ("https://www.mof.go.jp/english/policy/international_policy/reference/itn_transactions_in_securities/week.csv",
              "https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/week.csv"):
        try:
            r = get(u)
            got[u] = len(r.content)
            save("mof_" + ("en" if "english" in u else "ja") + "_week.csv", r.content)
        except Exception as ex:                                              # noqa: BLE001
            got[u] = repr(ex)[:150]
    if not any(isinstance(v, int) for v in got.values()):
        raise RuntimeError(json.dumps(got))
    return {"files": got}


@source("fred_japan_trade")
def fred():
    ids = ("XTNTVA01JPM664S", "XTEXVA01JPM664S", "XTIMVA01JPM664S")
    for i in ids:
        save(f"fred_{i}.csv", get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": i}).content)
    return {"series": list(ids)}


@source("cftc_tff")
def tff():
    # history file 2006-2016, then one file a year named fut_fin_txt_YYYY.zip (futures only)
    files = ["fin_fut_txt_2006_2016.zip"] + [f"fut_fin_txt_{y}.zip" for y in range(2017, 2026)]
    n = 0
    for f in files:
        r = get("https://www.cftc.gov/files/dea/history/" + f)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for m in z.namelist():
            save(f"cftc_tff/{f[:-4]}_{os.path.basename(m)}", z.read(m))
            n += 1
    return {"files": n}


@source("central_bank_pages")
def cb():
    got, bad = 0, []
    pages = [("fed/fomccalendars.htm", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")]
    pages += [(f"fed/fomchistorical{y}.htm", f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
              for y in range(2007, 2021)]
    pages += [(f"ecb/pr_{y}.htm", f"https://www.ecb.europa.eu/press/pr/date/{y}/html/index.en.html") for y in range(2007, 2026)]
    pages += [("boj/past.htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm"),
              ("boj/index.htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"),
              ("boe/bank_rate.htm", "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp"),
              ("boe/mps.htm", "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/monetary-policy-summary-and-minutes")]
    for name, u in pages:
        try:
            save("cb/" + name, get(u, page=True).content)
            got += 1
        except Exception as ex:                                              # noqa: BLE001
            bad.append(f"{name}: {repr(ex)[:120]}")
        time.sleep(1)
    if not got:
        raise RuntimeError("; ".join(bad[:5]))
    return {"pages": got, "failed": bad}


@source("calendar_actual_forecast")
def calendar():
    """Public GitHub dataset of past releases with actual and forecast values (scraped by its author; kept for
    research in the artifact only)."""
    d = os.path.join(RAW, "calendar_repo")
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/spliffli/news-trading-analyzer.git", d],
                   check=True, timeout=600)
    files = [f for f in os.listdir(os.path.join(d, "news_data")) if f.endswith(".csv")]
    subprocess.run(["rm", "-rf", os.path.join(d, ".git")], check=False)
    return {"csv_files": len(files), "examples": sorted(files)[:5]}


@source("dtcc_fx_options_probe")
def dtcc():
    ok = {}
    for day in ("2019_06_03", "2021_06_01", "2023_06_01", "2025_06_02"):
        u = f"https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/CFTC_CUMULATIVE_FOREX_{day}.zip"
        try:
            r = get(u)
            save(f"dtcc/{day}.zip", r.content)
            ok[day] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            ok[day] = repr(ex)[:120]
    if not any(isinstance(v, int) for v in ok.values()):
        raise RuntimeError(json.dumps(ok))
    return {"files": ok}


@source("google_trends")
def trends():
    from pytrends.request import TrendReq
    t = TrendReq(hl="en-US", tz=0)
    res = {}
    for kw in ("yen", "gold price", "forex", "dollar"):
        t.build_payload([kw], timeframe="2007-01-01 2025-12-31")
        df = t.interest_over_time()
        save(f"trends_{kw.replace(' ', '_')}.csv", df.to_csv().encode())
        res[kw] = len(df)
        time.sleep(5)
    return {"rows": res}


@source("oanda_position_book")
def oanda():
    tok = os.environ.get("OANDA_TOKEN", "")
    if not tok:
        raise RuntimeError("no OANDA_TOKEN secret (a free OANDA practice account token is needed)")
    h = {"Authorization": f"Bearer {tok}", **UA}
    r = requests.get("https://api-fxpractice.oanda.com/v3/instruments/USD_JPY/positionBook", headers=h, timeout=30)
    save("oanda_probe.json", r.content)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return {"probe": "ok"}


if __name__ == "__main__":
    only = set(sys.argv[2:])
    for fn in (treasury, cny, mof, fred, tff, cb, calendar, dtcc, trends, oanda):
        if not only or fn.__name__ in only:
            fn()
    json.dump(REPORT, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(json.dumps({k: v["ok"] for k, v in REPORT.items()}))
