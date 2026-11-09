"""Round 15 part D probe: which free sources answer, their format and date range. Values are not inspected
(only headers, row counts, first / last dates, and for Dukascopy the integer scale of one candle)."""
import io
import json
import lzma
import os
import sys
import time
from datetime import date

import numpy as np
import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/probe15"
os.makedirs(OUT, exist_ok=True)
R = {}


def save():
    json.dump(R, open(f"{OUT}/probe15.json", "w"), indent=1, default=str)


def get(url, **kw):
    try:
        r = requests.get(url, headers=UA, timeout=30, **kw)
        return r
    except Exception as ex:                                               # noqa: BLE001
        return ex


def info(name, url, text=True):
    r = get(url)
    if isinstance(r, Exception):
        R[name] = {"url": url, "error": repr(r)[:200]}
        return None
    d = {"url": url, "status": r.status_code, "bytes": len(r.content), "type": r.headers.get("content-type", "")}
    if text and r.status_code == 200:
        lines = r.text.splitlines()
        d["lines"] = len(lines)
        d["head"] = [l[:200] for l in lines[:2]]
    R[name] = d
    return r


# D1 Forex Factory calendar archives on HuggingFace
for ds in ("Tropstan/Forex_Factory_Calendar", "obrotowy/Forex_Factory_Calendar", "Ehsanrs2/Forex_Factory_Calendar",
           "oprizzle/Forex_Factory_Calendar", "stephanyuran/Forex_Factory_Calendar"):
    r = get(f"https://huggingface.co/api/datasets/{ds}")
    if isinstance(r, Exception) or r.status_code != 200:
        R[f"hf:{ds}"] = {"status": getattr(r, "status_code", repr(r)[:100])}
        continue
    j = r.json()
    files = [s["rfilename"] for s in j.get("siblings", [])]
    rec = {"files": files[:20], "license": (j.get("cardData") or {}).get("license"), "lastModified": j.get("lastModified")}
    for f in files:
        if f.lower().endswith((".csv", ".parquet", ".json", ".jsonl")) and "readme" not in f.lower():
            rr = get(f"https://huggingface.co/datasets/{ds}/resolve/main/{f}")
            if not isinstance(rr, Exception) and rr.status_code == 200:
                rec["file"] = f
                rec["bytes"] = len(rr.content)
                try:
                    import pandas as pd
                    df = pd.read_parquet(io.BytesIO(rr.content)) if f.endswith(".parquet") else pd.read_csv(io.BytesIO(rr.content))
                    rec["columns"] = list(df.columns)
                    rec["rows"] = int(len(df))
                    dc = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                    if dc:
                        s = df[dc[0]].astype(str)
                        rec["first"], rec["last"] = s.min(), s.max()
                    cc = [c for c in df.columns if c.lower() in ("currency", "ccy", "country")]
                    if cc:
                        rec["currencies"] = sorted(map(str, df[cc[0]].dropna().unique()))[:20]
                    fc = [c for c in df.columns if "forecast" in c.lower() or "consensus" in c.lower()]
                    if fc:
                        rec["forecast_nonnull_share"] = float(df[fc[0]].notna().mean())
                    os.makedirs(f"{OUT}/hf", exist_ok=True)
                    df.to_parquet(f"{OUT}/hf/{ds.replace('/', '__')}.parquet")
                except Exception as ex:                                   # noqa: BLE001
                    rec["parse_error"] = repr(ex)[:200]
                break
    R[f"hf:{ds}"] = rec

save()
# D3 Tokyo Financial Exchange (Click 365) historical pages
for nm, url in (("tfx_hist", "https://www.tfx.co.jp/historical/fx/"), ("tfx_hist_en", "https://www.tfx.co.jp/en/historical/fx/"),
                ("tfx_root", "https://www.tfx.co.jp/historical/")):
    r = info(nm, url)
    if r is not None and getattr(r, "status_code", 0) == 200:
        import re
        links = sorted(set(re.findall(r'href="([^"]+\.(?:csv|zip|xls|xlsx))"', r.text)))[:40]
        R[nm]["data_links"] = links

# D4 CBOE strategy benchmark indices
for ix in ("BXM", "BXMD", "PUT", "CNDR", "BFLY", "CLL", "CMBO", "WPUT", "PPUT", "BXY", "CLLZ"):
    info(f"cboe:{ix}", f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{ix}_History.csv")

# D5 two-year government yields
y5 = {
    "us_treasury_2024": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2024/all?type=daily_treasury_yield_curve&field_tdr_date_value=2024&page&_format=csv",
    "ecb_yc_2y": "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y?format=csvdata",
    "boc_2y": "https://www.bankofcanada.ca/valet/observations/BD.CDN.2YR.DQ.YLD/csv",
    "rba_f2": "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv",
    "rba_f2_hist": "https://www.rba.gov.au/statistics/tables/csv/f2.1-data.csv",
    "snb_rendoblid": "https://data.snb.ch/api/cube/rendoblid/data/csv/en",
    "mof_jgb_all": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv",
    "mof_jgb_recent": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv",
    "boe_2y": "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?Travel=NIxAZxSUx&FromSeries=1&ToSeries=50&DAT=RNG&FD=1&FM=Jan&FY=2006&TD=31&TM=Dec&TY=2025&FNY=Y&CSVF=TT&html.x=66&html.y=26&SeriesCodes=IUDSNZC&UsingCodes=Y&Filter=N&title=IUDSNZC&VPD=Y",
    "rbnz_b2": "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily.xlsx",
    "bundesbank_2y": "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A?format=csv",
}
for k, u in y5.items():
    info(f"yield:{k}", u, text=not u.endswith(".xlsx"))

# D7 Dukascopy SWFX sentiment (public widget feed, if any)
for k, u in (("swfx_feed", "https://freeserv.dukascopy.com/2.0/?path=api/sentimentIndex&type=swfx&jsonp=_"),
             ("swfx_page", "https://www.dukascopy.com/swiss/english/marketwatch/sentiment/")):
    info(f"sentiment:{k}", u)

# D8 Google Trends (unofficial endpoint)
info("gtrends:explore", "https://trends.google.com/trends/api/explore?hl=en-US&tz=0&req=%7B%22comparisonItem%22:%5B%7B%22keyword%22:%22forex%22,%22geo%22:%22%22,%22time%22:%22all%22%7D%5D,%22category%22:0,%22property%22:%22%22%7D")

save()
# D2 Dukascopy instruments not in the store: years with data (sampled) and the integer scale of one candle
DT = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])
cands = ["BRENTCMDUSD", "LIGHTCMDUSD", "GASCMDUSD", "COPPERCMDUSD", "DIESELCMDUSD", "XPTCMDUSD", "XPDCMDUSD",
         "SOYBEANCMDUSX", "WHEATCMDUSX", "CORNCMDUSX", "SUGARCMDUSD", "COCOACMDUSD", "COFFEECMDUSX", "COTTONCMDUSX",
         "BUNDTREUR", "USTBONDTRUSD", "UKGILTTRGBP", "SCHATZTREUR", "BOBLTREUR",
         "DEUIDXEUR", "GBRIDXGBP", "FRAIDXEUR", "EUSIDXEUR", "HKGIDXHKD", "AUSIDXAUD", "USA30IDXUSD", "CHEIDXCHF",
         "ESPIDXEUR", "NLDIDXEUR", "ITAIDXEUR",
         "EURSEK", "EURNOK", "USDSEK", "USDNOK", "USDMXN", "USDZAR", "USDPLN", "EURPLN", "USDTRY", "EURTRY", "USDSGD",
         "USDHKD", "USDHUF", "EURHUF", "USDCZK", "EURCZK", "USDDKK", "XPTUSD", "XPDUSD", "BTCUSD", "ETHUSD"]
YEARS = (2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025)


def duka_day(pair, d):
    url = f"https://datafeed.dukascopy.com/datafeed/{pair}/{d.year:04d}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
    for k in range(2):
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code == 404:
                return b""
            if r.status_code == 200:
                return r.content
        except Exception:                                                 # noqa: BLE001
            pass
    return None


def probe_pair(p):
    rec = {"years": {}}
    for y in YEARS:
        d = date(y, 3, 11)
        while d.weekday() >= 5:
            d = date(y, 3, d.day + 1)
        raw = duka_day(p, d)
        if raw is None:
            rec["years"][y] = "no answer"
            continue
        if not raw:
            rec["years"][y] = 0
            continue
        try:
            data = lzma.decompress(raw)
            a = np.frombuffer(data[: (len(data) // 24) * 24], dtype=DT)
            rec["years"][y] = int(len(a))
            if len(a):
                rec["raw_close_int_magnitude"] = int(np.log10(max(int(a["c"][len(a) // 2]), 1)))
        except Exception as ex:                                           # noqa: BLE001
            rec["years"][y] = "decode error"
    have = [y for y, v in rec["years"].items() if isinstance(v, int) and v > 0]
    rec["first_year_sampled"] = min(have) if have else None
    return p, rec


from concurrent.futures import ThreadPoolExecutor                         # noqa: E402
with ThreadPoolExecutor(12) as ex:
    R["dukascopy"] = dict(ex.map(probe_pair, cands))
save()

print(json.dumps(R, indent=1, default=str)[:20000])
