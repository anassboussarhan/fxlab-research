"""Money-flow, volume and positioning data for every market of the project (FX majors and crosses, gold, silver,
platinum, copper), plus daily yields. Every source is tried on its own; failures are recorded, never fatal.
Raw files go to STORE/ (kept as a rolling GitHub artifact, never committed: some sources do not allow
redistribution); STORE/report_<mode>.json says what worked.

--history  one-off: everything with free history.
  futures volume and open interest proxies: CME currency and metal futures (front contract, daily OHLCV, Yahoo);
  ETF money flows: shares outstanding / holdings of gold (GLD), silver (SLV) and currency ETFs (FXE, FXY, FXB, FXA,
    FXC, FXF, UUP), daily close x volume as dollar volume (Yahoo);
  CFTC weekly positions for every currency and metal: Traders in Financial Futures (currencies) and the
    disaggregated report (metals), 2006-today;
  US Treasury TIC monthly cross-border flows (major foreign holders);
  daily yields: US 2y / 10y (FRED), Japan JGB curve (Ministry of Finance), euro-area AAA 2y / 10y (ECB);
  Japan's foreign-exchange intervention record (Ministry of Finance).
--daily  every day (snapshots that have no free history, accumulated from now on):
  DTCC public FX option trades of the previous day (strikes, notionals, expiries);
  CME daily volume and open interest by contract (if the site allows it);
  retail positioning: Myfxbook community outlook; OANDA order and position books (needs the OANDA_TOKEN secret);
  the latest GLD holdings row.
Usage: python ci/flow_collect.py --history|--daily STORE"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import time
import zipfile

import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
MODE = sys.argv[1]
STORE = sys.argv[2] if len(sys.argv) > 2 else "flow_store"
os.makedirs(STORE, exist_ok=True)
REPORT = {}
TODAY = dt.datetime.utcnow().date()

FUT = {"6E=F": "EUR", "6J=F": "JPY", "6B=F": "GBP", "6A=F": "AUD", "6C=F": "CAD", "6S=F": "CHF", "6N=F": "NZD",
       "6M=F": "MXN", "DX=F": "DXY", "GC=F": "GOLD", "SI=F": "SILVER", "PL=F": "PLATINUM", "PA=F": "PALLADIUM",
       "HG=F": "COPPER"}
ETF = ["GLD", "IAU", "SLV", "FXE", "FXY", "FXB", "FXA", "FXC", "FXF", "UUP", "UDN"]


def get(url, **kw):
    last = ""
    for k in range(3):
        try:
            r = requests.get(url, headers={**UA, **kw.pop("headers", {})}, timeout=90, **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as ex:                                              # noqa: BLE001
            last = repr(ex)[:200]
        time.sleep(3 + 5 * k)
    raise RuntimeError(f"{url[:120]}: {last}")


def save(rel, data):
    p = os.path.join(STORE, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb" if isinstance(data, bytes) else "w") as f:
        f.write(data)
    return p


def source(fn):
    def run():
        t = time.time()
        try:
            REPORT[fn.__name__] = {"ok": True, **(fn() or {})}
        except Exception as ex:                                              # noqa: BLE001
            REPORT[fn.__name__] = {"ok": False, "error": repr(ex)[:400]}
        REPORT[fn.__name__]["seconds"] = round(time.time() - t, 1)
        print(fn.__name__, json.dumps(REPORT[fn.__name__])[:600], flush=True)
    run.__name__ = fn.__name__
    return run


# ------------------------------------------------------------------------------------------- history
@source
def yahoo_futures():
    import yfinance as yf
    out = {}
    for t, name in FUT.items():
        d = yf.download(t, start="2000-01-01", end=str(TODAY), progress=False, auto_adjust=False)
        if d is None or not len(d):
            out[name] = 0
            continue
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        save(f"history/futures/{name}.csv", d.to_csv())
        out[name] = int(len(d))
        time.sleep(1)
    return {"rows": out}


@source
def yahoo_etfs():
    import yfinance as yf
    out = {}
    for t in ETF:
        d = yf.download(t, start="2004-01-01", end=str(TODAY), progress=False, auto_adjust=False)
        if d is None or not len(d):
            out[t] = 0
            continue
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        save(f"history/etf/{t}.csv", d.to_csv())
        out[t] = int(len(d))
        try:
            so = yf.Ticker(t).get_shares_full(start="2010-01-01")
            if so is not None and len(so):
                save(f"history/etf/{t}_shares.csv", so.to_csv())
                out[t + "_shares"] = int(len(so))
        except Exception as ex:                                              # noqa: BLE001
            out[t + "_shares"] = repr(ex)[:80]
        time.sleep(1)
    return {"rows": out}


@source
def gld_holdings():
    r = get("https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv")
    save("history/etf/GLD_holdings_archive.csv", r.content)
    return {"bytes": len(r.content)}


@source
def cftc():
    out = {}
    for name, pat in (("tff", "https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip"),
                      ("disagg", "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip")):
        n = 0
        for y in range(2006, TODAY.year + 1):
            try:
                r = get(pat.format(y=y))
                z = zipfile.ZipFile(io.BytesIO(r.content))
                for m in z.namelist():
                    save(f"history/cftc_{name}/{y}_{os.path.basename(m)}", z.read(m))
                n += 1
            except Exception as ex:                                          # noqa: BLE001
                out[f"{name}_{y}"] = repr(ex)[:80]
        out[name + "_years"] = n
    return out


@source
def tic():
    r = get("https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table5.txt")
    save("history/tic/slt_table5.txt", r.content)
    out = {"slt_table5_bytes": len(r.content)}
    try:
        r2 = get("https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/mfh.txt")
        save("history/tic/mfh.txt", r2.content)
        out["mfh_bytes"] = len(r2.content)
    except Exception as ex:                                                  # noqa: BLE001
        out["mfh"] = repr(ex)[:100]
    return out


@source
def yields():
    out = {}
    for sid in ("DGS2", "DGS10", "DFII10", "T10YIE", "VIXCLS", "DTWEXBGS"):
        try:
            r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid})
            save(f"history/yields/fred_{sid}.csv", r.content)
            out[sid] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            out[sid] = repr(ex)[:80]
    for nm, u in (("jgbcme_all", "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"),
                  ("jgbcme", "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv")):
        try:
            r = get(u)
            save(f"history/yields/mof_{nm}.csv", r.content)
            out[nm] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            out[nm] = repr(ex)[:80]
    for tenor in ("SR_2Y", "SR_10Y"):
        try:
            r = get(f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{tenor}",
                    params={"format": "csvdata"})
            save(f"history/yields/ecb_aaa_{tenor}.csv", r.content)
            out["ecb_" + tenor] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            out["ecb_" + tenor] = repr(ex)[:80]
    return out


@source
def intervention():
    out = {}
    for nm, u in (("en", "https://www.mof.go.jp/english/policy/international_policy/reference/feio/foreign_exchange_intervention_operations.csv"),
                  ("ja", "https://www.mof.go.jp/policy/international_policy/reference/feio/foreign_exchange_intervention_operations.csv")):
        try:
            r = get(u)
            save(f"history/intervention/mof_{nm}.csv", r.content)
            out[nm] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            out[nm] = repr(ex)[:100]
    if not any(isinstance(v, int) for v in out.values()):
        raise RuntimeError(json.dumps(out))
    return out


# ------------------------------------------------------------------------------------------- daily
@source
def dtcc_options():
    got = {}
    for back in range(1, 5):
        d = TODAY - dt.timedelta(days=back)
        tag = d.strftime("%Y_%m_%d")
        u = f"https://kgc0418-tdw-data-0.s3.amazonaws.com/cftc/eod/CFTC_CUMULATIVE_FOREX_{tag}.zip"
        p = os.path.join(STORE, "daily", "dtcc_fx", f"{tag}.zip")
        if os.path.exists(p):
            got[tag] = "have"
            continue
        try:
            r = get(u)
            save(f"daily/dtcc_fx/{tag}.zip", r.content)
            got[tag] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            got[tag] = repr(ex)[:60]
    if not any(isinstance(v, int) or v == "have" for v in got.values()):
        raise RuntimeError(json.dumps(got))
    return got


@source
def cme_volume():
    # CME's public volume pages; the site may refuse automated clients, in which case this is recorded as failed
    out = {}
    ids = {"EUR": 58, "JPY": 69, "GBP": 42, "AUD": 37, "CAD": 48, "CHF": 86, "GOLD": 437, "SILVER": 458}
    for name, pid in ids.items():
        u = f"https://www.cmegroup.com/CmeWS/mvc/Volume/Details/F/{pid}/{TODAY.strftime('%Y%m%d')}/P"
        try:
            r = get(u, headers={"Accept": "application/json"})
            save(f"daily/cme/{TODAY}_{name}.json", r.content)
            out[name] = len(r.content)
        except Exception as ex:                                              # noqa: BLE001
            out[name] = repr(ex)[:60]
    if not any(isinstance(v, int) for v in out.values()):
        raise RuntimeError(json.dumps(out)[:300])
    return out


@source
def myfxbook_outlook():
    r = get("https://www.myfxbook.com/community/outlook")
    save(f"daily/myfxbook/{dt.datetime.utcnow():%Y-%m-%dT%H%M}.html", r.content)
    return {"bytes": len(r.content)}


@source
def oanda_books():
    tok = os.environ.get("OANDA_TOKEN", "")
    if not tok:
        raise RuntimeError("no OANDA_TOKEN secret (a free OANDA practice account token is needed)")
    h = {"Authorization": f"Bearer {tok}"}
    out = {}
    for ins in ("EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_JPY", "GBP_JPY",
                "EUR_GBP", "XAU_USD", "XAG_USD"):
        for book in ("orderBook", "positionBook"):
            try:
                r = get(f"https://api-fxpractice.oanda.com/v3/instruments/{ins}/{book}", headers=h)
                save(f"daily/oanda/{dt.datetime.utcnow():%Y-%m-%dT%H%M}_{ins}_{book}.json", r.content)
                out[f"{ins}_{book}"] = len(r.content)
            except Exception as ex:                                          # noqa: BLE001
                out[f"{ins}_{book}"] = repr(ex)[:60]
    return out


@source
def gld_daily():
    r = get("https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv")
    save(f"daily/gld/{TODAY}.csv", r.content)
    return {"bytes": len(r.content)}


if __name__ == "__main__":
    fns = {"--history": (yahoo_futures, yahoo_etfs, gld_holdings, cftc, tic, yields, intervention),
           "--daily": (dtcc_options, cme_volume, myfxbook_outlook, oanda_books, gld_daily)}[MODE]
    for fn in fns:
        fn()
    json.dump(REPORT, open(os.path.join(STORE, f"report_{MODE.strip('-')}_{TODAY}.json"), "w"), indent=1)
    print(json.dumps({k: v["ok"] for k, v in REPORT.items()}))
