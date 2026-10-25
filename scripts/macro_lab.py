"""Macro-fundamental currency signals (run 18), per USD pair: exports the monthly FX series and the country's
macro data (fetched from FRED / OECD Main Economic Indicators on the runner) for scripts/macro_report.py, which
builds the cross-sectional strategies. Development data only (FX 2007-2018, EURUSD to 2022; macro cut at
2025-12-31).

Exported per pair (foreign currency vs USD, sign convention: + = foreign currency appreciates):
  monthly: month-end mid close, spot return of the foreign currency, close spread (median of the month's last 5
           daily closes, as a fraction of price), point-in-time 3-month rate differential foreign - USD
  macro:   industrial production, CPI, unemployment rate, retail sales volume, producer prices - for the foreign
           country (euro area for EUR) and the US, with the source id that worked (run 20: DBnomics mirrors of the
           OECD MEI and IMF IFS; run 18 tried FRED, which timed out on every request from the runner). Monthly OECD series are
           treated as available 2 months after their observation month, quarterly ones 5 months after the start
           of the quarter (revised data: real-time vintages are not available here).
Usage: python scripts/macro_lab.py --pair EURUSD [--out DIR]"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEV_END = "2025-12-31"
PAIRS = {"EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "JPY", "AUDUSD": "AUD", "USDCAD": "CAD", "USDCHF": "CHF",
         "USDSEK": "SEK", "USDNOK": "NOK", "NZDUSD": "NZD"}
ISO = {"USD": ("USA", "US"), "EUR": ("EA19", "EZ"), "GBP": ("GBR", "GB"), "JPY": ("JPN", "JP"), "CHF": ("CHE", "CH"),
       "CAD": ("CAN", "CA"), "AUD": ("AUS", "AU"), "NZD": ("NZL", "NZ"), "SEK": ("SWE", "SE"), "NOK": ("NOR", "NO")}


def candidates(ccy, var):
    i3, i2 = ISO[ccy]
    c = {"ip": [f"{i3}PROINDMISMEI", f"{i3}PROINDQISMEI"],
         "cpi": [f"CPALTT01{i2}M661S", f"CPALTT01{i2}M661N", f"CPALTT01{i2}Q661S", f"CPALTT01{i2}Q661N"],
         "unemp": [f"LRHUTTTT{i2}M156S", f"LRUNTTTT{i2}M156S", f"LRHUTTTT{i2}Q156S", f"LRUNTTTT{i2}Q156S"],
         "retail": [f"SLRTTO01{i2}M661S", f"SLRTTO02{i2}M661S", f"SLRTTO01{i2}Q661S", f"SLRTTO02{i2}Q661S"],
         "ppi": [f"PIEAMP01{i2}M661N", f"PITGIG01{i2}M661N", f"PIEATI01{i2}M661N", f"PIEAMP01{i2}Q661N"]}[var]
    if ccy == "EUR":
        c = {"ip": ["EA19PRINTO01IXOBSAM", "PRINTO01EZM661S", "EA19PROINDMISMEI", "DEUPROINDMISMEI"],
             "cpi": ["CP0000EZ19M086NEST", "CPHPTT01EZM661N", "EA19CPALTT01IXOBSAM"],
             "unemp": ["LRHUTTTTEZM156S", "LRHUTTTTEA19M156S"],
             "retail": ["SLRTTO01EZM661S", "EA19SLRTTO01IXOBSAM", "DEUSLRTTO01IXOBSAM"],
             "ppi": ["PIEAMP01EZM661N", "PITGIG01EZM661N"]}[var] + c
    if ccy == "USD" and var == "unemp":
        c = ["UNRATE"] + c
    return c


# DBnomics (api.db.nomics.world) mirrors of the OECD Main Economic Indicators and the IMF International Financial
# Statistics. Run 18 got nothing from FRED (every request timed out on the GitHub runner); a probe on 2026-10-25
# found DBnomics reachable there. Order: OECD MEI (monthly, then quarterly), IMF IFS, FRED last with a short timeout.
MEI = {"ip": [("PRINTO01", ["IXOBSA", "IXNBSA", "IXOB"])],
       "cpi": [("CPALTT01", ["IXOB", "IXOBSA", "IXNB"])],
       "unemp": [("LRHUTTTT", ["STSA", "ST"]), ("LRUNTTTT", ["STSA", "ST"])],
       "retail": [("SLRTTO01", ["IXOBSA", "IXOB"]), ("SLRTTO02", ["IXOBSA", "IXOB"])],
       "ppi": [("PIEAMP01", ["IXOB", "IXOBSA"]), ("PITGIG01", ["IXOB", "IXOBSA"])]}
MEI_LOC = {"EUR": ["EA19", "EA20", "EA"]}
IFS = {"ip": "AIP_IX", "cpi": "PCPI_IX", "unemp": "LUR_PT", "ppi": "PPPI_IX"}
IFS_AREA = {"USD": "US", "EUR": "U2", "GBP": "GB", "JPY": "JP", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ",
            "SEK": "SE", "NOK": "NO"}
DBN = "https://api.db.nomics.world/v22/series"


def _dbn_docs(url):
    J = json.loads(_get(url, timeout=60))
    return J.get("series", {}).get("docs", [])


def _dbn_to_series(doc):
    per, val = doc.get("period", []), doc.get("value", [])
    idx, v = [], []
    for p_, x in zip(per, val):
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        idx.append(pd.Period(p_.replace("-Q", "Q"), freq="Q" if "Q" in p_ else "M").to_timestamp(how="start"))
        v.append(x)
    s = pd.Series(v, index=pd.DatetimeIndex(idx)).sort_index()
    s = s[s.index <= DEV_END]
    if len(s) < 40:
        raise ValueError(f"only {len(s)} observations")
    return s


def dbnomics_series(ccy, var):
    """(series, source id) from DBnomics: OECD MEI by dimension query (monthly before quarterly, measures in the
    preferred order), then IMF IFS."""
    import urllib.parse
    errs = []
    locs = MEI_LOC.get(ccy, [ISO[ccy][0]])
    for subj, measures in MEI.get(var, []):
        dims = json.dumps({"LOCATION": locs, "SUBJECT": [subj]})
        try:
            docs = _dbn_docs(f"{DBN}/OECD/MEI?dimensions={urllib.parse.quote(dims)}&observations=1&limit=100")
        except Exception as ex:                                           # noqa: BLE001
            errs.append(f"MEI {subj}: {str(ex)[:60]}")
            continue
        byc = {d.get("series_code"): d for d in docs}
        for loc in locs:
            for freq in ("M", "Q"):
                for meas in measures:
                    d = byc.get(f"{loc}.{subj}.{meas}.{freq}")
                    if d is None:
                        continue
                    try:
                        return _dbn_to_series(d), f"dbnomics:OECD/MEI/{loc}.{subj}.{meas}.{freq}"
                    except Exception as ex:                               # noqa: BLE001
                        errs.append(f"{loc}.{subj}.{meas}.{freq}: {str(ex)[:40]}")
        errs.append(f"MEI {subj}: no usable series among {sorted(byc)[:8]}")
    if var in IFS:
        for freq in ("M", "Q"):
            code = f"{freq}.{IFS_AREA[ccy]}.{IFS[var]}"
            try:
                docs = _dbn_docs(f"{DBN}/IMF/IFS/{code}?observations=1")
                if docs:
                    return _dbn_to_series(docs[0]), f"dbnomics:IMF/IFS/{code}"
                errs.append(f"IFS {code}: no docs")
            except Exception as ex:                                       # noqa: BLE001
                errs.append(f"IFS {code}: {str(ex)[:60]}")
    raise RuntimeError("; ".join(errs))


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 fxlab-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fred_series(sid, timeout=60):
    t = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=timeout)
    d = pd.read_csv(io.StringIO(t))
    if d.shape[1] < 2:
        raise ValueError("no data")
    d.columns = ["date", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    s = d.dropna().set_index("date")["value"]
    s.index = pd.to_datetime(s.index)
    s = s[s.index <= DEV_END]
    if len(s) < 40:
        raise ValueError(f"only {len(s)} observations")
    return s


def fetch(ccy, var, cache):
    path = os.path.join(cache, f"{ccy}_{var}.csv")
    if os.path.exists(path):
        d = pd.read_csv(path, index_col=0, parse_dates=True)
        return d["value"], str(d["sid"].iloc[0])
    errs = []
    try:
        s, sid = dbnomics_series(ccy, var)
        pd.DataFrame({"value": s, "sid": sid}).to_csv(path)
        return s, sid
    except Exception as ex:                                               # noqa: BLE001
        errs.append(f"dbnomics: {str(ex)[:300]}")
    for sid in candidates(ccy, var)[:1]:                                  # FRED last, one try, short timeout
        try:
            s = fred_series(sid, timeout=20)
            pd.DataFrame({"value": s, "sid": sid}).to_csv(path)
            return s, sid
        except Exception as ex:                                           # noqa: BLE001
            errs.append(f"{sid}: {str(ex)[:60]}")
    raise RuntimeError("; ".join(errs))


def monthly_fx(pair):
    import feature_scan as FS
    from fxlab.data.rates import rate_diff
    b = FS.cached_bars(pair, "D1", "2007-01-01", DEV_END)
    mid = 0.5 * (b["bid_c"] + b["ask_c"])
    spr = (b["ask_c"] - b["bid_c"]) / mid
    m = pd.DataFrame({"mid": mid, "spr": spr})
    m["month"] = m.index.tz_convert("UTC").tz_localize(None).to_period("M")
    g = m.groupby("month")
    out = pd.DataFrame({"mid": g["mid"].last(), "spr": g["spr"].apply(lambda x: float(x.tail(5).median()))})
    sign = 1.0 if pair.startswith(PAIRS[pair]) else -1.0                # + = foreign currency appreciates
    out["spot_ret"] = sign * np.log(out["mid"]).diff()
    ts = out.index.to_timestamp(how="end").tz_localize("UTC")
    rd = rate_diff(pair, pd.DatetimeIndex(ts)).to_numpy()               # base - quote
    out["rate_diff"] = sign * rd                                         # foreign - USD (NaN without rates)
    out.index = out.index.astype(str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default=DEV_END)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "macro"))
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.pair not in PAIRS:
        json.dump({"pair": a.pair, "skipped": True}, open(os.path.join(a.out, f"{a.pair}.json"), "w"))
        print(f"{a.pair}: skipped")
        return
    ccy = PAIRS[a.pair]
    import tempfile
    cache = os.path.join(tempfile.gettempdir(), "fxlab_macro_cache")      # outside the output folder (files only there)
    os.makedirs(cache, exist_ok=True)
    meta = {"pair": a.pair, "ccy": ccy, "series": {}, "errors": {}}
    fx = monthly_fx(a.pair)
    fx.to_csv(os.path.join(a.out, f"fx_{ccy}.csv"))
    for c in (ccy, "USD"):
        for var in ("ip", "cpi", "unemp", "retail", "ppi"):
            try:
                s, sid = fetch(c, var, cache)
                pd.DataFrame({"value": s, "sid": sid}).to_csv(os.path.join(a.out, f"macro_{c}_{var}.csv"))
                meta["series"][f"{c}_{var}"] = {"sid": sid, "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
                                                "n": int(len(s))}
            except Exception as ex:                                       # noqa: BLE001
                meta["errors"][f"{c}_{var}"] = str(ex)[:400]
    meta["seconds"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(a.out, f"{a.pair}.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
