"""Macro-fundamental currency signals (run 18), per USD pair: exports the monthly FX series and the country's
macro data (fetched from FRED / OECD Main Economic Indicators on the runner) for scripts/macro_report.py, which
builds the cross-sectional strategies. Development data only (FX 2007-2018, EURUSD to 2022; macro cut at
2025-12-31).

Exported per pair (foreign currency vs USD, sign convention: + = foreign currency appreciates):
  monthly: month-end mid close, spot return of the foreign currency, close spread (median of the month's last 5
           daily closes, as a fraction of price), point-in-time 3-month rate differential foreign - USD
  macro:   industrial production, CPI, unemployment rate, retail sales volume, producer prices - for the foreign
           country (euro area for EUR) and the US, with the FRED series id that worked. Monthly OECD series are
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


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 fxlab-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fred_series(sid):
    t = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
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
    for sid in candidates(ccy, var):
        try:
            s = fred_series(sid)
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
