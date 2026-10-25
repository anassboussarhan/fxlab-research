"""Daily closes for the options strategy lab, from Yahoo (yfinance) on the runner, cut at 2025-12-31.

For each underlying: <SYM>_raw = the close as traded on that day (Yahoo's split-adjusted close multiplied back by
every later split, so it matches the strikes quoted on that day) and <SYM>_adj = the dividend- and split-adjusted
close (total return). Also VIX and VIX3M, and the Cboe volatility indices used by the synthetic layer.
Usage: python scripts/options_closes_fetch.py --symbols SPY,QQQ,... --out FILE"""
from __future__ import annotations

import argparse
import json

import pandas as pd

END = "2025-12-31"
VOL = {"VIX": "^VIX", "VIX3M": "^VIX3M", "VIX9D": "^VIX9D", "VXN": "^VXN", "RVX": "^RVX", "VXD": "^VXD",
       "GVZ": "^GVZ", "OVX": "^OVX", "VXEEM": "^VXEEM", "SKEW": "^SKEW", "VVIX": "^VVIX", "GSPC": "^GSPC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import yfinance as yf
    cols, meta = {}, {}
    for sym in [s for s in a.symbols.split(",") if s]:
        try:
            t = yf.Ticker(sym.replace(".", "-"))
            h = t.history(start=a.start, end="2026-01-01", auto_adjust=False, actions=True)
            h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
            h = h[h.index <= END]
            spl = h["Stock Splits"].replace(0, 1.0).fillna(1.0)
            later = spl[::-1].cumprod()[::-1].shift(-1).fillna(1.0)     # product of splits strictly after each day
            cols[f"{sym}_raw"] = h["Close"] * later
            cols[f"{sym}_adj"] = h["Adj Close"]
            meta[sym] = {"first": str(h.index[0].date()), "last": str(h.index[-1].date()), "n": int(len(h)),
                         "splits": {str(d.date()): float(v) for d, v in h["Stock Splits"][h["Stock Splits"] > 0].items()}}
        except Exception as ex:                                          # noqa: BLE001
            meta[sym] = {"error": str(ex)[:200]}
    for k, s in VOL.items():
        try:
            h = yf.download(s, start=a.start, end="2026-01-01", progress=False, auto_adjust=False)
            c = h["Close"]
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.to_datetime(c.index)
            c = c[c.index <= END].dropna()
            if len(c):
                cols[k] = c
                meta[k] = {"first": str(c.index[0].date()), "last": str(c.index[-1].date()), "n": int(len(c))}
            else:
                meta[k] = {"n": 0}
        except Exception as ex:                                          # noqa: BLE001
            meta[k] = {"error": str(ex)[:200]}
    X = pd.DataFrame(cols).sort_index()
    X.index.name = "date"
    X.to_csv(a.out)
    with open(a.out.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
