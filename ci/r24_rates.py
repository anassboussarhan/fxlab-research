"""Round 24 R1 (addendum 2): central-bank policy rates for every currency of the FX instruments, monthly, from the
BIS (WS_CBPOL), to data/rates/oecd_ir3m.csv (month, ccy, rate in per cent; the file name is kept for the engine).
OECD MEI, the first source, ends in January 2024 and its Turkish interbank series in 2008. Sources tried in order:
the BIS SDMX API, then the DBnomics mirror. CNH uses China's rate; SGD has no policy rate and is left out.
Usage: python ci/r24_rates.py"""
import io
import json
import os
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
BIS = {"USD": "US", "EUR": "XM", "JPY": "JP", "GBP": "GB", "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA",
       "SEK": "SE", "NOK": "NO", "PLN": "PL", "HUF": "HU", "CZK": "CZ", "TRY": "TR", "MXN": "MX", "ZAR": "ZA",
       "CNH": "CN"}


def try_get(url, tries=4, js=False):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 200 and len(r.content) > 100:
                return r.json() if js else r.text
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 404):
                break
        except Exception as ex:                                            # noqa: BLE001
            last = repr(ex)[:120]
        time.sleep(8 * (k + 1))
    raise RuntimeError(last)


def bis_api(cc, freq):
    t = try_get(f"https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/{freq}.{cc}?format=csv")
    d = pd.read_csv(io.StringIO(t))
    col = "OBS_VALUE" if "OBS_VALUE" in d else d.columns[-1]
    s = pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(), index=pd.to_datetime(d["TIME_PERIOD"])).dropna()
    return s.sort_index()


def dbnomics(cc, freq):
    doc = try_get(f"https://api.db.nomics.world/v22/series/BIS/WS_CBPOL/{freq}.{cc}?observations=1", js=True)
    d = doc["series"]["docs"][0]
    return pd.Series(pd.to_numeric(pd.Series(d["value"]), errors="coerce").to_numpy(),
                     index=pd.to_datetime(d["period"])).dropna().sort_index()


rows, src = [], {}
for ccy, cc in BIS.items():
    s, how = None, []
    for name, f in (("bis_api_M", lambda: bis_api(cc, "M")), ("dbnomics_M", lambda: dbnomics(cc, "M")),
                    ("bis_api_D", lambda: bis_api(cc, "D")), ("dbnomics_D", lambda: dbnomics(cc, "D"))):
        try:
            x = f()
            x = x.resample("ME").mean() if name.endswith("_D") else x
            x.index = x.index.to_period("M")
            x = x[(x.index >= pd.Period("2000-01", "M")) & (x.index <= pd.Period("2025-12", "M"))]
            if len(x) >= 24:
                s, how = x, name
                break
            how.append(f"{name}: {len(x)} months")
        except Exception as ex:                                            # noqa: BLE001
            how.append(f"{name}: {ex}"[:140])
    if s is None:
        src[ccy] = {"error": how}
        continue
    rows += [{"month": str(p), "ccy": ccy, "rate": float(v)} for p, v in s.items()]
    src[ccy] = {"source": how, "first": str(s.index[0]), "last": str(s.index[-1]), "months": int(len(s))}
os.makedirs("data/rates", exist_ok=True)
pd.DataFrame(rows).to_csv("data/rates/oecd_ir3m.csv", index=False)
json.dump(src, open("data/rates/oecd_ir3m_sources.json", "w"), indent=1)
for k, v in src.items():
    print(k, json.dumps(v)[:200])
bad = [k for k, v in src.items() if "error" in v or v["last"] < "2025-11"]
print("currencies without a complete series to 2025:", bad)
sys.exit(0 if not bad else 1)
