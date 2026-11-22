"""Round 24 R1: OECD MEI short rates for every currency of the FX instruments, from DBnomics, to
data/rates/oecd_ir3m.csv (month, ccy, rate in per cent). 3-month interbank (IR3TIB01), else call money (IRSTCI01).
Usage: python ci/r24_rates.py"""
import json
import os
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
CODES = {"USD": ["USA"], "EUR": ["EA19", "EA20", "EA"], "JPY": ["JPN"], "GBP": ["GBR"], "CHF": ["CHE"],
         "AUD": ["AUS"], "NZD": ["NZL"], "CAD": ["CAN"], "SEK": ["SWE"], "NOK": ["NOR"], "PLN": ["POL"],
         "HUF": ["HUN"], "CZK": ["CZE"], "TRY": ["TUR"], "MXN": ["MEX"], "ZAR": ["ZAF"], "CNH": ["CHN"],
         "SGD": ["SGP"]}


def get(url, tries=5):
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code == 404:
                break
        except Exception as ex:                                            # noqa: BLE001
            last = repr(ex)[:120]
        time.sleep(10 * (k + 1))
    raise RuntimeError(last)


rows, src = [], {}
for ccy, codes in CODES.items():
    done = False
    for code in codes:
        for subj in ("IR3TIB01", "IRSTCI01"):
            try:
                doc = get(f"https://api.db.nomics.world/v22/series/OECD/MEI/{code}.{subj}.ST.M?observations=1")
                d = doc["series"]["docs"][0]
                s = pd.Series(pd.to_numeric(pd.Series(d["value"]), errors="coerce").to_numpy(), index=d["period"]).dropna()
                s = s[(s.index >= "2000-01") & (s.index <= "2025-12")]
                if len(s) < 24:
                    continue
                rows += [{"month": p, "ccy": ccy, "rate": float(v)} for p, v in s.items()]
                src[ccy] = {"code": code, "subject": subj, "first": s.index[0], "last": s.index[-1], "months": len(s)}
                done = True
                break
            except Exception as ex:                                        # noqa: BLE001
                src.setdefault(ccy + "_errors", []).append(f"{code}.{subj}: {ex}"[:120])
        if done:
            break
    if not done:
        src[ccy] = None
os.makedirs("data/rates", exist_ok=True)
pd.DataFrame(rows).to_csv("data/rates/oecd_ir3m.csv", index=False)
json.dump(src, open("data/rates/oecd_ir3m_sources.json", "w"), indent=1)
print(json.dumps({k: v for k, v in src.items() if not k.endswith("_errors")}, indent=0))
missing = [c for c in CODES if not src.get(c)]
print("currencies without a series:", missing)
sys.exit(0 if not set(missing) - {"SGD"} else 1)
