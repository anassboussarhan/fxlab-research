"""Round 9 data, fourth pass: ECB euro reference rates (FRED timed out from GitHub), GDELT France retry."""
import os, sys, time, requests
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OUT = sys.argv[1]
os.makedirs(f"{OUT}/ecb", exist_ok=True); os.makedirs(f"{OUT}/gdelt", exist_ok=True)
u = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD+JPY+GBP+CHF+CAD+AUD+NZD+SEK+NOK+DKK+CNY.EUR.SP00.A"
r = requests.get(u, params={"format": "csvdata"}, headers=UA, timeout=300)
open(f"{OUT}/ecb/exr.csv", "wb").write(r.content); print("ecb", r.status_code, len(r.content))
for k in range(5):
    time.sleep(30)
    q = '(economy OR inflation OR "interest rates" OR "central bank") sourcecountry:FR'
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", headers=UA, timeout=180,
                     params={"query": q, "mode": "timelinetone", "format": "csv", "startdatetime": "20170101000000",
                             "enddatetime": "20251231235959"})
    print("gdelt FR", r.status_code, len(r.content))
    if r.status_code == 200 and len(r.content) > 1000:
        open(f"{OUT}/gdelt/FR.csv", "wb").write(r.content); break
