"""Round 23 part T2: event dates (UTC calendar days) for the BoJ, ECB and US CPI filters. Usage: python ci/r23_events.py OUT
  BoJ     statement links on the BoJ's monetary-policy meeting pages (as edges E8): k(yy)(mm)(dd)a.pdf / .htm
  ECB     the public calendar dataset (Eurozone interest-rate decisions, 2017-2023), plus the ECB's press-release
          index pages, titles "Monetary policy decisions"
  CPI US  the public calendar dataset (US CPI MoM, 2017-2023), plus the BLS release archive (cpi_MMDDYYYY.htm links)
Only dates up to 2025-12-31 are written."""
import glob
import os
import re
import subprocess
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
END = pd.Timestamp("2025-12-31")
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rep = {}


def get(u):
    for k in range(4):
        try:
            r = requests.get(u, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.text
        except Exception:                                                  # noqa: BLE001
            pass
        time.sleep(3 * (k + 1))
    return ""


boj = set()
for u in ("https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"):
    t = get(u)
    for m in re.finditer(r"k(\d{2})(\d{2})(\d{2})[a-z]?\.(?:pdf|htm)", t):
        try:
            boj.add(pd.Timestamp(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
ecb, cpi = set(), set()
d = os.path.join(out, "_cal")
try:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/spliffli/news-trading-analyzer.git", d],
                   check=True, timeout=600)
    for pat, dst in (("*Eurozone_Interest_Rate_Decision*.csv", ecb), ("*US_Consumer_Price_Index_(CPI)_MoM*.csv", cpi)):
        for f in glob.glob(os.path.join(d, "news_data", pat)):
            x = pd.read_csv(f)
            dst.update(pd.to_datetime(x.Timestamp).dt.normalize())
except Exception as ex:                                                    # noqa: BLE001
    rep["calendar_error"] = repr(ex)[:200]
for y in range(2007, 2026):
    t = get(f"https://www.ecb.europa.eu/press/pr/date/{y}/html/index.en.html")
    for m in re.finditer(r"(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December) "
                         rf"{y}[^<]{{0,400}}?Monetary policy decisions", t):
        try:
            ecb.add(pd.Timestamp(f"{m.group(1)} {m.group(2)} {y}"))
        except ValueError:
            pass
t = get("https://www.bls.gov/bls/news-release/cpi.htm")
for m in re.finditer(r"cpi_(\d{2})(\d{2})(\d{4})\.(?:htm|pdf)", t):
    try:
        cpi.add(pd.Timestamp(int(m.group(3)), int(m.group(1)), int(m.group(2))))
    except ValueError:
        pass
for name, s in (("boj", boj), ("ecb", ecb), ("cpi_us", cpi)):
    s = sorted(x for x in s if x <= END)
    open(os.path.join(out, f"{name}_dates.txt"), "w").write("\n".join(str(x.date()) for x in s))
    rep[name] = {"dates": len(s), "first": str(s[0].date()) if s else None, "last": str(s[-1].date()) if s else None}
print(rep)
