"""CPI release dates 2019-2025 from the BLS news-release archive (the archived file names carry the release date,
cpi_MMDDYYYY.htm). Writes a CSV with a 'date' column; empty if the archive cannot be read."""
import re
import sys
import urllib.request

import pandas as pd

out = sys.argv[1]
dates = set()
for url in ("https://www.bls.gov/bls/news-release/cpi.htm",):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9"})
        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        for m, d, y in re.findall(r"cpi_(\d{2})(\d{2})(\d{4})\.htm", html):
            dates.add(pd.Timestamp(int(y), int(m), int(d)))
        print(url, "ok", len(dates))
    except Exception as ex:  # noqa: BLE001
        print(url, "failed", repr(ex)[:200])
D = pd.DataFrame({"date": sorted(x for x in dates if pd.Timestamp("2019-01-01") <= x <= pd.Timestamp("2025-12-31"))})
D.to_csv(out, index=False)
print("CPI releases 2019-2025:", len(D), D.date.min() if len(D) else None, D.date.max() if len(D) else None)
