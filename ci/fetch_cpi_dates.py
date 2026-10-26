"""CPI release dates 2019-2025. Sources, in order: ALFRED's release-date list for release 10 (Consumer Price Index,
St. Louis Fed), then the BLS news-release archive (archived file names carry the release date, cpi_MMDDYYYY.htm).
Writes a CSV with a 'date' column (empty if no source can be read) and a log next to it."""
import re
import sys
import urllib.parse
import urllib.request

import pandas as pd

out = sys.argv[1]
log = open(out.replace(".csv", "_log.txt"), "w")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/html,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")


def say(*x):
    print(*x)
    print(*x, file=log)


dates = set()
base = "https://alfred.stlouisfed.org/release/downloaddates?rid=10"
try:
    html = get(base)
    links = re.findall(r'href="([^"]+)"', html)
    cand = [l for l in links if "txt" in l.lower() and ("release_dates" in l or "downloaddates" in l or "ff=" in l)]
    say("alfred page ok; candidate links:", cand[:6])
    for l in cand[:4] + [base + "&ff=txt"]:
        u = urllib.parse.urljoin(base, l.replace("&amp;", "&"))
        try:
            txt = get(u)
        except Exception as ex:  # noqa: BLE001
            say(u, "failed", repr(ex)[:150])
            continue
        got = {pd.Timestamp(x) for x in re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", txt)}
        say(u, "dates found", len(got))
        if len(got) > 100:
            dates |= got
            break
except Exception as ex:  # noqa: BLE001
    say("alfred failed", repr(ex)[:200])
if len([d for d in dates if d.year >= 2019]) < 60:
    try:
        html = get("https://www.bls.gov/bls/news-release/cpi.htm")
        for m, d, y in re.findall(r"cpi_(\d{2})(\d{2})(\d{4})\.htm", html):
            dates.add(pd.Timestamp(int(y), int(m), int(d)))
        say("bls archive ok", len(dates))
    except Exception as ex:  # noqa: BLE001
        say("bls failed", repr(ex)[:200])
D = pd.DataFrame({"date": sorted(x for x in dates if pd.Timestamp("2019-01-01") <= x <= pd.Timestamp("2025-12-31"))})
D.to_csv(out, index=False)
say("CPI releases 2019-2025:", len(D), D.date.min() if len(D) else None, D.date.max() if len(D) else None)
