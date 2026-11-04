"""Scheduled FOMC decision dates from the Federal Reserve's calendar pages (historical pages for 2010-2020, the
current calendar for later years). The decision date is the last day of each scheduled meeting; unscheduled meetings
and conference calls are left out. Usage: python ci/fomc_dates.py OUT.csv"""
import re
import sys
import urllib.request

import pandas as pd

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
MON = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                   "September", "October", "November", "December"], 1)}


def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read().decode("utf-8", "ignore")


rows, log = [], {}
for y in range(2010, 2021):
    try:
        t = get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
    except Exception as ex:                                                # noqa: BLE001
        log[y] = repr(ex)[:80]
        continue
    n = 0
    for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                         r"\s+(\d{1,2})(?:\s*-\s*(?:(January|February|March|April|May|June|July|August|September|October|"
                         r"November|December)\s+)?(\d{1,2}))?\s*(?:\(unscheduled\))?\s*Meeting\s*-\s*(\d{4})", t):
        ctx = t[max(0, m.start() - 40):m.end() + 40].lower()
        if "unscheduled" in ctx or "conference call" in ctx or int(m.group(5)) != y:
            continue
        mon = MON[m.group(3) or m.group(1)]
        day = int(m.group(4) or m.group(2))
        rows.append(pd.Timestamp(y, mon, day))
        n += 1
    log[y] = n
cur = get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
for blk in re.split(r"(?=\b(20\d\d) FOMC Meetings)", cur):
    my = re.match(r"(20\d\d) FOMC Meetings", blk)
    if not my:
        continue
    y = int(my.group(1))
    if y <= 2020:
        continue
    n = 0
    for m in re.finditer(r'fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+)\s*(?:</strong>)?.*?'
                         r'fomc-meeting__date[^>]*>\s*([\d\-]+)(\*?)', blk, re.S):
        months = m.group(1).split("/")
        days = m.group(2).split("-")
        mon = MON.get(months[-1], None) or next((v for k, v in MON.items() if k.startswith(months[-1][:3])), None)
        if mon is None or not days[-1].isdigit():
            continue
        rows.append(pd.Timestamp(y, mon, int(days[-1])))
        n += 1
    log[y] = n
d = pd.DataFrame({"date": sorted(set(rows))})
d.to_csv(sys.argv[1], index=False)
print("FOMC meetings per year:", log, "total", len(d))
