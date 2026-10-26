"""Matrix of Dukascopy downloads (2019-2025) for scripts/untested_intraday_lab.py, in priority order. Each entry
fetches only the UTC hours / weekdays / dates its tests read (the feed serves about one hourly file a second)."""
import json
import os

END = "2025-12-31T23:59:00"


def quarters(y0=2019, y1=2025):
    out = []
    for y in range(y0, y1 + 1):
        for q in range(4):
            s = f"{y}-{3 * q + 1:02d}-01T00:00:00"
            e = f"{y + (q == 3)}-{(3 * q + 3) % 12 + 1:02d}-01T00:00:00" if not (y == y1 and q == 3) else END
            out.append((f"{y}q{q + 1}", s, e))
    return out


def halves(y0=2019, y1=2025):
    out = []
    for y in range(y0, y1 + 1):
        out.append((f"{y}h1", f"{y}-01-01T00:00:00", f"{y}-07-01T00:00:00"))
        out.append((f"{y}h2", f"{y}-07-01T00:00:00", f"{y + 1}-01-01T00:00:00" if y < y1 else END))
    return out


def job(sym, tag, start, end, hours="", weekdays="", dates=""):
    return {"sym": sym, "tag": tag, "start": start, "end": end, "hours": hours, "weekdays": weekdays, "dates": dates}


inc = []
for sym in ("USA500IDXUSD", "USATECHIDXUSD"):                      # I12: 09:30-16:00 New York = 13-21 UTC
    inc += [job(sym, t, s, e, "13-21", "0-4") for t, s, e in quarters()]
inc.append(job("EURJPY", "gprev", "2019-01-01T00:00:00", END, "18-23", "", "gotobi_prev"))   # I4: 03:00 Tokyo
inc.append(job("EURJPY", "gday", "2019-01-01T00:00:00", END, "0-1", "", "gotobi"))           # I4: 09:55 Tokyo
for sym in ("EURJPY", "AUDUSD"):                                    # I10: Friday close, Sunday open, the next 24 h
    inc.append(job(sym, "wfri", "2019-01-01T00:00:00", END, "19-22", "4"))
    inc.append(job(sym, "wsun", "2019-01-01T00:00:00", END, "20-23", "6"))
    inc += [job(sym, f"wmon{y}", f"{y}-01-01T00:00:00", f"{y + 1}-01-01T00:00:00" if y < 2025 else END, "0-23", "0")
            for y in range(2019, 2026)]
for sym in ("FRAIDXEUR", "CHEIDXCHF"):                              # I11: Friday closing auctions
    inc.append(job(sym, "fri", "2019-01-01T00:00:00", END, "14-17", "4"))
for sym in ("DEUIDXEUR", "GBRIDXGBP"):                              # I12b DAX/FTSE, I11 FTSE: 07-16 UTC
    inc += [job(sym, t, s, e, "7-16", "0-4") for t, s, e in quarters()]
inc += [job("EURUSD", t, s, e, "", "0-4,6") for t, s, e in quarters(2023, 2025)]          # I6b/c, I8 (mirror to 2022)
inc += [job("XAUUSD", t, s, e, "0-16,23", "0-4,6") for t, s, e in quarters()]           # I9: London 00:00-16:00
for sym in ("AUDNZD", "EURGBP", "GBPCAD"):                          # I7: bands warm up from 19:20 UTC
    inc += [job(sym, t, s, e, "0-1,19-23", "0-4") for t, s, e in halves()]
assert len(inc) <= 256, len(inc)
print(f"{len(inc)} jobs", flush=True)
with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
    f.write("matrix=" + json.dumps({"include": inc}) + "\n")
