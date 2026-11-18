"""Round 23: probe Dukascopy minute candles for candidate instruments (one recent and one old day) and write the
download matrices. Prints each candidate's median close at the default price scale. Usage: python ci/r23_probe.py"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.getcwd())
from fxlab.data.acquire_dukascopy_candles import day_frame  # noqa: E402

C = json.load(open("ci/r23_candidates.json"))
found, rep = [], {}
for p in C:
    r = {}
    for d in (date(2024, 3, 5), date(2012, 3, 6), date(2008, 3, 4)):
        try:
            f = day_frame(p, d)
        except Exception as ex:                                              # noqa: BLE001
            f = None
            r[str(d)] = f"error {ex!r}"[:80]
            continue
        if f is not None and f.height:
            r[str(d)] = {"rows": f.height, "median_close": float(f["bid_c"].median()) if "bid_c" in f.columns else None}
        else:
            r[str(d)] = None
    rep[p] = r
    if isinstance(r.get("2024-03-05"), dict):
        found.append(p)
    print(p, json.dumps(r), flush=True)
years = list(range(2007, 2026))
cells = [{"pair": p, "year": y} for p in found for y in years]
half = (len(cells) + 1) // 2
os.makedirs("probe", exist_ok=True)
json.dump(rep, open("probe/probe.json", "w"), indent=1)
with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as g:
    g.write(f"a={json.dumps({'include': cells[:half][:256]})}\n")
    g.write(f"b={json.dumps({'include': cells[half:][:256]})}\n")
print("found", found, "cells", len(cells))
