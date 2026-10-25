"""Fetch every macro series of scripts/macro_lab.py (10 currencies x 5 variables) from DBnomics on the runner and
publish them with a coverage table to the `probe` branch, to check sources and parsing before the macro sweep."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import macro_lab as M  # noqa: E402

out = os.path.join("probe_results", "macro")
os.makedirs(out, exist_ok=True)
cache = tempfile.mkdtemp()
cov = []
for c in ["USD"] + sorted(set(M.PAIRS.values())):
    for v in ("ip", "cpi", "unemp", "retail", "ppi"):
        try:
            s, sid = M.fetch(c, v, cache)
            s.rename("value").to_csv(os.path.join(out, f"macro_{c}_{v}.csv"))
            cov.append({"ccy": c, "var": v, "sid": sid, "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
                        "n": int(len(s))})
            print(f"::notice title={c}_{v}::{sid} {s.index[0].date()}..{s.index[-1].date()} n={len(s)}", flush=True)
        except Exception as ex:                                             # noqa: BLE001
            cov.append({"ccy": c, "var": v, "error": str(ex)[:600]})
            print(f"{c}_{v}: ERROR {str(ex)[:300]}", flush=True)
json.dump(cov, open(os.path.join(out, "coverage.json"), "w"), indent=1)
