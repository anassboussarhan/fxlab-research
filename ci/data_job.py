"""Cloud sweep, step 1 (one job per instrument): download the Dukascopy tick history, build M1 bid/ask bars,
audit them and write per-timeframe close series for the cross-instrument panel.

Writes data/m1/<PAIR>/*.parquet, data/manifests/acq_<PAIR>.jsonl, data/closes/<PAIR>_<TF>.parquet,
out/audit_<PAIR>.json. Years with no quotes are skipped. Nothing at or after 2026-01-01 is downloaded (holdout)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxlab.config import M1_DIR, MANIFEST_DIR, DATA_DIR  # noqa: E402

PANEL_TFS = ["M5", "M15", "M30", "H1", "H2", "H4", "H8", "D1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--first-year", type=int, default=2005)
    ap.add_argument("--last-year", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-download", action="store_true", help="M1 files already restored from cache")
    a = ap.parse_args()
    assert a.last_year <= 2025, "2026 is the locked holdout"
    os.makedirs("out", exist_ok=True)
    from fxlab.data.acquire_dukascopy import download_year
    log = {"pair": a.pair, "years": {}}
    if not a.skip_download:
        for y in range(a.first_year, a.last_year + 1):
            t = time.time()
            try:
                p = download_year(a.pair, y, datetime(y, 1, 1), datetime(y + 1, 1, 1), workers=a.workers)
                log["years"][y] = {"file": str(p) if p else None, "seconds": round(time.time() - t, 1)}
            except Exception as ex:          # recorded, never silent
                log["years"][y] = {"error": repr(ex)[:300], "seconds": round(time.time() - t, 1)}
            print(a.pair, y, log["years"][y], flush=True)
        man = MANIFEST_DIR / "acquisition.jsonl"
        if man.exists():
            shutil.copy(man, MANIFEST_DIR / f"acq_{a.pair}.jsonl")
    from fxlab.data.quality import audit_file
    audits = [audit_file(f) for f in sorted((M1_DIR / a.pair).glob("*.parquet"))]
    log["audit"] = audits
    from fxlab.data.store import load_bars
    cdir = DATA_DIR / "closes"
    cdir.mkdir(parents=True, exist_ok=True)
    if audits:
        for tf in PANEL_TFS:
            b = load_bars(a.pair, tf, "2005-01-01", "2025-12-31")
            b[["close"]].to_parquet(cdir / f"{a.pair}_{tf}.parquet")
    (open(f"out/audit_{a.pair}.json", "w")).write(json.dumps(log, indent=1, default=str))
    bad = [x for x in audits if not x.get("price_level_plausible", True)]
    print(f"{a.pair}: {len(audits)} years, implausible price level in {len(bad)}", flush=True)


if __name__ == "__main__":
    main()
