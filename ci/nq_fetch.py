"""NQ futures minute bars (Databento GLBX.MDP3 ohlcv-1m, front contract by volume) for the fair-value-gap audit.
Development years only (to 2025-12-31). Prices the request first and downloads nothing above the cap.
Key from the DATABENTO_API_KEY secret. Usage: python ci/nq_fetch.py OUTDIR CAP_USD"""
from __future__ import annotations

import json
import os
import sys


def main():
    import databento as db
    out, cap = sys.argv[1], float(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if key and not key.startswith("db-"):
        key = "db-" + key
    c = db.Historical(key)
    rng = c.metadata.get_dataset_range(dataset="GLBX.MDP3")
    start = str(rng["start"])[:10] if isinstance(rng, dict) else "2010-06-06"
    rep = {"cap_usd": cap, "priced": [], "downloaded": False}
    q = None
    for st in (start, "2015-01-01", "2019-01-01"):                       # the longest history within the cap
        qq = dict(dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=["NQ.v.0"], stype_in="continuous",
                  start=st, end="2026-01-01")
        cost = float(c.metadata.get_cost(**qq))
        rep["priced"].append({"start": st, "usd": cost, "bytes": int(c.metadata.get_billable_size(**qq))})
        print(json.dumps(rep["priced"][-1]), flush=True)
        if cost <= cap:
            q, rep["request"], rep["usd"] = qq, qq, cost
            break
    if q is None:
        json.dump(rep, open(os.path.join(out, "nq_cost.json"), "w"), indent=1)
        print("every request above the cap: nothing downloaded")
        return
    data = c.timeseries.get_range(**q, path=os.path.join(out, "nq.dbn.zst"))
    try:
        df = data.to_df(price_type="float")
    except TypeError:
        df = data.to_df()
    df = df[["open", "high", "low", "close", "volume", "instrument_id"]]
    df.index.name = "ts"
    df.to_parquet(os.path.join(out, "nq_1m.parquet"))
    rep.update(downloaded=True, rows=int(len(df)), first=str(df.index.min()), last=str(df.index.max()),
               contracts=int(df.instrument_id.nunique()))
    json.dump(rep, open(os.path.join(out, "nq_cost.json"), "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
