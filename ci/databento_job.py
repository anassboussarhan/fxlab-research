"""Databento requests for the testing list (O1 option volatility selling, K1 pre-FOMC drift with the fed funds
contracts, F1 real futures volume). Mode "cost" (default) prices every request with metadata.get_cost /
get_billable_size and downloads NOTHING; mode "download ITEM ..." downloads only the items named (after the user has
approved the itemised cost). The key comes from the DATABENTO_API_KEY secret only. Data goes to an artifact, never
to the repository (licence). Usage: python ci/databento_job.py cost|download [ITEM ...] OUTDIR"""
from __future__ import annotations

import json
import os
import sys

START, END = "2010-06-06", "2026-01-01"          # GLBX.MDP3 history starts in June 2010; development data ends 2025
REQ = {
    # O1: CME options on euro, yen and gold futures: daily bars, settlements / open interest, and the definitions
    # (strike, expiry, call / put) needed to compute implied volatility.
    "O1_bars": dict(schema="ohlcv-1d", symbols=["6E.OPT", "EUU.OPT", "6J.OPT", "JPU.OPT", "OG.OPT", "GC.OPT"]),
    "O1_stats": dict(schema="statistics", symbols=["6E.OPT", "EUU.OPT", "6J.OPT", "JPU.OPT", "OG.OPT", "GC.OPT"]),
    "O1_defs": dict(schema="definition", symbols=["6E.OPT", "EUU.OPT", "6J.OPT", "JPU.OPT", "OG.OPT", "GC.OPT"]),
    # the underlying futures, for the option deltas and the hedges
    "O1_under": dict(schema="ohlcv-1d", symbols=["6E.FUT", "6J.FUT", "GC.FUT"]),
    # K1: every fed funds futures contract (Karnaukh: the next-month contract 3 days before each meeting)
    "K1_zq": dict(schema="ohlcv-1d", symbols=["ZQ.FUT"]),
    "K1_zq_defs": dict(schema="definition", symbols=["ZQ.FUT"]),
    # the FX-only versions of O1 (gold options exceed the free credit)
    "O1fx_bars": dict(schema="ohlcv-1d", symbols=["6E.OPT", "EUU.OPT", "6J.OPT", "JPU.OPT"]),
    "O1fx_stats": dict(schema="statistics", symbols=["6E.OPT", "EUU.OPT", "6J.OPT", "JPU.OPT"]),
    "O1fx_defs": dict(schema="definition", symbols=["6E.OPT", "EUU.OPT", "6J.OPT", "JPU.OPT"]),
    "O1fx_under": dict(schema="ohlcv-1d", symbols=["6E.FUT", "6J.FUT"]),
    # F1: real daily volume of the currency and metal futures, every contract
    "F1_volume": dict(schema="ohlcv-1d", symbols=["6E.FUT", "6J.FUT", "6B.FUT", "6A.FUT", "6C.FUT", "6S.FUT", "6N.FUT",
                                                  "6M.FUT", "GC.FUT", "SI.FUT"]),
}


def main():
    import databento as db
    mode = sys.argv[1]
    out = sys.argv[-1]
    os.makedirs(out, exist_ok=True)
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        print("no DATABENTO_API_KEY secret")
        json.dump({"error": "no DATABENTO_API_KEY secret"}, open(os.path.join(out, "cost.json"), "w"))
        return
    print("key format:", "starts with db-" if key.startswith("db-") else "does NOT start with db-", "| length", len(key))
    if key and not key.startswith("db-"):
        key = "db-" + key                                                # the prefix is often left out when pasting
        print("prefix db- added")
    c = db.Historical(key)
    rep = {"dataset_range": None, "items": {}}
    try:
        rep["dataset_range"] = str(c.metadata.get_dataset_range(dataset="GLBX.MDP3"))
    except Exception as ex:                                              # noqa: BLE001
        rep["dataset_range"] = repr(ex)[:200]
    for item, r in REQ.items():
        per = {}
        for s in r["symbols"]:
            q = dict(dataset="GLBX.MDP3", symbols=[s], stype_in="parent", schema=r["schema"], start=START, end=END)
            try:
                per[s] = {"usd": float(c.metadata.get_cost(**q)),
                          "gb": round(c.metadata.get_billable_size(**q) / 1e9, 4)}
            except Exception as ex:                                      # noqa: BLE001
                per[s] = {"error": repr(ex)[:200]}
        rep["items"][item] = {"schema": r["schema"], "per_symbol": per,
                              "usd": round(sum(v.get("usd", 0.0) for v in per.values()), 2)}
        print(item, json.dumps(rep["items"][item]), flush=True)
    rep["total_usd_all_items"] = round(sum(v["usd"] for v in rep["items"].values()), 2)
    json.dump(rep, open(os.path.join(out, "cost.json"), "w"), indent=1)
    print("TOTAL (all items) USD", rep["total_usd_all_items"])
    if mode == "batch":
        batch(c, sys.argv[2:-1], rep, out)
        return
    if mode == "download":
        items = sys.argv[2:-1]
        total = sum(rep["items"][i]["usd"] for i in items)
        cap = float(os.environ.get("DATABENTO_CAP_USD", "80"))
        print("selected items", items, "priced at USD", round(total, 2), "cap", cap, flush=True)
        if total > cap:
            print("ABORT: the selected items cost more than the cap; nothing downloaded")
            return
        for item in items:
            r = REQ[item]
            for s in r["symbols"]:
                if "error" in rep["items"][item]["per_symbol"].get(s, {}):
                    continue
                path = os.path.join(out, "data", f"{item}_{s.replace('.', '_')}.dbn.zst")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                c.timeseries.get_range(dataset="GLBX.MDP3", symbols=[s], stype_in="parent", schema=r["schema"],
                                       start=START, end=END, path=path)
                print("downloaded", item, s, os.path.getsize(path), flush=True)


def batch(c, items, rep, out):
    """Batch jobs (billed once, re-downloadable for 30 days) for the large schemas, then compact tables: option
    definitions -> one row per instrument; statistics -> settlement prices and open interest only."""
    import time
    import databento as db
    import pandas as pd
    total = sum(rep["items"][i]["usd"] for i in items)
    cap = float(os.environ.get("DATABENTO_CAP_USD", "80"))
    print("batch items", items, "priced at USD", round(total, 2), "cap", cap, flush=True)
    if total > cap:
        print("ABORT: above the cap; nothing submitted")
        return
    comp = os.path.join(out, "compact")
    os.makedirs(comp, exist_ok=True)
    for item in items:
        r = REQ[item]
        for s in r["symbols"]:
            if "error" in rep["items"][item]["per_symbol"].get(s, {}):
                continue
            job = c.batch.submit_job(dataset="GLBX.MDP3", symbols=[s], schema=r["schema"], start=START, end=END,
                                     stype_in="parent", split_duration="month", compression="zstd")
            jid = job["id"]
            print("submitted", item, s, jid, flush=True)
            for _ in range(240):
                done = [j for j in c.batch.list_jobs(states="done") if j.get("id") == jid]
                if done:
                    break
                time.sleep(30)
            else:
                print("timeout waiting for", jid)
                continue
            d = os.path.join(out, "raw", item, s.replace(".", "_"))
            files = c.batch.download(job_id=jid, output_dir=d)
            dbn = sorted(str(f) for f in files if str(f).endswith(".dbn.zst"))
            print("downloaded", item, s, len(dbn), "files", flush=True)
            parts = []
            for f in dbn:
                store = db.DBNStore.from_file(f)
                for df in store.to_df(count=500_000, map_symbols=False):
                    df = df.reset_index()
                    if r["schema"] == "statistics":
                        st = df["stat_type"].astype(str).str.upper()
                        keep = st.isin(["3", "9", "STATTYPE.SETTLEMENT_PRICE", "STATTYPE.OPEN_INTEREST",
                                        "SETTLEMENT_PRICE", "OPEN_INTEREST"])
                        cols = [x for x in ("ts_event", "ts_ref", "instrument_id", "stat_type", "price", "quantity")
                                if x in df.columns]
                        parts.append(df.loc[keep, cols])
                    elif r["schema"].startswith("ohlcv"):
                        cols = [x for x in ("ts_event", "instrument_id", "open", "high", "low", "close", "volume",
                                            "symbol") if x in df.columns]
                        parts.append(df[cols])
                    else:
                        cols = [x for x in ("ts_recv", "instrument_id", "raw_symbol", "expiration", "strike_price",
                                            "instrument_class", "underlying", "asset", "group") if x in df.columns]
                        parts.append(df[cols].drop_duplicates("instrument_id", keep="last"))
            if parts:
                T = pd.concat(parts, ignore_index=True)
                if r["schema"] == "definition":
                    T = T.drop_duplicates("instrument_id", keep="last")
                T["stat_type"] = T["stat_type"].astype(str) if "stat_type" in T else None
                path = os.path.join(comp, f"{item}_{s.replace('.', '_')}.parquet")
                T.to_parquet(path, index=False)
                print("compact", path, len(T), flush=True)
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
