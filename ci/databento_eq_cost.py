"""Price US-equity requests on Databento (cost only; downloads nothing). Key from the DATABENTO_API_KEY secret."""
from __future__ import annotations

import json
import os
import sys


def main():
    import databento as db
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if key and not key.startswith("db-"):
        key = "db-" + key
    c = db.Historical(key)
    rep = {"datasets": {}, "items": {}}
    for ds in ("XNAS.ITCH", "EQUS.SUMMARY", "EQUS.MINI", "DBEQ.BASIC", "XNYS.PILLAR"):
        try:
            rep["datasets"][ds] = str(c.metadata.get_dataset_range(dataset=ds))
        except Exception as ex:                                          # noqa: BLE001
            rep["datasets"][ds] = repr(ex)[:200]
    Q = {
        "itch_1d_2018_2025": dict(dataset="XNAS.ITCH", schema="ohlcv-1d", start="2018-05-01", end="2026-01-01"),
        "itch_1m_2024_01": dict(dataset="XNAS.ITCH", schema="ohlcv-1m", start="2024-01-01", end="2024-02-01"),
                "eqsum_1d_all": dict(dataset="EQUS.SUMMARY", schema="ohlcv-1d", start="2018-05-01", end="2026-01-01"),
        "eqmini_1m_2024_01": dict(dataset="EQUS.MINI", schema="ohlcv-1m", start="2024-01-01", end="2024-02-01"),
        "dbeq_1m_2024_01": dict(dataset="DBEQ.BASIC", schema="ohlcv-1m", start="2024-01-01", end="2024-02-01"),
        "itch_defs_2024_01": dict(dataset="XNAS.ITCH", schema="definition", start="2024-01-02", end="2024-01-03"),
        "eqsum_1d_2024_07_2025": dict(dataset="EQUS.SUMMARY", schema="ohlcv-1d", start="2024-07-01", end="2026-01-01"),
    }
    # per symbol-day cost of minute bars for active small caps (one day, a few tickers)
    for ds in ("XNAS.ITCH", "EQUS.MINI"):
        for sym in ("GME", "AMC", "SOUN", "MULN"):
            q = dict(dataset=ds, schema="ohlcv-1m", symbols=[sym], start="2024-09-03", end="2024-09-04")
            try:
                rep["items"][f"{ds}_1m_{sym}_1day"] = {"usd": float(c.metadata.get_cost(**q)),
                                                       "bytes": c.metadata.get_billable_size(**q)}
            except Exception as ex:                                      # noqa: BLE001
                rep["items"][f"{ds}_1m_{sym}_1day"] = {"error": repr(ex)[:200]}
            print(ds, sym, rep["items"][f"{ds}_1m_{sym}_1day"], flush=True)
    for k, q in Q.items():
        try:
            rep["items"][k] = {"usd": float(c.metadata.get_cost(symbols="ALL_SYMBOLS", **q)),
                               "gb": round(c.metadata.get_billable_size(symbols="ALL_SYMBOLS", **q) / 1e9, 3)}
        except Exception as ex:                                          # noqa: BLE001
            rep["items"][k] = {"error": repr(ex)[:300]}
        print(k, rep["items"][k], flush=True)
    json.dump(rep, open(os.path.join(out, "eq_cost.json"), "w"), indent=1)
    print(json.dumps(rep["datasets"], indent=1))


if __name__ == "__main__":
    main()
