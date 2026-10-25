"""Normalise option-chain exports into the lab's format: one parquet per underlying with columns
date, expiration, strike, cp (P / C), bid, ask, iv, delta. Rows after 2025-12-31 are dropped (2026 is locked).

Input: CSVs exported from the DoltHub database post-no-preference/options (table option_chain) with
`dolt sql -r csv`, one file per underlying named raw_<SYM>.csv. Column names are mapped from the common
spellings (act_symbol, call_put, vol / implied_volatility, ...).
Usage: python scripts/options_chain_fetch.py --raw DIR --out DIR"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

END = pd.Timestamp("2025-12-31")
MAP = {"call_put": "cp", "type": "cp", "option_type": "cp", "put_call": "cp", "vol": "iv",
       "implied_volatility": "iv", "impliedvolatility": "iv", "iv": "iv", "expiry": "expiration",
       "expiration_date": "expiration", "quote_date": "date", "bid_price": "bid", "ask_price": "ask"}


def norm(f):
    C = pd.read_csv(f, low_memory=False)
    C.columns = [c.strip().lower() for c in C.columns]
    C = C.rename(columns={c: MAP[c] for c in C.columns if c in MAP})
    need = ["date", "expiration", "strike", "cp", "bid", "ask"]
    miss = [c for c in need if c not in C.columns]
    if miss:
        raise ValueError(f"{f}: missing {miss}; columns {list(C.columns)}")
    C["date"] = pd.to_datetime(C["date"])
    C["expiration"] = pd.to_datetime(C["expiration"])
    C = C[C.date <= END]
    C["cp"] = C["cp"].astype(str).str.strip().str[0].str.upper()
    for c in ("strike", "bid", "ask", "iv", "delta"):
        if c in C.columns:
            C[c] = pd.to_numeric(C[c], errors="coerce")
    keep = [c for c in ("date", "expiration", "strike", "cp", "bid", "ask", "iv", "delta") if c in C.columns]
    return C[keep].dropna(subset=["strike", "bid", "ask"]).drop_duplicates(["date", "expiration", "strike", "cp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for f in sorted(os.listdir(a.raw)):
        if not (f.startswith("raw_") and f.endswith(".csv")):
            continue
        sym = f[4:-4]
        try:
            C = norm(os.path.join(a.raw, f))
        except Exception as ex:                                  # noqa: BLE001
            meta[sym] = {"error": str(ex)[:300]}
            continue
        if C.empty:
            meta[sym] = {"rows": 0}
            continue
        C.to_parquet(os.path.join(a.out, f"chain_{sym}.parquet"), index=False)
        d = C.groupby("date").size()
        gaps = d.index.to_series().diff().dt.days
        meta[sym] = {"rows": int(len(C)), "dates": int(len(d)), "first": str(d.index.min().date()),
                     "last": str(d.index.max().date()), "median_rows_per_date": float(d.median()),
                     "dates_per_year": {str(y): int(n) for y, n in d.groupby(d.index.year).size().items()},
                     "max_gap_days": int(gaps.max()) if len(gaps) > 1 else None,
                     "weekday_share": {str(k): float(v) for k, v in
                                       d.index.to_series().dt.weekday.value_counts(normalize=True).sort_index().items()},
                     "expirations_per_date_median": float(C.groupby("date").expiration.nunique().median()),
                     "has_iv": bool("iv" in C.columns and C["iv"].notna().mean() > 0.5),
                     "has_delta": bool("delta" in C.columns and C["delta"].notna().mean() > 0.5),
                     "zero_bid_share": float((C.bid <= 0).mean())}
        print(sym, json.dumps(meta[sym])[:400], flush=True)
    with open(os.path.join(a.out, "chain_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
