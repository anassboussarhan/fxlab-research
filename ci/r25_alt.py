"""Round 25 part ALT data (docs/round25_preregistration.md): the point-in-time altcoin universe and its bars, from the
Binance public archive (data.binance.vision; free, no key).

universe: every USDT-margined perpetual in the archive (delisted ones included; BTC, ETH, stablecoins and dated
  contracts left out), its daily quote volume from the monthly 1d klines, and for each year Y in 2020-2025 the 30 with
  the highest mean daily quote volume over Y - 1 among those with 90 or more days in Y - 1. Writes universe.json
  (members by year, counts only; no prices).
bars: for the given members, the 1h klines of their whole history to 2025-12 as store bars (bid / ask = price x
  (1 -/+ 3.1 bp), the half-spread with fees of BTC and ETH since round 19), resampled to H2-D1 as every instrument's
  bars are, plus the funding-rate files (funding_SYM.csv: ts, rate).
Usage: python ci/r25_alt.py universe --out DIR
       python ci/r25_alt.py bars --universe DIR/universe.json --part i/k --out DIR
"""
import argparse
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import r24_common as C  # noqa: E402
from r24_crypto import ARCH, s3_list  # noqa: E402

BASE = "data/futures/um/monthly/"
STABLE = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "UST", "PAX", "SUSD", "EUR", "GBP", "AEUR", "USDE", "PYUSD",
          "XUSD", "RLUSD", "USD1", "BFUSD", "EURI", "USTC"}
HALF = 3.1e-4
KN = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
      "taker_buy_quote_volume", "ignore"]
DEV_END = pd.Timestamp(os.environ.get("ALT_END", "2025-12-31") + " 23:59", tz="UTC")


def zip_csv(key, names):
    r = C.get(ARCH + key, tries=5, timeout=180)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    d = pd.read_csv(io.StringIO(raw), header=None)
    if not str(d.iloc[0, 0]).replace(".", "").isdigit():
        d = d.iloc[1:]
    d = d.iloc[:, : len(names)]
    d.columns = names[: d.shape[1]]
    return d


def safe(f, *a):
    try:
        return f(*a)
    except Exception as ex:                                                # noqa: BLE001
        print("skip", a[:1], repr(ex)[:120], flush=True)
        return None


def ts_of(x):
    x = pd.to_numeric(x).astype("int64")
    return pd.to_datetime(np.where(x > 1e14, x // 1000, x), unit="ms", utc=True)


def perps():
    out = []
    for p in s3_list(f"{BASE}klines/"):
        s = p.rstrip("/").split("/")[-1]
        if not s.endswith("USDT") or "_" in s or s in ("BTCUSDT", "ETHUSDT"):
            continue
        base = s[:-4]
        if base in STABLE or base.startswith("1000000") and base[7:] in STABLE:
            continue
        out.append(s)
    return sorted(set(out))


def months_of(sym, interval, years):
    keys = [k for k in s3_list(f"{BASE}klines/{sym}/{interval}/") if k.endswith(".zip")]
    return [k for k in keys if int(k.rsplit("-", 2)[-2]) in years]


def daily_volume(sym):
    keys = months_of(sym, "1d", range(2019, DEV_END.year + 1))
    parts = [p for p in (safe(zip_csv, k, KN) for k in keys) if p is not None and len(p)]
    if not parts:
        return None
    d = pd.concat(parts)
    s = pd.Series(pd.to_numeric(d["quote_volume"], errors="coerce").to_numpy(float), index=ts_of(d["open_time"]))
    s = s[~s.index.duplicated()].sort_index()
    return s[s.index <= DEV_END]


def cmd_universe(a):
    os.makedirs(a.out, exist_ok=True)
    syms = perps()
    print("USDT perpetuals in the archive (after exclusions):", len(syms), flush=True)
    with ThreadPoolExecutor(8) as ex:
        vols = dict(zip(syms, ex.map(lambda s: safe(daily_volume, s), syms)))
    vols = {s: v for s, v in vols.items() if v is not None and len(v)}
    members, ranks = {}, {}
    for Y in range(2020, int(os.environ.get("ALT_LAST_YEAR", "2025")) + 1):
        cand = []
        for s, v in vols.items():
            w = v[v.index.year == Y - 1].dropna()
            if len(w) >= 90:
                cand.append((float(w.mean()), s))
        top = [s for _, s in sorted(cand, reverse=True)[:30]]
        ranks[str(Y)] = {"eligible": len(cand), "members": top}
        for s in top:
            members.setdefault(s, []).append(Y)
    first = {s: str(v.index.min().date()) for s, v in vols.items() if s in members}
    json.dump({"members": members, "by_year": ranks, "first_day": first, "n_perpetuals": len(syms),
               "n_with_volume": len(vols)}, open(os.path.join(a.out, "universe.json"), "w"), indent=1)
    print({Y: (r["eligible"], len(r["members"])) for Y, r in ranks.items()}, "distinct members", len(members), flush=True)


def build(sym, out):
    from fxlab.data.store import resample
    keys = months_of(sym, "1h", range(2019, DEV_END.year + 1))
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda k: safe(zip_csv, k, KN), keys) if p is not None and len(p)]
    if not parts:
        return {"error": "no 1h klines"}
    K = pd.concat(parts, ignore_index=True)
    K["ts"] = ts_of(K["open_time"]).floor("h")
    K = K.drop_duplicates("ts").sort_values("ts")
    K = K[K.ts <= DEV_END]
    m = pd.DataFrame(index=pd.DatetimeIndex(K.ts, name="ts"))
    for c, s in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
        v = pd.to_numeric(K[s], errors="coerce").to_numpy(float)
        m[f"bid_{c}"] = v * (1 - HALF)
        m[f"ask_{c}"] = v * (1 + HALF)
    m["n_ticks"] = pd.to_numeric(K["count"], errors="coerce").fillna(0).to_numpy(float)
    m["spread_mean"] = m["ask_c"] - m["bid_c"]
    m = m.dropna(subset=["bid_c", "ask_c"])
    os.makedirs(os.path.join(out, "bars"), exist_ok=True)
    info = {"hours": int(len(m)), "first": str(m.index.min()), "last": str(m.index.max())}
    for tf in ("H1", "H2", "H4", "H8", "H12", "D1"):
        b = resample(m, tf)
        b = b[b.index <= DEV_END]
        b.to_parquet(os.path.join(out, "bars", f"{sym}_{tf}.parquet"), compression="zstd")
        info[tf] = int(len(b))
    fk = [k for k in s3_list(f"{BASE}fundingRate/{sym}/") if k.endswith(".zip")]
    fparts = [p for p in (safe(zip_csv, k, ["calc_time", "funding_interval_hours", "last_funding_rate"]) for k in fk)
              if p is not None and len(p)]
    if fparts:
        F = pd.concat(fparts, ignore_index=True)
        F["ts"] = ts_of(F["calc_time"]).round("min")
        F["rate"] = pd.to_numeric(F["last_funding_rate"], errors="coerce")
        F = F.dropna(subset=["rate"]).drop_duplicates("ts").sort_values("ts")
        F = F[F.ts <= DEV_END][["ts", "rate"]]
        F.to_csv(os.path.join(out, f"funding_{sym}.csv"), index=False)
        info["funding_stamps"] = int(len(F))
    else:
        info["funding_stamps"] = 0
    return info


def cmd_bars(a):
    U = json.load(open(a.universe))
    syms = sorted(U["members"])
    i, k = (int(x) for x in a.part.split("/"))
    mine = syms[i::k] if not a.syms else [x for x in a.syms.split(",") if x]
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for s in mine:
        meta[s] = safe(build, s, a.out) or {"error": "failed"}
        print(s, meta[s], flush=True)
    json.dump(meta, open(os.path.join(a.out, f"alt_bars_meta_{i}.json"), "w"), indent=1)
    bad = [s for s, v in meta.items() if "error" in v]
    print("coins", len(mine), "failed", bad, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["universe", "bars"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--universe", default="")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--syms", default="")
    a = ap.parse_args()
    {"universe": cmd_universe, "bars": cmd_bars}[a.cmd](a)


if __name__ == "__main__":
    main()
