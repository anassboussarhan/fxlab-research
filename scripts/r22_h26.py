"""Round 22: the pre-registered 2026 check of the streams that counted (docs/round22_preregistration.md).
Configurations are selected on data up to 2025 only (the walk-forward's 2026 choice) and traded on
2026-01-01 .. 2026-10-31. Verdict: supported (mean > 0, t >= 1.645), contradicted (the difference from the
2019-2025 mean has t <= -1.645), otherwise inconclusive.

  fvg --symbols A,B,... --extra DIR --out DIR     I-FVG per instrument and pooled (2026 M1 candles in --extra)
  cc  --funding DIR --out DIR                     K cash-and-carry, BTC and ETH (2026 Binance public files fetched here)
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402

H0, H1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-31")
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}


def verdict(dev, h):
    dev, h = dev.dropna(), h.dropna()
    d2 = dev[(dev.index >= "2019-01-01") & (dev.index <= wf22.DEV_END)]
    if len(h) < 5:
        return {"n": int(len(h)), "verdict": "no data"}
    m, se = float(h.mean()), float(h.std(ddof=1) / math.sqrt(len(h)))
    md, sed = float(d2.mean()), float(d2.std(ddof=1) / math.sqrt(len(d2)))
    t = m / se if se > 0 else float("nan")
    td = (m - md) / math.sqrt(se ** 2 + sed ** 2)
    v = "supported" if (m > 0 and t >= 1.645) else ("contradicted" if td <= -1.645 else "inconclusive")
    return {"n": int(len(h)), "mean": m, "t": t, "dev_2019_2025_mean": md, "t_vs_dev": td, "verdict": v,
            "by_month": {str(k): float(x) for k, x in h.groupby(h.index.to_period("M")).mean().items()}}


def load_mid(sym, extra):
    from fxlab.data.store import load_m1
    COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
    m = load_m1(sym, "2007-01-01", "2025-12-31", columns=COLS)
    parts = [m]
    for f in sorted(glob.glob(os.path.join(extra, "**", f"{sym}*2026*.parquet"), recursive=True)):
        x = pd.read_parquet(f)
        if "ts" in x.columns:
            x = x.set_index("ts")
        x.index = pd.DatetimeIndex(x.index)
        if x.index.tz is None:
            x.index = x.index.tz_localize("UTC")
        parts.append(x[COLS])
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    m = m[m.index <= pd.Timestamp("2026-10-31 23:59", tz="UTC")]
    mid = pd.DataFrame({k: 0.5 * (m[f"bid_{k}"] + m[f"ask_{k}"]) for k in "ohlc"})
    mid["spr"] = (m.ask_o - m.bid_o).clip(lower=0)
    return mid[np.isfinite(mid.o) & (mid.o > 0)], len(parts) - 1


def cmd_fvg(a):
    import r22_I as I
    os.makedirs(a.out, exist_ok=True)
    res, h26 = {}, {}
    for sym in a.symbols.split(","):
        mid, nfiles = load_mid(sym, a.extra)
        if "IDX" in sym:
            pip = float(1e-4 * np.nanmedian(mid.c))
        elif sym.startswith("XAU"):
            pip = 0.1
        elif sym.startswith("XAG"):
            pip = 0.01
        else:
            pip = 0.01 if sym.endswith("JPY") else 1e-4
        index = "IDX" in sym
        tfs = ("5min", "15min") if index else ("15min", "1h")
        BB = {tf: I.Bars(I.bars(mid, tf, index), sym, pip) for tf in tfs}
        cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]].day)))
        U = qmc.Sobol(I.FAMS["FVG"] + 1, scramble=True, seed=22 + len("FVG")).random(128)
        M = np.vstack([I.daily(I.fam_FVG(BB[I.pick(u[-1], list(tfs))], u), BB[I.pick(u[-1], list(tfs))], cal).to_numpy() for u in U])
        first = max(2010, int(cal.year.min()) + 3)
        s5, s1, ch = wf22.walk_forward(M, cal, first, min_active=20, last_year=2026)
        dev = s5[s5.index <= wf22.DEV_END]
        h = s5[(s5.index >= H0) & (s5.index <= H1)]
        res[sym] = {"files_2026": nfiles, "dev": wf22.stream_stats(dev), "h2026": verdict(dev, h),
                    "chosen_2026": ch.get(2026, {})}
        h26[sym] = s5
        print(sym, json.dumps(res[sym]["h2026"], default=str)[:400], flush=True)
    P = pd.concat(h26, axis=1)
    pooled = P.mean(axis=1, skipna=True).dropna()
    res["POOLED"] = {"dev": wf22.stream_stats(pooled[pooled.index <= wf22.DEV_END]),
                     "h2026": verdict(pooled[pooled.index <= wf22.DEV_END], pooled[(pooled.index >= H0) & (pooled.index <= H1)])}
    json.dump(res, open(os.path.join(a.out, "h26_fvg.json"), "w"), indent=1, default=str)
    P.to_parquet(os.path.join(a.out, "h26_fvg_series.parquet"))
    print("POOLED", json.dumps(res["POOLED"]["h2026"], default=str)[:400])


def bin_daily(kind, sym, months):
    import requests
    base = f"https://data.binance.vision/data/{kind}/monthly/klines/{sym}/1d"
    rows = []
    for m in months:
        r = requests.get(f"{base}/{sym}-1d-{m}.zip", headers=UA, timeout=60)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        d = pd.read_csv(io.StringIO(z.read(z.namelist()[0]).decode()), header=None)
        if not str(d.iloc[0, 0]).isdigit():
            d = d.iloc[1:]
        d = d.iloc[:, [0, 4]].astype(float)
        rows.append(d)
    D = pd.concat(rows)
    t = D.iloc[:, 0]
    idx = pd.to_datetime(t, unit="us" if t.max() > 1e14 else "ms").dt.normalize()
    return pd.Series(D.iloc[:, 1].to_numpy(), index=pd.DatetimeIndex(idx)).sort_index()


def bin_funding(sym, months):
    import requests
    rows = []
    for m in months:
        r = requests.get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip",
                         headers=UA, timeout=60)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        rows.append(pd.read_csv(io.StringIO(z.read(z.namelist()[0]).decode())))
    F = pd.concat(rows)
    ct = pd.to_numeric(F.iloc[:, 0])
    F["ts"] = pd.to_datetime(ct, unit="us" if ct.max() > 1e14 else "ms")
    F["rate"] = pd.to_numeric(F.iloc[:, 2])
    return F.groupby(F.ts.dt.normalize()).rate.sum()


def cmd_cc(a):
    from scipy.stats import qmc as Q
    os.makedirs(a.out, exist_ok=True)
    months = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13) if (y, m) <= (2026, 10)]
    res, ser = {}, {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        sp = bin_daily("spot", sym, months)
        pc = bin_daily("futures/um", sym, months)
        fd = bin_funding(sym, months)
        idx2 = pc.index.intersection(sp.index)
        idx2 = idx2[idx2 <= H1]
        pc, sp2 = pc.reindex(idx2), sp.reindex(idx2)
        basis_ret = sp2.pct_change(fill_method=None) - pc.pct_change(fill_method=None)
        fdd = fd.reindex(idx2).fillna(0.0)
        U = Q.Sobol(3, scramble=True, seed=37 + len(sym)).random(128)
        M = []
        for u in U:
            w = int(1 + u[0] * 29)
            en = u[1] * 3e-4 * 3
            ex = en * u[2]
            tv = fdd.rolling(w).mean().shift(1).to_numpy()
            pos = np.zeros(len(idx2))
            cur = 0.0
            for t in range(len(idx2)):
                if cur == 0 and np.isfinite(tv[t]) and tv[t] > en:
                    cur = 1.0
                elif cur == 1 and np.isfinite(tv[t]) and tv[t] < ex:
                    cur = 0.0
                pos[t] = cur
            ps = pd.Series(pos, index=idx2)
            M.append((ps * (fdd + basis_ret.fillna(0.0)) - ps.diff().abs().fillna(0.0) * 2 * 10e-4).to_numpy())
        M = np.vstack(M)
        s5, s1, ch = wf22.walk_forward(M, idx2, 2023, min_active=30, last_year=2026)
        dev = s5[s5.index <= wf22.DEV_END]
        h = s5[(s5.index >= H0) & (s5.index <= H1)]
        res[sym] = {"dev": wf22.stream_stats(dev), "h2026": verdict(dev, h), "chosen_2026": ch.get(2026, {}),
                    "funding_2026_mean_per_day": float(fdd[fdd.index >= H0].mean()),
                    "days_2026": int((idx2 >= H0).sum())}
        ser[sym] = s5
        print(sym, json.dumps(res[sym]["h2026"], default=str)[:400], flush=True)
    json.dump(res, open(os.path.join(a.out, "h26_cc.json"), "w"), indent=1, default=str)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, "h26_cc_series.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--funding", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fvg": cmd_fvg, "cc": cmd_cc}[a.cmd](a)


if __name__ == "__main__":
    main()
