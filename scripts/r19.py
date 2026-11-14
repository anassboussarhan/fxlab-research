"""Round 19 (docs/round19_preregistration.md). Development data only (to 2025-12-31).

  idx    --symbol USA500IDXUSD|USATECHIDXUSD --out DIR   part A: ORB5, noise-area breakout, C2b at futures costs
  runs   --symbol SYM --out DIR                          part F: prior-day high / low stop runs
  eia    --out DIR                                       part D: oil around the EIA report (needs WTI, Brent M1)
  cdata  --out DIR [--symbols BTCUSDT,ETHUSDT]           Binance perpetual klines -> M1 store files; funding CSV
  crypto --funding DIR --out DIR                         part E: funding-time trades (needs the M1 files)
  carry  --out DIR                                       part C: weekly dip-buying with the carry filter
  lib    --inp DIR --out DIR                             part L: library units on the new instruments
  combine --inp DIR --out DIR                            family H: stats, BH, counting
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = "2025-12-31"
NY = "America/New_York"
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
FUT_PTS = {"USA500IDXUSD": 0.25 + 2.5 / 50, "USATECHIDXUSD": 0.25 + 2.5 / 20}   # points per side
LAB = {"USA500IDXUSD": "US500", "USATECHIDXUSD": "NAS100"}
GRID = [(n, k0) for n in (15, 30, 60) for k0 in (0, 30)]                  # round-18 C2b grid
SESSION = 390                                                              # 09:30-15:59 minutes


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) >= 20 and x.std() > 0 else np.nan


def bh(p, q=0.10):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.flatnonzero(np.isfinite(p))
    o = ok[np.argsort(p[ok])]
    thr = q * np.arange(1, len(ok) + 1) / max(len(ok), 1)
    passed = np.flatnonzero(p[o] <= thr)
    if len(passed):
        out[o[: passed.max() + 1]] = True
    return out


# ============================================================================================ part A
def session_days(m1):
    """Per New York date: minute-indexed arrays over 09:30-15:59 (forward-filled), or skipped if thin."""
    ny = m1.index.tz_convert(NY)
    mins = ny.hour * 60 + ny.minute - 570
    keep = (mins >= 0) & (mins < SESSION) & (ny.dayofweek < 5)
    x = m1[keep]
    k = mins[keep]
    d = ny[keep].normalize().tz_localize(None)
    out = {}
    for day, idx in pd.Series(np.arange(len(x)), index=d).groupby(level=0):
        ii = idx.to_numpy()
        kk = np.asarray(k[ii])
        if len(ii) < 300 or kk[0] != 0:
            continue
        g = x.iloc[ii]
        A = {}
        for nm, b, a_ in (("o", "bid_o", "ask_o"), ("h", "bid_h", "ask_h"), ("l", "bid_l", "ask_l"), ("c", "bid_c", "ask_c")):
            v = np.full(SESSION, np.nan)
            v[kk] = 0.5 * (g[b].to_numpy() + g[a_].to_numpy())
            A[nm] = v
        spr = np.full(SESSION, np.nan)
        spr[kk] = g["ask_o"].to_numpy() - g["bid_o"].to_numpy()
        w = np.ones(SESSION)
        if "n_ticks" in g and g["n_ticks"].notna().mean() > 0.9:
            w = np.zeros(SESSION)
            w[kk] = np.where(np.isfinite(g["n_ticks"].to_numpy(float)) & (g["n_ticks"].to_numpy(float) > 0),
                             g["n_ticks"].to_numpy(float), 1.0)
        cl = pd.Series(A["c"]).ffill().to_numpy()
        for nm in ("o", "h", "l"):                                         # missing minutes: flat at the last close
            A[nm] = np.where(np.isfinite(A[nm]), A[nm], np.r_[cl[0], cl[:-1]])
        A["c"] = cl
        A["spr"] = pd.Series(spr).ffill().bfill().to_numpy()
        A["w"] = w
        if not (np.isfinite(A["o"]).all() and np.isfinite(A["c"]).all() and (A["o"] > 0).all()):
            continue
        out[day] = A
    return out


def orb5_day(A):
    o, c = A["o"][0], A["c"][4]
    if o == c:
        return None
    d = 1 if c > o else -1
    ext = float(np.min(A["l"][:5])) if d > 0 else float(np.max(A["h"][:5]))
    E = A["o"][5]
    sd = (E - ext) * d
    if sd <= 0:
        return None
    stop, tgt = ext, E + d * 10 * sd
    X, ko = None, SESSION - 1
    for k in range(5, SESSION):
        lo, hi, op = A["l"][k], A["h"][k], A["o"][k]
        if d > 0:
            if lo <= stop:
                X, ko = min(op, stop), k
                break
            if hi >= tgt:
                X, ko = max(op, tgt), k
                break
        else:
            if hi >= stop:
                X, ko = max(op, stop), k
                break
            if lo <= tgt:
                X, ko = min(op, tgt), k
                break
    if X is None:
        X = A["c"][-1]
    return [(d, E, X, 5, ko)]


def nb_day(A, sigma, prev_close):
    """Noise-area breakout (VM = 1) with the VWAP trailing stop; trades as (dir, entry, exit, k_in, k_out)."""
    op = A["o"][0]
    ub0, lb0 = max(op, prev_close), min(op, prev_close)
    tp = (A["h"] + A["l"] + A["c"]) / 3
    vwap = np.cumsum(tp * A["w"]) / np.cumsum(A["w"])
    pos, ent, kin, trades = 0, None, None, []
    for m in range(30, SESSION, 30):                                       # 10:00 ... 15:30
        p = A["c"][m - 1]
        ub, lb = ub0 * (1 + sigma[m - 1]), lb0 * (1 - sigma[m - 1])
        tgt = pos
        if p > ub:
            tgt = 1
        elif p < lb:
            tgt = -1
        elif pos > 0 and p < max(ub, vwap[m - 1]):
            tgt = 0
        elif pos < 0 and p > min(lb, vwap[m - 1]):
            tgt = 0
        if tgt != pos:
            px = A["o"][m]
            if pos != 0:
                trades.append((pos, ent, px, kin, m))
            pos = tgt
            if pos != 0:
                ent, kin = px, m
    if pos != 0:
        trades.append((pos, ent, A["c"][-1], kin, SESSION - 1))
    return trades


def costs_of(trades, A, fut_pts, pip):
    from fxlab.execution.costs import BASE
    sf, sp, comm = BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4
    g = fu = cf = 0.0
    for d, E, X, ki, ko in trades:
        g += d * (X / E - 1.0)
        fu += fut_pts / E + fut_pts / X
        si, so = A["spr"][ki], A["spr"][min(ko, SESSION - 1)]
        cf += (0.5 * si + sf * si + sp * pip) / E + (0.5 * so + sf * so + sp * pip) / X + 2 * comm
    return g, fu, cf


def cmd_idx(a):
    from fxlab.data.store import load_m1
    from fxlab.execution.costs import BASE
    sf, sp, comm = BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4
    try:
        m1 = load_m1(a.symbol, "2007-01-01", DEV_END, columns=COLS + ["n_ticks"])
    except Exception:                                                       # noqa: BLE001
        m1 = load_m1(a.symbol, "2007-01-01", DEV_END, columns=COLS)
    pip = float(1e-4 * np.nanmedian(0.5 * (m1.bid_c + m1.ask_c)))
    fut = FUT_PTS[a.symbol]
    D = session_days(m1)
    days = sorted(D)
    print(a.symbol, "session days", len(days), days[0], days[-1], flush=True)
    rows = []
    moves = []                                                             # per day |close(m)/open - 1|
    closes = []                                                            # 16:00 closes (last minute close)
    for i, day in enumerate(days):
        A = D[day]
        mv = np.abs(A["c"] / A["o"][0] - 1.0)
        # ORB5
        t = orb5_day(A)
        if t:
            g, fu, cf = costs_of(t, A, fut, pip)
            rows.append({"test": "ORB5", "day": day, "gross": g, "net_fut": g - fu, "net_fut2": g - 2 * fu, "net_cfd": g - cf,
                         "trades": 1})
        # noise area: needs 14 prior days and the previous close
        if len(moves) >= 14 and closes:
            sigma = np.mean(np.vstack(moves[-14:]), axis=0)
            t = nb_day(A, sigma, closes[-1])
            g, fu, cf = costs_of(t, A, fut, pip) if t else (0.0, 0.0, 0.0)
            dr = np.diff(np.log(np.r_[closes[-15:]])) if len(closes) >= 15 else None
            lev = min(4.0, 0.02 / float(np.std(dr, ddof=1))) if dr is not None and np.std(dr) > 0 else np.nan
            rows.append({"test": "NB", "day": day, "gross": g, "net_fut": g - fu, "net_fut2": g - 2 * fu, "net_cfd": g - cf,
                         "trades": len(t), "lev": lev})
        moves.append(mv)
        closes.append(A["c"][-1])
        # C2b grid (round 18), positions in units; the cost of each unit traded at that minute's price
        mo, mc, n = A["o"], A["c"], SESSION
        vwap = np.cumsum((A["h"] + A["l"] + A["c"]) / 3 * A["w"]) / np.cumsum(A["w"])
        sig = np.sign(mc - vwap)
        k = np.arange(n)
        ret = mo[1:] / mo[:-1] - 1.0
        uf = fut / mo
        uc = (0.5 * A["spr"] + sf * A["spr"] + sp * pip) / mo + comm
        for N, k0 in GRID:
            dec = (k >= k0) & ((k - k0) % N == 0)
            held = pd.Series(np.where(dec, sig, np.nan)).ffill().fillna(0.0).to_numpy()
            pos = np.zeros(n)
            pos[1:] = held[:-1]
            pos[-1] = 0.0
            dpos = np.abs(np.diff(np.r_[0.0, pos]))
            g = float((pos[:-1] * ret).sum())
            fu = float((dpos * uf).sum())
            rows.append({"test": f"C2b_N{N}_K{k0}", "day": day, "gross": g, "net_fut": g - fu, "net_fut2": g - 2 * fu,
                         "net_cfd": g - float((dpos * uc).sum()), "trades": float(dpos.sum()) / 2})
    R = pd.DataFrame(rows)
    R["pair"] = a.symbol
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, f"idx_{a.symbol}.parquet"))
    print(R.groupby("test")[["gross", "net_fut", "net_cfd", "trades"]].mean().mul([1e4, 1e4, 1e4, 1]).round(2).to_string())


# ============================================================================================ part F
def cmd_runs(a):
    import edges_lab2 as E2
    import flows_lab as FL
    from fxlab.validation.metrics import trading_day
    bk = E2.book(a.symbol)
    if bk is None:
        print("no data", a.symbol)
        return
    bk._hl()
    keep = bk.idx <= pd.Timestamp(DEV_END + " 23:59", tz="UTC")
    td = trading_day(bk.idx)
    H = pd.Series(bk.mh, index=bk.idx)[keep].groupby(td[keep]).max()
    L = pd.Series(bk.ml, index=bk.idx)[keep].groupby(td[keep]).min()
    rows = []
    tdays = list(H.index)
    for j in range(1, len(tdays)):
        d, pd_ = tdays[j], tdays[j - 1]
        if (d - pd_).days > 4:
            continue
        pdh, pdl = H[pd_], L[pd_]
        i0 = int(bk.idx.searchsorted(FL.local_ts(d, NY, 8, 0)))
        i1 = int(bk.idx.searchsorted(FL.local_ts(d, NY, 15, 30)))
        jx = bk.at(FL.local_ts(d, NY, 16, 0), 5)
        if jx < 0 or i1 <= i0:
            continue
        for side, hit in (("up", np.flatnonzero(bk.mh[i0:i1] > pdh)), ("down", np.flatnonzero(bk.ml[i0:i1] < pdl))):
            if not len(hit):
                continue
            i = i0 + int(hit[0]) + 1
            if i >= jx:
                continue
            brk = 1 if side == "up" else -1
            for rule, dd in (("fade", -brk), ("follow", brk)):
                r = bk.trade_ij(i, jx, dd)
                if r is None:
                    continue
                rows.append({"pair": a.symbol, "day": d, "side": side, "rule": rule, "net": r[0], "gross": r[1]})
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, f"runs_{a.symbol}.parquet"))
    if len(R):
        print(R.groupby(["rule", "side"])[["net", "gross"]].agg(["mean", "count"]).to_string())


# ============================================================================================ part D
def eia_stocks():
    import requests
    errs = []
    try:
        r = requests.get("https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls", headers=UA, timeout=120)
        r.raise_for_status()
        x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
        x.columns = ["date", "stocks"]
        x["date"] = pd.to_datetime(x["date"])
        return x.dropna().set_index("date")["stocks"].astype(float).sort_index(), "eia.gov WCESTUS1w.xls"
    except Exception as ex:                                                # noqa: BLE001
        errs.append(repr(ex)[:200])
    try:
        r = requests.get("https://api.db.nomics.world/v22/series/EIA/PET/WCESTUS1.W?observations=1", headers=UA, timeout=120)
        r.raise_for_status()
        doc = r.json()["series"]["docs"][0]
        s = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").to_numpy(), index=pd.to_datetime(doc["period"]))
        return s.dropna().sort_index(), "DBnomics EIA/PET/WCESTUS1.W"
    except Exception as ex:                                                # noqa: BLE001
        errs.append(repr(ex)[:200])
    raise RuntimeError("; ".join(errs))


def cmd_eia(a):
    import holidays
    import edges_lab2 as E2
    import flows_lab as FL
    S, src = eia_stocks()
    print("EIA stocks", src, len(S), S.index.min(), S.index.max(), flush=True)
    S = S[(S.index >= "2007-01-01") & (S.index <= DEV_END)]
    dS = S.diff()
    iso = S.index.isocalendar()
    wk, yr = iso.week.to_numpy(), iso.year.to_numpy()
    seas = pd.Series(np.nan, index=S.index)
    for i in range(len(S)):
        m = (wk == wk[i]) & (yr < yr[i]) & (yr >= yr[i] - 5)
        v = dS[m].dropna()
        if len(v) >= 3:
            seas.iloc[i] = v.mean()
    surprise = dS - seas
    us = holidays.US(years=range(2007, 2027))
    books = {s: E2.book(s) for s in ("LIGHTCMDUSD", "BRENTCMDUSD")}
    wti = books["LIGHTCMDUSD"]
    if wti is None:
        raise RuntimeError("no WTI data")
    wti._hl()
    rows, timing = [], []
    for wend, sv in surprise.items():
        if not np.isfinite(sv) or wend < pd.Timestamp("2013-01-01"):
            continue
        fri = wend.normalize()
        wed = fri + pd.Timedelta(days=5)
        hol = any((fri + pd.Timedelta(days=k)).date() in us for k in (3, 4, 5))
        cands = [FL.local_ts(wed, NY, 10, 30)]
        if hol:
            thu = fri + pd.Timedelta(days=6)
            cands = [FL.local_ts(wed, NY, 10, 30), FL.local_ts(thu, NY, 10, 30), FL.local_ts(thu, NY, 11, 0)]
            best, bv = None, -1
            for c in cands:
                i = wti.at(c, 1)
                if i < 0 or i + 1 >= len(wti.idx):
                    continue
                v = abs(wti.mo[i + 1] / wti.mo[i] - 1)
                if v > bv:
                    best, bv = c, v
            cands = [best] if best is not None else []
        if not cands:
            continue
        t0 = cands[0]
        timing.append({"week": fri, "release": t0, "holiday_week": hol})
        t_in = t0 + pd.Timedelta(minutes=5)
        t_out = FL.local_ts(t0.tz_convert(NY).tz_localize(None).normalize(), NY, 14, 30)
        for sym, bk in books.items():
            if bk is None:
                continue
            i0, i5 = bk.at(t0, 1), bk.at(t_in, 2)
            if i0 < 0 or i5 < 0:
                continue
            mv = np.sign((bk.ao[i5] + bk.bo[i5]) - (bk.ao[i0] + bk.bo[i0]))
            for test, d in (("E1", -np.sign(sv)), ("FADE", -mv), ("FOLLOW", mv)):
                if d == 0:
                    continue
                r = bk.trade(t_in, t_out, int(d), tol_min=3)
                if r is None:
                    continue
                rows.append({"pair": sym, "day": pd.Timestamp(t0.tz_convert(NY).date()), "test": test, "net": r[0],
                             "gross": r[1], "surprise": sv})
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, "eia_trades.parquet"))
    pd.DataFrame(timing).to_csv(os.path.join(a.out, "eia_timing.csv"), index=False)
    json.dump({"source": src, "weeks": int(len(timing)), "holiday_weeks": int(sum(t["holiday_week"] for t in timing))},
              open(os.path.join(a.out, "eia_meta.json"), "w"), indent=1)
    if len(R):
        print(R.groupby(["pair", "test"])[["net", "gross"]].agg(["mean", "count"]).to_string())


# ============================================================================================ part E data
def bin_get(url):
    import requests
    for k in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except Exception:                                                  # noqa: BLE001
            pass
        time.sleep(5 * (k + 1))
    return None


def read_zip_csv(blob, names):
    z = zipfile.ZipFile(io.BytesIO(blob))
    raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    hdr = 0 if any(ch.isalpha() for ch in first.split(",")[0]) else None
    return pd.read_csv(io.StringIO(raw), header=hdr, names=None if hdr == 0 else names)


def cmd_cdata(a):
    base = "https://data.binance.vision/data/futures/um/monthly"
    KN = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
          "taker_buy_quote_volume", "ignore"]
    months = pd.period_range("2020-01", "2025-12", freq="M")
    meta = {}
    os.makedirs(a.out, exist_ok=True)
    for sym in a.symbols.split(","):
        parts, fparts, miss = [], [], []
        for m in months:
            tag = f"{m.year}-{m.month:02d}"
            b = bin_get(f"{base}/klines/{sym}/1m/{sym}-1m-{tag}.zip")
            if b is None:
                miss.append(tag)
            else:
                k = read_zip_csv(b, KN)
                k.columns = KN[: len(k.columns)]
                parts.append(k[["open_time", "open", "high", "low", "close", "count"]])
            f = bin_get(f"{base}/fundingRate/{sym}/{sym}-fundingRate-{tag}.zip")
            if f is not None:
                fparts.append(read_zip_csv(f, ["calc_time", "funding_interval_hours", "last_funding_rate"]))
        if not parts:
            meta[sym] = {"error": "no klines", "missing": miss}
            continue
        K = pd.concat(parts, ignore_index=True)
        ot = pd.to_numeric(K.open_time)
        unit = "us" if ot.max() > 1e14 else "ms"
        K["ts"] = pd.to_datetime(ot, unit=unit, utc=True).dt.floor("min").astype("datetime64[ms, UTC]")
        K = K.drop_duplicates("ts").sort_values("ts")
        half = 3.1e-4
        out = pd.DataFrame({"ts": K.ts})
        for c, s in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
            v = pd.to_numeric(K[s]).to_numpy(float)
            out[f"bid_{c}"] = v * (1 - half)
            out[f"ask_{c}"] = v * (1 + half)
        out["n_ticks"] = pd.to_numeric(K["count"]).fillna(0).astype("int32").to_numpy()
        out["spread_mean"] = out["ask_c"] - out["bid_c"]
        out = out[out.ts <= pd.Timestamp(DEV_END + " 23:59", tz="UTC")]
        d = os.path.join(a.out, "m1", sym)
        os.makedirs(d, exist_ok=True)
        for y, g in out.groupby(out.ts.dt.year):
            g.reset_index(drop=True).to_parquet(os.path.join(d, f"{sym}_{y}.parquet"), index=False)
        F = pd.concat(fparts, ignore_index=True) if fparts else pd.DataFrame()
        if len(F):
            ct = pd.to_numeric(F.calc_time)
            F["ts"] = pd.to_datetime(ct, unit="us" if ct.max() > 1e14 else "ms", utc=True).dt.round("min")
            F["rate"] = pd.to_numeric(F.last_funding_rate)
            F = F.drop_duplicates("ts").sort_values("ts")[["ts", "rate"]]
            F.to_csv(os.path.join(a.out, f"funding_{sym}.csv"), index=False)
        meta[sym] = {"minutes": int(len(out)), "first": str(out.ts.min()), "last": str(out.ts.max()), "missing_months": miss,
                     "funding_stamps": int(len(F))}
        print(sym, meta[sym], flush=True)
    json.dump(meta, open(os.path.join(a.out, "cdata_meta.json"), "w"), indent=1)


def cmd_crypto(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.funding, "**", "funding_*.csv"), recursive=True)):
        sym = os.path.basename(f)[8:-4]
        F = pd.read_csv(f, parse_dates=["ts"])
        files = sorted(glob.glob(os.path.join(a.funding, "**", "m1", sym, "*.parquet"), recursive=True))
        if not files:
            continue
        M = pd.concat([pd.read_parquet(x, columns=["ts", "bid_o", "ask_o"]) for x in files]).drop_duplicates("ts")
        mid = pd.Series(0.5 * (M.bid_o.to_numpy() + M.ask_o.to_numpy()), index=pd.DatetimeIndex(M.ts)).sort_index()
        F = F[F.ts <= pd.Timestamp(DEV_END + " 23:59", tz="UTC")].reset_index(drop=True)
        prev = F.rate.shift(1).to_numpy()
        T = pd.DatetimeIndex(F.ts)
        h = pd.Timedelta(minutes=60)
        P = np.vstack([mid.reindex(T - h).to_numpy(), mid.reindex(T).to_numpy(), mid.reindex(T + h).to_numpy()]).T
        crowd = np.where(prev > 1e-4 + 1e-9, 1, np.where(prev < 1e-4 - 1e-9, -1, 0))
        ok = np.isfinite(prev) & (crowd != 0) & np.isfinite(P).all(axis=1)
        for test, d, e, x in (("PRE", -crowd, P[:, 0], P[:, 1]), ("POST", crowd, P[:, 1], P[:, 2])):
            g = d * (x / e - 1)
            rows.append(pd.DataFrame({"pair": sym, "day": T.tz_convert(None).normalize(), "test": test, "gross": g,
                                      "net": g - 10e-4, "net_maker": g - 4e-4})[ok])
    R = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, "crypto_trades.parquet"))
    if len(R):
        print(R.groupby(["pair", "test"])[["gross", "net"]].agg(["mean", "count"]).to_string())


# ============================================================================================ part C
def cmd_carry(a):
    import forward_d7 as FD
    import r18_long as RL
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    X = FD.load(None)
    X = X[X.index <= DEV_END]
    months = pd.period_range("1999-01", "2025-12", freq="M")
    RT, src = RL.oecd_rates(months)                                        # already lagged one month
    RT = RT.ffill()
    print("rates", src, flush=True)
    rows = []
    for s in RL.FX:
        b, q = s[:3], s[3:]
        px = (X[q] / X[b]).dropna()
        iso = px.index.isocalendar()
        key = (iso.year * 100 + iso.week).to_numpy()
        last = np.r_[key[1:] != key[:-1], True]
        w = px[last]
        nxt = pd.Series(px.index, index=px.index).shift(-1)[last]
        fill_t = nxt.to_numpy()
        fill_px = px.reindex(nxt.to_numpy()).to_numpy()
        N = w.diff().abs().ewm(alpha=1 / 20, adjust=False, min_periods=20).mean().to_numpy()
        wv = w.to_numpy(float)
        bars = pd.DataFrame({"open": np.r_[wv[0], wv[:-1]], "high": wv, "low": wv, "close": wv},
                            index=pd.DatetimeIndex(w.index).tz_localize("UTC"))
        for rid in RL.RULES:
            e = CATALOG[rid]
            side = e.fn(bars.copy(), **e.params)["side"].clip(lower=0).to_numpy()
            state, ent = 0, None
            for i in range(len(side)):
                if pd.isna(fill_px[i]):
                    break
                if state == 0 and side[i] > 0 and np.isfinite(N[i]):
                    per = pd.Timestamp(fill_t[i]).to_period("M")
                    diff = (RT.at[per, b] - RT.at[per, q]) if per in RT.index else np.nan
                    state, ent = 1, (fill_t[i], fill_px[i], N[i] / w.iloc[i], diff)
                elif state == 1 and side[i] <= 0:
                    days = (pd.Timestamp(fill_t[i]) - pd.Timestamp(ent[0])).days
                    carry = (ent[3] - 0.0075) * days / 365 if np.isfinite(ent[3]) else np.nan
                    r = fill_px[i] / ent[1] - 1 - FD.COST + (carry if np.isfinite(carry) else 0.0)
                    rows.append({"rule": rid, "symbol": s, "entry": ent[0], "exit": fill_t[i], "ret": r, "R": r / (2 * ent[2]),
                                 "diff": ent[3], "carry": carry})
                    state, ent = 0, None
    T = pd.DataFrame(rows)
    T["entry"], T["exit"] = pd.to_datetime(T.entry), pd.to_datetime(T.exit)
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, "carry_trades.csv"), index=False)
    json.dump(src, open(os.path.join(a.out, "carry_rates_source.json"), "w"), indent=1)
    print(T.groupby([T.rule, T["diff"] > 0]).R.agg(["mean", "count"]).to_string())


def month_t(T):
    if len(T) < 2:
        return np.nan
    x = T.R.astype(float)
    m = x.mean()
    g = (x - m).groupby(pd.to_datetime(T.exit).dt.to_period("M")).sum()
    se = math.sqrt((g ** 2).sum()) / len(x)
    return float(m / se) if se > 0 else np.nan


# ============================================================================================ part L
LIB_PERIODS = {"bond": ("2019-01-01", "2022-01-01"), "crypto": ("2020-01-01", "2023-01-01"), "other": ("2013-01-01", "2019-01-01")}


def kind_of(sym):
    if "TR" in sym and ("BOND" in sym or "GILT" in sym):
        return "bond"
    if sym.endswith("USDT"):
        return "crypto"
    return "other"


def cmd_lib(a):
    from scipy.stats import norm
    recs = []
    for f in sorted(glob.glob(os.path.join(a.inp, "**", "daily1[57]_*.npz"), recursive=True)):
        eng = "D" if os.path.basename(f).startswith("daily15") else "I"
        sym = os.path.basename(f).split("_", 1)[1][:-4]
        z = np.load(f, allow_pickle=True)
        days = pd.to_datetime(pd.Series(z["days"]).astype(str).str[:10])
        lo, split = (pd.Timestamp(x) for x in LIB_PERIODS[kind_of(sym)])
        m1 = ((days >= lo) & (days < split)).to_numpy()
        m2 = ((days >= split) & (days <= pd.Timestamp(DEV_END))).to_numpy()
        U = pd.DataFrame({"unit": z["unit"].astype(str), "cfg": z["cfg"].astype(str), "dir": z["dir"].astype(int)})
        U = U[U.cfg != "terciles"]                                         # rules only (round-17 PAIR-UNIT tier)
        P = z["pnl"]
        for (u, d), g in U.groupby(["unit", "dir"]):
            x = P[g.index.to_numpy()].astype(float).mean(axis=0)
            x1, x2 = x[m1], x[m2]
            recs.append({"symbol": sym, "engine": eng, "unit": u, "dir": d, "placebo": u.startswith("PLACEBO"),
                         "p1_mean": float(x1.mean()) if len(x1) else np.nan, "p2_mean": float(x2.mean()) if len(x2) else np.nan,
                         "p1_t": tstat(x1), "p2_t": tstat(x2), "n1": int(m1.sum()), "n2": int(m2.sum()),
                         "settings": int(len(g))})
        print(sym, eng, "units", U.groupby(["unit", "dir"]).ngroups, flush=True)
    R = pd.DataFrame(recs)
    pl = R[R.placebo].groupby(["symbol", "engine"]).p2_t.quantile(0.95).rename("placebo_p95")
    R = R.join(pl, on=["symbol", "engine"])
    T = R[~R.placebo].copy()
    T["p"] = 1 - norm.cdf(T.p2_t.fillna(-9))
    T["bh"] = bh(T.p.to_numpy())
    T["own"] = (T.p1_mean > 0) & (T.p2_mean > 0) & (T.p2_t >= 1.645) & (T.p2_t > T.placebo_p95)
    T["counts"] = T.own & T.bh
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "lib_units.csv"), index=False)
    summ = {"units": int(len(T)), "placebo_units": int(R.placebo.sum()), "own_rule": int(T.own.sum()), "bh": int(T.bh.sum()),
            "counts": int(T.counts.sum()), "first_bh_threshold": 0.10 / max(len(T), 1),
            "p2_t_ge_1645_and_p1_pos": int(((T.p1_mean > 0) & (T.p2_t >= 1.645)).sum()),
            "top": T.sort_values("p2_t", ascending=False).head(25).round(4).to_dict("records"),
            "by_symbol": T.groupby("symbol").agg(units=("unit", "size"), own=("own", "sum"), med_p2_t=("p2_t", "median"),
                                                 max_p2_t=("p2_t", "max")).round(3).reset_index().to_dict("records"),
            "placebo_p95": pl.round(3).reset_index().to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "lib_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: summ[k] for k in ("units", "own_rule", "bh", "counts", "p2_t_ge_1645_and_p1_pos")}))
    print(pd.DataFrame(summ["by_symbol"]).to_string())


# ============================================================================================ family H
def per(x, lo, split, hi):
    x = x.dropna()
    a, b = x[(x.index >= lo) & (x.index < split)], x[(x.index >= split) & (x.index <= hi)]
    return {"p1_n": len(a), "p1_bp": a.mean() * 1e4 if len(a) else np.nan, "p1_t": tstat(a),
            "p2_n": len(b), "p2_bp": b.mean() * 1e4 if len(b) else np.nan, "p2_t": tstat(b)}


def cmd_combine(a):
    from scipy.stats import norm
    os.makedirs(a.out, exist_ok=True)
    fam, rows = [], []
    E = pd.Timestamp(DEV_END)

    def add(tid, net, gross, lo, split, extra=None, p1_key="p1_bp"):
        sn, sg = per(net, lo, split, E), per(gross, lo, split, E)
        rec = {"id": tid, **{f"net_{k}": v for k, v in sn.items()}, **{f"gross_{k}": v for k, v in sg.items()}}
        if extra:
            rec.update(extra)
        rows.append(rec)
        fam.append({"id": tid, "t": sn["p2_t"], "p1_pos": bool(sn[p1_key] > 0), "p2_pos": bool(sn["p2_bp"] > 0)})

    S12 = pd.Timestamp("2019-01-01")
    # part A
    files = glob.glob(os.path.join(a.inp, "**", "idx_*.parquet"), recursive=True)
    if files:
        A = pd.concat([pd.read_parquet(f) for f in files])
        A["day"] = pd.to_datetime(A.day)
        surf = []
        for sym, lab in LAB.items():
            x = A[A.pair == sym]
            if not len(x):
                continue
            for t, g in x.groupby("test"):
                g = g.set_index("day").sort_index()
                s = {"index": lab, "test": t}
                for c in ("gross", "net_fut", "net_fut2", "net_cfd"):
                    st = per(g[c], pd.Timestamp("2007-01-01"), S12, E)
                    s.update({f"{c}_{k}": v for k, v in st.items()})
                s["trades_per_day"] = float(g.trades.mean())
                s["years_pos_p2"] = f"{int((g[g.index >= S12].net_fut.groupby(g[g.index >= S12].index.year).sum() > 0).sum())}/" \
                                    f"{g[g.index >= S12].index.year.nunique()}"
                if t == "NB":
                    v = g.net_fut * g.lev
                    s["nb_volsized_p2_bp"] = float(v[v.index >= S12].mean() * 1e4)
                    s["nb_volsized_p2_t"] = tstat(v[v.index >= S12])
                    s["nb_volsized_p2_sharpe"] = float(v[v.index >= S12].mean() / v[v.index >= S12].std() * math.sqrt(252))
                surf.append(s)
            S = pd.DataFrame([r for r in surf if r["index"] == lab and r["test"].startswith("C2b")])
            best = S.sort_values("net_fut_p1_t", ascending=False).iloc[0]["test"] if len(S) else None
            for t, tid in (("ORB5", f"A-ORB5-{lab}"), ("NB", f"A-NB-{lab}"), (best, f"A-C2b-{lab}")):
                if t is None:
                    continue
                g = x[x.test == t].set_index("day").sort_index()
                add(tid, g.net_fut, g.gross, pd.Timestamp("2007-01-01"), S12, extra={"setting": t,
                    "net_fut2_p2_bp": per(g.net_fut2, pd.Timestamp("2007-01-01"), S12, E)["p2_bp"],
                    "net_cfd_p2_bp": per(g.net_cfd, pd.Timestamp("2007-01-01"), S12, E)["p2_bp"]})
        pd.DataFrame(surf).to_csv(os.path.join(a.out, "A_surface.csv"), index=False)
    # part C
    f = glob.glob(os.path.join(a.inp, "**", "carry_trades.csv"), recursive=True)
    if f:
        T = pd.read_csv(f[0], parse_dates=["entry", "exit"])
        T = T[(T.entry >= "2000-01-01") & (T.exit <= DEV_END)]
        for tid, sub in (("C-D7", T[T.rule == "S.DAILY.DOUBLE_SEVEN"]), ("C-FAM", T)):
            inc, exc = sub[sub["diff"] > 0], sub[~(sub["diff"] > 0)]
            p1 = lambda z: z[z.entry < "2013-01-01"]                       # noqa: E731
            p2 = lambda z: z[z.entry >= "2013-01-01"]                      # noqa: E731
            i2, e2 = p2(inc), p2(exc)
            welch = (i2.R.mean() - e2.R.mean()) / math.sqrt(i2.R.var() / len(i2) + e2.R.var() / len(e2)) \
                if len(i2) > 2 and len(e2) > 2 else np.nan
            rec = {"id": tid, "p1_trades": len(p1(inc)), "p1_mean_R": p1(inc).R.mean(), "p1_t": month_t(p1(inc)),
                   "p2_trades": len(i2), "p2_mean_R": i2.R.mean(), "p2_t": month_t(i2),
                   "p2_excluded_trades": len(e2), "p2_excluded_mean_R": e2.R.mean(), "welch_t": welch,
                   "unfiltered_p1_mean_R": p1(sub).R.mean(), "unfiltered_p2_mean_R": p2(sub).R.mean(),
                   "unfiltered_p2_t": month_t(p2(sub))}
            rows.append(rec)
            fam.append({"id": tid, "t": rec["p2_t"], "p1_pos": bool(rec["p1_mean_R"] > 0), "p2_pos": bool(rec["p2_mean_R"] > 0),
                        "extra_ok": bool(welch >= 1.645)})
    # part D
    f = glob.glob(os.path.join(a.inp, "**", "eia_trades.parquet"), recursive=True)
    if f:
        D = pd.read_parquet(f[0])
        D["day"] = pd.to_datetime(D.day)
        for sym, lab in (("LIGHTCMDUSD", "WTI"), ("BRENTCMDUSD", "BRENT")):
            for test in ("E1", "FADE", "FOLLOW"):
                g = D[(D.pair == sym) & (D.test == test)].groupby("day")
                if g.ngroups:
                    add(f"D-{test}-{lab}", g.net.mean(), g.gross.mean(), pd.Timestamp("2013-01-01"), S12)
    # part E
    f = glob.glob(os.path.join(a.inp, "**", "crypto_trades.parquet"), recursive=True)
    if f:
        C = pd.read_parquet(f[0])
        C["day"] = pd.to_datetime(C.day).dt.tz_localize(None)
        for sym, lab in (("BTCUSDT", "BTC"), ("ETHUSDT", "ETH")):
            for test in ("PRE", "POST"):
                g = C[(C.pair == sym) & (C.test == test)].groupby("day")
                if g.ngroups:
                    mk = per(g.net_maker.sum(), pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01"), E)
                    add(f"E-{test}-{lab}", g.net.sum(), g.gross.sum(), pd.Timestamp("2020-01-01"), pd.Timestamp("2023-01-01"),
                        extra={"maker_p2_bp": mk["p2_bp"], "maker_p2_t": mk["p2_t"]})
    # part F
    files = glob.glob(os.path.join(a.inp, "**", "runs_*.parquet"), recursive=True)
    if files:
        F = pd.concat([pd.read_parquet(x) for x in files])
        F["day"] = pd.to_datetime(F.day)
        grp = {"IDX": ("USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY"), "MET": ("XAUUSD", "XAGUSD")}
        for gname, syms in grp.items():
            for rule in ("fade", "follow"):
                x = F[F.pair.isin(syms) & (F.rule == rule)]
                daily = x.groupby(["day", "pair"])[["net", "gross"]].sum().groupby("day").mean()
                add(f"F-{rule.upper()}-{gname}", daily.net, daily.gross, pd.Timestamp("2007-01-01"), S12)
    Fm = pd.DataFrame(fam)
    Fm["p"] = 1 - norm.cdf(Fm.t.fillna(-9).astype(float))
    Fm["bh"] = bh(Fm.p.to_numpy())
    Fm["own"] = Fm.p1_pos & Fm.p2_pos & (Fm.t >= 1.645)
    if "extra_ok" in Fm:
        Fm["own"] = Fm.own & Fm.extra_ok.fillna(True).astype(bool)
    Fm["counts"] = Fm.own & Fm.bh
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "H_rows.csv"), index=False)
    Fm.to_csv(os.path.join(a.out, "H_family.csv"), index=False)
    summ = {"tests": int(len(Fm)), "counts": int(Fm.counts.sum()), "bh": int(Fm.bh.sum()), "own": int(Fm.own.sum()),
            "family": Fm.round(4).to_dict("records"), "rows": pd.DataFrame(rows).round(4).to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "H_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string())
    print(Fm.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--funding", default="")
    ap.add_argument("--inp", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"idx": cmd_idx, "runs": cmd_runs, "eia": cmd_eia, "cdata": cmd_cdata, "crypto": cmd_crypto, "carry": cmd_carry,
     "lib": cmd_lib, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    main()
