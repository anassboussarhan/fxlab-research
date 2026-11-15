"""Round 22 parts E (scheduled events) and C1 (commodity spreads and ratios), every parameter optimised walk-forward
(docs/round22_preregistration.md).

  eia    --out DIR    E-EIA crude (WTI, Brent) and E-NG natural-gas storage (gas CFD)
  fund   --funding DIR --out DIR   E-FUND crypto funding times (BTC, ETH perpetuals)
  spread --out DIR    C1 Brent-WTI, gold / silver, copper / gold (D1 and H4)
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402
from r22_daily import pick, rng, run_wf, save  # noqa: E402

NY = "America/New_York"
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}


def eia_series(url):
    import requests
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    x = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
    x = x.iloc[:, :2]
    x.columns = ["date", "v"]
    x["date"] = pd.to_datetime(x["date"])
    return x.dropna().set_index("date")["v"].astype(float).sort_index()


def surprises(S):
    S = S[(S.index >= "2007-01-01") & (S.index <= wf22.DEV_END)]
    dS = S.diff()
    iso = S.index.isocalendar()
    wk, yr = iso.week.to_numpy(), iso.year.to_numpy()
    out = pd.Series(np.nan, index=S.index)
    for i in range(len(S)):
        m = (wk == wk[i]) & (yr < yr[i]) & (yr >= yr[i] - 5)
        v = dS[m].dropna()
        if len(v) >= 3:
            out.iloc[i] = dS.iloc[i] - v.mean()
    sd = out.rolling(156, min_periods=52).std().shift()
    return out, out / sd


def releases(z, mid, weekday_offset, hh, mm, alt):
    """Release time per report: default weekday (days after the week-ending Friday) at hh:mm New York; in US-holiday
    weeks the candidate with the largest absolute 1-minute move (size only)."""
    import holidays
    import flows_lab as FL
    us = holidays.US(years=range(2007, 2027))
    o = mid.o
    out = []
    for wend, zz in z.items():
        if not np.isfinite(zz):
            continue
        fri = wend.normalize()
        base = fri + pd.Timedelta(days=weekday_offset)
        hol = any((fri + pd.Timedelta(days=k)).date() in us for k in range(3, weekday_offset + 1))
        cands = [FL.local_ts(base, NY, hh, mm)]
        if hol:
            cands = [FL.local_ts(fri + pd.Timedelta(days=dd), NY, h2, m2) for dd, h2, m2 in alt]
        best, bv = None, -1
        for c in cands:
            i = o.index.searchsorted(c)
            if i + 1 >= len(o) or abs((o.index[i] - c).total_seconds()) > 60:
                continue
            v = abs(o.iloc[i + 1] / o.iloc[i] - 1)
            if v > bv:
                best, bv = c, v
        if best is not None:
            out.append((best, zz))
    return out


def side_cost(px, spr, pip):
    from fxlab.execution.costs import BASE
    return (0.5 * spr + BASE.slip_spread_frac * spr + BASE.slip_fixed_pips * pip) / px + BASE.commission_bps_per_side * 1e-4


def event_family(mid, pip, events, U, label, rows, ser, first):
    o = mid.o
    ix = o.index
    oo, sp = o.to_numpy(float), mid.spr.to_numpy(float)
    days = pd.DatetimeIndex(sorted({t.tz_convert(NY).normalize().tz_localize(None) for t, _ in events}))
    M, desc = [], []
    pos = [(ix.searchsorted(t), t, z) for t, z in events]
    for u in U:
        dly = int(rng(u[0], 1, 30))
        mode = pick(u[1], ["exit_min", "close"])
        hold = int(rng(u[2], 15, 330))
        sig = pick(u[3], ["surprise", "follow", "fade"])
        k = int(rng(u[4], 1, 15))
        thr = rng(u[5], 0, 2)
        vals = []
        for i0, t, z in pos:
            if i0 + max(dly, k) + 2 >= len(oo):
                continue
            if sig == "surprise":
                if abs(z) < thr:
                    continue
                d = -np.sign(z)                                            # draw (negative surprise) -> long
            else:
                mv = np.sign(oo[i0 + k] - oo[i0])
                d = mv if sig == "follow" else -mv
            if d == 0:
                continue
            ie = ix.searchsorted(t + pd.Timedelta(minutes=max(dly, k if sig != "surprise" else 0)))
            if mode == "close":
                tx = pd.Timestamp(t.tz_convert(NY).date()).tz_localize(NY) + pd.Timedelta(hours=16)
                tx = tx.tz_convert("UTC")
            else:
                tx = ix[ie] + pd.Timedelta(minutes=hold) if ie < len(ix) else t
            ix_ = ix.searchsorted(tx)
            if ie >= len(oo) or ix_ >= len(oo) or ix_ <= ie or (ix[ix_] - ix[ie]) > pd.Timedelta(hours=10):
                continue
            g = d * (oo[ix_] / oo[ie] - 1)
            net = g - side_cost(oo[ie], sp[ie], pip) - side_cost(oo[ix_], sp[ix_], pip)
            vals.append((t.tz_convert(NY).normalize().tz_localize(None), net))
        s = pd.Series(0.0, index=days)
        if vals:
            v = pd.Series([x for _, x in vals], index=pd.DatetimeIndex([d for d, _ in vals])).groupby(level=0).sum()
            s = s.add(v.reindex(days).fillna(0.0), fill_value=0.0)
        M.append(s)
        desc.append({"delay": dly, "exit": mode, "hold": hold, "signal": sig, "k": k, "thr": round(thr, 2)})
    W = pd.concat(M, axis=1)
    run_wf(rows, ser, "E", label[0], label[1], W.to_numpy().T, W.index, first, desc=desc, min_active=20)


def cmd_eia(a):
    from r21_intraday import load
    rows, ser = [], {}
    U = qmc.Sobol(6, scramble=True, seed=51).random(128)
    crude = eia_series("https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls")
    _, zc = surprises(crude)
    for sym in ("LIGHTCMDUSD", "BRENTCMDUSD"):
        try:
            mid, pip = load(sym, "2013-01-01")
        except Exception as ex:                                            # noqa: BLE001
            print("no data", sym, repr(ex)[:100])
            continue
        pip = float(1e-4 * np.nanmedian(mid.c))
        ev = releases(zc, mid, 5, 10, 30, [(5, 10, 30), (6, 10, 30), (6, 11, 0)])
        event_family(mid, pip, ev, U, ("E-EIA-crude", sym), rows, ser, 2016)
    try:
        gas = eia_series("https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls")
        _, zg = surprises(gas)
        mid, pip = load("GASCMDUSD", "2013-01-01")
        pip = float(1e-4 * np.nanmedian(mid.c))
        ev = releases(zg, mid, 6, 10, 30, [(5, 10, 30), (5, 12, 0), (6, 10, 30), (6, 12, 0), (7, 10, 30)])
        event_family(mid, pip, ev, U, ("E-NG-storage", "GASCMDUSD"), rows, ser, 2016)
    except Exception as ex:                                                # noqa: BLE001
        print("NG failed", repr(ex)[:200], flush=True)
        rows.append({"part": "E", "family": "E-NG-storage", "symbol": "GASCMDUSD", "note": f"failed: {repr(ex)[:150]}"})
    save(a.out, "E", rows, ser)


def cmd_fund(a):
    rows, ser = [], {}
    U = qmc.Sobol(5, scramble=True, seed=53).random(128)
    for sym in ("BTCUSDT", "ETHUSDT"):
        ff = glob.glob(os.path.join(a.funding, "**", f"funding_{sym}.csv"), recursive=True)
        pf = sorted(glob.glob(os.path.join(a.funding, "**", "m1", sym, "*.parquet"), recursive=True))
        if not ff or not pf:
            continue
        F = pd.read_csv(ff[0], parse_dates=["ts"])
        F = F[F.ts <= pd.Timestamp("2025-12-31 23:59", tz="UTC")].reset_index(drop=True)
        P = pd.concat([pd.read_parquet(x, columns=["ts", "bid_o", "ask_o"]) for x in pf]).drop_duplicates("ts")
        mid = pd.Series(0.5 * (P.bid_o.to_numpy() + P.ask_o.to_numpy()), index=pd.DatetimeIndex(P.ts)).sort_index()
        T = pd.DatetimeIndex(F.ts)
        prev = F.rate.shift(1).to_numpy()
        days = pd.DatetimeIndex(sorted(set(T.tz_convert(None).normalize())))
        M, desc = [], []
        for u in U:
            pre = int(rng(u[0], 15, 240))
            post = int(rng(u[1], 15, 240))
            thr = rng(u[2], 0, 5e-4)
            side = pick(u[3], [-1, 1])                                      # -1: against the crowd, +1: with it
            mode = pick(u[4], ["pre", "post", "both"])
            dev = prev - 1e-4
            crowd = np.where(dev > thr, 1, np.where(dev < -thr, -1, 0))
            pA = mid.reindex(T - pd.Timedelta(minutes=pre)).to_numpy()
            pT = mid.reindex(T).to_numpy()
            pB = mid.reindex(T + pd.Timedelta(minutes=post)).to_numpy()
            net = np.zeros(len(T))
            if mode in ("pre", "both"):
                g = side * crowd * (pT / pA - 1)
                net += np.where(np.isfinite(g) & (crowd != 0), g - 10e-4, 0.0)
            if mode in ("post", "both"):
                g = -side * crowd * (pB / pT - 1)                           # the rebound after the stamp
                net += np.where(np.isfinite(g) & (crowd != 0), g - 10e-4, 0.0)
            s = pd.Series(net, index=T.tz_convert(None).normalize()).groupby(level=0).sum().reindex(days).fillna(0.0)
            M.append(s)
            desc.append({"pre": pre, "post": post, "thr": round(thr, 5), "side": side, "mode": mode})
        W = pd.concat(M, axis=1)
        run_wf(rows, ser, "E", "E-FUND-crypto", sym, W.to_numpy().T, W.index, 2023, desc=desc, min_active=30)
    save(a.out, "EF", rows, ser)


def cmd_spread(a):
    from r21_intraday import load, bars
    rows, ser = [], {}
    pairs = (("BRENTCMDUSD", "LIGHTCMDUSD"), ("XAUUSD", "XAGUSD"), ("COPPERCMDUSD", "XAUUSD"))
    cache = {}
    for A, B in pairs:
        try:
            for s in (A, B):
                if s not in cache:
                    m, p = load(s, "2007-01-01")
                    cache[s] = (m, p)
        except Exception as ex:                                            # noqa: BLE001
            print("no data", A, B, repr(ex)[:100])
            continue
        (mA, pA), (mB, pB) = cache[A], cache[B]
        U = qmc.Sobol(6, scramble=True, seed=57).random(128)
        BB = {}
        for tf in ("1D", "4h"):
            ba, bb = bars(mA, tf), bars(mB, tf)
            ix = ba.index.intersection(bb.index)
            BB[tf] = (ba.reindex(ix), bb.reindex(ix))
        days_all = pd.DatetimeIndex(sorted(set(BB["1D"][0].index.tz_convert(None).normalize())))
        M, desc = [], []
        for u in U:
            tf = pick(u[0], ["1D", "4h"])
            lb = int(rng(u[1], 10, 250))
            ez = rng(u[2], 1, 3)
            xz = rng(u[3], 0, 1)
            sz = ez + rng(u[4], 1, 2.5)
            mh = int(rng(u[5], 5, 120))
            ba, bb = BB[tf]
            lr = np.log(ba.c.to_numpy()) - np.log(bb.c.to_numpy())
            mu = pd.Series(lr).rolling(lb).mean().to_numpy()
            sd = pd.Series(lr).rolling(lb).std().to_numpy()
            z = (lr - mu) / sd
            oa, ob = ba.o.to_numpy(), bb.o.to_numpy()
            ca_, cb_ = ba.spr.to_numpy(), bb.spr.to_numpy()
            trades = []
            pos, e, held = 0, None, 0
            for t in range(lb, len(z) - 1):
                if not np.isfinite(z[t]):
                    continue
                if pos != 0:
                    held += 1
                    if (pos == 1 and (z[t] >= -xz or z[t] <= -sz)) or (pos == -1 and (z[t] <= xz or z[t] >= sz)) or held >= mh:
                        x = t + 1
                        g = pos * ((oa[x] / oa[e] - 1) - (ob[x] / ob[e] - 1))
                        c = side_cost(oa[e], ca_[e], pA) + side_cost(ob[e], cb_[e], pB) + side_cost(oa[x], ca_[x], pA) + \
                            side_cost(ob[x], cb_[x], pB)
                        trades.append((ba.index[e], g - c))
                        pos, held = 0, 0
                        continue
                if pos == 0:
                    if z[t] <= -ez and z[t] > -sz:
                        pos, e, held = 1, t + 1, 0                          # long A, short B
                    elif z[t] >= ez and z[t] < sz:
                        pos, e, held = -1, t + 1, 0
            s = pd.Series(0.0, index=days_all)
            if trades:
                v = pd.Series([x for _, x in trades], index=pd.DatetimeIndex([d.tz_convert(None).normalize() for d, _ in trades]))
                s = s.add(v.groupby(level=0).sum().reindex(days_all).fillna(0.0), fill_value=0.0)
            M.append(s)
            desc.append({"tf": tf, "lb": lb, "entry_z": round(ez, 2), "exit_z": round(xz, 2), "stop_z": round(sz, 2), "max_hold": mh})
        W = pd.concat(M, axis=1)
        first = max(2010, int(W.index.year.min()) + 3)
        run_wf(rows, ser, "C", "C1-spread", f"{A}-{B}", W.to_numpy().T, W.index, first, desc=desc, min_active=20)
    save(a.out, "C1", rows, ser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--funding", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"eia": cmd_eia, "fund": cmd_fund, "spread": cmd_spread}[a.cmd](a)


if __name__ == "__main__":
    main()
