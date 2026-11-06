"""Round 9 tests (rules: docs/round9_preregistration.md, fixed before the data was read).

Parts: --m1 (G1, AT1, AT2, AT4, S5 and, once their data is in, N1-N3, S3, S4), --opt (O3, O4, O5, AT6 on the
Databento CME options), --daily (S1, S2, AT3, AT5), --combine (Benjamini-Hochberg over the primary rows).
Usage: python scripts/round9_lab.py --m1 --out DIR [--r9data DIR] [--dbn DIR --compact DIR]"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import queue_lab as Q                                                        # noqa: E402  (month_t, summ, periods)

DEV_END = pd.Timestamp("2025-12-31")
FIT_END = pd.Timestamp("2018-12-31")
LOG = {}
RES = {}


def nv(x):
    return Q.naive(x)


def per(x, per_year=None, split=FIT_END):
    x = pd.Series(x, dtype=float).dropna()
    x.index = nv(x.index)
    x = x[x.index <= DEV_END]
    return {"all": Q.summ(x, per_year), "fit": Q.summ(x[x.index <= split], per_year),
            "check": Q.summ(x[x.index > split], per_year)}


def book(sym):
    import edges_lab2 as E
    return E.book(sym)


def net_of(r):
    import untested_intraday_lab as U
    return U.net_of(r)


def mid(bk, i):
    return 0.5 * (bk.bo[i] + bk.ao[i])


def cost_rt(bk, i):
    """Round-trip cost in price at bar i: the spread plus slippage on both sides plus commission."""
    s = bk.ao[i] - bk.bo[i]
    return s + 2 * (bk.sf * s + bk.sp * bk.pip) + 2 * bk.comm * mid(bk, i)


# ------------------------------------------------------------------------------------------- G1
def g1():
    import flow_edges_lab as F
    import flows_lab as FL
    import untested_intraday_lab as U
    out = {}
    for sym in ("USDJPY", "EURJPY"):
        bk = book(sym)
        if bk is None:
            out[sym] = {"status": "data missing"}
            continue
        G, _, _ = F.calendar(bk)
        rows = {}
        for d in G:
            d = pd.Timestamp(d)
            if d > DEV_END:
                continue
            i0, j0 = F.window(bk, d)
            if i0 < 0:
                continue                                                 # the B0 nights only
            t0 = FL.local_ts(d - pd.Timedelta(days=1), "America/Chicago", 17, 5)
            t1 = U.jst(d, 9, 53)
            i, j = bk.at(t0, 10), bk.at(t1, 65)
            if i < 0 or j <= i:
                continue
            P = mid(bk, i)
            g = mid(bk, j) / P - 1.0
            fut_cost = P * 1.15e-6 if sym == "USDJPY" else np.nan       # 1.5 ticks of 6J + $5 per 12.5m yen
            r = bk.trade_ij(i, j, 1, t0, t1)
            rb = bk.trade_ij(i0, j0, 1)
            rows[d] = {"gross": g, "fut_net": g - fut_cost, "spot_net": net_of(r) if r else np.nan,
                       "b0_spot_net": net_of(rb) if rb else np.nan, "entry_utc_hour": bk.idx[i].hour}
        D = pd.DataFrame.from_dict(rows, orient="index")
        D.to_csv(os.path.join(A.out, f"g1_{sym}.csv"))
        out[sym] = {c: per(D[c]) for c in D.columns if c != "entry_utc_hour"}
        out[sym]["entry_hours"] = D.entry_utc_hour.value_counts().to_dict()
        print("G1", sym, {c: round(out[sym][c]["check"].get("mean_bp", np.nan), 2) for c in ("gross", "fut_net", "spot_net", "b0_spot_net")}, flush=True)
    RES["G1"] = out
    if "fut_net" in out.get("USDJPY", {}):
        RES.setdefault("_primary_g1", {})["G1"] = out["USDJPY"]["fut_net"]


# ------------------------------------------------------------------------------------------- AT1
def at1():
    bk = book("XAUUSD")
    if bk is None:
        RES["AT1"] = {"status": "data missing"}
        return
    m = bk.m
    mc = 0.5 * (m["bid_c"] + m["ask_c"])
    mc.index = nv(mc.index)
    c5 = mc.resample("5min").last().dropna()
    sma, sd = c5.rolling(20).mean(), c5.rolling(20).std()
    z = (c5 - sma) / sd
    cand = z[(z.abs() > 2.2)].index
    rows = {}
    nxt = pd.Timestamp("1900-01-01")
    for t in cand:
        tc = t + pd.Timedelta(minutes=5)                                 # the M5 bar closes at its label + 5 min
        if tc < nxt or tc > DEV_END + pd.Timedelta(days=1):
            continue
        hm = tc.hour * 60 + tc.minute
        if 20 * 60 + 55 <= hm <= 22 * 60 + 5:
            continue
        tcu = tc.tz_localize("UTC")
        i = bk.at(tcu, 2)
        if i < 0:
            continue
        j = bk.at(tcu + pd.Timedelta(minutes=60), 10)
        if j <= i:
            continue
        c = cost_rt(bk, i)
        s = float(sd.loc[t])
        if not (np.isfinite(c) and s >= 3 * c):
            continue
        P = mid(bk, i)
        d = -1 if z.loc[t] > 0 else 1
        tp = d * (float(sma.loc[t]) - P)
        if tp <= 0:
            continue
        r = bk.bracket(i, j, d, tp, 1.5 * s)
        if r is None:
            continue
        net = net_of(r)
        rows[tc] = {"net": net, "net_1.5x": net - 0.5 * c / P, "net_2x": net - c / P, "gross": float(r[1]),
                    "cost_bp": c / P * 1e4, "z": float(z.loc[t])}
        nxt = tc + pd.Timedelta(minutes=60)
    D = pd.DataFrame.from_dict(rows, orient="index")
    D.to_csv(os.path.join(A.out, "at1_trades.csv"))
    RES["AT1"] = {c: per(D[c]) for c in ("net", "net_1.5x", "net_2x", "gross")} if len(D) else {"n": 0}
    if len(D):
        RES["AT1"]["median_cost_bp"] = float(D.cost_bp.median())
        RES.setdefault("_primary", {})["AT1"] = RES["AT1"]["net"]
    print("AT1", len(D), RES["AT1"].get("net", {}).get("check", {}).get("mean_bp"), flush=True)


# ------------------------------------------------------------------------------------------- AT2
def daily_px(sym, hh=21, mm=0):
    import flows_lab as FL
    bk = book(sym)
    if bk is None:
        return None
    rows = {}
    for d in pd.bdate_range("2007-01-01", DEV_END):
        t = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=hh, minutes=mm)
        i = bk.at(t, 5)
        if i < 0:
            continue
        P = mid(bk, i)
        s = bk.ao[i] - bk.bo[i]
        rows[pd.Timestamp(d)] = {"mid": P, "cs": (0.5 * s + bk.sf * s + bk.sp * bk.pip) / P + bk.comm,
                                 "swap": bk.swap_pips * bk.pip / P, "ts": bk.idx[i]}
    D = pd.DataFrame.from_dict(rows, orient="index")
    D["nights"] = [FL.Book.rollovers(a, b) for a, b in zip(D.ts.shift(1).fillna(D.ts.iloc[0]), D.ts)]
    return D


def hurst_rs(x):
    x = np.asarray(x, float)
    ns, rs = [], []
    for n in (10, 20, 50, 125, 250):
        if n > len(x):
            break
        vals = []
        for k in range(len(x) // n):
            c = x[k * n:(k + 1) * n]
            y = np.cumsum(c - c.mean())
            s = c.std()
            if s > 0:
                vals.append((y.max() - y.min()) / s)
        if vals:
            ns.append(n)
            rs.append(np.mean(vals))
    if len(ns) < 3:
        return np.nan
    return float(np.polyfit(np.log(ns), np.log(rs), 1)[0])


def at2():
    G, S = daily_px("XAUUSD"), daily_px("XAGUSD")
    if G is None or S is None:
        RES["AT2"] = {"status": "data missing"}
        return
    J = G.join(S, lsuffix="_g", rsuffix="_s", how="inner")
    R = np.log(J.mid_g) - np.log(J.mid_s)
    dR = R.diff()
    pos = pd.Series(0.0, index=J.index)
    H = pd.Series(np.nan, index=J.index)
    for k in range(260, len(J)):
        h = hurst_rs(dR.iloc[k - 249:k + 1].values)
        H.iloc[k] = h
        win = R.iloc[k - 59:k + 1]
        zz = (R.iloc[k] - win.mean()) / win.std() if win.std() > 0 else 0.0
        if h < 0.45 and abs(zz) > 1:
            pos.iloc[k] = -np.sign(zz)
        elif h > 0.55:
            pos.iloc[k] = np.sign(R.iloc[k] - R.iloc[k - 20])
    nxt = dR.shift(-1)
    turn = (pos - pos.shift(1).fillna(0)).abs()
    nights = J.nights_g.shift(-1).fillna(1)
    pnl = pos * nxt - turn * (J.cs_g + J.cs_s) - pos.abs() * nights * (J.swap_g + J.swap_s)
    pnl = pnl[J.index >= J.index[260]].dropna()
    pnl.to_csv(os.path.join(A.out, "at2_daily.csv"))
    RES["AT2"] = {"net": per(pnl, 252), "share_fade_days": float((H < 0.45).mean()), "share_follow_days": float((H > 0.55).mean()),
                  "hurst_median": float(H.median())}
    RES.setdefault("_primary", {})["AT2"] = RES["AT2"]["net"]
    print("AT2", RES["AT2"]["net"]["check"].get("mean_bp"), RES["AT2"]["hurst_median"], flush=True)


# ------------------------------------------------------------------------------------------- AT4
def at4():
    import flows_lab as FL
    out = {}
    for sym in ("USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCHF"):
        bk = book(sym)
        if bk is None:
            out[sym] = {"status": "data missing"}
            continue
        rows = {}
        for d in pd.bdate_range("2007-01-01", DEV_END):
            t0, t1 = FL.local_ts(d, "Europe/London", 8, 0), FL.local_ts(d, "Europe/London", 8, 30)
            t2, t3 = FL.local_ts(d, "Europe/London", 16, 0), FL.local_ts(d, "Europe/London", 12, 0)
            i0, i1, i2, i3 = bk.at(t0, 3), bk.at(t1, 3), bk.at(t2, 5), bk.at(t3, 5)
            if min(i0, i1) < 0:
                continue
            r = mid(bk, i1) / mid(bk, i0) - 1
            if r == 0:
                continue
            side = 1 if r > 0 else -1
            row = {}
            for k, j in (("to16", i2), ("to12", i3)):
                if j > i1:
                    tr = bk.trade_ij(i1, j, side)
                    if tr is not None:
                        row[k], row[k + "_gross"] = net_of(tr), float(tr[1])
            if row:
                rows[pd.Timestamp(d)] = row
        D = pd.DataFrame.from_dict(rows, orient="index")
        D.to_csv(os.path.join(A.out, f"at4_{sym}.csv"))
        out[sym] = {c: per(D[c], 252) for c in D.columns}
        print("AT4", sym, round(out[sym]["to16"]["check"].get("mean_bp", np.nan), 2), round(out[sym]["to16_gross"]["check"].get("mean_bp", np.nan), 2), flush=True)
    RES["AT4"] = out
    if "to16" in out.get("USDJPY", {}):
        RES.setdefault("_primary", {})["AT4"] = out["USDJPY"]["to16"]


# ------------------------------------------------------------------------------------------- S5
def s5():
    import flows_lab as FL
    import holidays
    import untested_intraday_lab as U
    bk = book("USDJPY")
    if bk is None:
        RES["S5"] = {"status": "data missing"}
        return
    us = holidays.US(years=range(2006, 2027))
    jp = FL.jp_holidays(range(2006, 2027))

    def next_bd(d, hol):
        d = d + pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in hol:
            d += pd.Timedelta(days=1)
        return d

    def roll(d, hol):
        while d.weekday() >= 5 or d in hol:
            d += pd.Timedelta(days=1)
        return d

    out = {}
    for name, dates in (("mid_quarter", [pd.Timestamp(y, m, 15) for y in range(2007, 2026) for m in (2, 5, 8, 11)]),
                        ("month_end", [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y in range(2007, 2026) for m in range(1, 13)])):
        rows = {}
        for d in dates:
            pay = roll(d, us)
            td = next_bd(pay, jp)
            if td > DEV_END:
                continue
            t0, t1 = U.jst(td, 10, 5), U.jst(td, 15, 0)            # the M1 store has no bars 00:00-01:00 UTC before 2019
            r = bk.trade(t0, t1, -1, 5)
            if r is not None:
                rows[td] = {"net": net_of(r), "gross": float(r[1])}
        D = pd.DataFrame.from_dict(rows, orient="index")
        out[name] = {c: per(D[c]) for c in D.columns}
    RES["S5"] = out
    RES.setdefault("_primary", {})["S5"] = out["mid_quarter"]["net"]
    print("S5", out["mid_quarter"]["net"]["check"].get("mean_bp"), out["mid_quarter"]["net"]["check"].get("n"), flush=True)


# ------------------------------------------------------------------------------------------- options (O3-O5, AT6)
TICK = {"EUR": 5e-5, "JPY": 5e-7}
PARENTS = {"EUR": ("6E", ("6E_OPT", "EUU_OPT")), "JPY": ("6J", ("6J_OPT", "JPU_OPT"))}
OPT_SPLIT = pd.Timestamp("2017-12-31")
_OPT = {}


def load_opts(ccy):
    """Settlements S (K, P, F scaled as in O1), open interest O (K, OI) and the futures closes C for one currency."""
    if ccy in _OPT:
        return _OPT[ccy]
    import databento_lab as DL
    fut, opts = PARENTS[ccy]
    fpath = glob.glob(os.path.join(A.dbn, f"F1_volume_{fut}_FUT.dbn.zst"))
    if not fpath:
        _OPT[ccy] = None
        return None
    C, _ = DL.futures_closes(fpath[0])
    defs, stats = [], []
    for o in opts:
        fd, fs = os.path.join(A.compact, f"O1fx_defs_{o}.parquet"), os.path.join(A.compact, f"O1fx_stats_{o}.parquet")
        if os.path.exists(fd) and os.path.exists(fs):
            defs.append(pd.read_parquet(fd))
            stats.append(pd.read_parquet(fs))
    if not defs:
        _OPT[ccy] = None
        return None
    D = pd.concat(defs).drop_duplicates("instrument_id", keep="last")
    D = D[D["instrument_class"].astype(str).str.upper().isin(["C", "P", "CALL", "PUT"])].copy()
    D["cp"] = D["instrument_class"].astype(str).str.upper().str[0]
    D["exp"] = pd.DatetimeIndex(pd.to_datetime(D["expiration"], utc=True)).tz_convert("America/Chicago").tz_localize(None).normalize()
    X = pd.concat(stats)
    ref = X["ts_ref"] if "ts_ref" in X else X["ts_event"]
    X["date"] = pd.DatetimeIndex(pd.to_datetime(ref, utc=True)).tz_convert("America/Chicago").tz_localize(None).normalize()
    X = X.merge(D[["instrument_id", "strike_price", "cp", "exp", "underlying"]], on="instrument_id", how="inner")
    X = X[X["underlying"].isin(C.columns) & (X.date <= X.exp)]
    fprice = np.array([C.at[d, u] if d in C.index else np.nan for d, u in zip(X.date, X.underlying)])
    X["F"] = fprice
    ratio = np.nanmedian(X.strike_price.values / fprice)
    scale = 10.0 ** (-round(math.log10(ratio))) if np.isfinite(ratio) and ratio > 0 else 1.0
    X["K"] = X.strike_price * scale
    is_set = X["stat_type"].astype(str).str.upper().str.contains("3$|SETTLEMENT", regex=True)
    is_oi = X["stat_type"].astype(str).str.upper().str.contains("9$|OPEN_INTEREST", regex=True)
    S = X[is_set & (X.price > 0)].sort_values("ts_event").drop_duplicates(["instrument_id", "date"], keep="last").copy()
    near = (S.K / S.F - 1).abs() < 0.01
    pr = float(np.nanmedian((S.price / S.F)[near])) if near.any() else np.nan
    pscale = 1.0 if (np.isfinite(pr) and 1e-4 <= pr <= 1e-1) else (10.0 ** (-round(math.log10(pr / 0.01))) if np.isfinite(pr) and pr > 0 else 1.0)
    S["P"] = S.price * pscale
    S = S.dropna(subset=["F"])
    O = X[is_oi].sort_values("ts_event").drop_duplicates(["instrument_id", "date"], keep="last").copy()
    O["OI"] = O["quantity"].astype(float)
    LOG.setdefault("opt_units", {})[ccy] = {"strike_scale": scale, "premium_scale": pscale, "settle_rows": int(len(S)),
                                            "oi_rows": int(len(O)), "oi_median": float(O.OI.median()) if len(O) else None}
    _OPT[ccy] = (S, O, C)
    return _OPT[ccy]


def opt_per(x):
    return per(x, None, OPT_SPLIT)


def rr25(day, F, T):
    import databento_lab as DL
    out = {}
    for cp, sel in (("C", day[(day.cp == "C") & (day.K >= F) & (day.K <= F * 1.06)]),
                    ("P", day[(day.cp == "P") & (day.K <= F) & (day.K >= F * 0.94)])):
        pts = []
        for K, P in zip(sel.K, sel.P):
            iv = DL.implied(F, K, T, P, cp)
            if np.isfinite(iv):
                pts.append((abs(DL.bs76(F, K, T, iv, cp)[1]), iv))
        pts.sort()
        if len(pts) < 2 or not (pts[0][0] <= 0.25 <= pts[-1][0]):
            return np.nan
        out[cp] = float(np.interp(0.25, [p[0] for p in pts], [p[1] for p in pts]))
    return out["C"] - out["P"]


def o3():
    res, series = {}, {}
    for ccy in ("EUR", "JPY"):
        L = load_opts(ccy)
        if L is None:
            res[ccy] = {"status": "data missing"}
            continue
        S, O, C = L
        dates = pd.Series(sorted(S.date.unique()))
        wk = dates.groupby(dates.dt.to_period("W-FRI")).max().values
        rows = {}
        for d in wk:
            d = pd.Timestamp(d)
            day = S[S.date == d]
            if day.empty or d > DEV_END:
                continue
            exps = [e for e in day.exp.unique() if 10 <= (pd.Timestamp(e) - d).days <= 45]
            if not exps:
                continue
            E = min(exps, key=lambda e: abs((pd.Timestamp(e) - d).days - 30))
            dd = day[day.exp == E]
            u = dd.underlying.value_counts().index[0]
            dd = dd[dd.underlying == u]
            F = float(dd.F.iloc[0])
            rows[d] = {"rr": rr25(dd, F, (pd.Timestamp(E) - d).days / 365.0), "u": u, "F": F}
        W = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        W["ret"] = [C[u].get(n, np.nan) / C[u].get(d, np.nan) - 1 if n is not None else np.nan
                    for d, u, n in zip(W.index, W.u, list(W.index[1:]) + [None])]
        W["drr"] = W.rr.diff()
        W["pos"] = np.sign(W.drr)
        W["z"] = (W.rr - W.rr.rolling(52, min_periods=26).mean()) / W.rr.rolling(52, min_periods=26).std()
        W["pos_z"] = np.where(W.z > 1, -1.0, np.where(W.z < -1, 1.0, 0.0))
        tick = TICK[ccy] / W.F
        for k in ("pos", "pos_z"):
            ch = (W[k] - W[k].shift(1).fillna(0)).abs()
            W["net_" + k] = W[k] * W.ret - ch * tick
        W.to_csv(os.path.join(A.out, f"o3_{ccy}.csv"))
        series[ccy] = W
        res[ccy] = {"momentum": opt_per(W.net_pos), "contrarian_z": opt_per(W.net_pos_z), "weeks_with_rr": int(W.rr.notna().sum())}
    if series:
        pool = pd.concat([v.net_pos for v in series.values()], axis=1).mean(axis=1)
        res["pooled"] = opt_per(pool)
        RES.setdefault("_primary", {})["O3"] = res["pooled"]
    RES["O3"] = res
    print("O3", res.get("pooled", {}).get("check", {}).get("mean_bp"), flush=True)


def spot_pair(ccy):
    return "EURUSD" if ccy == "EUR" else "USDJPY"


def o4():
    import flows_lab as FL
    res, allrows = {}, []
    for ccy in ("EUR", "JPY"):
        L = load_opts(ccy)
        if L is None:
            res[ccy] = {"status": "data missing"}
            continue
        S, O, C = L
        bk = book(spot_pair(ccy))
        exps = sorted(e for e in O.exp.unique() if pd.Timestamp("2010-07-01") <= e <= DEV_END)
        rows = {}
        for E in exps:
            E = pd.Timestamp(E)
            prev = O[(O.exp == E) & (O.date < E)]
            if prev.empty:
                continue
            dp = prev.date.max()
            if (E - dp).days > 4:
                continue
            oi = prev[prev.date == dp]
            u = oi.underlying.value_counts().index[0]
            if dp not in C.index or not np.isfinite(C.at[dp, u]):
                continue
            Fp = float(C.at[dp, u])
            ib = bk.at(FL.local_ts(dp, "America/Chicago", 16, 0), 30)
            i7 = bk.at(FL.local_ts(E, "America/New_York", 7, 0), 5)
            j = bk.at(FL.local_ts(E, "America/New_York", 9, 55), 5)
            if min(ib, i7) < 0 or j <= i7:
                continue
            sb, s7 = mid(bk, ib), mid(bk, i7)
            g = oi.groupby("K").OI.sum()
            if ccy == "EUR":
                ks = g.index.values * (sb / Fp)
            else:
                ks = 1.0 / (g.index.values * ((1.0 / sb) / Fp))
            dist = ks / s7 - 1
            ok = np.abs(dist) <= 0.005
            if not ok.any():
                continue
            k = int(np.argmax(np.where(ok, g.values, -1)))
            if abs(dist[k]) < 0.001:
                continue
            side = 1 if dist[k] > 0 else -1
            r = bk.trade_ij(i7, j, side)
            if r is None:
                continue
            rows[E] = {"net": net_of(r), "gross": float(r[1]), "dist_bp": dist[k] * 1e4, "oi": float(g.values[k])}
        D = pd.DataFrame.from_dict(rows, orient="index")
        D.to_csv(os.path.join(A.out, f"o4_{ccy}.csv"))
        res[ccy] = {c: opt_per(D[c]) for c in ("net", "gross")} if len(D) else {"n": 0}
        if len(D):
            allrows.append(D.net)
    if allrows:
        pool = pd.concat(allrows).sort_index()
        res["pooled"] = opt_per(pool)
        RES.setdefault("_primary", {})["O4"] = res["pooled"]
    RES["O4"] = res
    print("O4", res.get("pooled", {}).get("check", {}).get("mean_bp"), res.get("pooled", {}).get("all", {}).get("n"), flush=True)


def o5():
    import databento_lab as DL
    import flows_lab as FL
    from scipy.stats import norm
    res, allrows = {}, []
    for ccy in ("EUR", "JPY"):
        L = load_opts(ccy)
        if L is None:
            res[ccy] = {"status": "data missing"}
            continue
        S, O, C = L
        bk = book(spot_pair(ccy))
        days = sorted(d for d in O.date.unique() if pd.Timestamp("2010-07-01") <= d <= DEV_END)
        gam = {}
        Sg = S.groupby("date")
        Og = O.groupby("date")
        for d in days:
            if d not in Sg.groups:
                continue
            oi = Og.get_group(d)
            st_d = Sg.get_group(d)
            tot = 0.0
            for E, grp in oi.groupby("exp"):
                T = (pd.Timestamp(E) - pd.Timestamp(d)).days / 365.0
                if not (1 / 365 <= T <= 60 / 365):
                    continue
                sd = st_d[st_d.exp == E]
                if sd.empty:
                    continue
                F = float(sd.F.iloc[0])
                Ks = sd.K.values
                Ka = Ks[np.argmin(np.abs(Ks - F))]
                ivs = [DL.implied(F, Ka, T, float(sd[(sd.cp == cp) & (sd.K == Ka)].P.iloc[0]), cp)
                       for cp in ("C", "P") if len(sd[(sd.cp == cp) & (sd.K == Ka)])]
                iv = np.nanmean(ivs) if ivs else np.nan
                if not np.isfinite(iv):
                    continue
                K = grp.K.values
                d1 = (np.log(F / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
                tot += float(np.sum(grp.OI.values * norm.pdf(d1) / (F * iv * math.sqrt(T)) * F * F * 0.01))
            if tot > 0:
                gam[pd.Timestamp(d)] = tot
        Gm = pd.Series(gam).sort_index()
        hi = Gm.rolling(250, min_periods=120).quantile(2 / 3).shift(1)
        lo = Gm.rolling(250, min_periods=120).quantile(1 / 3).shift(1)
        rows = {}
        for d_prev, nxt in zip(Gm.index[:-1], Gm.index[1:]):
            if not np.isfinite(hi.get(d_prev, np.nan)):
                continue
            g = Gm[d_prev]
            mode = "fade" if g > hi[d_prev] else ("follow" if g < lo[d_prev] else None)
            if mode is None:
                continue
            i8 = bk.at(FL.local_ts(nxt, "America/New_York", 8, 0), 5)
            i12 = bk.at(FL.local_ts(nxt, "America/New_York", 12, 0), 5)
            j = bk.at(FL.local_ts(nxt, "America/New_York", 15, 55), 5)
            if min(i8, i12) < 0 or j <= i12:
                continue
            r1 = mid(bk, i12) / mid(bk, i8) - 1
            if r1 == 0:
                continue
            side = (1 if r1 > 0 else -1) * (-1 if mode == "fade" else 1)
            r = bk.trade_ij(i12, j, side)
            if r is not None:
                rows[nxt] = {"net": net_of(r), "gross": float(r[1]), "mode": mode}
        D = pd.DataFrame.from_dict(rows, orient="index")
        D.to_csv(os.path.join(A.out, f"o5_{ccy}.csv"))
        res[ccy] = {"all": opt_per(D.net) if len(D) else {}, "fade": opt_per(D[D["mode"] == "fade"].net) if len(D) else {},
                    "follow": opt_per(D[D["mode"] == "follow"].net) if len(D) else {}, "gamma_days": int(len(Gm))}
        if len(D):
            allrows.append(D.net)
    if allrows:
        pool = pd.concat(allrows).sort_index()
        res["pooled"] = opt_per(pool)
        RES.setdefault("_primary", {})["O5"] = res["pooled"]
    RES["O5"] = res
    print("O5", res.get("pooled", {}).get("check", {}).get("mean_bp"), flush=True)


def at6():
    L = load_opts("JPY")
    if L is None:
        RES["AT6"] = {"status": "data missing"}
        return
    S, O, C = L
    exps = sorted(e for e in S.exp.unique() if pd.Timestamp("2010-07-01") <= e <= DEV_END)
    rows = []
    for k in range(1, len(exps)):
        E, t0 = pd.Timestamp(exps[k]), pd.Timestamp(exps[k - 1])
        if not (10 <= (E - t0).days <= 45):
            continue
        day0 = S[(S.date >= t0) & (S.date <= t0 + pd.Timedelta(days=4)) & (S.exp == E) & (S.cp == "C")]
        if day0.empty:
            continue
        t0 = day0.date.min()
        day0 = day0[day0.date == t0]
        u = day0.underlying.value_counts().index[0]
        day0 = day0[day0.underlying == u]
        F0 = float(day0.F.iloc[0])
        K = min(day0.K, key=lambda x: abs(x - 1.04 * F0))
        c0 = float(day0[day0.K == K].P.iloc[0])
        path = C[u][(C.index >= t0) & (C.index <= E)].dropna()
        if len(path) < 5:
            continue
        FE = float(path.iloc[-1])
        carry = -(FE / F0 - 1) - TICK["JPY"] / F0
        hedge = (max(FE - K, 0.0) - 1.04 * c0) / F0
        rows.append({"expiry": E, "entry": t0, "F0": F0, "K": K, "unhedged": carry, "hedged": carry + hedge,
                     "premium_pct": c0 / F0})
    R = pd.DataFrame(rows).set_index("expiry")
    R.to_csv(os.path.join(A.out, "at6_cycles.csv"))
    out = {c: opt_per(R[c]) for c in ("hedged", "unhedged")}
    for c in ("hedged", "unhedged"):
        out[c + "_skew"] = float(R[c].skew())
        out[c + "_worst_bp"] = float(R[c].min() * 1e4)
    out["premium_median_bp"] = float(R.premium_pct.median() * 1e4)
    RES["AT6"] = out
    RES.setdefault("_primary", {})["AT6"] = out["hedged"]
    print("AT6", out["hedged"]["check"].get("mean_bp"), out["unhedged"]["check"].get("mean_bp"), flush=True)


# ------------------------------------------------------------------------------------------- daily (S1, AT3)
INV = {"JPY": "DEXJPUS", "CHF": "DEXSZUS", "CAD": "DEXCAUS", "SEK": "DEXSDUS", "NOK": "DEXNOUS", "DKK": "DEXDNUS",
       "DEM": "DEXGEUS"}                                                 # foreign currency per USD
DIR = {"GBP": "DEXUSUK", "AUD": "DEXUSAL", "NZD": "DEXUSNZ", "EUR": "DEXUSEU"}   # USD per foreign currency


def fred_usd_per(ccy):
    sid = INV.get(ccy) or DIR[ccy]
    f = os.path.join(A.r9data, "fred", f"{sid}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    x = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    x.index = pd.to_datetime(d.iloc[:, 0])
    x = x.dropna()
    x = x[x > 0]
    return (1.0 / x if ccy in INV else x).rename(ccy)


def monthly_ret(px):
    m = px.resample("ME").last()
    return m.pct_change()


def ls_portfolio(signal, ret, n=3, cost=2e-4, low_long=False):
    """Each month t: long the n highest signals (lowest if low_long), short the n lowest; return in month t+1 minus
    cost x turnover. signal and ret: DataFrame[month x currency]; signal at t uses data up to the end of t."""
    out, wprev = {}, pd.Series(0.0, index=ret.columns)
    for t, nx in zip(signal.index[:-1], signal.index[1:]):
        s = signal.loc[t].dropna()
        r = ret.loc[nx] if nx in ret.index else None
        if len(s) < 2 * n or r is None:
            continue
        s = s.sort_values(ascending=low_long)
        w = pd.Series(0.0, index=ret.columns)
        w[s.index[:n]] = 1.0 / n
        w[s.index[-n:]] = -1.0 / n
        rr = r.reindex(w.index)
        if rr[w != 0].isna().any():
            continue
        out[nx] = float((w * rr.fillna(0)).sum() - cost * (w - wprev).abs().sum())
        wprev = w
    return pd.Series(out)


def s1():
    ccys = ["JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK"]
    R = {c: monthly_ret(fred_usd_per(c)) for c in ccys if fred_usd_per(c) is not None}
    dem, eur = fred_usd_per("DEM"), fred_usd_per("EUR")
    if dem is not None and eur is not None:
        rd, re_ = monthly_ret(dem), monthly_ret(eur)
        R["EUR"] = pd.concat([rd[rd.index <= "1999-01-31"], re_[re_.index > "1999-01-31"]])
    if len(R) < 6:
        RES["S1"] = {"status": "data missing"}
        return
    M = pd.DataFrame(R).sort_index()
    M = M[M.index <= DEV_END]
    sig = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    for t in M.index:
        nxt_month = (t + pd.offsets.MonthEnd(1)).month                 # the signal at t is for month t+1
        past = M[(M.index.month == nxt_month) & (M.index < t) & (M.index >= t - pd.DateOffset(years=20))]
        cnt = past.notna().sum()
        sig.loc[t] = past.mean().where(cnt >= 10)
    a = ls_portfolio(sig, M)
    a = a[a.index >= "1991-01-01"]
    a.to_csv(os.path.join(A.out, "s1a_monthly.csv"))
    res = {"S1a seasonality": per(a, 12)}
    RES.setdefault("_primary", {})["S1a"] = res["S1a seasonality"]
    # S1b: realised skewness, excess returns, 7 currencies with rates
    from fxlab.data.rates import known_at
    c7 = ["EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    daily = {c: fred_usd_per(c) for c in c7}
    if all(v is not None for v in daily.values()):
        Dd = pd.DataFrame(daily).sort_index()
        Dd = Dd[(Dd.index >= "2005-01-01") & (Dd.index <= DEV_END)]
        lr = np.log(Dd).diff()
        ex = excess_monthly(Dd, c7, known_at)
        sk = pd.DataFrame({t: lr[(lr.index > t - pd.DateOffset(months=12)) & (lr.index <= t)].skew()
                           for t in ex.index}).T
        b = ls_portfolio(sk, ex, low_long=True)
        b = b[b.index >= "2007-01-01"]
        b.to_csv(os.path.join(A.out, "s1b_monthly.csv"))
        res["S1b skewness"] = per(b, 12)
        RES.setdefault("_primary", {})["S1b"] = res["S1b skewness"]
    RES["S1"] = res
    print("S1", {k: v["check"].get("mean_bp") for k, v in res.items()}, flush=True)


def excess_monthly(Dd, ccys, known_at):
    me = Dd.resample("ME").last()
    spot = me.pct_change()
    idx = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1, tz="UTC") for t in me.index])
    k = known_at(idx)                                                    # rates known at the start of month t
    k.index = me.index
    carry = pd.DataFrame({c: (k[c] - k["USD"]) / 12.0 for c in ccys})
    return (spot + carry)[ccys].dropna(how="all")


def at3():
    from scipy.optimize import minimize
    from fxlab.data.rates import known_at
    c7 = ["EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    daily = {c: fred_usd_per(c) for c in c7}
    if any(v is None for v in daily.values()):
        RES["AT3"] = {"status": "data missing"}
        return
    Dd = pd.DataFrame(daily).sort_index()
    Dd = Dd[(Dd.index >= "2005-12-01") & (Dd.index <= DEV_END)]
    ex = excess_monthly(Dd, c7, known_at)
    me = ex.index
    nxt_idx = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1, tz="UTC") + pd.DateOffset(months=1) for t in me])
    kn = known_at(nxt_idx)                                               # rates known at the start of month t+1
    kn.index = me
    mu_all = pd.DataFrame({c: (kn[c] - kn["USD"]) / 12.0 for c in c7})
    kappa, cost = 3e-4, 3e-4
    w_b_prev = np.zeros(7)
    w_p_prev = np.zeros(7)
    rows = {}
    for k in range(36, len(me) - 1):
        t, nx = me[k], me[k + 1]
        hist = ex.iloc[k - 35:k + 1].dropna()
        if len(hist) < 30:
            continue
        Sg = hist.cov().values
        Sg = 0.5 * Sg + 0.5 * np.diag(np.diag(Sg))
        mu = mu_all.loc[t].values
        if not np.isfinite(mu).all():
            continue
        raw = np.linalg.solve(Sg, mu)
        sc = 0.10 / math.sqrt(12 * raw @ Sg @ raw)
        wb = sc * raw
        gam = 1.0 / sc

        def f(z):
            u, v = z[:7], z[7:]
            w = w_p_prev + u - v
            val = -(mu @ w - 0.5 * gam * w @ Sg @ w - kappa * (u + v).sum())
            g_w = -(mu - gam * Sg @ w)
            return val, np.r_[g_w + kappa, -g_w + kappa]
        z0 = np.r_[np.clip(wb - w_p_prev, 0, None), np.clip(w_p_prev - wb, 0, None)]
        sol = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * 14)
        wp = w_p_prev + sol.x[:7] - sol.x[7:]
        r = ex.loc[nx].values
        if not np.isfinite(r).all():
            continue
        rk = pd.Series(mu, index=c7).sort_values()
        wr = pd.Series(0.0, index=c7)
        wr[rk.index[-3:]] = 1 / 3
        wr[rk.index[:3]] = -1 / 3
        rows[nx] = {"penalised": wp @ r - cost * np.abs(wp - w_p_prev).sum(),
                    "blind": wb @ r - cost * np.abs(wb - w_b_prev).sum(),
                    "rank": float(wr.values @ r), "turn_pen": np.abs(wp - w_p_prev).sum(),
                    "turn_blind": np.abs(wb - w_b_prev).sum(), "gross_pen": wp @ r}
        w_p_prev, w_b_prev = wp, wb
    D = pd.DataFrame.from_dict(rows, orient="index")
    D.to_csv(os.path.join(A.out, "at3_monthly.csv"))
    out = {c: per(D[c], 12) for c in ("penalised", "blind", "rank", "gross_pen")}
    out["turnover_median"] = {"penalised": float(D.turn_pen.median()), "blind": float(D.turn_blind.median())}
    RES["AT3"] = out
    RES.setdefault("_primary", {})["AT3"] = out["penalised"]
    print("AT3", {c: out[c]["check"].get("sharpe") for c in ("penalised", "blind", "rank")}, flush=True)


# ------------------------------------------------------------------------------------------- combine
def combine(files):
    from scipy.stats import norm
    rows = []
    for f in files:
        J = json.load(open(f))["results"]
        for name, P in J.get("_primary", {}).items():
            c, ft = P.get("check", {}), P.get("fit", {})
            t = c.get("t")
            rows.append({"test": name, "n_check": c.get("n"), "mean_check_bp": c.get("mean_bp"), "t_check": t,
                         "mean_fit_bp": ft.get("mean_bp"), "n_fit": ft.get("n"),
                         "p_one_sided": float(1 - norm.cdf(t)) if t is not None and np.isfinite(t) else 1.0})
        for name, P in J.get("_primary_g1", {}).items():
            c, ft = P.get("check", {}), P.get("fit", {})
            RES.setdefault("G1_separate", {})[name] = {"check": c, "fit": ft}
    T = pd.DataFrame(rows).drop_duplicates("test", keep="last").sort_values("p_one_sided").reset_index(drop=True)
    m = len(T)
    T["bh_threshold"] = [0.10 * (k + 1) / m for k in range(m)]
    passed = T.p_one_sided <= T.bh_threshold
    kmax = passed[passed].index.max() if passed.any() else -1
    T["bh_pass"] = T.index <= kmax
    T["counts"] = T.bh_pass & (T.mean_check_bp > 0) & (T.t_check >= 1.645) & (T.mean_fit_bp.fillna(1) > 0)
    T.to_csv(os.path.join(A.out, "round9_combined.csv"), index=False)
    print(T.to_string())


# ------------------------------------------------------------------------------------------- S2
def s2():
    f = os.path.join(A.r9data, "sge", "Au99.99.csv")
    if not os.path.exists(f):
        RES["S2"] = {"status": "not tested (data): no SGE history"}
        return
    S = pd.read_csv(f)
    dcol = [c for c in S.columns if "date" in c.lower()][0]
    ccol = [c for c in S.columns if c.lower() == "close"][0]
    sge = pd.Series(pd.to_numeric(S[ccol], errors="coerce").values, index=pd.to_datetime(S[dcol])).dropna()
    fx = os.path.join(A.r9data, "fred", "DEXCHUS.csv")
    d = pd.read_csv(fx)
    usdcny = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0])).dropna()
    bk = book("XAUUSD")
    rows = {}
    for day, px in sge.items():
        if day > DEV_END or day not in usdcny.index:
            continue
        i = bk.at(pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=7, minutes=30), 15)
        if i < 0:
            continue
        rows[pd.Timestamp(day)] = px * 31.1035 / usdcny[day] / mid(bk, i) - 1
    P = pd.Series(rows).sort_index()
    sig = P - P.rolling(20).mean()
    out = {}
    for day, v in sig.dropna().items():
        t0 = pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=21)
        nxt = day + pd.offsets.BDay(1)
        t1 = pd.Timestamp(nxt).tz_localize("UTC") + pd.Timedelta(hours=21)
        r = bk.trade(t0, t1, 1 if v > 0 else -1, 10)
        if r is not None:
            out[pd.Timestamp(day)] = net_of(r)
    X = pd.Series(out)
    X.to_csv(os.path.join(A.out, "s2_daily.csv"))
    n = len(X)
    split = X.index[int(0.6 * n)] if n else FIT_END
    RES["S2"] = {"net": per(X, 252, split), "premium_median_pct": float(P.median() * 100), "first_day": str(P.index.min())[:10],
                 "split_day": str(split)[:10]}
    RES.setdefault("_primary", {})["S2"] = RES["S2"]["net"]
    print("S2", RES["S2"]["net"]["check"].get("mean_bp"), RES["S2"]["premium_median_pct"], flush=True)


# ------------------------------------------------------------------------------------------- AT5
def gdelt_month(name):
    f = os.path.join(A.r9data, "gdelt", f"{name}.csv")
    if not os.path.exists(f) or os.path.getsize(f) < 200:
        return None
    d = pd.read_csv(f)
    dc = [c for c in d.columns if "date" in c.lower()][0]
    vc = [c for c in d.columns if "value" in c.lower()][0]
    x = pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d[dc], errors="coerce")).dropna()
    x.index = nv(x.index)
    return x.resample("ME").mean()


def at5():
    from sklearn.linear_model import LassoCV
    from fxlab.data.rates import known_at
    tone = {k: gdelt_month(k) for k in ("US", "DE", "FR", "JP", "UK", "CH", "CA", "AU", "NZ")}
    LOG["at5_tone_months"] = {k: (int(v.notna().sum()) if v is not None else 0) for k, v in tone.items()}
    if any(v is None or v.notna().sum() < 60 for v in tone.values()):
        RES["AT5"] = {"status": "not tested (data): GDELT tone incomplete", "months": LOG["at5_tone_months"]}
        return
    T = pd.DataFrame(tone)
    T["EZ"] = T[["DE", "FR"]].mean(axis=1)
    T = T.drop(columns=["DE", "FR"])
    feats = pd.concat([T.add_prefix("lvl_"), T.diff().add_prefix("chg_")], axis=1)
    c7 = ["EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
    daily = {c: fred_usd_per(c) for c in c7}
    Dd = pd.DataFrame(daily).sort_index()
    Dd = Dd[(Dd.index >= "2016-06-01") & (Dd.index <= DEV_END)]
    ex = excess_monthly(Dd, c7, known_at)
    nxt_idx = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1, tz="UTC") + pd.DateOffset(months=1) for t in ex.index])
    kn = known_at(nxt_idx)
    kn.index = ex.index
    fd = pd.DataFrame({c: (kn[c] - kn["USD"]) for c in c7})
    months = [m for m in feats.dropna().index if m in ex.index]
    fc = pd.DataFrame(index=months, columns=c7, dtype=float)
    for k in range(36, len(months) - 1):
        t = months[k]
        for c in c7:
            X = pd.concat([feats.loc[months[1:k + 1]], fd[c].reindex(months[1:k + 1]).rename("fd")], axis=1)
            y = ex[c].shift(-1).reindex(months[1:k + 1])
            ok = X.notna().all(axis=1) & y.notna()
            X, y = X[ok], y[ok]
            if len(y) < 30:
                continue
            mu, sd = X.mean(), X.std().replace(0, 1)
            mdl = LassoCV(cv=5, n_alphas=30, max_iter=5000).fit(((X - mu) / sd).values, y.values)
            xt = pd.concat([feats.loc[[t]], pd.Series({t: fd[c].get(t)}, name="fd")], axis=1)
            if xt.isna().any(axis=None):
                continue
            fc.loc[t, c] = float(mdl.predict(((xt - mu) / sd).values)[0])
    port = ls_portfolio(fc.dropna(how="all"), ex)
    port.to_csv(os.path.join(A.out, "at5_monthly.csv"))
    out = {"net": {"all": Q.summ(port, 12), "check": Q.summ(port, 12), "fit": {}}, "months_with_forecast": int(fc.notna().any(axis=1).sum())}
    RES["AT5"] = out
    RES.setdefault("_primary", {})["AT5"] = out["net"]
    print("AT5", out["net"]["check"].get("mean_bp"), out["net"]["check"].get("t"), flush=True)


def save():
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(clean({"results": RES, "log": LOG}), open(os.path.join(A.out, f"round9_{A.tag}.json"), "w"), indent=1)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", action="store_true")
    ap.add_argument("--opt", action="store_true")
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--combine", nargs="*")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r9data", default="r9data")
    ap.add_argument("--dbn", default="dbn")
    ap.add_argument("--compact", default="compact")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    A.tag = "m1" if A.m1 else "opt" if A.opt else "daily" if A.daily else "combine"
    parts = []
    if A.m1:
        parts = [g1, at1, at2, at4, s5] + [globals()[f] for f in ("s2", "n_tests", "s3", "s4") if f in globals()]
    if A.opt:
        parts = [globals()[f] for f in ("o3", "o4", "o5", "at6") if f in globals()]
    if A.daily:
        parts = [globals()[f] for f in ("s1", "at3", "at5") if f in globals()]
    only = set(A.only.split(",")) - {""}
    for f in parts:
        if only and f.__name__ not in only:
            continue
        try:
            f()
        except Exception as ex:                                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            RES[f.__name__ + "_error"] = repr(ex)[:400]
        save()
    if A.combine is not None:
        combine(A.combine)


A = None

if __name__ == "__main__":
    main()
