"""Flow / volume / positioning batch V1-V9 (docs/flow_volume_preregistration.md). 2007-2025 only.
Reads the flow-collect store (STORE) and the M1 bid/ask files; every test is traded with flows_lab Book fills.
Usage: python scripts/flow_volume_lab.py --store STORE --out DIR"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edges_lab2 as E  # noqa: E402
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

END = pd.Timestamp("2025-12-31")
FUT_MAP = {"EUR": ("EURUSD", 1), "JPY": ("USDJPY", -1), "GBP": ("GBPUSD", 1), "AUD": ("AUDUSD", 1),
           "NZD": ("NZDUSD", 1), "CAD": ("USDCAD", -1), "CHF": ("USDCHF", -1), "MXN": ("USDMXN", -1),
           "GOLD": ("XAUUSD", 1), "SILVER": ("XAGUSD", 1)}
CCY_PAIR = {k: v for k, v in FUT_MAP.items() if k not in ("GOLD", "SILVER", "MXN")}
ROWS, LOG = [], {}
BOOKS = {}


def book(sym):
    if sym not in BOOKS:
        if len(BOOKS) > 1:
            BOOKS.clear()
        BOOKS[sym] = E.book(sym)
        E.BOOKS.clear()
    return BOOKS[sym]


def ny(d, hh=17, mm=5):
    return FL.local_ts(pd.Timestamp(d), "America/New_York", hh, mm)


def bday_after(d, n=1):
    return pd.Timestamp(d) + pd.offsets.BDay(n)


PASS = {"mode": "direct", "want": set(), "done": {}}


def tr(sym, t_in, t_out, side):
    """(net, gross in the trade's direction) or None.
    Two passes so each market's minute book is loaded once: 'record' notes the trade and returns a placeholder,
    'replay' looks up the trade priced in between (signals never depend on trade results)."""
    key = (sym, pd.Timestamp(t_in), pd.Timestamp(t_out), int(side))
    if PASS["mode"] == "record":
        PASS["want"].add(key)
        return 0.0, 0.0
    if PASS["mode"] == "replay":
        return PASS["done"].get(key)
    return tr_direct(*key)


def price_all():
    by = {}
    for k in PASS["want"]:
        by.setdefault(k[0], []).append(k)
    for sym, keys in sorted(by.items()):
        for k in sorted(keys, key=lambda x: x[1]):
            PASS["done"][k] = tr_direct(*k)
        BOOKS.clear()
        E.BOOKS.clear()
        print("priced", sym, len(keys), flush=True)


def tr_direct(sym, t_in, t_out, side):
    bk = book(sym)
    if bk is None:
        return None
    r = bk.trade(t_in, t_out, int(side), tol_min=30)
    if r is None:
        return None
    return U.net_of(r), float(r[1])


def daily_trade(sym, d, side, n):
    t_in = ny(bday_after(d, 1))
    t_out = ny(bday_after(d, 1 + n))
    return tr(sym, t_in, t_out, side)


def weekly_trade(sym, monday, side):
    t_in = pd.Timestamp(monday).tz_localize("UTC") + pd.Timedelta(hours=14)
    return tr(sym, t_in, t_in + pd.Timedelta(days=4), side)


def month_t(x: pd.Series):
    x = x.dropna()
    if len(x) < 10:
        return np.nan
    e = x - x.mean()
    g = e.groupby(pd.DatetimeIndex(x.index).to_period("M")).sum()
    m = len(g)
    se = math.sqrt((g ** 2).sum()) / len(x) * math.sqrt(m / max(m - 1, 1))
    return float(x.mean() / se) if se > 0 else np.nan


def evaluate(test, ev, info=False, note=""):
    """ev: DataFrame index=date, columns follow_net, follow_gross (gross in the follow direction), fade_net."""
    if ev is None or not len(ev):
        ROWS.append({"test": test, "status": "no events / data missing", "note": note})
        return
    ev = ev.sort_index()
    ev = ev[ev.index <= END]
    if ev.index.min() > pd.Timestamp("2008-12-31"):
        k = int(0.6 * len(ev))
        fit = np.arange(len(ev)) < k
        split = f"60/40 by events (check from {ev.index[k].date()})"
    else:
        fit = ev.index <= pd.Timestamp("2018-12-31")
        split = "2007-2018 / 2019-2025"
    g = ev.follow_gross[fit].mean()
    d = "follow" if g > 0 else "fade"
    chk = ev[f"{d}_net"][~fit]
    t = month_t(chk)
    ROWS.append({"test": test, "split": split, "fit_n": int(fit.sum()), "fit_gross_follow_bp": float(g * 1e4),
                 "direction": d, "check_n": int(len(chk)), "check_net_bp": float(chk.mean() * 1e4) if len(chk) else np.nan,
                 "check_t": t, "check_win": float((chk > 0).mean()) if len(chk) else np.nan,
                 "fit_net_dir_bp": float(ev[f"{d}_net"][fit].mean() * 1e4), "info_only": info, "note": note})


def frame(rows):
    if not rows:
        return None
    f = pd.DataFrame(rows).set_index("date")
    return f.groupby(level=0).mean()


def ev_row(d, follow, fade):
    if follow is None or fade is None:
        return None
    return {"date": pd.Timestamp(d), "follow_net": follow[0], "follow_gross": follow[1], "fade_net": fade[0]}


# --------------------------------------------------------------------------------------------- loaders
def find(store, *pat):
    return sorted(glob.glob(os.path.join(store, "**", *pat), recursive=True))


def yahoo_csv(path):
    d = pd.read_csv(path)
    dc = d.columns[0]
    d = d[pd.to_datetime(d[dc], errors="coerce").notna()]
    d.index = pd.to_datetime(d[dc]).dt.tz_localize(None).dt.normalize()
    for c in ("Close", "Volume", "Open", "High", "Low"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def futures(store):
    out = {}
    for name in FUT_MAP:
        f = find(store, "futures", f"{name}.csv")
        if f:
            out[name] = yahoo_csv(f[0])
    return out


# --------------------------------------------------------------------------------------------- tests
def v1(store):
    F = futures(store)
    LOG["futures"] = {k: [str(v.index.min().date()), str(v.index.max().date()), int(len(v))] for k, v in F.items()}
    Z = {}
    for name, d in F.items():
        v = np.log(d.Volume.where(d.Volume > 0))
        z = (v - v.rolling(60, min_periods=40).mean().shift(1)) / v.rolling(60, min_periods=40).std().shift(1)
        r = np.log(d.Close).diff()
        Z[name] = pd.DataFrame({"z": z, "r": r})
        sym, sg = FUT_MAP[name]
        for n in (1, 5):
            rows = []
            for dd, x in Z[name][(Z[name].z >= 2) & Z[name].r.notna()].iterrows():
                if dd > END or dd < pd.Timestamp("2007-01-01"):
                    continue
                side = int(np.sign(x.r) * sg)
                if side == 0:
                    continue
                rows.append(ev_row(dd, daily_trade(sym, dd, side, n), daily_trade(sym, dd, -side, n)))
            evaluate(f"V1a {name} volume z >= 2, {n} day(s)", frame([r for r in rows if r]))
    # V1b cross-sectional low-volume reversal, currencies only
    cc = [k for k in Z if k in CCY_PAIR]
    if len(cc) >= 5:
        P = pd.concat({k: Z[k].z for k in cc}, axis=1).dropna(thresh=5)
        R = pd.concat({k: Z[k].r for k in cc}, axis=1)
        rows = []
        for dd, zr in P.iterrows():
            if dd > END or dd < pd.Timestamp("2007-01-01"):
                continue
            low = zr.dropna().nsmallest(3).index
            fo, fa = [], []
            for k in low:
                rr = R.at[dd, k]
                if not np.isfinite(rr) or rr == 0:
                    continue
                sym, sg = CCY_PAIR[k]
                side = int(np.sign(rr) * sg)            # follow = the day's direction
                a, b = daily_trade(sym, dd, side, 1), daily_trade(sym, dd, -side, 1)
                if a and b:
                    fo.append(a)
                    fa.append(b)
            if fo:
                rows.append({"date": dd, "follow_net": np.mean([x[0] for x in fo]), "follow_gross": np.mean([x[1] for x in fo]),
                             "fade_net": np.mean([x[0] for x in fa])})
        evaluate("V1b low-volume reversal, 3 lowest-volume currency futures, 1 day", frame(rows),
                 note="pre-registered as fade; direction still taken from the fit period")


def gld_tonnes(store):
    f = find(store, "etf", "GLD_holdings_archive.csv")
    if not f:
        return None
    raw = open(f[0], "rb").read().decode("utf-8", "ignore")
    lines = raw.splitlines()
    hi = next((i for i, l in enumerate(lines) if l.lower().startswith("date")), None)
    if hi is None:
        return None
    d = pd.read_csv(io.StringIO("\n".join(lines[hi:])), on_bad_lines="skip")
    d.columns = [c.strip() for c in d.columns]
    tc = next((c for c in d.columns if "tonne" in c.lower()), None)
    if tc is None:
        return None
    s = pd.Series(pd.to_numeric(d[tc].astype(str).str.replace(",", ""), errors="coerce").values,
                  index=pd.to_datetime(d[d.columns[0]], errors="coerce", dayfirst=True)).dropna()
    s = s[~s.index.duplicated()].sort_index()
    return s


def v2_v9(store, mm_gold):
    s = gld_tonnes(store)
    if s is None or len(s) < 500:
        evaluate("V2 GLD holdings change", None, note="GLD archive not parsed")
        return None
    LOG["gld"] = [str(s.index.min().date()), str(s.index.max().date()), int(len(s))]
    ch = s.pct_change()
    z = (ch - ch.rolling(60, min_periods=40).mean().shift(1)) / ch.rolling(60, min_periods=40).std().shift(1)
    for n in (1, 5):
        rows = []
        for dd, zz in z[z.abs() >= 2].items():
            if dd > END:
                continue
            side = int(np.sign(zz))
            rows.append(ev_row(dd, daily_trade("XAUUSD", dd, side, n), daily_trade("XAUUSD", dd, -side, n)))
        evaluate(f"V2 GLD holdings change abs(z) >= 2, {n} day(s)", frame([r for r in rows if r]))
    # V9: weekly combination with managed money
    if mm_gold is not None:
        w5 = s.pct_change(5)
        wz = (w5 - w5.rolling(250, min_periods=100).mean().shift(5)) / w5.rolling(250, min_periods=100).std().shift(5)
        rows = []
        for rep_date, mz in mm_gold.items():
            mon = rep_date + pd.Timedelta(days=6)
            g = wz[wz.index <= rep_date + pd.Timedelta(days=3)]
            if not len(g) or not np.isfinite(g.iloc[-1]) or not np.isfinite(mz):
                continue
            if abs(mz) >= 1 and abs(g.iloc[-1]) >= 1 and np.sign(mz) == np.sign(g.iloc[-1]) and mon <= END:
                side = int(np.sign(mz))
                rows.append(ev_row(mon, weekly_trade("XAUUSD", mon, side), weekly_trade("XAUUSD", mon, -side)))
        evaluate("V9 GLD 5-day flow and managed-money change agree (abs z >= 1), gold next week", frame([r for r in rows if r]))
    return s


def cftc_frame(store, sub):
    fs = find(store, f"cftc_{sub}", "*.txt")
    D = []
    for f in fs:
        try:
            d = pd.read_csv(f, low_memory=False)
            d.columns = [c.strip() for c in d.columns]
            D.append(d)
        except Exception:                                                   # noqa: BLE001
            pass
    if not D:
        return None
    T = pd.concat(D, ignore_index=True)
    dc = next(c for c in T.columns if c.startswith("Report_Date_as_YYYY"))
    T["date"] = pd.to_datetime(T[dc], errors="coerce")
    T["code"] = T["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
    return T.dropna(subset=["date"])


TFF_CODES = {"JPY": "097741", "EUR": "099741", "GBP": "096742", "AUD": "232741", "CAD": "090741", "CHF": "092741",
             "NZD": "112741"}
DIS_CODES = {"GOLD": "088691", "SILVER": "084691"}


def change_z(g, lcol, scol):
    g = g.drop_duplicates("date").set_index("date").sort_index()
    net = (pd.to_numeric(g[lcol], errors="coerce") - pd.to_numeric(g[scol], errors="coerce")) / \
        pd.to_numeric(g["Open_Interest_All"], errors="coerce")
    dn = net.diff()
    return (dn - dn.rolling(52, min_periods=30).mean().shift(1)) / dn.rolling(52, min_periods=30).std().shift(1)


def v3(store):
    T = cftc_frame(store, "tff")
    out_mm = None
    if T is not None:
        LOG["tff_years"] = sorted(set(T.date.dt.year.astype(int)))
        for grp, lc, sc in (("V3a leveraged funds", "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
                            ("V3b asset managers", "Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All")):
            rows = []
            for ccy, code in TFF_CODES.items():
                g = T[T.code == code]
                if len(g) < 100:
                    continue
                z = change_z(g, lc, sc)
                sym, sg = CCY_PAIR[ccy]
                for d, zz in z[z.abs() >= 1.5].items():
                    mon = d + pd.Timedelta(days=6)
                    if mon > END:
                        continue
                    side = int(np.sign(zz) * sg)
                    r = ev_row(mon, weekly_trade(sym, mon, side), weekly_trade(sym, mon, -side))
                    if r:
                        rows.append(r)
            evaluate(f"{grp}: weekly net-position change abs(z) >= 1.5, next week, 7 currencies pooled",
                     pd.DataFrame(rows).set_index("date").sort_index() if rows else None)
    D = cftc_frame(store, "disagg")
    if D is not None:
        rows = []
        for m, code in DIS_CODES.items():
            g = D[D.code == code]
            if len(g) < 100:
                continue
            z = change_z(g, "M_Money_Positions_Long_All", "M_Money_Positions_Short_All")
            if m == "GOLD":
                out_mm = z
            sym = FUT_MAP[m][0]
            for d, zz in z[z.abs() >= 1.5].items():
                mon = d + pd.Timedelta(days=6)
                if mon > END:
                    continue
                side = int(np.sign(zz))
                r = ev_row(mon, weekly_trade(sym, mon, side), weekly_trade(sym, mon, -side))
                if r:
                    rows.append(r)
        evaluate("V3c managed money (gold, silver): weekly change abs(z) >= 1.5, next week",
                 pd.DataFrame(rows).set_index("date").sort_index() if rows else None)
    return out_mm


def v4(store):
    f = find(store, "snb", "sight_deposits.csv")
    if not f:
        evaluate("V4 SNB sight deposits", None, note="file missing")
        return
    raw = open(f[0], encoding="utf-8", errors="ignore").read().splitlines()
    hi = next(i for i, l in enumerate(raw) if l.startswith('"Date"') or l.startswith("Date"))
    d = pd.read_csv(io.StringIO("\n".join(raw[hi:])), sep=";")
    d.columns = [c.strip('"') for c in d.columns]
    d = d[d["D0"].astype(str).str.strip('"') == "GI"] if "D0" in d else d
    s = pd.Series(pd.to_numeric(d["Value"], errors="coerce").values, index=pd.to_datetime(d["Date"].astype(str).str.strip('"'), errors="coerce")).dropna().sort_index()
    LOG["snb"] = [str(s.index.min().date()), str(s.index.max().date()), int(len(s))]
    ch = s.diff().dropna()
    cut = ch[(ch.index >= "2007-01-01") & (ch.index <= "2018-12-31")].abs().quantile(0.8)
    rows = []
    for d0, c in ch[ch.abs() > cut].items():
        mon = d0 + pd.offsets.Week(weekday=0)
        if mon > END or d0 < pd.Timestamp("2007-01-01"):
            continue
        side = int(np.sign(c))                       # deposits up = SNB sold CHF -> CHF weaker -> EUR/CHF, USD/CHF up
        legs = [(weekly_trade(p, mon, side), weekly_trade(p, mon, -side)) for p in ("EURCHF", "USDCHF")]
        legs = [x for x in legs if x[0] and x[1]]
        if legs:
            rows.append({"date": mon, "follow_net": np.mean([a[0] for a, b in legs]),
                         "follow_gross": np.mean([a[1] for a, b in legs]), "fade_net": np.mean([b[0] for a, b in legs])})
    evaluate("V4 SNB sight deposits weekly change > fit 80th pct, EUR/CHF + USD/CHF next week", frame(rows),
             note=f"cut {cut:.0f} CHF m")


def v5(store):
    tone = {}
    for k in ("EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"):
        S = []
        for f in find(store, "gdelt", f"tone_{k}_*.csv"):
            try:
                d = pd.read_csv(f)
                d.columns = [c.strip() for c in d.columns]
                dc = d.columns[0]
                vc = d.columns[-1]
                S.append(pd.Series(pd.to_numeric(d[vc], errors="coerce").values, index=pd.to_datetime(d[dc], errors="coerce")))
            except Exception:                                               # noqa: BLE001
                pass
        if S:
            s = pd.concat(S).dropna()
            s = s[~s.index.duplicated()].sort_index()
            tone[k] = s.resample("D").mean()
    LOG["gdelt"] = {k: [str(v.index.min().date()), str(v.index.max().date()), int(v.notna().sum())] for k, v in tone.items()}
    if len(tone) < 5:
        evaluate("V5 GDELT news-tone reversal", None, note=f"only {len(tone)} currencies")
        return
    TT = pd.DataFrame(tone).rolling(7, min_periods=4).mean()
    rows = []
    for fri in pd.date_range("2017-01-06", "2025-12-26", freq="W-FRI"):
        x = TT.loc[:fri].iloc[-1] if len(TT.loc[:fri]) else None
        if x is None or x.notna().sum() < 5:
            continue
        x = x.dropna()
        mon = fri + pd.Timedelta(days=3)
        lo, hi = x.nsmallest(2).index, x.nlargest(2).index
        fo, fa = [], []
        for k, want in [(c, +1) for c in lo] + [(c, -1) for c in hi]:   # follow = long low tone (the reversal)
            sym, sg = CCY_PAIR[k]
            a, b = weekly_trade(sym, mon, want * sg), weekly_trade(sym, mon, -want * sg)
            if a and b:
                fo.append(a)
                fa.append(b)
        if fo:
            rows.append({"date": mon, "follow_net": np.mean([a[0] for a in fo]), "follow_gross": np.mean([a[1] for a in fo]),
                         "fade_net": np.mean([b[0] for b in fa])})
    evaluate("V5 GDELT tone reversal: long 2 lowest-tone, short 2 highest-tone currencies vs USD, next week", frame(rows),
             note="'follow' = the reversal (long low tone)")


def yields(store):
    out = {}
    U2 = []
    for f in find(store, "yields", "ust_curve_*.csv"):
        try:
            d = pd.read_csv(f)
            c2 = next(c for c in d.columns if c.strip() in ("2 Yr", "2 YR"))
            U2.append(pd.Series(pd.to_numeric(d[c2], errors="coerce").values, index=pd.to_datetime(d["Date"], errors="coerce")))
        except Exception:                                                   # noqa: BLE001
            pass
    if U2:
        s = pd.concat(U2).dropna()
        out["US"] = s[~s.index.duplicated()].sort_index()
    for f in find(store, "yields", "mof_jgbcme_all.csv") + find(store, "yields", "mof_jgbcme.csv"):
        try:
            raw = open(f, encoding="utf-8", errors="ignore").read().splitlines()
            hi = next(i for i, l in enumerate(raw) if l.startswith("Date"))
            d = pd.read_csv(io.StringIO("\n".join(raw[hi:])))
            s = pd.Series(pd.to_numeric(d["2Y"], errors="coerce").values, index=pd.to_datetime(d["Date"], errors="coerce")).dropna()
            out["JP"] = pd.concat([out.get("JP", pd.Series(dtype=float)), s])
        except Exception as ex:                                             # noqa: BLE001
            LOG.setdefault("yield_errors", []).append(f"{os.path.basename(f)}: {ex!r}"[:150])
    if "JP" in out:
        out["JP"] = out["JP"][~out["JP"].index.duplicated()].sort_index()
    for f in find(store, "yields", "ecb_aaa_SR_2Y.csv"):
        try:
            d = pd.read_csv(f)
            out["EU"] = pd.Series(pd.to_numeric(d["OBS_VALUE"], errors="coerce").values,
                                  index=pd.to_datetime(d["TIME_PERIOD"], errors="coerce")).dropna().sort_index()
        except Exception as ex:                                             # noqa: BLE001
            LOG.setdefault("yield_errors", []).append(f"ecb: {ex!r}"[:150])
    LOG["yields"] = {k: [str(v.index.min().date()), str(v.index.max().date()), int(len(v))] for k, v in out.items()}
    return out


def v6(store):
    Y = yields(store)
    if "US" not in Y:
        evaluate("V6 2-year yield gap", None, note="US yields missing")
        return
    for other, sym, usd_side in (("JP", "USDJPY", 1), ("EU", "EURUSD", -1)):
        if other not in Y:
            evaluate(f"V6 US-{other} 2y gap", None, note="yields missing")
            continue
        gap = (Y["US"] - Y[other]).dropna()
        c5 = gap.diff(5)
        z = (c5 - c5.rolling(250, min_periods=150).mean().shift(1)) / c5.rolling(250, min_periods=150).std().shift(1)
        for n in (1, 5):
            rows = []
            for dd, zz in z[z.abs() >= 1.5].items():
                if dd > END or dd < pd.Timestamp("2007-01-01"):
                    continue
                side = int(np.sign(zz) * usd_side)
                rows.append(ev_row(dd, daily_trade(sym, dd, side, n), daily_trade(sym, dd, -side, n)))
            evaluate(f"V6 US-{other} 2y gap, 5-day change abs(z) >= 1.5, {sym} {n} day(s)", frame([r for r in rows if r]))


def v7(store):
    f = find(store, "rates_futures", "fedfunds_front.csv")
    if not f:
        evaluate("V7 fed funds futures", None, note="file missing")
        return
    d = yahoo_csv(f[0])
    rate = 100 - d.Close
    ch = rate.diff()
    first = ch.index.to_series().groupby(ch.index.to_period("M")).transform("min") == ch.index.to_series()
    ch = ch[~first.values]
    legs = (("EURUSD", -1), ("USDJPY", 1), ("GBPUSD", -1), ("AUDUSD", -1))
    for n in (1, 5):
        rows = []
        for dd, c in ch[ch.abs() >= 0.05].items():
            if dd > END or dd < pd.Timestamp("2007-01-01"):
                continue
            s = int(np.sign(c))
            L = [(daily_trade(p, dd, s * u, n), daily_trade(p, dd, -s * u, n)) for p, u in legs]
            L = [x for x in L if x[0] and x[1]]
            if L:
                rows.append({"date": dd, "follow_net": np.mean([a[0] for a, b in L]), "follow_gross": np.mean([a[1] for a, b in L]),
                             "fade_net": np.mean([b[0] for a, b in L])})
        evaluate(f"V7 fed funds front implied rate change >= 5 bp, USD vs 4 majors, {n} day(s)", frame(rows))


def v8(store):
    fs = find(store, "intervention", "mof_en.csv") or find(store, "intervention", "mof_ja.csv")
    if not fs:
        evaluate("V8 MoF intervention", None, info=True, note="file missing")
        return
    raw = open(fs[0], "rb").read()
    txt = None
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:                                                   # noqa: BLE001
            pass
    days = []
    for l in (txt or "").splitlines():
        m = re.match(r"\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s*,\s*([-\d,\.]+)", l)
        if m:
            amt = float(m.group(4).replace(",", "") or 0)
            days.append((pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))), amt))
    LOG["intervention_rows"] = len(days)
    for n in (5, 20):
        rows = []
        for d0, amt in days:
            if d0 < pd.Timestamp("2007-01-01") or d0 > END or amt == 0:
                continue
            side = -1 if amt > 0 else 1        # positive amount read as yen bought (USD sold) -> USD/JPY down
            rows.append(ev_row(d0, daily_trade("USDJPY", d0, side, n), daily_trade("USDJPY", d0, -side, n)))
        evaluate(f"V8 MoF intervention days, USD/JPY next {n} days in the intervention's direction", frame([r for r in rows if r]),
                 info=True, note="information only; amount sign convention checked in the log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    PASS["mode"] = "record"
    run_all(a)
    print("trades to price:", len(PASS["want"]), flush=True)
    price_all()
    ROWS.clear()
    LOG.clear()
    PASS["mode"] = "replay"
    run_all(a)
    finish(a)


def run_all(a):
    for fn in (v1, v3, v4, v5, v6, v7, v8):
        try:
            r = fn(a.store)
            if fn is v3:
                mm = r
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            ROWS.append({"test": fn.__name__, "status": f"error {ex!r}"[:300]})
        print(fn.__name__, "done", len(ROWS), flush=True)
    try:
        v2_v9(a.store, locals().get("mm"))
    except Exception as ex:                                                 # noqa: BLE001
        import traceback
        traceback.print_exc()
        ROWS.append({"test": "v2/v9", "status": f"error {ex!r}"[:300]})


def finish(a):
    R = pd.DataFrame(ROWS)
    from scipy import stats as st
    if "check_t" in R:
        m = R.check_t.notna() & ~R.get("info_only", pd.Series(False, index=R.index)).fillna(False).astype(bool)
        p = pd.Series(1.0, index=R.index)
        p[m] = st.norm.sf(R.check_t[m])
        q = p.copy()
        idx = p[m].sort_values().index
        k = len(idx)
        if k:
            ranked = p[idx].values * k / np.arange(1, k + 1)
            q[idx] = np.minimum.accumulate(ranked[::-1])[::-1]
        R["bh_q"] = q.where(m)
        R["counts"] = m & (R.check_net_bp > 0) & (R.check_t >= 1.645) & (R.bh_q <= 0.10)
    R.to_csv(os.path.join(a.out, "flow_volume.csv"), index=False)
    json.dump(LOG, open(os.path.join(a.out, "log.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = [c for c in ("test", "fit_n", "direction", "check_n", "check_net_bp", "check_t", "bh_q", "counts", "status") if c in R]
    print(R[cols].round(3).to_string())
    print(json.dumps(LOG, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
