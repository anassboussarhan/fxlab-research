"""Round 24 stock parts (docs/round24_preregistration.md): R4 earnings-announcement drift from SEC 8-K dates, R6 FINRA
daily short-sale volume, R11a ETF pairs trading, R11b S&P 500 additions. Prices: Yahoo daily (yfinance); universe
and changes: Wikipedia; earnings dates: data.sec.gov submissions; short volume: FINRA Reg SHO files.
Usage: python scripts/r24_stocks.py {data,ear,shortvol,pairs,adds} --out DIR [--data DIR]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24_common as C  # noqa: E402

ETF_SECT = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ETF_CTRY = ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO", "EWP", "EWQ", "EWS",
            "EWU", "EWW"]


def yf_prices(tickers, start="2007-01-01"):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        for k in range(4):
            try:
                h = yf.download(chunk, start=start, end="2026-01-01", auto_adjust=False, progress=False, threads=True,
                                group_by="ticker")
                break
            except Exception as ex:                                        # noqa: BLE001
                print("yf retry", k, repr(ex)[:100], flush=True)
                time.sleep(20 * (k + 1))
        else:
            continue
        for t in chunk:
            try:
                d = h[t] if isinstance(h.columns, pd.MultiIndex) else h
                d = d.dropna(subset=["Close"])
                if len(d) < 250:
                    continue
                f = d["Adj Close"] / d["Close"]
                out[t] = pd.DataFrame({"open": d["Open"] * f, "close": d["Adj Close"], "dv": d["Close"] * d["Volume"]})
            except Exception:                                              # noqa: BLE001
                continue
        time.sleep(2)
    return out


def cmd_data(a):
    os.makedirs(a.out, exist_ok=True)
    def flat(cols):
        if isinstance(cols, pd.MultiIndex):
            return ["_".join(str(x).strip() for x in c if "Unnamed" not in str(x)).strip("_") for c in cols]
        return [str(c).strip() for c in cols]

    page = "List_of_S%26P_500_companies"
    srcs = [("page", f"https://en.wikipedia.org/wiki/{page}"),               # the same article, three ways of fetching
            ("rest", f"https://en.wikipedia.org/api/rest_v1/page/html/{page}"),
            ("parse", f"https://en.wikipedia.org/w/api.php?action=parse&page={page}&prop=text&format=json"
                      "&formatversion=2&redirects=1"),
            ("wikiwand", f"https://www.wikiwand.com/en/articles/{page}")]  # a mirror of the same article
    is_cur = is_chg = []
    for name, url in srcs:
        try:
            r = C.get(url, tries=3)
            html = r.json()["parse"]["text"] if name == "parse" else r.text
        except Exception as ex:                                            # noqa: BLE001
            print("wikipedia", name, "failed", repr(ex)[:120], flush=True)
            continue
        print("wikipedia", name, "bytes", len(html), "<table", html.count("<table"), flush=True)
        for flavor in ("lxml", "bs4"):                                     # html5lib when lxml drops a table
            try:
                T = pd.read_html(io.StringIO(html), flavor=flavor)
            except Exception as ex:                                        # noqa: BLE001
                print("  read_html", flavor, "failed", repr(ex)[:120], flush=True)
                T = []
            for t in T:                                                    # tables are chosen by content, not position
                t.columns = flat(t.columns)
            is_cur = [t for t in T if {"Symbol", "CIK", "Security"} <= set(t.columns)]
            is_chg = [t for t in T if any(c.lower().startswith("added") and "ticker" in c.lower() for c in t.columns)
                      and any(c.lower().startswith(("date", "effective")) for c in t.columns)]
            print("  ", flavor, "tables", len(T), "members", [len(t) for t in is_cur], "changes",
                  [len(t) for t in is_chg], flush=True)
            if is_cur and is_chg:
                break
        if is_cur and is_chg:
            break
        for i, t in enumerate(T):
            print("  table", i, t.shape, list(t.columns)[:8], flush=True)
    if not is_cur or not is_chg:
        raise SystemExit("S&P 500 tables not found")
    cur, chg = max(is_cur, key=len).copy(), max(is_chg, key=len).copy()
    cur["ticker"] = cur["Symbol"].astype(str).str.replace(".", "-", regex=False)
    cur[["ticker", "Security", "GICS Sector", "CIK"]].to_csv(os.path.join(a.out, "sp500_current.csv"), index=False)
    chg.to_csv(os.path.join(a.out, "sp500_changes.csv"), index=False)
    print("current", len(cur), "changes", len(chg), list(chg.columns), flush=True)
    add_col = [c for c in chg.columns if c.lower().startswith("added") and "ticker" in c.lower()][0]
    added = chg[add_col].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
    tick = sorted(set(cur.ticker) | set(added) | {"SPY"} | set(ETF_SECT) | set(ETF_CTRY))
    P = yf_prices(tick)
    print("prices", len(P), "of", len(tick), flush=True)
    for k in ("open", "close", "dv"):
        pd.DataFrame({t: d[k] for t, d in P.items()}).to_parquet(os.path.join(a.out, f"px_{k}.parquet"))
    # SEC 8-K item 2.02 dates for the current members
    def subs(cik):
        cik = int(cik)
        try:
            j = C.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", js=True, tries=4)
        except Exception as ex:                                            # noqa: BLE001
            return cik, None, repr(ex)[:80]
        recs = [j["filings"]["recent"]]
        for f in j["filings"].get("files", []):
            try:
                recs.append(C.get(f"https://data.sec.gov/submissions/{f['name']}", js=True, tries=4))
            except Exception:                                              # noqa: BLE001
                pass
            time.sleep(0.12)
        rows = []
        for r in recs:
            for form, acc, items in zip(r.get("form", []), r.get("acceptanceDateTime", []), r.get("items", [])):
                if form == "8-K" and "2.02" in str(items):
                    rows.append((cik, acc))
        time.sleep(0.12)
        return cik, rows, None
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(subs, cur["CIK"].tolist()))
    ev = [r for _, rows, _ in res if rows for r in rows]
    bad = [c for c, rows, e in res if rows is None]
    E = pd.DataFrame(ev, columns=["cik", "accepted"])
    E = E.merge(cur[["CIK", "ticker"]].rename(columns={"CIK": "cik"}), on="cik")
    E.to_csv(os.path.join(a.out, "earnings_8k.csv"), index=False)
    json.dump({"current": len(cur), "changes": len(chg), "prices": len(P), "tickers": len(tick), "events": len(E),
               "sec_failed": len(bad)}, open(os.path.join(a.out, "data_meta.json"), "w"), indent=1)
    print("8-K 2.02 events", len(E), "SEC failures", len(bad), flush=True)


def load(dd):
    O, Cl, DV = (pd.read_parquet(os.path.join(dd, f"px_{k}.parquet")) for k in ("open", "close", "dv"))
    for X in (O, Cl, DV):
        X.index = pd.to_datetime(X.index).tz_localize(None)
    return O.sort_index(), Cl.sort_index(), DV.sort_index()


# ------------------------------------------------------------------------------------------------ R4 earnings drift
def cmd_ear(a):
    O, Cl, DV = load(a.data)
    E = pd.read_csv(os.path.join(a.data, "earnings_8k.csv"))
    cur = pd.read_csv(os.path.join(a.data, "sp500_current.csv"))
    tick = [t for t in cur.ticker if t in Cl.columns]
    days = Cl.index[(Cl.index >= "2008-01-01") & (Cl.index <= "2025-12-31")]
    Cl, O, DV = Cl.reindex(days), O.reindex(days), DV.reindex(days)
    spy_c, spy_o = Cl["SPY"].to_numpy(float), O["SPY"].to_numpy(float)
    acc = pd.to_datetime(E["accepted"], utc=True).dt.tz_convert("America/New_York")
    E["d0"] = acc.dt.tz_localize(None).dt.normalize()
    after = acc.dt.hour * 60 + acc.dt.minute >= 16 * 60
    pos = np.searchsorted(days.values, E["d0"].values)                     # first trading day on or after the date
    same = (pos < len(days)) & (days.values[np.minimum(pos, len(days) - 1)] == E["d0"].values)
    E["i0"] = pos + (after.to_numpy() & same).astype(int)                  # after the close: the next trading day
    E = E[(E.i0 > 0) & (E.i0 < len(days) - 70) & E.ticker.isin(tick)].copy()
    E = E.sort_values(["ticker", "i0"])
    E = E[E.groupby("ticker").i0.diff().fillna(99) > 5]                    # duplicates within 5 trading days
    E = E[days.values[E.i0.values] >= np.datetime64("2009-01-01")]
    col = {t: i for i, t in enumerate(Cl.columns)}
    CC, OO = Cl.to_numpy(float), O.to_numpy(float)
    dvm = DV.rolling(60, min_periods=20).median().shift(1).to_numpy(float)
    dvmed = np.nanmedian(dvm[:, [col[t] for t in tick]], axis=1)
    rng = np.random.default_rng(2440)
    shift = {t: int(rng.integers(20, 121)) for t in tick}

    def build(ev, placebo):
        rows = []
        for t, i0 in zip(ev.ticker.values, ev.i0.values):
            i0 = int(i0) + (shift[t] if placebo else 0)
            if i0 >= len(days) - 70:
                continue
            j = col[t]
            if not (np.isfinite(CC[i0, j]) and np.isfinite(CC[i0 - 1, j])):
                continue
            sig = (CC[i0, j] / CC[i0 - 1, j] - 1) - (spy_c[i0] / spy_c[i0 - 1] - 1)
            k = np.arange(1, 67)
            oc = (CC[i0 + k, j] / OO[i0 + k, j] - 1) - (spy_c[i0 + k] / spy_o[i0 + k] - 1)
            cc = (CC[i0 + k, j] / CC[i0 + k - 1, j] - 1) - (spy_c[i0 + k] / spy_c[i0 + k - 1] - 1)
            liq = 1 if dvm[i0, j] >= dvmed[i0] else 0
            rows.append((i0, sig, liq, np.nan_to_num(oc), np.nan_to_num(cc)))
        return rows

    rows_out, ser = [], {}
    for placebo in (False, True):
        R = build(E, placebo)
        i0 = np.array([r[0] for r in R]); sg = np.array([r[1] for r in R]); lq = np.array([r[2] for r in R])
        OC = np.stack([r[3] for r in R]); CCa = np.stack([r[4] for r in R])
        print("events", len(R), "placebo", placebo, flush=True)
        U = C.sobol(7, 128, seed=2441)
        desc, cols = [], {}
        for ci, u in enumerate(U):
            p = {"D": C.ilin(u[0], 1, 5), "q": C.lin(u[1], 0.1, 0.5), "delay": C.ilin(u[2], 0, 5), "H": C.ilin(u[3], 1, 60),
                 "dir": C.cat(u[4], ["drift", "reversal"]), "form": C.cat(u[5], ["long_short", "long_vs_spy"]),
                 "liq": C.cat(u[6], ["all", "high", "low"])}
            desc.append(p)
            cols[ci] = ear_run(i0, sg, lq, OC, CCa, p, len(days))
        W = pd.DataFrame(cols, index=days)
        r, s_ = C.units("R4", "R4-earnings-drift", "SP500", W, placebo=placebo, desc=None if placebo else desc)
        rows_out += r
        ser.update(s_)
    C.write(a.out, "R4", rows_out, ser, {"events": int(len(E))})


def ear_run(i0, sg, lq, OC, CCa, p, nd, cost=5e-4):
    keep = np.ones(len(i0), bool) if p["liq"] == "all" else (lq == (1 if p["liq"] == "high" else 0))
    i0, sg, OC, CCa = i0[keep], sg[keep], OC[keep], CCa[keep]
    order = np.argsort(i0, kind="stable")
    i0, sg, OC, CCa = i0[order], sg[order], OC[order], CCa[order]
    side = np.zeros(len(i0))
    lo_ptr = 0
    for e in range(len(i0)):
        while i0[lo_ptr] < i0[e] - p["D"] + 1:
            lo_ptr += 1
        hi_ptr = np.searchsorted(i0, i0[e], side="right")
        pool = sg[lo_ptr:hi_ptr]
        if len(pool) < 5:
            continue
        pct = (pool < sg[e]).mean() + 0.5 * (pool == sg[e]).mean()
        if pct >= 1 - p["q"]:
            side[e] = 1
        elif pct <= p["q"]:
            side[e] = -1
    if p["dir"] == "reversal":
        side = -side
    if p["form"] == "long_vs_spy":
        side = np.where(side > 0, 1.0, 0.0)
    sl, cl_, ss, cs = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
    d0, H = p["delay"], p["H"]
    for e in np.flatnonzero(side != 0):
        k0 = d0                                                            # offset index: day i0 + 1 + delay
        r = np.r_[OC[e, k0], CCa[e, k0 + 1:k0 + H]]
        r = r.copy()
        r[0] -= cost
        r[-1] -= cost
        dd = i0[e] + 1 + d0 + np.arange(len(r))
        ok = dd < nd
        if side[e] > 0:
            np.add.at(sl, dd[ok], r[ok]); np.add.at(cl_, dd[ok], 1)
        else:
            np.add.at(ss, dd[ok], r[ok]); np.add.at(cs, dd[ok], 1)
    ml = np.where(cl_ > 0, sl / np.maximum(cl_, 1), 0.0)
    ms = np.where(cs > 0, ss / np.maximum(cs, 1), 0.0)
    if p["form"] == "long_vs_spy":
        return ml
    return 0.5 * (ml - ms)


# ------------------------------------------------------------------------------------------------ R6 short volume
def cmd_shortvol(a):
    O, Cl, DV = load(a.data)
    cur = pd.read_csv(os.path.join(a.data, "sp500_current.csv"))
    tick = [t for t in cur.ticker if t in O.columns]
    sv_path = os.path.join(a.data, "finra_sv.parquet")
    if not os.path.exists(sv_path):
        days = O.index[(O.index >= "2009-08-01") & (O.index <= "2025-12-31")]
        fin = set(t.replace("-", ".") for t in tick) | set(tick)

        def one(d):
            ds = d.strftime("%Y%m%d")
            try:
                r = C.get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ds}.txt", tries=3, timeout=60)
            except Exception:                                              # noqa: BLE001
                return d, None
            x = pd.read_csv(io.StringIO(r.text), sep="|")
            x = x[x["Symbol"].isin(fin)]
            return d, x[["Symbol", "ShortVolume", "TotalVolume"]]
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(one, days))
        parts = []
        for d, x in got:
            if x is not None and len(x):
                x = x.assign(date=d)
                parts.append(x)
        S = pd.concat(parts)
        S["Symbol"] = S["Symbol"].str.replace(".", "-", regex=False)
        S.to_parquet(sv_path)
        print("FINRA days", len(parts), "of", len(days), "first", parts[0].date.iloc[0] if parts else None, flush=True)
    S = pd.read_parquet(sv_path)
    ratio = (S.pivot_table(index="date", columns="Symbol", values="ShortVolume", aggfunc="sum")
             / S.pivot_table(index="date", columns="Symbol", values="TotalVolume", aggfunc="sum"))
    ratio.index = pd.to_datetime(ratio.index)
    days = ratio.index
    tk = [t for t in tick if t in ratio.columns]
    Rt = ratio[tk].to_numpy(float)
    Oo = O.reindex(days)[tk].to_numpy(float)
    roo = np.full_like(Oo, np.nan)
    roo[:-1] = Oo[1:] / Oo[:-1] - 1                                       # open t -> open t + 1, booked on day t
    rows, ser = [], {}
    rng = np.random.default_rng(2460)
    Rp = Rt.copy()
    for i in range(len(Rp)):
        f = np.flatnonzero(np.isfinite(Rp[i]))
        Rp[i, f] = Rp[i, rng.permutation(f)]
    for placebo, X in ((False, Rt), (True, Rp)):
        U = C.sobol(6, 128, seed=2461)
        desc, cols = [], {}
        for ci, u in enumerate(U):
            p = {"L": C.ilin(u[0], 1, 20), "kind": C.cat(u[1], ["level", "change"]), "q": C.lin(u[2], 0.1, 0.5),
                 "side": C.cat(u[3], ["long_low", "long_high"]), "H": C.ilin(u[4], 1, 20),
                 "reb": C.cat(u[5], ["daily", "weekly"])}
            desc.append(p)
            cols[ci] = sv_run(X, roo, days, p)
        r, s_ = C.units("R6", "R6-short-volume", "SP500", pd.DataFrame(cols, index=days), placebo=placebo,
                        desc=None if placebo else desc)
        rows += r
        ser.update(s_)
    C.write(a.out, "R6", rows, ser, {"days": int(len(days)), "first": str(days[0].date()), "stocks": len(tk)})


def sv_run(X, roo, days, p, cost=5e-4):
    Xs = pd.DataFrame(X).rolling(p["L"], min_periods=max(1, p["L"] // 2)).mean().to_numpy()
    if p["kind"] == "change":
        Xs = Xs - pd.DataFrame(Xs).shift(p["L"]).to_numpy()
    n, m = Xs.shape
    w = np.zeros((n, m))
    form = np.ones(n, bool) if p["reb"] == "daily" else (pd.DatetimeIndex(days).dayofweek == 0)
    for t in np.flatnonzero(form):
        x = Xs[t]
        f = np.flatnonzero(np.isfinite(x) & np.isfinite(roo[t]))
        if len(f) < 20:
            continue
        r = x[f].argsort().argsort() / (len(f) - 1)
        lo, hi = f[r <= p["q"]], f[r >= 1 - p["q"]]
        sgn = 1.0 if p["side"] == "long_low" else -1.0
        w[t, lo] += sgn * 0.5 / max(len(lo), 1)
        w[t, hi] -= sgn * 0.5 / max(len(hi), 1)
    k = p["H"] if p["reb"] == "daily" else p["H"]
    cw = np.cumsum(w, axis=0)
    W = cw - np.vstack([np.zeros((k, m)), cw[:-k]]) if n > k else cw
    W = W / (k if p["reb"] == "daily" else max(1, int(np.ceil(k / 5))))
    gross = np.abs(W).sum(1, keepdims=True)
    W = np.where(gross > 0, W / np.maximum(gross, 1e-12), 0.0)
    ret = np.nansum(W * np.nan_to_num(roo), axis=1)
    turn = np.abs(np.diff(np.vstack([np.zeros((1, m)), W]), axis=0)).sum(1)
    return ret - turn * cost


# ------------------------------------------------------------------------------------------------ R11a ETF pairs
def cmd_pairs(a):
    O, Cl, DV = load(a.data)
    uni = {"sectors": ETF_SECT, "countries": ETF_CTRY, "all": ETF_SECT + ETF_CTRY}
    Px = Cl[[t for t in ETF_SECT + ETF_CTRY if t in Cl.columns]]
    Px = Px[(Px.index >= "1998-01-01") & (Px.index <= "2025-12-31")]
    R = Px.pct_change()
    rows, ser = [], {}
    rng = np.random.default_rng(2475)
    for placebo in (False, True):
        U = C.sobol(6, 128, seed=2476)
        desc, cols = [], {}
        for ci, u in enumerate(U):
            p = {"F": C.ilin(u[0], 60, 500), "n": C.ilin(u[1], 1, 20), "k": C.lin(u[2], 1.0, 3.0), "f": C.lin(u[3], 0.5, 1.0),
                 "H": C.ilin(u[4], 5, 60), "uni": C.cat(u[5], list(uni))}
            desc.append(p)
            cols[ci] = pairs_run(Px[[t for t in uni[p["uni"]] if t in Px.columns]], R, p, rng if placebo else None)
        r, s_ = C.units("R11", "R11a-etf-pairs", "ETF", pd.DataFrame(cols), placebo=placebo, desc=None if placebo else desc)
        rows += r
        ser.update(s_)
    C.write(a.out, "R11a", rows, ser)


def pairs_run(Px, R, p, rng=None, cost=2e-4):
    idx = Px.index
    ret = pd.Series(0.0, index=idx)
    months = pd.DatetimeIndex(sorted({d.replace(day=1) for d in idx}))
    V = Px.to_numpy(float)
    Rv = R.reindex(columns=Px.columns).to_numpy(float)
    for m0, m1 in zip(months[:-1], months[1:]):
        t0 = idx.searchsorted(m0)
        t1 = idx.searchsorted(m1)
        if t0 < p["F"]:
            continue
        win = V[t0 - p["F"]:t0]
        ok = np.flatnonzero(np.all(np.isfinite(win), 0) & np.isfinite(V[t0 - 1]))
        if len(ok) < 2:
            continue
        N = win[:, ok] / win[0, ok]
        pr = [(i, j) for i in range(len(ok)) for j in range(i + 1, len(ok))]
        ssd = np.array([np.sum((N[:, i] - N[:, j]) ** 2) for i, j in pr])
        sel = np.argsort(ssd)[:p["n"]] if rng is None else rng.choice(len(pr), size=min(p["n"], len(pr)), replace=False)
        daily = np.zeros(t1 - t0)
        for s_ in sel:
            i, j = pr[s_]
            ci, cj = ok[i], ok[j]
            sd = np.std(N[:, i] - N[:, j])
            if not sd > 0:
                continue
            ni = V[t0:t1, ci] / V[t0 - p["F"], ci]                          # normalised as in the formation
            nj = V[t0:t1, cj] / V[t0 - p["F"], cj]
            spread = ni - nj
            pos, entry, held = 0, 0.0, 0
            for t in range(t1 - t0):
                r_ = 0.0
                if pos != 0:
                    r_ += pos * 0.5 * (Rv[t0 + t, ci] - Rv[t0 + t, cj])
                    held += 1
                    if abs(spread[t]) <= (1 - p["f"]) * abs(entry) or held >= p["H"] or t == t1 - t0 - 1:
                        r_ -= 2 * cost
                        pos = 0
                elif abs(spread[t]) > p["k"] * sd and t < t1 - t0 - 1:
                    pos = -1 if spread[t] > 0 else 1                       # short the leader, long the laggard
                    entry, held = spread[t], 0
                    r_ -= 2 * cost
                daily[t] += np.nan_to_num(r_)
        ret.iloc[t0:t1] += daily / p["n"]
    return ret


# ------------------------------------------------------------------------------------------------ R11b S&P additions
def cmd_adds(a):
    O, Cl, DV = load(a.data)
    chg = pd.read_csv(os.path.join(a.data, "sp500_changes.csv"))
    dcol = [c for c in chg.columns if c.lower().startswith("date") or c.lower().startswith("effective")][0]
    acol = [c for c in chg.columns if c.lower().startswith("added") and "ticker" in c.lower()][0]
    ev = pd.DataFrame({"date": pd.to_datetime(chg[dcol], errors="coerce"),
                       "ticker": chg[acol].astype(str).str.replace(".", "-", regex=False)}).dropna()
    ev = ev[(ev.date >= "2008-01-01") & (ev.date <= "2025-12-31") & ev.ticker.isin(Cl.columns)]
    days = Cl.index[(Cl.index >= "2008-01-01") & (Cl.index <= "2025-12-31")]
    CC = Cl.reindex(days)
    ab = CC.pct_change().sub(CC["SPY"].pct_change(), axis=0)
    A = ab.to_numpy(float)
    col = {t: i for i, t in enumerate(CC.columns)}
    ev["i"] = days.searchsorted(ev.date)
    ev = ev[(ev.i > 25) & (ev.i < len(days) - 25)]
    rng = np.random.default_rng(2477)
    rows, ser = [], {}
    for placebo in (False, True):
        E = ev.copy()
        if placebo:
            E["i"] = rng.integers(25, len(days) - 25, len(E))
        U = C.sobol(3, 128, seed=2478)
        desc, cols = [], {}
        for ci, u in enumerate(U):
            p = {"e": C.ilin(u[0], 0, 1), "x": C.ilin(u[1], 1, 20), "side": C.cat(u[2], ["long", "short"])}
            desc.append(p)
            s, c = np.zeros(len(days)), np.zeros(len(days))
            for t, i in zip(E.ticker, E.i):
                dd = np.arange(i - p["e"], i + p["x"] + 1)
                r = np.nan_to_num(A[dd, col[t]]) * (1 if p["side"] == "long" else -1)
                r[0] -= 5e-4
                r[-1] -= 5e-4
                np.add.at(s, dd, r); np.add.at(c, dd, 1)
            cols[ci] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
        r, s_ = C.units("R11", "R11b-sp500-adds", "SP500", pd.DataFrame(cols, index=days), placebo=placebo,
                        desc=None if placebo else desc)
        rows += r
        ser.update(s_)
    C.write(a.out, "R11b", rows, ser, {"events": int(len(ev))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "ear", "shortvol", "pairs", "adds"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="sdata")
    a = ap.parse_args()
    {"data": cmd_data, "ear": cmd_ear, "shortvol": cmd_shortvol, "pairs": cmd_pairs, "adds": cmd_adds}[a.cmd](a)


if __name__ == "__main__":
    main()
