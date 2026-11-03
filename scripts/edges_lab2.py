"""The edges batch, as fixed in docs/edges_batch_preregistration.md (E2-E11, E13-E16, E18, E19, E23, E24; E1, E12,
E17, E20-E22 have their own scripts). Rules are copied from that file; where the data forced a detail, it is
written next to the test (DATA NOTE) and in the results.

Common rules:
  - Development data only (to 2025-12-31). M1 bid/ask fills and costs of the flows lab (Book.trade): crossing the
    spread + 25% of it + 0.1 pip each way, 0.35 bp commission a side, swap (rate differential where rates exist,
    else the flat platform swap).
  - Split: 2007-2018 fit, 2019-2025 check; a series whose first event is after 2008-12-31 is split at its first
    60% of events.
  - Reported per test: fit and check n, net mean (bp), one-sided t and p in the chosen direction, win rate, and the
    event-minus-control gross difference where a control exists. Benjamini-Hochberg over the whole batch is done
    by --aggregate.
Usage: python scripts/edges_lab2.py --tests E4 E7 ... --raw RAWDIR --out DIR
       python scripts/edges_lab2.py --aggregate DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_END = pd.Timestamp("2018-12-31")
END = pd.Timestamp("2025-12-31")
BOOKS = {}
RAW = "."
OUT = "."


# ------------------------------------------------------------------------------------------------ helpers
def pip_of(sym):
    if sym.startswith("XAU"):
        return 0.01
    if sym.startswith("XAG"):
        return 0.001
    return 0.01 if sym.endswith("JPY") else 1e-4


def book(sym):
    if sym not in BOOKS:
        BOOKS.clear()                                                       # one instrument in memory at a time
        bk = None
        try:
            m = U.load(sym)
            if m is not None and len(m) > 1000:
                bk = FL.Book(m, pip_of(sym), sym if len(sym) == 6 else None)
                bk._hl()
        except Exception as ex:                                             # noqa: BLE001
            print("book", sym, repr(ex)[:200], flush=True)
        BOOKS[sym] = bk
    return BOOKS[sym]


def tnet(r):
    return U.net_of(r)


def both(bk, t_in, t_out, tol=5):
    """Long and short nets (with swap) and the long gross of one window, or None."""
    a = bk.trade(t_in, t_out, +1, tol_min=tol)
    if a is None:
        return None
    b = bk.trade(t_in, t_out, -1, tol_min=tol)
    if b is None:
        return None
    return {"long": tnet(a), "short": tnet(b), "gross": float(a[1])}


def frame(rows):
    f = pd.DataFrame.from_dict(rows, orient="index")
    if f.empty:
        return pd.DataFrame(columns=["long", "short", "gross"], index=pd.DatetimeIndex([]))
    f.index = pd.DatetimeIndex(f.index)
    if f.index.tz is not None:
        f.index = f.index.tz_convert(None)
    return f.sort_index()


def split_mask(idx):
    idx = pd.DatetimeIndex(idx)
    if len(idx) and idx.min() > pd.Timestamp("2008-12-31"):
        cut = idx.sort_values()[int(0.6 * len(idx))]
        return idx < cut, "60/40 by events"
    return idx <= TRAIN_END, "2007-2018 / 2019-2025"


def tstat(x):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 8 or x.std() == 0:
        return np.nan, np.nan
    t = float(x.mean() / x.std() * np.sqrt(len(x)))
    return t, float(st.t.sf(t, len(x) - 1))


def stats_block(prefix, x):
    x = pd.Series(x, dtype=float).dropna()
    t, p = tstat(x)
    return {f"{prefix}_n": int(len(x)), f"{prefix}_net_bp": float(x.mean() * 1e4) if len(x) else np.nan,
            f"{prefix}_t": t, f"{prefix}_p": p, f"{prefix}_win": float((x > 0).mean()) if len(x) else np.nan}


def result(test, inst, ev, direction, ctl=None, note=""):
    """ev: frame with long/short/gross. direction: +1 / -1 fixed, or 'train' = sign of the fit-period
    event (minus control) gross mean."""
    out = {"test": test, "instrument": inst, "note": note, "n_events": int(len(ev))}
    if len(ev) < 10:
        out["status"] = "too few events"
        return out
    fit, how = split_mask(ev.index)
    out["split"] = how
    if direction == "train":
        g = ev.gross[fit].mean()
        if ctl is not None and len(ctl):
            check_start = ev.index[~fit].min() if (~fit).any() else ev.index.max() + pd.Timedelta(days=1)
            g = g - ctl.gross[ctl.index < check_start].mean()
        direction = 1 if g > 0 else -1
        out["direction_from"] = "fit period"
    out["direction"] = "long" if direction > 0 else "short"
    net = ev.long if direction > 0 else ev.short
    out.update(stats_block("fit", net[fit]))
    out.update(stats_block("check", net[~fit]))
    if ctl is not None and len(ctl):
        for nm, m_e in (("fit", fit), ("check", ~fit)):
            lo, hi = ev.index[m_e].min(), ev.index[m_e].max()
            c = ctl[(ctl.index >= lo) & (ctl.index <= hi)]
            out[f"{nm}_diff_vs_control_bp"] = float((ev.gross[m_e].mean() - c.gross.mean()) * direction * 1e4)
    out["p_for_bh"] = out.get("check_p", np.nan)
    out["counts"] = bool(out.get("check_net_bp", -1) > 0 and (out.get("check_t") or -9) >= 1.645)
    return out


def signed_result(test, inst, net, note="", fit=None):
    """net: per-event net already in the rule's direction."""
    net = pd.Series(net, dtype=float).dropna().sort_index()
    out = {"test": test, "instrument": inst, "note": note, "n_events": int(len(net))}
    if len(net) < 10:
        out["status"] = "too few events"
        return out
    if fit is None:
        fit, how = split_mask(net.index)
    else:
        how = "given"
    out["split"] = how
    out.update(stats_block("fit", net[fit]))
    out.update(stats_block("check", net[~fit]))
    out["p_for_bh"] = out.get("check_p", np.nan)
    out["counts"] = bool(out.get("check_net_bp", -1) > 0 and (out.get("check_t") or -9) >= 1.645)
    return out


def utc(d, h, m=0):
    return pd.Timestamp(pd.Timestamp(d).date()).tz_localize("UTC") + pd.Timedelta(hours=h, minutes=m)


def us_bdays(start="2006-12-01", end="2026-01-31"):
    from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
    hol = set(USFederalHolidayCalendar().holidays(start, end))
    gf = set(GoodFriday.dates(start, end))
    return pd.DatetimeIndex([d for d in pd.bdate_range(start, end) if d not in hol and d not in gf])


def mid_at(bk, ts, tol=5, before=False):
    if before:
        k = int(bk.idx.searchsorted(ts, side="right")) - 1
        if k < 0 or (ts - bk.idx[k]) > pd.Timedelta(minutes=tol):
            return np.nan
    else:
        k = bk.at(ts, tol)
        if k < 0:
            return np.nan
    return 0.5 * (bk.bo[k] + bk.ao[k])


def raw(*parts):
    hits = sorted(glob.glob(os.path.join(RAW, "**", *parts), recursive=True))
    return hits


def fred_csv(sid):
    for p in raw(f"fred_{sid}.csv"):
        df = pd.read_csv(p)
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        return s.dropna()
    try:
        import requests
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
        df = pd.read_csv(io.StringIO(r.text))
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        return s.dropna()
    except Exception as ex:                                                 # noqa: BLE001
        print("FRED", sid, repr(ex)[:100], flush=True)
        return pd.Series(dtype=float)


def yahoo(t, start="1999-01-01"):
    import yfinance as yf
    d = yf.download(t, start=start, end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    d = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    return d[d.index <= END].dropna()


def gotobi_nights():
    B = pd.read_csv(os.path.join(ROOT, "docs/gotobi_losses/trades_basket_stop_80_bp.csv"), index_col=0, parse_dates=True)
    B.index = pd.DatetimeIndex(B.index).normalize()
    return B[B.index <= END]


def spearman_test(test, feat, y, note=""):
    """Feature vs Gotobi night net (rules of F in docs/panel_hedge_preregistration.md)."""
    j = pd.concat([feat.rename("x"), y.rename("y")], axis=1).dropna()
    tr, te = j[j.index <= TRAIN_END], j[j.index > TRAIN_END]
    out = {"test": test, "instrument": "Gotobi basket", "note": note, "fit_n": int(len(tr)), "check_n": int(len(te))}
    if len(tr) < 30 or len(te) < 20:
        out["status"] = "too few nights"
        return out
    r1, p1 = st.spearmanr(tr.x, tr.y)
    r2, p2 = st.spearmanr(te.x, te.y)
    sgn = 1 if r1 > 0 else -1
    p_one = float(p2 / 2 if np.sign(r2) == sgn else 1 - p2 / 2)
    out.update({"fit_rho": float(r1), "fit_p": float(p1), "check_rho": float(r2), "check_p_one_sided": p_one,
                "p_for_bh": p_one if p1 < 0.05 else 1.0,
                "counts": bool(p1 < 0.05 and np.sign(r2) == sgn and p_one < 0.05)})
    return out


# ------------------------------------------------------------------------------------------------ E2
def e2(rows, frames):
    """Cheaper Gotobi entry: buy limit at the 14:00 UTC bid + 25% of the spread; if not filled by 14:10, market."""
    legs = {}
    for p in ("USDJPY", "EURJPY"):
        bk = book(p)
        if bk is None:
            continue
        G, _, _ = F.calendar(bk)
        out = {}
        for d in G:
            if pd.Timestamp(d) > END:
                continue
            i, j = F.window(bk, d)
            if i < 0 or not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j]]).all():
                continue
            k10 = int(bk.idx.searchsorted(bk.idx[i] + pd.Timedelta(minutes=10)))
            if k10 >= j:
                continue

            def exit_ret(k, E):
                S = E * (1 - 0.008)
                q = np.flatnonzero(bk.bl[k:j] <= S)
                if len(q):
                    q = int(q[0]) + k
                    X = (min(S, bk.bo[q]) if q > k else S) - R.slip(bk, q)
                else:
                    X = bk.bo[j] - R.slip(bk, j)
                return X / E - 1.0
            Em = bk.ao[i] + R.slip(bk, i)
            L = bk.bo[i] + 0.25 * (bk.ao[i] - bk.bo[i])
            f = np.flatnonzero(bk.al[i:k10] <= L)
            if len(f):
                k = i + int(f[0])
                El, kl = min(L, bk.ao[k]), k
            else:
                El, kl = bk.ao[k10] + R.slip(bk, k10), k10
            out[d] = {"market": exit_ret(i, Em), "limit": exit_ret(kl, El), "filled": bool(len(f))}
        legs[p] = frame(out)
    if len(legs) < 2:
        rows.append({"test": "E2 cheaper Gotobi entry", "status": "no data"})
        return
    a, b = legs["USDJPY"], legs["EURJPY"]
    ix = a.index.union(b.index)
    m = pd.concat([a.market.reindex(ix), b.market.reindex(ix)], axis=1).mean(axis=1)
    lm = pd.concat([a.limit.reindex(ix), b.limit.reindex(ix)], axis=1).mean(axis=1)
    frames["E2"] = pd.DataFrame({"market": m, "limit": lm})
    r = signed_result("E2 cheaper Gotobi entry (limit minus market, basket)", "USDJPY+EURJPY", lm - m,
                      note=f"fill rate USDJPY {a.filled.mean():.0%}, EURJPY {b.filled.mean():.0%}; paired difference, "
                           "gross of commission and swap (the same for both entries)")
    rows.append(r)


# ------------------------------------------------------------------------------------------------ E3
def parse_mof():
    fs = raw("mof_ja_week.csv")
    if not fs:
        return None
    L = open(fs[0], encoding="cp932", errors="replace").read().splitlines()
    import csv
    out = {}
    for row in csv.reader(L):
        if not row:
            continue
        m = re.match(r"\s*(\d{4})[．.](\d{1,2})[．.](\d{1,2})\s*～\s*(\d{1,2})[．.](\d{1,2})", row[0])
        if not m or len(row) < 7:
            continue
        y, m1, d1, m2, d2 = map(int, m.groups())
        start = pd.Timestamp(y, m1, d1)
        endy = y + 1 if m2 < m1 else y
        end = pd.Timestamp(endy, m2, d2)
        try:
            v = float(row[6].replace(",", "").strip())
        except ValueError:
            continue
        out[end] = v
    return pd.Series(out).sort_index()


def e3(rows, frames):
    B = gotobi_nights()
    y = B.net
    entry = pd.Series(B.index - pd.Timedelta(days=1), index=B.index)
    s = parse_mof()
    if s is None or len(s) < 100:
        rows.append({"test": "E3a MoF weekly foreign bond purchases", "status": "untestable (file did not parse)"})
    else:
        z = (s - s.rolling(52, min_periods=26).mean()) / s.rolling(52, min_periods=26).std()
        avail = pd.Series(z.values, index=z.index + pd.Timedelta(days=5))   # released the Thursday after the week
        feat = pd.Series({d: avail[avail.index <= e].iloc[-1] if (avail.index <= e).any() else np.nan
                          for d, e in entry.items()})
        frames["E3a"] = feat.to_frame("z")
        rows.append(spearman_test("E3a MoF weekly net foreign long-term bond purchases (z, 52 weeks)", feat, y,
                                  note=f"{len(s)} weeks parsed; usable from the Thursday after each week"))
    tb = fred_csv("XTNTVA01JPM664S")
    if len(tb) < 100:
        rows.append({"test": "E3b Japan trade balance", "status": "untestable (no FRED data)"})
        return
    tb = tb[tb.index >= "2004-01-01"]
    z = (tb - tb.rolling(36, min_periods=24).mean()) / tb.rolling(36, min_periods=24).std()
    # DATA NOTE: month m is published around the 20th of m+1; a night uses the latest month published before it
    pub = pd.Series(z.values, index=[d + pd.DateOffset(months=1, days=19) for d in z.index])
    feat = pd.Series({d: pub[pub.index <= e].iloc[-1] if (pub.index <= e).any() else np.nan for d, e in entry.items()})
    frames["E3b"] = feat.to_frame("z")
    rows.append(spearman_test("E3b Japan trade balance (z, 36 months, latest published)", feat, y,
                              note="publication lag: month m used from the 20th of m+1"))


# ------------------------------------------------------------------------------------------------ E4
def parse_cny():
    fs = raw("cny_fixing.json")
    if not fs:
        return pd.Series(dtype=float)
    J = json.load(open(fs[0]))
    out = {}
    for ch in J:
        recs = (ch.get("payload") or {}).get("records") or ((ch.get("payload") or {}).get("data") or {}).get("records") or []
        for r in recs:
            try:
                out[pd.Timestamp(r["date"])] = float(r["values"][0])
            except (KeyError, ValueError, IndexError, TypeError):
                continue
    return pd.Series(out).sort_index()


def e4(rows, frames):
    fx = parse_cny()
    if len(fx) < 100:
        rows.append({"test": "E4 yuan fixing", "status": "untestable (no fixing data)"})
        return
    bk = book("USDCNH")
    if bk is None:
        rows.append({"test": "E4 yuan fixing", "status": "untestable (no USD/CNH bars)"})
        return
    sig, ev = {}, {}
    for d, fix in fx.items():
        if d > END or d < bk.idx[0].tz_convert(None).normalize():
            continue
        m = mid_at(bk, utc(d, 1, 14), tol=3, before=True)
        if not np.isfinite(m):
            continue
        r = both(bk, utc(d, 1, 16), utc(d, 8, 0))
        if r is None:
            continue
        sig[d] = fix - m
        ev[d] = r
    E = frame(ev)
    S = pd.Series(sig)
    S.index = pd.DatetimeIndex(S.index)
    S = S.reindex(E.index)
    fit, how = split_mask(E.index)
    check_start = E.index[~fit].min()
    cut = float(S[fit].abs().quantile(0.75))
    allnet = pd.Series(np.where(S > 0, E.long, E.short), index=E.index)
    net = allnet[S.abs() > cut]
    frames["E4_USDCNH"] = pd.DataFrame({"signal": S, "net_in_direction": allnet})
    rows.append(signed_result("E4 yuan fixing: USD/CNH 01:16-08:00 UTC in the direction of fix minus market",
                              "USDCNH", net, note=f"|signal| > fit 75th pct ({cut * 1e4:.0f} pips); split {how}",
                              fit=np.asarray(net.index < check_start)))
    bk = book("AUDUSD")
    if bk is None:
        return
    ev2 = {}
    for d in net.index:
        r = both(bk, utc(d, 1, 16), utc(d, 8, 0))
        if r is not None:
            ev2[d] = r["short"] if S[d] > 0 else r["long"]
    n2 = pd.Series(ev2)
    n2.index = pd.DatetimeIndex(n2.index)
    rows.append(signed_result("E4 yuan fixing: AUD/USD opposite to the signal", "AUDUSD", n2,
                              note="same days as the USD/CNH rule", fit=np.asarray(n2.index < check_start)))


# ------------------------------------------------------------------------------------------------ E6 / E9
def gold_expiry_days(bd):
    s = pd.Series(bd, index=bd)
    return pd.DatetimeIndex(s.groupby([bd.year, bd.month]).apply(lambda x: x.iloc[-4] if len(x) >= 4 else pd.NaT).dropna())


def msci_days(bd):
    s = pd.Series(bd, index=bd)
    last = s.groupby([bd.year, bd.month]).max()
    return pd.DatetimeIndex([d for d in last if d.month in (2, 5, 8, 11)])


def windows(bk, days, fi, fo, tol=5):
    out = {}
    for d in days:
        if d > END:
            continue
        r = both(bk, fi(d), fo(d), tol)
        if r is not None:
            out[d] = r
    return frame(out)


def e6(rows, frames):
    bk = book("XAUUSD")
    if bk is None:
        rows.append({"test": "E6", "status": "no XAUUSD"})
        return
    bd = us_bdays()
    ex = gold_expiry_days(bd)
    prev = {d: bd[bd.get_loc(d) - 1] for d in bd[1:]}
    others = bd[1:].difference(ex)
    for name, fi, fo in (("E6a gold options expiry 13:30-17:30 UTC", lambda d: utc(d, 13, 30), lambda d: utc(d, 17, 30)),
                         ("E6b gold, 13:30 UTC the day before expiry -> 13:30 UTC expiry day",
                          lambda d: utc(prev[d], 13, 30), lambda d: utc(d, 13, 30))):
        e, c = windows(bk, ex[ex > bd[0]], fi, fo), windows(bk, others, fi, fo)
        frames[name.split()[0]] = e
        rows.append(result(name, "XAUUSD", e, "train", c))


def e9(rows, frames):
    bd = us_bdays()
    ms = msci_days(bd)
    nxt = {d: bd[bd.get_loc(d) + 1] for d in bd[:-1]}
    others = bd[:-1].difference(ms)
    for sym in ("USDJPY", "EURJPY"):
        bk = book(sym)
        if bk is None:
            continue
        # DATA NOTE: 2007-2018 files have no 00:00-01:00 UTC bars; E9b enters at the first quote within 65 minutes
        for name, fi, fo, tol in (("E9a MSCI review day 14:00-21:00 UTC", lambda d: utc(d, 14), lambda d: utc(d, 21), 5),
                                  ("E9b MSCI review, next day 00:00-03:00 UTC", lambda d: utc(nxt[d], 0),
                                   lambda d: utc(nxt[d], 3), 65)):
            e, c = windows(bk, ms, fi, fo, tol), windows(bk, others, fi, fo, tol)
            frames[f"{name.split()[0]}_{sym}"] = e
            rows.append(result(name, sym, e, "train", c))


# ------------------------------------------------------------------------------------------------ E7
def auctions():
    fs = raw("treasury_auctions.json")
    if not fs:
        return []
    J = json.load(open(fs[0]))
    out = []
    for r in J:
        typ = r.get("security_type", "")
        term = r.get("original_security_term") or r.get("security_term") or ""
        term2 = r.get("security_term") or ""
        if typ not in ("Note", "Bond"):
            continue
        ten = any(t.startswith(("10-Year", "9-Year")) for t in (term, term2))      # reopenings: "9-Year 10-Month"
        thirty = any(t.startswith(("30-Year", "29-Year")) for t in (term, term2))
        if not (ten or thirty):
            continue
        try:
            d = pd.Timestamp(r["auction_date"])
        except Exception:                                                   # noqa: BLE001
            continue
        ct = r.get("closing_time_comp") or "01:00 PM"
        m = re.match(r"\s*(\d{1,2}):(\d{2})\s*([AP]M)", str(ct))
        hh, mm = (13, 0)
        if m:
            hh, mm = int(m.group(1)) % 12 + (12 if m.group(3) == "PM" else 0), int(m.group(2))
        close = FL.local_ts(d, "America/New_York", hh, mm)
        out.append((d.normalize(), "10y" if ten else "30y", close))
    return sorted(set(out))


def e7(rows, frames):
    A = auctions()
    if len(A) < 50:
        rows.append({"test": "E7 Treasury auctions", "status": "untestable (auction dates did not parse)"})
        return
    bd = us_bdays()
    nxt = {d: bd[bd.get_loc(d) + 1] for d in bd[:-1]}
    for sym, pre_dir in (("USDJPY", +1), ("XAUUSD", -1)):
        bk = book(sym)
        if bk is None:
            continue
        pre, post = {}, {}
        for d, kind, close in A:
            if d > END or d not in nxt:
                continue
            a = both(bk, utc(d, 13, 30), close - pd.Timedelta(minutes=5))
            if a is not None:
                pre[(d, kind)] = a
            b = both(bk, close, utc(nxt[d], 14, 0))
            if b is not None:
                post[(d, kind)] = b
        for nm, dd, dirn in (("E7a pre-auction 13:30 UTC -> close - 5 min", pre, pre_dir),
                             ("E7b post-auction close -> 14:00 UTC next day", post, -pre_dir)):
            f = pd.DataFrame({k[0]: v for k, v in dd.items()}).T
            f.index = pd.DatetimeIndex(f.index)
            f = f.sort_index()
            frames[f"{nm.split()[0]}_{sym}"] = f
            rows.append(result(f"{nm} (10y and 30y)", sym, f, dirn,
                               note="direction fixed by Lou, Yan & Zhang (2013): dollar up / gold down before, the reverse after"))


# ------------------------------------------------------------------------------------------------ E8
def boj_dates():
    out = set()
    for p in raw("cb", "boj", "*.htm"):
        t = open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"k(\d{2})(\d{2})(\d{2})[a-z]?\.(?:pdf|htm)", t):
            y, mo, d = 2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                out.add(pd.Timestamp(y, mo, d))
            except ValueError:
                pass
    return sorted(d for d in out if pd.Timestamp("2006-01-01") <= d <= END)


def calendar_file(pattern):
    fs = [f for f in raw("calendar_repo", "news_data", "*.csv") if re.search(pattern, os.path.basename(f))]
    if not fs:
        return None
    df = pd.read_csv(fs[0])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df[~df.Timestamp.dt.normalize().duplicated(keep="first")]          # one release a day (the earliest stamp)
    return df.reset_index(drop=True)


def e8(rows, frames):
    B = boj_dates()
    if len(B) >= 20:
        bk = book("USDJPY")
        if bk is not None:
            ev = {}
            for d in B:
                dec = utc(d, 3, 0)                                          # DATA NOTE: BoJ decisions ~12:00 JST
                r = both(bk, dec - pd.Timedelta(hours=24), dec - pd.Timedelta(minutes=15))
                if r is not None:
                    ev[d] = r
            f = frame(ev)
            frames["E8_BoJ"] = f
            rows.append(result("E8 BoJ pre-decision drift, 24 h -> 15 min before (03:00 UTC assumed)", "USDJPY", f,
                               "train", note=f"{len(B)} meeting dates from the BoJ statement links"))
    else:
        rows.append({"test": "E8 BoJ pre-decision drift", "status": f"untestable ({len(B)} dates parsed)"})
    E = calendar_file(r"Eurozone_Interest_Rate_Decision")
    if E is None or len(E) < 20:
        rows.append({"test": "E8 ECB pre-decision drift", "status": "untestable (no dates)"})
    else:
        bk = book("EURUSD")
        if bk is not None:
            ev = {}
            for ts in E.Timestamp:
                dec = pd.Timestamp(ts).tz_localize("UTC")
                r = both(bk, dec - pd.Timedelta(hours=24), dec - pd.Timedelta(minutes=15))
                if r is not None:
                    ev[dec.tz_convert(None).normalize()] = r
            f = frame(ev)
            frames["E8_ECB"] = f
            rows.append(result("E8 ECB pre-decision drift, 24 h -> 15 min before", "EURUSD", f, "train",
                               note="dates 2017-2023 from the calendar dataset"))
    rows.append({"test": "E8 BoE pre-decision drift", "status": "untestable (Bank of England pages are rendered by script)"})


# ------------------------------------------------------------------------------------------------ E10
def num(x):
    if not isinstance(x, str):
        return float(x) if pd.notna(x) else np.nan
    s = x.strip().replace(",", "")
    mult = 1.0
    if s.endswith("K"):
        s, mult = s[:-1], 1e3
    elif s.endswith("M"):
        s, mult = s[:-1], 1e6
    elif s.endswith("B"):
        s, mult = s[:-1], 1e9
    s = s.rstrip("%")
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def e10(rows, frames):
    specs = (("E10 US non-farm payrolls", r"^10000_.*Nonfarm"), ("E10 US CPI m/m", r"^10030_.*Consumer_Price_Index_\(CPI\)_MoM"),
             ("E10 US retail sales m/m", r"^10130_.*Retail_Sales_MoM"))
    rows.append({"test": "E10 ISM manufacturing", "status": "untestable (not in the calendar dataset)"})
    for name, pat in specs:
        df = calendar_file(pat)
        if df is None:
            rows.append({"test": name, "status": "untestable (file missing)"})
            continue
        df["a"], df["f"] = df.Actual.map(num), df.Forecast.map(num)
        df = df.dropna(subset=["a", "f"])
        df = df[df.Timestamp <= END]
        s = (df.a - df.f).to_numpy()
        n = len(df)
        if n < 30:
            rows.append({"test": name, "status": "too few releases"})
            continue
        fitn = int(0.6 * n)
        sd = float(np.std(s[:fitn], ddof=1))
        z = s / sd if sd > 0 else np.zeros_like(s)
        for sym, sgn in (("USDJPY", +1), ("XAUUSD", -1)):
            bk = book(sym)
            if bk is None:
                continue
            out, fitmask = {}, []
            for k, (ts, zz) in enumerate(zip(df.Timestamp, z)):
                if abs(zz) < 1:
                    continue
                t0 = pd.Timestamp(ts).tz_localize("UTC")
                r = both(bk, t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(hours=4))
                if r is None:
                    continue
                d = r["long"] if sgn * zz > 0 else r["short"]
                out[t0.tz_convert(None)] = d
                fitmask.append(k < fitn)
            x = pd.Series(out).sort_index()
            rows.append(signed_result(f"{name}: {sym} +5 min -> +4 h in the surprise's direction (|z| >= 1)", sym, x,
                                      note=f"{n} releases 2017-2023; z by the fit-period sd", fit=np.array(fitmask)))


# ------------------------------------------------------------------------------------------------ E11
def e11(rows, frames):
    fs = raw("cftc_tff", "*.txt")
    if fs:
        D = []
        for f in fs:
            d = pd.read_csv(f, low_memory=False)
            d.columns = [c.strip() for c in d.columns]
            D.append(d)
        T = pd.concat(D, ignore_index=True)
        T["date"] = pd.to_datetime(T["Report_Date_as_YYYY-MM-DD"])
        T["code"] = T["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
        for code, sym, sgn, nm in (("097741", "USDJPY", -1, "yen"), ("099741", "EURUSD", +1, "euro")):
            g = T[T.code == code].drop_duplicates("date").set_index("date").sort_index()
            if len(g) < 200:
                rows.append({"test": f"E11a TFF leveraged funds {nm}", "status": "too few reports"})
                continue
            net = (pd.to_numeric(g.Lev_Money_Positions_Long_All) - pd.to_numeric(g.Lev_Money_Positions_Short_All)) / \
                pd.to_numeric(g.Open_Interest_All)
            z = (net - net.rolling(156, min_periods=52).mean()) / net.rolling(156, min_periods=52).std()
            bk = book(sym)
            if bk is None:
                continue
            ev = {}
            for d, zz in z.dropna().items():
                if abs(zz) <= 1 or d > END:
                    continue
                r = both(bk, utc(d + pd.Timedelta(days=6), 14), utc(d + pd.Timedelta(days=10), 14))
                if r is None:
                    continue
                # trend = position in the pair that follows leveraged funds' side of the currency
                trend_dir = sgn * (1 if zz > 0 else -1)
                ev[d] = {"long": r["long"] if trend_dir > 0 else r["short"],
                         "short": r["short"] if trend_dir > 0 else r["long"],
                         "gross": r["gross"] * trend_dir}
            f = frame(ev)
            frames[f"E11a_{nm}"] = f
            res = result(f"E11a TFF leveraged funds, {nm}: next week (Mon 14:00 -> Fri 14:00 UTC), |z| > 1", sym, f,
                         "train", note="'long' = follow leveraged funds; direction (follow or fade) from the fit period")
            res["direction"] = "follow funds" if res.get("direction") == "long" else "fade funds"
            rows.append(res)
    else:
        rows.append({"test": "E11a TFF", "status": "untestable (no files)"})
    rows.append({"test": "E11a TFF gold", "status": "untestable (gold is not in the TFF report)"})
    for kw, sym in (("yen", "USDJPY"), ("gold_price", "XAUUSD"), ("forex", "EURUSD"), ("dollar", "EURUSD")):
        fs = raw(f"trends_{kw}.csv")
        if not fs:
            rows.append({"test": f"E11b search interest {kw}", "status": "untestable"})
            continue
        s = pd.read_csv(fs[0], parse_dates=["date"]).set_index("date").iloc[:, 0].astype(float)
        ratio = s / s.shift(1).rolling(12, min_periods=12).mean()
        fitm = ratio.index <= TRAIN_END
        cut = float(ratio[fitm].quantile(0.9))
        spikes = ratio[ratio > cut].index
        bk = book(sym)
        if bk is None:
            continue
        bd = us_bdays()
        ev = {}
        for m in spikes:
            nm_ = m + pd.offsets.MonthBegin(1)
            days = bd[(bd.year == nm_.year) & (bd.month == nm_.month)]
            if len(days) < 10 or days[-1] > END:
                continue
            r = both(bk, utc(days[0], 14), utc(days[-1], 14))
            if r is not None:
                ev[nm_] = r
        f = frame(ev)
        rows.append(result(f"E11b search-interest spike ({kw}): the next month", sym, f, "train",
                           note=f"{len(spikes)} spikes above the fit 90th pct of value / 12-month mean"))


# ------------------------------------------------------------------------------------------------ E13
def e13(rows, frames):
    bk = book("USDJPY")
    if bk is None:
        return
    G, _, _ = F.calendar(bk)
    mv, ev = {}, {}
    for d in G:
        d = pd.Timestamp(d)
        if d > END:
            continue
        a = mid_at(bk, U.jst(d, 3, 0), 10)
        b = mid_at(bk, U.jst(d, 9, 50), 65, before=True)                      # DATA NOTE: 2007-2018: last quote before 01:00 UTC
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        r = bk.trade(U.jst(d, 9, 55), U.jst(d, 10, 30), -1, tol_min=10)        # 2007-2018: enters at 01:00 UTC (10:00 JST)
        if r is None:
            continue
        mv[d] = b / a - 1
        ev[d] = tnet(r)
    M, N = pd.Series(mv), pd.Series(ev)
    cut = float(M[M.index <= TRAIN_END].quantile(0.75))
    sel = N[M > cut]
    frames["E13"] = pd.DataFrame({"move": M, "net_short": N})
    rows.append(signed_result("E13 short USD/JPY 09:55 -> 10:30 JST after a large rise 03:00 -> 09:50 JST (Gotobi days)",
                              "USDJPY", sel, note=f"move above the fit 75th pct ({cut * 1e4:.1f} bp); all Gotobi days "
                              f"mean {N.mean() * 1e4:.2f} bp"))


# ------------------------------------------------------------------------------------------------ E14
def e14(rows, frames):
    def eod(sym, mv_from, mv_to, tr_from, tr_to):
        bk = book(sym)
        if bk is None:
            return None
        mv, ev = {}, {}
        for d in us_bdays("2007-01-01", "2025-12-31"):
            t0 = FL.local_ts(d, mv_from[0], *mv_from[1:])
            t1 = FL.local_ts(d, mv_to[0], *mv_to[1:])
            a, b = mid_at(bk, t0, 5), mid_at(bk, t1, 5)
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            r = both(bk, FL.local_ts(d, tr_from[0], *tr_from[1:]), FL.local_ts(d, tr_to[0], *tr_to[1:]))
            if r is None:
                continue
            mv[d] = b / a - 1
            ev[d] = r
        M, E = pd.Series(mv), frame(ev)
        cut = float(M[M.index <= TRAIN_END].abs().quantile(0.8))
        sel = M.abs() > cut
        net = pd.Series(np.where(M > 0, E.long, E.short), index=E.index)[sel]
        return net, cut
    g = eod("XAUUSD", ("Europe/London", 8, 0), ("America/New_York", 13, 0), ("America/New_York", 13, 0),
            ("America/New_York", 13, 30))
    if g is not None:
        frames["E14_gold"] = g[0].to_frame("net")
        rows.append(signed_result("E14 gold 13:00-13:30 NY following the 08:00 London -> 13:00 NY move (top 20%)", "XAUUSD",
                                  g[0], note=f"|move| > {g[1] * 1e4:.0f} bp"))
    pool = []
    for sym in ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"):
        x = eod(sym, ("Europe/London", 8, 0), ("America/New_York", 14, 30), ("America/New_York", 14, 30),
                ("America/New_York", 15, 0))
        if x is not None:
            pool.append(x[0])
    if pool:
        P = pd.concat(pool).sort_index()
        frames["E14_fx"] = P.to_frame("net")
        rows.append(signed_result("E14 FX 14:30-15:00 NY following the 08:00 London -> 14:30 NY move (top 20%), 7 majors pooled",
                                  "7 majors", P))


# ------------------------------------------------------------------------------------------------ E15
def e15(rows, frames):
    import holidays
    try:
        gc, gld = yahoo("GC=F", "2000-01-01"), yahoo("GLD", "2004-01-01")
        si = yahoo("SI=F", "2000-01-01")
        irx = yahoo("^IRX", "2000-01-01")
    except Exception as ex:                                                 # noqa: BLE001
        rows.append({"test": "E15", "status": f"untestable (Yahoo: {repr(ex)[:80]})"})
        return
    g = pd.concat([gc[gc.index < "2004-12-01"], gld[gld.index >= "2004-12-01"]]).sort_index()
    rf = (irx / 100 / 252).reindex(g.index).ffill().fillna(0)

    def window_ret(px, ev_dates, label):
        r = px.pct_change().dropna()
        ex = r - rf.reindex(r.index).ffill().fillna(0)
        out = {}
        for e in ev_dates:
            before = ex[ex.index < e]
            if len(before) < 20 or e > END:
                continue
            out[e] = float((1 + before.iloc[-15:]).prod() - 1) - 2e-4
        return pd.Series(out)
    dh = []
    for y in range(2000, 2026):
        for d, n in holidays.India(years=y).items():
            if "iwali" in n or "eepavali" in n:
                dh.append(pd.Timestamp(d) - pd.Timedelta(days=2))
    x = window_ret(g, sorted(set(dh)), "Dhanteras")
    frames["E15_dhanteras"] = x.to_frame("ret")
    rows.append(signed_result("E15 gold over the 15 trading days before Dhanteras (Diwali - 2 days)", "gold (GC=F / GLD)",
                              x, note="excess return minus 2 bp; 2000-2018 fit / 2019-2025 check",
                              fit=np.asarray(x.index <= TRAIN_END)))
    rows.append({"test": "E15 gold before Akshaya Tritiya", "status": "untestable (not in the holidays package)"})
    try:
        from lunardate import LunarDate
        cny = [pd.Timestamp(LunarDate(y, 1, 1).toSolarDate()) for y in range(2001, 2026)]
        s = window_ret(si, cny, "CNY")
        rows.append(signed_result("E15 silver over the 15 trading days before the Chinese New Year", "silver (SI=F)", s,
                                  note="excess return minus 2 bp", fit=np.asarray(s.index <= TRAIN_END)))
    except ImportError:
        rows.append({"test": "E15 silver before the Chinese New Year", "status": "untestable (lunardate missing)"})


# ------------------------------------------------------------------------------------------------ E16 / E24
def month_starts(bd):
    s = pd.Series(bd, index=bd)
    return pd.DatetimeIndex(s.groupby([bd.year, bd.month]).min().values)


def vix_month_end():
    try:
        v = yahoo("^VIX", "2006-01-01")
        return v.resample("ME").last()
    except Exception:                                                       # noqa: BLE001
        return pd.Series(dtype=float)


def monthly_pair_trades(sym, starts, carry=None):
    """Long and short nets of holding `sym` from each month start 14:00 UTC to the next; carry: annual rate
    differential (long base) Series by month for pairs without stored rates (swap then = carry - mark-up)."""
    bk = book(sym)
    if bk is None:
        return None
    out = {}
    for a, b in zip(starts[:-1], starts[1:]):
        if b > END + pd.Timedelta(days=31):
            continue
        ra, rb = bk.trade(utc(a, 14), utc(b, 14), +1, 30), bk.trade(utc(a, 14), utc(b, 14), -1, 30)
        if ra is None or rb is None:
            continue
        if carry is None:
            out[a] = {"long": tnet(ra), "short": tnet(rb)}
        else:
            c = carry.asof(a) if len(carry) else np.nan
            if not np.isfinite(c):
                continue
            from fxlab.data.rates import SWAP_MARKUP_PA
            n = ra[4]
            out[a] = {"long": ra[0] + n * (c - SWAP_MARKUP_PA) / 365, "short": rb[0] + n * (-c - SWAP_MARKUP_PA) / 365}
    return frame(out)


def e24(rows, frames):
    from fxlab.data.rates import known_at
    bd = us_bdays("2006-12-01", "2026-01-31")
    starts = month_starts(bd)
    pairs = {"EURUSD": ("EUR", 1), "GBPUSD": ("GBP", 1), "AUDUSD": ("AUD", 1), "NZDUSD": ("NZD", 1),
             "USDJPY": ("JPY", -1), "USDCHF": ("CHF", -1), "USDCAD": ("CAD", -1)}
    K = known_at(pd.DatetimeIndex(starts).tz_localize("UTC"))
    K.index = K.index.tz_convert(None)
    vix = vix_month_end()
    port = {}
    tr = {s: monthly_pair_trades(s, starts) for s in pairs}
    for a in starts[:-1]:
        r = K.loc[a].dropna().sort_values()
        if len(r) < 8:
            continue
        lo, hi = set(r.index[:2]), set(r.index[-2:])
        tot, ok = 0.0, True
        for s, (c, sg) in pairs.items():
            w = 0.5 if c in hi else (-0.5 if c in lo else 0.0)
            if w == 0:
                continue
            f = tr[s]
            if f is None or a not in f.index:
                ok = False
                break
            d = w * sg
            tot += abs(d) * (f.long[a] if d > 0 else f.short[a])
        if ok:
            port[a] = tot
    P = pd.Series(port, dtype=float).sort_index()
    P.index = pd.DatetimeIndex(P.index)
    if len(P) < 10:
        rows.append({"test": "E24 G10 carry", "status": f"too few months ({len(P)}); pairs with bars: "
                     f"{sorted(k for k, v in tr.items() if v is not None)}"})
        return
    frames["E24"] = P.to_frame("net")
    fit = np.asarray(P.index <= TRAIN_END)
    rows.append(signed_result("E24 G10 carry factor (long top 2 / short bottom 2 by 3-month rate), monthly", "7 USD pairs", P,
                              note="real swaps", fit=fit))
    if len(vix):
        prev = vix.shift(0)
        keep = [a for a in P.index if (prev[prev.index < a].iloc[-1] if (prev.index < a).any() else 0) <= 25]
        Pv = P.reindex(keep)
        rows.append(signed_result("E24 G10 carry, flat after a month-end VIX above 25", "7 USD pairs", Pv,
                                  fit=np.asarray(Pv.index <= TRAIN_END)))


def e16(rows, frames):
    bd = us_bdays("2006-12-01", "2026-01-31")
    starts = month_starts(bd)
    ids = {"MXN": "IR3TIB01MXM156N", "ZAR": "IR3TIB01ZAM156N", "CNH": "IR3TIB01CNM156N", "SGD": "IR3TIB01SGM156N"}
    us = fred_csv("IR3TIB01USM156N") / 100
    rates = {}
    for c, sid in ids.items():
        s = fred_csv(sid) / 100
        if len(s) > 24:
            s.index = s.index + pd.DateOffset(months=1)                      # month m known from m+1
            rates[c] = s
    rows.append({"test": "E16 TRY carry", "status": "untestable (no USD/TRY bars; only TRY/JPY was downloaded)"})
    if len(rates) < 3 or len(us) < 24:
        rows.append({"test": "E16 EM carry", "status": f"untestable (rates for {sorted(rates)})"})
        return
    us.index = us.index + pd.DateOffset(months=1)
    trades = {}
    for c in rates:
        diff = (us.reindex(us.index.union(rates[c].index)).ffill() - rates[c].reindex(us.index.union(rates[c].index)).ffill())
        trades[c] = monthly_pair_trades("USD" + c, starts, carry=diff.dropna())
    vix = vix_month_end()
    port = {}
    for a in starts[:-1]:
        avail = {c: rates[c].asof(a) for c in rates if trades[c] is not None and a in trades[c].index}
        avail = {c: v for c, v in avail.items() if np.isfinite(v)}
        if len(avail) < 3:
            continue
        top = sorted(avail, key=avail.get)[-2:]
        v = vix[vix.index < a]
        if len(v) and v.iloc[-1] > 25:
            port[a] = 0.0
            continue
        port[a] = float(np.mean([trades[c].short[a] for c in top]))         # short USDxxx = long the EM currency
    P = pd.Series(port, dtype=float).sort_index()
    P.index = pd.DatetimeIndex(P.index)
    frames["E16"] = P.to_frame("net")
    rows.append(signed_result("E16 EM carry: long the 2 highest-yielding of MXN, ZAR, CNH, SGD vs USD, flat after VIX > 25",
                              "USDMXN/USDZAR/USDCNH/USDSGD", P,
                              note=f"currencies with rates: {sorted(rates)}; swap = rate differential - 0.75% a year"))


# ------------------------------------------------------------------------------------------------ E18
def e18(rows, frames):
    specs = (("EURUSD", 1e-4), ("GBPUSD", 1e-4), ("USDJPY", 0.01), ("XAUUSD", None))
    for sym, pip in specs:
        bk = book(sym)
        if bk is None:
            continue
        if pip is None:
            step, off, tgt = 50.0, 2.0, 5.0                                  # DATA NOTE: gold target / stop read as $5
        else:
            step, off, tgt = 100 * pip, 2 * pip, 10 * pip
        idx = bk.idx
        out = {}
        days = pd.bdate_range(str(idx[0].date()), str(min(idx[-1], pd.Timestamp(END).tz_localize("UTC")).date()))
        for d in days:
            a, e = int(idx.searchsorted(utc(d, 0))), int(idx.searchsorted(utc(d, 21)))
            if e - a < 300:
                continue
            o = 0.5 * (bk.bo[a] + bk.ao[a])
            res = []
            for side in (+1, -1):
                if side > 0:
                    lim = np.floor(o / step) * step + off
                    if bk.ao[a] <= lim:
                        continue
                    f = np.flatnonzero(bk.al[a + 1:e] <= lim)
                else:
                    lim = np.ceil(o / step) * step - off
                    if bk.bo[a] >= lim:
                        continue
                    f = np.flatnonzero(bk.bh[a + 1:e] >= lim)
                if not len(f):
                    continue
                k = a + 1 + int(f[0])
                E = min(lim, bk.ao[k]) if side > 0 else max(lim, bk.bo[k])
                T, S = E + side * tgt, E - side * tgt
                X = None
                for q in range(k, e):
                    if side > 0:
                        if bk.bl[q] <= S:
                            X = min(S, bk.bo[q]) - R.slip(bk, q) if q > k else S - R.slip(bk, q)
                            break
                        if bk.bh[q] >= T and q > k:
                            X = T
                            break
                    else:
                        if bk.ah[q] >= S:
                            X = max(S, bk.ao[q]) + R.slip(bk, q) if q > k else S + R.slip(bk, q)
                            break
                        if bk.al[q] <= T and q > k:
                            X = T
                            break
                if X is None:
                    X = bk.bo[e] - R.slip(bk, e) if side > 0 else bk.ao[e] + R.slip(bk, e)
                res.append(side * (X / E - 1.0) - 2 * bk.comm)
            if res:
                out[d] = float(np.sum(res))
        x = pd.Series(out)
        frames[f"E18_{sym}"] = x.to_frame("net")
        rows.append(signed_result("E18 round-number limit orders (2 pips inside a 100-pip level, 10 / 10, out at 21:00 UTC)",
                                  sym, x, note="per day, both sides summed; limit fills only where the quote reaches the price"))


# ------------------------------------------------------------------------------------------------ E19
def night_fade_after_rollover(sym):
    """I7 with entries 23:00-02:00 UTC (Mon-Thu nights) and the time exit at 02:00 UTC."""
    bk = book(sym)
    if bk is None:
        return None
    m = bk.m
    c1 = pd.Series(0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy()), index=bk.idx)
    c5s = c1.resample("5min", label="left", closed="left").last().dropna()
    c5 = c5s.to_numpy()
    ma = c5s.rolling(20).mean().to_numpy()
    sd = c5s.rolling(20).std().to_numpy()
    up, lo = ma + 2 * sd, ma - 2 * sd
    t5 = c5s.index
    hrs, wds = t5.hour.to_numpy(), t5.weekday.to_numpy()
    in_win = ((hrs == 23) & (wds <= 3)) | ((hrs <= 1) & (wds >= 1) & (wds <= 4))
    gap_next = np.r_[np.asarray((t5[1:] - t5[:-1]) > pd.Timedelta(minutes=30)), True]
    out = {}
    n, N = 20, len(t5) - 1
    while n < N:
        if not in_win[n] or not np.isfinite(up[n]) or gap_next[n]:
            n += 1
            continue
        side = -1 if c5[n] > up[n] else (+1 if c5[n] < lo[n] else 0)
        if side == 0:
            n += 1
            continue
        t_in = t5[n + 1]
        i = bk.at(t_in, 5)
        day0 = t5[n].normalize()
        lim = (day0 + pd.Timedelta(days=1) if hrs[n] >= 23 else day0) + pd.Timedelta(hours=2)
        if i < 0 or t_in >= lim:
            n += 1
            continue
        k, t_out = n + 1, lim
        while k < N and t5[k] < lim:
            crossed = (c5[k] <= ma[k]) if side < 0 else (c5[k] >= ma[k])
            if crossed or gap_next[k]:
                t_out = min(t5[k + 1], lim)
                break
            k += 1
        j = U.first_after(bk, t_out, 10)
        if j <= i:
            n += 1
            continue
        r = bk.bracket(i, j, side, 1e9, float(up[n] - lo[n]))
        if r is None:
            n += 1
            continue
        out[t_in] = U.net_of(r)
        n = max(n + 1, int(t5.searchsorted(bk.idx[j], side="right")) - 1)
    s = pd.Series(out)
    s.index = pd.DatetimeIndex(s.index).tz_convert(None)
    return s[s.index <= END]


def e19(rows, frames):
    for sym in ("AUDNZD", "EURGBP", "GBPCAD"):
        x = night_fade_after_rollover(sym)
        if x is None:
            continue
        frames[f"E19_{sym}"] = x.to_frame("net")
        rows.append(signed_result("E19 night Bollinger fade, entries 23:00-02:00 UTC (after the rollover)", sym, x,
                                  note="DATA NOTE: 2007-2018 files have no 00:00-01:00 UTC bars (positions close at the gap)"))


# ------------------------------------------------------------------------------------------------ E23
CCY = ("USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF")
ORDER = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")           # market convention for the pair names


def pair_of(a, b):
    """(symbol, sign): sign +1 if long `symbol` = long a / short b."""
    s = a + b if ORDER.index(a) < ORDER.index(b) else b + a
    return s, (1 if s.startswith(a) else -1)


def e23_pair(sym, out):
    """Per-pair daily table for E23: mid at 21:00 UTC and the long / short net of 21:00 -> 21:00 next business day."""
    bd = us_bdays("2007-01-01", "2026-01-15")
    bk = book(sym)
    if bk is None:
        return
    m, L = {}, {}
    for d0, d1 in zip(bd[:-1], bd[1:]):
        if d0 > END:
            break
        x = mid_at(bk, utc(d0, 21), 10)
        if np.isfinite(x):
            m[d0] = x
        r = both(bk, utc(d0, 21), utc(d1, 21), 10)
        if r is not None:
            L[d0] = r
    f = frame(L)
    f["mid"] = pd.Series(m)
    f.to_csv(os.path.join(out, f"e23_{sym}.csv"))
    print("E23", sym, len(m), flush=True)


def e23(rows, frames):
    fs = glob.glob(os.path.join(OUT, "**", "e23_*.csv"), recursive=True)
    mids, legs = {}, {}
    for f in fs:
        s = os.path.basename(f)[4:10]
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        mids[s], legs[s] = d["mid"].dropna(), d[["long", "short"]].dropna()
    if len(mids) < 20:
        rows.append({"test": "E23 currency strength", "status": f"only {len(mids)} pairs"})
        return
    M = pd.DataFrame(mids).sort_index()
    for look in (5, 20):
        ret = np.log(M).diff(look)
        strength = pd.DataFrame(0.0, index=M.index, columns=list(CCY))
        cnt = pd.DataFrame(0.0, index=M.index, columns=list(CCY))
        for s in M.columns:
            a, b = s[:3], s[3:]
            v = ret[s]
            ok = v.notna()
            strength.loc[ok, a] += v[ok]
            strength.loc[ok, b] -= v[ok]
            cnt.loc[ok, a] += 1
            cnt.loc[ok, b] += 1
        strength = strength / cnt.replace(0, np.nan)
        for rev in (False, True):
            out = {}
            for d, row in strength.dropna(thresh=6).iterrows():
                row = row.dropna()
                hi, lo = row.idxmax(), row.idxmin()
                a, b = (lo, hi) if rev else (hi, lo)
                s, sg = pair_of(a, b)
                f = legs.get(s)
                if f is None or d not in f.index:
                    continue
                out[d] = f.long[d] if sg > 0 else f.short[d]
            x = pd.Series(out)
            nm = f"E23 currency strength, {look}-day ranking, {'weakest vs strongest' if rev else 'strongest vs weakest'}"
            frames[f"E23_{look}_{int(rev)}"] = x.to_frame("net")
            rows.append(signed_result(nm, f"{len(mids)} pairs", x, note="held 21:00 -> 21:00 UTC next business day"))


# ------------------------------------------------------------------------------------------------ aggregate
def aggregate(d, out):
    fs = glob.glob(os.path.join(d, "**", "rows_*.json"), recursive=True)
    rows = []
    for f in fs:
        rows += json.load(open(f))
    R_ = pd.DataFrame(rows)
    if "p_for_bh" in R_:
        p = pd.to_numeric(R_.p_for_bh, errors="coerce")
        ok = p.notna()
        q = pd.Series(np.nan, index=R_.index)
        if ok.sum():
            from statsmodels.stats.multitest import multipletests
            q[ok] = multipletests(p[ok], method="fdr_bh")[1]
        R_["bh_q"] = q
        R_["counts_after_bh"] = R_.get("counts", False).fillna(False).astype(bool) & (q <= 0.10)
    os.makedirs(out, exist_ok=True)
    R_.to_csv(os.path.join(out, "edges_batch.csv"), index=False)
    cols = [c for c in ("test", "instrument", "status", "direction", "split", "fit_n", "fit_net_bp", "fit_t", "check_n",
                        "check_net_bp", "check_t", "check_win", "fit_rho", "check_rho", "bh_q", "counts_after_bh", "note")
            if c in R_]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(R_[cols].round(3).to_string())


TESTS = {"E2": e2, "E3": e3, "E4": e4, "E6": e6, "E7": e7, "E8": e8, "E9": e9, "E10": e10, "E11": e11, "E13": e13,
         "E14": e14, "E15": e15, "E16": e16, "E18": e18, "E19": e19, "E23": e23, "E24": e24}


def main():
    global RAW, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", default=[])
    ap.add_argument("--raw", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--aggregate", default="")
    ap.add_argument("--pair", default="")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate, a.out)
        return
    RAW, OUT = a.raw, a.out
    os.makedirs(a.out, exist_ok=True)
    if a.pair:
        e23_pair(a.pair, a.out)
        return
    for t in a.tests:
        rows, frames = [], {}
        try:
            TESTS[t](rows, frames)
        except Exception:                                                   # noqa: BLE001
            import traceback
            rows.append({"test": t, "status": "error", "error": traceback.format_exc()[-1500:]})
        for r in rows:
            print(json.dumps(r, default=float), flush=True)
        json.dump(rows, open(os.path.join(a.out, f"rows_{t}.json"), "w"), indent=1, default=float)
        for k, f in frames.items():
            f.to_csv(os.path.join(a.out, f"events_{k}.csv"))


if __name__ == "__main__":
    main()
