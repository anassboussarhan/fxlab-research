"""P batch of docs/push_preregistration.md: the leads tested on years and pairs never looked at. 2026 is never read.

--p34  P3 (frozen Gotobi basket B0 on 2003-2006 Dukascopy M1) and P4 (post-fix reversal: P4a five yen crosses on
       2019-2025, P4b USD/JPY + EUR/JPY on 2003-2006). Early files: EARLY/{PAIR}/*.parquet (Dukascopy candles).
--p1   P1: 2-year yield gaps on GBP/USD, USD/CAD, AUD/USD, USD/CHF, NZD/USD (STORE = the flow store).
--p2   P2: gold (and silver) before the Chinese New Year, 1975-2000.
--combine  BH over the primary rows of every part found in OUT.
Usage: python scripts/push_lab.py --p34 --early data/m1_early --out OUT
       python scripts/push_lab.py --p1 --p2 --store flow_store --out OUT
       python scripts/push_lab.py --combine --out OUT"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
LOG = {}


def tstat(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def month_t(x):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 10:
        return np.nan
    e = x - x.mean()
    g = e.groupby(pd.DatetimeIndex(x.index).to_period("M")).sum()
    m = len(g)
    se = math.sqrt((g ** 2).sum()) / len(x) * math.sqrt(m / max(m - 1, 1))
    return float(x.mean() / se) if se > 0 else np.nan


def summary(x, cluster=True):
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": month_t(x) if cluster else tstat(x),
            "t_iid": tstat(x), "win": float((x > 0).mean()),
            "by_year_bp": {int(k): round(float(v * 1e4), 2) for k, v in x.groupby(pd.DatetimeIndex(x.index).year).mean().items()}}


# ------------------------------------------------------------------------------------------- P3 / P4
def early_book(early, pair):
    import polars as pl
    import flows_lab as FL
    fs = sorted(glob.glob(os.path.join(early, pair, "*.parquet")))
    if not fs:
        return None
    df = pl.concat([pl.read_parquet(f) for f in fs]).unique(subset=["ts"]).sort("ts").to_pandas()
    df = df.set_index(pd.DatetimeIndex(df.ts).tz_convert("UTC") if pd.DatetimeIndex(df.ts).tz is not None
                      else pd.DatetimeIndex(df.ts).tz_localize("UTC"))
    m = df[["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]].dropna()
    m = m[(m.ask_o >= m.bid_o) & (m.bid_o > 0) & (m.index < pd.Timestamp("2007-01-01", tz="UTC"))]
    LOG.setdefault("early_coverage", {})[pair] = {int(k): int(v) for k, v in m.groupby(m.index.year).size().items()}
    bk = FL.Book(m, 0.01, None)
    bk._hl()
    return bk


def m1_book(pair):
    import flows_lab as FL
    import untested_intraday_lab as U
    m = U.load(pair)
    if m is None:
        return None
    bk = FL.Book(m, 0.01, pair)
    bk._hl()
    return bk


def b0_nights(bk, days):
    """Per night: fill-based B0 leg (spread + slippage, 80 bp stop, before commission and swap), mid gross, the
    quoted spread at entry (bp)."""
    import flow_edges_lab as F
    import prop_checks as P
    rows = {}
    for d in days:
        i, j = F.window(bk, d)
        if i < 0 or not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all():
            continue
        mi, mj = 0.5 * (bk.bo[i] + bk.ao[i]), 0.5 * (bk.bo[j] + bk.ao[j])
        rows[pd.Timestamp(d)] = {"fill": P.leg(bk, i, j)[1], "mid": mj / mi - 1.0,
                                 "spread_bp": (bk.ao[i] - bk.bo[i]) / mi * 1e4}
    return pd.DataFrame.from_dict(rows, orient="index")


def postfix(bk, days):
    """Short 09:55 -> 10:20 Tokyo; net with commission (no rollover inside the window)."""
    import untested_intraday_lab as U
    out = {}
    for d in days:
        r = bk.trade(U.jst(pd.Timestamp(d), 9, 55), U.jst(pd.Timestamp(d), 10, 20), -1, tol_min=5)
        if r is not None:
            out[pd.Timestamp(d)] = U.net_of(r)
    return pd.Series(out, dtype=float)


def p34(a):
    import flow_edges_lab as F
    res = {"P3": {}, "P4b": {}, "P4a": {}, "reference_2019_2025": {}}
    nights, ctrl, pf = {}, {}, {}
    for pair in ("USDJPY", "EURJPY"):
        bk = early_book(a.early, pair)
        if bk is None:
            res["P3"][pair] = {"status": "data missing"}
            continue
        G, C, _ = F.calendar(bk)
        G = [d for d in G if pd.Timestamp(d) < pd.Timestamp("2007-01-01")]
        C = [d for d in C if pd.Timestamp(d) < pd.Timestamp("2007-01-01")]
        nights[pair], ctrl[pair] = b0_nights(bk, G), b0_nights(bk, C)
        pf[pair] = postfix(bk, G)
        res["P3"][pair] = {"gotobi_fill": summary(nights[pair]["fill"], False) if len(nights[pair]) else {"n": 0},
                           "gotobi_mid": summary(nights[pair]["mid"], False) if len(nights[pair]) else {"n": 0},
                           "control_fill": summary(ctrl[pair]["fill"], False) if len(ctrl[pair]) else {"n": 0},
                           "entry_spread_bp_median": float(nights[pair]["spread_bp"].median()) if len(nights[pair]) else None}
    if len(nights) == 2:
        ix = nights["USDJPY"].index.intersection(nights["EURJPY"].index)
        bas = (nights["USDJPY"].loc[ix] + nights["EURJPY"].loc[ix]) / 2
        cx = ctrl["USDJPY"].index.intersection(ctrl["EURJPY"].index)
        cb = (ctrl["USDJPY"].loc[cx] + ctrl["EURJPY"].loc[cx]) / 2
        s = summary(bas["fill"], False)
        res["P3"]["basket"] = {"fill": s, "mid": summary(bas["mid"], False), "control_fill": summary(cb["fill"], False),
                               "gotobi_minus_control_bp": float((bas["fill"].mean() - cb["fill"].mean()) * 1e4),
                               "confirms": bool(s["n"] > 2 and s["mean_bp"] > 0 and (s["t_iid"] or 0) >= 1.645)}
        bas.to_csv(os.path.join(a.out, "p3_basket_nights.csv"))
    if pf:
        P4b = pd.concat(pf, axis=1).mean(axis=1)
        res["P4b"] = {"pooled": summary(P4b), **{p: summary(v) for p, v in pf.items()}}
    # P4a: the five other crosses on 2019-2025, and the two seen pairs for reference
    cr = {}
    for pair in ("GBPJPY", "AUDJPY", "CHFJPY", "CADJPY", "NZDJPY", "USDJPY", "EURJPY"):
        bk = m1_book(pair)
        if bk is None:
            (res["P4a"] if pair not in ("USDJPY", "EURJPY") else res["reference_2019_2025"])[pair] = {"status": "data missing"}
            continue
        G, _, _ = F.calendar(bk)
        G = [d for d in G if pd.Timestamp("2019-01-01") <= pd.Timestamp(d) <= pd.Timestamp("2025-12-31")]
        s = postfix(bk, G)
        if pair in ("USDJPY", "EURJPY"):
            res["reference_2019_2025"][pair] = summary(s)
        else:
            cr[pair] = s
            res["P4a"][pair] = summary(s)
        del bk
    if cr:
        res["P4a"]["pooled"] = summary(pd.concat(cr, axis=1).mean(axis=1))
    json.dump(res, open(os.path.join(a.out, "p34.json"), "w"), indent=1, default=str)
    return res


# ------------------------------------------------------------------------------------------- P1
def find(store, *pat):
    return sorted(glob.glob(os.path.join(store, "**", *pat), recursive=True))


def foreign_2y(store):
    """Daily foreign 2-year yields from the push probe files; a country that cannot be read is left out (logged)."""
    out, err = {}, {}
    # Canada: Bank of Canada Valet JSON
    try:
        j = json.load(open(find(store, "push", "yields", "boc_2y.json")[0]))
        s = pd.Series({pd.Timestamp(o["d"]): float(o["BD.CDN.2YR.DQ.YLD"]["v"]) for o in j["observations"]
                       if o.get("BD.CDN.2YR.DQ.YLD", {}).get("v") not in (None, "")})
        out["CA"] = s.sort_index()
    except Exception as ex:                                                 # noqa: BLE001
        err["CA"] = repr(ex)[:200]
    # Australia: RBA F2 daily (historical .xls to 2013 + current .xlsx); column FCMYGBAG2 = 2-year AGB
    try:
        parts = []
        for f in find(store, "push", "yields", "rba_f02d*"):
            X = pd.read_excel(f, header=None)
            hdr = next(i for i in range(min(30, len(X))) if (X.iloc[i].astype(str) == "Series ID").any())
            ids = X.iloc[hdr].astype(str).tolist()
            c = ids.index("FCMYGBAG2")
            body = X.iloc[hdr + 1:, [0, c]].dropna()
            parts.append(pd.Series(pd.to_numeric(body.iloc[:, 1], errors="coerce").values,
                                   index=pd.to_datetime(body.iloc[:, 0], errors="coerce")).dropna())
        s = pd.concat(parts)
        out["AU"] = s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as ex:                                                 # noqa: BLE001
        err["AU"] = repr(ex)[:200]
    # UK: Bank of England nominal curve archive (zip of xlsx, sheet with the spot curve; the 2-year column)
    try:
        parts = []
        for f in find(store, "push", "yields", "boe_*.zip"):
            z = zipfile.ZipFile(f)
            for nm in z.namelist():
                if "nominal" not in nm.lower() or not nm.lower().endswith((".xlsx", ".xls")):
                    continue
                yrs = [int(y) for y in re.findall(r"(19\d\d|20\d\d)", nm)]
                if yrs and max(yrs) < 2005:                                 # the z needs 250 days before 2007
                    continue
                xl = pd.ExcelFile(io.BytesIO(z.read(nm)))
                sh = next((k for k in xl.sheet_names if "spot" in k.lower() and "short" not in k.lower()), None) or \
                    next((k for k in xl.sheet_names if "spot" in k.lower()), None)
                if sh is None:
                    continue
                X = xl.parse(sh, header=None)
                hr = next(i for i in range(min(10, len(X))) if pd.to_numeric(X.iloc[i, 1:], errors="coerce").notna().sum() > 5)
                mats = pd.to_numeric(X.iloc[hr, 1:], errors="coerce")
                col = int(np.flatnonzero(np.isclose(mats.values.astype(float), 2.0))[0]) + 1
                body = X.iloc[hr + 1:, [0, col]]
                parts.append(pd.Series(pd.to_numeric(body.iloc[:, 1], errors="coerce").values,
                                       index=pd.to_datetime(body.iloc[:, 0], errors="coerce")).dropna())
        s = pd.concat(parts)
        out["UK"] = s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as ex:                                                 # noqa: BLE001
        err["UK"] = repr(ex)[:200]
    # Switzerland: SNB cube (csv; the 2-year row, daily only)
    try:
        f = find(store, "push", "yields", "snb_rendoblid.csv")[0]
        raw = open(f, encoding="utf-8", errors="ignore").read().splitlines()
        hi = next(i for i, l in enumerate(raw) if l.lower().startswith('"date"') or l.lower().startswith("date"))
        d = pd.read_csv(io.StringIO("\n".join(raw[hi:])), sep=";")
        d.columns = [c.strip('"') for c in d.columns]
        dim = [c for c in d.columns if c not in ("Date", "Value")]
        k2 = d[dim[0]].astype(str).str.contains(r"(^|\D)2J|2Y|2 ?year|_2$", regex=True)
        s = d[k2]
        s = pd.Series(pd.to_numeric(s.Value, errors="coerce").values, index=pd.to_datetime(s.Date, errors="coerce")).dropna()
        if len(s) < 1000 or s.index.to_series().diff().dt.days.median() > 5:
            raise RuntimeError(f"not a daily 2-year series ({len(s)} rows; keys {d[dim[0]].unique()[:8].tolist()})")
        out["CH"] = s.sort_index()
    except Exception as ex:                                                 # noqa: BLE001
        err["CH"] = repr(ex)[:300]
    # New Zealand: RBNZ B2 daily (column with "2 year" government bond)
    try:
        parts = []
        for f in find(store, "push", "yields", "rbnz_hb2*"):
            X = pd.read_excel(f, header=None)
            r = next(i for i in range(min(10, len(X))) if X.iloc[i].astype(str).str.contains("Series Id", case=False).any())
            ids = X.iloc[r].astype(str).tolist()
            c = next(k for k, v in enumerate(ids) if v.strip().upper() in ("INM.DG102.N", "INM.DG102.NZ.Z"))
            body = X.iloc[r + 1:, [0, c]]
            parts.append(pd.Series(pd.to_numeric(body.iloc[:, 1], errors="coerce").values,
                                   index=pd.to_datetime(body.iloc[:, 0], errors="coerce")).dropna())
        s = pd.concat(parts)
        out["NZ"] = s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as ex:                                                 # noqa: BLE001
        err["NZ"] = repr(ex)[:200]
    LOG["foreign_2y"] = {k: [str(v.index.min().date()), str(v.index.max().date()), int(len(v))] for k, v in out.items()}
    LOG["foreign_2y_errors"] = err
    return out


def p1(a):
    import flow_volume_lab as V
    US = V.yields(a.store).get("US")
    F2 = foreign_2y(a.store)
    pairs = {"UK": ("GBPUSD", -1), "CA": ("USDCAD", 1), "AU": ("AUDUSD", -1), "CH": ("USDCHF", 1), "NZ": ("NZDUSD", -1)}
    res, per = {}, {1: {}, 5: {}}
    if US is None:
        return {"status": "US yields missing"}
    for c, (sym, usd_side) in pairs.items():
        if c not in F2:
            res[sym] = {"status": "data missing", "error": LOG["foreign_2y_errors"].get(c)}
            continue
        gap = (US - F2[c]).dropna()
        c5 = gap.diff(5)
        z = (c5 - c5.rolling(250, min_periods=150).mean().shift(1)) / c5.rolling(250, min_periods=150).std().shift(1)
        for n in (1, 5):
            rows = {}
            for dd, zz in z[z.abs() >= 1.5].items():
                if dd > pd.Timestamp("2025-12-31") or dd < pd.Timestamp("2007-01-01"):
                    continue
                r = V.daily_trade(sym, dd, int(np.sign(zz) * usd_side), n)      # follow, fixed in advance
                if r is not None:
                    rows[pd.Timestamp(dd)] = r[0]
            per[n][sym] = pd.Series(rows, dtype=float)
    # one market book in memory at a time: daily_trade loads books lazily (two-pass not needed: one pair per loop)
    for n in (1, 5):
        for sym, s in per[n].items():
            res.setdefault(sym, {})[f"{n}d"] = {"all": summary(s), "2007-2018": summary(s[s.index.year <= 2018]),
                                                "2019-2025": summary(s[s.index.year >= 2019])}
        if per[n]:
            pool = pd.concat(per[n], axis=1).mean(axis=1)
            res[f"pooled_{n}d"] = {"all": summary(pool), "2007-2018": summary(pool[pool.index.year <= 2018]),
                                   "2019-2025": summary(pool[pool.index.year >= 2019])}
            pool.to_csv(os.path.join(a.out, f"p1_pooled_{n}d.csv"))
    p = res.get("pooled_5d")
    if p:
        p["counts_own_rule"] = bool(p["all"].get("mean_bp", -1) > 0 and (p["all"].get("t") or 0) >= 1.645
                                    and p["2007-2018"].get("mean_bp", -1) > 0 and p["2019-2025"].get("mean_bp", -1) > 0)
    json.dump(res, open(os.path.join(a.out, "p1.json"), "w"), indent=1, default=str)
    return res


# ------------------------------------------------------------------------------------------- P2
def bbk(store, names):
    """The London fixing as republished by the Bundesbank (BBEX3): any CSV with date, value rows."""
    for nm in names:
        for f in find(store, "push", "metals", nm):
            rows = {}
            for l in open(f, encoding="utf-8", errors="ignore"):
                m = re.match(r'\s*"?(\d{4}-\d{2}-\d{2})"?\s*[,;]\s*"?([0-9]+(?:\.[0-9]+)?)"?', l)
                if m:
                    rows[pd.Timestamp(m.group(1))] = float(m.group(2))
            if len(rows) > 5000:
                LOG.setdefault("metal_source", {})[nm] = len(rows)
                s = pd.Series(rows).sort_index()
                return s[s > 0]
    return None


def lbma(store, name):
    alt = {"lbma_gold_pm.json": ("bbk_gold_pm.csv", "bbk_gold_pm_its.csv"),
           "lbma_silver.json": ("bbk_silver.csv", "bbk_silver_its.csv")}[name]
    f = find(store, "push", "metals", name)
    if not f:
        return bbk(store, alt)
    j = json.load(open(f[0]))
    s = pd.Series({pd.Timestamp(r["d"]): (r["v"][0] if isinstance(r["v"], list) else r["v"]) for r in j
                   if r.get("v") and (r["v"][0] if isinstance(r["v"], list) else r["v"])}, dtype=float)
    return s[s > 0].sort_index()


def french_rf(store):
    f = find(store, "push", "rates", "ff_factors.zip")
    if not f:
        return None
    z = zipfile.ZipFile(f[0])
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    rows = []
    for l in txt:
        m = re.match(r"\s*(\d{6})\s*,([^,]*),([^,]*),([^,]*),\s*([-\d\.]+)\s*$", l)
        if m:
            rows.append((pd.Period(m.group(1)[:4] + "-" + m.group(1)[4:], "M"), float(m.group(5)) / 100))
        elif rows and re.match(r"\s*Annual", l):
            break
    return pd.Series(dict(rows))


def p2(a):
    from lunardate import LunarDate
    rf = french_rf(a.store)
    res = {}
    for nm, fn in (("gold", "lbma_gold_pm.json"), ("silver", "lbma_silver.json")):
        px = lbma(a.store, fn)
        if px is None and nm == "gold":
            px = boe_gold(a.store)
        if px is None or rf is None:
            res[nm] = {"status": "data missing", "rf": rf is not None}
            continue
        r = px.pct_change().dropna()
        out = {}
        for y in range(1975, 2026):
            cny = pd.Timestamp(LunarDate(y, 1, 1).toSolarDate())
            before = r[r.index < cny]
            if len(before) < 20 or before.index[-1] < cny - pd.Timedelta(days=10):
                continue
            w = before.iloc[-15:]
            rfd = np.array([rf.get(pd.Period(t, "M"), np.nan) / 21 for t in w.index])
            if not np.isfinite(rfd).all():
                continue
            out[cny] = float((1 + w).prod() - 1) - float(np.sum(rfd)) - 2e-4
        s = pd.Series(out)
        pre, post = s[s.index.year <= 2000], s[s.index.year >= 2001]
        res[nm] = {"1975-2000": summary(pre, False), "2001-2025 (D6 sample, reference)": summary(post, False),
                   "1975-2025": summary(s, False), "first_price": str(px.index.min().date())}
        s.to_csv(os.path.join(a.out, f"p2_{nm}.csv"))
    g = res.get("gold", {}).get("1975-2000")
    if g and g.get("n"):
        res["gold"]["counts_own_rule"] = bool(g["mean_bp"] > 0 and (g["t_iid"] or 0) >= 1.645)
    json.dump(res, open(os.path.join(a.out, "p2.json"), "w"), indent=1, default=str)
    return res


# ------------------------------------------------------------------------------------------- P5 and extra fetches
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}


def fetch_extra(store):
    """GLD holdings archive from the API address in the fund page; BoE daily gold price in USD (candidate codes)."""
    import requests
    info = {}
    d = os.path.join(store, "history", "push", "extra")
    os.makedirs(d, exist_ok=True)
    try:
        r = requests.get("https://api.spdrgoldshares.com/api/v1/historical-archive?product=gld&exchange=NYSE&lang=en",
                         headers=UA, timeout=120)
        info["gld_api"] = {"status": r.status_code, "type": r.headers.get("content-type", ""), "bytes": len(r.content),
                           "head": r.content[:300].decode("utf-8", "replace")}
        open(os.path.join(d, "gld_api.bin"), "wb").write(r.content)
        body = r.content
        if r.ok and body[:1] in (b"{", b"["):
            j = r.json()
            urls = re.findall(r'https?://[^"\s]+', json.dumps(j))
            info["gld_api_urls"] = urls[:10]
            for u in urls:
                if re.search(r"(?i)\.(csv|xlsx?)", u):
                    body = requests.get(u, headers=UA, timeout=120).content
                    break
        tonnes = None
        if body[:2] == b"PK" or body[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            book = pd.read_excel(io.BytesIO(body), sheet_name=None, header=None)
            for nm, X in book.items():
                for i in range(min(40, len(X))):
                    row = X.iloc[i].astype(str).str.lower()
                    if row.str.contains("date").any() and row.str.contains("tonne").any():
                        dc = int(np.flatnonzero(row.str.contains("date"))[0])
                        tc = int(np.flatnonzero(row.str.contains("tonne"))[0])
                        b = X.iloc[i + 1:, [dc, tc]]
                        tonnes = pd.Series(pd.to_numeric(b.iloc[:, 1].astype(str).str.replace(",", ""), errors="coerce").values,
                                           index=pd.to_datetime(b.iloc[:, 0], errors="coerce", dayfirst=True)).dropna()
                        info["gld_sheet"] = nm
                        break
                if tonnes is not None:
                    break
        elif body[:4] != b"%PDF":
            txt = body.decode("utf-8", "ignore").splitlines()
            hi = next((i for i, l in enumerate(txt) if "date" in l.lower() and "tonne" in l.lower()), None)
            if hi is not None:
                X = pd.read_csv(io.StringIO("\n".join(txt[hi:])), on_bad_lines="skip")
                tc = next(c for c in X.columns if "tonne" in c.lower())
                tonnes = pd.Series(pd.to_numeric(X[tc].astype(str).str.replace(",", ""), errors="coerce").values,
                                   index=pd.to_datetime(X[X.columns[0]], errors="coerce", dayfirst=True)).dropna()
        if tonnes is not None and len(tonnes) > 500:
            tonnes = tonnes[~tonnes.index.duplicated()].sort_index()
            for f in glob.glob(os.path.join(store, "**", "GLD_holdings_archive.csv"), recursive=True):
                os.rename(f, f + ".pdf")
            out = os.path.join(store, "history", "etf", "GLD_holdings_archive.csv")
            with open(out, "w") as fh:
                fh.write("Date,Tonnes\n")
                for t, v in tonnes.items():
                    fh.write(f"{t.strftime('%d-%b-%Y')},{v}\n")
            info["gld_tonnes"] = [str(tonnes.index.min().date()), str(tonnes.index.max().date()), int(len(tonnes))]
    except Exception as ex:                                                 # noqa: BLE001
        info["gld_error"] = repr(ex)[:300]
    for code in ("XUDLGPD", "XUDLGPS", "XUDLGOD", "XUDLGLD"):
        u = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1975"
             f"&Dateto=31/Dec/2001&SeriesCodes={code}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
        try:
            r = requests.get(u, headers=UA, timeout=120)
            info[f"boe_{code}"] = {"status": r.status_code, "bytes": len(r.content), "head": r.content[:200].decode("utf-8", "replace")}
            if r.ok and len(r.content) > 20000 and b"DATE" in r.content[:200].upper():
                open(os.path.join(d, f"boe_gold_{code}.csv"), "wb").write(r.content)
        except Exception as ex:                                             # noqa: BLE001
            info[f"boe_{code}"] = repr(ex)[:200]
    LOG["fetch_extra"] = info
    return info


def boe_gold(store):
    for f in sorted(glob.glob(os.path.join(store, "**", "extra", "boe_gold_*.csv"), recursive=True)):
        d = pd.read_csv(f)
        s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values,
                      index=pd.to_datetime(d.iloc[:, 0], format="%d %b %Y", errors="coerce")).dropna()
        if len(s) > 3000 and 30 < s.median() < 3000:                      # a dollar price per ounce, daily
            LOG.setdefault("metal_source", {})["gold"] = f"BoE {os.path.basename(f)}"
            return s[s > 0].sort_index()
    return None


def p5(a):
    import flow_volume_lab as V
    V.ROWS.clear()
    D = V.cftc_frame(a.store, "disagg")
    mm = None
    if D is not None:
        g = D[D.code == V.DIS_CODES["GOLD"]]
        mm = V.change_z(g, "M_Money_Positions_Long_All", "M_Money_Positions_Short_All") if len(g) >= 100 else None
    V.v2_v9(a.store, mm)
    R = pd.DataFrame(V.ROWS)
    R.to_csv(os.path.join(a.out, "p5_v2_v9.csv"), index=False)
    return R.to_dict("records")


# ------------------------------------------------------------------------------------------- combine
def combine(a):
    from scipy import stats as st
    rows = []
    for nm, key in (("p1.json", ("pooled_5d", "all")), ("p2.json", ("gold", "1975-2000")),
                    ("p34.json", ("P4a", "pooled")), ("p34.json", ("P4b", "pooled"))):
        f = os.path.join(a.out, nm)
        if not os.path.exists(f):
            rows.append({"test": f"{nm}:{'/'.join(key)}", "status": "not run"})
            continue
        j = json.load(open(f))
        x = j.get(key[0], {}).get(key[1], {})
        t = x.get("t") if key[0] != "gold" else x.get("t_iid")
        rows.append({"test": f"{nm}:{'/'.join(key)}", "n": x.get("n"), "mean_bp": x.get("mean_bp"), "t": t})
    f5 = os.path.join(a.out, "p5_v2_v9.csv")
    if os.path.exists(f5):
        for r in pd.read_csv(f5).to_dict("records"):
            rows.append({"test": "P5 " + str(r.get("test")), "n": r.get("check_n"), "mean_bp": r.get("check_net_bp"),
                         "t": r.get("check_t") if pd.notna(r.get("check_t")) else None, "status": r.get("status")})
    R = pd.DataFrame(rows)
    m = R.t.notna() if "t" in R else pd.Series(False, index=R.index)
    p = pd.Series(np.nan, index=R.index)
    p[m] = st.norm.sf(R.t[m].astype(float))
    k = int(m.sum())
    if k:
        idx = p[m].sort_values().index
        q = p[idx].values * k / np.arange(1, k + 1)
        R.loc[idx, "bh_q"] = np.minimum.accumulate(q[::-1])[::-1]
    R["counts"] = m & (R.get("mean_bp", 0).astype(float) > 0) & (R.t.astype(float) >= 1.645) & (R.get("bh_q", 1.0) <= 0.10)
    R.to_csv(os.path.join(a.out, "p_combined.csv"), index=False)
    print(R.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p34", action="store_true")
    ap.add_argument("--p1", action="store_true")
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--p5", action="store_true")
    ap.add_argument("--fetch-extra", action="store_true")
    ap.add_argument("--early", default="data/m1_early")
    ap.add_argument("--store", default="flow_store")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.fetch_extra:
        print("fetch_extra", json.dumps(fetch_extra(a.store), indent=1, default=str)[:4000], flush=True)
    for flag, fn in ((a.p34, p34), (a.p1, p1), (a.p2, p2), (a.p5, p5)):
        if flag:
            try:
                r = fn(a)
                print(fn.__name__, json.dumps(r, indent=1, default=str)[:6000], flush=True)
            except Exception:                                               # noqa: BLE001
                import traceback
                traceback.print_exc()
    if a.combine:
        combine(a)
    json.dump(LOG, open(os.path.join(a.out, f"log_{'_'.join(k for k in ('p34', 'p1', 'p2', 'p5') if getattr(a, k))}.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
