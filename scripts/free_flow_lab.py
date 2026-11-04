"""Free retail-flow and daily positioning tests of docs/free_flow_preregistration.md (FF1, FF2, FD1). 2007-2025.
Usage: python scripts/free_flow_lab.py --store flow_store --out OUT [--ffaj DIR]"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_volume_lab as V  # noqa: E402
import queue_lab as Q  # noqa: E402

DEV_END = pd.Timestamp("2025-12-31")
FIT_END = pd.Timestamp("2018-12-31")
LOG = {}


def periods(x):
    return Q.periods(x)


def rule_ok(p, need_all=True):
    a, b, c = p["all"], p["2007-2018"], p["2019-2025"]
    ok = b.get("mean_bp", -1) > 0 and c.get("mean_bp", -1) > 0
    if need_all:
        ok &= a.get("mean_bp", -1) > 0 and (a.get("t") or 0) >= 1.645
    else:
        ok &= (c.get("t") or 0) >= 1.645
    return bool(ok)


# ------------------------------------------------------------------------------------------- FF1
def level_z(g, lcol, scol):
    g = g.drop_duplicates("date").set_index("date").sort_index()
    net = (pd.to_numeric(g[lcol], errors="coerce") - pd.to_numeric(g[scol], errors="coerce")) / \
        pd.to_numeric(g["Open_Interest_All"], errors="coerce")
    return (net - net.rolling(52, min_periods=30).mean().shift(1)) / net.rolling(52, min_periods=30).std().shift(1)


def ff1(a):
    T = V.cftc_frame(a.store, "tff")
    if T is None:
        return {"status": "TFF missing"}
    lc, sc = "NonRept_Positions_Long_All", "NonRept_Positions_Short_All"
    if lc not in T.columns:
        return {"status": f"no {lc} column", "columns": [c for c in T.columns if "NonRept" in c][:10]}
    out = {}
    for name, zfun in (("level (primary)", level_z), ("change (secondary)", V.change_z)):
        per = {}
        for ccy, code in V.TFF_CODES.items():
            g = T[T.code == code]
            if len(g) < 100:
                continue
            z = zfun(g, lc, sc)
            sym, sg = V.CCY_PAIR[ccy]
            rows = {}
            for d, zz in z[z.abs() >= 1.5].items():
                mon = d + pd.Timedelta(days=6)
                if mon > DEV_END or mon < pd.Timestamp("2007-01-01"):
                    continue
                r = V.weekly_trade(sym, mon, int(-np.sign(zz) * sg))            # against the small traders
                if r is not None:
                    rows[mon] = r[0]
            per[ccy] = pd.Series(rows, dtype=float)
        pool = pd.concat(per, axis=1).mean(axis=1) if per else pd.Series(dtype=float)
        out[name] = {"pooled": periods(pool), "per_currency": {k: periods(v)["all"] for k, v in per.items()}}
    D = V.cftc_frame(a.store, "disagg")
    if D is not None and lc in D.columns:
        per = {}
        for m, code in V.DIS_CODES.items():
            g = D[D.code == code]
            if len(g) < 100:
                continue
            z = level_z(g, lc, sc)
            rows = {}
            for d, zz in z[z.abs() >= 1.5].items():
                mon = d + pd.Timedelta(days=6)
                if mon > DEV_END or mon < pd.Timestamp("2007-01-01"):
                    continue
                r = V.weekly_trade(V.FUT_MAP[m][0], mon, int(-np.sign(zz)))
                if r is not None:
                    rows[mon] = r[0]
            per[m] = pd.Series(rows, dtype=float)
        if per:
            out["metals level (secondary)"] = {"pooled": periods(pd.concat(per, axis=1).mean(axis=1))}
    out["counts_own_rule"] = rule_ok(out["level (primary)"]["pooled"])
    return out


# ------------------------------------------------------------------------------------------- FF2
CCY_TOKENS = {"USD": ("USD", "米ドル", "ドル"), "EUR": ("EUR", "ユーロ"), "GBP": ("GBP", "ポンド"),
              "AUD": ("AUD", "豪ドル", "豪"), "NZD": ("NZD", "NZドル", "ニュージーランド")}
BUY = ("買", "LONG", "BUY")
SELL = ("売", "SHORT", "SELL")


def parse_month(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
        try:
            t = pd.Timestamp(v)
            return pd.Period(t, "M")
        except Exception:                                               # noqa: BLE001
            return None
    s = str(v)
    m = re.search(r"(20\d\d|19\d\d)\s*[年/.\-]\s*(\d{1,2})", s)
    if m:
        return pd.Period(f"{m.group(1)}-{int(m.group(2)):02d}", "M")
    return None


def ffaj_positions(d):
    """Month-end long / short open positions by currency from the FFAJ xls files (heuristic header search; every
    sheet's first rows are logged so the parser can be checked)."""
    out, heads = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "*.xls*"))):
        try:
            book = pd.read_excel(f, sheet_name=None, header=None)
        except Exception as ex:                                         # noqa: BLE001
            heads[os.path.basename(f)] = repr(ex)[:200]
            continue
        for sh, X in book.items():
            heads[f"{os.path.basename(f)}:{sh}"] = X.head(12).astype(str).values.tolist()
            X = X.astype(object)
            # find the header rows: one with currency tokens, the next (or the same) with buy / sell tokens
            for hr in range(min(30, len(X))):
                row = X.iloc[hr].astype(str).str.upper().tolist()
                ccy_cols = {}
                for j, v in enumerate(row):
                    for c, toks in CCY_TOKENS.items():
                        if any(t.upper() in v for t in toks) and c not in ccy_cols:
                            ccy_cols[c] = j
                if len(ccy_cols) < 2:
                    continue
                sub = X.iloc[min(hr + 1, len(X) - 1)].astype(str).str.upper().tolist()
                cols = {}
                for c, j in ccy_cols.items():
                    span = range(j, min(j + 4, len(sub)))
                    b = next((k for k in span if any(t in sub[k] for t in BUY)), None)
                    s_ = next((k for k in span if any(t in sub[k] for t in SELL)), None)
                    if b is not None and s_ is not None:
                        cols[c] = (b, s_)
                if len(cols) < 2:
                    continue
                for i in range(hr + 2, len(X)):
                    mth = parse_month(X.iat[i, 0]) or parse_month(X.iat[i, 1])
                    if mth is None:
                        continue
                    for c, (b, s_) in cols.items():
                        L = pd.to_numeric(str(X.iat[i, b]).replace(",", ""), errors="coerce")
                        S = pd.to_numeric(str(X.iat[i, s_]).replace(",", ""), errors="coerce")
                        if np.isfinite(L) and np.isfinite(S) and L + S > 0:
                            out.setdefault(c, {})[mth] = (L, S)
                break
    LOG["ffaj_heads"] = heads
    res = {c: pd.DataFrame.from_dict(v, orient="index", columns=["long", "short"]).sort_index() for c, v in out.items()}
    LOG["ffaj_parsed"] = {c: [str(v.index.min()), str(v.index.max()), int(len(v))] for c, v in res.items()}
    return res


def ff2(a):
    if not a.ffaj or not os.path.isdir(a.ffaj):
        return {"status": "FFAJ files missing"}
    P = ffaj_positions(a.ffaj)
    if len(P) < 2:
        return {"status": "data missing: the FFAJ files could not be parsed", "parsed": LOG.get("ffaj_parsed")}
    pair = {"USD": ("USDJPY", 1), "EUR": ("EURJPY", 1), "GBP": ("GBPJPY", 1), "AUD": ("AUDJPY", 1), "NZD": ("NZDJPY", 1)}
    per = {}
    for c, f in P.items():
        if c not in pair:
            continue
        n = (f.long - f.short) / (f.long + f.short)
        dn = n.diff().dropna()
        sym, sg = pair[c]
        rows = {}
        for mth, x in dn.items():
            if x == 0:
                continue
            t0 = (mth + 1).start_time + pd.Timedelta(days=15)                   # the 16th of the next month
            t0 = pd.bdate_range(t0, t0 + pd.Timedelta(days=7))[0]
            t1 = pd.bdate_range(t0 + pd.DateOffset(months=1), t0 + pd.DateOffset(months=1) + pd.Timedelta(days=7))[0]
            if t1 > DEV_END or t0 < pd.Timestamp("2007-01-01"):
                continue
            r = V.tr(sym, pd.Timestamp(t0).tz_localize("UTC") + pd.Timedelta(hours=14),
                     pd.Timestamp(t1).tz_localize("UTC") + pd.Timedelta(hours=14), int(-np.sign(x) * sg))
            if r is not None:
                rows[t0] = r[0]
        per[c] = pd.Series(rows, dtype=float)
    if not per:
        return {"status": "no tradable months"}
    pool = pd.concat(per, axis=1).mean(axis=1)
    p = {"all": {**Q.summ(pool), "t_iid": Q.tiid(pool)},
         "pre-2019": {**Q.summ(pool[pool.index <= FIT_END]), "t_iid": Q.tiid(pool[pool.index <= FIT_END])},
         "2019-2025": {**Q.summ(pool[pool.index > FIT_END]), "t_iid": Q.tiid(pool[pool.index > FIT_END])}}
    ok = (p["all"].get("mean_bp", -1) > 0 and (p["all"].get("t_iid") or 0) >= 1.645
          and p["pre-2019"].get("mean_bp", -1) > 0 and p["2019-2025"].get("mean_bp", -1) > 0)
    return {"pooled": p, "per_currency": {k: Q.summ(v) for k, v in per.items()}, "parsed": LOG.get("ffaj_parsed"),
            "counts_own_rule": bool(ok)}


# ------------------------------------------------------------------------------------------- FD1
FD_SYMS = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "XAGUSD")
FD_CFTC = {"EURUSD": ("tff", "EUR", 1), "GBPUSD": ("tff", "GBP", 1), "USDJPY": ("tff", "JPY", -1),
           "USDCHF": ("tff", "CHF", -1), "USDCAD": ("tff", "CAD", -1), "AUDUSD": ("tff", "AUD", 1),
           "NZDUSD": ("tff", "NZD", 1), "XAUUSD": ("disagg", "GOLD", 1), "XAGUSD": ("disagg", "SILVER", 1)}


def cta_position(px):
    r = np.log(px).diff()
    sd = r.rolling(63, min_periods=40).std()
    parts = []
    for L in (80, 160, 260):
        m = r.rolling(L, min_periods=int(0.8 * L)).mean()
        parts.append((m / sd * math.sqrt(L)).clip(-1, 1))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def fd1(a):
    import edges_lab2 as E
    import untested_intraday_lab as U
    T = V.cftc_frame(a.store, "tff")
    D = V.cftc_frame(a.store, "disagg")
    flow, crowd, valid = {}, {}, {}
    for sym in FD_SYMS:
        bk = E.book(sym)
        if bk is None:
            continue
        px = {}
        for d in pd.bdate_range("2007-01-01", DEV_END):
            i = bk.at(V.ny(d, 16, 55), 10)
            if i >= 0:
                px[pd.Timestamp(d)] = 0.5 * (bk.bo[i] + bk.ao[i])
        px = pd.Series(px)
        pos = cta_position(px).dropna()
        # validation against the CFTC positions (reported, not used)
        src, key, sg = FD_CFTC[sym]
        F = T if src == "tff" else D
        lc, sc = (("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All") if src == "tff"
                  else ("M_Money_Positions_Long_All", "M_Money_Positions_Short_All"))
        code = V.TFF_CODES.get(key) if src == "tff" else V.DIS_CODES.get(key)
        if F is not None and code is not None:
            g = F[F.code == code].drop_duplicates("date").set_index("date").sort_index()
            net = (pd.to_numeric(g[lc], errors="coerce") - pd.to_numeric(g[sc], errors="coerce")) / \
                pd.to_numeric(g["Open_Interest_All"], errors="coerce") * sg
            est = pd.Series([pos.asof(t) for t in net.index], index=net.index)
            ok = net.notna() & est.notna()
            valid[sym] = {"weeks": int(ok.sum()), "corr_level": float(net[ok].corr(est[ok])) if ok.sum() > 50 else None,
                          "corr_change": float(net[ok].diff().corr(est[ok].diff())) if ok.sum() > 50 else None}
        dp = pos.diff()
        thr = dp.abs()[dp.index <= FIT_END].quantile(0.80)
        rows = {}
        for d, x in dp[dp.abs() > thr].items():
            if d > DEV_END:
                continue
            r = bk.trade(V.ny(d), V.ny(V.bday_after(d, 1)), int(np.sign(x)), tol_min=30)
            if r is not None:
                rows[d] = U.net_of(r)
        flow[sym] = pd.Series(rows, dtype=float)
        rows = {}
        hit = (pos.abs() >= 0.95) & (pos.abs().shift(1) < 0.95)
        for d in pos.index[hit.values]:
            if d > DEV_END:
                continue
            r = bk.trade(V.ny(d), V.ny(V.bday_after(d, 5)), int(-np.sign(pos[d])), tol_min=30)
            if r is not None:
                rows[d] = U.net_of(r)
        crowd[sym] = pd.Series(rows, dtype=float)
        E.BOOKS.clear()
        del bk
        print("FD1", sym, len(flow[sym]), len(crowd[sym]), flush=True)
    res = {"validation_vs_cftc": valid}
    for nm, per in (("FD1a flow", flow), ("FD1b crowding", crowd)):
        if not per:
            res[nm] = {"status": "data missing"}
            continue
        pool = pd.concat(per, axis=1).mean(axis=1)
        res[nm] = {"pooled": periods(pool), "per_instrument": {k: periods(v)["all"] for k, v in per.items()}}
        res[nm]["counts_own_rule"] = rule_ok(res[nm]["pooled"], need_all=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="flow_store")
    ap.add_argument("--ffaj", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for nm, fn in (("FF2", ff2), ("FF1", ff1), ("FD1", fd1)):
        try:
            res[nm] = fn(a)
        except Exception as ex:                                         # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[nm] = {"status": f"error {ex!r}"[:300]}
        print(nm, json.dumps(res[nm], default=str)[:3000], flush=True)
    json.dump(res, open(os.path.join(a.out, "free_flow.json"), "w"), indent=1, default=str)
    json.dump(LOG, open(os.path.join(a.out, "log_free_flow.json"), "w"), indent=1, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
