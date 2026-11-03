"""M1 of docs/edges_batch_preregistration.md (addendum 3): one weekly model pooled over the 28 pairs of USD, EUR, GBP,
JPY, CHF, CAD, AUD, NZD. 2007-2025 only (nothing from 2026).

Timing. The rule says Friday 21:00 UTC; in US summer time the FX market closes at 21:00 UTC, so no price exists at
or after it. Decisions and fills use the M1 bar at 20:45 UTC on Friday (both seasons), held to 20:45 UTC the next
Friday. Every price feature uses 20:45 UTC marks; VIX and the S&P 500 (the Dukascopy USA500 CFD) use Thursday's
values, the last that are certainly closed at 20:45 UTC Friday; the TFF report of the same Tuesday is published on
Friday at 15:30 New York, before the decision.

Target: the next 5-day mid log return over its 20-day volatility (regression). LightGBM with the 8.5 settings
(regression objective), refitted every year on weeks whose label ended before 1 January of the test year (model B).
Model A, fitted one year earlier, scores the year before the test year; the 90th percentile of |prediction| there
is the cut. Each week: the 5 pairs with the largest |prediction| above the cut, long if > 0, short if < 0, 1/5 of
capital each. Fills: bid/ask + slippage, commission, real rate swaps (rate gap minus the 0.75% mark-up; the flat
0.5-pip charge when no rate exists). Counts if the weekly portfolio net > 0 with t >= 1.645 on 2019-2025.
Usage: python scripts/ml_weekly_lab.py --raw RAWDIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CCY = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")       # market quoting order
PAIRS = [a + b for i, a in enumerate(CCY) for b in CCY[i + 1:]]
END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
TFF_CODE = {"JPY": "097741", "EUR": "099741", "GBP": "096742", "AUD": "232741", "CAD": "090741", "CHF": "092741",
            "NZD": "112741", "MXN": "095741"}
LGBR = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=4)


def month_t(x: pd.Series):
    if len(x) < 10:
        return np.nan
    e = x - x.mean()
    g = e.groupby(pd.DatetimeIndex(x.index).tz_localize(None).to_period("M")).sum()
    m = len(g)
    se = math.sqrt((g ** 2).sum()) / len(x) * math.sqrt(m / max(m - 1, 1))
    return float(x.mean() / se) if se > 0 else np.nan


def marks(bk, days):
    """Mid at the first M1 bar in [20:45, 20:55] UTC of each day, NaN otherwise."""
    out = np.full(len(days), np.nan)
    for k, d in enumerate(days):
        i = bk.at(d + pd.Timedelta(hours=20, minutes=45), 10)
        if i >= 0 and bk.ao[i] >= bk.bo[i] > 0:
            out[k] = 0.5 * (bk.bo[i] + bk.ao[i])
    return pd.Series(out, index=days)


def book(sym):
    import flows_lab as FL
    from fxlab.data.store import load_m1
    try:
        m1 = load_m1(sym, "2007-01-01", "2025-12-31")
    except FileNotFoundError:
        return None
    m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)]
    m1 = m1[m1.index <= END]
    pip = 0.01 if "JPY" in sym else 1e-4
    return FL.Book(m1[["bid_o", "ask_o"]], pip, sym if len(sym) == 6 else None)


def fred(sid, raw):
    for p in sorted(glob.glob(os.path.join(raw, "**", f"fred_{sid}.csv"), recursive=True)):
        df = pd.read_csv(p)
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        return s.dropna()
    try:
        import requests
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=90)
        df = pd.read_csv(io.StringIO(r.text))
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        return s.dropna()
    except Exception as ex:                                                 # noqa: BLE001
        print("FRED", sid, repr(ex)[:120], flush=True)
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def tff(raw):
    fs = sorted(glob.glob(os.path.join(raw, "**", "cftc_tff", "*.txt"), recursive=True))
    if not fs:
        return {}
    D = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        D.append(d)
    T = pd.concat(D, ignore_index=True)
    T["date"] = pd.to_datetime(T["Report_Date_as_YYYY-MM-DD"])
    T["code"] = T["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
    out = {}
    for c, code in TFF_CODE.items():
        g = T[T.code == code].drop_duplicates("date").set_index("date").sort_index()
        if len(g) < 60:
            continue
        net = (pd.to_numeric(g.Lev_Money_Positions_Long_All, errors="coerce") -
               pd.to_numeric(g.Lev_Money_Positions_Short_All, errors="coerce")) / \
            pd.to_numeric(g.Open_Interest_All, errors="coerce")
        out[c] = (net - net.rolling(156, min_periods=52).mean()) / net.rolling(156, min_periods=52).std()
    return out


def build(a):
    days = pd.date_range("2007-01-01", "2025-12-31", freq="B", tz="UTC")
    fri = days[days.weekday == 4]
    vix = fred("VIXCLS", a.raw)
    spx_bk = book("USA500IDXUSD")
    spx = marks(spx_bk, days) if spx_bk is not None else pd.Series(np.nan, index=days)
    del spx_bk
    Z = tff(a.raw)
    print("VIX", len(vix), "SPX marks", int(spx.notna().sum()), "TFF", sorted(Z), flush=True)
    from fxlab.data.rates import has_rates, rate_diff
    rows = []
    for p in PAIRS:
        bk = book(p)
        if bk is None:
            print("no data", p, flush=True)
            continue
        m = marks(bk, days)
        lr = np.log(m).diff()
        v20 = lr.rolling(20, min_periods=15).std()
        v120 = lr.rolling(120, min_periods=90).std()
        hi, lo = m.rolling(252, min_periods=200).max(), m.rolling(252, min_periods=200).min()
        rd = rate_diff(p, days) if has_rates(p) else pd.Series(np.nan, index=days)
        for k, t in enumerate(fri):
            if not np.isfinite(m.get(t, np.nan)):
                continue
            ti = days.get_loc(t)
            prev = days[ti - 1] if ti > 0 else t
            r = {"ts": t, "pair": p, "base": p[:3], "quote": p[3:]}
            for w in (1, 4, 12, 26):
                j = ti - 5 * w
                r[f"mom{w}"] = float(np.log(m.iloc[ti] / m.iloc[j])) / (v20.iloc[ti] * math.sqrt(5 * w)) \
                    if j >= 0 and np.isfinite(m.iloc[j]) and v20.iloc[ti] > 0 else np.nan
            r["carry"] = float(rd.iloc[ti]) if np.isfinite(rd.iloc[ti]) else np.nan
            r["carry_chg3m"] = float(rd.iloc[ti] - rd.iloc[ti - 63]) if ti >= 63 else np.nan
            r["vol20"], r["vol_ratio"] = float(v20.iloc[ti]), float(v20.iloc[ti] / v120.iloc[ti]) if v120.iloc[ti] > 0 else np.nan
            r["dist_hi"] = float(np.log(m.iloc[ti] / hi.iloc[ti]) / v20.iloc[ti]) if v20.iloc[ti] > 0 else np.nan
            r["dist_lo"] = float(np.log(m.iloc[ti] / lo.iloc[ti]) / v20.iloc[ti]) if v20.iloc[ti] > 0 else np.nan
            vv = vix[vix.index <= prev.tz_convert(None).normalize()]
            r["vix"] = float(vv.iloc[-1]) if len(vv) else np.nan
            r["vix_chg1w"] = float(vv.iloc[-1] - vv.iloc[-6]) if len(vv) > 6 else np.nan
            j = ti - 1
            r["spx_4w"] = float(np.log(spx.iloc[j] / spx.iloc[j - 20])) if j >= 20 else np.nan
            for side, c in (("base", p[:3]), ("quote", p[3:])):
                z = Z.get(c)
                if z is None:
                    r[f"tff_{side}"] = np.nan
                    continue
                zz = z[z.index <= (t.tz_convert(None) - pd.Timedelta(days=3)).normalize()]
                r[f"tff_{side}"] = float(zz.iloc[-1]) if len(zz) and (t.tz_convert(None) - zz.index[-1]).days <= 10 else np.nan
            r["month"], r["wom"] = t.month, (t.day - 1) // 7 + 1
            # target and the trade to next Friday 20:45 UTC
            t1 = t + pd.Timedelta(days=7)
            if t1 in days and ti + 5 < len(m) and np.isfinite(m.get(t1, np.nan)) and v20.iloc[ti] > 0:
                r["fwd"] = float(np.log(m[t1] / m.iloc[ti]))
                r["y"] = r["fwd"] / (v20.iloc[ti] * math.sqrt(5))
                for d, nm in ((1, "long"), (-1, "short")):
                    x = bk.trade(t + pd.Timedelta(hours=20, minutes=45), t1 + pd.Timedelta(hours=20, minutes=45), d, 10)
                    if x is None:
                        r[nm] = np.nan
                        continue
                    net, gross, _, _, n, flat, rate = x
                    r[nm] = net + (rate if np.isfinite(rate) else flat)
                    r[nm + "_noswap"] = net
            rows.append(r)
        print(p, len(rows), flush=True)
        del bk
    T = pd.DataFrame(rows)
    T.to_parquet(os.path.join(a.out, "weekly_table.parquet"))
    return T


FEATS = ["mom1", "mom4", "mom12", "mom26", "carry", "carry_chg3m", "vol20", "vol_ratio", "dist_hi", "dist_lo", "vix",
         "vix_chg1w", "spx_4w", "tff_base", "tff_quote", "month", "wom"]


def model(T, a):
    import lightgbm as lgb
    T = T.sort_values(["ts", "pair"]).reset_index(drop=True)
    T["ts"] = pd.to_datetime(T.ts, utc=True)
    yr = T.ts.dt.year.to_numpy()
    lab_end = (T.ts + pd.Timedelta(days=7)).dt.year.to_numpy()
    ok = T.y.notna().to_numpy()
    X = T[FEATS].to_numpy(np.float32)
    y = T.y.to_numpy()
    pred = np.full(len(T), np.nan)
    cut = {}
    for Y in range(2011, 2026):
        trA, trB = ok & (lab_end < Y - 1), ok & (lab_end < Y)
        cal, te = (yr == Y - 1), np.flatnonzero(yr == Y)
        if trA.sum() < 2000 or not len(te):
            continue
        mA = lgb.LGBMRegressor(**LGBR).fit(X[trA], y[trA])
        cut[Y] = float(np.quantile(np.abs(mA.predict(X[cal])), 0.9))
        mB = lgb.LGBMRegressor(**LGBR).fit(X[trB], y[trB])
        pred[te] = mB.predict(X[te])
        print("M1 year", Y, "cut", round(cut[Y], 4), "train", int(trB.sum()), flush=True)
    T["pred"] = pred
    T["cut"] = T.ts.dt.year.map(cut)
    trades = []
    weeks = []
    for t, g in T[T.pred.notna()].groupby("ts"):
        g = g[(g.pred.abs() > g.cut) & g.long.notna() & g.short.notna()]
        g = g.reindex(g.pred.abs().sort_values(ascending=False).index)[:5]
        nets = [(r.long if r.pred > 0 else r.short) for r in g.itertuples()]
        for r, n in zip(g.itertuples(), nets):
            trades.append({"ts": t, "pair": r.pair, "side": 1 if r.pred > 0 else -1, "net": n, "pred": r.pred})
        weeks.append({"ts": t, "n": len(nets), "port": float(np.sum(nets) / 5) if nets else 0.0})
    W = pd.DataFrame(weeks).set_index("ts")
    TR = pd.DataFrame(trades)
    # information: rank IC of the prediction with the next-week return, by period
    out = {"weeks": int(len(W)), "trades": int(len(TR)), "periods": {}}
    for per, m in (("2011-2018", W.index.year <= 2018), ("2019-2025", W.index.year >= 2019)):
        w = W[m]
        tr = TR[(TR.ts.dt.year <= 2018) if per.startswith("2011") else (TR.ts.dt.year >= 2019)] if len(TR) else TR
        sub = T[(T.pred.notna()) & ((T.ts.dt.year <= 2018) if per.startswith("2011") else (T.ts.dt.year >= 2019))]
        ic = sub.groupby("ts").apply(lambda g: g.pred.rank().corr(g.fwd.rank()) if len(g) > 5 else np.nan).dropna()
        out["periods"][per] = {
            "weeks": int(len(w)), "weeks_with_trades": int((w.n > 0).sum()), "trades": int(len(tr)),
            "weekly_port_net_bp": float(w.port.mean() * 1e4), "t_weekly": month_t(w.port),
            "per_trade_net_bp": float(tr.net.mean() * 1e4) if len(tr) else None,
            "per_trade_win": float((tr.net > 0).mean()) if len(tr) else None,
            "mean_weekly_rank_ic": float(ic.mean()) if len(ic) else None,
            "ic_t": float(ic.mean() / ic.std() * math.sqrt(len(ic))) if len(ic) > 2 and ic.std() > 0 else None}
    c = out["periods"].get("2019-2025", {})
    out["counts"] = bool(c.get("weekly_port_net_bp", -1) > 0 and (c.get("t_weekly") or 0) >= 1.645)
    W.to_csv(os.path.join(a.out, "m1_weeks.csv"))
    TR.to_csv(os.path.join(a.out, "m1_trades.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "m1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_parquet(a.table) if a.table else build(a)
    model(T, a)


if __name__ == "__main__":
    main()
