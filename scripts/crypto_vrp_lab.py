"""Crypto variance risk premium from free data: Deribit's DVOL (30-day implied volatility index of BTC / ETH options,
public API, no account) against the realised volatility that followed, as a short-variance proxy (the same method
as the S&P test in docs/options_retail.md). Development data only: cut at 2025-12-31.

PRE-REGISTERED (one-sided, BH over the four):
  C1  BTC: each month start, sell 30-day variance at DVOL; P&L in vol points = (K^2 - RV^2) / 2K, RV = realised vol
      of daily log returns over the next 30 days (annualised with 365 days); minus 2 vol points of cost
  C2  ETH, the same
  C3  C1 only when DVOL > the realised vol of the previous 30 days
  C4  C2 only when DVOL > the realised vol of the previous 30 days
Reported: before costs, by year, worst month, and BTC / ETH DVOL coverage.
Usage: python scripts/crypto_vrp_lab.py --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats as st

END = pd.Timestamp("2025-12-31")
COST_VP = 2.0


def get_json(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:                                           # noqa: BLE001
            time.sleep(2 ** k)
    return None


def dvol(cur):
    """Daily DVOL closes from Deribit, paging backwards with the continuation token."""
    end = int(pd.Timestamp("2026-01-01").timestamp() * 1000)
    start = int(pd.Timestamp("2019-01-01").timestamp() * 1000)
    rows = []
    for _ in range(200):
        js = get_json(f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={cur}"
                      f"&start_timestamp={start}&end_timestamp={end}&resolution=1D")
        if not js or "result" not in js:
            break
        data = js["result"].get("data") or []
        rows += data
        cont = js["result"].get("continuation")
        if not cont or not data:
            break
        end = int(cont)
    if not rows:
        return pd.Series(dtype=float)
    D = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c"]).drop_duplicates("ts")
    D["date"] = pd.to_datetime(D.ts, unit="ms").dt.normalize()
    return D.set_index("date").c.sort_index()


def spot(sym):
    import yfinance as yf
    h = yf.download(sym, start="2018-01-01", end="2026-01-01", progress=False, auto_adjust=False)
    c = h["Close"]
    c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
    c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.to_datetime(c.index)
    return c[c.index <= END].dropna()


def nw_t(x, lag=1):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.nan
    e = x - x.mean()
    s = e @ e / n + sum(2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n for L in range(1, lag + 1))
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def months(K, S):
    lr = np.log(S).diff()
    out = []
    for m in pd.date_range(K.index[0] + pd.offsets.MonthBegin(1), END, freq="MS"):
        k = K.loc[:m]
        if not len(k) or (m - k.index[-1]).days > 3:
            continue
        k0 = float(k.iloc[-1])
        fwd = lr.loc[m + pd.Timedelta(days=1): m + pd.Timedelta(days=30)].dropna()
        back = lr.loc[m - pd.Timedelta(days=29): m].dropna()
        if len(fwd) < 25 or m + pd.Timedelta(days=30) > END:
            continue
        rv = float(fwd.std() * math.sqrt(365) * 100)
        rvb = float(back.std() * math.sqrt(365) * 100) if len(back) >= 20 else np.nan
        gross = (k0 ** 2 - rv ** 2) / (2 * k0)
        out.append({"month": m, "K": k0, "rv": rv, "rv_prev": rvb, "gross_vp": gross, "net_vp": gross - COST_VP})
    return pd.DataFrame(out).set_index("month")


def stats(x):
    x = x.dropna()
    if len(x) < 8:
        return {"n": int(len(x))}
    t = nw_t(x.to_numpy())
    yrs = x.groupby(x.index.year).sum()
    return {"n": int(len(x)), "mean_vp": float(x.mean()), "t_nw": t, "p_one_sided": float(st.norm.sf(t)),
            "sharpe": float(x.mean() / x.std() * math.sqrt(12)), "months_positive": float((x > 0).mean()),
            "worst_vp": float(x.min()), "worst_month": str(x.idxmin().date()),
            **{f"y{y}": float(v) for y, v in yrs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, meta = [], {}
    for cur, ysym in (("BTC", "BTC-USD"), ("ETH", "ETH-USD")):
        K = dvol(cur)
        S = spot(ysym)
        meta[cur] = {"dvol_first": str(K.index[0].date()) if len(K) else None, "dvol_n": int(len(K)),
                     "spot_first": str(S.index[0].date()) if len(S) else None}
        if not len(K) or not len(S):
            continue
        M = months(K[K.index <= END], S)
        M.to_csv(os.path.join(a.out, f"months_{cur}.csv"))
        tag = "C1" if cur == "BTC" else "C2"
        rows.append({"test": f"{tag} {cur} short variance at DVOL", **stats(M.net_vp),
                     "gross_mean_vp": float(M.gross_vp.mean()), "gross_t": nw_t(M.gross_vp.to_numpy()),
                     "mean_K": float(M.K.mean()), "mean_rv": float(M.rv.mean())})
        F = M[M.K > M.rv_prev]
        tag = "C3" if cur == "BTC" else "C4"
        rows.append({"test": f"{tag} {cur} only when DVOL > past 30-day RV", **stats(F.net_vp),
                     "gross_mean_vp": float(F.gross_vp.mean()), "gross_t": nw_t(F.gross_vp.to_numpy())})
    T = pd.DataFrame(rows)
    if len(T):
        p = T.p_one_sided.to_numpy()
        o = np.argsort(p)
        q = p[o] * len(p) / (np.arange(len(p)) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        T.loc[T.index[o], "bh_q"] = np.minimum(q, 1)
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump({"meta": meta, "cost_vol_points": COST_VP}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps(meta, indent=1))
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
