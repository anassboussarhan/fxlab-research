"""Information nobody in the earlier runs had: the CFTC Commitments of Traders positioning of speculators
(non-commercials) and hedgers (commercials) in the currency and gold futures, as weekly signals for FX and gold.
Free public data (cftc.gov legacy futures-only reports). Development data only: weeks to 2025-12-31.

FIXED BEFORE ANY RESULT:
  Markets: EUR, JPY, GBP, CHF, CAD, AUD, NZD (the CME futures; a position in the currency is against the USD) and
    gold (COMEX). Prices: Yahoo daily closes (EURUSD=X, JPY=X, GBPUSD=X, CHF=X, CAD=X, AUDUSD=X, NZDUSD=X; GLD for
    gold), 2004-2025.
  Timing: positions are as of Tuesday and published on Friday at 15:30 New York; a signal is traded from the close
    of the Monday after publication (Tuesday + 6 days, or the next trading day) to the close of the following
    Monday, when the next signal takes over. No position is ever based on a report not yet published.
  Returns: the currency's spot return against the USD plus the 3-month rate differential (point-in-time, one-month
    lag) minus a 0.75% a year financing mark-up, for longs and shorts alike; gold: GLD's return minus the USD rate
    plus 2.5% a year for longs, plus the USD rate minus 2.5% for shorts (CFD financing). Costs on every change of
    position: 1 bp a side for EUR, JPY, GBP, CHF, CAD, AUD; 1.5 bp for NZD; 2 bp for gold.
  Positioning: speculators' net = (non-commercial long - short) / open interest; z = its distance from the
    average of the previous 156 weeks in standard deviations (at least 104 weeks of history); commercials alike.
  Signals:
    C1 crowding reversal: short the currency (or gold) when the speculators' z > 1.5, long when z < -1.5.
    C2 positioning momentum: the sign of the 4-week change in the speculators' net, always in the market.
    C3 cross-sectional crowding: each week long the two currencies with the lowest speculator z and short the two
       with the highest (half a unit each), currencies only.
    C4 follow the hedgers: long when the commercials' z > 1.5, short when < -1.5.
PRE-REGISTERED TESTS (one-sided, mean weekly net return > 0, Newey-West t with 4 lags): each signal pooled (the
  equal-weight average over the markets) and per market; Benjamini-Hochberg over all of them.
Usage: python scripts/cot_lab.py --fetch DATA_DIR            (CI: downloads the reports and prices)
       python scripts/cot_lab.py --data DATA_DIR --out DIR"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = pd.Timestamp("2025-12-31")
CODES = {"EUR": "099741", "JPY": "097741", "GBP": "096742", "CHF": "092741", "CAD": "090741", "AUD": "232741",
         "NZD": "112741", "XAU": "088691"}
YAHOO = {"EUR": ("EURUSD=X", 1), "JPY": ("JPY=X", -1), "GBP": ("GBPUSD=X", 1), "CHF": ("CHF=X", -1),
         "CAD": ("CAD=X", -1), "AUD": ("AUDUSD=X", 1), "NZD": ("NZDUSD=X", 1), "XAU": ("GLD", 1)}
COST = {"EUR": 1e-4, "JPY": 1e-4, "GBP": 1e-4, "CHF": 1e-4, "CAD": 1e-4, "AUD": 1e-4, "NZD": 1.5e-4, "XAU": 2e-4}
CCY = ("EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")
FIN_MARKUP, CFD_MARKUP = 0.0075, 0.025


# ----------------------------------------------------------------------------------------------- data
def fetch(out):
    import requests
    import yfinance as yf
    os.makedirs(out, exist_ok=True)
    frames = []
    urls = ["https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"] + \
           [f"https://www.cftc.gov/files/dea/history/deacot{y}.zip" for y in range(2017, 2026)]
    for u in urls:
        r = requests.get(u, timeout=120, headers={"User-Agent": "Mozilla/5.0 research"})
        print(u, r.status_code, len(r.content), flush=True)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for n in z.namelist():
            if n.lower().endswith(".txt"):
                frames.append(pd.read_csv(z.open(n), low_memory=False))
    raw = pd.concat(frames, ignore_index=True)
    print("COT columns:", list(raw.columns)[:20], flush=True)
    cot = tidy_cot(raw)
    cot.to_csv(os.path.join(out, "cot.csv"), index=False)
    print(cot.groupby("market").date.agg(["min", "max", "size"]), flush=True)
    px = {}
    for k, (tick, _) in YAHOO.items():
        d = yf.download(tick, start="2003-01-01", end="2026-01-01", auto_adjust=True, progress=False)
        c = d["Close"]
        c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        px[k] = c
        print(tick, len(c), c.index.min(), c.index.max(), flush=True)
    P = pd.DataFrame(px)
    P.index = pd.DatetimeIndex(P.index).tz_localize(None)
    P[P.index <= END].to_csv(os.path.join(out, "prices.csv"))


def tidy_cot(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c: " ".join(str(c).lower().split()) for c in raw.columns}

    def find(*keys, avoid=()):
        for c, n in cols.items():
            if all(k in n for k in keys) and not any(a in n for a in avoid):
                return c
        raise KeyError(keys)
    c_date = next((c for c, n in cols.items() if "yyyy-mm-dd" in n), None)
    c_code = find("contract market code")
    c_oi = find("open interest", "(all)", avoid=("change", "%", "traders"))
    c_ncl = find("noncommercial", "long", "(all)", avoid=("change", "%", "spread", "traders"))
    c_ncs = find("noncommercial", "short", "(all)", avoid=("change", "%", "spread", "traders"))
    c_cl = find("commercial", "long", "(all)", avoid=("noncommercial", "change", "%", "traders"))
    c_cs = find("commercial", "short", "(all)", avoid=("noncommercial", "change", "%", "traders"))
    code = raw[c_code].astype(str).str.strip().str.zfill(6)
    if c_date is not None:
        date = pd.to_datetime(raw[c_date], errors="coerce")
    else:
        date = pd.to_datetime(raw[find("yymmdd")].astype(str).str.zfill(6), format="%y%m%d", errors="coerce")
    inv = {v: k for k, v in CODES.items()}
    df = pd.DataFrame({"market": code.map(inv), "date": date,
                       "oi": pd.to_numeric(raw[c_oi], errors="coerce"),
                       "nc_long": pd.to_numeric(raw[c_ncl], errors="coerce"),
                       "nc_short": pd.to_numeric(raw[c_ncs], errors="coerce"),
                       "c_long": pd.to_numeric(raw[c_cl], errors="coerce"),
                       "c_short": pd.to_numeric(raw[c_cs], errors="coerce")})
    df = df.dropna(subset=["market", "date", "oi"])
    df = df[(df.oi > 0) & (df.date <= END)]
    return df.drop_duplicates(["market", "date"]).sort_values(["market", "date"])


# ----------------------------------------------------------------------------------------------- signals
def zscore(x: pd.Series, n=156, min_n=104) -> pd.Series:
    """Distance of x_t from the mean of the previous n weeks (t excluded), in their standard deviations."""
    prev = x.shift(1)
    m = prev.rolling(n, min_periods=min_n).mean()
    s = prev.rolling(n, min_periods=min_n).std()
    return (x - m) / s


def weekly_panel(cot: pd.DataFrame, prices: pd.DataFrame):
    """Per market: signals at each report and the net return of holding from the Monday close after publication
    to the next such Monday close. Returns dict market -> DataFrame indexed by entry date."""
    from fxlab.data import rates as R
    out = {}
    for mkt, g in cot.groupby("market"):
        g = g.set_index("date").sort_index()
        g["ns"] = (g.nc_long - g.nc_short) / g.oi
        g["nc"] = (g.c_long - g.c_short) / g.oi
        g["z"] = zscore(g.ns)
        g["zc"] = zscore(g.nc)
        g["d4"] = g.ns - g.ns.shift(4)
        px = prices[mkt].dropna()
        if px.empty:
            continue
        sign = YAHOO[mkt][1]
        entry_target = g.index + pd.Timedelta(days=6)                  # the Monday after the Friday release
        pos = px.index.searchsorted(entry_target)
        ok = pos < len(px)
        g = g[ok].copy()
        g["entry"] = px.index[pos[ok]]
        g = g[~g.entry.duplicated(keep="last")]
        nxt = g.entry.shift(-1)
        g = g.iloc[:-1].copy()
        g["exit"] = nxt.iloc[:-1].values
        p0 = px.reindex(g.entry).to_numpy()
        p1 = px.reindex(g["exit"]).to_numpy()
        spot = (p1 / p0) ** sign - 1                                 # return of being long the currency / gold
        days = (pd.DatetimeIndex(g["exit"]) - pd.DatetimeIndex(g.entry)).days.to_numpy()
        k = R.known_at(pd.DatetimeIndex(g.entry).tz_localize("UTC"))
        usd = np.nan_to_num(k["USD"].to_numpy())
        if mkt == "XAU":
            g["r_long"] = spot - (usd + CFD_MARKUP) * days / 360
            g["r_short"] = -spot + (usd - CFD_MARKUP) * days / 360
        else:
            fr = np.nan_to_num(k[mkt].to_numpy())
            g["r_long"] = spot + (fr - usd - FIN_MARKUP) * days / 365
            g["r_short"] = -spot + (usd - fr - FIN_MARKUP) * days / 365
        g = g[np.isfinite(g.r_long) & (g.entry >= pd.Timestamp("2004-01-01")) & (g["exit"] <= END)]
        out[mkt] = g.set_index("entry")
    return out


def positions(panel):
    """Signal -> {market: position series (+1 long, -1 short, 0 flat; C3 uses +-0.5)}."""
    P = {}
    P["C1"] = {m: pd.Series(np.where(g.z > 1.5, -1.0, np.where(g.z < -1.5, 1.0, 0.0)), g.index).where(g.z.notna())
               for m, g in panel.items()}
    P["C2"] = {m: np.sign(g.d4) for m, g in panel.items()}
    P["C4"] = {m: pd.Series(np.where(g.zc > 1.5, 1.0, np.where(g.zc < -1.5, -1.0, 0.0)), g.index).where(g.zc.notna())
               for m, g in panel.items()}
    Z = pd.DataFrame({m: panel[m].z for m in CCY if m in panel})
    c3 = {m: pd.Series(0.0, index=Z.index) for m in Z.columns}
    for t, row in Z.iterrows():
        r = row.dropna()
        if len(r) < 5:
            for m in Z.columns:
                c3[m].loc[t] = np.nan
            continue
        order = r.sort_values()
        for m in order.index[:2]:
            c3[m].loc[t] = 0.5
        for m in order.index[-2:]:
            c3[m].loc[t] = -0.5
    P["C3"] = {m: s.reindex(panel[m].index) for m, s in c3.items()}
    return P


def pnl(panel, pos):
    """Weekly net returns per market: position x the long or short return, minus costs on changes."""
    out = {}
    for m, p in pos.items():
        g = panel[m]
        p = p.reindex(g.index)
        r = np.where(p > 0, p * g.r_long, np.where(p < 0, -p * g.r_short, 0.0))
        turn = p.fillna(0).diff().abs().fillna(p.fillna(0).abs())
        x = pd.Series(r, g.index) - turn * COST[m]
        out[m] = x[p.notna()]
    return out


def nw_t(x, lags=4):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def run(a):
    from scipy import stats as st
    from cross_market_ta_lab import bh_q
    os.makedirs(a.out, exist_ok=True)
    cot = pd.read_csv(os.path.join(a.data, "cot.csv"), parse_dates=["date"])
    prices = pd.read_csv(os.path.join(a.data, "prices.csv"), index_col=0, parse_dates=True)
    panel = weekly_panel(cot, prices)
    P = positions(panel)
    rows, series = [], {}
    for sig, pos in P.items():
        X = pnl(panel, pos)
        for m, x in X.items():
            active = x[pos[m].reindex(x.index).fillna(0) != 0]
            t = nw_t(x)
            rows.append({"signal": sig, "market": m, "weeks": int(len(x)), "active_weeks": int(len(active)),
                         "mean_bp_week": float(x.mean() * 1e4), "t": t,
                         "p_one_sided": float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan,
                         "ann_return": float(x.mean() * 52), "sharpe": float(x.mean() / x.std() * math.sqrt(52))
                         if x.std() > 0 else np.nan, "first": str(x.index.min())[:10]})
        D = pd.DataFrame(X)
        pooled = D.mean(axis=1, skipna=True).dropna()
        series[sig] = pooled
        t = nw_t(pooled)
        yrs = pooled.groupby(pooled.index.year).sum()
        rows.append({"signal": sig, "market": "pooled", "weeks": int(len(pooled)),
                     "mean_bp_week": float(pooled.mean() * 1e4), "t": t,
                     "p_one_sided": float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan,
                     "ann_return": float(pooled.mean() * 52),
                     "sharpe": float(pooled.mean() / pooled.std() * math.sqrt(52)) if pooled.std() > 0 else np.nan,
                     "first": str(pooled.index.min())[:10],
                     "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}",
                     "2004_2014_bp": float(pooled[pooled.index < "2015-01-01"].mean() * 1e4),
                     "2015_2025_bp": float(pooled[pooled.index >= "2015-01-01"].mean() * 1e4)})
    R = pd.DataFrame(rows)
    R["q_bh"] = bh_q(R.p_one_sided.to_numpy())
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame(series).to_csv(os.path.join(a.out, "pooled_weekly.csv"))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(R.round(3).to_string())
    json.dump({"tests": int(len(R)), "q10": int((R.q_bh <= 0.10).sum()), "t_gt_2": int((R.t > 2).sum()),
               "t_lt_m2": int((R.t < -2).sum())}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    fetch(a.fetch) if a.fetch else run(a)


if __name__ == "__main__":
    main()
