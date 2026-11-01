"""Fourth literature round, daily and monthly data: FX and gold seasonality as blogs, banks and one paper report it,
and gold around the turn of the month. Rules fixed before any result; development data only (to 2025-12-31).

FIXED BEFORE ANY RESULT:
  Currencies vs the US dollar: EUR (DEM x 1.95583 before 1999), JPY, GBP, CHF, CAD, AUD, NZD. Month-end spot from
    the Federal Reserve H.10 daily rates (FRED, 1971 on), else BIS end-of-month rates or IMF IFS end-of-period rates
    (DBnomics), else Yahoo (2003 on). Monthly excess return of holding the foreign currency = spot change + (foreign
    - US 3-month interbank rate) / 12, the rates (OECD, FRED or DBnomics) lagged one month; spot only where a rate
    is missing (share reported). Gold: GC=F to 2004-11, GLD after (Yahoo), excess over T-bills.
  S1 Walk-forward seasonality with the 15-of-20 rule (Chandler, "Seasonal patterns in FX: caveat emptor", 2021):
    for each test year from 1991, each currency and calendar month, long the currency in that month if its excess
    return was positive in at least 15 of the previous 20 years, short if in at most 5; equal weight across the
    active positions; 2 bp a position (NZD 3 bp). Test series: the months with at least one position.
  S1b Every month, every currency: the sign of the previous 20 years' mean excess return in that calendar month;
    equal weight across the seven currencies; costs as S1.
  S2 The published patterns, after publication (June 2021 - December 2025): the twelve currency patterns of
    Chandler (2021) with 14+ of 20 years (USD/JPY falls in August; GBP rises in April, falls in May and August; CAD
    rises in April; EUR rises in June and December; AUD rises in February, April and June, falls in May and
    August) and gold's three (rises in January, April and August); one pooled observation per pattern-year, costs
    as S1 (gold 2 bp).
  G1 Gold's turn of the month (Seasonax / StoneX notes): long gold over the last trading day of each month and the
    first three of the next (close of day -2 to close of day +3), excess over T-bills, 2 bp a round trip;
    2000-2025. Also reported: the window's days against all other days (Welch t).
PRE-REGISTERED TESTS (one-sided): mean > 0 for S1, S1b, S2 and G1; t-statistics are Newey-West (3 lags) with the
  variance floored at the iid variance; Benjamini-Hochberg over the four.
Usage: python scripts/lit4_daily_lab.py --out DIR"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = pd.Timestamp("2025-12-31")
CCYS = ("EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD")
# FRED H.10 id and whether it quotes USD per foreign unit (True) or foreign per USD (False)
FRED_FX = {"EUR": ("DEXUSEU", True), "DEM": ("DEXGEUS", False), "JPY": ("DEXJPUS", False),
           "GBP": ("DEXUSUK", True), "CHF": ("DEXSZUS", False), "CAD": ("DEXCAUS", False),
           "AUD": ("DEXUSAL", True), "NZD": ("DEXUSNZ", True)}
BIS_AREA = {"EUR": ("XM", "EUR"), "DEM": ("DE", "DEM"), "JPY": ("JP", "JPY"), "GBP": ("GB", "GBP"),
            "CHF": ("CH", "CHF"), "CAD": ("CA", "CAD"), "AUD": ("AU", "AUD"), "NZD": ("NZ", "NZD")}
IFS_AREA = {"EUR": "U2", "DEM": "DE", "JPY": "JP", "GBP": "GB", "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ"}
YAHOO = {"EUR": ("EURUSD=X", True), "JPY": ("JPY=X", False), "GBP": ("GBPUSD=X", True), "CHF": ("CHF=X", False),
         "CAD": ("CAD=X", False), "AUD": ("AUDUSD=X", True), "NZD": ("NZDUSD=X", True)}
RATE_LOC = {"USD": ("US", "USA"), "EUR": ("EZ", "EA19"), "DEM": ("DE", "DEU"), "JPY": ("JP", "JPN"),
            "GBP": ("GB", "GBR"), "CHF": ("CH", "CHE"), "CAD": ("CA", "CAN"), "AUD": ("AU", "AUS"),
            "NZD": ("NZ", "NZL")}
COST = {c: (3e-4 if c == "NZD" else 2e-4) for c in CCYS} | {"GOLD": 2e-4}
CHANDLER = [("JPY", 8, +1), ("GBP", 4, +1), ("GBP", 5, -1), ("GBP", 8, -1), ("CAD", 4, +1), ("EUR", 6, +1),
            ("EUR", 12, +1), ("AUD", 2, +1), ("AUD", 4, +1), ("AUD", 6, +1), ("AUD", 5, -1), ("AUD", 8, -1),
            ("GOLD", 1, +1), ("GOLD", 4, +1), ("GOLD", 8, +1)]
PUBLISHED = pd.Timestamp("2021-06-01")
SOURCES = {}


def _get(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 fxlab-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def fred(sid, timeout=45):
    try:
        d = pd.read_csv(io.StringIO(_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout)))
        s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(d.iloc[:, 0])
        return s[s.index <= END].dropna()
    except Exception as ex:                                                 # noqa: BLE001
        print("FRED", sid, repr(ex)[:100], flush=True)
        return pd.Series(dtype=float)


def dbnomics(path, timeout=60):
    try:
        J = json.loads(_get(f"https://api.db.nomics.world/v22/series/{path}?observations=1", timeout))
        docs = J.get("series", {}).get("docs", [])
        if not docs:
            return pd.Series(dtype=float)
        per, val = docs[0].get("period", []), docs[0].get("value", [])
        s = pd.Series(pd.to_numeric(pd.Series(val), errors="coerce").to_numpy(),
                      index=pd.PeriodIndex(per, freq="M").to_timestamp(how="end").normalize())
        return s[s.index <= END].dropna()
    except Exception as ex:                                                 # noqa: BLE001
        print("DBnomics", path, repr(ex)[:100], flush=True)
        return pd.Series(dtype=float)


def month_end(s):
    s = s.dropna()
    s = s.groupby(s.index.to_period("M")).last()
    s.index = s.index.to_timestamp(how="end").normalize()
    return s


def usd_value(c):
    """Month-end value of one unit of currency c in USD, and the source used."""
    sid, usd_per = FRED_FX[c]
    s = fred(sid)
    if len(s) > 120:
        s = month_end(s)
        return (s if usd_per else 1.0 / s), f"FRED {sid}"
    area, code = BIS_AREA[c]
    for ds in ("WS_XRU", "XRU"):
        s = dbnomics(f"BIS/{ds}/M.{area}.{code}.E")
        if len(s) > 120:
            return 1.0 / s, f"BIS {ds} M.{area}.{code}.E"
    s = dbnomics(f"IMF/IFS/M.{IFS_AREA[c]}.ENDE_XDC_USD_RATE")
    if len(s) > 120:
        return 1.0 / s, "IMF IFS ENDE"
    if c in YAHOO:
        import lit3_daily_lab as L3
        t, usd_per = YAHOO[c]
        s = L3.yahoo(t, "1999-01-01")
        if len(s):
            s = month_end(s)
            return (s if usd_per else 1.0 / s), f"Yahoo {t}"
    return pd.Series(dtype=float), "none"


def rate(c):
    a2, a3 = RATE_LOC[c]
    s = fred(f"IR3TIB01{a2}M156N")
    src = "FRED"
    if len(s) < 120:
        s, src = dbnomics(f"OECD/MEI/{a3}.IR3TIB01.ST.M"), "OECD MEI (DBnomics)"
    if len(s):
        s.index = s.index.to_period("M").to_timestamp(how="end").normalize()
    return s / 100.0, src


def fx_excess():
    """Monthly excess return of each currency vs USD (index: month end), and the share of months with carry."""
    vals, srcs = {}, {}
    for c in CCYS:
        v, srcs[c] = usd_value(c)
        if c == "EUR":
            dem, srcs["DEM"] = usd_value("DEM")
            if len(dem):
                dem_eur = dem * 1.95583
                v = pd.concat([dem_eur[dem_eur.index < "1999-01-01"], v[v.index >= "1999-01-01"]])
        vals[c] = v.sort_index()
    rates = {}
    for c in ("USD",) + CCYS:
        rates[c], srcs[f"rate {c}"] = rate(c)
    dem_r, srcs["rate DEM"] = rate("DEM")
    if len(dem_r):
        e = rates["EUR"]
        rates["EUR"] = pd.concat([dem_r[dem_r.index < (e.index.min() if len(e) else END)], e]).sort_index()
    SOURCES.update(srcs)
    X, carry_share = {}, {}
    for c in CCYS:
        v = vals[c]
        if not len(v):
            continue
        spot = v.pct_change()
        rd = (rates[c].reindex(spot.index) - rates["USD"].reindex(spot.index)) if len(rates[c]) and len(rates["USD"]) \
            else pd.Series(np.nan, index=spot.index)
        rd = rd.shift(1)                                       # month m-1's average, known at the start of month m
        carry_share[c] = float(rd.reindex(spot.dropna().index).notna().mean())
        X[c] = (spot + (rd / 12).fillna(0.0)).dropna()
    return pd.DataFrame(X), carry_share


def gold_monthly():
    import lit3_daily_lab as L3
    r = L3.gold_series()
    rf = L3.tbill(r.index).shift(1).fillna(0) / 252
    ex = r - rf
    m = (1 + ex).groupby(ex.index.to_period("M")).prod() - 1
    m.index = m.index.to_timestamp(how="end").normalize()
    return m, ex


def nw_t(x, lags=3):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    s0 = (e @ e) / n
    s = s0
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    s = max(s, s0)                                             # never below the iid variance
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def summ(name, x, per_year, **extra):
    x = pd.Series(x, dtype=float).dropna().sort_index()
    t = nw_t(x)
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    late = x[x.index >= "2011-01-01"]
    return {"test": name, "n": int(len(x)), "mean": float(x.mean()) if len(x) else np.nan, "t": t,
            "t_iid": float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan,
            "p_one_sided": float(st.norm.sf(t)) if np.isfinite(t) else np.nan,
            "sharpe": float(x.mean() / x.std() * math.sqrt(per_year)) if len(x) > 2 and x.std() > 0 else np.nan,
            "hit": float((x > 0).mean()) if len(x) else np.nan,
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}", "first": str(x.index.min())[:10] if len(x) else "",
            "mean_2011_2025": float(late.mean()) if len(late) else np.nan, "t_2011_2025": nw_t(late), **extra}


def seasonal_positions(X, lookback=20, first=1991, rule="15of20"):
    """{(month end, currency): position} for every test month; uses only years before the test year."""
    pos = {}
    for c in X.columns:
        x = X[c].dropna()
        by = pd.DataFrame({"y": x.index.year, "m": x.index.month, "x": x.to_numpy()})
        for t in x.index:
            if t.year < first:
                continue
            h = by[(by.m == t.month) & (by.y >= t.year - lookback) & (by.y < t.year)].x
            if len(h) < lookback:
                continue
            if rule == "15of20":
                k = int((h > 0).sum())
                p = 1 if k >= 15 else (-1 if k <= lookback - 15 else 0)
            else:
                p = int(np.sign(h.mean()))
            if p != 0:
                pos[(t, c)] = p
    return pos


def portfolio(X, pos):
    rows = {}
    for (t, c), p in pos.items():
        rows.setdefault(t, []).append(p * X.at[t, c] - COST[c])
    return pd.Series({t: float(np.mean(v)) for t, v in rows.items()}).sort_index()


def s2_published(X, gold_m):
    out = {}
    for c, mth, sgn in CHANDLER:
        x = gold_m if c == "GOLD" else (X[c] if c in X else pd.Series(dtype=float))
        for t, v in x.items():
            if t >= PUBLISHED and t.month == mth and t <= END and np.isfinite(v):
                out[(t, f"{c}{mth:02d}")] = sgn * v - COST[c]
    s = pd.Series(out)
    if len(s):
        s.index = pd.MultiIndex.from_tuples(s.index)
        s = s.droplevel(1).sort_index()
    return s


def g1_tom(ex, cost=2e-4):
    """Per month: gold's excess return over the last trading day and the first three of the next month."""
    d = pd.Series(ex.index, index=ex.index)
    mon = d.dt.to_period("M")
    from_end = mon.groupby(mon).cumcount(ascending=False)
    from_start = mon.groupby(mon).cumcount()
    win = (from_end == 0) | (from_start <= 2)
    # label each window by the month it ends in (the last day of month m joins the first days of m+1)
    lab = mon.where(from_start <= 2, mon + 1)
    w = ex[win].groupby(lab[win]).apply(lambda r: float((1 + r).prod() - 1) - cost)
    n = ex[win].groupby(lab[win]).size()
    w = w[n == 4]
    w.index = w.index.to_timestamp(how="start")
    welch = float(st.ttest_ind(ex[win], ex[~win], equal_var=False).statistic)
    return w, welch, float(ex[win].mean()), float(ex[~win].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, series, info = [], {}, {}
    X, carry_share = fx_excess()
    info["fx_months"] = {c: [str(X[c].dropna().index.min())[:10], int(X[c].notna().sum())] for c in X}
    info["carry_share"] = carry_share
    gold_m, gold_d = gold_monthly()
    for name, rule in (("S1 FX seasonality, walk-forward, 15-of-20 rule", "15of20"),
                       ("S1b FX seasonality, walk-forward, sign of the 20-year mean", "sign")):
        pos = seasonal_positions(X, rule=rule)
        p = portfolio(X, pos)
        series[name] = p
        rows.append(summ(name, p, 12, positions=len(pos)))
        info[name + " positions by currency"] = pd.Series([c for (_, c) in pos]).value_counts().to_dict()
    s2 = s2_published(X, gold_m)
    series["S2 published patterns after publication (2021-06 to 2025-12)"] = s2
    rows.append(summ("S2 published patterns after publication (2021-06 to 2025-12)", s2, 12))
    w, welch, m_in, m_out = g1_tom(gold_d)
    series["G1 gold turn of the month (days -1..+3)"] = w
    rows.append(summ("G1 gold turn of the month (days -1..+3)", w, 12, welch_window_vs_rest=welch,
                     mean_day_in=m_in, mean_day_out=m_out))
    # information: the same 15-of-20 rule on gold (history from 2000 only, test years from 2020)
    gpos = seasonal_positions(pd.DataFrame({"GOLD": gold_m}), first=2020)
    info["gold 15-of-20 positions 2020-2025"] = len(gpos)
    R = pd.DataFrame(rows)
    from cross_market_ta_lab import bh_q
    R["q_bh"] = bh_q(R.p_one_sided.to_numpy())
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame(series).to_csv(os.path.join(a.out, "series.csv"))
    json.dump({"sources": SOURCES, **info}, open(os.path.join(a.out, "info.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string())
    print(json.dumps({"sources": SOURCES, **info}, indent=1, default=str))


if __name__ == "__main__":
    main()
