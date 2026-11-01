"""Third literature round, daily and monthly data (Yahoo closes, the project's point-in-time 3-month rates, the
options-edges branch's per-stock straddle results). Rules fixed before any result; nothing tuned; development
data only (to 2025-12-31).

PRE-REGISTERED TESTS (one-sided, mean net > 0; Newey-West t; Benjamini-Hochberg over D1-D8):
  D1 Composite currency predictor (Bartram, Djuranovik, Garratt & Xu, JFQA 2025): each month-end, for EUR, JPY,
     GBP, CHF, CAD, AUD, NZD against the USD, rank five signals across the seven currencies - carry (3-month rate
     differential), momentum over 1, 3 and 12 months, value (minus the 60-month spot change) - and average the
     percentile ranks (signals with history only); long the two highest, short the two lowest for the next month.
     Return = spot change + rate differential - 0.75% a year financing mark-up; costs 1 bp a side (1.5 bp NZD).
  D2 Equity rebalancing across currencies (Hau & Rey): rank the seven by last month's local equity index return
     minus the S&P 500 (Euro Stoxx 50, Nikkei 225, FTSE 100, SMI, S&P/TSX, S&P/ASX 200, NZX 50); short the two whose
     markets outperformed most, long the two that underperformed most, for the next month. Same returns and costs.
  D3 Gold variance premium: on each month-end sell one-month gold variance at the Cboe gold volatility index (GVZ):
     P&L in vol points (K^2 - RV^2) / 2K, RV = GLD's realised volatility over the next 21 trading days, minus 1
     vol point of costs.
  D4 D3 only when GVZ is above GLD's realised volatility of the past 21 days.
  D5 Gold's autumn effect (Baur 2013): hold gold (GLD; GC=F before 2004-12) in September-November, cash otherwise;
     excess return over T-bills minus 2 bp a switch. Also reported: Sep-Nov minus the other months (Welch t).
  D6 Gold before the Chinese New Year: long gold from the close 15 trading days before the holiday to the last
     close before it; excess return minus 2 bp.
  D7 Variance premium as an equity-timing signal (Bollerslev, Tauchen & Zhou 2009): at each month-end, VRP = VIX^2
     - (21-day realised S&P variance, annualised); hold SPY next month when VRP is above its expanding median, else
     T-bills; the test is the timed return minus buy-and-hold (Newey-West t), and the timed return itself.
  D8 Option momentum (Heston, Jones, Khorram, Li & Mo, JF 2023): each month, rank stocks by the mean return on
     premium of their unhedged ATM straddles over months t-12..t-2 (at least 6); long straddles on the top fifth,
     short straddles on the bottom fifth (each side pays its own costs, as in the options-edges lab), 2019-2025.
Usage: python scripts/lit3_daily_lab.py --out DIR [--positions DIR]"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = pd.Timestamp("2025-12-31")
CCY = {"EUR": ("EURUSD=X", 1), "JPY": ("JPY=X", -1), "GBP": ("GBPUSD=X", 1), "CHF": ("CHF=X", -1),
       "CAD": ("CAD=X", -1), "AUD": ("AUDUSD=X", 1), "NZD": ("NZDUSD=X", 1)}
EQ = {"EUR": "^STOXX50E", "JPY": "^N225", "GBP": "^FTSE", "CHF": "^SSMI", "CAD": "^GSPTSE", "AUD": "^AXJO",
      "NZD": "^NZ50"}
COST = {k: (1.5e-4 if k == "NZD" else 1e-4) for k in CCY}


def yahoo(t, start="1999-01-01"):
    import yfinance as yf
    try:
        d = yf.download(t, start=start, end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    except Exception:                                                       # noqa: BLE001
        return pd.Series(dtype=float)
    d = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    return d[d.index <= END].dropna()


def fred(sid):
    import io
    import requests
    try:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60)
        df = pd.read_csv(io.StringIO(r.text))
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0])
        return s[s.index <= END].dropna()
    except Exception as ex:                                                 # noqa: BLE001
        print("FRED", sid, repr(ex)[:120], flush=True)
        return pd.Series(dtype=float)


def nw_t(x, lags=3):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def summ(name, x, per_year, unit="return", **extra):
    x = pd.Series(x, dtype=float).dropna().sort_index()
    t = nw_t(x)
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    return {"test": name, "unit": unit, "n": int(len(x)), "mean": float(x.mean()) if len(x) else np.nan, "t": t,
            "p_one_sided": float(st.norm.sf(t)) if np.isfinite(t) else np.nan,
            "sharpe": float(x.mean() / x.std() * math.sqrt(per_year)) if len(x) > 2 and x.std() > 0 else np.nan,
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}", "first": str(x.index.min())[:10] if len(x) else "",
            "mean_2015_2025": float(x[x.index >= "2015-01-01"].mean()) if len(x) else np.nan,
            "t_2015_2025": nw_t(x[x.index >= "2015-01-01"]), **extra}


# ----------------------------------------------------------------------------------------------- FX month panel
def fx_panel():
    from fxlab.data import rates as R
    px = {k: yahoo(t) for k, (t, _) in CCY.items()}
    M = pd.DataFrame({k: v.resample("ME").last() for k, v in px.items()})
    for k, (_, sgn) in CCY.items():
        if sgn < 0:
            M[k] = 1.0 / M[k]                                # value of the foreign currency in USD
    kn = R.known_at(pd.DatetimeIndex(M.index).tz_localize("UTC"))
    kn.index = M.index
    rd = pd.DataFrame({k: kn[k] - kn["USD"] for k in CCY}, index=M.index)
    spot = M.pct_change().shift(-1)                          # next month's spot return
    long_r = spot + (rd - 0.0075) / 12
    short_r = -spot + (-rd - 0.0075) / 12
    return M, rd, long_r, short_r


def long_short(scores, long_r, short_r, n=2):
    out, prev = {}, {}
    for t in scores.index:
        s = scores.loc[t].dropna()
        lr, sr = long_r.loc[t], short_r.loc[t]
        if len(s) < 5 or lr.isna().all():
            continue
        o = s.sort_values()
        longs, shorts = list(o.index[-n:]), list(o.index[:n])
        pos = {k: 0.5 for k in longs} | {k: -0.5 for k in shorts}
        r = sum(0.5 * lr[k] for k in longs) + sum(0.5 * sr[k] for k in shorts)
        turn = sum(abs(pos.get(k, 0) - prev.get(k, 0)) * COST[k] for k in CCY)
        if np.isfinite(r):
            out[t] = r - turn
        prev = pos
    return pd.Series(out)


def d1_composite(M, rd, long_r, short_r):
    sig = {"carry": rd, "mom1": M.pct_change(1), "mom3": M.pct_change(3), "mom12": M.pct_change(12),
           "value": -M.pct_change(60)}
    ranks = [s.rank(axis=1, pct=True) for s in sig.values()]
    comp = sum(r.fillna(0) for r in ranks) / sum(r.notna().astype(float) for r in ranks)
    return long_short(comp, long_r, short_r)


def d2_hau_rey(M, long_r, short_r):
    us = yahoo("^GSPC").resample("ME").last().pct_change()
    rel = pd.DataFrame({k: yahoo(t).resample("ME").last().pct_change() for k, t in EQ.items()}).reindex(M.index)
    rel = rel.sub(us.reindex(M.index), axis=0)
    return long_short(-rel, long_r, short_r)                  # outperformers' currencies sold


# ----------------------------------------------------------------------------------------------- gold and equities
def vrp_proxy(K, px, cost, only_rich=False):
    lr = np.log(px).diff()
    out = {}
    for t in K.resample("ME").last().index:
        k = K.loc[:t]
        if not len(k):
            continue
        k0 = float(k.iloc[-1])
        fwd = lr[lr.index > t].iloc[:21].dropna()
        past = lr[lr.index <= t].iloc[-21:].dropna()
        if len(fwd) < 18 or fwd.index[-1] > END or len(past) < 18:
            continue
        if only_rich and k0 <= past.std() * math.sqrt(252) * 100:
            continue
        rv = float(fwd.std() * math.sqrt(252) * 100)
        out[t] = (k0 ** 2 - rv ** 2) / (2 * k0) - cost
    return pd.Series(out)


def tbill(idx):
    irx = yahoo("^IRX")
    return (irx.reindex(idx).ffill().fillna(0) / 100)


def gold_series():
    gld, gc = yahoo("GLD", "2004-01-01"), yahoo("GC=F", "2000-01-01")
    r = gc.pct_change()
    rg = gld.pct_change()
    r = pd.concat([r[r.index < "2004-12-01"], rg[rg.index >= "2004-12-01"]]).dropna()
    return r


def d5_autumn(r):
    rf = tbill(r.index).shift(1).fillna(0) / 252
    ex = (r - rf)
    m = ex.groupby(ex.index.to_period("M")).sum()
    m.index = m.index.to_timestamp("M")
    aut = m[m.index.month.isin([9, 10, 11])]
    rest = m[~m.index.month.isin([9, 10, 11])]
    sw = pd.Series(0.0, index=m.index)
    sw[m.index.month.isin([9, 11])] = 2e-4                    # one switch in at September, one out after November
    strat = (m.where(m.index.month.isin([9, 10, 11]), 0.0) - sw)
    return strat, float(st.ttest_ind(aut, rest, equal_var=False).statistic)


def d6_cny(r):
    from lunardate import LunarDate
    rf = tbill(r.index).shift(1).fillna(0) / 252
    ex = r - rf
    out = {}
    for y in range(2001, 2026):
        cny = pd.Timestamp(LunarDate(y, 1, 1).toSolarDate())
        before = ex[ex.index < cny]
        if len(before) < 20:
            continue
        w = before.iloc[-15:]
        out[cny] = float((1 + w).prod() - 1) - 2e-4
    return pd.Series(out)


def d7_vrp_timing():
    spy, vix = yahoo("SPY", "1993-01-01"), yahoo("^VIX", "1993-01-01")
    r = spy.pct_change().dropna()
    rf = tbill(r.index).shift(1).fillna(0) / 252
    me = r.resample("ME").last().index
    rows = {}
    hist = []
    for t in me:
        past = r[r.index <= t].iloc[-21:]
        v = vix[vix.index <= t]
        nxt = r[(r.index > t) & (r.index <= t + pd.offsets.MonthEnd(1))]
        if len(past) < 18 or not len(v) or not len(nxt):
            continue
        vrp = (float(v.iloc[-1]) / 100) ** 2 - past.var() * 252
        if len(hist) >= 36:
            on = vrp > np.median(hist)
            ret_bh = float((1 + nxt).prod() - 1)
            ret_rf = float((1 + rf.reindex(nxt.index).fillna(0)).prod() - 1)
            rows[t] = (ret_bh if on else ret_rf, ret_bh, on)
        hist.append(vrp)
    D = pd.DataFrame.from_dict(rows, orient="index", columns=["timed", "bh", "on"])
    sw = D.on.astype(int).diff().abs().fillna(0) * 2e-4
    D["timed"] -= sw
    return D


def d8_option_momentum(pos_dir):
    import options_edges_lab as OE
    fs = glob.glob(os.path.join(pos_dir, "**", "positions_*.csv.gz"), recursive=True)
    if not fs:
        return None
    P = pd.concat([pd.read_csv(f, parse_dates=["entry", "expiry"]) for f in fs], ignore_index=True)
    P = P[~P.symbol.isin(["SPY", "DIA", "XLF", "XLE"]) & (P.expiry <= END)]
    P["date"] = P.entry.dt.to_period("M").dt.to_timestamp()
    lu = P[P.kind == "long_straddle_unhedged"].copy()
    lu["rp"] = lu.pnl / lu.premium
    R = lu.pivot_table(index="date", columns="symbol", values="rp", aggfunc="mean").sort_index()
    sig = R.shift(2).rolling(11, min_periods=6).mean()           # months t-12..t-2
    S = sig.stack().rename("mom").reset_index()
    M = P.merge(S, on=["date", "symbol"], how="inner")
    return OE.quintile_two_books(M, "mom", "long_straddle_unhedged", "short_straddle_unhedged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, series = [], {}

    def add(name, x, per_year, **kw):
        series[name] = x
        rows.append(summ(name, x, per_year, **kw))
        print(rows[-1], flush=True)
    try:
        M, rd, lr, sr = fx_panel()
        add("D1 composite currency predictor (carry, momentum 1/3/12, value)", d1_composite(M, rd, lr, sr), 12)
        add("D2 Hau-Rey equity rebalancing across currencies", d2_hau_rey(M, lr, sr), 12)
    except Exception as ex:                                                 # noqa: BLE001
        print("FX error", repr(ex)[:300], flush=True)
    gvz = yahoo("^GVZ", "2008-01-01")
    if not len(gvz):
        gvz = fred("GVZCLS")
    gld = yahoo("GLD", "2004-01-01")
    if len(gvz) and len(gld):
        add("D3 gold variance premium at GVZ (vol points)", vrp_proxy(gvz, gld, 1.0), 12, unit="vol points")
        add("D4 D3 only when GVZ is above past realised vol", vrp_proxy(gvz, gld, 1.0, True), 12, unit="vol points")
    else:
        print("no GVZ / GLD data", flush=True)
    try:
        g = gold_series()
        strat, welch = d5_autumn(g)
        add("D5 gold autumn effect (Sep-Nov long, cash otherwise)", strat, 12, autumn_minus_rest_welch_t=welch)
        add("D6 gold 15 days before the Chinese New Year", d6_cny(g), 1)
    except Exception as ex:                                                 # noqa: BLE001
        print("gold error", repr(ex)[:300], flush=True)
    try:
        D = d7_vrp_timing()
        add("D7 VRP equity timing, timed minus buy-and-hold", D.timed - D.bh, 12, share_months_in=float(D.on.mean()))
        add("D7b VRP equity timing, timed return", D.timed, 12)
    except Exception as ex:                                                 # noqa: BLE001
        print("VRP error", repr(ex)[:300], flush=True)
    if a.positions:
        x = d8_option_momentum(a.positions)
        if x is not None:
            add("D8 option momentum (straddles, months t-12..t-2)", x, 12)
    R = pd.DataFrame(rows)
    from cross_market_ta_lab import bh_q
    prim = ~R.test.str.startswith("D7b")
    R["q_bh"] = np.nan
    R.loc[prim, "q_bh"] = bh_q(R.loc[prim, "p_one_sided"].to_numpy())
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame(series).to_csv(os.path.join(a.out, "series.csv"))
    pd.set_option("display.width", 250)
    print(R.round(4).to_string())


if __name__ == "__main__":
    main()
