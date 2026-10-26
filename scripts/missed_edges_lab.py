"""Edges this project had not tested, from the literature, with parameters fixed in advance (nothing tuned), on free
data: Yahoo daily prices (ETFs, indices, FX, the S&P 100 closes of the options lab), Deribit perpetual funding
history, DoltHub earnings dates. Development data only: everything is cut at 2025-12-31.

PRE-REGISTERED TESTS (one-sided, mean > 0; Benjamini-Hochberg over M1-M16; sub-periods reported):
  M1  Cross-asset time-series momentum (Moskowitz, Ooi & Pedersen 2012) on 22 ETFs (US / world / emerging equity,
      Treasuries, credit, TIPS, gold, silver, oil, gas, agriculture, commodities, dollar, euro, yen, pound, REITs):
      each month-end, the sign of the asset's 12-month excess return, sized to 40% / sqrt(252 x 60-day variance) / N
      (EWMA vol, centre of mass 60 days); excess returns over the 13-week T-bill; 5 bp per unit traded; shorts pay
      0.5% a year to borrow. 2008-2025.
  M2  SPY overnight: buy the close, sell the next open (Cliff, Cooper & Gulen 2008); 1 bp per round trip.
  M3  QQQ overnight, the same.     M4  IWM overnight, the same.
  M5  Turn of the month, SPY (Lakonishok & Smidt 1988): long from the close of the second-to-last trading day of
      the month to the close of the third trading day of the next; excess over T-bill; 1 bp per round trip.
  M6  Pre-holiday, SPY (Ariel 1990): the close-to-close return of the last trading day before a US market holiday;
      excess; 1 bp per round trip.
  M7  Pre-FOMC drift, SPY (Lucca & Moench 2015): close before to close of each scheduled FOMC statement day
      (2019-2025); excess; 1 bp.
  M8  Short volatility through SVXY when VIX < VIX3M at the prior close (Simon & Campasano 2014: the VIX futures
      roll), otherwise T-bills; 3 bp per switch. 2012-2025.
  M9  SVXY held always (no filter), excess over T-bill. 2012-2025.
  M10 EUR/USD variance premium (Della Corte, Ramadorai & Sarno 2016): each month-end sell one-month variance at EVZ
      against the realised volatility of EUR/USD over the next 21 trading days, 0.5 vol point cost. 2008-2025.
  M11 Earnings-announcement drift (Chan, Jegadeesh & Lakonishok 1996; the announcement-return form of PEAD) on the
      S&P 100: each month-end, stocks that announced in that month ranked by the announcement return (close before
      to close after, minus SPY); long the top fifth, short the bottom fifth for the next month; 5 bp per side.
  M12 Cross-sectional momentum (Jegadeesh & Titman 1993) on the S&P 100: 12-1 month return, long top / short bottom
      fifth, monthly; 5 bp per side. 2008-2025. (Today's members: survivorship bias, favourable to momentum.)
  M13 Low volatility (Frazzini & Pedersen 2014 form, beta-neutral): long the lowest 60-day-volatility fifth, short
      the highest, each leg scaled to beta 1 against SPY; monthly; 5 bp per side. 2008-2025.
  M14 BTC cash-and-carry: long spot, short the Deribit perpetual, collecting funding every hour; 10 bp per leg to
      enter and to exit; P&L per month.            M15 ETH, the same.
  M16 FX daily mean reversion portfolio (the project's best pre-2019 near-miss, docs/cloud_sweep.md 8.1c):
      Bollinger(20, 2) entry / middle-band exit, signal at the close, traded at the next close, on 19 pairs,
      equal risk; 1 bp per side on majors, 2 bp on crosses, 0.3 bp a night. 2019-2025 (new data for this rule).
Usage: python scripts/missed_edges_lab.py --stage data --out DIR [--closes chains/closes.csv --earnings chains/earnings.csv]
       python scripts/missed_edges_lab.py --stage tests --data DIR --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats as st

END = pd.Timestamp("2025-12-31")
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "SLV", "USO", "UNG",
        "DBA", "DBC", "UUP", "FXE", "FXY", "FXB", "VNQ"]
IDX = ["^VIX", "^VIX3M", "^IRX", "^EVZ", "SVXY"]
FX = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X", "USDSEK=X", "USDNOK=X",
      "EURGBP=X", "EURJPY=X", "EURCHF=X", "GBPJPY=X", "AUDJPY=X", "EURSEK=X", "EURNOK=X", "AUDNZD=X", "EURCAD=X",
      "GBPCHF=X"]
MAJORS = {"EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X"}
FOMC = ["2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30",
        "2019-12-11", "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
        "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22",
        "2021-11-03", "2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
        "2022-09-21", "2022-11-02", "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20", "2024-05-01",
        "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19",
        "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"]


# ----------------------------------------------------------------------------------------------- data
def yahoo(symbols, start="1993-01-01"):
    import yfinance as yf
    out = {}
    for s in symbols:
        for k in range(3):
            try:
                h = yf.download(s, start=start, end="2026-01-01", progress=False, auto_adjust=False, actions=True)
                break
            except Exception:                                          # noqa: BLE001
                h = pd.DataFrame()
                time.sleep(3)
        if h is None or h.empty:
            print("no data", s, flush=True)
            continue
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) else pd.to_datetime(h.index)
        h = h[h.index <= END]
        cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Dividends") if c in h.columns]
        out[s] = h[cols].astype(float)
        print(s, len(h), h.index[0].date(), flush=True)
    return out


def get_json(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:                                              # noqa: BLE001
            time.sleep(2 ** k)
    return None


def deribit_funding(inst):
    """Hourly funding of a Deribit perpetual (interest_1h), paging by 30 days."""
    rows = []
    t = pd.Timestamp("2019-01-01")
    while t < pd.Timestamp("2026-01-01"):
        t1 = min(t + pd.Timedelta(days=30), pd.Timestamp("2026-01-01"))
        js = get_json("https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name="
                      f"{inst}&start_timestamp={int(t.timestamp() * 1000)}&end_timestamp={int(t1.timestamp() * 1000)}")
        if js and js.get("result"):
            rows += js["result"]
        t = t1
    if not rows:
        return pd.DataFrame()
    D = pd.DataFrame(rows).drop_duplicates("timestamp")
    D["time"] = pd.to_datetime(D.timestamp, unit="ms")
    return D.set_index("time").sort_index()


def stage_data(a):
    os.makedirs(a.out, exist_ok=True)
    allp = yahoo(ETFS + IDX + FX)
    for s, h in allp.items():
        h.to_csv(os.path.join(a.out, f"y_{s.replace('^', 'I_').replace('=', '_')}.csv"))
    for inst in ("BTC-PERPETUAL", "ETH-PERPETUAL"):
        D = deribit_funding(inst)
        print(inst, len(D), D.index[0] if len(D) else None, flush=True)
        if len(D):
            D.to_csv(os.path.join(a.out, f"funding_{inst}.csv"))
    meta = {"yahoo": {s: [str(h.index[0].date()), str(h.index[-1].date()), int(len(h))] for s, h in allp.items()}}
    json.dump(meta, open(os.path.join(a.out, "data_meta.json"), "w"), indent=1)


def load(d, s):
    f = os.path.join(d, f"y_{s.replace('^', 'I_').replace('=', '_')}.csv")
    if not os.path.exists(f):
        return None
    h = pd.read_csv(f, index_col=0, parse_dates=True)
    return h[h.index <= END]


# ----------------------------------------------------------------------------------------------- statistics
def nw_t(x, lag=1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan
    e = x - x.mean()
    s = e @ e / n + sum(2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n for L in range(1, lag + 1))
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def stats(x, per_year, split=pd.Timestamp("2019-01-01")):
    x = pd.Series(x).dropna()
    x = x[x.index <= END]
    if len(x) < 8:
        return {"n": int(len(x))}
    t = nw_t(x.to_numpy(), lag=1 if per_year <= 52 else 5)
    yrs = x.groupby(x.index.year).sum()
    a, b = x[x.index < split], x[x.index >= split]
    return {"n": int(len(x)), "first": str(x.index[0].date()), "mean": float(x.mean()), "t_nw": t,
            "p_one_sided": float(st.norm.sf(t)) if np.isfinite(t) else np.nan,
            "sharpe": float(x.mean() / x.std() * math.sqrt(per_year)) if x.std() > 0 else np.nan,
            "ann_return": float(x.mean() * per_year), "win_rate": float((x > 0).mean()), "worst": float(x.min()),
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}",
            "mean_before_2019": float(a.mean()) if len(a) else np.nan,
            "t_before_2019": nw_t(a.to_numpy()) if len(a) >= 8 else np.nan,
            "mean_2019_2025": float(b.mean()) if len(b) else np.nan,
            "t_2019_2025": nw_t(b.to_numpy()) if len(b) >= 8 else np.nan,
            **{f"y{y}": float(v) for y, v in yrs.items()}}


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pp = p[ok]
        o = np.argsort(pp)
        r = pp[o] * len(pp) / (np.arange(len(pp)) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty(len(pp))
        qq[o] = np.minimum(r, 1)
        q[ok] = qq
    return q


# ----------------------------------------------------------------------------------------------- tests
def tbill_daily(d):
    h = load(d, "^IRX")
    if h is None:
        return None
    r = (h["Close"] / 100.0).clip(lower=0)
    return (r / 252.0).rename("rf")


def tr(h):
    """Daily total return from the adjusted close."""
    return h["Adj Close"].pct_change()


def month_end_idx(idx):
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().to_numpy())


def m1_tsmom(d, rf):
    R = {}
    for s in ETFS:
        h = load(d, s)
        if h is not None:
            R[s] = tr(h)
    R = pd.DataFrame(R).dropna(how="all")
    rfd = rf.reindex(R.index).ffill().fillna(0.0)
    X = R.sub(rfd, axis=0)                                              # excess returns
    var = (X ** 2).ewm(com=60, min_periods=60).mean()
    vol = np.sqrt(var * 252)
    cum = np.log1p(X.fillna(0)).cumsum()
    me = month_end_idx(X.index)
    me = me[(me >= pd.Timestamp("2007-12-31")) & (me < END)]
    out, prev_w = {}, pd.Series(0.0, index=X.columns)
    for i, t in enumerate(me[:-1]):
        t1 = me[i + 1]
        lb = cum.loc[:t]
        if len(lb) < 253:
            continue
        sig = np.sign(lb.iloc[-1] - lb.iloc[-253])
        v = vol.loc[t]
        ok = sig.notna() & v.notna() & (v > 0) & X.loc[:t].iloc[-253:].notna().all()
        n = int(ok.sum())
        if n < 5:
            continue
        w = (sig[ok] * 0.40 / v[ok] / n).reindex(X.columns).fillna(0.0)
        period = X.loc[(X.index > t) & (X.index <= t1)]
        gross = float(((1 + period.fillna(0)).prod() - 1).mul(w).sum())
        cost = float((w - prev_w).abs().sum() * 5e-4)
        borrow = float(w.clip(upper=0).abs().sum() * 0.005 * len(period) / 252)
        out[t1] = gross - cost - borrow
        prev_w = w
    return pd.Series(out)


def overnight(d, s, rf=None, cost=1e-4):
    h = load(d, s)
    if h is None:
        return None, None
    f = h["Adj Close"] / h["Close"]                                    # adjustment factor of the day
    o, c = h["Open"] * f, h["Adj Close"]
    on = (o / c.shift(1) - 1) - cost                                    # buy close, sell next open (dividend incl.)
    intra = c / o - 1
    return on.dropna(), intra.dropna()


def trading_days(h):
    return pd.DatetimeIndex(h.index)


def m5_turn_of_month(d, rf):
    h = load(d, "SPY")
    r = tr(h)
    rfd = rf.reindex(r.index).ffill().fillna(0)
    days = trading_days(h)
    per = days.to_period("M")
    out = {}
    months = sorted(set(per))
    for i in range(len(months) - 1):
        cur = days[per == months[i]]
        nxt = days[per == months[i + 1]]
        if len(cur) < 3 or len(nxt) < 3:
            continue
        hold = days[(days > cur[-2]) & (days <= nxt[2])]
        x = r.reindex(hold) - rfd.reindex(hold)
        out[nxt[2]] = float((1 + x).prod() - 1) - 1e-4
    return pd.Series(out)


def m6_pre_holiday(d, rf):
    h = load(d, "SPY")
    r = tr(h)
    rfd = rf.reindex(r.index).ffill().fillna(0)
    days = trading_days(h)
    out = {}
    bdays = pd.bdate_range(days[0], days[-1])
    missing = set(bdays) - set(days)                                   # weekday market holidays
    for i, t in enumerate(days[:-1]):
        nb = t + pd.offsets.BDay(1)
        if nb in missing:
            out[t] = float(r.loc[t] - rfd.loc[t]) - 1e-4
    return pd.Series(out).dropna()


def m7_pre_fomc(d, rf):
    h = load(d, "SPY")
    r = tr(h)
    rfd = rf.reindex(r.index).ffill().fillna(0)
    out = {}
    for x in FOMC:
        t = pd.Timestamp(x)
        if t in r.index:
            out[t] = float(r.loc[t] - rfd.loc[t]) - 1e-4
    return pd.Series(out)


def m8_m9_svxy(d, rf):
    h, v, v3 = load(d, "SVXY"), load(d, "^VIX"), load(d, "^VIX3M")
    if h is None or v is None or v3 is None:
        return None, None
    r = tr(h)
    rfd = rf.reindex(r.index).ffill().fillna(0)
    cont = (v["Close"] < v3["Close"]).reindex(r.index).ffill().shift(1).fillna(False).astype(bool)
    pos = cont.astype(float)
    switch = pos.diff().abs().fillna(0)
    filt = pos * (r - rfd) - switch * 3e-4
    filt = filt[filt.index >= pd.Timestamp("2012-01-01")]
    always = (r - rfd)[r.index >= pd.Timestamp("2012-01-01")]
    to_m = lambda x: x.groupby(x.index.to_period("M")).apply(lambda y: float((1 + y).prod() - 1)).to_timestamp("M")  # noqa: E731
    return to_m(filt.dropna()), to_m(always.dropna())


def m10_evz(d):
    e, fx = load(d, "^EVZ"), load(d, "EURUSD=X")
    if e is None or fx is None or not len(e):
        return None
    K = e["Close"].dropna()
    lr = np.log(fx["Close"]).diff()
    out = {}
    for t in month_end_idx(K.index):
        if t < pd.Timestamp("2008-01-01"):
            continue
        k0 = float(K.loc[:t].iloc[-1])
        fwd = lr.loc[lr.index > t].iloc[:21].dropna()
        if len(fwd) < 18 or fwd.index[-1] > END:
            continue
        rv = float(fwd.std() * math.sqrt(252) * 100)
        out[t] = (k0 ** 2 - rv ** 2) / (2 * k0) - 0.5
    return pd.Series(out)


def quintile_ls(scores, fwd, cost=5e-4, beta=None):
    """Per date: mean forward return of the top fifth minus the bottom fifth by score (optionally beta-scaled legs),
    minus 2 x cost per side for full turnover."""
    out = {}
    for t, s in scores.items():
        s = s.dropna()
        if len(s) < 20 or t not in fwd.index:
            continue
        f = fwd.loc[t].reindex(s.index)
        ok = f.notna()
        s, f = s[ok], f[ok]
        if len(s) < 20:
            continue
        q = s.rank(pct=True)
        top, bot = f[q > 0.8], f[q <= 0.2]
        if beta is not None:
            b = beta.loc[t].reindex(s.index)
            bt, bb = b[q > 0.8].mean(), b[q <= 0.2].mean()
            if not (np.isfinite(bt) and np.isfinite(bb) and bt > 0.2 and bb > 0.2):      # no 5x+ leverage on a leg
                continue
            out[t] = top.mean() / bt - bot.mean() / bb - 4 * cost
        else:
            out[t] = top.mean() - bot.mean() - 4 * cost
    return pd.Series(out)


def stock_panel(closes):
    X = pd.read_csv(closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    adj = X[[c for c in X.columns if c.endswith("_adj")]]
    adj.columns = [c[:-4] for c in adj.columns]
    return adj


def m11_ear(closes, earnings):
    P = stock_panel(closes)
    spy = P["SPY"]
    E = pd.read_csv(earnings)
    E.columns = [c.lower() for c in E.columns]
    E["date"] = pd.to_datetime(E["date"], errors="coerce")
    E = E.dropna(subset=["date"])
    E = E[(E.date <= END) & E.act_symbol.isin(P.columns) & ~E.act_symbol.isin(["SPY", "DIA", "XLF", "XLE"])]
    idx = P.index
    rows = []
    for sym, dt in zip(E.act_symbol, E.date):
        i = idx.searchsorted(dt)
        if i < 1 or i + 1 >= len(idx):
            continue
        p0, p1 = P[sym].iloc[i - 1], P[sym].iloc[i + 1]
        s0, s1 = spy.iloc[i - 1], spy.iloc[i + 1]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            rows.append({"symbol": sym, "date": idx[i + 1], "ar": p1 / p0 - s1 / s0})
    A = pd.DataFrame(rows)
    if not len(A):
        return None, 0
    me = month_end_idx(idx)
    mret = P.reindex(me).pct_change().shift(-1)                         # next month's return, indexed by month-end
    scores = {}
    for t in me:
        a = A[(A.date > t - pd.offsets.MonthEnd(1)) & (A.date <= t)]
        if len(a):
            scores[t] = a.groupby("symbol").ar.last()
    return quintile_ls(scores, mret), len(A)


def m12_m13_xs(closes):
    P = stock_panel(closes)
    stocks = [c for c in P.columns if c not in ("SPY", "DIA", "XLF", "XLE", "QQQ", "IWM", "GLD", "SLV", "TLT", "EEM")]
    me = month_end_idx(P.index)
    me = me[me >= pd.Timestamp("2007-12-31")]
    M = P.reindex(me)
    mret = M.pct_change().shift(-1)
    mom = (M.shift(1) / M.shift(12) - 1)[stocks]
    dr = P.pct_change()
    vol = dr[stocks].rolling(60, min_periods=50).std().reindex(me)
    spy = dr["SPY"]
    cov = dr[stocks].rolling(252, min_periods=200).cov(spy)
    beta = cov.div(spy.rolling(252, min_periods=200).var(), axis=0).reindex(me)
    m12 = quintile_ls({t: mom.loc[t] for t in me}, mret[stocks])
    m13 = quintile_ls({t: -vol.loc[t] for t in me}, mret[stocks], beta=beta)
    return m12, m13


def m14_carry(d, inst):
    f = os.path.join(d, f"funding_{inst}.csv")
    if not os.path.exists(f):
        return None
    D = pd.read_csv(f, index_col=0, parse_dates=True)
    D = D[D.index <= END + pd.Timedelta(hours=23)]
    if "interest_1h" not in D.columns or not len(D):
        return None
    fund = D["interest_1h"].astype(float)                               # short perp receives positive funding
    m = fund.groupby(fund.index.to_period("M")).sum().to_timestamp("M")
    m.iloc[0] -= 2 * 10e-4                                              # enter spot and perp
    m.iloc[-1] -= 2 * 10e-4                                             # exit both
    return m


def m16_fx_mr(d):
    rets = {}
    for s in FX:
        h = load(d, s)
        if h is None:
            continue
        c = h["Close"].dropna()
        c = c[c.index >= pd.Timestamp("2018-09-01")]
        ma, sd = c.rolling(20).mean(), c.rolling(20).std()
        pos = pd.Series(0.0, index=c.index)
        p = 0.0
        for i, t in enumerate(c.index):
            if not np.isfinite(sd.iloc[i]):
                pos.iloc[i] = 0.0
                continue
            x = c.iloc[i]
            if p == 0:
                p = 1.0 if x < ma.iloc[i] - 2 * sd.iloc[i] else (-1.0 if x > ma.iloc[i] + 2 * sd.iloc[i] else 0.0)
            elif (p > 0 and x >= ma.iloc[i]) or (p < 0 and x <= ma.iloc[i]):
                p = 0.0
            pos.iloc[i] = p
        r = c.pct_change()
        held = pos.shift(1).fillna(0)                                   # traded at the next close
        cost = (1e-4 if s in MAJORS else 2e-4)
        vol = r.rolling(60, min_periods=40).std()
        scale = (0.10 / math.sqrt(252)) / vol.shift(1)
        pnl = held * scale * r - held.diff().abs().fillna(0) * scale * cost - held.abs() * scale * 0.3e-4
        rets[s] = pnl[pnl.index >= pd.Timestamp("2019-01-01")]
    if not rets:
        return None, 0
    R = pd.DataFrame(rets).fillna(0)
    port = R.mean(axis=1)
    return port.groupby(port.index.to_period("M")).sum().to_timestamp("M"), len(rets)


def stage_tests(a):
    os.makedirs(a.out, exist_ok=True)
    d = a.data
    rf = tbill_daily(d)
    if rf is None:
        rf = pd.Series(dtype=float)
    tests, series, notes = [], {}, {}

    def reg(name, x, per_year, **extra):
        if x is None or not len(x):
            tests.append({"test": name, "n": 0, **extra})
            return
        x = x.sort_index()
        series[name] = x
        tests.append({"test": name, **stats(x, per_year), **extra})

    reg("M1 Cross-asset time-series momentum, 22 ETFs", m1_tsmom(d, rf), 12)
    for k, s in (("M2", "SPY"), ("M3", "QQQ"), ("M4", "IWM")):
        on, intra = overnight(d, s)
        reg(f"{k} {s} overnight (close to next open), 1 bp", on, 252,
            intraday_mean=float(intra.mean()) if intra is not None else np.nan,
            gross_overnight_mean=float(on.mean() + 1e-4) if on is not None else np.nan)
    reg("M5 Turn of the month, SPY", m5_turn_of_month(d, rf), 12)
    reg("M6 Pre-holiday, SPY", m6_pre_holiday(d, rf), 9)
    reg("M7 Pre-FOMC day, SPY (2019-2025)", m7_pre_fomc(d, rf), 8)
    f, al = m8_m9_svxy(d, rf)
    reg("M8 Short volatility (SVXY) only in VIX contango", f, 12)
    reg("M9 SVXY held always", al, 12)
    reg("M10 EUR/USD variance premium at EVZ (vol points)", m10_evz(d), 12)
    if a.closes and os.path.exists(a.closes):
        if a.earnings and os.path.exists(a.earnings):
            x, n = m11_ear(a.closes, a.earnings)
            reg("M11 Earnings-announcement drift, S&P 100 (long-short)", x, 12, announcements=n)
        else:
            reg("M11 Earnings-announcement drift, S&P 100 (long-short)", None, 12)
        m12, m13 = m12_m13_xs(a.closes)
        reg("M12 Cross-sectional momentum 12-1, S&P 100 (long-short)", m12, 12)
        reg("M13 Low volatility, beta-neutral, S&P 100 (long-short)", m13, 12)
    reg("M14 BTC cash-and-carry (Deribit perpetual funding)", m14_carry(d, "BTC-PERPETUAL"), 12)
    reg("M15 ETH cash-and-carry (Deribit perpetual funding)", m14_carry(d, "ETH-PERPETUAL"), 12)
    x, n = m16_fx_mr(d)
    reg("M16 FX daily Bollinger mean reversion, 19 pairs (2019-2025)", x, 12, pairs=n)
    T = pd.DataFrame(tests)
    T["bh_q"] = bh(T["p_one_sided"].to_numpy()) if "p_one_sided" in T else np.nan
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    if series:
        pd.concat(series, axis=1).to_csv(os.path.join(a.out, "series.csv"))
    json.dump({"tests": T.to_dict("records")}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 260)
    cols = ["test", "n", "first", "mean", "t_nw", "sharpe", "ann_return", "win_rate", "worst", "years_positive",
            "t_before_2019", "t_2019_2025", "bh_q"]
    print(T[[c for c in cols if c in T.columns]].round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("data", "tests"), required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--closes", default="")
    ap.add_argument("--earnings", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "data":
        stage_data(a)
    else:
        stage_tests(a)


if __name__ == "__main__":
    main()
