"""Daily-data tests of every remaining idea (MQL5 candidates and the free batch), rules fixed in advance (nothing
tuned), free data only, everything cut at 2025-12-31 (2026 stays locked).

PRE-REGISTERED TESTS (one-sided unless marked; Benjamini-Hochberg over the whole batch in the combined report):
  D1  RSI(2) dip-buying (the PropDip rule) on SPY, QQQ, DIA, IWM: at the close, RSI(2) < 10 and close > 200-day
      average -> buy that close; sell at the first close above the 5-day average, at a close 3 x ATR(14) below the
      entry, or after 20 trading days. 1 bp per side, excess over the T-bill. Pooled trades; 1993-2018 and
      2019-2025 reported. Control: the same holding periods started on random days (same ETF, same year).
  D1b D1 but every position closed at the Friday close (no weekends).
  D2  TLT month-end: long from the close 6 trading days before the month's last trading day to that last close
      (the last 5 days); excess over T-bill; 1 bp per side. 2002-2025.
  D3  COT speculative pressure (Fan, Fernandez-Perez, Fuertes & Miffre 2020) on EUR, GBP, JPY, CHF, CAD, AUD, NZD:
      each month-end, the 52-week mean of non-commercial (long - short) / (long + short) from the CFTC legacy
      futures-only report (the last one released by month-end: as-of Tuesday + 3 days), cross-sectionally standardised; weights
      proportional, long and short legs 50% each; the next month's spot return against USD plus the 3-month rate
      differential; 2 bp per unit of turnover. 2007-2025; post-sample June 2018 - December 2025 reported.
  D4  Volatility-managed SPY (Moreira & Muir 2017): monthly weight c / (previous month's realised variance), c set
      so the full-sample average weight is 1 (Sharpe does not depend on c), capped at 2; 1 bp per unit traded;
      alpha against SPY (one-sided).
  D5  Overnight (close to next open, Monday-Thursday nights) on DAX, FTSE 100, CAC 40, Euro Stoxx 50, Nikkei 225,
      Hang Seng: 1 bp per round trip; also net of index-CFD financing (T-bill + 2.5% a year per night).
  D6  BTC calendar (Yahoo daily, UTC): (a) Saturday + Sunday only; (b) Monday only; (c) turn of the month (last day
      + first 3 days); 10 bp per round trip; excess over T-bill.
  D7  BTC 14/56-day moving-average trend: long when SMA14 > SMA56 at the close, traded next close, else T-bills;
      10 bp per switch; alpha against buy-and-hold BTC.
  D8  BTC / ETH cash-and-carry on Binance and Bybit perpetuals (funding history), as M14 / M15 on Deribit: long spot,
      short the perpetual, 10 bp per leg in and out; monthly P&L. Run only if the venue answers the runner.
  D9  Commodity seasonality (Heston & Sadka form) on 10 commodity ETFs: each month-end, the average return of the
      coming calendar month in all earlier years (at least 3); long the top 3, short the bottom 3; 5 bp per side.
  D10 Factor ETFs MTUM, QUAL, USMV, VLUE, SIZE: CAPM alpha against SPY, each and equal-weight (one-sided).
  D10b Russell reconstitution (two-sided, descriptive): IWM minus SPY from the close 5 days before to the close of the
      reconstitution day (last Friday of June), and from that close to 5 days after.
  D11 Currency value (5-year reversal) on the same 7 currencies: long the 3 with the lowest 5-year spot return
      against USD, short the 3 highest; monthly; carry included; 2 bp per unit of turnover.
  D12 Cross-asset into FX, next-day sign rules entered at the FX close AFTER the signal day (one day skipped so
      that the signal is known): (a) oil up -> short USDCAD; (b) SPY up -> long AUDJPY; (c) VIX up -> short AUDJPY;
      (d) gold up -> long AUDUSD; (e) US 10-year yield up -> long USDJPY. 1 bp per side (2 bp AUDJPY). 2007-2025.
  D13 Earnings-announcement premium (Frazzini & Lamont 2007; Savor & Wilson 2016) on the S&P 100 names of the
      options lab: long each stock from the close 5 trading days before its earnings date to the close of the day
      after, minus SPY over the same days; 5 bp per side; monthly average.
  D14 Gold multi-length Donchian (the "Gold Channel Rider" family) on GLD: 20 / 55 / 100-day breakouts (long and
      short) with exits on the opposite 10 / 20 / 50-day channel, equal weight; 2 bp per side; against 200 sets of
      random entries with the same exits and holding rules (one-sided: real mean above the random 95th percentile).
Usage: python scripts/untested_edges_lab.py --stage data --out DIR [--chain chains]
       python scripts/untested_edges_lab.py --stage tests --data DIR --out DIR"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
END = pd.Timestamp("2025-12-31")
RSI_ETFS = ["SPY", "QQQ", "DIA", "IWM"]
IDX_O = ["^GDAXI", "^FTSE", "^FCHI", "^STOXX50E", "^N225", "^HSI"]
COMMOD = ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC", "CORN", "WEAT", "SOYB"]
FACTORS = ["MTUM", "QUAL", "USMV", "VLUE", "SIZE"]
FXY = {"EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPY=X", "CHF": "CHF=X", "CAD": "CAD=X", "AUD": "AUDUSD=X",
       "NZD": "NZDUSD=X"}
INVERT = {"JPY", "CHF", "CAD"}          # Yahoo quotes these as USD per ... -> invert to USD value of the currency
CROSS = ["CL=F", "GC=F", "^TNX", "^VIX", "AUDJPY=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "TLT", "GLD", "BTC-USD",
         "ETH-USD", "^IRX", "SPY", "IWM"]
COT_CODES = {"EUR": "099741", "GBP": "096742", "JPY": "097741", "CHF": "092741", "CAD": "090741", "AUD": "232741",
             "NZD": "112741"}


# ----------------------------------------------------------------------------------------------- data
def yahoo(symbols, start="1990-01-01"):
    import yfinance as yf
    out = {}
    for s in symbols:
        h = pd.DataFrame()
        for k in range(3):
            try:
                h = yf.download(s, start=start, end="2026-01-01", progress=False, auto_adjust=False)
                if h is not None and len(h):
                    break
            except Exception:                                          # noqa: BLE001
                time.sleep(3)
        if h is None or h.empty:
            print("no data", s, flush=True)
            continue
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) else pd.to_datetime(h.index)
        h = h[h.index <= END]
        out[s] = h[[c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in h.columns]].astype(float)
        print(s, len(h), h.index[0].date(), flush=True)
    return out


def fetch(url, timeout=90, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as ex:                                        # noqa: BLE001
            err = repr(ex)[:200]
            time.sleep(2 ** k)
    raise RuntimeError(err)


def cot_legacy():
    """CFTC legacy futures-only COT, all years, the currency contracts only."""
    frames = []
    urls = ["https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"] + \
           [f"https://www.cftc.gov/files/dea/history/deacot{y}.zip" for y in range(2017, 2026)]
    for u in urls:
        try:
            z = zipfile.ZipFile(io.BytesIO(fetch(u)))
            for n in z.namelist():
                df = pd.read_csv(z.open(n), low_memory=False)
                frames.append(df)
            print("cot", u, "ok", flush=True)
        except Exception as ex:                                        # noqa: BLE001
            print("cot", u, "failed", ex, flush=True)
    if not frames:
        return pd.DataFrame()
    D = pd.concat(frames, ignore_index=True)
    cols = {c: c.strip() for c in D.columns}
    D = D.rename(columns=cols)
    code = [c for c in D.columns if "Contract Market Code" in c][0]
    date = [c for c in D.columns if "YYYY-MM-DD" in c][0]
    lo = [c for c in D.columns if c.startswith("Noncommercial Positions-Long")][0]
    sh = [c for c in D.columns if c.startswith("Noncommercial Positions-Short")][0]
    D = D[[code, date, lo, sh]].copy()
    D.columns = ["code", "date", "long", "short"]
    D["code"] = D.code.astype(str).str.strip().str.zfill(6)
    D["date"] = pd.to_datetime(D.date, errors="coerce")
    inv = {v: k for k, v in COT_CODES.items()}
    D = D[D.code.isin(inv)].dropna()
    D["ccy"] = D.code.map(inv)
    return D[D.date <= END].drop_duplicates(["ccy", "date"]).sort_values(["ccy", "date"])


def funding_binance(sym):
    rows, t = [], int(pd.Timestamp("2019-09-01").timestamp() * 1000)
    end = int(pd.Timestamp("2026-01-01").timestamp() * 1000)
    while t < end:
        js = json.loads(fetch(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={t}&limit=1000"))
        if not js:
            break
        rows += js
        t = int(js[-1]["fundingTime"]) + 1
        if len(js) < 1000:
            break
    D = pd.DataFrame(rows)
    D["time"] = pd.to_datetime(D.fundingTime, unit="ms")
    D["rate"] = D.fundingRate.astype(float)
    return D.set_index("time")[["rate"]]


def funding_bybit(sym):
    rows, end = [], int(pd.Timestamp("2026-01-01").timestamp() * 1000)
    while True:
        js = json.loads(fetch(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={sym}"
                              f"&endTime={end}&limit=200"))
        lst = js.get("result", {}).get("list", [])
        if not lst:
            break
        rows += lst
        end = int(lst[-1]["fundingRateTimestamp"]) - 1
        if len(lst) < 200 or end < int(pd.Timestamp("2019-01-01").timestamp() * 1000):
            break
    D = pd.DataFrame(rows)
    D["time"] = pd.to_datetime(D.fundingRateTimestamp.astype("int64"), unit="ms")
    D["rate"] = D.fundingRate.astype(float)
    return D.set_index("time")[["rate"]].sort_index()


def stage_data(a):
    os.makedirs(a.out, exist_ok=True)
    syms = sorted(set(RSI_ETFS + IDX_O + COMMOD + FACTORS + list(FXY.values()) + CROSS))
    for s, h in yahoo(syms).items():
        h.to_csv(os.path.join(a.out, f"y_{s.replace('^', 'I_').replace('=', '_')}.csv"))
    D = cot_legacy()
    D.to_csv(os.path.join(a.out, "cot_fx.csv"), index=False)
    print("cot rows", len(D), flush=True)
    for venue, fn in (("binance", funding_binance), ("bybit", funding_bybit)):
        for sym in ("BTCUSDT", "ETHUSDT"):
            try:
                F = fn(sym)
                F.to_csv(os.path.join(a.out, f"funding_{venue}_{sym}.csv"))
                print(venue, sym, len(F), F.index.min(), F.index.max(), flush=True)
            except Exception as ex:                                    # noqa: BLE001
                print(venue, sym, "unavailable", ex, flush=True)
                open(os.path.join(a.out, f"funding_{venue}_{sym}.unavailable"), "w").write(str(ex))
    if a.chain:
        for f in ("closes.csv", "earnings.csv"):
            p = os.path.join(a.chain, f)
            if os.path.exists(p):
                import shutil
                shutil.copy(p, os.path.join(a.out, f))


# ----------------------------------------------------------------------------------------------- stats
def nw_t(x, lag=None):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 8 or x.std() == 0:
        return np.nan
    lag = lag if lag is not None else int(4 * (n / 100) ** (2 / 9))
    e = x - x.mean()
    s = e @ e / n
    for L_ in range(1, lag + 1):
        s += 2 * (1 - L_ / (lag + 1)) * (e[L_:] @ e[:-L_]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def summ(x, per_year, label=None):
    x = pd.Series(x).dropna()
    if len(x) < 8:
        return {"n": int(len(x))}
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    t = nw_t(x)
    return {"n": int(len(x)), "mean_pct": float(x.mean() * 100), "t": t, "p_one_sided": float(1 - st.norm.cdf(t)),
            "sharpe": float(x.mean() / x.std() * math.sqrt(per_year)), "worst_pct": float(x.min() * 100),
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}", "first": str(x.index.min())[:10],
            "last": str(x.index.max())[:10]}


def load(dirn, s):
    p = os.path.join(dirn, f"y_{s.replace('^', 'I_').replace('=', '_')}.csv")
    if not os.path.exists(p):
        return None
    h = pd.read_csv(p, index_col=0, parse_dates=True)
    return h[h.index <= END]


def rf_daily(data):
    irx = load(data, "^IRX")
    if irx is None:
        return pd.Series(dtype=float)
    return (irx["Close"] / 100 / 252).rename("rf")


def rsi(c, n=2):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ----------------------------------------------------------------------------------------------- tests
def t_rsi2(data, rf, no_weekend=False, seed=5):
    trades, ctrl = [], []
    rng = np.random.default_rng(seed)
    for s in RSI_ETFS:
        h = load(data, s)
        if h is None:
            continue
        c = h["Adj Close"].dropna()
        hi, lo, cl = h["High"] * c / h["Close"], h["Low"] * c / h["Close"], c
        tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        r2, sma200, sma5 = rsi(c), c.rolling(200).mean(), c.rolling(5).mean()
        idx = c.index
        r_rf = rf.reindex(idx).ffill().fillna(0)
        i = 200
        n = len(idx)
        while i < n - 1:
            if r2.iloc[i] < 10 and c.iloc[i] > sma200.iloc[i]:
                e = c.iloc[i]
                stop = e - 3 * atr.iloc[i]
                j = i + 1
                while j < n:
                    if c.iloc[j] > sma5.iloc[j] or c.iloc[j] < stop or j - i >= 20:
                        break
                    if no_weekend and idx[j].weekday() == 4:
                        break
                    j += 1
                j = min(j, n - 1)
                if no_weekend and idx[i].weekday() == 4:          # a Friday entry would be held over the weekend
                    i += 1
                    continue
                ret = c.iloc[j] / e - 1 - 2e-4 - r_rf.iloc[i + 1:j + 1].sum()
                trades.append({"etf": s, "entry": idx[i], "exit": idx[j], "days": j - i, "ret": ret})
                # control: same holding length from a random day of the same year
                yr = np.where(idx.year == idx[i].year)[0]
                yr = yr[(yr >= 200) & (yr < n - (j - i) - 1)]
                if len(yr):
                    k = int(rng.choice(yr))
                    ctrl.append({"etf": s, "entry": idx[k], "ret": c.iloc[k + j - i] / c.iloc[k] - 1 - 2e-4 -
                                 r_rf.iloc[k + 1:k + j - i + 1].sum(), "days": j - i})
                i = j + 1
            else:
                i += 1
    T, Cc = pd.DataFrame(trades), pd.DataFrame(ctrl)
    if T.empty:
        return {}, None
    x = T.set_index("entry").ret.sort_index()
    out = {"all": summ(x, len(x) / max((x.index.max() - x.index.min()).days / 365.25, 1)),
           "bp_per_trade": float(x.mean() * 1e4),
           "control_bp_per_trade": float(Cc.ret.mean() * 1e4) if len(Cc) else np.nan,
           "minus_control_t": float(st.ttest_ind(T.ret, Cc.ret, equal_var=False).statistic) if len(Cc) > 10 else np.nan,
           "dev_1993_2018_bp": float(x[x.index < "2019-01-01"].mean() * 1e4),
           "dev_t": nw_t(x[x.index < "2019-01-01"]),
           "hold_2019_2025_bp": float(x[x.index >= "2019-01-01"].mean() * 1e4),
           "hold_t": nw_t(x[x.index >= "2019-01-01"]),
           "trades": int(len(T)), "mean_days": float(T.days.mean()),
           "weekend_share": float(((T.exit - T.entry).dt.days > (T.days * 1.0)).mean())}
    by = T.groupby("etf").ret.agg(["count", "mean"])
    out["by_etf_bp"] = {k: round(v * 1e4, 1) for k, v in by["mean"].items()}
    m = x.groupby(x.index.to_period("M")).sum()
    return out, m


def t_tlt_month_end(data, rf):
    h = load(data, "TLT")
    if h is None:
        return {}, None
    c = h["Adj Close"].dropna()
    r_rf = rf.reindex(c.index).ffill().fillna(0)
    last = c.groupby(c.index.to_period("M")).apply(lambda s: s.index[-1])
    rows = []
    pos = {d: i for i, d in enumerate(c.index)}
    for d in last:
        i = pos[d]
        if i < 6 or d.month == 12 and d.year == 2025 and d < END - pd.Timedelta(days=3):
            continue
        r = c.iloc[i] / c.iloc[i - 5] - 1 - 2e-4 - r_rf.iloc[i - 4:i + 1].sum()
        rows.append((d, r))
    x = pd.Series(dict(rows)).sort_index()
    # permutation: random 5-day windows, same count per year
    rng = np.random.default_rng(3)
    null = []
    years = x.index.year
    for _ in range(2000):
        s = 0.0
        for y in np.unique(years):
            ii = np.where(c.index.year == y)[0]
            ii = ii[ii >= 5]
            k = rng.choice(ii, size=int((years == y).sum()))
            s += float(np.sum(c.iloc[k].to_numpy() / c.iloc[k - 5].to_numpy() - 1 - 2e-4))
        null.append(s / len(x))
    out = summ(x, 12)
    out["perm_p"] = float((np.array(null) >= x.mean()).mean())
    out["y2023_2024_pct"] = float(x[(x.index.year >= 2023) & (x.index.year <= 2024)].mean() * 100)
    return out, x


def fx_monthly(data):
    """Monthly USD-based spot return of each currency and the 3-month rate differential (point in time)."""
    from fxlab.data import rates as R
    px = {}
    for c, s in FXY.items():
        h = load(data, s)
        if h is None:
            continue
        v = h["Close"].dropna()
        v = v[v > 0]
        px[c] = 1 / v if c in INVERT else v
    P = pd.DataFrame(px)
    Pm = P.groupby(P.index.to_period("M")).last()
    spot = Pm.pct_change()
    rm = R.monthly().copy()                                   # cached: never modify in place
    rm.index = rm.index.tz_localize(None).to_period("M")
    known = rm.shift(1)                                         # month m's average is known from month m+1
    carry = pd.DataFrame({c: (known[c] - known["USD"]) / 12 for c in P.columns if c in known})
    return spot, carry.reindex(spot.index)


def ls_returns(W, spot, carry, cost=2e-4):
    """W: weights decided at the end of month m (index m) -> returns of month m+1."""
    W = W.reindex(spot.index).fillna(0.0)
    nxt = (spot + carry.fillna(0)).shift(-1)
    gross = (W * nxt).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    r = (gross - turn * cost).shift(1)                         # realised in month m+1
    return r[W.abs().sum(axis=1).shift(1) > 0].dropna()


def t_cot(data):
    p = os.path.join(data, "cot_fx.csv")
    if not os.path.exists(p):
        return {}, None
    D = pd.read_csv(p, parse_dates=["date"])
    if D.empty:
        return {"error": "no COT data"}, None
    D["x"] = (D["long"] - D["short"]) / (D["long"] + D["short"])
    sp = D.pivot_table(index="date", columns="ccy", values="x").sort_index()
    sp52 = sp.rolling(52, min_periods=40).mean()
    sp52.index = sp52.index + pd.Timedelta(days=3)            # as-of Tuesday, released on the Friday
    m = sp52.groupby(sp52.index.to_period("M")).last()          # last report RELEASED by the end of month m
    spot, carry = fx_monthly(data)
    m = m.reindex(spot.index)
    z = m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1), axis=0)
    pos, neg = z.clip(lower=0), (-z).clip(lower=0)
    W = 0.5 * pos.div(pos.sum(axis=1), axis=0) - 0.5 * neg.div(neg.sum(axis=1), axis=0)
    W = W[(m.notna().sum(axis=1) >= 5)]
    r = ls_returns(W, spot, carry)
    r.index = r.index.to_timestamp()
    r = r[(r.index >= "2007-01-01") & (r.index <= END)]
    out = summ(r, 12)
    post = r[r.index >= "2018-06-01"]
    out["post_sample_2018_06"] = summ(post, 12)
    return out, r


def t_currency_value(data):
    spot, carry = fx_monthly(data)
    lvl = (1 + spot.fillna(0)).cumprod()
    five = lvl / lvl.shift(60) - 1
    W = pd.DataFrame(0.0, index=five.index, columns=five.columns)
    for d, row in five.iterrows():
        row = row.dropna()
        if len(row) >= 6:
            o = row.sort_values()
            W.loc[d, o.index[:3]] = 1 / 3
            W.loc[d, o.index[-3:]] = -1 / 3
    r = ls_returns(W, spot, carry)
    r.index = r.index.to_timestamp()
    return summ(r[r.index <= END], 12), r


def t_vol_managed(data, rf):
    h = load(data, "SPY")
    c = h["Adj Close"].dropna()
    d = c.pct_change().dropna()
    rfd = rf.reindex(d.index).ffill().fillna(0)
    ex = d - rfd
    mo = ex.groupby(ex.index.to_period("M")).apply(lambda s: (1 + s).prod() - 1)
    rv = d.groupby(d.index.to_period("M")).apply(lambda s: (s ** 2).sum())
    w = (1 / rv).shift(1)
    w = w / w.mean()
    w = w.clip(upper=2.0)
    man = w * mo - w.diff().abs().fillna(0) * 1e-4
    j = pd.concat([man.rename("y"), mo.rename("x")], axis=1).dropna()
    b = np.cov(j.y, j.x)[0, 1] / j.x.var()
    res = j.y - b * j.x
    out = {"managed": summ(j.y.set_axis(j.index.to_timestamp()), 12), "spy": summ(j.x.set_axis(j.index.to_timestamp()), 12),
           "beta": float(b), "alpha_month_pct": float(res.mean() * 100), "t": nw_t(res),
           "p_one_sided": float(1 - st.norm.cdf(nw_t(res)))}
    s = res.copy()
    s.index = s.index.to_timestamp()
    return out, s


def t_overnight_indices(data, rf):
    outs, series = {}, {}
    for s in IDX_O:
        h = load(data, s)
        if h is None:
            continue
        o, c = h["Open"], h["Close"]
        same_open = float((o == c.shift(1)).mean())
        night = (o / c.shift(1) - 1)
        weekday_night = h.index.to_series().diff().dt.days == 1
        x = night[weekday_night].dropna()
        x = x[(x.abs() < 0.2)]
        gross = x - 1e-4
        rfn = rf.reindex(x.index).ffill().fillna(0) * 252 / 360
        net = gross - (rfn + 0.025 / 360)
        outs[s] = {"gross": summ(gross, 252), "net_cfd": summ(net, 252), "share_open_equals_prev_close": same_open}
        series[s] = gross
    if series:
        P = pd.DataFrame(series).mean(axis=1)
        outs["pooled"] = summ(P, 252)
    return outs, series


def t_btc_calendar(data, rf):
    h = load(data, "BTC-USD")
    c = h["Close"].dropna()
    r = c.pct_change().dropna()
    rfd = rf.reindex(r.index).ffill().fillna(0) * 252 / 365
    ex = r - rfd
    wd = r.index.weekday
    out, ser = {}, {}
    wk = ex[(wd >= 5)]
    wk = wk.groupby(wk.index.to_period("W")).sum() - 1e-3
    out["a_weekend"] = summ(wk.set_axis(wk.index.to_timestamp()), 52)
    mon = ex[wd == 0] - 1e-3
    out["b_monday"] = summ(mon, 52)
    dom = r.index.day
    last = r.index.to_series().groupby(r.index.to_period("M")).transform("max") == r.index.to_series()
    tom = ex[(dom <= 3) | last.to_numpy()]
    tom = tom.groupby((tom.index + pd.Timedelta(days=4)).to_period("M")).sum() - 1e-3
    out["c_turn_of_month"] = summ(tom.set_axis(tom.index.to_timestamp()), 12)
    ser = {"a_weekend": wk, "b_monday": mon, "c_turn_of_month": tom}
    return out, ser


def t_btc_trend(data, rf):
    h = load(data, "BTC-USD")
    c = h["Close"].dropna()
    r = c.pct_change()
    rfd = rf.reindex(c.index).ffill().fillna(0) * 252 / 365
    sig = (c.rolling(14).mean() > c.rolling(56).mean()).astype(float)
    pos = sig.shift(2)                                    # decided at the close, executed at the next close
    strat = pos * r + (1 - pos) * rfd - pos.diff().abs().fillna(0) * 1e-3
    ex = (strat - rfd).dropna()[56:]
    bh = (r - rfd).reindex(ex.index)
    mo = ex.groupby(ex.index.to_period("M")).sum()
    mb = bh.groupby(bh.index.to_period("M")).sum()
    b = np.cov(mo, mb)[0, 1] / mb.var()
    res = mo - b * mb
    out = {"strategy": summ(mo.set_axis(mo.index.to_timestamp()), 12),
           "buy_hold": summ(mb.set_axis(mb.index.to_timestamp()), 12),
           "beta": float(b), "alpha_month_pct": float(res.mean() * 100), "t": nw_t(res),
           "p_one_sided": float(1 - st.norm.cdf(nw_t(res)))}
    return out, res.set_axis(res.index.to_timestamp())


def t_carry_venues(data):
    out, ser = {}, {}
    for venue in ("binance", "bybit"):
        for sym in ("BTCUSDT", "ETHUSDT"):
            p = os.path.join(data, f"funding_{venue}_{sym}.csv")
            if not os.path.exists(p):
                out[f"{venue}_{sym}"] = {"unavailable": True}
                continue
            F = pd.read_csv(p, index_col=0, parse_dates=True)
            F = F[F.index <= END]
            m = F.rate.groupby(F.index.to_period("M")).sum()
            m.iloc[0] -= 2 * 1e-3
            m.iloc[-1] -= 2 * 1e-3
            m.index = m.index.to_timestamp()
            out[f"{venue}_{sym}"] = summ(m, 12)
            ser[f"{venue}_{sym}"] = m
    return out, ser


def t_commodity_season(data):
    rets = {}
    for s in COMMOD:
        h = load(data, s)
        if h is None:
            continue
        c = h["Adj Close"].dropna()
        rets[s] = c.groupby(c.index.to_period("M")).last().pct_change()
    R = pd.DataFrame(rets)
    W = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    for i, m in enumerate(R.index):
        nxt = (m + 1).month
        hist = R.iloc[:i + 1]
        hist = hist[hist.index.month == nxt]
        sc = hist.mean()[hist.count() >= 3]
        if len(sc) >= 6:
            o = sc.sort_values()
            W.loc[m, o.index[-3:]] = 1 / 3
            W.loc[m, o.index[:3]] = -1 / 3
    nxt = R.shift(-1)
    gross = (W * nxt).sum(axis=1)
    turn = W.diff().abs().sum(axis=1)
    r = (gross - turn * 5e-4).shift(1)
    r = r[W.abs().sum(axis=1).shift(1) > 0].dropna()
    r.index = r.index.to_timestamp()
    return summ(r[r.index <= END], 12), r


def t_factors(data):
    spy = load(data, "SPY")["Adj Close"]
    sm = spy.groupby(spy.index.to_period("M")).last().pct_change()
    out, res_all = {}, []
    for s in FACTORS:
        h = load(data, s)
        if h is None:
            continue
        c = h["Adj Close"].dropna()
        m = c.groupby(c.index.to_period("M")).last().pct_change()
        j = pd.concat([m.rename("y"), sm.rename("x")], axis=1).dropna()
        b = np.cov(j.y, j.x)[0, 1] / j.x.var()
        res = (j.y - b * j.x)
        out[s] = {"beta": float(b), "alpha_month_pct": float(res.mean() * 100), "t": nw_t(res), "months": int(len(j))}
        res_all.append(res.rename(s))
    if res_all:
        ew = pd.concat(res_all, axis=1).mean(axis=1)
        ew.index = ew.index.to_timestamp()
        out["equal_weight"] = summ(ew, 12)
        return out, ew
    return out, None


def t_russell(data):
    iwm, spy = load(data, "IWM")["Adj Close"], load(data, "SPY")["Adj Close"]
    rel = (iwm.pct_change() - spy.pct_change()).dropna()
    rows = []
    for y in range(2001, 2026):
        june = pd.date_range(f"{y}-06-01", f"{y}-06-30")
        fr = [d for d in june if d.weekday() == 4]
        d = fr[-1]
        idx = rel.index[rel.index <= d]
        if not len(idx):
            continue
        i = rel.index.get_loc(idx[-1])
        rows.append({"year": y, "pre": float(rel.iloc[i - 4:i + 1].sum()), "post": float(rel.iloc[i + 1:i + 6].sum())})
    D = pd.DataFrame(rows)
    return {"pre_mean_pct": float(D.pre.mean() * 100), "pre_t": float(st.ttest_1samp(D.pre, 0).statistic),
            "post_mean_pct": float(D.post.mean() * 100), "post_t": float(st.ttest_1samp(D.post, 0).statistic),
            "years": int(len(D))}, None


def t_cross_asset(data):
    def c_(s):
        h = load(data, s)
        return None if h is None else h["Close"].dropna()
    pairs = {"a_oil_to_USDCAD": ("CL=F", "USDCAD=X", -1, 1e-4), "b_SPY_to_AUDJPY": ("SPY", "AUDJPY=X", 1, 2e-4),
             "c_VIX_to_AUDJPY": ("^VIX", "AUDJPY=X", -1, 2e-4), "d_gold_to_AUDUSD": ("GC=F", "AUDUSD=X", 1, 1e-4),
             "e_yield_to_USDJPY": ("^TNX", "USDJPY=X", 1, 1e-4)}
    out, ser = {}, {}
    for name, (sig_s, fx_s, sign, cost) in pairs.items():
        s, f = c_(sig_s), c_(fx_s)
        if s is None or f is None:
            continue
        sr = s.diff() if sig_s in ("^TNX", "^VIX") else s.pct_change()
        fr = f.pct_change()
        sr, fr = sr[sr.index >= "2007-01-01"], fr[fr.index >= "2007-01-01"]
        dates = fr.index
        sg = np.sign(sr).reindex(dates, method="ffill")
        pos = (sign * sg).shift(2)                         # signal day t -> held from FX close t+1 to t+2
        r = (pos * fr - pos.diff().abs().fillna(0) * cost).dropna()
        out[name] = summ(r, 252)
        ser[name] = r
    return out, ser


def t_earnings_premium(data):
    pc, pe = os.path.join(data, "closes.csv"), os.path.join(data, "earnings.csv")
    if not (os.path.exists(pc) and os.path.exists(pe)):
        return {"error": "no closes / earnings"}, None
    X = pd.read_csv(pc, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    E = pd.read_csv(pe)
    E.columns = [c.lower() for c in E.columns]
    E["date"] = pd.to_datetime(E["date"], errors="coerce")
    spy = X["SPY_adj"].dropna()
    rows = []
    for sym, g in E.groupby("act_symbol"):
        col = f"{sym}_adj"
        if col not in X.columns:
            continue
        p = X[col].dropna()
        if len(p) < 300:
            continue
        idx = p.index
        for d in g.date.dropna().unique():
            d = pd.Timestamp(d)
            if d > END - pd.Timedelta(days=5) or d < idx[0] + pd.Timedelta(days=30):
                continue
            j = idx.searchsorted(d)
            if j < 6 or j + 1 >= len(idx):
                continue
            i0, i1 = j - 5, j + 1
            r = p.iloc[i1] / p.iloc[i0] - 1
            m = spy.reindex([idx[i0], idx[i1]])
            if m.isna().any():
                continue
            rows.append({"symbol": sym, "date": d, "ret": r - (m.iloc[1] / m.iloc[0] - 1) - 2 * 5e-4})
    D = pd.DataFrame(rows)
    if D.empty:
        return {"error": "no events"}, None
    mo = D.groupby(D.date.dt.to_period("M")).ret.mean()
    mo.index = mo.index.to_timestamp()
    out = summ(mo, 12)
    out.update({"events": int(len(D)), "mean_event_pct": float(D.ret.mean() * 100)})
    return out, mo


def t_gold_donchian(data, n_random=200):
    h = load(data, "GLD")
    c = h["Adj Close"].dropna()
    r = c.pct_change().fillna(0)
    systems = [(20, 10), (55, 20), (100, 50)]

    def run(entries_long, entries_short, ex_n):
        pos = np.zeros(len(c))
        lo_x = c.rolling(ex_n).min().shift(1).to_numpy()
        hi_x = c.rolling(ex_n).max().shift(1).to_numpy()
        cc = c.to_numpy()
        p = 0
        for i in range(len(c)):
            if p == 1 and cc[i] < lo_x[i]:
                p = 0
            elif p == -1 and cc[i] > hi_x[i]:
                p = 0
            if p == 0:
                if entries_long[i]:
                    p = 1
                elif entries_short[i]:
                    p = -1
            pos[i] = p
        ps = pd.Series(pos, index=c.index).shift(1).fillna(0)
        return ps * r - ps.diff().abs().fillna(0) * 2e-4

    real = []
    rng = np.random.default_rng(9)
    rand_means = np.zeros(n_random)
    for n, ex_n in systems:
        hi = c.rolling(n).max().shift(1)
        lo = c.rolling(n).min().shift(1)
        el, es = (c > hi).to_numpy(), (c < lo).to_numpy()
        real.append(run(el, es, ex_n))
        rate = (el.sum() + es.sum()) / len(c)
        for k in range(n_random):
            u = rng.random(len(c))
            rl = u < rate / 2
            rs = (u >= rate / 2) & (u < rate)
            rand_means[k] += float(run(rl, rs, ex_n).iloc[100:].mean()) / len(systems)
    port = pd.concat(real, axis=1).mean(axis=1).iloc[100:]
    mo = port.groupby(port.index.to_period("M")).sum()
    mo.index = mo.index.to_timestamp()
    out = summ(mo, 12)
    out["random_p"] = float((rand_means >= port.mean()).mean())
    out["random_mean_bp_day"] = float(rand_means.mean() * 1e4)
    out["real_mean_bp_day"] = float(port.mean() * 1e4)
    return out, mo


def stage_tests(a):
    os.makedirs(a.out, exist_ok=True)
    rf = rf_daily(a.data)
    res, rows, series = {}, [], {}

    def add(test, family, stats, s=None, primary=True):
        res[test] = stats
        flat = stats if "t" in stats else {}
        rows.append({"test": test, "family": family, "primary": primary, "n": flat.get("n"),
                     "mean_pct": flat.get("mean_pct", flat.get("alpha_month_pct")), "t": flat.get("t"),
                     "p_one_sided": flat.get("p_one_sided", (float(1 - st.norm.cdf(flat["t"])) if np.isfinite(flat.get("t", np.nan)) else np.nan)),
                     "sharpe": flat.get("sharpe"), "years_positive": flat.get("years_positive")})
        if s is not None:
            series[test] = s

    for name, fn in [("D1 RSI(2) dip-buying, 4 ETFs", lambda: t_rsi2(a.data, rf)),
                     ("D1b RSI(2), no weekends", lambda: t_rsi2(a.data, rf, no_weekend=True))]:
        try:
            o, m = fn()
            s = o.get("all", {})
            add(name, "D1", {**s, **{k: v for k, v in o.items() if k != "all"}}, m)
        except Exception as ex:                                          # noqa: BLE001
            res[name] = {"error": repr(ex)[:300]}
    simple = [("D2 TLT month-end", "D2", lambda: t_tlt_month_end(a.data, rf)),
              ("D3 COT speculative pressure, 7 currencies", "D3", lambda: t_cot(a.data)),
              ("D4 Volatility-managed SPY (alpha)", "D4", lambda: t_vol_managed(a.data, rf)),
              ("D7 BTC 14/56 trend (alpha vs buy-and-hold)", "D7", lambda: t_btc_trend(a.data, rf)),
              ("D9 Commodity ETF seasonality", "D9", lambda: t_commodity_season(a.data)),
              ("D10 Factor ETFs, equal-weight alpha", "D10", lambda: t_factors(a.data)),
              ("D11 Currency value (5-year reversal)", "D11", lambda: t_currency_value(a.data)),
              ("D13 Earnings-announcement premium, S&P 100", "D13", lambda: t_earnings_premium(a.data)),
              ("D14 Gold multi-length Donchian vs random entries", "D14", lambda: t_gold_donchian(a.data))]
    for name, fam, fn in simple:
        try:
            o, s = fn()
            if fam == "D10":
                add(name, fam, o.get("equal_weight", {}), s)
                res[name + " (each)"] = {k: v for k, v in o.items() if k != "equal_weight"}
            elif fam == "D14":
                o["p_one_sided"] = o.get("random_p", np.nan)
                add(name, fam, o, s)
            else:
                add(name, fam, o, s)
        except Exception as ex:                                          # noqa: BLE001
            res[name] = {"error": repr(ex)[:300]}
    for name, fam, fn, key in [("D5 Overnight, other indices", "D5", lambda: t_overnight_indices(a.data, rf), "gross"),
                               ("D6 BTC calendar", "D6", lambda: t_btc_calendar(a.data, rf), None),
                               ("D8 Crypto carry, other venues", "D8", lambda: t_carry_venues(a.data), None),
                               ("D12 Cross-asset into FX", "D12", lambda: t_cross_asset(a.data), None)]:
        try:
            o, ser = fn()
            res[name] = o
            for k, v in o.items():
                st_ = v.get(key, v) if key else v
                if isinstance(st_, dict) and "t" in st_:
                    add(f"{name}: {k}", fam, st_, ser.get(k) if isinstance(ser, dict) else None, primary=(k != "pooled"))
        except Exception as ex:                                          # noqa: BLE001
            res[name] = {"error": repr(ex)[:300]}
    try:
        o, _ = t_russell(a.data)
        res["D10b Russell reconstitution (two-sided)"] = o
    except Exception as ex:                                              # noqa: BLE001
        res["D10b Russell reconstitution (two-sided)"] = {"error": repr(ex)[:300]}
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    for k, s in series.items():
        if s is None:
            continue
        pd.Series(s).to_csv(os.path.join(a.out, "series_" + "".join(ch if ch.isalnum() else "_" for ch in k)[:60] + ".csv"))
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(json.dumps(res, indent=1, default=str)[:20000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "tests"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--chain", default="")
    a = ap.parse_args()
    stage_data(a) if a.stage == "data" else stage_tests(a)


if __name__ == "__main__":
    main()
