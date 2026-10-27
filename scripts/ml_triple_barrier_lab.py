"""Entries as a machine-learning problem with the triple barrier as the trade (Lopez de Prado): an upper barrier
(take-profit), a lower barrier (stop) and a vertical barrier (time). Gradient boosting learns which barrier a
trade opened at the next bar touches first; the strategy takes the model's strongest calls and is managed by the
same three barriers. One instrument and one timeframe per run, on every timeframe from M1 to D1.
Development data only: the loader refuses anything after 2025-12-31.

FIXED BEFORE ANY RESULT:
  Data: M1 bid/ask bars of the store (the FX-Data mirror for FX and gold, 2007-2018 with EURUSD to mid-2022;
    Dukascopy 2019-2025 for the US500 and NAS100 CFDs, cash-session hours only), aggregated to the timeframe.
  Barriers (three pre-registered shapes, in ATR(14) of the timeframe's bars, horizon in bars):
    B1 symmetric  take-profit +1.5 ATR, stop -1.5 ATR, 12 bars  (primary)
    B2 trend      take-profit +2.0 ATR, stop -1.0 ATR, 24 bars
    B3 reversion  take-profit +1.0 ATR, stop -2.0 ATR, 12 bars
  Label: for each bar t, a long opened at the open of t+1; class 2 = the upper barrier is touched first within the
    horizon, class 0 = the lower barrier first, class 1 = the vertical barrier (or both in one bar). Shorts use the
    mirrored reading of the same classes.
  Features (known at the close of t): returns over 1-200 bars scaled by 20-bar volatility; RSI(2), RSI(14);
    internal bar strength; distance to the 5/20/50/200-bar averages and to the 20/55/250-bar highs and lows in ATRs;
    the 50-bar average's slope; Bollinger %b; ATR / price; 5/20/60-bar volatility and their ratio; 60-bar skew and
    kurtosis; the up/down streak; the bar's range in ATRs; the quoted spread in ATRs and its 100-bar z-score; the tick
    count's 100-bar z-score; hour (UTC) and weekday.
  Model: LightGBM multiclass, 300 trees, learning rate 0.05, 31 leaves, 200 samples per leaf, 80% row and column
    subsampling, seed 0; nothing tuned. Training rows are every bar when the timeframe has fewer than 20,000 bars
    a year and every horizon-th bar otherwise (non-overlapping labels).
  Walk-forward: one test year at a time from the fourth year of data. Model B learns from all bars whose horizon
    ends before the test year; thresholds come from model A, trained on the bars before the previous year and scored
    on that year: the 90th percentile of P(up) - P(down) for longs and of P(down) - P(up) for shorts.
  Trading: at the close of t, with no open position, long if P(up) - P(down) beats its threshold and 0, else short
    by the mirror rule; the position is opened at the next bar's open and closed at the first barrier, with the
    M1 path deciding the order (stop first within one minute); costs, slippage, commission and swap as in the flows
    lab (FX: the rate-differential swap; gold and index CFDs: the 3-month USD rate plus a 2.5% mark-up a night).
  Control: 20 sets of random entries per test year with the model's number of trades and long/short mix and the
    same barriers, compared on net and on gross (mid-price) returns: barriers in ATR units make costs a smaller share
    of a trade when volatility is high, so a model can beat random entries on net by choosing volatile times alone;
    the gross comparison measures direction skill.
PRE-REGISTERED TESTS (one-sided): for each instrument x timeframe x barrier shape, mean net return per trade > 0
  (t clustered by month) and above the random entries (p = share of draws with a mean at least as high);
  Benjamini-Hochberg over all of them in the combine stage.
Usage: python scripts/ml_triple_barrier_lab.py --symbol EURUSD --tf H1 --out DIR
       python scripts/ml_triple_barrier_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = "2025-12-31 23:59"
BARRIERS = {"B1": (1.5, 1.5, 12), "B2": (2.0, 1.0, 24), "B3": (1.0, 2.0, 12)}
PIPS = {"USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01, "XAUUSD": 0.01, "USA500IDXUSD": 0.1, "USATECHIDXUSD": 0.1}
CFD = {"XAUUSD", "USA500IDXUSD", "USATECHIDXUSD"}
MARKUP = 0.025
LGB = dict(objective="multiclass", num_class=3, n_estimators=300, learning_rate=0.05, num_leaves=31,
           min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
           verbose=-1, n_jobs=4)


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def features(b: pd.DataFrame):
    o, h, l, c = (0.5 * (b[f"bid_{k}"] + b[f"ask_{k}"]) for k in "ohlc")
    lr = np.log(c).diff()
    v20 = lr.rolling(20, min_periods=15).std()
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    F = pd.DataFrame(index=b.index)
    for k in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        F[f"r{k}"] = np.log(c / c.shift(k)) / (v20 * math.sqrt(k) + 1e-12)
    F["rsi2"], F["rsi14"] = rsi(c, 2), rsi(c, 14)
    F["ibs"] = (c - l) / (h - l + 1e-12)
    for n in (5, 20, 50, 200):
        F[f"dma{n}"] = (c - c.rolling(n, min_periods=n).mean()) / (a + 1e-12)
    s50 = c.rolling(50, min_periods=50).mean()
    F["slope50"] = (s50 / s50.shift(10) - 1) / (v20 + 1e-12)
    for n in (20, 55, 250):
        F[f"dhi{n}"] = (c - h.rolling(n, min_periods=n).max()) / (a + 1e-12)
        F[f"dlo{n}"] = (c - l.rolling(n, min_periods=n).min()) / (a + 1e-12)
    s20, sd20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
    F["pctb"] = (c - (s20 - 2 * sd20)) / (4 * sd20 + 1e-12)
    F["atr_pct"] = a / c
    for n in (5, 20, 60):
        F[f"vol{n}"] = lr.rolling(n, min_periods=n).std()
    F["vol_ratio"] = F["vol5"] / (F["vol60"] + 1e-12)
    F["skew60"], F["kurt60"] = lr.rolling(60, min_periods=60).skew(), lr.rolling(60, min_periods=60).kurt()
    sg = np.sign(lr.fillna(0)).to_numpy()
    stk = np.zeros(len(sg))
    for i in range(1, len(sg)):
        stk[i] = stk[i - 1] + sg[i] if sg[i] != 0 and np.sign(stk[i - 1]) in (0, sg[i]) else sg[i]
    F["streak"] = stk
    F["range_atr"] = (h - l) / (a + 1e-12)
    sp = (b["ask_c"] - b["bid_c"])
    F["spread_atr"] = sp / (a + 1e-12)
    F["spread_z"] = (sp - sp.rolling(100, min_periods=50).mean()) / (sp.rolling(100, min_periods=50).std() + 1e-12)
    if "n_ticks" in b:
        nt = b["n_ticks"].astype(float)
        F["ticks_z"] = (nt - nt.rolling(100, min_periods=50).mean()) / (nt.rolling(100, min_periods=50).std() + 1e-12)
    F["hour"] = b.index.hour
    F["weekday"] = b.index.weekday
    return F.astype("float32"), a.to_numpy(), o.to_numpy(), h.to_numpy(), l.to_numpy()


@njit(cache=True)
def tb_labels(o, h, l, a, up_k, dn_k, hz):
    n = len(o)
    y = np.full(n, -1)
    for t in range(n - hz - 1):
        if not (a[t] > 0):
            continue
        e = o[t + 1]
        up, dn = e + up_k * a[t], e - dn_k * a[t]
        lab = 1
        for j in range(t + 1, t + 1 + hz):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                lab = 1
                break
            if hu:
                lab = 2
                break
            if hd:
                lab = 0
                break
        y[t] = lab
    return y


def month_t(x: pd.Series):
    if len(x) < 10:
        return np.nan
    e = x - x.mean()
    g = e.groupby(pd.DatetimeIndex(x.index).tz_localize(None).to_period("M")).sum()
    m = len(g)
    se = math.sqrt((g ** 2).sum()) / len(x) * math.sqrt(m / max(m - 1, 1))
    return float(x.mean() / se) if se > 0 else np.nan


def cfd_rates(index):
    from fxlab.data import rates as R
    k = R.known_at(pd.DatetimeIndex(index.floor("D").unique()))
    return k["USD"]


def run(a):
    import lightgbm as lgb
    import flows_lab as FL
    from fxlab.data.store import load_m1, resample
    from scipy import stats as st
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    m1 = load_m1(a.symbol, "2007-01-01", END)
    m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)].dropna(subset=["bid_o", "ask_o", "bid_c", "ask_c"])
    bars = resample(m1, a.tf) if a.tf != "M1" else m1.copy()
    bars = bars.dropna(subset=["bid_o", "ask_o", "bid_h", "ask_l"])
    pip = PIPS.get(a.symbol, 1e-4)
    bk = FL.Book(m1[["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]], pip,
                 a.symbol if len(a.symbol) == 6 and a.symbol not in CFD else None)
    first_m1 = bk.idx.searchsorted(bars.index)                     # first M1 bar of each timeframe bar
    F, at, o, h, l = features(bars)
    feats = list(F.columns)
    years = bars.index.year.to_numpy()
    yrs = sorted(set(years))
    per_year = len(bars) / max(len(yrs), 1)
    usd = cfd_rates(bk.idx) if a.symbol in CFD else None
    print(a.symbol, a.tf, "bars", len(bars), "years", yrs[0], yrs[-1], "features", len(feats), flush=True)
    rows, all_trades = [], []
    for name, (uk, dk, hz) in BARRIERS.items():
        y = tb_labels(o, h, l, at, uk, dk, hz)
        ok = (y >= 0) & np.isfinite(F["r200"].to_numpy()) & np.isfinite(at)
        step = hz if per_year > 20000 else 1
        tidx = np.arange(len(bars))
        lab_end_year = np.r_[years[hz:], np.repeat(years[-1] + 1, hz)]
        trades = []
        ctrl_plan = []
        for Y in yrs[3:]:
            trB = ok & (lab_end_year < Y) & (tidx % step == 0) & (years < Y)
            trA = ok & (lab_end_year < Y - 1) & (tidx % step == 0) & (years < Y - 1)
            cal = ok & (years == Y - 1)
            te = np.flatnonzero(years == Y)
            if trA.sum() < 1000 or cal.sum() < 200 or not len(te):
                continue
            mA = lgb.LGBMClassifier(**LGB).fit(F.values[trA], y[trA])
            pc = mA.predict_proba(F.values[cal])
            th_l = float(np.quantile(pc[:, 2] - pc[:, 0], 0.9))
            th_s = float(np.quantile(pc[:, 0] - pc[:, 2], 0.9))
            mB = lgb.LGBMClassifier(**LGB).fit(F.values[trB], y[trB])
            p = mB.predict_proba(F.values[te])
            sl, ss = p[:, 2] - p[:, 0], p[:, 0] - p[:, 2]
            busy_until = -1
            n_y, n_long = 0, 0
            for k, t in enumerate(te):
                if t + 1 >= len(bars) or not np.isfinite(at[t]) or at[t] <= 0:
                    continue
                i = int(first_m1[t + 1])
                if i <= busy_until or i >= len(bk.idx):
                    continue
                side = 1 if (sl[k] > th_l and sl[k] > 0) else (-1 if (ss[k] > th_s and ss[k] > 0) else 0)
                if side == 0:
                    continue
                r, jx = trade(bk, first_m1, t, side, at[t], uk, dk, hz, a.symbol, usd)
                if r is None:
                    continue
                trades.append({"time": bk.idx[i], "side": side, "net": r[0], "gross": r[1]})
                busy_until = jx
                n_y += 1
                n_long += side > 0
            ctrl_plan.append((Y, n_y, n_long / max(n_y, 1)))
            print(name, Y, "trades", n_y, f"{time.time() - t0:.0f}s", flush=True)
        T = pd.DataFrame(trades)
        # random-entry control with the same barriers, counts and long/short mix
        rng = np.random.default_rng(19)
        draws, draws_g = [], []
        for d in range(a.draws):
            tot, tot_g, cnt = 0.0, 0.0, 0
            for Y, n_y, ls in ctrl_plan:
                if n_y == 0:
                    continue
                cand = np.flatnonzero((years == Y) & np.isfinite(at) & (at > 0))
                cand = cand[cand + 1 < len(bars)]
                picks = np.sort(rng.choice(cand, size=min(len(cand), n_y * 3), replace=False))
                busy, kk = -1, 0
                for t in picks:
                    if kk >= n_y:
                        break
                    i = int(first_m1[t + 1])
                    if i <= busy:
                        continue
                    side = 1 if rng.random() < ls else -1
                    r, jx = trade(bk, first_m1, t, side, at[t], uk, dk, hz, a.symbol, usd)
                    if r is None:
                        continue
                    tot += r[0]
                    tot_g += r[1]
                    cnt += 1
                    kk += 1
                    busy = jx
            draws.append(tot / max(cnt, 1))
            draws_g.append(tot_g / max(cnt, 1))
        draws, draws_g = np.array(draws), np.array(draws_g)
        if len(T):
            x = T.set_index("time").net.sort_index()
            tt = month_t(x)
            rows.append({"symbol": a.symbol, "tf": a.tf, "barrier": name, "tp_atr": uk, "sl_atr": dk, "horizon": hz,
                         "trades": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": tt,
                         "p_one_sided": float(1 - st.norm.cdf(tt)) if np.isfinite(tt) else np.nan,
                         "gross_bp": float(T.gross.mean() * 1e4), "hit": float((x > 0).mean()),
                         "long_share": float((T.side > 0).mean()), "random_mean_bp": float(draws.mean() * 1e4),
                         "random_p": float((draws >= x.mean()).mean()),
                         "random_gross_bp": float(draws_g.mean() * 1e4),
                         "random_p_gross": float((draws_g >= T.gross.mean()).mean()), "first": str(x.index[0])[:10],
                         "years_positive": f"{int((x.groupby(x.index.year).sum() > 0).sum())}/{x.index.year.nunique()}"})
            T.assign(barrier=name).to_csv(os.path.join(a.out, f"trades_{a.symbol}_{a.tf}_{name}.csv.gz"), index=False)
        else:
            rows.append({"symbol": a.symbol, "tf": a.tf, "barrier": name, "trades": 0})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"tests_{a.symbol}_{a.tf}.csv"), index=False)
    json.dump({"symbol": a.symbol, "tf": a.tf, "bars": int(len(bars)), "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(a.out, f"meta_{a.symbol}_{a.tf}.json"), "w"))


def trade(bk, first_m1, t, side, atr_t, uk, dk, hz, symbol, usd):
    """Open at the first M1 bar of timeframe bar t+1, close at the first barrier (M1 path), or at the first M1 bar of
    timeframe bar t+1+hz. Returns ((net, gross), exit M1 index) or (None, -1)."""
    i = int(first_m1[t + 1])
    jt = min(t + 1 + hz, len(first_m1) - 1)
    j = int(first_m1[jt]) if jt > t + 1 else len(bk.idx) - 1
    j = min(j, len(bk.idx) - 1)
    if j <= i:
        return None, -1
    r = bk.bracket(i, j, side, uk * atr_t, dk * atr_t)
    if r is None:
        return None, -1
    net, gross, _, _, n, flat, rate = r
    if usd is not None:                                      # CFD financing: side x USD rate + mark-up, per night
        if n:
            rf = float(usd.asof(bk.idx[i].floor("D"))) if len(usd) else 0.0
            fin = n * (side * (rf if np.isfinite(rf) else 0.0) + MARKUP) / 360.0
        else:
            fin = 0.0
        net_total = net - fin
    else:
        net_total = net + (rate if np.isfinite(rate) else flat)
    # where the trade actually ended (first barrier touch on the M1 path, else the time exit)
    bk._hl()
    si = bk.sf * (bk.ao[i] - bk.bo[i]) + bk.sp * bk.pip
    E = bk.ao[i] + si if side > 0 else bk.bo[i] - si
    if side > 0:
        hit = np.flatnonzero((bk.bl[i:j] <= E - dk * atr_t) | (bk.bh[i:j] >= E + uk * atr_t))
    else:
        hit = np.flatnonzero((bk.ah[i:j] >= E + dk * atr_t) | (bk.al[i:j] <= E - uk * atr_t))
    jx = i + int(hit[0]) if len(hit) else j
    return (net_total, gross), jx


def combine(a):
    from scipy import stats as st
    fs = glob.glob(os.path.join(a.combine, "**", "tests_*.csv"), recursive=True)
    T = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    T = T[T.trades > 0].copy()
    p = T.p_one_sided.to_numpy(float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    qq = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pv[i] * m / rank)
        qq[i] = prev
    q[ok] = qq
    T["q_bh"] = q
    os.makedirs(a.out, exist_ok=True)
    T.sort_values("p_one_sided").to_csv(os.path.join(a.out, "tests_all.csv"), index=False)
    piv = T.pivot_table(index="tf", columns="barrier", values="mean_bp", aggfunc="median")
    piv.to_csv(os.path.join(a.out, "median_mean_bp_by_tf.csv"))
    beat = T.assign(beat=T.random_p < 0.05).pivot_table(index="tf", columns="barrier", values="beat", aggfunc="mean")
    beat.to_csv(os.path.join(a.out, "share_beating_random_by_tf.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(T.sort_values("p_one_sided")[["symbol", "tf", "barrier", "trades", "mean_bp", "t", "gross_bp", "hit",
                                         "random_mean_bp", "random_p", "q_bh", "years_positive"]].round(3).to_string())
    print(piv.round(2))
    print(beat.round(2))
    json.dump({"tests": int(len(T)), "q10": int((T.q_bh <= 0.10).sum()), "positive_t2": int((T.t > 2).sum()),
               "negative_t2": int((T.t < -2).sum()), "beat_random_p05": int((T.random_p < 0.05).sum())},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--combine", default="")
    a = ap.parse_args()
    if a.combine:
        combine(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
