"""Entries as a machine-learning problem: gradient boosting (LightGBM) trained on three outcomes per candidate trade,
traded with three take-profit targets. Daily bars of the cross-market universe of scripts/cross_market_ta_lab.py
(excess-return prices, the same per-instrument costs and financing mark-up). Development data only, to 2025-12-31.

FIXED BEFORE ANY RESULT:
  Label (triple barrier, three classes): for every instrument and day t, a position opened at the next open
    o(t+1); class 2 = the open +2 ATR(14, at t) is touched before the open -2 ATR within the next 10 bars
    (t+1 .. t+10, daily highs and lows); class 0 = the lower barrier first; class 1 = neither, or both in the same
    bar (order unknown on daily bars).
  Features (all known at the close of t): returns over 1, 2, 3, 5, 10, 21, 63, 126 and 252 days scaled by the
    21-day volatility; RSI(2), RSI(14); internal bar strength; distance of the close from its 5/20/50/200-day
    averages in ATRs; the 50-day average's 10-day slope; distance to the 20/55/252-day high and low in ATRs;
    Bollinger %b(20); ATR / price; 5/21/63-day volatility and the 5/63 ratio; 63-day skew and kurtosis; the up/down
    streak; the opening gap; the bar's range in ATRs; weekday; month; the VIX level, its 1-year z-score and 5-day
    change; the 10-year minus 3-month Treasury yield and its 21-day change; SPY's 1/5/21-day vol-scaled returns;
    the asset class (one-hot).
  Model: one LightGBM multiclass model pooled over all instruments; 400 trees, learning rate 0.03, 15 leaves,
    200 samples per leaf, 80% row and 70% column subsampling, L2 1.0, seed 0. Nothing is tuned.
  Walk-forward: one test year at a time (2001-2025). Model B is trained on every row whose label ends at least 10
    trading days before the test year (expanding window). Thresholds come from model A, trained the same way but
    ending two years earlier, scored on those two years: the 90th percentiles of P(up) - P(down) (long) and
    P(down) - P(up) (short). Test-year rows are never used for training or thresholds.
  Entry: at the close of t, an instrument without an open position goes long if P(up) - P(down) is above its
    threshold and above 0, else short if P(down) - P(up) is above its threshold and above 0; filled at the next
    open plus the one-way cost.
  Exits:
    V1 (primary, three targets): stop 2 ATR; a third of the position at +1, +2 and +3 ATR; after the first target
       the stop moves to the entry; anything left exits at the open after 10 bars.
    V2 (the labelled trade): one target at +2 ATR, stop 2 ATR, time exit after 10 bars.
    V3 (entry skill alone): no stop or target, exit at the open after 10 bars.
    Daily-bar ordering is conservative: when a bar touches the stop and a target, the stop is taken first; a bar
    that opens beyond a level fills at its open. Every fill pays the instrument's one-way cost; open positions pay
    the financing mark-up per night (the cash rate is already in the excess-return price).
  Controls: C1 = 200 sets of random entries with the model's number of trades per instrument and year and its
    long/short mix, the same one-position-at-a-time rule and the V1 exits; compared on net and (added with the
    second run, before it) on gross returns, because ATR-scaled barriers make costs and financing a smaller share
    of a trade in volatile times, so beating random entries on net alone can reflect timing volatility.
PRE-REGISTERED TESTS (one-sided): M1 (primary) mean net return per V1 trade > 0 (t clustered by month) and above
  the random-entry distribution (p = share of C1 draws with a mean at least as high; the draws share the model's
  instruments, years and long/short mix, so market drift is controlled); M2/M3 the same mean test for V2 and V3;
  also by asset class and long vs short.
Usage: python scripts/ml_entry_lab.py --data DIR --out DIR [--first-test 2001]"""
from __future__ import annotations

import argparse
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

H = 10
K_BARRIER = 2.0
CLASSES = ("equity", "index", "bond", "commodity", "crypto")
LGB = dict(objective="multiclass", num_class=3, n_estimators=400, learning_rate=0.03, num_leaves=15,
           min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
           random_state=0, verbose=-1, n_jobs=4)


def safe(s):
    return s.replace("^", "I_").replace("=", "_").replace("-", "_")


# ----------------------------------------------------------------------------------------------- data
def mid_ohlc(b):
    return pd.DataFrame({k: 0.5 * (b[f"bid_{k}"] + b[f"ask_{k}"]) for k in "ohlc"}, index=b.index.normalize())


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def atr(m, n=14):
    pc = m.c.shift(1)
    tr = pd.concat([m.h - m.l, (m.h - pc).abs(), (m.l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(m, macro, spy_r, cls):
    c = m.c
    lr = np.log(c).diff()
    v21 = lr.rolling(21, min_periods=15).std()
    a = atr(m)
    F = pd.DataFrame(index=m.index)
    for k in (1, 2, 3, 5, 10, 21, 63, 126, 252):
        F[f"r{k}"] = np.log(c / c.shift(k)) / (v21 * math.sqrt(k) + 1e-12)
    F["rsi2"], F["rsi14"] = rsi(c, 2), rsi(c, 14)
    F["ibs"] = (c - m.l) / (m.h - m.l + 1e-12)
    for n in (5, 20, 50, 200):
        F[f"dma{n}"] = (c - c.rolling(n, min_periods=n).mean()) / (a + 1e-12)
    s50 = c.rolling(50, min_periods=50).mean()
    F["slope50"] = (s50 / s50.shift(10) - 1) / (v21 + 1e-12)
    for n in (20, 55, 252):
        F[f"dhi{n}"] = (c - m.h.rolling(n, min_periods=n).max()) / (a + 1e-12)
        F[f"dlo{n}"] = (c - m.l.rolling(n, min_periods=n).min()) / (a + 1e-12)
    s20, sd20 = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
    F["pctb"] = (c - (s20 - 2 * sd20)) / (4 * sd20 + 1e-12)
    F["atr_pct"] = a / c
    for n in (5, 21, 63):
        F[f"vol{n}"] = lr.rolling(n, min_periods=n).std() * math.sqrt(252)
    F["vol_ratio"] = F["vol5"] / (F["vol63"] + 1e-12)
    F["skew63"], F["kurt63"] = lr.rolling(63, min_periods=63).skew(), lr.rolling(63, min_periods=63).kurt()
    sgn = np.sign(lr.fillna(0)).to_numpy()
    st_ = np.zeros(len(sgn))
    for i in range(1, len(sgn)):
        st_[i] = st_[i - 1] + sgn[i] if sgn[i] != 0 and np.sign(st_[i - 1]) in (0, sgn[i]) else sgn[i]
    F["streak"] = st_
    F["gap"] = np.log(m.o / c.shift(1)) / (v21 + 1e-12)
    F["range_atr"] = (m.h - m.l) / (a + 1e-12)
    F["weekday"] = m.index.weekday
    F["month"] = m.index.month
    mac = macro.reindex(macro.index.union(m.index)).ffill().reindex(m.index)
    F["vix"] = mac["VIX"]
    F["vix_z"] = (mac["VIX"] - mac["VIX"].rolling(252, min_periods=200).mean()) / (mac["VIX"].rolling(252, min_periods=200).std() + 1e-12)
    F["vix_d5"] = mac["VIX"] - mac["VIX"].shift(5)
    F["term"] = mac["TNX"] - mac["IRX"]
    F["term_d21"] = F["term"] - F["term"].shift(21)
    sr = spy_r.reindex(spy_r.index.union(m.index)).ffill().reindex(m.index)
    for k in (1, 5, 21):
        F[f"spy{k}"] = sr[f"spy{k}"]
    for k in CLASSES:
        F[f"cls_{k}"] = float(cls == k)
    return F.astype("float32"), a


@njit(cache=True)
def tb_labels(o, h, l, a, hz, k):
    n = len(o)
    y = np.full(n, -1)
    for t in range(n - hz - 1):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        e = o[t + 1]
        up, dn = e + k * a[t], e - k * a[t]
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


@njit(cache=True)
def sim_trade(o, h, l, i, side, a, cost, fin_night, nights, variant, hz):
    """Trade opened at the open of bar i (decided at the close of i-1). Returns (net, exit bar)."""
    e = o[i] * (1 + side * cost)                      # entry fill including the one-way cost
    ref = o[i]
    n = len(o)
    last = min(i + hz, n - 1)                          # time exit at the open of bar i + hz
    if variant == 3:
        stop_d, tgts, fr = 1e18, np.array([1e18]), np.array([1.0])
    elif variant == 2:
        stop_d, tgts, fr = 2.0 * a, np.array([2.0 * a]), np.array([1.0])
    else:
        stop_d, tgts, fr = 2.0 * a, np.array([1.0 * a, 2.0 * a, 3.0 * a]), np.array([1 / 3, 1 / 3, 1 / 3])
    stop = ref - side * stop_d
    left = 1.0
    pnl = 0.0
    hit = np.zeros(len(tgts), dtype=np.bool_)
    j = i
    while j < last:
        # financing for holding through bar j (open j -> open j+1)
        # stop first (conservative), then targets in order
        lo_j, hi_j, op_j = l[j], h[j], o[j]
        stopped = (lo_j <= stop) if side > 0 else (hi_j >= stop)
        if stopped:
            px = (min(stop, op_j) if side > 0 else max(stop, op_j)) if j > i else stop
            px = px * (1 - side * cost)
            pnl += left * side * (px / e - 1)
            left = 0.0
            return pnl, j
        for q in range(len(tgts)):
            if hit[q]:
                continue
            lvl = ref + side * tgts[q]
            touched = (hi_j >= lvl) if side > 0 else (lo_j <= lvl)
            if touched:
                px = (max(lvl, op_j) if side > 0 else min(lvl, op_j)) if j > i else lvl
                px = px * (1 - side * cost)
                pnl += fr[q] * side * (px / e - 1)
                left -= fr[q]
                hit[q] = True
                if variant == 1 and q == 0:
                    stop = ref                         # breakeven after the first target
        if left <= 1e-9:
            return pnl, j
        pnl -= left * fin_night * nights[j]
        j += 1
    px = o[last] * (1 - side * cost)
    pnl += left * side * (px / e - 1)
    return pnl, last


# ----------------------------------------------------------------------------------------------- main
def load_universe(data):
    meta = json.load(open(os.path.join(data, "meta.json")))
    macro = pd.read_parquet(os.path.join(data, "macro.parquet"))
    macro.index = pd.DatetimeIndex(macro.index).tz_localize(None).normalize() if macro.index.tz is not None else macro.index
    U = {}
    for s, mt in meta.items():
        if mt.get("status") != "ok":
            continue
        b = pd.read_parquet(os.path.join(data, f"bars_{safe(s)}.parquet"))
        m = mid_ohlc(b)
        m.index = m.index.tz_localize(None)
        U[s] = (m, mt)
    spy = U["SPY"][0].c
    lr = np.log(spy).diff()
    v = lr.rolling(21, min_periods=15).std()
    spy_r = pd.DataFrame({f"spy{k}": np.log(spy / spy.shift(k)) / (v * math.sqrt(k)) for k in (1, 5, 21)})
    return U, macro, spy_r


def month_clustered_t(x: pd.Series):
    if len(x) < 10:
        return np.nan
    g = x.groupby(pd.DatetimeIndex(x.index).to_period("M")).sum()
    n_m = len(g)
    mean_per_trade = x.mean()
    # cluster-robust se of the mean: sqrt(sum_g (sum_i e_i)^2) / N
    e = x - mean_per_trade
    ge = e.groupby(pd.DatetimeIndex(x.index).to_period("M")).sum()
    se = math.sqrt((ge ** 2).sum()) / len(x) * math.sqrt(n_m / max(n_m - 1, 1))
    return float(mean_per_trade / se) if se > 0 else np.nan


def run(a):
    import lightgbm as lgb
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    U, macro, spy_r = load_universe(a.data)
    rows, store, feat_of = [], {}, {}
    for s, (m, mt) in U.items():
        F, at = features(m, macro, spy_r, mt["class"])
        y = tb_labels(m.o.to_numpy(), m.h.to_numpy(), m.l.to_numpy(), at.to_numpy(), H, K_BARRIER)
        d = F.copy()
        d["y"] = y
        d["sym"] = s
        d["t"] = np.arange(len(d))
        lab_end = np.r_[m.index[H:], np.repeat(m.index[-1], H)][: len(d)]   # date of bar t + H
        d["label_end"] = lab_end
        rows.append(d)
        nights = np.r_[pd.Series(m.index).diff().dt.days.to_numpy()[1:], 0.0].astype(float)
        store[s] = (m, at.to_numpy(), mt, nights)
        feat_of[s] = F
    D = pd.concat(rows)
    D = D[D.y >= 0]
    feats = [c for c in D.columns if c not in ("y", "sym", "t", "label_end")]
    D = D.dropna(subset=["r252", "dma200", "vol63"])
    print("rows", len(D), "features", len(feats), "class shares", D.y.value_counts(normalize=True).round(3).to_dict(),
          flush=True)
    scores = []
    imp = pd.Series(0.0, index=feats)
    years = list(range(a.first_test, 2026))
    thr_rows = []
    for Y in years:
        ys = pd.Timestamp(f"{Y}-01-01")
        emb = pd.Timedelta(days=16)                          # 10 trading days and a margin
        trB = D[D.label_end < ys - emb]
        trA = D[D.label_end < ys - pd.DateOffset(years=2) - emb]
        cal = D[(D.index >= ys - pd.DateOffset(years=2)) & (D.label_end < ys - emb)]
        te_idx = [(s, store[s][0].index[(store[s][0].index >= ys) & (store[s][0].index <= pd.Timestamp(f"{Y}-12-31"))])
                  for s in store]
        if len(trA) < a.min_train or len(cal) < a.min_cal:
            print("skip", Y, len(trA), len(cal), flush=True)
            continue
        mA = lgb.LGBMClassifier(**LGB).fit(trA[feats], trA.y)
        pc = mA.predict_proba(cal[feats])
        th_l = float(np.quantile(pc[:, 2] - pc[:, 0], 0.9))
        th_s = float(np.quantile(pc[:, 0] - pc[:, 2], 0.9))
        mB = lgb.LGBMClassifier(**LGB).fit(trB[feats], trB.y)
        imp += pd.Series(mB.booster_.feature_importance("gain"), index=feats)
        thr_rows.append({"year": Y, "train_rows": len(trB), "th_long": th_l, "th_short": th_s})
        for s, idx in te_idx:
            if not len(idx):
                continue
            X = feat_of[s].loc[idx, feats]
            p = mB.predict_proba(X)
            scores.append(pd.DataFrame({"sym": s, "date": idx, "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2],
                                        "th_l": th_l, "th_s": th_s}))
        print("year", Y, "train", len(trB), "thresholds", round(th_l, 3), round(th_s, 3),
              f"{time.time() - t0:.0f}s", flush=True)
    Sc = pd.concat(scores, ignore_index=True)
    Sc.to_parquet(os.path.join(a.out, "scores.parquet"))
    pd.DataFrame(thr_rows).to_csv(os.path.join(a.out, "thresholds.csv"), index=False)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(os.path.join(a.out, "feature_importance.csv"))
    # ------------------------------------------------------------------ trading
    trades = []
    for s, g in Sc.groupby("sym"):
        m, at, mt, nights = store[s]
        o, h, l = m.o.to_numpy(), m.h.to_numpy(), m.l.to_numpy()
        pos_of = {d: i for i, d in enumerate(m.index)}
        cost = mt["cost_bp"] * 1e-4
        fin = mt["markup"] / 360.0
        g = g.sort_values("date")
        for v in (1, 2, 3):
            busy = -1
            for r in g.itertuples(index=False):
                t = pos_of[r.date]
                if t <= busy or t + 1 >= len(o):
                    continue
                sl, ss = r.p2 - r.p0, r.p0 - r.p2
                side = 1 if (sl > r.th_l and sl > 0) else (-1 if (ss > r.th_s and ss > 0) else 0)
                if side == 0 or not np.isfinite(at[t]):
                    continue
                net, xj = sim_trade(o, h, l, t + 1, side, at[t], cost, fin, nights, v, H)
                gross, _ = sim_trade(o, h, l, t + 1, side, at[t], 0.0, 0.0, nights, v, H)
                trades.append({"sym": s, "class": mt["class"], "variant": v, "date": r.date, "side": side,
                               "net": net, "gross": gross, "exit": m.index[xj]})
                busy = xj - 1 if v != 3 else t + H
    T = pd.DataFrame(trades)
    T.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)
    # ------------------------------------------------------------------ random-entry control (V1 exits)
    rng = np.random.default_rng(19)
    v1 = T[T.variant == 1]
    plan = v1.groupby(["sym", v1.date.dt.year]).agg(n=("net", "size"), longs=("side", lambda x: (x > 0).mean()))
    ctrl, ctrl_g = [], []
    for draw in range(a.draws):
        tot, tot_g, cnt = 0.0, 0.0, 0
        for (s, Y), r in plan.iterrows():
            m, at, mt, nights = store[s]
            o, h, l = m.o.to_numpy(), m.h.to_numpy(), m.l.to_numpy()
            cand = np.flatnonzero((m.index.year == Y))
            cand = cand[(cand + 1 < len(o)) & np.isfinite(at[cand])]
            if not len(cand):
                continue
            picks = np.sort(rng.choice(cand, size=min(int(r.n) * 3, len(cand)), replace=False))
            busy, k = -1, 0
            for t in picks:
                if k >= r.n:
                    break
                if t <= busy:
                    continue
                side = 1 if rng.random() < r.longs else -1
                net, xj = sim_trade(o, h, l, t + 1, side, at[t], mt["cost_bp"] * 1e-4, mt["markup"] / 360.0, nights, 1, H)
                g_, _ = sim_trade(o, h, l, t + 1, side, at[t], 0.0, 0.0, nights, 1, H)
                tot += net
                tot_g += g_
                cnt += 1
                k += 1
                busy = xj - 1
        ctrl.append(tot / max(cnt, 1))
        ctrl_g.append(tot_g / max(cnt, 1))
    ctrl, ctrl_g = np.array(ctrl), np.array(ctrl_g)
    np.save(os.path.join(a.out, "control_means.npy"), ctrl)
    np.save(os.path.join(a.out, "control_gross_means.npy"), ctrl_g)
    # ------------------------------------------------------------------ statistics
    from scipy import stats as st
    res = {"runtime_min": round((time.time() - t0) / 60, 1), "rows": int(len(D)), "features": feats}
    out_rows = []
    for v in (1, 2, 3):
        x = T[T.variant == v].set_index("date").net.sort_index()
        tt = month_clustered_t(x)
        row = {"test": f"M{v}", "variant": {1: "three targets", 2: "labelled trade", 3: "time exit"}[v],
               "trades": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": tt,
               "p_one_sided": float(1 - st.norm.cdf(tt)) if np.isfinite(tt) else np.nan,
               "hit": float((x > 0).mean()), "long_share": float((T[T.variant == v].side > 0).mean()),
               "years_positive": f"{int((x.groupby(x.index.year).sum() > 0).sum())}/{x.index.year.nunique()}"}
        xg = T[T.variant == v].gross
        row["gross_bp"] = float(xg.mean() * 1e4)
        if v == 1:
            row["random_entry_mean_bp"] = float(ctrl.mean() * 1e4)
            row["random_entry_p"] = float((ctrl >= x.mean()).mean())
            row["random_entry_gross_bp"] = float(ctrl_g.mean() * 1e4)
            row["random_entry_p_gross"] = float((ctrl_g >= xg.mean()).mean())
        for cls in CLASSES:
            xc = T[(T.variant == v) & (T["class"] == cls)].set_index("date").net.sort_index()
            row[f"{cls}_bp"] = float(xc.mean() * 1e4) if len(xc) else np.nan
            row[f"{cls}_t"] = month_clustered_t(xc) if len(xc) > 30 else np.nan
        for sd, nm in ((1, "long"), (-1, "short")):
            xs = T[(T.variant == v) & (T.side == sd)].set_index("date").net.sort_index()
            row[f"{nm}_bp"] = float(xs.mean() * 1e4) if len(xs) else np.nan
            row[f"{nm}_t"] = month_clustered_t(xs) if len(xs) > 30 else np.nan
        out_rows.append(row)
        by_year = x.groupby(x.index.year).agg(["size", "mean"])
        res[f"M{v}_by_year_bp"] = {int(k): [int(r["size"]), round(float(r["mean"]) * 1e4, 2)] for k, r in by_year.iterrows()}
    R = pd.DataFrame(out_rows)
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(R.round(3).T.to_string())
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-test", type=int, default=2001)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--min-train", type=int, default=20000, help="rows needed to fit model A (smoke tests lower it)")
    ap.add_argument("--min-cal", type=int, default=2000)
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
