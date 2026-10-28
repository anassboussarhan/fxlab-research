"""Adaptive triple barriers: the take-profit, the stop and the time limit of every trade are chosen by the model, as a
nonlinear function of the market state, instead of being fixed. One gradient-boosting regressor predicts the net
return (after spread, slippage, commission and swap) of a trade opened at the next bar for any (side, take-profit,
stop, time limit); at each bar the strategy takes the combination with the highest prediction, if that prediction
clears a threshold. Model hyper-parameters, loss, linear leaves and the threshold are all chosen inside the
walk-forward; the test year is never used for any choice. Development data only (to 2025-12-31).

FIXED BEFORE ANY RESULT (the search spaces, not the values):
  Barrier grid (ATR(14) of the timeframe's bars; time limit in bars): side +1 / -1; take-profit 0.5, 1, 1.5, 2, 3, 4
    or none; stop 0.5, 1, 1.5, 2, 3, 4 or none; time limit 3, 6, 12, 24 or 48 bars; the combination with neither a
    take-profit nor a stop is the pure time exit. 2 x 7 x 7 x 5 = 490 combinations.
  Training rows: sampled bars x 6 random combinations each, the target being the combination's realised net return
    on the M1 bid/ask path (stop first within a minute; costs as in the flows lab; FX swap from point-in-time rate
    differentials; gold and index CFDs pay the 3-month USD rate plus 2.5% a year per night on longs and receive the
    rate minus 2.5% on shorts).
  Features (known at the close of the bar): the 38 of scripts/ml_triple_barrier_lab.py, plus longer context (returns
    over 500 and 1,000 bars, 250-bar volatility, distance to the 500- and 1,000-bar averages), calendar (day of
    month, last business day of the month, first Friday of the month, Tokyo Gotobi day, minutes to the 09:55 Tokyo
    and 16:00 London fixes), the point-in-time carry of the pair, and the combination itself (side, take-profit,
    stop, time limit, their ratio).
  Model search (Optuna TPE, 8 trials per test year, objective = the policy's total net return on the validation
    year): leaves 15-127, learning rate 0.02-0.2, trees 100-600, samples per leaf 50-1,000, row and column
    subsampling 0.5-1, L2 0-10, loss L2 or Huber, linear leaves on or off. Threshold: the prediction quantile that
    maximises validation net return, from the top 0.5%, 1%, 2%, 5%, 10% or 20% of bars, or any positive
    prediction; that threshold, in predicted bp, is then applied unchanged to the refitted model.
  Walk-forward: test years from the fourth year of data; validation = the year before the test year; the final
    model is refitted on every earlier year with the chosen settings. One position at a time.
  Control: 20 sets of random entries per test year with the model's trade count, long share and its chosen
    barrier combinations, compared before and after costs.
PRE-REGISTERED TEST (one-sided): mean net return per trade > 0 (t clustered by month), per instrument and timeframe;
  Benjamini-Hochberg over all of them in the combine stage.
Usage: python scripts/ml_adaptive_barrier_lab.py --symbol EURUSD --tf H1 --out DIR [--trials 8]
       python scripts/ml_adaptive_barrier_lab.py --combine DIR --out DIR"""
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
import ml_triple_barrier_lab as TB  # noqa: E402

END = "2025-12-31 23:59"
TPS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf)
SLS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, np.inf)
HZS = (3, 6, 12, 24, 48)
GRID = np.array([(s, tp, sl, h) for s in (1.0, -1.0) for tp in TPS for sl in SLS for h in HZS], dtype=float)
QUANTS = (0.995, 0.99, 0.98, 0.95, 0.90, 0.80, None)
MARKUP = 0.025
NS_PER_MIN = 60 * 10 ** 9


# ----------------------------------------------------------------------------------------------- costs and paths
def rollover_index(idx: pd.DatetimeIndex) -> np.ndarray:
    """Cumulative count of swap nights (17:00 New York instants, Wednesday x3) up to each M1 bar."""
    ny = idx.tz_convert("America/New_York")
    day = (ny.tz_localize(None) - pd.Timedelta(hours=17)).floor("D")     # the NY day whose 17:00 has passed
    d = day.to_numpy().astype("datetime64[D]")
    base = d.min()
    days = np.arange(base, d.max() + np.timedelta64(2, "D"), dtype="datetime64[D]")
    wd = (days.view("int64") - 4) % 7
    w = np.where(wd == 2, 3, np.where(wd < 5, 1, 0))
    C = np.r_[0, np.cumsum(w)]
    return C[(d - base).astype(np.int64) + 1].astype(np.int64)


@njit(cache=True)
def sim(bo, ao, bh, bl, ah, al, i, j, side, tp_d, sl_d, sf, sp, comm, rc, sw_long, sw_short):
    """Market entry at the open of M1 bar i, take-profit / stop at price distances from the fill (inf = none), time
    exit at the open of bar j. Returns (net, gross, exit bar)."""
    si = sf * (ao[i] - bo[i]) + sp
    mi = 0.5 * (ao[i] + bo[i])
    if side > 0:
        E = ao[i] + si
        TP, SL = E + tp_d, E - sl_d
    else:
        E = bo[i] - si
        TP, SL = E - tp_d, E + sl_d
    k = j
    px = 0.0
    hit = False
    for x in range(i, j):
        if side > 0:
            if bl[x] <= SL:
                sx = sf * (ao[x] - bo[x]) + sp
                px = min(SL, bo[x]) - sx if x > i else SL - sx
                k = x
                hit = True
                break
            if bh[x] >= TP:
                px = TP
                k = x
                hit = True
                break
        else:
            if ah[x] >= SL:
                sx = sf * (ao[x] - bo[x]) + sp
                px = max(SL, ao[x]) + sx if x > i else SL + sx
                k = x
                hit = True
                break
            if al[x] <= TP:
                px = TP
                k = x
                hit = True
                break
    if not hit:
        sj = sf * (ao[j] - bo[j]) + sp
        px = bo[j] - sj if side > 0 else ao[j] + sj
    net = (px / E - 1.0) if side > 0 else (1.0 - px / E)
    nights = rc[k] - rc[i]
    swap = nights * (sw_long[i] if side > 0 else sw_short[i])
    mid_x = 0.5 * (ao[k] + bo[k])
    gross = side * (mid_x / mi - 1.0)
    return net - 2 * comm + swap, gross, k


@njit(cache=True)
def sim_many(bo, ao, bh, bl, ah, al, I, J, S, TPD, SLD, sf, sp, comm, rc, sw_long, sw_short):
    n = len(I)
    out = np.empty(n)
    gr = np.empty(n)
    ex = np.empty(n, dtype=np.int64)
    for q in range(n):
        out[q], gr[q], ex[q] = sim(bo, ao, bh, bl, ah, al, I[q], J[q], S[q], TPD[q], SLD[q], sf, sp, comm, rc,
                                   sw_long, sw_short)
    return out, gr, ex


# ----------------------------------------------------------------------------------------------- features
def extra_features(b: pd.DataFrame, symbol: str, carry: np.ndarray | None):
    import flows_lab as FL
    c = 0.5 * (b["bid_c"] + b["ask_c"])
    lr = np.log(c).diff()
    E = pd.DataFrame(index=b.index)
    v20 = lr.rolling(20, min_periods=15).std()
    for k in (500, 1000):
        E[f"r{k}"] = np.log(c / c.shift(k)) / (v20 * math.sqrt(k) + 1e-12)
        E[f"dma{k}"] = (c / c.rolling(k, min_periods=k).mean() - 1) / (v20 * math.sqrt(20) + 1e-12)
    E["vol250"] = lr.rolling(250, min_periods=200).std()
    d = b.index.tz_convert("UTC").tz_localize(None).normalize()
    E["dom"] = d.day
    E["month_end"] = (d == d + pd.offsets.BMonthEnd(0)).astype(float)
    E["nfp_friday"] = ((d.weekday == 4) & (d.day <= 7)).astype(float)
    years = range(d.min().year, d.max().year + 1)
    hol = FL.jp_holidays(years)
    G = set(FL.gotobi_days(str(d.min().date()), str(d.max().date()), hol))
    tky = b.index.tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    E["gotobi"] = pd.DatetimeIndex(tky).isin(list(G)).astype(float)
    t_tok = b.index.tz_convert("Asia/Tokyo")
    E["min_to_tokyo_fix"] = ((9 * 60 + 55) - (t_tok.hour * 60 + t_tok.minute)).astype(float)
    t_ldn = b.index.tz_convert("Europe/London")
    E["min_to_london_fix"] = ((16 * 60) - (t_ldn.hour * 60 + t_ldn.minute)).astype(float)
    E["carry"] = carry if carry is not None else 0.0
    return E.astype("float32")


def cfg_cols(g: np.ndarray, n: int):
    s, tp, sl, h = g
    tpv, slv = (tp if np.isfinite(tp) else 10.0), (sl if np.isfinite(sl) else 10.0)
    return np.tile(np.array([s, tpv, slv, h, tpv / slv], dtype=np.float32), (n, 1))


# ----------------------------------------------------------------------------------------------- run
class Market:
    def __init__(self, symbol, tf):
        import flows_lab as FL
        from fxlab.data.store import load_m1, resample
        m1 = load_m1(symbol, "2007-01-01", END)
        m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)].dropna(subset=["bid_o", "ask_o", "bid_c", "ask_c"])
        bars = resample(m1, tf) if tf != "M1" else m1.copy()
        self.bars = bars.dropna(subset=["bid_o", "ask_o", "bid_h", "ask_l"])
        self.pip = TB.PIPS.get(symbol, 1e-4)
        cols = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
        self.bk = FL.Book(m1[cols], self.pip, symbol if len(symbol) == 6 and symbol not in TB.CFD else None)
        bk = self.bk
        bk._hl()
        self.arr = (bk.bo.astype(float), bk.ao.astype(float), bk.bh.astype(float), bk.bl.astype(float),
                    bk.ah.astype(float), bk.al.astype(float))
        self.sf, self.sp, self.comm = bk.sf, bk.sp * bk.pip, bk.comm
        self.rc = rollover_index(bk.idx)
        days = bk.idx.floor("D")
        if symbol in TB.CFD:
            from fxlab.data import rates as R
            usd = R.known_at(pd.DatetimeIndex(days.unique()))["USD"].reindex(days).to_numpy()
            usd = np.nan_to_num(usd)
            self.sw_long = -(usd + MARKUP) / 360.0
            self.sw_short = (usd - MARKUP) / 360.0
            carry = None
        elif bk.rd is not None:
            from fxlab.data.rates import SWAP_MARKUP_PA
            rd = bk.rd.reindex(days, method="ffill").to_numpy()
            rd = np.nan_to_num(rd)
            self.sw_long = (rd - SWAP_MARKUP_PA) / 365.0
            self.sw_short = (-rd - SWAP_MARKUP_PA) / 365.0
            carry = bk.rd.reindex(self.bars.index.floor("D"), method="ffill").to_numpy()
        else:
            flat = -0.5 * self.pip / (0.5 * (self.arr[0] + self.arr[1]))
            self.sw_long = self.sw_short = flat
            carry = None
        self.first_m1 = bk.idx.searchsorted(self.bars.index)
        F, self.atr, *_ = TB.features(self.bars)
        X2 = extra_features(self.bars, symbol, carry)
        self.F = pd.concat([F, X2], axis=1)
        self.feats = list(self.F.columns) + ["c_side", "c_tp", "c_sl", "c_h", "c_ratio"]
        self.years = self.bars.index.year.to_numpy()

    def simulate(self, T, G):
        """T: timeframe bar indices of the decisions; G: (n, 4) combinations. Returns net, gross, exit M1 index."""
        I = self.first_m1[np.minimum(T + 1, len(self.first_m1) - 1)].astype(np.int64)
        jt = np.minimum(T + 1 + G[:, 3].astype(np.int64), len(self.first_m1) - 1)
        J = np.minimum(self.first_m1[jt], len(self.bk.idx) - 1).astype(np.int64)
        a = self.atr[T]
        TPD = np.where(np.isfinite(G[:, 1]), G[:, 1] * a, 1e18)
        SLD = np.where(np.isfinite(G[:, 2]), G[:, 2] * a, 1e18)
        ok = (J > I) & np.isfinite(a) & (a > 0)
        net = np.full(len(T), np.nan)
        gr = np.full(len(T), np.nan)
        ex = np.full(len(T), -1, dtype=np.int64)
        if ok.any():
            n_, g_, e_ = sim_many(*self.arr, I[ok], J[ok], G[ok, 0], TPD[ok], SLD[ok], self.sf, self.sp, self.comm,
                                  self.rc, self.sw_long, self.sw_short)
            net[ok], gr[ok], ex[ok] = n_, g_, e_
        return net, gr, ex


def build_training(mk: Market, mask: np.ndarray, rng, per_bar=6, max_bars=60_000):
    T = np.flatnonzero(mask)
    if len(T) > max_bars:
        T = np.sort(rng.choice(T, size=max_bars, replace=False))
    TT = np.repeat(T, per_bar)
    G = GRID[rng.integers(0, len(GRID), size=len(TT))]
    net, _, _ = mk.simulate(TT, G)
    ok = np.isfinite(net)
    Xb = mk.F.values[TT[ok]]
    Xc = np.column_stack([G[ok, 0], np.where(np.isfinite(G[ok, 1]), G[ok, 1], 10.0),
                          np.where(np.isfinite(G[ok, 2]), G[ok, 2], 10.0), G[ok, 3],
                          np.where(np.isfinite(G[ok, 1]), G[ok, 1], 10.0) / np.where(np.isfinite(G[ok, 2]), G[ok, 2], 10.0)])
    return np.hstack([Xb, Xc.astype(np.float32)]), net[ok] * 1e4          # target in bp


def best_combo(model, Xbars: np.ndarray):
    """Highest predicted net return over the grid for each bar: (value, grid index)."""
    n = len(Xbars)
    best = np.full(n, -np.inf, dtype=np.float32)
    arg = np.zeros(n, dtype=np.int32)
    for k, g in enumerate(GRID):
        p = model.predict(np.hstack([Xbars, cfg_cols(g, n)]))
        better = p > best
        best[better] = p[better]
        arg[better] = k
    return best, arg


def policy(mk: Market, T: np.ndarray, best: np.ndarray, arg: np.ndarray, thr: float):
    """One position at a time: walk the decision bars in order, open the best combination when it clears thr."""
    trades = []
    busy = -1
    cand = np.flatnonzero(best > thr)
    for q in cand:
        t = T[q]
        i = int(mk.first_m1[min(t + 1, len(mk.first_m1) - 1)])
        if i <= busy:
            continue
        g = GRID[arg[q]][None, :]
        net, gr, ex = mk.simulate(np.array([t]), g)
        if not np.isfinite(net[0]):
            continue
        trades.append((t, i, arg[q], net[0], gr[0]))
        busy = int(ex[0])
    return trades


def run(a):
    import lightgbm as lgb
    import optuna
    from scipy import stats as st
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    mk = Market(a.symbol, a.tf)
    yrs = sorted(set(mk.years))
    per_year = len(mk.bars) / max(len(yrs), 1)
    step = 1 if per_year < 20000 else max(1, int(per_year // 20000))
    valid = np.isfinite(mk.F["r200"].to_numpy()) & np.isfinite(mk.atr) & (mk.atr > 0)
    tidx = np.arange(len(mk.bars))
    rng = np.random.default_rng(0)
    print(a.symbol, a.tf, "bars", len(mk.bars), "features", len(mk.feats), "grid", len(GRID), "step", step,
          f"{time.time() - t0:.0f}s", flush=True)
    rows, trades_all, chosen = [], [], []
    hz_max = int(GRID[:, 3].max())
    for Y in yrs[3:]:
        if time.time() - t0 > a.budget_min * 60:
            print("time budget reached before", Y, "- later years not tested", flush=True)
            break
        # rows whose longest trade ends before the year in question (purging by the longest time limit)
        last_ok = {y: np.flatnonzero(mk.years < y) for y in (Y - 1, Y)}
        def before(y):
            m = np.zeros(len(mk.bars), bool)
            ix = last_ok[y]
            if len(ix) > hz_max + 1:
                m[ix[: len(ix) - hz_max - 1]] = True
            return m & valid & (tidx % step == 0)
        trA, trB = before(Y - 1), before(Y)
        va = np.flatnonzero((mk.years == Y - 1) & valid)
        te = np.flatnonzero((mk.years == Y) & valid)
        if trA.sum() < a.min_train or len(va) < a.min_val or not len(te):
            continue
        XA, yA = build_training(mk, trA, rng)
        va_s = va[:: max(1, len(va) // 25000)]                    # validation decisions (subsampled for M1-M15)

        def fit(params, X, y):
            p = dict(params)
            p.update(verbose=-1, n_jobs=4, random_state=0, subsample_freq=1)
            return lgb.LGBMRegressor(**p).fit(X, y)

        def evaluate(params):
            m = fit(params, XA, yA)
            best, arg = best_combo(m, mk.F.values[va_s])
            out = []
            for qv in QUANTS:
                thr = 0.0 if qv is None else max(0.0, float(np.quantile(best, qv)))
                tr = policy(mk, va_s, best, arg, thr)
                tot = float(np.sum([x[3] for x in tr])) if tr else 0.0
                out.append((tot, thr, qv, len(tr)))
            return max(out, key=lambda z: z[0])

        def objective(trial):
            params = {"num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
                      "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                      "n_estimators": trial.suggest_int("n_estimators", 100, 600, log=True),
                      "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
                      "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                      "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                      "objective": trial.suggest_categorical("objective", ["l2", "huber"]),
                      "linear_tree": trial.suggest_categorical("linear_tree", [False, True])}
            tot, thr, qv, n = evaluate(params)
            trial.set_user_attr("thr_q", qv)
            trial.set_user_attr("thr_bp", thr)
            trial.set_user_attr("val_trades", n)
            return tot
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=Y))
        study.optimize(objective, n_trials=a.trials)
        bp = study.best_params
        qv = study.best_trial.user_attrs["thr_q"]
        thr = float(study.best_trial.user_attrs["thr_bp"])          # in predicted bp, set out of sample on Y-1
        XB, yB = build_training(mk, trB, rng)
        mB = fit(bp, XB, yB)
        best_t, arg_t = best_combo(mB, mk.F.values[te])
        tr = policy(mk, te, best_t, arg_t, thr)
        for (t, i, k, net, gr) in tr:
            s, tp, sl, h = GRID[k]
            trades_all.append({"time": mk.bk.idx[i], "side": int(s), "tp": tp, "sl": sl, "h": int(h), "net": net,
                               "gross": gr, "pred_bp": float(best_t[np.searchsorted(te, t)])})
        chosen.append({"year": Y, "params": bp, "threshold_quantile": qv, "threshold_bp": thr,
                       "validation_net_bp_total": study.best_value, "trades": len(tr)})
        print(Y, "val", round(study.best_value, 1), "q", qv, "thr", round(thr, 2), "test trades", len(tr),
              f"{time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(trades_all)
    json.dump(chosen, open(os.path.join(a.out, f"chosen_{a.symbol}_{a.tf}.json"), "w"), indent=1, default=str)
    row = {"symbol": a.symbol, "tf": a.tf, "trades": int(len(T)), "years_tested": len(chosen)}
    if len(T):
        T.to_csv(os.path.join(a.out, f"trades_{a.symbol}_{a.tf}.csv.gz"), index=False)
        x = T.set_index("time").net.sort_index()
        tt = TB.month_t(x)
        # random entries with the model's yearly counts, long share and chosen combinations
        draws, draws_g = [], []
        rr = np.random.default_rng(19)
        T["year"] = pd.DatetimeIndex(T.time).year
        for d in range(a.draws):
            nets, grs = [], []
            for Y, g in T.groupby("year"):
                cand = np.flatnonzero((mk.years == Y) & valid)
                picks = np.sort(rr.choice(cand, size=min(len(cand), 3 * len(g)), replace=False))
                combos = g[["side", "tp", "sl", "h"]].to_numpy(float)
                busy, k = -1, 0
                for t in picks:
                    if k >= len(g):
                        break
                    i = int(mk.first_m1[min(t + 1, len(mk.first_m1) - 1)])
                    if i <= busy:
                        continue
                    gg = combos[rr.integers(0, len(combos))].copy()
                    gg[0] = 1.0 if rr.random() < (g.side > 0).mean() else -1.0
                    n_, g_, e_ = mk.simulate(np.array([t]), gg[None, :])
                    if not np.isfinite(n_[0]):
                        continue
                    nets.append(n_[0])
                    grs.append(g_[0])
                    busy = int(e_[0])
                    k += 1
            draws.append(np.mean(nets) if nets else np.nan)
            draws_g.append(np.mean(grs) if grs else np.nan)
        draws, draws_g = np.array(draws), np.array(draws_g)
        row.update({"mean_bp": float(x.mean() * 1e4), "t": tt,
                    "p_one_sided": float(1 - st.norm.cdf(tt)) if np.isfinite(tt) else np.nan,
                    "gross_bp": float(T.gross.mean() * 1e4), "hit": float((x > 0).mean()),
                    "pred_bp_mean": float(T.pred_bp.mean()), "long_share": float((T.side > 0).mean()),
                    "median_tp": float(np.nanmedian(np.where(np.isfinite(T.tp), T.tp, np.nan))),
                    "share_no_tp": float((~np.isfinite(T.tp)).mean()), "share_no_sl": float((~np.isfinite(T.sl)).mean()),
                    "median_sl": float(np.nanmedian(np.where(np.isfinite(T.sl), T.sl, np.nan))),
                    "median_h": float(T.h.median()),
                    "random_mean_bp": float(np.nanmean(draws) * 1e4), "random_p": float(np.nanmean(draws >= x.mean())),
                    "random_gross_bp": float(np.nanmean(draws_g) * 1e4),
                    "random_p_gross": float(np.nanmean(draws_g >= T.gross.mean())),
                    "first": str(x.index[0])[:10],
                    "years_positive": f"{int((x.groupby(x.index.year).sum() > 0).sum())}/{x.index.year.nunique()}"})
    print(row, flush=True)
    pd.DataFrame([row]).to_csv(os.path.join(a.out, f"tests_{a.symbol}_{a.tf}.csv"), index=False)
    json.dump({"symbol": a.symbol, "tf": a.tf, "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(a.out, f"meta_{a.symbol}_{a.tf}.json"), "w"))


def combine(a):
    fs = glob.glob(os.path.join(a.combine, "**", "tests_*.csv"), recursive=True)
    T = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    T = T[T.trades > 0].copy()
    from cross_market_ta_lab import bh_q
    T["q_bh"] = bh_q(T.p_one_sided.to_numpy())
    os.makedirs(a.out, exist_ok=True)
    T.sort_values("p_one_sided").to_csv(os.path.join(a.out, "tests_all.csv"), index=False)
    g = T.groupby("tf").agg(tests=("t", "size"), median_net=("mean_bp", "median"), median_gross=("gross_bp", "median"),
                            median_random_gross=("random_gross_bp", "median"),
                            share_gross_beats=("random_p_gross", lambda x: (x < 0.05).mean()),
                            share_net_positive=("mean_bp", lambda x: (x > 0).mean()),
                            median_trades=("trades", "median"), median_pred=("pred_bp_mean", "median"))
    g.to_csv(os.path.join(a.out, "by_timeframe.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(g.round(2).to_string())
    print(T.sort_values("p_one_sided")[["symbol", "tf", "trades", "mean_bp", "t", "gross_bp", "pred_bp_mean", "hit",
                                        "median_tp", "median_sl", "median_h", "random_gross_bp", "random_p_gross",
                                        "q_bh", "years_positive"]].round(2).to_string())
    json.dump({"tests": int(len(T)), "q10": int((T.q_bh <= 0.10).sum()), "t_gt_2": int((T.t > 2).sum()),
               "t_lt_m2": int((T.t < -2).sum())}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--combine", default="")
    ap.add_argument("--budget-min", type=float, default=270, help="stop starting new test years after this")
    ap.add_argument("--min-train", type=int, default=2000, help="training bars needed for model A")
    ap.add_argument("--min-val", type=int, default=300, help="validation-year bars needed (D1 has ~260 a year)")
    a = ap.parse_args()
    combine(a) if a.combine else run(a)


if __name__ == "__main__":
    main()
