"""ALL-FEATURE walk-forward scan: every output column of every registered feature is turned into a trading rule
and tested out of sample, after retail costs, with the same placebo columns run through the same pipeline.

Rule for a column x (decided at the close of bar t, executed at the next open):
    long  when x_t > hi_q      short when x_t < lo_q      (or the reverse, dir = -1)
    the position is kept for h bars after the last trigger, then closed (unless re-triggered)
    hi_q / lo_q = the (1-q) / q quantiles of x over the TRAINING window only
Grid per fold: q in {0.05, 0.1, 0.2, 0.33} x h in {1, 3, 6, 12, 24, 48} bars x dir in {+1, -1} = 48 rules.
Walk-forward: calendar, 3-year training window, 6-month test window, first test 2010-01-01, last test ends
2025-12-31 (2026 is never loaded). In every fold the rule with the best TRAINING net Sharpe (daily, after costs)
is used for the test window; if no rule has a positive training net Sharpe the column ABSTAINS (stays flat).
Selection uses a fast cost approximation identical in form to the vector engine; the reported out-of-sample
numbers come from re-running the chosen positions through the real vector engine (fxlab.execution).

Placebos: 20 pure-noise columns and 20 time-shuffled copies of real columns (same values, order destroyed).
Outputs (results/reports/feature_scan/<SYMBOL>_<TF>/): summary.csv (one row per column), oos_daily.parquet
(daily net returns of every column), feature_build.json (time and errors per feature, skipped features)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

import fxlab.features  # noqa: E402,F401
from fxlab.config import RESULTS_DIR, DEV_END  # noqa: E402
from fxlab.registry import by_kind  # noqa: E402
from fxlab.feature_store import compute_feature  # noqa: E402
from fxlab.orchestration.runners import cached_bars, load_panel_closes  # noqa: E402
from fxlab.execution.costs import BASE  # noqa: E402
from fxlab.execution.simulator import vector_backtest, rollover_counts  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.validation.metrics import daily, trading_day  # noqa: E402

ALL = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
       "GBPJPY", "AUDJPY", "XAUUSD", "XAGUSD"]
QS = np.array([0.05, 0.1, 0.2, 0.33])
HS = np.array([1, 3, 6, 12, 24, 48], dtype=np.int64)
OUT = RESULTS_DIR / "reports" / "feature_scan"


@njit(cache=False)
def _sim(x, lo, hi, h, r, c, fin, a, b, T, G, K):
    """Simulate the threshold rule over bars [a, b). T = target decided at close t; G/K = gross and cost of
    holding during bar t (held = T[t-1])."""
    tgt = 0.0
    cnt = 0
    prev_held = 0.0
    for t in range(a, b):
        held = T[t - 1] if t > a else 0.0
        G[t] = held * r[t]
        K[t] = abs(held - prev_held) * c[t] + abs(held) * fin[t]
        prev_held = held
        xv = x[t]
        s = 0.0
        if xv == xv:
            if xv > hi:
                s = 1.0
            elif xv < lo:
                s = -1.0
        if s != 0.0:
            tgt = s
            cnt = h
        else:
            if cnt > 0:
                cnt -= 1
            if cnt == 0:
                tgt = 0.0
        T[t] = tgt


@njit(cache=False)
def _daily_sharpe(v, day, a, b):
    d0 = day[a]
    nd = day[b - 1] - d0 + 1
    acc = np.zeros(nd)
    for t in range(a, b):
        acc[day[t] - d0] += v[t]
    m = acc.mean()
    s = acc.std()
    if s <= 0:
        return -99.0
    return m / s * np.sqrt(260.0)


@njit(cache=False)
def _scan_column(x, r, c, fin, day, tr_lo, tr_hi, te_hi, thr, hs):
    """thr[f, q, 0/1] = lo/hi thresholds of fold f. Returns the out-of-sample target (NaN outside test windows)
    and per fold: chosen q index, h index, direction, best training Sharpe."""
    n = len(x)
    out = np.full(n, np.nan)
    nf = len(tr_lo)
    choice = np.zeros((nf, 4))
    T = np.zeros(n)
    G = np.zeros(n)
    K = np.zeros(n)
    tmp = np.zeros(n)
    for f in range(nf):
        a, m, b = tr_lo[f], tr_hi[f], te_hi[f]
        best = -1e9
        bq, bh, bd = 0, 0, 1.0
        for qi in range(thr.shape[1]):
            lo, hi = thr[f, qi, 0], thr[f, qi, 1]
            if not (lo == lo and hi == hi) or hi <= lo:
                continue
            for hj in range(len(hs)):
                _sim(x, lo, hi, hs[hj], r, c, fin, a, m, T, G, K)
                for t in range(a, m):
                    tmp[t] = G[t] - K[t]
                s1 = _daily_sharpe(tmp, day, a, m)
                for t in range(a, m):
                    tmp[t] = -G[t] - K[t]
                s2 = _daily_sharpe(tmp, day, a, m)
                if s1 > best:
                    best, bq, bh, bd = s1, qi, hj, 1.0
                if s2 > best:
                    best, bq, bh, bd = s2, qi, hj, -1.0
        choice[f, 0], choice[f, 1], choice[f, 2], choice[f, 3] = bq, bh, bd, best
        if best > 0.0:
            _sim(x, thr[f, bq, 0], thr[f, bq, 1], hs[bh], r, c, fin, a, b, T, G, K)
            for t in range(m, b):
                out[t] = bd * T[t]
        else:
            for t in range(m, b):
                out[t] = 0.0
    return out, choice


def folds(index, first_test="2010-01-01", train_years=3, test_months=6):
    t0 = pd.Timestamp(first_test, tz="UTC")
    end = min(index[-1] + pd.Timedelta(seconds=1), pd.Timestamp(DEV_END))
    lo, mid, hi = [], [], []
    while t0 < end:
        t1 = min(t0 + pd.DateOffset(months=test_months), end)
        a = index.searchsorted(t0 - pd.DateOffset(years=train_years))
        m = index.searchsorted(t0)
        b = index.searchsorted(t1)
        if m - a > 500 and b > m:
            lo.append(a); mid.append(m); hi.append(b)
        t0 = t1
    return np.array(lo, np.int64), np.array(mid, np.int64), np.array(hi, np.int64)


def build_features(bars, sym, tf, panel, max_est, bench):
    fids = sorted(e.id for e in by_kind("feature"))
    scale = len(bars) / bench.get("bars", 12501)
    cols, log = [], {"skipped_estimated_too_slow": {}, "errors": {}, "seconds": {}}
    for fid in fids:
        est = bench.get("per_feature", {}).get(fid, 0.0) * scale
        if est > max_est:
            log["skipped_estimated_too_slow"][fid] = round(est, 1)
            continue
        t = time.time()
        try:
            cols.append(compute_feature(bars, fid, sym, tf, panel))
        except Exception as ex:
            log["errors"][fid] = repr(ex)[:300]
        log["seconds"][fid] = round(time.time() - t, 2)
    print(f"  {sym} {tf}: built {len(cols)} features in {sum(log['seconds'].values()):.0f}s "
          f"({len(log['errors'])} errors, {len(log['skipped_estimated_too_slow'])} skipped)", flush=True)
    X = pd.concat(cols, axis=1)
    return X, log


def scan(sym, tf, start, end, max_est, n_placebo=20):
    dest = OUT / f"{sym}_{tf}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "summary.csv").exists():
        print("exists", dest)
        return
    t_all = time.time()
    bars = cached_bars(sym, tf, start, end)
    panel = load_panel_closes([s for s in ALL if s != sym], tf, bars.index, start, end)
    bench = json.loads((RESULTS_DIR / "benchmark.json").read_text())["features"]
    X, log = build_features(bars, sym, tf, panel, max_est, bench)
    log["feature_build_seconds"] = round(time.time() - t_all, 1)
    rng = np.random.default_rng(12345)
    real_cols = list(X.columns)
    plac = {}
    for k in range(n_placebo):
        plac[f"PLACEBO.noise:{k}"] = rng.normal(size=len(X))
    for k, col in enumerate(rng.choice(real_cols, n_placebo, replace=False)):
        v = X[col].to_numpy(np.float64).copy()
        ok = np.isfinite(v)
        v[ok] = rng.permutation(v[ok])
        plac[f"PLACEBO.shuffled:{k}:{col}"] = v
    X = pd.concat([X, pd.DataFrame(plac, index=X.index)], axis=1)
    pip = instrument(sym).pip
    bo, ao = bars["bid_o"].to_numpy(float), bars["ask_o"].to_numpy(float)
    mo = (bo + ao) / 2
    mc = ((bars["bid_c"] + bars["ask_c"]) / 2).to_numpy(float)
    r = np.r_[mo[1:], mc[-1]] / mo - 1.0
    sp = ao - bo
    c = (BASE.spread_mult * sp / 2 + BASE.slip_spread_frac * sp + BASE.slip_fixed_pips * pip) / mo \
        + BASE.commission_bps_per_side * 1e-4
    roll = rollover_counts(bars.index)
    fin = roll * BASE.swap_pips_per_night * pip / mo
    day = pd.factorize(trading_day(bars.index))[0].astype(np.int64)
    tr_lo, tr_hi, te_hi = folds(bars.index)
    rows, D = [], {}
    t_scan = time.time()
    for col in X.columns:
        x = X[col].to_numpy(np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        thr = np.full((len(tr_lo), len(QS), 2), np.nan)
        for f in range(len(tr_lo)):
            w = x[tr_lo[f]:tr_hi[f]]
            w = w[np.isfinite(w)]
            if len(w) > 200 and np.nanstd(w) > 0:
                thr[f, :, 0] = np.quantile(w, QS)
                thr[f, :, 1] = np.quantile(w, 1 - QS)
        tgt, ch = _scan_column(x, r, c, fin, day, tr_lo, tr_hi, te_hi, thr, HS)
        oos = np.isfinite(tgt)
        if not oos.any():
            continue
        pos = pd.Series(np.nan_to_num(tgt), index=bars.index)
        bt = vector_backtest(bars, pos, pip, BASE, roll)
        net = bt["net"][oos]
        dn = daily(net)
        dg = daily(bt["gross"][oos])
        sd = dn.std()
        D[col] = dn.astype("float32")
        yrs = oos.sum() / len(bars) * (bars.index[-1] - bars.index[0]).days / 365.25
        turn = np.abs(np.diff(np.r_[0.0, bt["position"].to_numpy()]))[oos].sum() / max(yrs, 1e-9)
        active = (ch[:, 3] > 0).mean()
        rows.append({"column": col, "feature": col.split(":")[0], "placebo": col.startswith("PLACEBO"),
                     "oos_sharpe": float(dn.mean() / sd * np.sqrt(260)) if sd > 0 else 0.0,
                     "oos_gross_sharpe": float(dg.mean() / dg.std() * np.sqrt(260)) if dg.std() > 0 else 0.0,
                     "oos_ann_return": float(dn.mean() * 260), "oos_days": int(len(dn)),
                     "turnover_per_year": float(turn), "share_folds_active": float(active),
                     "t_stat": float(dn.mean() / sd * np.sqrt(len(dn))) if sd > 0 else 0.0,
                     "median_train_sharpe": float(np.median(ch[:, 3])),
                     "modal_q": float(QS[int(pd.Series(ch[:, 0]).mode()[0])]),
                     "modal_h": int(HS[int(pd.Series(ch[:, 1]).mode()[0])]),
                     "share_dir_plus": float((ch[:, 2] > 0).mean())})
    S = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    S.to_csv(dest / "summary.csv", index=False)
    pd.DataFrame(D).to_parquet(dest / "oos_daily.parquet")
    log.update({"symbol": sym, "tf": tf, "bars": len(bars), "columns": len(real_cols), "folds": len(tr_lo),
                "scan_seconds": round(time.time() - t_scan, 1), "total_seconds": round(time.time() - t_all, 1)})
    (dest / "feature_build.json").write_text(json.dumps(log, indent=1))
    real = S[~S.placebo]
    pl = S[S.placebo]
    print(f"{sym} {tf}: {len(real)} columns, median OOS net Sharpe {real.oos_sharpe.median():.2f}, "
          f"positive {(real.oos_sharpe > 0).mean():.2f}, best {real.oos_sharpe.max():.2f} | placebo median "
          f"{pl.oos_sharpe.median():.2f} best {pl.oos_sharpe.max():.2f} | {log['total_seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=ALL)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-est-seconds", type=float, default=150.0,
                    help="skip features whose benchmark-extrapolated build time exceeds this (documented)")
    a = ap.parse_args()
    for s in a.symbols:
        try:
            scan(s, a.tf, a.start, a.end, a.max_est_seconds)
        except FileNotFoundError as ex:
            print("missing data", s, ex)


if __name__ == "__main__":
    main()
