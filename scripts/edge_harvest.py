"""Edge harvest: try to turn the detected (small) predictive edge into net profit with every lever available:
probability thresholds, sizing, pyramiding, holding rules, stops/targets/trailing, and passive (limit-order)
execution that earns the spread instead of paying it. One instrument x one timeframe per run.

1. FORECAST (walk-forward, 3-year training / 6-month test, first test 2010): in every fold the K = 30 feature
   columns (all 269 features, raw and |d20) with the largest training information coefficient are combined by
   ridge regression into a forecast of the next-h-bar return (h = 1, 6, 24). Features standardised and winsorised
   with training statistics only.
2. POLICY SELECTION without look-ahead: inside each training window the model is fitted on the first 75% and
   forecasts the last 25% (pseudo out-of-sample); every trading policy is simulated there after costs and the
   best one (training net Sharpe, enough trades) is chosen. The model is then refitted on the whole training
   window and the chosen policy trades the test window.
3. POLICY GRID (86 policies per horizon):
   - threshold (probability filter): trade only when |forecast| >= theta x round-trip cost, theta in {0.5, 1, 2};
     exit by hysteresis (|forecast| < theta/2 x cost) or after holding h bars since the last trigger;
   - sizing: unit or proportional to the forecast (Kelly-like, capped at 1 x notional);
   - pyramiding: start at 1/3 and add 1/3 per further bar of signal, up to 1;
   - execution: market at the next open, or a passive limit at the best bid/ask that fills only on a strict
     trade-through (no queue priority, adverse selection marked) and falls back to market after 1 or 3 bars;
   - stop/target variants through the trade engine: 2 ATR stop + 2 ATR target, or 3 ATR trailing stop, entry at
     market or a resting limit at the signal close (3 bars), with risk-based sizing (0.5% per trade);
   - plus "always in" sign(forecast) at market and passive, as references.
Outputs DIR/<pair>_<tf>.json (selected-policy OOS results per horizon, every policy's OOS Sharpe as a diagnostic)
and DIR/<pair>_<tf>_daily.parquet (daily OOS net returns of the selected policy per horizon).
Usage: python scripts/edge_harvest.py --pair EURUSD --tf H1 [--out DIR]"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

HS = (1, 6, 24)
K_FEAT = 30
RIDGE = 10.0
MIN_TRADES = 10


@njit(cache=False)
def _policy(f, thr, exit_mode, size_mode, pyr, h):
    """Target position from a forecast series f (return units; NaN = no forecast) and a per-bar threshold array
    thr (theta x the round-trip cost estimated on the fold's TRAINING window). exit_mode 0 = hysteresis,
    1 = hold h bars after the last trigger. size_mode 0 = unit, 1 = proportional. pyr: pyramid in thirds."""
    n = len(f)
    out = np.zeros(n)
    d = 0.0; size = 0.0; last = -1; units = 0.0
    for t in range(n):
        x = f[t]
        th = thr[t]
        if x != x:
            d = 0.0; size = 0.0; units = 0.0
            continue
        trig = abs(x) >= th
        sx = 1.0 if x > 0 else -1.0
        if trig:
            if d != sx:
                d = sx; units = 1.0 / 3.0 if pyr else 1.0
            elif pyr:
                units = min(1.0, units + 1.0 / 3.0)
            size = min(1.0, abs(x) / (2.0 * th)) if size_mode == 1 else 1.0
            last = t
        elif d != 0.0:
            if exit_mode == 0:
                if abs(x) < 0.5 * th or (x > 0) != (d > 0):
                    d = 0.0
            else:
                if t - last >= h:
                    d = 0.0
        out[t] = d * size * (units if pyr else 1.0)
    return out


def build_policies():
    P = []
    for th, ex, sz, py, exe in itertools.product((0.5, 1.0, 2.0), (0, 1), (0, 1), (0, 1), ("market", "passive1", "passive3")):
        P.append({"kind": "signal", "theta": th, "exit": ["hysteresis", "hold_h"][ex], "size": ["unit", "prop"][sz],
                  "pyramid": bool(py), "exec": exe})
    for th, risk, ent in itertools.product((0.5, 1.0, 2.0), ("sl2_tp2", "trail3"), ("market", "limit_touch")):
        P.append({"kind": "stops", "theta": th, "risk": risk, "entry": ent})
    for exe in ("market", "passive1"):
        P.append({"kind": "always", "exec": exe})
    return P


RISK = {"sl2_tp2": {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine", "target_params": {"a": 0, "b": 1.0}},
        "trail3": {"stop": "atr", "stop_params": {"mult": 3.0}, "target": "none", "target_params": {}, "trail_atr": 3.0}}


def simulate(bars, f, pol, c_rt, h, pip, cost, roll):
    """Net bar returns (pd.Series) and number of position changes for one policy on the bar slice `bars`.
    c_rt: per-bar round-trip cost estimate (array, from each fold's training window)."""
    from fxlab.execution.simulator import vector_backtest, passive_backtest, simulate_trades
    from fxlab.execution.risk import RiskConfig, build_orders
    k = pol["kind"]
    if k == "always":
        tgt = np.where(np.isfinite(f), np.sign(f), 0.0)
        ex = pol["exec"]
    elif k == "signal":
        tgt = _policy(f, pol["theta"] * c_rt, 0 if pol["exit"] == "hysteresis" else 1,
                      1 if pol["size"] == "prop" else 0, pol["pyramid"], h)
        ex = pol["exec"]
    else:
        side = _policy(f, pol["theta"] * c_rt, 0, 0, False, h)
        orders = build_orders(bars, pd.Series(side, index=bars.index), RiskConfig(**RISK[pol["risk"]]), pip)
        res = simulate_trades(bars, orders, pip, cost, exit_on_flat=True, rollovers=roll,
                              entry=pol["entry"], k=0.0, ttl=3 if pol["entry"] != "market" else 0)
        return res.bars["net"], int(len(res.trades))
    ps = pd.Series(tgt, index=bars.index)
    if ex == "market":
        bt = vector_backtest(bars, ps, pip, cost, roll)
    else:
        bt = passive_backtest(bars, ps, pip, cost, offset=0.0, max_wait=1 if ex == "passive1" else 3, rollovers=roll)
    return bt["net"], int((np.abs(np.diff(np.r_[0.0, tgt])) > 0).sum())


def dsharpe(r):
    from fxlab.validation.metrics import daily
    d = daily(r)
    return (float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0), d


def fit_forecast(Xs, y, idx_fit, idx_pred, K=K_FEAT, lam=RIDGE):
    """Top-K columns by training IC, ridge on standardised/winsorised features; forecasts for idx_pred."""
    x_fit = Xs[idx_fit]
    lo, hi = np.nanpercentile(x_fit, [1, 99], axis=0)
    x_fit = np.clip(x_fit, lo, hi)
    mu, sd = np.nanmean(x_fit, 0), np.nanstd(x_fit, 0) + 1e-15
    z = np.nan_to_num((x_fit - mu) / sd)
    yy = np.nan_to_num(y[idx_fit])
    ic = z.T @ yy / len(yy) / (np.sqrt((yy ** 2).mean()) + 1e-15)
    top = np.argsort(-np.abs(ic))[:K]
    Z = z[:, top]
    beta = np.linalg.solve(Z.T @ Z + lam * len(Z) / 100.0 * np.eye(len(top)), Z.T @ yy)
    zp = np.nan_to_num((np.clip(Xs[idx_pred][:, top], lo[top], hi[top]) - mu[top]) / sd[top])
    return zp @ beta, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-est-seconds", type=float, default=150.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "edge_harvest"))
    a = ap.parse_args()
    import feature_scan as FS
    from fxlab.config import RESULTS_DIR
    from fxlab.execution.costs import BASE, STRESS
    from fxlab.execution.simulator import rollover_counts
    from fxlab.data.instruments import get as instrument
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    bars = FS.cached_bars(a.pair, a.tf, a.start, a.end)
    panel = FS.load_panel_closes([s for s in FS.ALL if s != a.pair], a.tf, bars.index, a.start, a.end)
    bench = json.loads((RESULTS_DIR / "benchmark.json").read_text())["features"]
    X, _ = FS.build_features(bars, a.pair, a.tf, panel, a.max_est_seconds, bench)
    real = list(X.columns)
    D = (X[real] - X[real].ewm(span=20, adjust=False, min_periods=20).mean()).astype("float32")
    D.columns = [f"{c}|d20" for c in real]
    X = pd.concat([X.astype("float32"), D], axis=1)
    cols = list(X.columns)
    Xs = X.to_numpy(np.float32)
    del X, D
    n = len(bars)
    pip = instrument(a.pair).pip
    mid = ((bars["bid_c"] + bars["ask_c"]) / 2).to_numpy(float)
    lr = np.r_[np.diff(np.log(mid)), 0.0]
    cs = np.r_[0.0, np.cumsum(lr)]
    sp = (bars["ask_c"] - bars["bid_c"]).to_numpy(float)
    c_rt_series = 2 * ((BASE.spread_mult / 2 + BASE.slip_spread_frac) * sp + BASE.slip_fixed_pips * pip) / mid \
        + 2 * BASE.commission_bps_per_side * 1e-4
    roll = rollover_counts(bars.index)
    tr_lo, tr_hi, te_hi = FS.folds(bars.index)
    P = build_policies()
    out = {"pair": a.pair, "tf": a.tf, "bars": n, "features_columns": len(cols), "policies": len(P), "horizons": {}}
    daily_out = {}
    for h in HS:
        y = np.full(n, np.nan)
        y[: n - h] = cs[h + 1: n + 1] - cs[1: n - h + 1]
        f_oos = np.full(n, np.nan)
        inner = []
        tops = []
        for f in range(len(tr_lo)):
            tr = np.arange(tr_lo[f], max(tr_lo[f], tr_hi[f] - h))
            te = np.arange(tr_hi[f], te_hi[f])
            if len(tr) < 400 or len(te) < 20:
                continue
            k = int(len(tr) * 0.75)
            a_idx, b_idx = tr[: k - h], tr[k:]
            fb, _ = fit_forecast(Xs, y, a_idx, b_idx)
            fte, top = fit_forecast(Xs, y, tr, te)
            f_oos[te] = fte
            c_rt = float(np.nanmedian(c_rt_series[tr]))
            inner.append({"fold": f, "b": b_idx, "fb": fb, "te": te, "c_rt": c_rt})
            tops.append([cols[j] for j in top[:5]])
        # OOS forecast quality
        ok = np.isfinite(f_oos) & np.isfinite(y)
        ic_oos = float(np.corrcoef(f_oos[ok], y[ok])[0, 1]) if ok.sum() > 50 else np.nan
        hit = float((np.sign(f_oos[ok]) == np.sign(y[ok]))[y[ok] != 0].mean()) if ok.sum() > 50 else np.nan
        # every policy on the full OOS forecast series (one simulation each); per-fold slices are used below
        oos_net, oos_trades = [], []
        c_arr = np.full(n, np.inf)                                 # thresholds from each fold's TRAINING costs
        for it in inner:
            c_arr[it["te"]] = it["c_rt"]
        c_rt_all = float(np.nanmedian([x["c_rt"] for x in inner])) if inner else np.nan
        for pol in P:
            r, ntr = simulate(bars, f_oos, pol, c_arr, h, pip, BASE, roll)
            oos_net.append(r.to_numpy()); oos_trades.append(ntr)
        oos_net = np.array(oos_net)
        # policy selection per fold on the inner (pseudo-OOS) window
        chosen, sel = np.full(n, -1), []
        for it in inner:
            b = it["b"]
            bb = bars.iloc[b[0]: b[-1] + 1]
            fb_full = np.full(len(bb), np.nan); fb_full[b - b[0]] = it["fb"]
            best, best_s = None, -np.inf
            for pi, pol in enumerate(P):
                r, ntr = simulate(bb, fb_full, pol, np.full(len(bb), it["c_rt"]), h, pip, BASE, roll[b[0]: b[-1] + 1])
                s, _ = dsharpe(r)
                if ntr >= MIN_TRADES and s > best_s:
                    best, best_s = pi, s
            if best is None:
                best = len(P) - 2                                  # no policy trades enough: always-in market
            chosen[it["te"]] = best
            sel.append({"fold": it["fold"], "policy": best, "inner_sharpe": best_s})
        m = chosen >= 0
        sel_net = np.zeros(n)
        sel_net[m] = oos_net[chosen[m], np.flatnonzero(m)]
        rs = pd.Series(sel_net, index=bars.index)[m]
        s_sel, d_sel = dsharpe(rs)
        # gross of the selected policies (zero costs) and 2x costs, re-simulated per chosen policy
        from fxlab.execution.costs import ZERO
        gross = np.zeros(n); x2 = np.zeros(n)
        for pi in sorted(set(chosen[m].tolist())):
            mm = chosen == pi
            rg, _ = simulate(bars, f_oos, P[pi], c_arr, h, pip, ZERO, roll)
            r2, _ = simulate(bars, f_oos, P[pi], c_arr, h, pip, STRESS[2.0], roll)
            gross[mm] = rg.to_numpy()[mm]; x2[mm] = r2.to_numpy()[mm]
        s_gross, _ = dsharpe(pd.Series(gross, index=bars.index)[m])
        s_x2, _ = dsharpe(pd.Series(x2, index=bars.index)[m])
        yrs = max(1e-9, (bars.index[m][-1] - bars.index[m][0]).days / 365.25) if m.any() else np.nan
        per_policy = []
        for pi, pol in enumerate(P):
            s, _ = dsharpe(pd.Series(oos_net[pi], index=bars.index)[m])
            per_policy.append({"policy": pi, "sharpe": s, "trades_per_year": oos_trades[pi] / yrs if yrs == yrs else np.nan})
        best_hind = max(per_policy, key=lambda x: x["sharpe"])
        counts = pd.Series([P[s["policy"]].get("exec", P[s["policy"]].get("entry", "")) + ":" + P[s["policy"]]["kind"]
                            for s in sel]).value_counts().to_dict()
        out["horizons"][str(h)] = {
            "ic_oos": ic_oos, "hit_rate": hit, "cost_rt_bps": c_rt_all * 1e4,
            "selected": {"net_sharpe": s_sel, "gross_sharpe": s_gross, "cost_x2_sharpe": s_x2,
                         "ann_return": float(d_sel.mean() * 260), "days": int(len(d_sel)),
                         "t": float(d_sel.mean() / d_sel.std() * np.sqrt(len(d_sel))) if d_sel.std() > 0 else 0.0},
            "selection": sel, "choice_counts": counts,
            "per_policy_oos_sharpe": per_policy, "best_in_hindsight": {**best_hind, **P[best_hind["policy"]]},
            "top_features_last_folds": tops[-3:]}
        daily_out[f"h{h}"] = d_sel
        print(f"{a.pair} {a.tf} h={h}: OOS IC {ic_oos:+.4f} hit {hit:.4f} | selected net {s_sel:+.2f} gross {s_gross:+.2f} "
              f"x2 {s_x2:+.2f} | hindsight best {best_hind['sharpe']:+.2f} | {counts}", flush=True)
    out["policy_grid"] = P
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, f"{a.pair}_{a.tf}.json"), "w"), indent=1, default=float)
    pd.DataFrame(daily_out).astype("float32").to_parquet(os.path.join(a.out, f"{a.pair}_{a.tf}_daily.parquet"))
    print(f"done {a.pair} {a.tf} in {out['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
