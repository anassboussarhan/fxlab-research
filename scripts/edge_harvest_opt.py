"""Edge harvest with OPTIMISED trade management: every lever is a continuous (or integer) parameter searched by
TPE (Optuna) inside each walk-forward fold, instead of a fixed grid.

Forecast: as in scripts/edge_harvest.py (top-30 features by training IC -> ridge forecast of the next-h-bar return,
h = 1, 6, 24; 3-year training / 6-month test, first test 2010).
Optimisation without look-ahead: inside each training window the model is fitted on the first 75% and forecasts
the last 25%; the search maximises the daily net Sharpe (after costs, >= 10 trades) of the trading rule on that
pseudo-out-of-sample slice. The model is then refitted on the whole training window and the best parameters trade
the following 6-month test window.
Search space (N_TRIALS = 80 per fold and horizon):
  mode            signal-managed position, or stop/target-managed trades (trade engine)
  theta           entry threshold, 0.2 - 5 x round-trip cost (log)            -- probability filter
  exit            hysteresis (exit when |f| < exit_frac x theta, 0 - 0.95) or hold k bars (1 - 4h)
  sizing          unit, or proportional with scale 1 - 5 (Kelly-like, capped at 1 x notional)
  pyramiding      off, or add pyr_step (0.1 - 1) of a unit per further bar of signal
  execution       market, or passive limit at bid/ask + offset (0 - 0.5 spread towards mid), falls back to
                  market after 1 - 12 bars (fills only on strict trade-through)
  stops mode      stop 0.3 - 8 x ATR; target none or 0.3 - 6 R; trailing none or 0.5 - 8 x ATR; breakeven none or
                  0.3 - 3 R; partial take-profit (half) none or 0.3 - 3 R; time exit none or 2 - 100 bars; entry market,
                  limit at the signal close, limit pullback k x ATR (0.05 - 1.5) or stop breakout k x ATR, order life
                  1 - 12 bars; sizing 0.5% risk per trade
Outputs DIR/<pair>_<tf>.json and DIR/<pair>_<tf>_daily.parquet in the format of edge_harvest.py, plus the chosen
parameters per fold. Usage: python scripts/edge_harvest_opt.py --pair EURUSD --tf H1 [--out DIR]"""
from __future__ import annotations

import argparse
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

from edge_harvest import fit_forecast, dsharpe, HS  # noqa: E402

N_TRIALS = int(os.environ.get("EHO_TRIALS", 80))
MIN_TRADES = 10
WARMUP = 300


@njit(cache=False)
def _policy2(f, thr, exit_mode, exit_frac, hold, size_mode, scale, pyr, pyr_step):
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
                d = sx; units = pyr_step if pyr else 1.0
            elif pyr:
                units = min(1.0, units + pyr_step)
            size = min(1.0, abs(x) / (scale * th)) if size_mode == 1 else 1.0
            last = t
        elif d != 0.0:
            if exit_mode == 0:
                if abs(x) < exit_frac * th or (x > 0) != (d > 0):
                    d = 0.0
            else:
                if t - last >= hold:
                    d = 0.0
        out[t] = d * size * (units if pyr else 1.0)
    return out


def suggest(trial, h):
    p = {"mode": trial.suggest_categorical("mode", ["signal", "stops"]),
         "theta": trial.suggest_float("theta", 0.2, 5.0, log=True)}
    if p["mode"] == "signal":
        p["exit"] = trial.suggest_categorical("exit", ["hysteresis", "hold"])
        if p["exit"] == "hysteresis":
            p["exit_frac"] = trial.suggest_float("exit_frac", 0.0, 0.95)
        else:
            p["hold"] = trial.suggest_int("hold", 1, max(2, 4 * h))
        p["size"] = trial.suggest_categorical("size", ["unit", "prop"])
        if p["size"] == "prop":
            p["scale"] = trial.suggest_float("scale", 1.0, 5.0)
        p["pyramid"] = trial.suggest_categorical("pyramid", [False, True])
        if p["pyramid"]:
            p["pyr_step"] = trial.suggest_float("pyr_step", 0.1, 1.0)
        p["exec"] = trial.suggest_categorical("exec", ["market", "passive"])
        if p["exec"] == "passive":
            p["offset"] = trial.suggest_float("offset", 0.0, 0.5)
            p["wait"] = trial.suggest_int("wait", 1, 12)
    else:
        p["sl_atr"] = trial.suggest_float("sl_atr", 0.3, 8.0, log=True)
        if trial.suggest_categorical("use_tp", [False, True]):
            p["tp_r"] = trial.suggest_float("tp_r", 0.3, 6.0, log=True)
        if trial.suggest_categorical("use_trail", [False, True]):
            p["trail_atr"] = trial.suggest_float("trail_atr", 0.5, 8.0, log=True)
        if trial.suggest_categorical("use_be", [False, True]):
            p["be_r"] = trial.suggest_float("be_r", 0.3, 3.0)
        if trial.suggest_categorical("use_partial", [False, True]):
            p["partial_r"] = trial.suggest_float("partial_r", 0.3, 3.0)
        if trial.suggest_categorical("use_time", [False, True]):
            p["max_hold"] = trial.suggest_int("max_hold", 2, 100)
        p["entry"] = trial.suggest_categorical("entry", ["market", "limit_touch", "limit_pullback", "stop_breakout"])
        if p["entry"] in ("limit_pullback", "stop_breakout"):
            p["k"] = trial.suggest_float("k", 0.05, 1.5)
        if p["entry"] != "market":
            p["ttl"] = trial.suggest_int("ttl", 1, 12)
    return p


def simulate(bars, f, p, c_rt, pip, cost, roll):
    from fxlab.execution.simulator import vector_backtest, passive_backtest, simulate_trades
    from fxlab.execution.risk import RiskConfig, build_orders
    thr = p["theta"] * c_rt
    if p["mode"] == "signal":
        tgt = _policy2(f, thr, 0 if p["exit"] == "hysteresis" else 1, p.get("exit_frac", 0.5), p.get("hold", 1),
                       1 if p["size"] == "prop" else 0, p.get("scale", 2.0), bool(p["pyramid"]), p.get("pyr_step", 1.0))
        ps = pd.Series(tgt, index=bars.index)
        if p["exec"] == "market":
            bt = vector_backtest(bars, ps, pip, cost, roll)
        else:
            bt = passive_backtest(bars, ps, pip, cost, offset=p["offset"], max_wait=p["wait"], rollovers=roll)
        return bt["net"], int((np.abs(np.diff(np.r_[0.0, tgt])) > 0).sum())
    side = _policy2(f, thr, 0, 0.5, 1, 0, 2.0, False, 1.0)
    rc = RiskConfig(stop="atr", stop_params={"mult": p["sl_atr"]},
                    target="affine" if "tp_r" in p else "none",
                    target_params={"a": 0.0, "b": p["tp_r"]} if "tp_r" in p else {},
                    trail_atr=p.get("trail_atr"), breakeven_r=p.get("be_r"), partial_r=p.get("partial_r"),
                    partial_frac=0.5, max_hold=p.get("max_hold"))
    orders = build_orders(bars, pd.Series(side, index=bars.index), rc, pip)
    res = simulate_trades(bars, orders, pip, cost, exit_on_flat=True, rollovers=roll, entry=p["entry"],
                          k=p.get("k", 0.0), ttl=p.get("ttl", 0))
    return res.bars["net"], int(len(res.trades))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-est-seconds", type=float, default=150.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "edge_harvest_opt"))
    a = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import feature_scan as FS
    from fxlab.config import RESULTS_DIR
    from fxlab.execution.costs import BASE, STRESS, ZERO
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
    Xs = X.to_numpy(np.float32)
    del X, D
    n = len(bars)
    pip = instrument(a.pair).pip
    mid = ((bars["bid_c"] + bars["ask_c"]) / 2).to_numpy(float)
    cs = np.r_[0.0, np.cumsum(np.r_[np.diff(np.log(mid)), 0.0])]
    sp = (bars["ask_c"] - bars["bid_c"]).to_numpy(float)
    c_rt_series = 2 * ((BASE.spread_mult / 2 + BASE.slip_spread_frac) * sp + BASE.slip_fixed_pips * pip) / mid \
        + 2 * BASE.commission_bps_per_side * 1e-4
    roll = rollover_counts(bars.index)
    tr_lo, tr_hi, te_hi = FS.folds(bars.index)
    out = {"pair": a.pair, "tf": a.tf, "bars": n, "trials_per_fold": N_TRIALS, "horizons": {}}
    daily_out = {}
    for h in HS:
        y = np.full(n, np.nan)
        y[: n - h] = cs[h + 1: n + 1] - cs[1: n - h + 1]
        net = pd.Series(np.nan, index=bars.index); gross = net.copy(); x2 = net.copy()
        f_all = np.full(n, np.nan)
        sel = []
        for fo in range(len(tr_lo)):
            tr = np.arange(tr_lo[fo], max(tr_lo[fo], tr_hi[fo] - h))
            te = np.arange(tr_hi[fo], te_hi[fo])
            if len(tr) < 400 or len(te) < 20:
                continue
            k = int(len(tr) * 0.75)
            a_idx, b_idx = tr[: k - h], tr[k:]
            fb, _ = fit_forecast(Xs, y, a_idx, b_idx)
            c_rt = float(np.nanmedian(c_rt_series[tr]))
            lo_b = max(0, b_idx[0] - WARMUP)
            bb = bars.iloc[lo_b: b_idx[-1] + 1]
            fbf = np.full(len(bb), np.nan); fbf[b_idx - lo_b] = fb
            cb = np.full(len(bb), c_rt)
            rb = roll[lo_b: b_idx[-1] + 1]
            seg_b = slice(b_idx[0] - lo_b, None)

            def objective(trial):
                p = suggest(trial, h)
                try:
                    r, ntr = simulate(bb, fbf, p, cb, pip, BASE, rb)
                except Exception:
                    return -10.0
                if ntr < MIN_TRADES:
                    return -10.0
                s, _ = dsharpe(r.iloc[seg_b])
                return s if np.isfinite(s) else -10.0
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=fo * 7 + h))
            study.optimize(objective, n_trials=N_TRIALS)
            best = suggest(optuna.trial.FixedTrial(study.best_params), h)
            fte, _ = fit_forecast(Xs, y, tr, te)
            f_all[te] = fte
            lo_t = max(0, te[0] - WARMUP)
            bt_ = bars.iloc[lo_t: te[-1] + 1]
            ft = np.full(len(bt_), np.nan); ft[te - lo_t] = fte
            ct = np.full(len(bt_), c_rt)
            rt = roll[lo_t: te[-1] + 1]
            for ser, cm in ((net, BASE), (gross, ZERO), (x2, STRESS[2.0])):
                r, _ = simulate(bt_, ft, best, ct, pip, cm, rt)
                ser.iloc[te] = r.iloc[te - lo_t].to_numpy()
            sel.append({"fold": fo, "inner_sharpe": float(study.best_value), "params": best})
        m = net.notna()
        s_net, d_net = dsharpe(net[m])
        s_gross, _ = dsharpe(gross[m])
        s_x2, _ = dsharpe(x2[m])
        ok = np.isfinite(f_all) & np.isfinite(y)
        ic = float(np.corrcoef(f_all[ok], y[ok])[0, 1]) if ok.sum() > 50 else np.nan
        hit = float((np.sign(f_all[ok]) == np.sign(y[ok]))[y[ok] != 0].mean()) if ok.sum() > 50 else np.nan
        modes = pd.Series([s["params"]["mode"] + ":" + s["params"].get("exec", s["params"].get("entry", ""))
                           for s in sel]).value_counts().to_dict()
        out["horizons"][str(h)] = {
            "ic_oos": ic, "hit_rate": hit, "cost_rt_bps": float(np.nanmedian(c_rt_series)) * 1e4,
            "selected": {"net_sharpe": s_net, "gross_sharpe": s_gross, "cost_x2_sharpe": s_x2,
                         "ann_return": float(d_net.mean() * 260), "days": int(len(d_net)),
                         "t": float(d_net.mean() / d_net.std() * np.sqrt(len(d_net))) if d_net.std() > 0 else 0.0},
            "median_inner_sharpe": float(np.median([s["inner_sharpe"] for s in sel])) if sel else None,
            "choice_counts": modes, "selection": sel,
            "best_in_hindsight": {"sharpe": float("nan")}}
        daily_out[f"h{h}"] = d_net
        print(f"{a.pair} {a.tf} h={h}: OOS IC {ic:+.4f} hit {hit:.4f} | optimised net {s_net:+.2f} gross {s_gross:+.2f} "
              f"x2 {s_x2:+.2f} | median inner (training) Sharpe {out['horizons'][str(h)]['median_inner_sharpe']:+.2f} | {modes}",
              flush=True)
    out["policy_grid"] = []
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, f"{a.pair}_{a.tf}.json"), "w"), indent=1, default=float)
    pd.DataFrame(daily_out).astype("float32").to_parquet(os.path.join(a.out, f"{a.pair}_{a.tf}_daily.parquet"))
    print(f"done {a.pair} {a.tf} in {out['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
