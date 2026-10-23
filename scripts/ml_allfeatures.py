"""All-feature gradient-boosting model with a COST-AWARE trading threshold, purged walk-forward.

Per instrument / timeframe / horizon h:
  features : every cached feature column of the all-feature scan (same build, same exclusions)
  target   : forward h-bar mid log return from the next open (labels end at t+1+h)
  folds    : calendar walk-forward, 3-year training window, 6-month test window; training labels that end
             inside the test window or within an embargo of h+1 bars before it are purged
  inner    : the model is fitted on the first 75% of the training window; the threshold multiplier kappa
             is chosen on the last 25% (genuinely out-of-sample for that model, purged as well); the same
             model is then used on the test window (no refit, so kappa is never tuned on fitted data)
  rule     : long if prediction > kappa x round-trip cost, short if < -kappa x cost, otherwise flat;
             kappa in {0.1, 0.25, 0.5, 1, 1.5, 2, 3, inf} (inf = abstain; small
             kappas allowed because boosted predictions are shrunk towards zero)
  model    : LightGBM regressor, 300 trees, learning rate 0.03, 15 leaves, 30% features per tree,
             >= 200 rows per leaf, 1 thread, fixed seed
OOS positions are priced by the vector engine (retail ECN costs)."""
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

import fxlab.features  # noqa: E402,F401
from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.registry import by_kind  # noqa: E402
from fxlab.feature_store import compute_feature  # noqa: E402
from fxlab.orchestration.runners import cached_bars, load_panel_closes  # noqa: E402
from fxlab.execution.costs import BASE, STRESS  # noqa: E402
from fxlab.execution.simulator import vector_backtest, rollover_counts  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.validation.metrics import daily  # noqa: E402
from fxlab.validation.stats import sharpe_ci, psr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_scan import ALL, folds  # noqa: E402

KAPPAS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
OUT = RESULTS_DIR / "reports" / "ml_allfeatures"


def sr(d):
    d = pd.Series(d).dropna()
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0


def load_matrix(bars, sym, tf, start, end, max_est=150.0):
    panel = load_panel_closes([s for s in ALL if s != sym], tf, bars.index, start, end)
    bench = json.loads((RESULTS_DIR / "benchmark.json").read_text())["features"]
    scale = len(bars) / bench.get("bars", 12501)
    cols = []
    for e in sorted(by_kind("feature"), key=lambda e: e.id):
        if bench.get("per_feature", {}).get(e.id, 0.0) * scale > max_est:
            continue
        try:
            cols.append(compute_feature(bars, e.id, sym, tf, panel))
        except Exception:
            pass
    X = pd.concat(cols, axis=1).astype("float32")
    return X.replace([np.inf, -np.inf], np.nan)


def positions(pred, cost_rt, kappa):
    if not np.isfinite(kappa):
        return np.zeros(len(pred))
    s = pred / cost_rt
    return np.where(s > kappa, 1.0, np.where(s < -kappa, -1.0, 0.0))


def run(sym, tf, h, start, end):
    import lightgbm as lgb
    dest = OUT / f"{sym}_{tf}_h{h}"
    if (dest / "summary.json").exists():
        print("exists", dest)
        return json.loads((dest / "summary.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bars = cached_bars(sym, tf, start, end)
    X = load_matrix(bars, sym, tf, start, end)
    pip = instrument(sym).pip
    mo = ((bars["bid_o"] + bars["ask_o"]) / 2).to_numpy()
    y = pd.Series(np.log(np.r_[mo[1 + h:], [np.nan] * (1 + h)] / np.r_[mo[1:], [np.nan]]), index=bars.index)
    sp = (bars["ask_o"] - bars["bid_o"]).to_numpy()
    c1 = (sp / 2 + BASE.slip_spread_frac * sp + BASE.slip_fixed_pips * pip) / mo + BASE.commission_bps_per_side * 1e-4
    cost_rt = pd.Series(2 * c1, index=bars.index).rolling(500, min_periods=50).median().shift(1).fillna(pd.Series(2 * c1, index=bars.index)).to_numpy()
    roll = rollover_counts(bars.index)
    tr_lo, tr_hi, te_hi = folds(bars.index)
    stride = max(1, h // 4)              # overlapping labels carry little new information
    pos = np.full(len(bars), np.nan)
    pred_oos = np.full(len(bars), np.nan)
    fold_info = []
    Xv = X.to_numpy(np.float32)
    yv = y.to_numpy()
    for f in range(len(tr_lo)):
        a, m, b = int(tr_lo[f]), int(tr_hi[f]), int(te_hi[f])
        emb = h + 1
        split = a + int(0.75 * (m - a))
        fit_idx = np.arange(a, split - emb, stride)
        val_idx = np.arange(split, m - emb)
        fit_idx = fit_idx[np.isfinite(yv[fit_idx])]
        val_idx = val_idx[np.isfinite(yv[val_idx])]
        if len(fit_idx) < 1000 or len(val_idx) < 200:
            continue
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.3,
                                min_child_samples=200, subsample=0.8, subsample_freq=1, n_jobs=1, random_state=0,
                                verbose=-1)
        mdl.fit(Xv[fit_idx], yv[fit_idx])
        pv = mdl.predict(Xv[split:m])
        best_k, best_s = np.inf, 0.0
        vb = bars.iloc[split:m]
        for k in KAPPAS[:-1]:
            ps = positions(pv, cost_rt[split:m], k)
            r = vector_backtest(vb, pd.Series(ps, index=vb.index), pip, BASE, roll[split:m])["net"]
            s = sr(daily(r))
            if s > best_s:
                best_k, best_s = k, s
        pt = mdl.predict(Xv[m:b])
        pred_oos[m:b] = pt
        pos[m:b] = positions(pt, cost_rt[m:b], best_k)
        ic = pd.Series(pv).corr(pd.Series(yv[split:m]), method="spearman")
        fold_info.append({"test_start": str(bars.index[m].date()), "kappa": best_k, "val_net_sharpe": best_s,
                          "val_ic": float(ic) if np.isfinite(ic) else None})
    oos = np.isfinite(pos)
    bt = vector_backtest(bars, pd.Series(np.nan_to_num(pos), index=bars.index), pip, BASE, roll)
    dn, dg = daily(bt["net"][oos]), daily(bt["gross"][oos])
    bt2 = vector_backtest(bars, pd.Series(np.nan_to_num(pos), index=bars.index), pip, STRESS[2.0], roll)
    ok = oos & np.isfinite(yv)
    ic = pd.Series(pred_oos[ok]).corr(pd.Series(yv[ok]), method="spearman")
    _, lo, hi = sharpe_ci(dn.to_numpy(), n_boot=1000)
    res = {"symbol": sym, "tf": tf, "h": h, "features": int(X.shape[1]), "folds": len(fold_info),
           "oos_net_sharpe": sr(dn), "oos_net_ci95": [lo, hi], "oos_gross_sharpe": sr(dg),
           "oos_ann_return": float(dn.mean() * 260), "psr_0": float(psr(dn.to_numpy())),
           "cost_x2_sharpe": sr(daily(bt2["net"][oos])), "oos_ic_spearman": float(ic),
           "oos_hit_rate": float((np.sign(pred_oos[ok]) == np.sign(yv[ok])).mean()),
           "share_folds_abstain": float(np.mean([not np.isfinite(f["kappa"]) for f in fold_info])) if fold_info else None,
           "time_in_market": float(np.abs(np.nan_to_num(pos[oos])).mean()),
           "runtime_s": round(time.time() - t0, 1), "fold_info": fold_info}
    (dest / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    dn.to_frame("net").to_parquet(dest / "oos_daily.parquet")
    print(f"{sym} {tf} h={h}: net {res['oos_net_sharpe']:.2f} gross {res['oos_gross_sharpe']:.2f} IC {ic:.3f} "
          f"abstain {res['share_folds_abstain']:.2f} in-market {res['time_in_market']:.2f} ({res['runtime_s']}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["EURUSD"])
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 12, 24])
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    a = ap.parse_args()
    for s in a.symbols:
        for h in a.horizons:
            run(s, a.tf, h, a.start, a.end)


if __name__ == "__main__":
    main()
