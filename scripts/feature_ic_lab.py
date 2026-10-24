"""Feature predictability lab: does each feature carry information about the NEXT price move?

For every output column of every registered feature (and its deviation from its own 20-bar EMA, `|d20`) on one
instrument x timeframe, a direct measure of predictive information, free of trading-rule search:

  ic_oos   out-of-sample information coefficient. In each walk-forward fold (3-year training, 6-month test, first
           test 2010) the SIGN of the feature's (uncentred) correlation with the next-h-bar return is learned on the
           training window, and the signed uncentred correlation is measured on the test window (feature
           winsorised and standardised with TRAINING statistics only; nothing is demeaned with test-window
           means, which would leak the window's path and bias persistent, level-like features). It equals the
           normalised out-of-sample P&L of holding a position proportional to the standardised feature.
           n-weighted average over folds. Horizons h = 1, 6, 24 bars.
  z        against a NULL that keeps the feature's own persistence: the target return series is circularly shifted
           by K = 20 random offsets (>= 10% of the sample). The same sign-learning / testing is run for every
           shift; z = (ic_oos - mean_null) / sd_null. (The earlier feature-scan placebos - i.i.d. noise and
           time-shuffled columns - did not keep persistence and mostly abstained, so they are not a valid null
           for slowly moving features.)
  spread_bps  economic size: mean next-h-bar return of the training top decile minus bottom decile of the
           feature, signed by the training sign, measured on the test window (bps); compared with the
           round-trip cost (spread x 1.25 + slippage + commission, both sides) in cost_rt_bps.
Usage: python scripts/feature_ic_lab.py --pair EURUSD --tf H1 [--out DIR]  ->  DIR/<pair>_<tf>.csv.gz"""
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

HS = (1, 6, 24)
K_NULL = int(os.environ.get("FI_NULLS", 20))
CHUNK = 192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-est-seconds", type=float, default=150.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "feature_ic"))
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import feature_scan as FS
    from fxlab.config import RESULTS_DIR
    from fxlab.execution.costs import BASE
    from fxlab.data.instruments import get as instrument
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    bars = FS.cached_bars(a.pair, a.tf, a.start, a.end)
    panel = FS.load_panel_closes([s for s in FS.ALL if s != a.pair], a.tf, bars.index, a.start, a.end)
    bench = json.loads((RESULTS_DIR / "benchmark.json").read_text())["features"]
    X, log = FS.build_features(bars, a.pair, a.tf, panel, a.max_est_seconds, bench)
    real = list(X.columns)
    D = (X[real] - X[real].ewm(span=20, adjust=False, min_periods=20).mean()).astype("float32")
    D.columns = [f"{c}|d20" for c in real]
    X = pd.concat([X.astype("float32"), D], axis=1)
    cols = list(X.columns)
    tb = time.time() - t0
    mid = ((bars["bid_c"] + bars["ask_c"]) / 2).to_numpy(float)
    lr = np.r_[np.diff(np.log(mid)), 0.0]                       # lr[t] = log return from close t to t+1
    n = len(mid)
    pip = instrument(a.pair).pip
    sp = (bars["ask_c"] - bars["bid_c"]).to_numpy(float)
    c_rt = 2 * ((BASE.spread_mult / 2 + BASE.slip_spread_frac) * sp + BASE.slip_fixed_pips * pip) / mid \
        + 2 * BASE.commission_bps_per_side * 1e-4
    tr_lo, tr_hi, te_hi = FS.folds(bars.index)
    rng = np.random.default_rng(777)
    shifts = rng.integers(int(0.1 * n), int(0.9 * n), size=K_NULL)
    cs = np.r_[0.0, np.cumsum(lr)]
    cost_bps = float(np.nanmedian(c_rt)) * 1e4
    ys = {}
    for h in HS:
        y = np.full(n, np.nan)
        y[: n - h] = cs[h + 1: n + 1] - cs[1: n - h + 1]         # sum of lr[t+1 .. t+h]: next-h-bar return
        ys[h] = y
    # target + K circularly shifted nulls for every horizon: columns [h0: real, nulls..., h1: real, nulls...]
    Y = np.column_stack([np.roll(ys[h], int(s)) if k else ys[h] for h in HS for k, s in enumerate([0] + list(shifts))])
    G = K_NULL + 1
    acc = np.zeros((len(cols), len(HS) * G)); wsum = np.zeros(len(HS))
    spread = np.zeros((len(cols), len(HS)))
    fold_rows = []
    for f in range(len(tr_lo)):
        tr = np.arange(tr_lo[f], max(tr_lo[f], tr_hi[f] - max(HS)))      # targets must end inside training
        te = np.arange(tr_hi[f], min(te_hi[f], n - max(HS)))
        if len(tr) >= 200 and len(te) >= 50:
            fold_rows.append((tr, te))
    for c0 in range(0, len(cols), CHUNK):
        cc = cols[c0:c0 + CHUNK]
        xt = X[cc].to_numpy(np.float64)
        for tr, te in fold_rows:
            # NO centering inside the test window: demeaning with the test-window mean leaks the window's own
            # path into the correlation and biases level-like (persistent) features (Stambaugh-type bias).
            # Uncentred, sum(z_t y_t) is the out-of-sample P&L of holding z_t, which is causal.
            Ytr_c = np.nan_to_num(Y[tr]); Yte_c = np.nan_to_num(Y[te])
            sy_tr = np.sqrt((Ytr_c ** 2).mean(0)) + 1e-15; sy_te = np.sqrt((Yte_c ** 2).mean(0)) + 1e-15
            xtr, xte = xt[tr], xt[te]
            q = np.nanpercentile(xtr, [1, 10, 90, 99], axis=0)
            lo_q, q10, q90, hi_q = q
            xtr_c = np.clip(xtr, lo_q, hi_q); xte_c = np.clip(xte, lo_q, hi_q)
            mu = np.nanmean(xtr_c, 0); sd = np.nanstd(xtr_c, 0) + 1e-15
            ztr = np.nan_to_num((xtr_c - mu) / sd); zte = np.nan_to_num((xte_c - mu) / sd)
            ic_tr = (ztr.T @ Ytr_c) / len(tr) / sy_tr                       # (c, H*G)
            ic_te = (zte.T @ Yte_c) / len(te) / (np.sqrt((zte ** 2).mean(0))[:, None] + 1e-15) / sy_te
            acc[c0:c0 + len(cc)] += np.sign(ic_tr) * ic_te * len(te)
            with np.errstate(invalid="ignore", divide="ignore"):
                ttop, tbot = xtr >= q90, xtr <= q10
                etop, ebot = xte >= q90, xte <= q10
                for hi_, h in enumerate(HS):
                    ytr_, yte_ = np.nan_to_num(ys[h][tr])[:, None], np.nan_to_num(ys[h][te])[:, None]
                    s_tr = (np.where(ttop, ytr_, 0).sum(0) / ttop.sum(0)) - (np.where(tbot, ytr_, 0).sum(0) / tbot.sum(0))
                    s_te = (np.where(etop, yte_, 0).sum(0) / etop.sum(0)) - (np.where(ebot, yte_, 0).sum(0) / ebot.sum(0))
                    spread[c0:c0 + len(cc), hi_] += np.nan_to_num(np.sign(s_tr) * s_te) * len(te)
    wtot = sum(len(te) for _, te in fold_rows)
    ic = acc / max(wtot, 1)
    rows = []
    for hi_, h in enumerate(HS):
        real_ic, null = ic[:, hi_ * G], ic[:, hi_ * G + 1:(hi_ + 1) * G]
        z = (real_ic - null.mean(1)) / (null.std(1) + 1e-12)
        for j, col in enumerate(cols):
            rows.append({"pair": a.pair, "tf": a.tf, "h": h, "column": col, "feature": col.split(":")[0],
                         "derived": col.endswith("|d20"), "ic_oos": real_ic[j], "null_mean": null[j].mean(),
                         "null_sd": null[j].std(), "z": z[j], "spread_bps": spread[j, hi_] / max(wtot, 1) * 1e4,
                         "cost_rt_bps": cost_bps})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, f"{a.pair}_{a.tf}.csv.gz"), index=False)
    log.update({"pair": a.pair, "tf": a.tf, "bars": n, "columns": len(cols), "build_seconds": round(tb, 1),
                "total_seconds": round(time.time() - t0, 1)})
    json.dump(log, open(os.path.join(a.out, f"{a.pair}_{a.tf}.log.json"), "w"), indent=1, default=str)
    s1 = out[out.h == 1]
    print(f"{a.pair} {a.tf}: {len(cols)} columns, h=1 median z {s1.z.median():.2f}, z>3 {(s1.z > 3).mean():.3f}, "
          f"max z {s1.z.max():.1f}, median |spread| {s1.spread_bps.abs().median():.2f}bp vs cost {s1.cost_rt_bps.iloc[0]:.2f}bp "
          f"| {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
