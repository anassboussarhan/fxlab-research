"""Round 22 walk-forward core (docs/round22_preregistration.md).

walk_forward(P, dates, first_oos_year): P is (n_cfg, n_periods) net returns per period (0 = flat, NaN = no data).
At the start of each year Y >= first_oos_year every configuration is scored on all periods before Y (expanding
window) by the t-statistic of its returns; the top 5 eligible (>= min_active active periods) are traded in Y with
equal weight. Returns the out-of-sample series (top-5 and top-1) and the choices.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEV_END = pd.Timestamp("2025-12-31")
SPLIT = pd.Timestamp("2019-01-01")


def walk_forward(P, dates, first_oos_year, top=5, min_active=30, last_year=2025, min_train_years=3, min_valid=50):
    P = np.asarray(P, np.float64)
    dates = pd.DatetimeIndex(dates)
    yrs = dates.year.to_numpy()
    X = np.where(np.isfinite(P), P, 0.0)
    A = (np.isfinite(P) & (P != 0)).astype(np.float64)
    V = np.isfinite(P).astype(np.float64)
    y0 = int(yrs.min())
    first = max(first_oos_year, y0 + min_train_years)
    years = list(range(y0, last_year + 1))
    # per-year aggregates: sum, sum of squares, active count, valid count
    agg = {}
    for y in years:
        m = yrs == y
        agg[y] = (X[:, m].sum(1), (X[:, m] ** 2).sum(1), A[:, m].sum(1), V[:, m].sum(1))
    cs = np.zeros(P.shape[0])
    cq = np.zeros(P.shape[0])
    ca = np.zeros(P.shape[0])
    cv = np.zeros(P.shape[0])
    out5 = pd.Series(np.nan, index=dates)
    out1 = pd.Series(np.nan, index=dates)
    choices = {}
    for y in years:
        if y >= first:
            n = np.maximum(cv, 1)
            mu = cs / n
            var = np.maximum(cq / n - mu ** 2, 1e-30) * n / np.maximum(n - 1, 1)
            t = mu / np.sqrt(var / n)
            t[(ca < min_active) | (cv < min_valid)] = -np.inf
            order = np.argsort(-t)
            pick = [int(i) for i in order[:top] if np.isfinite(t[i])]
            m = yrs == y
            if pick and m.any():
                out5[m] = X[pick][:, m].mean(0)
                out1[m] = X[pick[0], m]
                choices[y] = {"cfg": pick, "train_t": [float(t[i]) for i in pick]}
            elif m.any():
                out5[m] = 0.0
                out1[m] = 0.0
                choices[y] = {"cfg": [], "train_t": []}
        s, q, a_, v = agg[y]
        cs += s
        cq += q
        ca += a_
        cv += v
    keep = out5.notna()
    return out5[keep], out1[keep], choices


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) >= 12 and x.std() > 0 else float("nan")


def stream_stats(s):
    s = s.dropna()
    s = s[s.index <= DEV_END]
    a, b = s[s.index < SPLIT], s[s.index >= SPLIT]
    return {"oos_first": str(s.index.min().date()) if len(s) else None, "n": int(len(s)),
            "p1_n": int(len(a)), "p1_mean": float(a.mean()) if len(a) else float("nan"), "p1_t": tstat(a),
            "p2_n": int(len(b)), "p2_mean": float(b.mean()) if len(b) else float("nan"), "p2_t": tstat(b),
            "all_mean": float(s.mean()) if len(s) else float("nan"), "all_t": tstat(s),
            "active_share": float((s != 0).mean()) if len(s) else float("nan")}


def counts_rule(st):
    p1_ok = (st["p1_n"] == 0) or (st["p1_mean"] > 0)
    return bool(p1_ok and st["p2_mean"] > 0 and st["p2_t"] >= 1.645)
