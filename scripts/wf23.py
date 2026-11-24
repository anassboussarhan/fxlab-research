"""Round 23 selection core (docs/round23_preregistration.md): a general walk-forward and the nested walk-forward that
chooses the selection setting itself.

walk_forward(agg, first_oos_year, setting): at each re-selection point the configurations are scored on a training
window of earlier months (expanding or the last w months) by their t-statistic or mean; the top k eligible ones are
traded with equal weight until the next re-selection. With setting DEFAULT it reproduces wf22.walk_forward exactly.
nested(agg, first_oos_year): every setting in SETTINGS gives an inner out-of-sample stream; each January from the
inner start + 3 years the setting whose inner stream has the best t over all earlier years is chosen for that year.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import wf22

DEFAULT = (None, 5, "Y", "t")
SETTINGS = [DEFAULT] + [(w, k, r, s) for w in (None, 12, 24, 36, 60) for k in (1, 3, 5, 10) for r in ("M", "Q", "Y")
                        for s in ("t", "mean") if (w, k, r, s) != DEFAULT]


LAST_YEAR = 2025                       # the last selection year (2026 only in pre-registered checks)

def name(s):
    w, k, r, sc = s
    return f"w{'exp' if w is None else w}_top{k}_{r}_{sc}"


class Agg:
    """Per-month aggregates (sum, sum of squares, active count, valid count) of each configuration, plus a way to
    get daily rows. Give either P (n_cfg x n_days, NaN = no data) or the aggregates and a rows function."""

    def __init__(self, dates, P=None, S=None, Q=None, A=None, V=None, rows=None):
        self.dates = pd.DatetimeIndex(dates)
        per = self.dates.to_period("M")
        mk = np.asarray(per.year * 12 + per.month - 1)
        change = np.r_[True, mk[1:] != mk[:-1]]
        self.mstart = np.flatnonzero(change)                               # first day index of each month
        self.mkey = mk[self.mstart]                                        # year * 12 + month - 1
        if P is not None:
            P = np.asarray(P, np.float64)
            X = np.where(np.isfinite(P), P, 0.0)
            self.X = X
            red = lambda Z: np.add.reduceat(Z, self.mstart, axis=1)        # noqa: E731
            S, Q = red(X), red(X ** 2)
            A, V = red((np.isfinite(P) & (P != 0)).astype(np.float64)), red(np.isfinite(P).astype(np.float64))
            rows = lambda idx, d0, d1: self.X[idx][:, d0:d1]               # noqa: E731
        self.n = S.shape[0]
        z = np.zeros((self.n, 1))
        self.cS, self.cQ = np.hstack([z, np.cumsum(S, 1)]), np.hstack([z, np.cumsum(Q, 1)])
        self.cA, self.cV = np.hstack([z, np.cumsum(A, 1)]), np.hstack([z, np.cumsum(V, 1)])
        self.rows = rows


def walk_forward(ag: Agg, first_oos_year, setting=DEFAULT, min_active=30, min_valid=50, min_train_years=3,
                 last_year=None):
    last_year = LAST_YEAR if last_year is None else last_year
    w, top, rs, sc = setting
    yrs = ag.mkey // 12
    mon = ag.mkey % 12 + 1
    y0 = int(ag.dates.year.min())
    first = max(first_oos_year, y0 + min_train_years)
    step = {"M": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), "Q": (1, 4, 7, 10), "Y": (1,)}[rs]
    pts = [b for b in range(len(ag.mkey)) if yrs[b] >= first and yrs[b] <= last_year and mon[b] in step]
    nd = len(ag.dates)
    out = np.full(nd, np.nan)
    choices = {}
    last_day = np.searchsorted(ag.dates.values, np.datetime64(pd.Timestamp(f"{last_year}-12-31")), side="right")
    for j, b in enumerate(pts):
        nxt = pts[j + 1] if j + 1 < len(pts) else None
        d0 = ag.mstart[b]
        d1 = ag.mstart[nxt] if nxt is not None else last_day
        lo = 0 if w is None else max(0, b - w)
        S, Q = ag.cS[:, b] - ag.cS[:, lo], ag.cQ[:, b] - ag.cQ[:, lo]
        A, V = ag.cA[:, b] - ag.cA[:, lo], ag.cV[:, b] - ag.cV[:, lo]
        n = np.maximum(V, 1)
        mu = S / n
        if sc == "t":
            var = np.maximum(Q / n - mu ** 2, 1e-30) * n / np.maximum(n - 1, 1)
            score = mu / np.sqrt(var / n)
        else:
            score = mu.copy()
        score[(A < min_active) | (V < min_valid)] = -np.inf
        order = np.argsort(-score, kind="stable")
        pick = [int(i) for i in order[:top] if np.isfinite(score[i])]
        if d1 > d0:
            out[d0:d1] = ag.rows(pick, d0, d1).mean(0) if pick else 0.0
        key = f"{yrs[b]}-{mon[b]:02d}"
        choices[key] = pick
    s = pd.Series(out, index=ag.dates).dropna()
    return s, choices


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 12 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def nested(ag: Agg, first_oos_year, min_active=30, min_valid=50, min_train_years=3, outer_years=3, last_year=None,
           settings=SETTINGS):
    last_year = LAST_YEAR if last_year is None else last_year
    inner = {}
    for s in settings:
        inner[name(s)], _ = walk_forward(ag, first_oos_year, s, min_active, min_valid, min_train_years, last_year)
    D = pd.DataFrame(inner)
    if D.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([])), {}, D
    y_first = int(D.index.year.min())
    out, ch = [], {}
    for Y in range(y_first + outer_years, last_year + 1):
        tr = D[D.index.year < Y]
        te = D[D.index.year == Y]
        if te.empty:
            continue
        ts = np.array([tstat(tr[c].to_numpy()) for c in D.columns])
        ts = np.where(np.isfinite(ts), ts, -np.inf)
        best = D.columns[int(np.argmax(ts))]                               # ties go to the earlier setting (default first)
        ch[Y] = best
        out.append(te[best])
    s = pd.concat(out) if out else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return s, ch, D


def both(ag: Agg, first_oos_year, **kw):
    """The two counted streams of a unit: default and nested, with their statistics."""
    lim = {k: kw[k] for k in ("min_active", "min_valid", "min_train_years", "last_year") if k in kw}
    d, dch = walk_forward(ag, first_oos_year, DEFAULT, **lim)
    n, nch, _ = nested(ag, first_oos_year, **kw)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    sd = wf22.stream_stats(d if len(d) else empty)
    sn = wf22.stream_stats(n if len(n) else empty)
    return {"default": (d, sd, wf22.counts_rule(sd) if len(d) else False, dch),
            "nested": (n, sn, wf22.counts_rule(sn) if len(n) else False, nch)}


def nested_row(base_row, P, dates, first_oos_year, **kw):
    """Round 23 part N: the nested stream for a round-22 unit, as a row like base_row (selection = nested)."""
    s, ch, _ = nested(Agg(dates, P=P), first_oos_year, **kw)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    st = wf22.stream_stats(s if len(s) else empty)
    r = {k: v for k, v in base_row.items() if k not in st and k not in ("top1_p2_mean", "top1_p2_t", "own", "choices")}
    r.update(st)
    r.update({"selection": "nested", "own": wf22.counts_rule(st) if len(s) else False,
              "choices": json.dumps({str(k): v for k, v in list(ch.items())[-3:]})})
    return r, s
