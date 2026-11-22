"""Round 25 selection core (docs/round25_preregistration.md): the round-23 walk-forward with three more scores, the
288-setting nested selection ("nested-plus"), panel selection over instruments and the Newey-West t.

Scores at each re-selection point, over the training months [lo, b):
- t, mean: as scripts/wf23.py (bit for bit: the default setting reproduces wf23.walk_forward);
- nbr: the median t of the configuration and its 8 nearest neighbours in the unit's Sobol space;
- share: the share of training years with a positive sum, ties broken by t;
- decayH: t with monthly weights 0.5 ** (age / H) (H in months), expanding window only.
A plan is the list of (month index b, next month index, picked configurations); streams are built from a plan on any
instrument whose months are mapped onto the plan's months.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import wf22
import wf23

SETTINGS23 = wf23.SETTINGS
SETTINGS_PLUS = list(SETTINGS23) \
    + [(w, k, r, "nbr") for w in (None, 12, 24, 36, 60) for k in (1, 3, 5, 10) for r in ("M", "Q", "Y")] \
    + [(w, k, r, "share") for w in (None, 12, 24, 36, 60) for k in (1, 3, 5, 10) for r in ("M", "Q", "Y")] \
    + [(None, k, r, f"decay{h}") for h in (12, 24, 36, 60) for k in (1, 3, 5, 10) for r in ("M", "Q", "Y")]
FIXED_DIAG = {"nbr": (None, 5, "Y", "nbr"), "share": (None, 5, "Y", "share"), "decay24": (None, 5, "Y", "decay24")}
assert len(SETTINGS_PLUS) == 288


def name(s):
    return wf23.name(s)


class Agg25(wf23.Agg):
    """wf23.Agg plus the monthly arrays (for share and decay) and the unit's nearest-neighbour table."""

    def __init__(self, dates, P, U=None, k_nbr=8):
        super().__init__(dates, P=P)
        X = self.X
        P = np.asarray(P, np.float64)
        red = lambda Z: np.add.reduceat(Z, self.mstart, axis=1)            # noqa: E731
        self.S, self.Q = red(X), red(X ** 2)
        self.A = red((np.isfinite(P) & (P != 0)).astype(np.float64))
        self.V = red(np.isfinite(P).astype(np.float64))
        self.nbr = knn(U, k_nbr) if U is not None else None


def knn(U, k):
    U = np.asarray(U, float)
    d = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1, kind="stable")[:, : k + 1]                # itself first (distance 0)


class Pooled:
    """Panel aggregates: the monthly arrays of several instruments' Agg25 added on the union of their months."""

    def __init__(self, aggs):
        self.aggs = aggs
        self.mkey = np.array(sorted(set(np.concatenate([a.mkey for a in aggs]))))
        n = aggs[0].n
        M = len(self.mkey)
        self.S, self.Q, self.A, self.V = (np.zeros((n, M)) for _ in range(4))
        for a in aggs:
            pos = np.searchsorted(self.mkey, a.mkey)
            for dst, src in ((self.S, a.S), (self.Q, a.Q), (self.A, a.A), (self.V, a.V)):
                dst[:, pos] += src
        z = np.zeros((n, 1))
        self.cS, self.cQ = np.hstack([z, np.cumsum(self.S, 1)]), np.hstack([z, np.cumsum(self.Q, 1)])
        self.cA, self.cV = np.hstack([z, np.cumsum(self.A, 1)]), np.hstack([z, np.cumsum(self.V, 1)])
        self.n = n
        self.nbr = aggs[0].nbr
        self.y0 = int(min(int(a.dates.year.min()) for a in aggs))


def _t(S, Q, V):
    n = np.maximum(V, 1)
    mu = S / n
    var = np.maximum(Q / n - mu ** 2, 1e-30) * n / np.maximum(n - 1, 1)
    return mu / np.sqrt(var / n)


def scores(ag, lo, b, sc, min_active, min_valid):
    S, Q = ag.cS[:, b] - ag.cS[:, lo], ag.cQ[:, b] - ag.cQ[:, lo]
    A, V = ag.cA[:, b] - ag.cA[:, lo], ag.cV[:, b] - ag.cV[:, lo]
    bad = (A < min_active) | (V < min_valid)
    if sc == "mean":
        out = S / np.maximum(V, 1)
    else:
        t = _t(S, Q, V)
        if sc == "t":
            out = t
        elif sc == "nbr":
            tt = np.where(bad, -np.inf, t)
            out = np.median(tt[ag.nbr], axis=1) if ag.nbr is not None else tt
        elif sc == "share":
            yrs = ag.mkey[lo:b] // 12
            if len(yrs):
                cut = np.r_[0, np.flatnonzero(yrs[1:] != yrs[:-1]) + 1]
                Sy = np.add.reduceat(ag.S[:, lo:b], cut, axis=1)
                Vy = np.add.reduceat(ag.V[:, lo:b], cut, axis=1)
                share = ((Sy > 0) & (Vy > 0)).sum(1) / np.maximum((Vy > 0).sum(1), 1)
            else:
                share = np.zeros(ag.n)
            out = 1000.0 * share + np.clip(np.nan_to_num(t, nan=-100.0), -100, 100)
        elif sc.startswith("decay"):
            h = float(sc[5:])
            age = (b - 1) - np.arange(lo, b)
            w = 0.5 ** (age / h)
            Sw, Qw, Vw = ag.S[:, lo:b] @ w, ag.Q[:, lo:b] @ w, ag.V[:, lo:b] @ w
            V2 = ag.V[:, lo:b] @ (w * w)
            mu = Sw / np.maximum(Vw, 1e-12)
            var = np.maximum(Qw / np.maximum(Vw, 1e-12) - mu ** 2, 1e-30)
            neff = Vw ** 2 / np.maximum(V2, 1e-12)
            out = mu / np.sqrt(var / np.maximum(neff, 1))
        else:
            raise ValueError(sc)
    out = np.asarray(out, float).copy()
    out[bad | ~np.isfinite(out)] = -np.inf
    return out


def plan(ag, first_oos_year, setting, min_active=30, min_valid=50, min_train_years=3, last_year=2025, y0=None):
    """The re-selection plan: [(b, nxt, pick)] on ag's months (nxt None = to the end)."""
    w, top, rs, sc = setting
    if sc.startswith("decay"):
        w = None
    yrs = ag.mkey // 12
    mon = ag.mkey % 12 + 1
    y0 = int(ag.dates.year.min()) if y0 is None else y0
    first = max(first_oos_year, y0 + min_train_years)
    step = {"M": tuple(range(1, 13)), "Q": (1, 4, 7, 10), "Y": (1,)}[rs]
    pts = [b for b in range(len(ag.mkey)) if first <= yrs[b] <= last_year and mon[b] in step]
    out = []
    for j, b in enumerate(pts):
        lo = 0 if w is None else max(0, b - w)
        s = scores(ag, lo, b, sc, min_active, min_valid)
        order = np.argsort(-s, kind="stable")
        out.append((b, pts[j + 1] if j + 1 < len(pts) else None, [int(i) for i in order[:top] if np.isfinite(s[i])]))
    return out


def stream(ag, pl, mkey=None, last_year=2025):
    """Daily stream of instrument ag (an Agg25) following plan pl, whose month indices refer to mkey (default:
    ag's own months)."""
    mk = ag.mkey if mkey is None else mkey
    nd = len(ag.dates)
    last_day = np.searchsorted(ag.dates.values, np.datetime64(pd.Timestamp(f"{last_year}-12-31")), side="right")
    # first day index of the instrument at or after the start of each plan month
    dmk = np.asarray(ag.dates.to_period("M").year * 12 + ag.dates.to_period("M").month - 1)
    out = np.full(nd, np.nan)
    for b, nxt, pick in pl:
        d0 = int(np.searchsorted(dmk, mk[b], side="left"))
        d1 = int(np.searchsorted(dmk, mk[nxt], side="left")) if nxt is not None else last_day
        d1 = min(d1, last_day)
        if d1 > d0:
            out[d0:d1] = ag.X[pick][:, d0:d1].mean(0) if pick else 0.0
    return pd.Series(out, index=ag.dates).dropna()


def walk_forward(ag, first_oos_year, setting, **kw):
    pl = plan(ag, first_oos_year, setting, **kw)
    return stream(ag, pl, last_year=kw.get("last_year", 2025)), pl


def nested_from_inner(inner, outer_years=3, last_year=2025):
    """wf23.nested's outer step on a dict name -> inner stream."""
    D = pd.DataFrame(inner)
    if D.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([])), {}
    y_first = int(D.index.year.min())
    out, ch = [], {}
    for Y in range(y_first + outer_years, last_year + 1):
        tr, te = D[D.index.year < Y], D[D.index.year == Y]
        if te.empty:
            continue
        ts = np.array([wf23.tstat(tr[c].to_numpy()) for c in D.columns])
        ts = np.where(np.isfinite(ts), ts, -np.inf)
        best = D.columns[int(np.argmax(ts))]
        ch[Y] = best
        out.append(te[best].dropna())
    s = pd.concat(out) if out else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return s, ch


def nested(ag, first_oos_year, settings, **kw):
    inner = {name(s): walk_forward(ag, first_oos_year, s, **kw)[0] for s in settings}
    return nested_from_inner(inner, last_year=kw.get("last_year", 2025))


def pooled_stream(per_inst):
    """Equal weight over the instruments with a stream value that day."""
    D = pd.DataFrame(per_inst)
    return D.mean(axis=1, skipna=True).dropna() if not D.empty else pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def panel(aggs, first_oos_year, setting, **kw):
    """Panel selection: one plan on the pooled aggregates, applied to every instrument. Returns (pooled stream,
    {i: instrument stream}, plan)."""
    P = Pooled(aggs)
    P.dates = pd.DatetimeIndex([])                                         # unused by plan when y0 is given
    pl = plan(P, first_oos_year, setting, y0=P.y0, **kw)
    per = {i: stream(a, pl, mkey=P.mkey, last_year=kw.get("last_year", 2025)) for i, a in enumerate(aggs)}
    return pooled_stream(per), per, pl


def panel_pit(aggs, members, first_oos_year, setting, **kw):
    """Point-in-time panel (part ALT): in year Y the plan is made on the pooled aggregates of the year's members only
    (their whole earlier history) and applied to them during Y. members: list of year lists, one per instrument.
    Returns (pooled stream, {i: instrument stream}, plans by year)."""
    last_year = kw.get("last_year", 2025)
    years = sorted({y for ys in members for y in ys if first_oos_year <= y <= last_year})
    per, plans = {}, {}
    for Y in years:
        idx = [i for i, ys in enumerate(members) if Y in ys]
        if not idx:
            continue
        P = Pooled([aggs[i] for i in idx])
        P.dates = pd.DatetimeIndex([])
        pl = [pt for pt in plan(P, first_oos_year, setting, y0=P.y0, **kw) if P.mkey[pt[0]] // 12 == Y]
        plans[Y] = pl
        for i in idx:
            s = stream(aggs[i], pl, mkey=P.mkey, last_year=last_year)
            s = s[s.index.year == Y]
            per[i] = pd.concat([per[i], s]) if i in per else s
    return pooled_stream(per), per, plans


def panel_pit_nested(aggs, members, first_oos_year, settings, **kw):
    inner = {name(s): panel_pit(aggs, members, first_oos_year, s, **kw)[0] for s in settings}
    return nested_from_inner(inner, last_year=kw.get("last_year", 2025))


def panel_nested(aggs, first_oos_year, settings, **kw):
    inner = {}
    for s in settings:
        inner[name(s)] = panel(aggs, first_oos_year, s, **kw)[0]
    return nested_from_inner(inner, last_year=kw.get("last_year", 2025))


# ------------------------------------------------------------------------------------------------ statistics
def nw_t(x):
    """Newey-West t of the mean, lag floor(4 (n / 100) ^ (2/9))."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12 or x.std() == 0:
        return float("nan")
    L = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    lrv = e @ e / n
    for l in range(1, L + 1):
        lrv += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    if lrv <= 0:
        return float("nan")
    return float(x.mean() / math.sqrt(lrv / n))


def stats(s):
    s = s.dropna()
    st = wf22.stream_stats(s if len(s) else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    b = s[(s.index >= wf22.SPLIT) & (s.index <= wf22.DEV_END)]
    st["p2_hac_t"] = nw_t(b.to_numpy()) if len(b) else float("nan")
    return st


def counts_rule(st, use_p1=True):
    """Round-25 own rule: round 22-24's plus the Newey-West condition; part AD drops the P1 condition."""
    if not st.get("n"):
        return False
    p1_ok = (not use_p1) or (st["p1_n"] == 0) or (st["p1_mean"] > 0)
    hac = st.get("p2_hac_t")
    return bool(p1_ok and st["p2_mean"] > 0 and st["p2_t"] >= 1.645 and hac is not None and np.isfinite(hac)
                and hac >= 1.645)
