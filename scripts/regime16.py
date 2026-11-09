"""Round 16 engine (docs/round16_preregistration.md): walk-forward regime detection and per-regime strategy
selection over the strategy library, with benchmarks, placebos, the part-A diagnostics, the options tests (O2, O3)
and the dip-buying filter (D).

Usage:
  python scripts/regime16.py --data DIR --out DIR [--placebos 200] [--jobs 4]
  python scripts/regime16.py --synthetic leak|planted|null --out DIR      (checks of section 9)

Conventions
- Library month j (2008-01 .. 2025-12). The decision for month j is made at the close of month j-1.
- Feature row of month m = the market state at the close of month m.
- M2 / M3 / M5 (and the boosting model) are predictive: month j is labelled by feature row j-1, and the decision
  uses row j-1.
- M4 (HMM) is contemporaneous: month j is weighted by the smoothed state posterior of row j; the decision uses the
  filtered posterior of row j-1 times the transition matrix.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

INNER = ("2011-01", "2018-12")
CHECK = ("2019-01", "2025-12")
OOS0 = "2011-01"
FIT_A = ("2008-01", "2018-12")
COST = 0.03
WIN = 5.0
KGRID = {"FACTOR": (1, 2, 3), "UNIT": (1, 5, 20, 100), "CFG": (1, 10, 50, 200)}
PRICE = ["RV21", "RV_CHG", "RV252", "CORR63", "DISP21", "ER63", "VR5", "AC_W", "TREND12", "REV12", "CARRY_SD",
         "CARRY6", "USD3", "SPX3", "SPX_RV21", "GOLD3"]
OPT = ["VIX", "VIX_TS", "VVIX", "SKEW", "GVZ", "OVX", "EVZ", "VRP_SPX", "DVIX"]
CME = ["EUR_IV", "EUR_RR25", "JPY_IV", "JPY_RR25", "FX_VRP"]
MACRO = ["US2Y_CHG3", "SLOPE", "NFCI", "EPU"]
FEATS = PRICE + OPT + CME + MACRO
MIN_ROWS = 30
FACTORS = [f"F{i}" for i in range(1, 10)]


# ================================================================================================ statistics
def nw_t(x, lags=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else np.nan


def sharpe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * math.sqrt(12)) if len(x) > 2 and x.std() > 0 else np.nan


def bh(p, q=0.10):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok])]
    thr = q * np.arange(1, len(ok) + 1) / len(ok)
    passed = np.flatnonzero(p[o] <= thr)
    if len(passed):
        out[o[: passed.max() + 1]] = True
    return out


# ================================================================================================ data
class Tier:
    def __init__(self, name, R, names):
        self.name, self.names = name, list(names)
        self.R = np.asarray(R, float)                                      # T x S raw monthly net P&L
        self.z = scale(self.R)
        self.zw = np.clip(self.z, -WIN, WIN)
        self.elig = np.isfinite(self.z)


def scale(R):
    """z[t] = r[t] / RMS(r[t-36 .. t-1]), needing >= 12 active months in that window."""
    T, S = R.shape
    sq = np.vstack([np.zeros(S), np.cumsum(R * R, axis=0)])
    ac = np.vstack([np.zeros(S), np.cumsum(R != 0, axis=0)])
    sig = np.full((T, S), np.nan)
    for t in range(1, T):
        a = max(0, t - 36)
        ms = (sq[t] - sq[a]) / (t - a)
        ok = ((ac[t] - ac[a]) >= 12) & (ms > 0)
        sig[t, ok] = np.sqrt(ms[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        return R / sig


def unit_tier(cfg: Tier, index: pd.MultiIndex, name="UNIT"):
    keys = pd.Index(list(zip(index.get_level_values("unit").astype(str), index.get_level_values("dir").astype(int))))
    codes, uniq = pd.factorize(keys)
    T, S = cfg.z.shape
    U = len(uniq)
    zsum = np.zeros((T, U))
    cnt = np.zeros((T, U))
    zf = np.nan_to_num(cfg.z)
    fin = np.isfinite(cfg.z).astype(float)
    for u in range(U):
        cols = np.flatnonzero(codes == u)
        zsum[:, u] = zf[:, cols].sum(1)
        cnt[:, u] = fin[:, cols].sum(1)
    t = Tier.__new__(Tier)
    t.name, t.names = name, [f"{a}|{b:+d}" for a, b in uniq]
    with np.errstate(invalid="ignore", divide="ignore"):
        t.z = np.where(cnt > 0, zsum / cnt, np.nan)
    t.R = t.z
    t.zw = np.clip(t.z, -WIN, WIN)
    t.elig = np.isfinite(t.z)
    return t


class Data:
    def __init__(self, months, feat: pd.DataFrame, tiers: dict, placebo_tiers: dict):
        self.months = list(months)
        self.T = len(months)
        self.j0 = self.months.index(OOS0)
        self.tiers, self.ptiers = tiers, placebo_tiers
        fac = tiers["FACTOR"]
        F = feat.copy()
        # performance features 9, 10, 12 from the FACTOR tier's scaled returns (known at the month's close)
        zf = pd.DataFrame(fac.zw, index=self.months, columns=fac.names)
        for col, f, L in (("TREND12", "F2", 12), ("REV12", "F5", 12), ("CARRY6", "F1", 6)):
            F[col] = zf[f].rolling(L, min_periods=L).mean().reindex(F.index)
        self.dropped = [c for c in FEATS if c not in F or F[c].loc[self.months[0]:].notna().sum() < 24]
        self.feats = [c for c in FEATS if c not in self.dropped]
        self.sets = {"PRICE": [c for c in PRICE if c in self.feats], "OPT": [c for c in OPT if c in self.feats],
                     "CME": [c for c in CME if c in self.feats], "MACRO": [c for c in MACRO if c in self.feats]}
        self.sets["ALL"] = self.sets["PRICE"] + self.sets["OPT"] + self.sets["MACRO"]
        self.sets["ALLC"] = self.sets["ALL"] + self.sets["CME"]
        # aligned feature rows: cur[j] = row of month j; prev[j] = row of month j-1
        self.ftab = F
        rows = list(F.index.astype(str))
        self.ridx = np.array([rows.index(m) for m in self.months])
        self.FV = F[self.feats].to_numpy(float)                          # all rows (from 2007-01)
        self.fcol = {c: i for i, c in enumerate(self.feats)}
        self.mask_inner = np.array([INNER[0] <= m <= INNER[1] for m in self.months])
        self.mask_check = np.array([CHECK[0] <= m <= CHECK[1] for m in self.months])

    def rolled(self, offset):
        """A copy whose feature rows are circularly shifted by `offset` rows (placebo P3)."""
        d = object.__new__(Data)
        d.__dict__.update(self.__dict__)
        d.FV = np.roll(self.FV, offset, axis=0)
        return d


def load(data_dir):
    lib = pd.read_parquet(os.path.join(data_dir, "lib_cfg.parquet"))
    plc = pd.read_parquet(os.path.join(data_dir, "lib_placebo.parquet"))
    fac = pd.read_parquet(os.path.join(data_dir, "factors.parquet"))
    feat = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
    months = [str(c) for c in lib.columns]
    assert months[0] == "2008-01" and months[-1] == "2025-12", (months[0], months[-1])
    cfg = Tier("CFG", lib.to_numpy(float).T, [f"{u}|{c}|{int(d):+d}" for u, c, d in lib.index])
    unit = unit_tier(cfg, lib.index)
    facT = Tier("FACTOR", fac.reindex(months)[FACTORS].fillna(0.0).to_numpy(float), FACTORS)
    pcfg = Tier("CFG", plc.to_numpy(float).T, [f"{u}|{c}|{int(d):+d}" for u, c, d in plc.index])
    punit = unit_tier(pcfg, plc.index)
    feat.index = feat.index.astype(str)
    return Data(months, feat, {"FACTOR": facT, "UNIT": unit, "CFG": cfg}, {"UNIT": punit, "CFG": pcfg})


# ================================================================================================ allocation
def evaluate(D: Data, tier: Tier, scores, ks, mode="topk"):
    """scores: T x S (NaN = not eligible / no score) for the decision months. Returns {k: R series (T,)}."""
    out = {}
    T, S = tier.z.shape
    for k in (ks if mode == "topk" else (None,)):
        R = np.full(T, np.nan)
        wprev = np.zeros(S)
        for j in range(D.j0, T):
            s = scores[j]
            ok = np.isfinite(s) & tier.elig[j]
            w = np.zeros(S)
            if mode == "topk":
                idx = np.flatnonzero(ok & (s > 0))
                if len(idx) > k:
                    idx = idx[np.argsort(-s[idx], kind="stable")[:k]]
                w[idx] = 1.0 / k
            else:                                                          # equal weight over every positive
                idx = np.flatnonzero(ok & (s > 0))
                if len(idx):
                    w[idx] = 1.0 / len(idx)
            R[j] = float(np.nan_to_num(tier.z[j]) @ w) - COST * np.abs(w - wprev).sum()
            wprev = w
        out[k] = R
    return out


def window_start(j, window):
    return 0 if window == "exp" else max(0, j - 60)


def static_scores(D, tier, window, zw=None):
    zw = tier.zw if zw is None else zw
    z0 = np.nan_to_num(zw)
    fin = np.isfinite(zw)
    cs = np.vstack([np.zeros(z0.shape[1]), np.cumsum(z0, 0)])
    cn = np.vstack([np.zeros(z0.shape[1]), np.cumsum(fin, 0)])
    Sc = np.full(zw.shape, np.nan)
    for j in range(D.j0, D.T):
        a = window_start(j, window)
        n = cn[j] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            Sc[j] = np.where(n >= 6, (cs[j] - cs[a]) / n, np.nan)
    return Sc


def mom_scores(D, tier, L, zw=None):
    zw = tier.zw if zw is None else zw
    z0 = np.nan_to_num(zw)
    fin = np.isfinite(zw)
    cs = np.vstack([np.zeros(z0.shape[1]), np.cumsum(z0, 0)])
    cn = np.vstack([np.zeros(z0.shape[1]), np.cumsum(fin, 0)])
    Sc = np.full(zw.shape, np.nan)
    need = max(1, L // 2)
    for j in range(D.j0, D.T):
        a = max(0, j - L)
        n = cn[j] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            Sc[j] = np.where(n >= need, (cs[j] - cs[a]) / n, np.nan)
    return Sc


def conditional_scores(D, tier, G, P, window, n0, static):
    """G: T x K training weights of each library month; P: T x K decision-month regime probabilities (NaN row =
    fall back to the static score). Score = sum_r P[j, r] x shrunk conditional mean_r."""
    zw = tier.zw
    z0 = np.nan_to_num(zw)
    fin = np.isfinite(zw).astype(float)
    Sc = static.copy()
    for j in range(D.j0, D.T):
        p = P[j]
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(G[:j])):
            continue
        a = window_start(j, window)
        g = G[a:j]                                                         # (n, K)
        m_all = static[j]
        sc = np.zeros(z0.shape[1])
        for r in np.flatnonzero(p > 1e-12):
            wr = g[:, r]
            nr = wr @ fin[a:j]
            with np.errstate(invalid="ignore", divide="ignore"):
                mr = (wr @ z0[a:j]) / nr
                sh = (nr * mr + n0 * m_all) / (nr + n0)
            sh = np.where(nr >= 6, sh, m_all)
            sc += p[r] * sh
        Sc[j] = np.where(np.isfinite(m_all), sc, np.nan)
    return Sc


# ================================================================================================ regime models
def year_starts(D):
    return [j for j in range(D.j0, D.T) if j == D.j0 or D.months[j].endswith("-01")]


def refit_periods(D):
    ys = year_starts(D)
    return [(s, (ys[i + 1] if i + 1 < len(ys) else D.T)) for i, s in enumerate(ys)]


def set_matrix(D, set_name, tier_unit=None):
    if set_name == "RET":
        return None
    return D.FV[:, [D.fcol[c] for c in D.sets[set_name]]]


def fit_transform(Xtr, ncomp=3):
    from sklearn.decomposition import PCA
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    nc = min(ncomp, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=nc, random_state=0).fit((Xtr - mu) / sd)
    def tr(X):
        out = np.full((len(X), nc), np.nan)
        ok = np.all(np.isfinite(X), axis=1)
        if ok.any():
            out[ok] = pca.transform((X[ok] - mu) / sd)
        return out
    return tr


def cluster_labels(D, X, algo, K, seed=0):
    """Predictive labels, refitted each January on the rows known then: per refit period, G (training weight of
    each library month = one-hot label of row j-1, 0 if missing) and Gp (the decision months' one-hot labels)."""
    rows_prev = D.ridx - 1
    for s, e in refit_periods(D):
        last_row = D.ridx[s] - 1                                           # row of month s-1 (the last known)
        Xtr_all = X[: last_row + 1]
        ok = np.all(np.isfinite(Xtr_all), axis=1)
        if ok.sum() < MIN_ROWS:
            yield (s, e, None, None)
            continue
        tr = fit_transform(Xtr_all[ok])
        Ztr = tr(Xtr_all[ok])
        if algo == "kmeans":
            from sklearn.cluster import KMeans
            mdl = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(Ztr)
        else:
            from sklearn.mixture import GaussianMixture
            mdl = GaussianMixture(n_components=K, covariance_type="diag", n_init=5, random_state=seed,
                                  reg_covar=1e-4).fit(Ztr)
        Zall = tr(X)
        okall = np.all(np.isfinite(Zall), axis=1)
        oh = np.full((len(X), K), np.nan)
        oh[okall] = np.eye(K)[mdl.predict(Zall[okall])]
        Gp = np.where(rows_prev[:, None] >= 0, oh[np.maximum(rows_prev, 0)], np.nan)
        G = Gp.copy()
        G[~np.all(np.isfinite(G), axis=1)] = 0.0
        yield (s, e, G, Gp)


def split_labels(D, f, K):
    x = D.FV[:, D.fcol[f]]
    for s, e in refit_periods(D):
        last_row = D.ridx[s] - 1
        xt = x[: last_row + 1]
        xt = xt[np.isfinite(xt)]
        if len(xt) < MIN_ROWS:
            yield (s, e, None, None)
            continue
        cuts = np.quantile(xt, np.arange(1, K) / K)
        lab = np.where(np.isfinite(x), np.digitize(x, cuts), -1)
        oh = np.zeros((len(x), K))
        ok = lab >= 0
        oh[ok] = np.eye(K)[lab[ok]]
        oh[~ok] = np.nan
        rows_prev = D.ridx - 1
        Gp = np.where(rows_prev[:, None] >= 0, oh[np.maximum(rows_prev, 0)], np.nan)
        G = Gp.copy()
        G[~np.all(np.isfinite(G), axis=1)] = 0.0
        yield (s, e, G, Gp)


def hmm_labels(D, X, K, seed=0):
    """Contemporaneous HMM: training weight of month j = smoothed posterior of row j; decision probabilities for
    month j = filtered posterior of row j-1 x transition matrix."""
    from hmmlearn.hmm import GaussianHMM
    for s, e in refit_periods(D):
        last_row = D.ridx[s] - 1
        Xtr_all = X[: last_row + 1]
        ok = np.all(np.isfinite(Xtr_all), axis=1)
        if ok.sum() < MIN_ROWS:
            yield (s, e, None, None)
            continue
        first = int(np.argmax(ok))                                         # use the contiguous block from here
        blk = np.all(np.isfinite(Xtr_all[first:]), axis=1)
        if not blk.all():
            first = first + int(np.flatnonzero(~blk).max()) + 1
        if last_row + 1 - first < MIN_ROWS:
            yield (s, e, None, None)
            continue
        tr = fit_transform(Xtr_all[first:])
        Ztr = tr(Xtr_all[first:])
        best, bs = None, -np.inf
        for k in range(10):
            try:
                m = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, tol=1e-4,
                                random_state=seed + 101 * k, min_covar=1e-3).fit(Ztr)
                sc = m.score(Ztr)
                if np.isfinite(sc) and sc > bs:
                    best, bs = m, sc
            except Exception:                                              # noqa: BLE001
                continue
        if best is None:
            yield (s, e, None, None)
            continue
        Zall = tr(X)
        gam = np.full((len(X), K), np.nan)
        gam[first: last_row + 1] = best.predict_proba(Ztr)
        filt = np.full((len(X), K), np.nan)
        filt[last_row] = gam[last_row]
        last_need = D.ridx[e - 1] - 1 if e - 1 < D.T else last_row
        for r in range(last_row + 1, last_need + 1):
            seg = Zall[first: r + 1]
            if np.all(np.isfinite(seg)):
                filt[r] = best.predict_proba(seg)[-1]
                gam[r] = filt[r]                                           # later training months: filtered
        A = best.transmat_
        rows_cur = D.ridx
        G = gam[rows_cur].copy()
        G[~np.all(np.isfinite(G), axis=1)] = 0.0
        P = np.full((D.T, K), np.nan)
        for j in range(s, e):
            r = D.ridx[j] - 1
            if r >= 0 and np.all(np.isfinite(filt[r])):
                P[j] = filt[r] @ A
        yield (s, e, G, P)


def regime_scores(D, tier, gen, window, n0, static):
    Sc = static.copy()
    for s, e, G, P in gen:
        if G is None:
            continue
        part = conditional_scores(D, tier, G, P, window, n0, static)
        Sc[s:e] = part[s:e]
    return Sc


def ridge_scores(D, tier, set_name, alpha, static):
    X = set_matrix(D, set_name)
    Sc = static.copy()
    zw = tier.zw
    for s, e in refit_periods(D):
        rows_prev = D.ridx[:s] - 1
        ok_m = rows_prev >= 0
        months = np.flatnonzero(ok_m)
        Xm = X[rows_prev[months]]
        okx = np.all(np.isfinite(Xm), axis=1)
        months, Xm = months[okx], Xm[okx]
        if len(months) < MIN_ROWS:
            continue
        mu, sd = Xm.mean(0), Xm.std(0)
        sd[sd == 0] = 1.0
        Xs = (Xm - mu) / sd
        Y = zw[months]
        fin = np.isfinite(Y)
        with np.errstate(invalid="ignore", divide="ignore"):
            ym = np.nansum(Y, 0) / fin.sum(0)
        Yc = np.where(fin, Y - ym, 0.0)
        beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ Yc)
        for j in range(s, e):
            x = X[D.ridx[j] - 1]
            if not np.all(np.isfinite(x)):
                continue
            pred = ym + ((x - mu) / sd) @ beta
            Sc[j] = np.where(np.isfinite(static[j]) & (fin.sum(0) >= 12), pred, np.nan)
    return Sc


def gb_scores(D, tier, set_name, static):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = set_matrix(D, set_name)
    S = tier.zw.shape[1]
    Sc = static.copy()
    for s, e in refit_periods(D):
        rows, ys = [], []
        for j in range(1, s):
            x = X[D.ridx[j] - 1]
            if not np.all(np.isfinite(x)):
                continue
            for i in range(S):
                if np.isfinite(tier.zw[j, i]):
                    rows.append(np.r_[x, np.eye(S)[i]])
                    ys.append(tier.zw[j, i])
        if len(rows) < MIN_ROWS * S // 2:
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=20,
                                          random_state=0).fit(np.array(rows), np.array(ys))
        for j in range(s, e):
            x = X[D.ridx[j] - 1]
            if not np.all(np.isfinite(x)):
                continue
            Q = np.array([np.r_[x, np.eye(S)[i]] for i in range(S)])
            Sc[j] = np.where(np.isfinite(static[j]), m.predict(Q), np.nan)
    return Sc


def ret_matrix(D):
    """RET set: the UNIT tier's scaled returns as the HMM input (row = the month itself); a month counts once at
    least half of the units have a scaled return."""
    U = D.tiers["UNIT"].zw
    X = np.full((len(D.FV), U.shape[1]), np.nan)
    good = np.isfinite(U).mean(1) >= 0.5
    X[D.ridx[good]] = np.nan_to_num(U[good])
    return X


# ================================================================================================ variants
def variants(D, tier_name):
    V = []
    for w in ("exp", "60"):
        V.append(("B1", {"window": w}))
    for L in (1, 3, 6, 12, 24, 36):
        V.append(("M1", {"L": L}))
    for L in (3, 6, 12):
        V.append(("M1b", {"L": L}))
    for algo, K, st, n0, w in itertools.product(("kmeans", "gmm"), (2, 3, 4, 5, 6, 8), ("PRICE", "OPT", "ALL", "ALLC"),
                                                (0, 12), ("exp", "60")):
        V.append(("M2", {"algo": algo, "K": K, "set": st, "n0": n0, "window": w}))
    for f, K, n0, w in itertools.product(D.feats, (2, 3), (0, 12), ("exp", "60")):
        V.append(("M3", {"f": f, "K": K, "n0": n0, "window": w}))
    for K, st, n0 in itertools.product((2, 3, 4), ("PRICE", "OPT", "ALL", "ALLC", "RET"), (0, 12)):
        V.append(("M4", {"K": K, "set": st, "n0": n0}))
    for st, al in itertools.product(("PRICE", "OPT", "ALL", "ALLC"), (10, 100, 1000)):
        V.append(("M5", {"model": "ridge", "set": st, "alpha": al}))
    if tier_name == "FACTOR":
        for st in ("PRICE", "OPT", "ALL", "ALLC"):
            V.append(("M5", {"model": "gb", "set": st}))
    return V


class Cache:
    def __init__(self):
        self.lab = {}


def scores_for(D, tier, fam, p, cache, zw_override=None):
    """Score matrix of one variant (k is applied later)."""
    if fam == "B0":
        return np.where(tier.elig, 1.0, np.nan)
    if fam == "B1":
        return static_scores(D, tier, p["window"], zw_override)
    if fam in ("M1", "M1b"):
        return mom_scores(D, tier, p["L"], zw_override)
    win = p.get("window", "exp")
    static = static_scores(D, tier, win)
    if fam == "M2":
        key = ("M2", p["algo"], p["K"], p["set"])
        if key not in cache.lab:
            cache.lab[key] = list(cluster_labels(D, set_matrix(D, p["set"]), p["algo"], p["K"]))
        return regime_scores(D, tier, iter(cache.lab[key]), win, p["n0"], static)
    if fam == "M3":
        key = ("M3", p["f"], p["K"])
        if key not in cache.lab:
            cache.lab[key] = list(split_labels(D, p["f"], p["K"]))
        return regime_scores(D, tier, iter(cache.lab[key]), win, p["n0"], static)
    if fam == "M4":
        key = ("M4", p["K"], p["set"])
        if key not in cache.lab:
            X = ret_matrix(D) if p["set"] == "RET" else set_matrix(D, p["set"])
            cache.lab[key] = list(hmm_labels(D, X, p["K"]))
        return regime_scores(D, tier, iter(cache.lab[key]), "exp", p["n0"], static)
    if fam == "M5":
        if p["model"] == "ridge":
            return ridge_scores(D, tier, p["set"], p["alpha"], static)
        return gb_scores(D, tier, p["set"], static)
    raise ValueError(fam)


def pname(fam, p, k=None):
    s = fam + ":" + ",".join(f"{a}={b}" for a, b in p.items())
    return s + (f",k={k}" if k is not None else "")


def series_stats(D, R):
    a, b = R[D.mask_inner], R[D.mask_check]
    return {"inner_mean": float(np.nanmean(a)), "inner_t": nw_t(a), "inner_sharpe": sharpe(a),
            "check_mean": float(np.nanmean(b)), "check_t": nw_t(b), "check_sharpe": sharpe(b)}


def run_surface(D, tier_name, tier, log=print, only_fams=None, cache=None):
    """Every variant x k: R series and stats."""
    cache = Cache() if cache is None else cache
    rows, series = [], {}
    ks = KGRID[tier_name]
    t0 = time.time()
    V = [("B0", {})] + variants(D, tier_name)
    for n, (fam, p) in enumerate(V):
        if only_fams and fam not in only_fams:
            continue
        Sc = scores_for(D, tier, fam, p, cache)
        mode = "ew" if fam in ("B0", "M1b") else "topk"
        res = evaluate(D, tier, Sc, ks, mode)
        for k, R in res.items():
            nm = pname(fam, p, k)
            series[nm] = R
            rows.append({"tier": tier_name, "family": fam, "variant": nm, **p, "k": k, **series_stats(D, R)})
        if n % 50 == 0:
            log(f"  {tier_name} {n}/{len(V)} variants, {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), series, cache


def choose(surf, fam):
    s = surf[(surf.family == fam) & np.isfinite(surf.inner_t)]
    if s.empty:
        return None
    return s.loc[s.inner_t.idxmax()]


def params_of(row):
    keys = {"B1": ["window"], "M1": ["L"], "M1b": ["L"], "M2": ["algo", "K", "set", "n0", "window"],
            "M3": ["f", "K", "n0", "window"], "M4": ["K", "set", "n0"], "M5": ["model", "set", "alpha"]}[row.family]
    p = {}
    for k in keys:
        v = row.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        p[k] = int(v) if k in ("K", "L", "n0", "alpha") else v
    if row.family == "M5" and p.get("model") == "gb":
        p.pop("alpha", None)
    return p


def placebo_run(D, tier_name, tier, fam, p, k, offsets, jobs=1):
    """P3: the same setting with its regime input circularly shifted in time."""
    def one(o):
        if fam in ("M1", "M1b"):
            zr = np.roll(tier.zw, o, axis=0)
            Sc = mom_scores(D, tier, p["L"], zr)
        else:
            Dp = D.rolled(o)
            if fam == "M4" and p.get("set") == "RET":
                Dp = D
                Xr = np.roll(ret_matrix(D), o, axis=0)
                static = static_scores(D, tier, "exp")
                Sc = regime_scores(D, tier, hmm_labels(D, Xr, p["K"]), "exp", p["n0"], static)
                return sharpe(evaluate(D, tier, Sc, (k,), "topk")[k][D.mask_check])
            Sc = scores_for(Dp, tier, fam, p, Cache())
        mode = "ew" if fam == "M1b" else "topk"
        R = evaluate(D, tier, Sc, (k,) if mode == "topk" else (None,), mode)
        return sharpe(list(R.values())[0][D.mask_check])
    if jobs > 1:
        from joblib import Parallel, delayed
        chunks = np.array_split(np.asarray(offsets), jobs)
        res = Parallel(n_jobs=jobs)(delayed(lambda c: [one(o) for o in c])(c) for c in chunks)
        return np.array([x for c in res for x in c])
    return np.array([one(o) for o in offsets])


def offsets_for(D, n, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    N = len(D.FV)
    return rng.integers(24, N - 24, n)


# ================================================================================================ part A
def part_A1(D):
    out = {}
    for tn in ("UNIT", "FACTOR"):
        t = D.tiers[tn]
        for L in (12, 36):
            sc = mom_scores(D, t, L)
            rho = np.full(D.T, np.nan)
            for j in range(D.j0, D.T):
                ok = np.isfinite(sc[j]) & np.isfinite(t.z[j])
                if ok.sum() >= 5:
                    a = pd.Series(sc[j][ok]).rank().to_numpy()
                    b = pd.Series(t.z[j][ok]).rank().to_numpy()
                    rho[j] = np.corrcoef(a, b)[0, 1]
            oos = rho[D.j0:]
            out[f"{tn}_L{L}"] = {"mean_rho": float(np.nanmean(oos)), "t": nw_t(oos),
                                 "inner_mean": float(np.nanmean(rho[D.mask_inner])),
                                 "check_mean": float(np.nanmean(rho[D.mask_check])), "months": int(np.isfinite(oos).sum())}
    return out


def a2_rho(D, tier, x, K=3):
    rows_prev = D.ridx - 1
    xm = np.where(rows_prev >= 0, x[np.maximum(rows_prev, 0)], np.nan)
    fitm = np.array([FIT_A[0] <= m <= FIT_A[1] for m in D.months])
    xf = xm[fitm & np.isfinite(xm)]
    if len(xf) < MIN_ROWS:
        return np.nan
    cuts = np.quantile(xf, np.arange(1, K) / K)
    lab = np.where(np.isfinite(xm), np.digitize(xm, cuts), -1)
    zw = tier.zw
    d1, d2 = [], []
    for r in range(K):
        for mask, dst in ((fitm, d1), (D.mask_check, d2)):
            base = np.nanmean(np.where(mask[:, None], zw, np.nan), axis=0)
            cond = np.nanmean(np.where((mask & (lab == r))[:, None], zw, np.nan), axis=0)
            dst.append(cond - base)
    a, b = np.concatenate(d1), np.concatenate(d2)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else np.nan


def part_A2(D, nplac):
    rows = []
    for tn in ("UNIT", "FACTOR"):
        t = D.tiers[tn]
        for f in D.feats:
            x = D.FV[:, D.fcol[f]]
            rho = a2_rho(D, t, x)
            offs = offsets_for(D, nplac, f"A2|{tn}|{f}")
            null = np.array([a2_rho(D, t, np.roll(x, o)) for o in offs])
            null = null[np.isfinite(null)]
            p = (1 + (null >= rho).sum()) / (1 + len(null)) if np.isfinite(rho) and len(null) else np.nan
            rows.append({"tier": tn, "feature": f, "rho": rho, "null_mean": float(null.mean()) if len(null) else np.nan,
                         "null_p95": float(np.quantile(null, 0.95)) if len(null) else np.nan, "p": p})
    A2 = pd.DataFrame(rows)
    A2["bh_pass"] = False
    for tn, g in A2.groupby("tier"):
        A2.loc[g.index, "bh_pass"] = bh(g.p.to_numpy())
    return A2


# ================================================================================================ parts O3 and D
def lagged_feature(D, f, std_period):
    x = D.FV[:, D.fcol[f]]
    rows_prev = D.ridx - 1
    xm = np.where(rows_prev >= 0, x[np.maximum(rows_prev, 0)], np.nan)
    per = np.array([std_period[0] <= m <= std_period[1] for m in D.months])
    v = xm[per & np.isfinite(xm)]
    if len(v) < MIN_ROWS:
        return None, per
    return (xm - v.mean()) / v.std(), per


def part_O3(D):
    fac = D.tiers["FACTOR"]
    rows = []
    for f in D.feats:
        grp = "OPT" if f in OPT else "CME" if f in CME else "PRICE" if f in PRICE else "MACRO"
        xs, fit = lagged_feature(D, f, FIT_A)
        if xs is None:
            continue
        for i, fn in enumerate(fac.names):
            y = fac.zw[:, i]
            ok = fit & np.isfinite(xs) & np.isfinite(y)
            if ok.sum() < MIN_ROWS:
                continue
            b = np.polyfit(xs[ok], y[ok], 1)[0]
            u = np.sign(b) * np.clip(xs, -2, 2) * fac.z[:, i]
            uc = u[D.mask_check]
            rows.append({"group": grp, "feature": f, "factor": fn, "fit_slope": float(b), "fit_months": int(ok.sum()),
                         "check_mean": float(np.nanmean(uc)), "check_t": nw_t(uc)})
    O3 = pd.DataFrame(rows)
    O3["p"] = 1 - norm.cdf(O3.check_t.fillna(-9))
    O3["bh_pass"] = False
    opt = O3.group.isin(["OPT", "CME"])
    O3.loc[opt, "bh_pass"] = bh(O3.loc[opt, "p"].to_numpy())
    O3.loc[~opt, "bh_pass_info"] = bh(O3.loc[~opt, "p"].to_numpy())
    O3["counts"] = opt & O3.bh_pass & (O3.check_mean > 0) & (O3.check_t >= 1.645)
    return O3


def part_D(D):
    fac = D.tiers["FACTOR"]
    i7 = fac.names.index("F7")
    y, z = fac.zw[:, i7], fac.z[:, i7]
    rows = []
    for f in D.feats:
        x = D.FV[:, D.fcol[f]]
        rows_prev = D.ridx - 1
        xm = np.where(rows_prev >= 0, x[np.maximum(rows_prev, 0)], np.nan)
        fit = np.array([FIT_A[0] <= m <= FIT_A[1] for m in D.months])
        xf = xm[fit & np.isfinite(xm)]
        if len(xf) < MIN_ROWS:
            continue
        cuts = np.quantile(xf, [1 / 3, 2 / 3])
        lab = np.where(np.isfinite(xm), np.digitize(xm, cuts), -1)
        means = [np.nanmean(y[fit & (lab == r)]) for r in range(3)]
        best = int(np.nanargmax(means))
        keep = lab == best
        filt = np.where(keep, z, 0.0)
        diff = (filt - z)[D.mask_check]
        rows.append({"feature": f, "best_tercile": best, "fit_means": [round(float(m), 3) for m in means],
                     "cuts": [float(c) for c in cuts], "check_kept_share": float(np.mean(keep[D.mask_check])),
                     "check_filtered_mean": float(np.nanmean(filt[D.mask_check])),
                     "check_unfiltered_mean": float(np.nanmean(z[D.mask_check])),
                     "check_diff_mean": float(np.nanmean(diff)), "check_diff_t": nw_t(diff)})
    R = pd.DataFrame(rows)
    R["p"] = 1 - norm.cdf(R.check_diff_t.fillna(-9))
    R["bh_pass"] = bh(R.p.to_numpy())
    R["counts"] = R.bh_pass & (R.check_diff_mean > 0) & (R.check_diff_t >= 1.645)
    return R


# ================================================================================================ regime map
def regime_map(D, tier, fam, p, cache, top=5):
    """The chosen setting's regimes at the last refit: feature profile (standardised means) and top strategies."""
    if fam not in ("M2", "M3", "M4"):
        return None
    if fam == "M2":
        key = ("M2", p["algo"], p["K"], p["set"])
        cols = D.sets[p["set"]]
    elif fam == "M3":
        key = ("M3", p["f"], p["K"])
        cols = [p["f"]]
    else:
        key = ("M4", p["K"], p["set"])
        cols = D.sets.get(p["set"], [])
    if key not in cache.lab:
        scores_for(D, tier, fam, p, cache)
    s, e, G, P = cache.lab[key][-1]
    if G is None:
        return None
    K = G.shape[1]
    j = e - 1
    X = D.FV[:, [D.fcol[c] for c in cols]] if cols else None
    out = {"refit_month": D.months[s], "regimes": []}
    zw = tier.zw[:s]
    fin = np.isfinite(zw)
    for r in range(K):
        w = G[:s, r]
        nr = w @ fin
        with np.errstate(invalid="ignore", divide="ignore"):
            mr = (w @ np.nan_to_num(zw)) / nr
        mr = np.where(nr >= 6, mr, np.nan)
        order = np.argsort(-np.nan_to_num(mr, nan=-9))[:top]
        prof = {}
        if X is not None:
            rows = (D.ridx[:s] - 1) if fam != "M4" else D.ridx[:s]
            xx = X[np.maximum(rows, 0)]
            mu, sd = np.nanmean(X[: D.ridx[s]], 0), np.nanstd(X[: D.ridx[s]], 0)
            ww = w[:, None]
            okr = np.all(np.isfinite(xx), axis=1)
            if (w[okr]).sum() > 0:
                prof = {c: round(float(((xx[okr] - mu) / sd * ww[okr]).sum(0)[i] / w[okr].sum()), 2) for i, c in enumerate(cols[:20])}
        out["regimes"].append({"regime": r, "training_months": round(float(w.sum()), 1), "feature_profile_z": prof,
                               "top": [{"strategy": tier.names[i], "cond_mean_z": round(float(mr[i]), 3)} for i in order
                                       if np.isfinite(mr[i])]})
    pj = P[j] if P is not None else None
    out["last_decision_probs"] = [round(float(v), 3) for v in pj] if pj is not None and np.all(np.isfinite(pj)) else None
    return out


def bic_k(D):
    from sklearn.mixture import GaussianMixture
    res = {}
    end_row = D.ridx[D.months.index(FIT_A[1])]
    for st in ("PRICE", "OPT", "ALL", "ALLC"):
        X = set_matrix(D, st)[: end_row + 1]
        ok = np.all(np.isfinite(X), axis=1)
        if ok.sum() < MIN_ROWS:
            continue
        Z = fit_transform(X[ok])(X[ok])
        b = {K: float(GaussianMixture(K, covariance_type="diag", n_init=5, random_state=0, reg_covar=1e-4).fit(Z).bic(Z))
             for K in range(1, 9)}
        res[st] = {"bic": b, "best_K": int(min(b, key=b.get)), "rows": int(ok.sum())}
    return res


# ================================================================================================ main
def run_all(D, out, nplac=200, jobs=1, log=print):
    os.makedirs(out, exist_ok=True)
    summary = {"dropped_features": D.dropped, "sets": D.sets, "tiers": {k: len(v.names) for k, v in D.tiers.items()},
               "placebo_tiers": {k: len(v.names) for k, v in D.ptiers.items()}}
    log("sets " + json.dumps(D.sets))
    surfaces, chosen = [], []
    caches, series_all = {}, {}
    shared = Cache()                                                       # regime labels do not depend on the tier
    for tn in ("FACTOR", "UNIT", "CFG"):
        t0 = time.time()
        surf, ser, cache = run_surface(D, tn, D.tiers[tn], log, cache=shared)
        surfaces.append(surf)
        caches[tn], series_all[tn] = cache, ser
        log(f"{tn}: {len(surf)} rows in {time.time() - t0:.0f}s")
        b1 = choose(surf, "B1")
        R_b1 = ser[b1.variant]
        R_b0 = ser[surf[surf.family == "B0"].variant.iloc[0]]
        for fam in ("B1", "M1", "M1b", "M2", "M3", "M4", "M5"):
            row = choose(surf, fam)
            if row is None:
                continue
            R = ser[row.variant]
            bench = R_b0 if fam == "B1" else R_b1
            diff = (R - bench)[D.mask_check]
            rec = {"tier": tn, "family": fam, "variant": row.variant, **series_stats(D, R),
                   "bench": "B0" if fam == "B1" else b1.variant, "diff_check_mean": float(np.nanmean(diff)),
                   "diff_check_t": nw_t(diff)}
            chosen.append(rec)
    S = pd.concat(surfaces, ignore_index=True)
    C = pd.DataFrame(chosen)
    # P1 with BH over the 18 regime tests (B1 excluded)
    reg = C.family != "B1"
    C["p"] = 1 - norm.cdf(C.check_t.fillna(-9))
    C["bh_pass"] = False
    C.loc[reg, "bh_pass"] = bh(C.loc[reg, "p"].to_numpy())
    C["P1"] = (C.check_mean > 0) & (C.check_t >= 1.645) & C.bh_pass
    C["P2"] = C.diff_check_t >= 1.645
    # P3 placebos for every chosen regime setting
    p3, p95 = [], []
    for i, r in C.iterrows():
        if r.family == "B1":
            p3.append(np.nan)
            p95.append(np.nan)
            continue
        row = S[S.variant == r.variant].iloc[0]
        p = params_of(row)
        k = int(row.k) if pd.notna(row.k) else None
        t0 = time.time()
        sh = placebo_run(D, r.tier, D.tiers[r.tier], r.family, p, k, offsets_for(D, nplac, f"P3|{r.variant}|{r.tier}"), jobs)
        sh = sh[np.isfinite(sh)]
        q95 = float(np.quantile(sh, 0.95)) if len(sh) else np.nan
        p95.append(q95)
        p3.append(bool(np.isfinite(r.check_sharpe) and r.check_sharpe >= q95))
        log(f"  P3 {r.tier} {r.variant}: real {r.check_sharpe:.2f} vs placebo p95 {q95:.2f} ({time.time() - t0:.0f}s)")
    C["placebo_p95_sharpe"] = p95
    C["P3"] = p3
    # P4: the same family's inner selection on the placebo library
    p4 = []
    for i, r in C.iterrows():
        if r.tier not in D.ptiers or r.family == "B1":
            p4.append(np.nan)
            continue
        Dp = D
        tp = D.ptiers[r.tier]
        surf_p, ser_p, _ = run_surface(Dp, r.tier, tp, lambda *_: None, only_fams={r.family}, cache=caches[r.tier])
        rp = choose(surf_p, r.family)
        ok = bool(rp is not None and rp.check_mean > 0 and rp.check_t >= 1.645)
        p4.append(not ok)
        log(f"  P4 {r.tier} {r.family}: placebo library picks {rp.variant if rp is not None else None}, "
            f"check t {rp.check_t if rp is not None else np.nan:.2f}")
    C["P4"] = p4
    C["counts"] = C.P1 & C.P2 & (C.P3 == True) & (C.P4 != False)                                   # noqa: E712
    C.loc[C.family == "B1", "counts"] = False
    S.to_csv(os.path.join(out, "surface.csv.gz"), index=False)
    C.to_csv(os.path.join(out, "chosen.csv"), index=False)
    # K curve for M2 / M4 (best inner setting per K)
    kc = []
    for fam in ("M2", "M4"):
        for (tn, K), g in S[S.family == fam].groupby(["tier", "K"]):
            b = g.loc[g.inner_t.idxmax()]
            kc.append({"tier": tn, "family": fam, "K": int(K), "variant": b.variant, "inner_t": b.inner_t,
                       "check_t": b.check_t, "check_sharpe": b.check_sharpe})
    pd.DataFrame(kc).to_csv(os.path.join(out, "k_curve.csv"), index=False)
    # overall pick per tier
    picks = {}
    for tn, g in C[C.family != "B1"].groupby("tier"):
        b = g.loc[g.inner_t.idxmax()]
        picks[tn] = b[["family", "variant", "inner_t", "check_t", "check_sharpe", "counts"]].to_dict()
    # regime maps
    maps = {}
    for i, r in C.iterrows():
        if r.family in ("M2", "M3", "M4"):
            row = S[S.variant == r.variant].iloc[0]
            maps[f"{r.tier}|{r.family}"] = regime_map(D, D.tiers[r.tier], r.family, params_of(row), caches[r.tier])
    json.dump(maps, open(os.path.join(out, "regime_maps.json"), "w"), indent=1, default=str)
    # O2: swap the feature set of the chosen M2 / M4 / M5 settings
    o2 = []
    for i, r in C[C.family.isin(["M2", "M4", "M5"])].iterrows():
        row = S[S.variant == r.variant].iloc[0]
        p = params_of(row)
        k = int(row.k)
        Rs = {}
        for st in ("PRICE", "OPT", "ALL", "ALLC"):
            q = dict(p)
            q["set"] = st
            nm = pname(r.family, q, k)
            Rs[st] = series_all[r.tier].get(nm)
            if Rs[st] is None:
                Rs[st] = evaluate(D, D.tiers[r.tier], scores_for(D, D.tiers[r.tier], r.family, q, caches[r.tier]), (k,))[k]
        for a_, b_ in (("OPT", "PRICE"), ("ALL", "PRICE"), ("ALLC", "ALL")):
            d = (Rs[a_] - Rs[b_])[D.mask_check]
            o2.append({"tier": r.tier, "family": r.family, "base_variant": r.variant, "compare": f"{a_}-{b_}",
                       "sharpe_a": sharpe(Rs[a_][D.mask_check]), "sharpe_b": sharpe(Rs[b_][D.mask_check]),
                       "diff_mean": float(np.nanmean(d)), "diff_t": nw_t(d)})
    O2 = pd.DataFrame(o2)
    if len(O2):
        O2["p"] = 1 - norm.cdf(O2.diff_t.fillna(-9))
        O2["bh_pass"] = bh(O2.p.to_numpy())
        O2["counts"] = O2.bh_pass & (O2.diff_t >= 1.645)
    O2.to_csv(os.path.join(out, "O2_options_in_regimes.csv"), index=False)
    log("part A")
    A1 = part_A1(D)
    A2 = part_A2(D, nplac)
    A2.to_csv(os.path.join(out, "A2_regime_repeat.csv"), index=False)
    O3 = part_O3(D)
    O3.to_csv(os.path.join(out, "O3_options_timing.csv"), index=False)
    PD = part_D(D)
    PD.to_csv(os.path.join(out, "D_dip_filter.csv"), index=False)
    summary.update({
        "chosen": C.round(4).to_dict("records"), "overall_pick": picks, "A1": A1,
        "A2_pass": A2[A2.bh_pass][["tier", "feature", "rho", "p"]].to_dict("records"),
        "A2_top": A2.sort_values("p").head(10)[["tier", "feature", "rho", "null_p95", "p"]].round(4).to_dict("records"),
        "O2_counts": O2[O2.get("counts", pd.Series(dtype=bool)) == True].to_dict("records") if len(O2) else [],  # noqa: E712
        "O3_counts": O3[O3.counts].to_dict("records"),
        "O3_top": O3.sort_values("p").head(10)[["group", "feature", "factor", "check_mean", "check_t", "bh_pass"]].round(4).to_dict("records"),
        "D_counts": PD[PD.counts].to_dict("records"),
        "D_top": PD.sort_values("p").head(8)[["feature", "check_diff_mean", "check_diff_t", "check_kept_share", "bh_pass"]].round(4).to_dict("records"),
        "bic_k": bic_k(D), "n_counts": int(C.counts.sum())})
    json.dump(summary, open(os.path.join(out, "summary16.json"), "w"), indent=1, default=str)
    log(json.dumps({k: summary[k] for k in ("n_counts", "overall_pick", "A1")}, indent=1, default=str))
    return summary


# ================================================================================================ synthetic checks
def synthetic(kind, seed=7):
    rng = np.random.default_rng(seed)
    months = list(pd.period_range("2008-01", "2025-12", freq="M").astype(str))
    rows = list(pd.period_range("2007-01", "2025-12", freq="M").astype(str))
    T, N = len(months), len(rows)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    F = pd.DataFrame(rng.normal(size=(N, len(FEATS))), index=rows, columns=FEATS)
    S = 40
    R = rng.normal(0, 1, (T, S))
    xprev = np.array([x[rows.index(m) - 1] for m in months])
    if kind == "planted":                                                 # strategy 0 wins when x was high, 1 when low
        R[:, 0] += 0.8 * np.sign(xprev)
        R[:, 1] -= 0.8 * np.sign(xprev)
        F["RV21"] = x
    elif kind == "leak":                                                  # feature row m = sign of month m+1's return
        nxt = np.r_[R[1:, 0], 0.0]
        lk = np.zeros(N)
        for j, m in enumerate(months):
            lk[rows.index(m)] = np.sign(nxt[j])
        F["RV21"] = lk
    elif kind == "leak_shift":                                            # the same feature, shifted one month later
        lk = np.zeros(N)
        for j, m in enumerate(months):
            lk[rows.index(m)] = np.sign(R[j, 0])
        F["RV21"] = lk
    fac = Tier("FACTOR", R[:, :9], FACTORS)
    cfg = Tier("CFG", R, [f"s{i}" for i in range(S)])
    idx = pd.MultiIndex.from_tuples([(f"u{i // 2}", "c", 1) for i in range(S)], names=["unit", "cfg", "dir"])
    unit = unit_tier(cfg, idx)
    return Data(months, F, {"FACTOR": fac, "UNIT": unit, "CFG": cfg}, {"UNIT": unit, "CFG": cfg})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placebos", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)       # noqa: E731
    D = synthetic(a.synthetic) if a.synthetic else load(a.data)
    run_all(D, a.out, a.placebos, a.jobs, log)


if __name__ == "__main__":
    main()
