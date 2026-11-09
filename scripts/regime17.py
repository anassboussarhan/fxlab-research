"""Round 17 engine (docs/round17_preregistration.md): walk-forward regime methods at daily, weekly and monthly
decision frequencies over the FACTOR, BOOK-UNIT, BOOK-CFG and PAIR-UNIT tiers, with incremental computation.

Usage:
  python scripts/regime17.py --data DIR --tier BOOK-CFG --freq D --fams A --out DIR [--placebos 200] [--jobs 4]
  python scripts/regime17.py --combine DIR --data DIR --out DIR        (BH over all tests, timing tests, summary)
  python scripts/regime17.py --synthetic planted|leak|leak_shift|null --tier FACTOR --freq M --fams all --out DIR

Family groups: A = B1, M1, M1b, M6, M8; B = M2, M4, M5; C = M3; D = M9; all = every family.

Conventions (as round 16)
- The decision for period j is made at the close of period j-1. Feature row j = the state at the close of j.
- M2 / M3 / M5 / M9 are predictive: period j is labelled by row j-1.
- M4 is contemporaneous: smoothed posterior of row j in training; the decision uses the filtered posterior of row
  j-1 times the transition matrix.
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

FREQ = {
    "M": dict(ppy=12, nsc=36, act=12, cost=0.03, n0=12, minreg=6, long=60, L=(1, 3, 6, 12, 24, 36), lags=3,
              m9=(3, 12, 36)),
    "W": dict(ppy=52, nsc=104, act=26, cost=0.0625, n0=52, minreg=26, long=260, L=(1, 4, 13, 26, 52, 104), lags=4,
              m9=(4, 26, 104)),
    "D": dict(ppy=252, nsc=252, act=60, cost=0.137, n0=252, minreg=126, long=1260, L=(1, 5, 21, 63, 126, 252), lags=5,
              m9=(21, 126, 252)),
}
KGRID = {"FACTOR": (1, 2, 3), "BOOK-UNIT": (1, 5, 20, 100), "BOOK-CFG": (1, 10, 50, 200), "PAIR-UNIT": (1, 10, 50, 200)}
GROUPS = {"A": ("B1", "M1", "M1b", "M6", "M8"), "B": ("B1", "M2", "M4", "M5"), "C": ("B1", "M3"), "D": ("B1", "M9"),
          "all": ("B1", "M1", "M1b", "M2", "M3", "M4", "M5", "M6", "M8", "M9")}
PAIR_FAMS = ("B1", "M1", "M1b", "M3", "M6", "M8", "M9")
PRICE = ["RV5", "RV21", "RV_CHG", "RV252", "CORR21", "CORR63", "DISP5", "DISP21", "ER21", "ER63", "VR5", "AC_D", "AC_W",
         "TREND12", "REV12", "CARRY6", "CARRY_SD", "USD5", "USD63", "SPX5", "SPX63", "SPX_RV21", "GOLD63"]
INTRA = ["RV_5M", "JUMP", "AC_5M", "ASIA_SHARE"]
OPT = ["VIX", "VIX_TS", "VVIX", "SKEW", "GVZ", "OVX", "VRP_SPX", "DVIX5", "DVIX21"]
CME = ["EUR_IV", "EUR_RR25", "JPY_IV", "JPY_RR25", "FX_VRP"]
MACRO = ["US2Y_CHG63", "SLOPE", "NFCI", "EPU"]
FEATS = PRICE + INTRA + OPT + CME + MACRO
PAIRF = ["RV21", "RV_CHG", "ER63", "VR5", "AC_D", "AC_W", "CARRY", "TS12", "MA200Z", "RV_5M", "AC_5M"]
TF_CODE = {"MN": 0, "W1": 1, "D1": 2, "H4": 3, "H1": 4, "M30": 5, "M15": 6}
MIN_ROWS = 30
WIN = 5.0
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


def sharpe(x, ppy):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * math.sqrt(ppy)) if len(x) > 2 and x.std() > 0 else np.nan


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
def scale(R, nsc, act):
    T, S = R.shape
    sq = np.vstack([np.zeros(S), np.cumsum(np.asarray(R, np.float64) ** 2, axis=0)])
    ac = np.vstack([np.zeros(S), np.cumsum(R != 0, axis=0)])
    sig = np.full((T, S), np.nan, np.float32)
    for t in range(1, T):
        a = max(0, t - nsc)
        ms = (sq[t] - sq[a]) / (t - a)
        ok = ((ac[t] - ac[a]) >= act) & (ms > 0)
        sig[t, ok] = np.sqrt(ms[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (R / sig).astype(np.float32)


class Tier:
    def __init__(self, name, z, names, inst=None, desc=None):
        self.name, self.names = name, list(names)
        self.z = np.asarray(z, np.float32)
        self.zw = np.clip(self.z, -WIN, WIN)
        self.elig = np.isfinite(self.z)
        self.inst = np.zeros(len(self.names), int) if inst is None else np.asarray(inst, int)
        self.desc = desc                                                  # DataFrame of descriptors (M9)


def unit_average(z, keys):
    codes, uniq = pd.factorize(pd.Index(keys))
    T = z.shape[0]
    U = len(uniq)
    zs = np.zeros((T, U), np.float64)
    cn = np.zeros((T, U), np.float64)
    zf = np.nan_to_num(z)
    fin = np.isfinite(z)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=U))]
    for u in range(U):
        cols = order[bounds[u]: bounds[u + 1]]
        zs[:, u] = zf[:, cols].sum(1)
        cn[:, u] = fin[:, cols].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cn > 0, zs / cn, np.nan).astype(np.float32), list(uniq), codes


class Data:
    """Periods, the tier, its placebo tier and the aligned features for one (tier, frequency)."""

    def __init__(self, periods, ends, tier, ptier, feats: pd.DataFrame, freq, pairfeats=None, inst_names=None):
        self.periods, self.ends = list(periods), pd.DatetimeIndex(ends)
        self.T = len(self.periods)
        self.cfg = FREQ[freq]
        self.freq = freq
        yr = self.ends.year
        self.mask_inner = np.asarray((yr >= 2011) & (yr <= 2018))
        self.mask_check = np.asarray((yr >= 2019) & (yr <= 2025))
        self.j0 = int(np.argmax(self.mask_inner))
        self.tier, self.ptier = tier, ptier
        self.dropped = [c for c in FEATS if c not in feats or feats[c].notna().sum() < 24]
        self.feats = [c for c in FEATS if c not in self.dropped]
        self.sets = {"PRICE": [c for c in PRICE if c in self.feats], "INTRA": [c for c in INTRA if c in self.feats],
                     "OPT": [c for c in OPT if c in self.feats], "CME": [c for c in CME if c in self.feats],
                     "MACRO": [c for c in MACRO if c in self.feats]}
        self.sets["ALL"] = self.sets["PRICE"] + self.sets["INTRA"] + self.sets["OPT"] + self.sets["MACRO"]
        self.sets["ALLC"] = self.sets["ALL"] + self.sets["CME"]
        self.FV = feats.reindex(columns=self.feats).to_numpy(float)       # T x F, row j = state at close of j
        self.fcol = {c: i for i, c in enumerate(self.feats)}
        self.PF = pairfeats or {}                                         # name -> T x I (per instrument)
        self.inst_names = inst_names or ["BOOK"]
        ys = [self.j0] + [j for j in range(self.j0 + 1, self.T) if self.ends[j].year != self.ends[j - 1].year]
        self.refits = [(s, ys[i + 1] if i + 1 < len(ys) else self.T) for i, s in enumerate(ys)]

    def rolled(self, offset):
        d = object.__new__(Data)
        d.__dict__.update(self.__dict__)
        d.FV = np.roll(self.FV, offset, axis=0)
        d.PF = {k: np.roll(v, offset, axis=0) for k, v in self.PF.items()}
        return d


def period_index(days: pd.DatetimeIndex, freq):
    if freq == "D":
        return np.array(days.strftime("%Y-%m-%d"))
    if freq == "W":
        iso = days.isocalendar()
        return (iso.year.astype(int) * 100 + iso.week.astype(int)).astype(str).to_numpy()
    return np.array(days.strftime("%Y-%m"))


def agg(mat_days_by_cols, keys):
    """mat: (S x days) -> (periods x S) sums, keys contiguous."""
    keys = np.asarray(keys).astype(str)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    out = np.add.reduceat(np.asarray(mat_days_by_cols, np.float64), starts, axis=1).T
    return keys[starts], out, starts


def asof(df, ends, tol=7):
    return df.reindex(pd.DatetimeIndex(ends), method="ffill", tolerance=pd.Timedelta(days=tol))


def load(data_dir, tier_name, freq):
    days = pd.DatetimeIndex(np.load(os.path.join(data_dir, "days.npy")))
    keys = period_index(days, freq)
    feats_d = pd.read_parquet(os.path.join(data_dir, "features_daily.parquet"))
    feats_d.index = pd.DatetimeIndex(feats_d.index)
    feats_d = feats_d.sort_index()
    cfgf = FREQ[freq]
    if tier_name == "PAIR-UNIT":
        if freq == "D":
            raise ValueError("PAIR-UNIT runs at W and M only")
        per = np.load(os.path.join(data_dir, f"pair_{freq}_periods.npy")).astype(str)
        Xr = np.load(os.path.join(data_dir, f"pair_{freq}_real.npy")).T
        Xp = np.load(os.path.join(data_dir, f"pair_{freq}_placebo.npy")).T
        mr = pd.read_parquet(os.path.join(data_dir, "pair_real_meta.parquet"))
        mp = pd.read_parquet(os.path.join(data_dir, "pair_placebo_meta.parquet"))
        insts = sorted(set(mr.symbol) | set(mp.symbol))
        ii = {s: i for i, s in enumerate(insts)}
        tier = Tier("PAIR-UNIT", scale(Xr, cfgf["nsc"], cfgf["act"]),
                    [f"{s}|{u}|{d:+d}" for s, u, d in zip(mr.symbol, mr.unit, mr.dir)], mr.symbol.map(ii), desc=mr)
        ptier = Tier("PAIR-UNIT", scale(Xp, cfgf["nsc"], cfgf["act"]),
                     [f"{s}|{u}|{d:+d}" for s, u, d in zip(mp.symbol, mp.unit, mp.dir)], mp.symbol.map(ii), desc=mp)
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        ends_idx = np.r_[starts[1:] - 1, len(days) - 1]
        assert list(keys[starts]) == list(per), "period keys differ between the matrices and the calendar"
        pf = {}
        for f in PAIRF:
            p = os.path.join(data_dir, f"pairfeat_{f}.parquet")
            if os.path.exists(p):
                W = pd.read_parquet(p)
                W.index = pd.DatetimeIndex(W.index)
                W = asof(W.sort_index(), days)
                pf[f] = W.reindex(columns=insts).to_numpy(float)[ends_idx]
        ends = days[ends_idx]
        return Data(per, ends, tier, ptier, asof(feats_d, ends), freq, pf, insts)
    if tier_name == "FACTOR":
        F = pd.read_parquet(os.path.join(data_dir, "factors_daily.parquet"))
        F.index = pd.DatetimeIndex(F.index)
        F = F.sort_index().fillna(0.0)[FACTORS]                           # the factors' own (FX panel) calendar
        fdays = pd.DatetimeIndex(F.index)
        per, X, starts = agg(F.to_numpy().T, period_index(fdays, freq))
        ends = fdays[np.r_[starts[1:] - 1, len(fdays) - 1]]
        tier = Tier("FACTOR", scale(X, cfgf["nsc"], cfgf["act"]), FACTORS,
                    desc=pd.DataFrame({"factor": FACTORS}))
        return Data(per, ends, tier, None, asof(feats_d, ends), freq)
    Br = np.load(os.path.join(data_dir, "book_real.npy"))
    Bp = np.load(os.path.join(data_dir, "book_placebo.npy"))
    mr = pd.read_parquet(os.path.join(data_dir, "book_real_meta.parquet"))
    mp = pd.read_parquet(os.path.join(data_dir, "book_placebo_meta.parquet"))
    per, Xr, starts = agg(Br, keys)
    _, Xp, _ = agg(Bp, keys)
    del Br, Bp
    ends = days[np.r_[starts[1:] - 1, len(days) - 1]]
    feats_e = asof(feats_d, ends)
    zr, zp = scale(Xr, cfgf["nsc"], cfgf["act"]), scale(Xp, cfgf["nsc"], cfgf["act"])
    del Xr, Xp
    if tier_name == "BOOK-CFG":
        tier = Tier("BOOK-CFG", zr, [f"{u}|{c}|{d:+d}" for u, c, d in zip(mr.unit, mr.cfg, mr.dir)], desc=mr)
        ptier = Tier("BOOK-CFG", zp, [f"{u}|{c}|{d:+d}" for u, c, d in zip(mp.unit, mp.cfg, mp.dir)], desc=mp)
    else:
        def unitize(z, m):
            keys_ = list(zip(m.unit, m.dir, m.engine))
            zu, uniq, codes = unit_average(z, keys_)
            dm = m.groupby(codes).first().reset_index(drop=True)
            return Tier("BOOK-UNIT", zu, [f"{u}|{d:+d}" for u, d, _ in uniq], desc=dm)
        tier, ptier = unitize(zr, mr), unitize(zp, mp)
    return Data(per, ends, tier, ptier, feats_e, freq)


# ================================================================================================ scores
def window_start(j, window, D):
    return 0 if window == "exp" else max(0, j - D.cfg["long"])


def static_scores(D, tier, window, zw=None):
    zw = tier.zw if zw is None else zw
    z0 = np.nan_to_num(zw).astype(np.float64)
    fin = np.isfinite(zw)
    cs = np.vstack([np.zeros(z0.shape[1]), np.cumsum(z0, 0)])
    cn = np.vstack([np.zeros(z0.shape[1]), np.cumsum(fin, 0)])
    Sc = np.full(zw.shape, np.nan, np.float32)
    mr = D.cfg["minreg"]
    for j in range(D.j0, D.T):
        a = window_start(j, window, D)
        n = cn[j] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            Sc[j] = np.where(n >= mr, (cs[j] - cs[a]) / n, np.nan)
    return Sc


def mom_scores(D, tier, L, zw=None):
    zw = tier.zw if zw is None else zw
    z0 = np.nan_to_num(zw).astype(np.float64)
    fin = np.isfinite(zw)
    cs = np.vstack([np.zeros(z0.shape[1]), np.cumsum(z0, 0)])
    cn = np.vstack([np.zeros(z0.shape[1]), np.cumsum(fin, 0)])
    Sc = np.full(zw.shape, np.nan, np.float32)
    need = max(1, L // 2)
    for j in range(D.j0, D.T):
        a = max(0, j - L)
        n = cn[j] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            Sc[j] = np.where(n >= need, (cs[j] - cs[a]) / n, np.nan)
    return Sc


def hedge_weights(D, tier, eta, hl, zw=None):
    """M6: weights proportional to exp(eta x discounted sum of z) over eligible strategies plus a cash expert."""
    zw = tier.zw if zw is None else zw
    delta = 1.0 if hl is None else 0.5 ** (1.0 / (hl * D.cfg["ppy"]))
    T, S = zw.shape
    Dsum = np.zeros(S)
    W = np.zeros((T, S), np.float32)
    z0 = np.nan_to_num(zw)
    for j in range(1, T):
        Dsum = delta * Dsum + z0[j - 1]
        if j < D.j0:
            continue
        ok = tier.elig[j]
        if not ok.any():
            continue
        x = eta * Dsum[ok]
        m = max(x.max(), 0.0)
        e = np.exp(x - m)
        tot = e.sum() + math.exp(-m)                                    # the cash expert has a score of 0
        W[j, ok] = e / tot
    return W


def cusum_scores(D, tier, h, zw=None, kappa=0.5):
    """M8: two-sided CUSUM per strategy on z; after an alarm its mean restarts; score = mean since the last alarm."""
    zw = tier.zw if zw is None else zw
    T, S = zw.shape
    gp, gm = np.zeros(S), np.zeros(S)
    sm, cnt = np.zeros(S), np.zeros(S)
    Sc = np.full((T, S), np.nan, np.float32)
    need = D.cfg["minreg"]
    for j in range(1, T):
        x = zw[j - 1]
        ok = np.isfinite(x)
        mu = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)
        dev = np.where(ok, x - mu, 0.0)
        gp = np.where(ok, np.maximum(0.0, gp + dev - kappa), gp)
        gm = np.where(ok, np.maximum(0.0, gm - dev - kappa), gm)
        alarm = (gp > h) | (gm > h)
        sm = np.where(alarm, 0.0, sm + np.where(ok, x, 0.0))
        cnt = np.where(alarm, 0.0, cnt + ok)
        gp = np.where(alarm, 0.0, gp)
        gm = np.where(alarm, 0.0, gm)
        if j >= D.j0:
            with np.errstate(invalid="ignore", divide="ignore"):
                Sc[j] = np.where(cnt >= need, sm / np.maximum(cnt, 1), np.nan)
    return Sc


def conditional_scores(D, tier, G, P, window, n0, static, s, e):
    """Decisions j in [s, e). G: T x K x I training weights of each period (by instrument; I = 1 for market-wide
    labels); P: T x K x I decision probabilities (NaN = fall back to static). Incremental sums over the window."""
    zw = tier.zw
    T, S = zw.shape
    K = G.shape[1]
    inst = tier.inst if G.shape[2] > 1 else np.zeros(S, int)
    mr = D.cfg["minreg"]
    Sc = static[s:e].copy()
    z0 = np.nan_to_num(zw).astype(np.float64)
    fin = np.isfinite(zw).astype(np.float64)
    a = window_start(s, window, D)
    SZ = np.zeros((K, S))
    SN = np.zeros((K, S))
    for c0 in range(a, s, 256):                                           # initial sums in row chunks
        c1 = min(s, c0 + 256)
        g = G[c0:c1][:, :, inst]                                          # rows x K x S
        SZ += np.einsum("rks,rs->ks", g, z0[c0:c1])
        SN += np.einsum("rks,rs->ks", g, fin[c0:c1])
    a_prev = a
    for j in range(s, e):
        if j > s:
            g = G[j - 1][:, inst]
            SZ += g * z0[j - 1]
            SN += g * fin[j - 1]
            a_new = window_start(j, window, D)
            while a_prev < a_new:
                g = G[a_prev][:, inst]
                SZ -= g * z0[a_prev]
                SN -= g * fin[a_prev]
                a_prev += 1
        p = P[j][:, inst]                                                 # K x S
        if not np.all(np.isfinite(p)):
            bad = ~np.all(np.isfinite(p), axis=0)
        else:
            bad = None
        m_all = static[j].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mk = SZ / SN
            sh = (SN * mk + n0 * m_all) / (SN + n0)
        sh = np.where(SN >= mr, sh, m_all)
        sc = np.nansum(np.nan_to_num(p) * sh, axis=0)
        sc = np.where(np.isfinite(m_all), sc, np.nan)
        if bad is not None:
            sc = np.where(bad, m_all, sc)
        Sc[j - s] = sc
    return Sc


def regime_scores(D, tier, periods, window, n0, static):
    Sc = static.copy()
    for s, e, G, P in periods:
        if G is None:
            continue
        Sc[s:e] = conditional_scores(D, tier, G, P, window, n0, static, s, e)
    return Sc


# ------------------------------------------------------------------------------------------------ label makers
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


def onehot_prev(lab, K):
    """lab: T labels (-1 missing) of rows; returns G (period j weighted by row j-1) and P (decision j from row j-1)."""
    T = len(lab)
    oh = np.full((T, K), np.nan)
    ok = lab >= 0
    oh[ok] = np.eye(K)[lab[ok]]
    prev = np.full((T, K), np.nan)
    prev[1:] = oh[:-1]
    G = np.where(np.isfinite(prev), prev, 0.0)
    return G[:, :, None], prev[:, :, None]


def cluster_periods(D, X, algo, K, seed=0):
    out = []
    for s, e in D.refits:
        Xtr = X[:s]
        ok = np.all(np.isfinite(Xtr), axis=1)
        if ok.sum() < MIN_ROWS:
            out.append((s, e, None, None))
            continue
        tr = fit_transform(Xtr[ok])
        Ztr = tr(Xtr[ok])
        if algo == "kmeans":
            from sklearn.cluster import KMeans
            mdl = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(Ztr)
        else:
            from sklearn.mixture import GaussianMixture
            mdl = GaussianMixture(n_components=K, covariance_type="diag", n_init=5, random_state=seed, reg_covar=1e-4).fit(Ztr)
        Z = tr(X[:e])
        lab = np.full(D.T, -1)
        okz = np.all(np.isfinite(Z), axis=1)
        lab[np.flatnonzero(okz)] = mdl.predict(Z[okz])
        G, P = onehot_prev(lab, K)
        out.append((s, e, G, P))
    return out


def split_periods(D, x, K):
    """x: T (market-wide) or T x I (per instrument). Quantile cut-points from the rows known at each refit."""
    xx = x if x.ndim == 2 else x[:, None]
    I = xx.shape[1]
    out = []
    for s, e in D.refits:
        G = np.zeros((D.T, K, I))
        P = np.full((D.T, K, I), np.nan)
        any_ok = False
        for i in range(I):
            xt = xx[:s, i]
            xt = xt[np.isfinite(xt)]
            if len(xt) < MIN_ROWS:
                continue
            any_ok = True
            cuts = np.quantile(xt, np.arange(1, K) / K)
            col = xx[:e, i]
            lab = np.full(D.T, -1)
            fin = np.isfinite(col)
            lab[np.flatnonzero(fin)] = np.digitize(col[fin], cuts)
            g, p = onehot_prev(lab, K)
            G[:, :, i], P[:, :, i] = g[:, :, 0], p[:, :, 0]
        out.append((s, e, G, P) if any_ok else (s, e, None, None))
    return out


def hmm_forward(model, Z):
    """Filtered state probabilities of a diagonal Gaussian HMM (rows with NaN are skipped: their filter is the
    one-step prediction)."""
    K = model.n_components
    A = model.transmat_
    mu = model.means_
    var = np.array([np.diag(c) if c.ndim == 2 else c for c in model.covars_])
    out = np.full((len(Z), K), np.nan)
    a = model.startprob_.copy()
    for t in range(len(Z)):
        pred = a @ A if t else a
        if np.all(np.isfinite(Z[t])):
            ll = -0.5 * (np.log(2 * np.pi * var) + (Z[t] - mu) ** 2 / var).sum(1)
            w = pred * np.exp(ll - ll.max())
            a = w / w.sum() if w.sum() > 0 else pred
        else:
            a = pred
        out[t] = a
    return out


def hmm_periods(D, X, K, seed=0):
    from hmmlearn.hmm import GaussianHMM
    out = []
    for s, e in D.refits:
        Xtr = X[:s]
        ok = np.all(np.isfinite(Xtr), axis=1)
        if ok.sum() < MIN_ROWS:
            out.append((s, e, None, None))
            continue
        first = int(np.argmax(ok))
        blk = np.all(np.isfinite(Xtr[first:]), axis=1)
        if not blk.all():
            first = first + int(np.flatnonzero(~blk).max()) + 1
        if s - first < MIN_ROWS:
            out.append((s, e, None, None))
            continue
        tr = fit_transform(Xtr[first:])
        Ztr = tr(Xtr[first:])
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
            out.append((s, e, None, None))
            continue
        Z = tr(X[:e])
        gam = np.full((D.T, K), np.nan)
        gam[first:s] = best.predict_proba(Ztr)
        filt = np.full((D.T, K), np.nan)
        filt[first:e] = hmm_forward(best, Z[first:e])
        gam[s:e] = filt[s:e]                                               # later rows: filtered
        G = np.where(np.isfinite(gam), gam, 0.0)[:, :, None]
        P = np.full((D.T, K), np.nan)
        P[1:e] = filt[0:e - 1] @ best.transmat_
        out.append((s, e, G, P[:, :, None]))
    return out


def ret_matrix(D):
    """RET: the BOOK-UNIT-like returns of the tier itself (rows = the period), for the HMM."""
    U = D.tier.zw
    if U.shape[1] > 3000:                                                 # keep the PCA tractable: a fixed subset
        idx = np.random.default_rng(0).choice(U.shape[1], 3000, replace=False)
        U = U[:, np.sort(idx)]
    X = np.where(np.isfinite(U), U, 0.0).astype(np.float64)
    X[np.isfinite(U).mean(1) < 0.5] = np.nan
    return X


def set_matrix(D, st):
    if st == "RET":
        return ret_matrix(D)
    return D.FV[:, [D.fcol[c] for c in D.sets[st]]]


def ridge_scores(D, tier, st, alpha, static):
    X = set_matrix(D, st)
    Sc = static.copy()
    zw = tier.zw
    for s, e in D.refits:
        rows = np.arange(1, s)
        Xm = X[rows - 1]
        ok = np.all(np.isfinite(Xm), axis=1)
        rows, Xm = rows[ok], Xm[ok]
        if len(rows) < MIN_ROWS:
            continue
        mu, sd = Xm.mean(0), Xm.std(0)
        sd[sd == 0] = 1.0
        Xs = (Xm - mu) / sd
        Y = zw[rows].astype(np.float64)
        fin = np.isfinite(Y)
        with np.errstate(invalid="ignore", divide="ignore"):
            ym = np.nansum(Y, 0) / fin.sum(0)
        Yc = np.where(fin, Y - ym, 0.0)
        beta = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ Yc)
        need = D.cfg["minreg"]
        for j in range(s, e):
            x = X[j - 1]
            if not np.all(np.isfinite(x)):
                continue
            pred = ym + ((x - mu) / sd) @ beta
            Sc[j] = np.where(np.isfinite(static[j]) & (fin.sum(0) >= need), pred, np.nan)
    return Sc


# ------------------------------------------------------------------------------------------------ M9 pooled boosting
def trailing_stats(D, tier, zw=None):
    zw = tier.zw if zw is None else zw
    z0 = np.nan_to_num(zw).astype(np.float64)
    fin = np.isfinite(zw)
    cs = np.vstack([np.zeros(z0.shape[1]), np.cumsum(z0, 0)])
    cq = np.vstack([np.zeros(z0.shape[1]), np.cumsum(z0 * z0, 0)])
    cn = np.vstack([np.zeros(z0.shape[1]), np.cumsum(fin, 0)])
    L1, L2, L3 = D.cfg["m9"]
    T, S = zw.shape
    out = {}
    for nm, L in (("m1", L1), ("m2", L2), ("m3", L3)):
        a = np.maximum(np.arange(T) - L, 0)
        n = cn[np.arange(T)] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nm] = np.where(n >= max(1, L // 2), (cs[np.arange(T)] - cs[a]) / n, np.nan).astype(np.float32)
    def rms(L):
        a = np.maximum(np.arange(T) - L, 0)
        n = cn[np.arange(T)] - cn[a]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= max(2, L // 2), np.sqrt((cq[np.arange(T)] - cq[a]) / n), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["volr"] = (rms(L1) / rms(L3)).astype(np.float32)
    C = cs[:T]                                                            # cumulative z up to (not incl.) row j
    roll_max = pd.DataFrame(C).rolling(L3, min_periods=1).max().to_numpy()
    out["dd"] = (C - roll_max).astype(np.float32)
    return out                                                            # value at row j uses rows < j


def m9_scores(D, tier, st, static, seed=0, stats=None, FV=None, PF=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    stats = trailing_stats(D, tier) if stats is None else stats
    FV = D.FV if FV is None else FV
    PF = D.PF if PF is None else PF
    cols = [D.fcol[c] for c in D.sets[st]]
    desc = tier.desc
    S = tier.zw.shape[1]
    tfc = np.zeros(S)
    dirc = np.zeros(S)
    famc = np.zeros(S)
    engc = np.zeros(S)
    if desc is not None and "unit" in desc:
        tfc = desc.unit.str.split("|").str[-1].map(TF_CODE).fillna(-1).to_numpy(float)
        dirc = desc["dir"].to_numpy(float)
        famc = pd.factorize(desc["family"])[0].astype(float) if "family" in desc else famc
        engc = (desc["engine"] == "I").to_numpy(float) if "engine" in desc else engc
    elif desc is not None and "factor" in desc:
        famc = np.arange(S, dtype=float)
    pnames = [k for k in PAIRF if k in PF] if tier.name == "PAIR-UNIT" else []

    def rows_for(j, idx):
        f = FV[j - 1, cols]
        parts = [np.column_stack([tfc[idx], dirc[idx], famc[idx], engc[idx]] +
                                 [stats[k][j, idx] for k in ("m1", "m2", "m3", "volr", "dd")]),
                 np.repeat(f[None, :], len(idx), 0)]
        if pnames:
            parts.append(np.column_stack([PF[k][j - 1, tier.inst[idx]] for k in pnames]))
        return np.hstack(parts)

    Sc = static.copy()
    rng = np.random.default_rng(seed)
    for s, e in D.refits:
        js = np.arange(1, s)
        # sample (period, strategy) pairs with a finite target
        fin = np.isfinite(tier.zw[js])
        tot = int(fin.sum())
        if tot < 1000:
            continue
        take = min(400_000, tot)
        flat = np.flatnonzero(fin.ravel())
        pick = np.sort(rng.choice(flat, take, replace=False))
        jj, ii = js[pick // S], pick % S
        Xtr = np.vstack([rows_for(j, ii[jj == j]) for j in np.unique(jj)])
        ytr = np.concatenate([tier.zw[j, ii[jj == j]] for j in np.unique(jj)])
        ok = np.isfinite(ytr)
        mdl = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                            categorical_features=[2], random_state=seed)
        Xtr[:, 2] = np.where(np.isfinite(Xtr[:, 2]) & (Xtr[:, 2] >= 0), Xtr[:, 2], 0)
        mdl.fit(Xtr[ok], ytr[ok])
        for j in range(s, e):
            idx = np.flatnonzero(np.isfinite(static[j]))
            if not len(idx):
                continue
            Xq = rows_for(j, idx)
            Xq[:, 2] = np.where(np.isfinite(Xq[:, 2]) & (Xq[:, 2] >= 0), Xq[:, 2], 0)
            row = np.full(S, np.nan, np.float32)
            row[idx] = mdl.predict(Xq)
            Sc[j] = row
    return Sc


# ================================================================================================ allocation
def evaluate(D, tier, scores, ks, mode="topk"):
    T, S = tier.z.shape
    out = {}
    cost = D.cfg["cost"]
    zf = np.nan_to_num(tier.z)
    for k in (ks if mode == "topk" else (None,)):
        R = np.full(T, np.nan)
        wprev = np.zeros(S, np.float32)
        for j in range(D.j0, T):
            s = scores[j]
            w = np.zeros(S, np.float32)
            if mode == "weights":
                w = np.where(tier.elig[j], np.nan_to_num(s), 0.0).astype(np.float32)
            else:
                ok = np.flatnonzero(np.isfinite(s) & tier.elig[j] & (s > 0))
                if mode == "topk":
                    if len(ok) > k:
                        part = np.argpartition(-s[ok], k - 1)[:k]
                        ok = ok[part]
                    w[ok] = 1.0 / k
                elif len(ok):
                    w[ok] = 1.0 / len(ok)
            R[j] = float(zf[j] @ w) - cost * float(np.abs(w - wprev).sum())
            wprev = w
        out[k] = R
    return out


# ================================================================================================ variants
def variants(D, fam):
    c = D.cfg
    tn = D.tier.name
    if fam == "B1":
        return [{"window": w} for w in ("exp", "long")]
    if fam == "M1":
        return [{"L": L} for L in c["L"]]
    if fam == "M1b":
        return [{"L": c["L"][i]} for i in (1, 3, 4)]
    if fam == "M2":
        return [{"algo": a, "K": K, "set": s, "n0": n} for a, K, s, n in itertools.product(
            ("kmeans", "gmm"), (2, 3, 4, 6, 8), ("PRICE", "INTRA", "OPT", "ALL", "ALLC"), (0, c["n0"]))]
    if fam == "M3":
        fl = list(D.feats) + ([f"PAIR:{k}" for k in PAIRF if k in D.PF] if tn == "PAIR-UNIT" else [])
        return [{"f": f, "K": K, "n0": n, "window": w} for f, K, n, w in itertools.product(fl, (2, 3), (0, c["n0"]),
                                                                                           ("exp", "long"))]
    if fam == "M4":
        return [{"K": K, "set": s, "n0": n} for K, s, n in itertools.product((2, 3, 4), ("PRICE", "INTRA", "OPT", "ALL",
                                                                                         "ALLC", "RET"), (0, c["n0"]))]
    if fam == "M5":
        return [{"set": s, "alpha": a} for s, a in itertools.product(("PRICE", "INTRA", "OPT", "ALL", "ALLC"),
                                                                     (10, 100, 1000))]
    if fam == "M6":
        return [{"eta": e, "hl": h} for e, h in itertools.product((0.1, 0.3, 1.0), (None, 3, 1))]
    if fam == "M8":
        return [{"h": h} for h in (4, 8)]
    if fam == "M9":
        return [{"set": s} for s in ("PRICE", "ALLC")]
    raise ValueError(fam)


class Cache:
    def __init__(self):
        self.lab = {}
        self.static = {}
        self.stats = None


def scores_for(D, tier, fam, p, cache, zw=None, FV=None):
    """Score (or weight) matrix of one setting; zw / FV override the trailing-performance input (placebos)."""
    if fam == "B0":
        return np.where(tier.elig, 1.0, np.nan).astype(np.float32)
    if fam == "B1":
        return static_scores(D, tier, p["window"], zw)
    if fam in ("M1", "M1b"):
        return mom_scores(D, tier, p["L"], zw)
    if fam == "M6":
        return hedge_weights(D, tier, p["eta"], p["hl"], zw)
    if fam == "M8":
        return cusum_scores(D, tier, p["h"], zw)
    win = p.get("window", "exp")
    key_s = (id(tier), win)
    if key_s not in cache.static:
        cache.static[key_s] = static_scores(D, tier, win)
    static = cache.static[key_s]
    if fam == "M9":
        stats = trailing_stats(D, tier, zw) if zw is not None else (cache.stats or trailing_stats(D, tier))
        if zw is None:
            cache.stats = stats
        return m9_scores(D, tier, p["set"], static, stats=stats)
    if fam == "M2":
        key = ("M2", p["algo"], p["K"], p["set"])
        if key not in cache.lab:
            cache.lab[key] = cluster_periods(D, set_matrix(D, p["set"]), p["algo"], p["K"])
        return regime_scores(D, tier, cache.lab[key], win, p["n0"], static)
    if fam == "M3":
        key = ("M3", p["f"], p["K"])
        if key not in cache.lab:
            f = p["f"]
            x = D.PF[f[5:]] if f.startswith("PAIR:") else D.FV[:, D.fcol[f]]
            cache.lab[key] = split_periods(D, x, p["K"])
        return regime_scores(D, tier, cache.lab[key], win, p["n0"], static)
    if fam == "M4":
        key = ("M4", p["K"], p["set"])
        if key not in cache.lab:
            cache.lab[key] = hmm_periods(D, set_matrix(D, p["set"]), p["K"])
        return regime_scores(D, tier, cache.lab[key], "exp", p["n0"], static)
    if fam == "M5":
        return ridge_scores(D, tier, p["set"], p["alpha"], static)
    raise ValueError(fam)


def mode_of(fam):
    return "weights" if fam == "M6" else ("ew" if fam in ("B0", "M1b") else "topk")


def pname(fam, p, k=None):
    s = fam + ":" + ",".join(f"{a}={b}" for a, b in p.items())
    return s + (f",k={k}" if k is not None else "")


def stats_of(D, R):
    a, b = R[D.mask_inner], R[D.mask_check]
    L, ppy = D.cfg["lags"], D.cfg["ppy"]
    return {"inner_mean": float(np.nanmean(a)), "inner_t": nw_t(a, L), "inner_sharpe": sharpe(a, ppy),
            "check_mean": float(np.nanmean(b)), "check_t": nw_t(b, L), "check_sharpe": sharpe(b, ppy)}


def run_family(D, tier, fam, cache, log=print):
    rows, series = [], {}
    ks = KGRID[tier.name]
    t0 = time.time()
    V = variants(D, fam)
    for n, p in enumerate(V):
        Sc = scores_for(D, tier, fam, p, cache)
        md = mode_of(fam)
        for k, R in evaluate(D, tier, Sc, ks, md).items():
            nm = pname(fam, p, k)
            series[nm] = R
            rows.append({"family": fam, "variant": nm, "params": json.dumps(p), "k": k, **stats_of(D, R)})
        if n % 40 == 0:
            log(f"    {fam} {n + 1}/{len(V)} {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), series


def choose(surf):
    s = surf[np.isfinite(surf.inner_t)]
    return None if s.empty else s.loc[s.inner_t.idxmax()]


def offsets(D, n, key):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    lo = 2 * D.cfg["ppy"]
    return rng.integers(lo, max(lo + 1, D.T - lo), n)


def placebo_sharpes(D, tier, fam, p, k, offs, jobs):
    md = mode_of(fam)

    def one(o):
        if fam in ("M1", "M1b", "M6", "M8"):
            Sc = scores_for(D, tier, fam, p, Cache(), zw=np.roll(tier.zw, o, axis=0))
        elif fam == "M9":
            Dp = D.rolled(o)
            Sc = scores_for(Dp, tier, fam, p, Cache(), zw=np.roll(tier.zw, o, axis=0))
        elif fam == "M4" and p.get("set") == "RET":
            static = static_scores(D, tier, "exp")
            per = hmm_periods(D, np.roll(ret_matrix(D), o, axis=0), p["K"])
            Sc = regime_scores(D, tier, per, "exp", p["n0"], static)
        else:
            Sc = scores_for(D.rolled(o), tier, fam, p, Cache())
        R = list(evaluate(D, tier, Sc, (k,) if md == "topk" else (None,), md).values())[0]
        return sharpe(R[D.mask_check], D.cfg["ppy"])
    if jobs > 1:
        from joblib import Parallel, delayed
        chunks = np.array_split(np.asarray(offs), jobs)
        res = Parallel(n_jobs=jobs)(delayed(lambda c: [one(o) for o in c])(c) for c in chunks)
        return np.array([x for c in res for x in c])
    return np.array([one(o) for o in offs])


def run_job(D, fams, out, nplac=200, jobs=1, log=print, tag=""):
    os.makedirs(out, exist_ok=True)
    tier = D.tier
    cache = Cache()
    surfs, ser = [], {}
    fams = [f for f in fams if not (tier.name == "PAIR-UNIT" and f not in PAIR_FAMS)]
    if tier.name == "PAIR-UNIT" or tier.name == "FACTOR":
        fams = [f for f in fams if not (tier.name == "PAIR-UNIT" and f in ("M2", "M4", "M5"))]
    log(f"{tag}: tier {tier.name} {len(tier.names)} strategies, freq {D.freq}, periods {D.T}, families {fams}")
    for fam in fams:
        t0 = time.time()
        s_, r_ = run_family(D, tier, fam, cache, log)
        surfs.append(s_)
        ser.update(r_)
        log(f"  {fam}: {len(s_)} settings, {time.time() - t0:.0f}s")
    S = pd.concat(surfs, ignore_index=True)
    S["tier"], S["freq"] = tier.name, D.freq
    b1 = choose(S[S.family == "B1"])
    Rb1 = ser[b1.variant]
    rows = []
    for fam in fams:
        if fam == "B1":
            continue
        r = choose(S[S.family == fam])
        if r is None:
            continue
        R = ser[r.variant]
        diff = (R - Rb1)[D.mask_check]
        rec = {"tier": tier.name, "freq": D.freq, "family": fam, "variant": r.variant, "params": r.params,
               "k": r.k, **{c: r[c] for c in ("inner_mean", "inner_t", "inner_sharpe", "check_mean", "check_t",
                                             "check_sharpe")},
               "b1": b1.variant, "b1_check_t": b1.check_t, "b1_check_sharpe": b1.check_sharpe,
               "diff_check_mean": float(np.nanmean(diff)), "diff_check_t": nw_t(diff, D.cfg["lags"])}
        pre = rec["check_mean"] > 0 and np.isfinite(rec["check_t"]) and rec["check_t"] >= 1.645
        rec["p1_pre"] = bool(pre)
        rec["placebo_p95"], rec["P3"], rec["P4"] = np.nan, None, None
        if pre:                                                            # P3 / P4 matter only when P1 can hold
            p = json.loads(r.params)
            k = int(r.k) if pd.notna(r.k) else None
            t1 = time.time()
            sh = placebo_sharpes(D, tier, fam, p, k, offsets(D, nplac, f"P3|{tag}|{r.variant}"), jobs)
            sh = sh[np.isfinite(sh)]
            q95 = float(np.quantile(sh, 0.95)) if len(sh) else np.nan
            rec["placebo_p95"], rec["P3"] = q95, bool(rec["check_sharpe"] >= q95)
            log(f"  P3 {fam}: real {rec['check_sharpe']:.2f} vs p95 {q95:.2f} ({time.time() - t1:.0f}s)")
            if D.ptier is not None:
                Dp = object.__new__(Data)
                Dp.__dict__.update(D.__dict__)
                Dp.tier = D.ptier
                sp, _ = run_family(Dp, D.ptier, fam, Cache(), lambda *_: None)
                rp = choose(sp)
                bad = bool(rp is not None and rp.check_mean > 0 and rp.check_t >= 1.645)
                rec["P4"] = not bad
                rec["placebo_lib_check_t"] = float(rp.check_t) if rp is not None else np.nan
                log(f"  P4 {fam}: placebo library check t {rec['placebo_lib_check_t']:.2f}")
        rows.append(rec)
    C = pd.DataFrame(rows)
    S.to_csv(os.path.join(out, f"surface_{tag}.csv.gz"), index=False)
    C.to_csv(os.path.join(out, f"chosen_{tag}.csv"), index=False)
    keep = {nm: ser[nm] for nm in set([b1.variant] + list(C.variant))}
    np.savez_compressed(os.path.join(out, f"series_{tag}.npz"), names=np.array(list(keep)),
                        R=np.vstack([keep[n] for n in keep]), ends=np.array(D.ends.strftime("%Y-%m-%d")))
    log(f"{tag}: done")
    return S, C


# ================================================================================================ timing (long-only)
def timing_tests(data_dir):
    rows = []
    for freq in ("D", "W", "M"):
        D = load(data_dir, "FACTOR", freq)
        fac = D.tier
        fit = np.asarray((D.ends.year >= 2008) & (D.ends.year <= 2018))
        for f in D.sets["OPT"] + D.sets["CME"] + D.sets["INTRA"]:
            x = D.FV[:, D.fcol[f]]
            xp = np.r_[np.nan, x[:-1]]
            v = xp[fit & np.isfinite(xp)]
            if len(v) < MIN_ROWS:
                continue
            xs = (xp - v.mean()) / v.std()
            for i, fn in enumerate(fac.names):
                y = fac.zw[:, i]
                ok = fit & np.isfinite(xs) & np.isfinite(y)
                if ok.sum() < MIN_ROWS:
                    continue
                b = np.polyfit(xs[ok], y[ok], 1)[0]
                w = np.maximum(0.0, np.sign(b) * np.clip(xs, -2, 2))
                u = w * np.nan_to_num(fac.z[:, i])
                hold = np.nan_to_num(fac.z[:, i]) * np.nanmean(w[D.mask_check])
                d = (u - hold)[D.mask_check]
                rows.append({"freq": freq, "feature": f, "factor": fn, "fit_slope": float(b),
                             "check_mean": float(np.nanmean(u[D.mask_check])), "check_t": nw_t(u[D.mask_check], D.cfg["lags"]),
                             "diff_mean": float(np.nanmean(d)), "diff_t": nw_t(d, D.cfg["lags"])})
    X = pd.DataFrame(rows)
    X["p"] = 1 - norm.cdf(X.diff_t.fillna(-9))
    X["bh_pass"] = bh(X.p.to_numpy())
    X["counts"] = X.bh_pass & (X.diff_t >= 1.645) & (X.check_mean > 0)
    return X


def combine(inp, data_dir, out):
    import glob
    os.makedirs(out, exist_ok=True)
    C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(inp, "**", "chosen_*.csv"), recursive=True))],
                  ignore_index=True)
    C = C.drop_duplicates(["tier", "freq", "family"], keep="first")
    C["p"] = 1 - norm.cdf(C.check_t.fillna(-9))
    C["bh_pass"] = bh(C.p.to_numpy())
    C["P1"] = (C.check_mean > 0) & (C.check_t >= 1.645) & C.bh_pass
    C["P2"] = C.diff_check_t >= 1.645
    C["counts"] = C.P1 & C.P2 & (C.P3 == True) & (C.P4 != False)                                  # noqa: E712
    C.to_csv(os.path.join(out, "chosen_all.csv"), index=False)
    S = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(inp, "**", "surface_*.csv.gz"), recursive=True))],
                  ignore_index=True)
    surf = S.groupby(["tier", "freq", "family"]).agg(settings=("variant", "size"), med_check_t=("check_t", "median"),
                                                     share_ge_1645=("check_t", lambda x: float((x >= 1.645).mean())),
                                                     best_check_t=("check_t", "max")).reset_index()
    surf.to_csv(os.path.join(out, "surface_summary.csv"), index=False)
    kc = []
    for fam in ("M2", "M4"):
        g = S[S.family == fam].copy()
        if g.empty:
            continue
        g["K"] = g.params.map(lambda s: json.loads(s).get("K"))
        for (tn, fr, K), h in g.groupby(["tier", "freq", "K"]):
            b = h.loc[h.inner_t.idxmax()]
            kc.append({"tier": tn, "freq": fr, "family": fam, "K": int(K), "variant": b.variant, "inner_t": b.inner_t,
                       "check_t": b.check_t})
    pd.DataFrame(kc).to_csv(os.path.join(out, "k_curve.csv"), index=False)
    X = timing_tests(data_dir) if data_dir else pd.DataFrame()
    if len(X):
        X.to_csv(os.path.join(out, "timing_long_only.csv"), index=False)
    picks = {}
    for (tn, fr), g in C.groupby(["tier", "freq"]):
        b = g.loc[g.inner_t.idxmax()]
        picks[f"{tn}|{fr}"] = {"family": b.family, "variant": b.variant, "inner_t": b.inner_t, "check_t": b.check_t,
                               "counts": bool(b.counts)}
    summ = {"tests": int(len(C)), "counts": int(C.counts.sum()),
            "p1_pre": int(C.p1_pre.sum()), "bh_pass": int(C.bh_pass.sum()),
            "counting": C[C.counts].to_dict("records"), "overall_pick": picks,
            "top10": C.sort_values("p").head(10)[["tier", "freq", "family", "variant", "inner_t", "check_t",
                                                  "check_sharpe", "diff_check_t", "P3", "P4", "bh_pass"]].to_dict("records"),
            "timing": {"tests": int(len(X)), "counts": int(X.counts.sum()) if len(X) else 0,
                       "top": X.sort_values("p").head(10).to_dict("records") if len(X) else []}}
    json.dump(summ, open(os.path.join(out, "summary17.json"), "w"), indent=1, default=str)
    print(json.dumps({k: summ[k] for k in ("tests", "counts", "p1_pre", "bh_pass", "overall_pick")}, indent=1, default=str))


# ================================================================================================ synthetic checks
def synthetic(kind, freq="M", seed=7, S=40):
    rng = np.random.default_rng(seed)
    if freq == "M":
        ends = pd.date_range("2007-04-30", "2025-12-31", freq="ME")
    elif freq == "W":
        ends = pd.date_range("2007-04-06", "2025-12-31", freq="W-FRI")
    else:
        ends = pd.bdate_range("2007-04-02", "2025-12-31")
    T = len(ends)
    x = np.zeros(T)
    for i in range(1, T):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    F = pd.DataFrame(rng.normal(size=(T, len(FEATS))), index=ends, columns=FEATS)
    R = rng.normal(0, 1, (T, S))
    xprev = np.r_[0.0, x[:-1]]
    if kind == "planted":
        R[:, 0] += 0.8 * np.sign(xprev)
        R[:, 1] -= 0.8 * np.sign(xprev)
        F["RV21"] = x
    elif kind == "leak":
        F["RV21"] = np.r_[np.sign(R[1:, 0]), 0.0]
    elif kind == "leak_shift":
        F["RV21"] = np.sign(R[:, 0])
    c = FREQ[freq]
    desc = pd.DataFrame({"unit": [f"S.X{i // 2}|D1" for i in range(S)], "cfg": "c", "dir": 1, "engine": "D",
                         "family": ["trend" if i % 2 else "mean_reversion" for i in range(S)], "symbol": ["EURUSD"] * S})
    tier = Tier("BOOK-CFG", scale(R, c["nsc"], c["act"]), [f"s{i}" for i in range(S)], desc=desc)
    ptier = Tier("BOOK-CFG", scale(rng.normal(0, 1, (T, S)), c["nsc"], c["act"]), [f"p{i}" for i in range(S)], desc=desc)
    return Data([str(e.date()) for e in ends], ends, tier, ptier, F, freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="")
    ap.add_argument("--tier", default="FACTOR")
    ap.add_argument("--freq", default="M")
    ap.add_argument("--fams", default="all")
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--combine", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placebos", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.0f}s] {s}", flush=True)        # noqa: E731
    if a.combine:
        combine(a.combine, a.data, a.out)
        return
    D = synthetic(a.synthetic, a.freq) if a.synthetic else load(a.data, a.tier, a.freq)
    tag = f"{a.tier}_{a.freq}_{a.fams}"
    run_job(D, GROUPS[a.fams], a.out, a.placebos, a.jobs, log, tag)


if __name__ == "__main__":
    main()
