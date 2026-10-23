"""7.15 Market-regime features. Every regime probability is FILTERED (uses data
up to t only). Model parameters are re-estimated on a fixed absolute schedule
on trailing windows; we never use smoothed (forward-backward) posteriors."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, atr, true_range, rolling_apply

F = "regime"


def _lr(df):
    return np.log(df["close"]).diff()


@feature("F.RG.VOL_PERCENTILE", F, "volatility regime: percentile rank of ATR_14/C within trailing n bars",
         params={"n": 1000}, space={"n": [250, 1000, 5000]}, inputs=("high", "low", "close"))
def f_vol_pct(df, n=1000):
    return (atr(df, 14) / df["close"]).rolling(n, min_periods=n // 2).rank(pct=True)


@feature("F.RG.CHOPPINESS", F, "choppiness index 100 log10(sum_n TR / (max_n H - min_n L)) / log10 n (trend vs range)",
         params={"n": 14}, space={"n": [14, 28, 50]}, inputs=("high", "low", "close"))
def f_chop(df, n=14):
    s = true_range(df).rolling(n, min_periods=n).sum()
    rng = df["high"].rolling(n, min_periods=n).max() - df["low"].rolling(n, min_periods=n).min()
    return 100 * np.log10(s / (rng + EPS)) / np.log10(n)


@njit(cache=True)
def _forward_filter(X, startprob, trans, means, covs):
    n, d = X.shape
    k = len(startprob)
    out = np.full((n, k), np.nan)
    alpha = startprob.copy()
    inv = np.empty((k, d, d)); logdet = np.empty(k)
    for j in range(k):
        inv[j] = np.linalg.inv(covs[j])
        logdet[j] = np.log(np.linalg.det(covs[j]))
    for t in range(n):
        if np.isnan(X[t]).any():
            out[t] = alpha
            continue
        pred = trans.T @ alpha if t > 0 else alpha
        lik = np.empty(k)
        for j in range(k):
            dv = X[t] - means[j]
            lik[j] = np.exp(-0.5 * (dv @ inv[j] @ dv) - 0.5 * logdet[j])
        a = pred * lik
        s = a.sum()
        alpha = a / s if s > 0 else pred
        out[t] = alpha
    return out


@feature("F.RG.HMM", F, "Gaussian HMM (k states, full cov) on [l_t, |l_t|] standardised by the estimation window; "
         "parameters fitted by Baum-Welch (hmmlearn) every `refit` bars on the trailing `window`; FILTERED state "
         "probabilities from a forward pass with frozen params; states ordered by variance (state 0 = calmest)",
         params={"k": 2, "window": 2000, "refit": 500}, space={"k": [2, 3], "window": [1000, 2000], "refit": [250, 500]},
         requires=("hmmlearn",), complexity="O(n·k^2 + fits)")
def f_hmm(df, k=2, window=2000, refit=500):
    from hmmlearn.hmm import GaussianHMM
    lr = _lr(df).to_numpy(float)
    X = np.column_stack([lr, np.abs(lr)])
    n = len(X)
    out = np.full((n, k), np.nan)
    for i in range(window, n, refit):
        seg = X[i - window:i]
        seg = seg[~np.isnan(seg).any(1)]
        mu, sd = seg.mean(0), seg.std(0) + EPS
        Z = (seg - mu) / sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=k, covariance_type="full", n_iter=50, random_state=0)
            try:
                m.fit(Z)
            except Exception:
                continue
        order = np.argsort([np.trace(c) for c in m.covars_])
        sp, tr = m.startprob_[order], m.transmat_[np.ix_(order, order)]
        means, covs = m.means_[order], m.covars_[order] + 1e-6 * np.eye(2)
        end = min(n, i + refit)
        # warm start the filter over the estimation window, then continue out-of-sample
        Zall = (X[i - window:end] - mu) / sd
        filt = _forward_filter(Zall, sp, tr, means, covs)
        out[i:end] = filt[window:]
    return pd.DataFrame(out, index=df.index, columns=[f"p{j}" for j in range(k)])


@njit(cache=True)
def _bocpd(x, hazard, mu0, kappa0, alpha0, beta0, rmax):
    n = len(x)
    erl = np.full(n, np.nan); pcp = np.full(n, np.nan)
    R = np.zeros(rmax + 1); R[0] = 1.0
    mu = np.full(rmax + 1, mu0); ka = np.full(rmax + 1, kappa0)
    al = np.full(rmax + 1, alpha0); be = np.full(rmax + 1, beta0)
    for t in range(n):
        xt = x[t]
        if np.isnan(xt):
            continue
        # student-t predictive
        df_ = 2 * al
        scale = np.sqrt(be * (ka + 1) / (al * ka))
        z = (xt - mu) / scale
        from_lg = np.empty(rmax + 1)
        for r in range(rmax + 1):
            v = df_[r]
            # log t-pdf
            from_lg[r] = (math.lgamma((v + 1) / 2) - math.lgamma(v / 2) - 0.5 * np.log(v * np.pi)
                          - np.log(scale[r]) - (v + 1) / 2 * np.log(1 + z[r] ** 2 / v))
        pred = np.exp(from_lg)
        growth = R * pred * (1 - hazard)
        cp = (R * pred * hazard).sum()
        newR = np.zeros(rmax + 1)
        newR[1:] = growth[:-1]
        newR[rmax] += growth[rmax]
        newR[0] = cp
        s = newR.sum()
        R = newR / s if s > 0 else newR
        # update sufficient stats (shift)
        nmu = np.empty(rmax + 1); nka = np.empty(rmax + 1); nal = np.empty(rmax + 1); nbe = np.empty(rmax + 1)
        nmu[0], nka[0], nal[0], nbe[0] = mu0, kappa0, alpha0, beta0
        for r in range(rmax):
            nmu[r + 1] = (ka[r] * mu[r] + xt) / (ka[r] + 1)
            nka[r + 1] = ka[r] + 1
            nal[r + 1] = al[r] + 0.5
            nbe[r + 1] = be[r] + ka[r] * (xt - mu[r]) ** 2 / (2 * (ka[r] + 1))
        mu, ka, al, be = nmu, nka, nal, nbe
        erl[t] = (R * np.arange(rmax + 1)).sum()
        pcp[t] = R[:5].sum()
    return erl, pcp


@feature("F.RG.BOCPD", F, "Bayesian online change-point detection (Adams & MacKay 2007), Normal-Gamma model on "
         "standardised returns, constant hazard 1/lambda, run length truncated at rmax; outputs expected run length "
         "and P(run length < 5)", params={"lam": 250, "rmax": 300}, space={"lam": [100, 250, 1000]},
         complexity="O(n·rmax)")
def f_bocpd(df, lam=250, rmax=300):
    lr = _lr(df)
    sd = lr.ewm(span=500, min_periods=100).std().shift(1)
    z = (lr / (sd + EPS)).to_numpy(float)
    erl, p = _bocpd(z, 1.0 / lam, 0.0, 1.0, 1.0, 1.0, rmax)
    return pd.DataFrame({"exp_run": erl, "p_recent_cp": p}, index=df.index)


@feature("F.RG.LIQUIDITY_REGIME", F, "liquidity regime: mean of percentile ranks (trailing n) of spread_mean (inverted) and "
         "quote intensity n_ticks; 1 = most liquid", params={"n": 2000}, space={"n": [500, 2000]},
         inputs=("spread_mean", "n_ticks"))
def f_liq_regime(df, n=2000):
    s = df["spread_mean"].rolling(n, min_periods=n // 2).rank(pct=True)
    t = df["n_ticks"].astype(float).rolling(n, min_periods=n // 2).rank(pct=True)
    return ((1 - s) + t) / 2


@feature("F.RG.GMM", F, "Gaussian-mixture regime (no temporal dynamics): mixture with k components on [ln vol_20, "
         "trend t-stat_20] fitted on the trailing window every `refit` bars; posterior of the highest-vol component",
         params={"k": 3, "window": 2000, "refit": 500}, space={"k": [2, 3, 4]})
def f_gmm(df, k=3, window=2000, refit=500):
    from sklearn.mixture import GaussianMixture
    lr = _lr(df)
    v = np.log(lr.rolling(20, min_periods=20).std() + EPS)
    t = lr.rolling(20, min_periods=20).mean() / (lr.rolling(20, min_periods=20).std() + EPS) * np.sqrt(20)
    X = np.column_stack([v, t])
    n = len(X)
    out = np.full(n, np.nan)
    for i in range(window, n, refit):
        seg = X[i - window:i]
        seg = seg[~np.isnan(seg).any(1)]
        if len(seg) < window // 2:
            continue
        g = GaussianMixture(k, random_state=0).fit(seg)
        hi = int(np.argmax(g.means_[:, 0]))
        end = min(n, i + refit)
        blk = X[i:end]
        ok = ~np.isnan(blk).any(1)
        res = np.full(len(blk), np.nan)
        if ok.any():
            res[ok] = g.predict_proba(blk[ok])[:, hi]
        out[i:end] = res
    return pd.Series(out, index=df.index)


@njit(cache=True)
def _binseg_var(x):
    n = len(x)
    s2 = x * x
    tot = s2.sum()
    best = 0.0; bk = -1
    cs = np.cumsum(s2)
    for k in range(20, n - 20):
        v1 = cs[k - 1] / k; v2 = (tot - cs[k - 1]) / (n - k); v = tot / n
        if v1 <= 0 or v2 <= 0:
            continue
        lr = n * np.log(v) - k * np.log(v1) - (n - k) * np.log(v2)
        if lr > best:
            best = lr; bk = k
    return best, (n - bk) if bk >= 0 else np.nan


@njit(cache=True)
def _binseg_stat(x):
    return _binseg_var(x)[0]


@njit(cache=True)
def _binseg_age(x):
    return _binseg_var(x)[1]


@feature("F.RG.VARIANCE_CHANGEPOINT", F, "single variance change point in the trailing window (Gaussian likelihood-ratio "
         "scan, 20-bar margins): LR statistic and bars since the most likely change point",
         params={"n": 300, "step": 5}, space={"n": [200, 300, 500]})
def f_var_cp(df, n=300, step=5):
    lr = _lr(df)
    return pd.DataFrame({"lr_stat": rolling_apply(lr, n, _binseg_stat, step=step),
                         "bars_since": rolling_apply(lr, n, _binseg_age, step=step)})


@njit(cache=True)
def _schmitt(z, up, dn):
    out = np.full(len(z), np.nan)
    s = 0.0
    for i in range(len(z)):
        if np.isnan(z[i]):
            continue
        if s <= 0 and z[i] > up:
            s = 1.0
        elif s >= 0 and z[i] < dn:
            s = -1.0
        out[i] = s
    return out


@feature("F.RG.VOL_HYSTERESIS", F, "two-threshold (Schmitt trigger) volatility state on the z-score of ln RV_20 vs trailing n: "
         "+1 high-vol entered above `up`, -1 low-vol entered below `dn`, state persists in between",
         params={"n": 1000, "up": 1.0, "dn": -0.5}, space={"up": [0.5, 1.0], "dn": [-0.5, 0.0]})
def f_vol_hyst(df, n=1000, up=1.0, dn=-0.5):
    lv = np.log(_lr(df).rolling(20, min_periods=20).std() + EPS)
    z = (lv - lv.rolling(n, min_periods=n // 2).mean()) / (lv.rolling(n, min_periods=n // 2).std() + EPS)
    return pd.Series(_schmitt(z.to_numpy(float), up, dn), index=df.index)


@feature("F.RG.DRAWDOWN_REGIME", F, "bull/bear regime proxy: ln(C/max_n C) (drawdown) and ln(C/min_n C) (rally) over a long "
         "trailing horizon n (e.g. ~1 year of bars)", params={"n": 6000}, space={"n": [1500, 6000]})
def f_dd_regime(df, n=6000):
    c = df["close"]
    return pd.DataFrame({"drawdown": np.log(c / c.rolling(n, min_periods=n // 4).max()),
                         "rally": np.log(c / c.rolling(n, min_periods=n // 4).min())})
