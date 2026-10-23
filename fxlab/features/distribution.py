"""7.5 Statistical distribution features of trailing log returns.
Not counted: Jarque-Bera (a function of SKEW and KURT) and the gain-loss ratio
(identical to Omega at threshold 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, logret, rolling_apply

F = "distribution"
NS = {"n": [50, 100, 250]}


@feature("F.DI.SKEW", F, "rolling sample skewness of l over n", params={"n": 100}, space=NS)
def f_skew(df, n=100):
    return logret(df["close"]).rolling(n, min_periods=n).skew()


@feature("F.DI.KURT", F, "rolling excess kurtosis of l over n", params={"n": 100}, space=NS)
def f_kurt(df, n=100):
    return logret(df["close"]).rolling(n, min_periods=n).kurt()


@feature("F.DI.QUANTILES", F, "rolling empirical quantiles q05, q50, q95 of l scaled by std", params={"n": 100}, space=NS)
def f_quantiles(df, n=100):
    lr = logret(df["close"])
    r = lr.rolling(n, min_periods=n)
    s = r.std() + EPS
    return pd.DataFrame({"q05": r.quantile(0.05) / s, "q50": r.quantile(0.5) / s, "q95": r.quantile(0.95) / s})


@njit(cache=True)
def _mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m)) * 1.4826


@feature("F.DI.MAD", F, "robust dispersion: 1.4826 median |l - median(l)| relative to std (outlier sensitivity)",
         params={"n": 100}, space=NS)
def f_mad(df, n=100):
    lr = logret(df["close"])
    return rolling_apply(lr, n, _mad) / (lr.rolling(n, min_periods=n).std() + EPS)


@feature("F.DI.TAIL_RATIO", F, "tail ratio |q95| / |q05| of l", params={"n": 100}, space=NS)
def f_tail_ratio(df, n=100):
    r = logret(df["close"]).rolling(n, min_periods=n)
    return r.quantile(0.95).abs() / (r.quantile(0.05).abs() + EPS)


@njit(cache=True)
def _hill(x):
    a = np.sort(np.abs(x))[::-1]
    k = max(5, int(0.1 * len(a)))
    if a[k] <= 0:
        return np.nan
    s = 0.0
    for i in range(k):
        s += np.log(a[i] / a[k])
    return k / s if s > 0 else np.nan


@feature("F.DI.HILL", F, "Hill tail-index estimator on the top 10% |l| order statistics", params={"n": 250}, space={"n": [250, 500]},
         complexity="O(n·w log w)")
def f_hill(df, n=250):
    return rolling_apply(logret(df["close"]), n, _hill)


@njit(cache=True)
def _gev_shape_pwm(x):
    # block maxima of |x| over blocks of 10, Hosking PWM estimator of GEV shape
    b = 10
    m = len(x) // b
    if m < 8:
        return np.nan
    mx = np.empty(m)
    for j in range(m):
        mx[j] = np.max(np.abs(x[j * b:(j + 1) * b]))
    mx = np.sort(mx)
    b0 = mx.mean()
    b1 = 0.0
    b2 = 0.0
    for i in range(m):
        b1 += (i / (m - 1)) * mx[i]
        b2 += (i * (i - 1) / ((m - 1) * (m - 2))) * mx[i]
    b1 /= m
    b2 /= m
    den = 3 * b2 - b0
    if den == 0:
        return np.nan
    c = (2 * b1 - b0) / den - np.log(2) / np.log(3)
    return -(7.859 * c + 2.9554 * c * c)


@feature("F.DI.GEV_SHAPE", F, "extreme-value: GEV shape xi of block maxima (blocks of 10) of |l| via Hosking PWM",
         params={"n": 250}, space={"n": [250, 500]}, complexity="O(n·w log w)")
def f_gev(df, n=250):
    return rolling_apply(logret(df["close"]), n, _gev_shape_pwm)


@njit(cache=True)
def _lmom(x):
    s = np.sort(x)
    n = len(s)
    b0 = s.mean(); b1 = 0.0; b2 = 0.0; b3 = 0.0
    for i in range(n):
        b1 += i / (n - 1) * s[i]
        b2 += i * (i - 1) / ((n - 1) * (n - 2)) * s[i]
        b3 += i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3)) * s[i]
    b1 /= n; b2 /= n; b3 /= n
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l3 / l2 if l2 != 0 else np.nan, l4 / l2 if l2 != 0 else np.nan


@njit(cache=True)
def _lskew(x):
    return _lmom(x)[0]


@njit(cache=True)
def _lkurt(x):
    return _lmom(x)[1]


@feature("F.DI.LMOMENTS", F, "L-skewness tau3 and L-kurtosis tau4 from probability-weighted moments", params={"n": 100}, space=NS)
def f_lmom(df, n=100):
    lr = logret(df["close"])
    return pd.DataFrame({"tau3": rolling_apply(lr, n, _lskew), "tau4": rolling_apply(lr, n, _lkurt)})


@feature("F.DI.BOWLEY", F, "Bowley quartile skewness (Q3 + Q1 - 2Q2)/(Q3 - Q1)", params={"n": 100}, space=NS)
def f_bowley(df, n=100):
    r = logret(df["close"]).rolling(n, min_periods=n)
    q1, q2, q3 = r.quantile(0.25), r.quantile(0.5), r.quantile(0.75)
    return (q3 + q1 - 2 * q2) / (q3 - q1 + EPS)


@feature("F.DI.OMEGA", F, "Omega ratio at threshold tau=0: E[(l-tau)^+]/E[(tau-l)^+], reported as log", params={"n": 100}, space=NS)
def f_omega(df, n=100):
    lr = logret(df["close"])
    g = lr.clip(lower=0).rolling(n, min_periods=n).mean()
    l = (-lr).clip(lower=0).rolling(n, min_periods=n).mean()
    return np.log((g + EPS) / (l + EPS))


@njit(cache=True)
def _cvar5(x):
    s = np.sort(x)
    k = max(1, int(0.05 * len(s)))
    return s[:k].mean()


@feature("F.DI.CVAR", F, "historical 5% expected shortfall of l over n (scaled by std)", params={"n": 250}, space={"n": [100, 250, 500]})
def f_cvar(df, n=250):
    lr = logret(df["close"])
    return rolling_apply(lr, n, _cvar5) / (lr.rolling(n, min_periods=n).std() + EPS)


@njit(cache=True)
def _ks_split(x):
    h = len(x) // 2
    a = np.sort(x[:h]); b = np.sort(x[h:])
    i = 0; j = 0; d = 0.0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


@feature("F.DI.KS_SHIFT", F, "two-sample Kolmogorov-Smirnov distance between first and second half of the trailing window "
         "(distribution shift detector)", params={"n": 200}, space={"n": [100, 200, 500]}, complexity="O(n·w log w)")
def f_ks(df, n=200):
    return rolling_apply(logret(df["close"]), n, _ks_split)
