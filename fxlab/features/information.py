"""7.11 Information-theoretic features on trailing windows of log returns.
Discretisation always uses the window's own standard deviation (never global
statistics) so values are point-in-time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import rolling_apply

F = "information_theory"
WS = {"w": [100, 200, 500]}


def _lr(df):
    return np.log(df["close"]).diff()


@njit(cache=True)
def _sym(x, edges):
    s = (x - x.mean()) / (x.std() + 1e-12)
    out = np.zeros(len(s), np.int64)
    for i in range(len(s)):
        k = 0
        for e in edges:
            if s[i] > e:
                k += 1
        out[i] = k
    return out


EDGES8 = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
EDGES3 = np.array([-0.5, 0.5])


@njit(cache=True)
def _entropy_counts(c):
    t = c.sum()
    h = 0.0
    for v in c:
        if v > 0:
            p = v / t
            h -= p * np.log(p)
    return h


@njit(cache=True)
def _shannon(x):
    s = _sym(x, EDGES8)
    c = np.zeros(8)
    for v in s:
        c[v] += 1
    return _entropy_counts(c) / np.log(8)


@feature("F.IT.SHANNON", F, "normalised Shannon entropy of returns discretised into 8 bins at {-2,-1,-.5,0,.5,1,2} "
         "window-std units", params={"w": 200}, space=WS)
def f_shannon(df, w=200):
    return rolling_apply(_lr(df), w, _shannon)


@njit(cache=True)
def _perm_entropy(x):
    m = 4
    n = len(x) - m + 1
    counts = np.zeros(256)
    for i in range(n):
        idx = np.argsort(x[i:i + m])
        code = 0
        for k in range(m):
            code = code * 4 + idx[k]
        counts[code] += 1
    return _entropy_counts(counts) / np.log(24.0)


@feature("F.IT.PERMUTATION", F, "Bandt-Pompe permutation entropy (order 4, delay 1), normalised by ln 4!", params={"w": 200}, space=WS)
def f_perm(df, w=200):
    return rolling_apply(_lr(df), w, _perm_entropy)


@njit(cache=True)
def _sampen(x):
    m = 2
    r = 0.2 * x.std()
    n = len(x)
    B = 0.0; A = 0.0
    for i in range(n - m):
        for j in range(i + 1, n - m):
            ok = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    ok = False
                    break
            if ok:
                B += 1
                if abs(x[i + m] - x[j + m]) <= r:
                    A += 1
    if A == 0 or B == 0:
        return np.nan
    return -np.log(A / B)


@feature("F.IT.SAMPLE_ENTROPY", F, "sample entropy SampEn(m=2, r=0.2 std) = -ln(A/B)", params={"w": 200, "step": 1},
         space={"w": [100, 200], "step": [1, 5]}, complexity="O(n·w^2/step)")
def f_sampen(df, w=200, step=1):
    return rolling_apply(_lr(df), w, _sampen, step=step)


@njit(cache=True)
def _apen(x):
    m = 2
    r = 0.2 * x.std()
    n = len(x)
    phi = np.zeros(2)
    for mm in range(m, m + 2):
        N = n - mm + 1
        acc = 0.0
        for i in range(N):
            c = 0
            for j in range(N):
                ok = True
                for k in range(mm):
                    if abs(x[i + k] - x[j + k]) > r:
                        ok = False
                        break
                if ok:
                    c += 1
            acc += np.log(c / N)
        phi[mm - m] = acc / N
    return phi[0] - phi[1]


@feature("F.IT.APPROX_ENTROPY", F, "approximate entropy ApEn(m=2, r=0.2 std) = Phi_m - Phi_{m+1}", params={"w": 150, "step": 1},
         space={"w": [100, 150], "step": [1, 5]}, complexity="O(n·w^2/step)")
def f_apen(df, w=150, step=1):
    return rolling_apply(_lr(df), w, _apen, step=step)


@njit(cache=True)
def _lz76(x):
    s = np.zeros(len(x), np.int64)
    for i in range(len(x)):
        s[i] = 1 if x[i] > 0 else 0
    n = len(s)
    c = 1; l = 1; i = 0; k = 1; kmax = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i = 0; k = 1; kmax = 1
            else:
                k = 1
    return c * np.log2(n) / n


@feature("F.IT.LEMPEL_ZIV", F, "normalised Lempel-Ziv (1976) complexity c(n) log2(n)/n of the binary up/down sequence",
         params={"w": 200}, space=WS)
def f_lz(df, w=200):
    return rolling_apply(_lr(df), w, _lz76)


@njit(cache=True)
def _mi_lag1(x):
    s = _sym(x, EDGES3)
    j = np.zeros((3, 3))
    for i in range(1, len(s)):
        j[s[i - 1], s[i]] += 1
    t = j.sum()
    px = j.sum(1) / t; py = j.sum(0) / t
    mi = 0.0
    for a in range(3):
        for b in range(3):
            if j[a, b] > 0:
                p = j[a, b] / t
                mi += p * np.log(p / (px[a] * py[b]))
    return mi


@feature("F.IT.AUTO_MI", F, "lag-1 mutual information I(r_t; r_{t-1}) with 3-state discretisation (±0.5 std)",
         params={"w": 200}, space=WS)
def f_auto_mi(df, w=200):
    return rolling_apply(_lr(df), w, _mi_lag1)


@njit(cache=True)
def _cond_mi(x):
    s = _sym(x, EDGES3)
    z = np.abs((x - x.mean()) / (x.std() + 1e-12))
    med = np.median(z)
    cmi = 0.0
    tot = len(s) - 1
    for state in range(2):
        j = np.zeros((3, 3))
        for i in range(1, len(s)):
            st = 1 if z[i - 1] > med else 0
            if st == state:
                j[s[i - 1], s[i]] += 1
        t = j.sum()
        if t == 0:
            continue
        px = j.sum(1) / t; py = j.sum(0) / t
        mi = 0.0
        for a in range(3):
            for b in range(3):
                if j[a, b] > 0:
                    p = j[a, b] / t
                    mi += p * np.log(p / (px[a] * py[b]))
        cmi += (t / tot) * mi
    return cmi


@feature("F.IT.COND_MI", F, "conditional MI I(r_t; r_{t-1} | V_{t-1}), V = 1[|z_{t-1}| > median] (volatility state)",
         params={"w": 300}, space={"w": [200, 300, 500]})
def f_cond_mi(df, w=300):
    return rolling_apply(_lr(df), w, _cond_mi)


@njit(cache=True)
def _kl_halves(x):
    h = len(x) // 2
    ref = x[:h]
    mu, sd = ref.mean(), ref.std() + 1e-12
    p = np.ones(8); q = np.ones(8)  # Laplace smoothing
    for i in range(len(x)):
        s = (x[i] - mu) / sd
        k = 0
        for e in EDGES8:
            if s > e:
                k += 1
        if i < h:
            p[k] += 1
        else:
            q[k] += 1
    p /= p.sum(); q /= q.sum()
    return (q * np.log(q / p)).sum()


@feature("F.IT.KL_DIVERGENCE", F, "KL(Q||P) between binned return distributions of the recent half (Q) and older half (P) "
         "of the trailing window, bins from P's mean/std, Laplace-smoothed", params={"w": 400}, space={"w": [200, 400, 1000]})
def f_kl(df, w=400):
    return rolling_apply(_lr(df), w, _kl_halves)


@njit(cache=True)
def _block_entropy(s, L):
    n = len(s) - L + 1
    codes = np.zeros(n, np.int64)
    for i in range(n):
        c = 0
        for k in range(L):
            c = c * 3 + s[i + k]
        codes[i] = c
    counts = np.zeros(3 ** L)
    for c in codes:
        counts[c] += 1
    return _entropy_counts(counts)


@njit(cache=True)
def _entropy_rate(x):
    s = _sym(x, EDGES3)
    return _block_entropy(s, 4) - _block_entropy(s, 3)


@feature("F.IT.ENTROPY_RATE", F, "entropy-rate estimate h = H_4 - H_3 of block entropies of the 3-symbol return sequence",
         params={"w": 500}, space={"w": [300, 500, 1000]})
def f_entropy_rate(df, w=500):
    return rolling_apply(_lr(df), w, _entropy_rate)


@njit(cache=True)
def _mse(x):
    tot = 0.0
    cnt = 0
    for scale in (1, 2, 4):
        n = len(x) // scale
        y = np.zeros(n)
        for i in range(n):
            y[i] = x[i * scale:(i + 1) * scale].mean()
        v = _sampen(y)
        if not np.isnan(v):
            tot += v; cnt += 1
    return tot / cnt if cnt > 0 else np.nan


@feature("F.IT.MULTISCALE", F, "multiscale entropy complexity index: mean SampEn of coarse-grained returns at scales 1,2,4 "
         "(Costa et al.)", params={"w": 256, "step": 5}, space={"w": [256], "step": [5, 10]}, complexity="O(n·w^2/step)")
def f_mse(df, w=256, step=5):
    return rolling_apply(_lr(df), w, _mse, step=step)
