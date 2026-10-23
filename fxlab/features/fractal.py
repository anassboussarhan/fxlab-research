"""7.10 Fractal and nonlinear dynamics. Each estimator is evaluated on a
trailing window; heavy estimators accept ``step`` (evaluate every k bars and
carry the last computed value forward, which keeps causality)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import rolling_apply

F = "fractal_nonlinear"


def _lr(df):
    return np.log(df["close"]).diff()


def _lp(df):
    return np.log(df["close"])


@njit(cache=True)
def _ols_slope(x, y):
    xm = x.mean(); ym = y.mean()
    return ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()


@njit(cache=True)
def _hurst_rs(r):
    n = len(r)
    sizes = np.array([8, 16, 32, 64, 128, 256])
    xs = []; ys = []
    for s in sizes:
        if s > n // 2:
            break
        m = n // s
        acc = 0.0; cnt = 0
        for j in range(m):
            seg = r[j * s:(j + 1) * s]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std()
            if S > 0:
                acc += R / S; cnt += 1
        if cnt > 0:
            xs.append(np.log(s)); ys.append(np.log(acc / cnt))
    if len(xs) < 3:
        return np.nan
    return _ols_slope(np.array(xs), np.array(ys))


@feature("F.FR.HURST_RS", F, "rescaled-range Hurst exponent: slope of ln E[R/S](s) on ln s, s in {8..256} within trailing w returns",
         params={"w": 256, "step": 1}, space={"w": [128, 256, 512]}, complexity="O(n·w)")
def f_hurst(df, w=256, step=1):
    return rolling_apply(_lr(df), w, _hurst_rs, step=step)


@njit(cache=True)
def _dfa(r):
    n = len(r)
    y = np.cumsum(r - r.mean())
    xs = []; ys = []
    for s in (8, 16, 32, 64, 128):
        if s > n // 4:
            break
        m = n // s
        t = np.arange(s).astype(np.float64)
        tm = t.mean(); stt = ((t - tm) ** 2).sum()
        f2 = 0.0
        for j in range(m):
            seg = y[j * s:(j + 1) * s]
            b = ((t - tm) * (seg - seg.mean())).sum() / stt
            res = seg - (seg.mean() + b * (t - tm))
            f2 += (res ** 2).mean()
        xs.append(np.log(s)); ys.append(0.5 * np.log(f2 / m))
    if len(xs) < 3:
        return np.nan
    return _ols_slope(np.array(xs), np.array(ys))


@feature("F.FR.DFA", F, "detrended fluctuation analysis exponent alpha (DFA-1, linear detrending per box)",
         params={"w": 256, "step": 1}, space={"w": [128, 256, 512]}, complexity="O(n·w)")
def f_dfa(df, w=256, step=1):
    return rolling_apply(_lr(df), w, _dfa, step=step)


@njit(cache=True)
def _higuchi(x):
    n = len(x)
    kmax = 10
    lk = np.zeros(kmax); lx = np.zeros(kmax)
    for k in range(1, kmax + 1):
        L = 0.0
        for m in range(k):
            cnt = (n - m - 1) // k
            if cnt < 1:
                continue
            s = 0.0
            for i in range(1, cnt + 1):
                s += abs(x[m + i * k] - x[m + (i - 1) * k])
            L += s * (n - 1) / (cnt * k) / k
        lk[k - 1] = np.log(L / k + 1e-300)
        lx[k - 1] = np.log(1.0 / k)
    return _ols_slope(lx, lk)


@feature("F.FR.HIGUCHI", F, "Higuchi fractal dimension (kmax=10) of trailing ln C", params={"w": 128, "step": 1},
         space={"w": [64, 128, 256]}, complexity="O(n·w·kmax)")
def f_higuchi(df, w=128, step=1):
    return rolling_apply(_lp(df), w, _higuchi, step=step)


@njit(cache=True)
def _katz(x):
    n = len(x) - 1
    d = np.abs(np.diff(x))
    L = d.sum()
    if L <= 0:
        return np.nan
    a = L / n
    dmax = np.abs(x - x[0]).max()
    if dmax <= 0:
        return np.nan
    return np.log10(n) / (np.log10(n) + np.log10(dmax / L))


@feature("F.FR.KATZ", F, "Katz fractal dimension log10(n)/(log10(n)+log10(d/L)) of trailing ln C", params={"w": 100},
         space={"w": [50, 100, 250]})
def f_katz(df, w=100):
    return rolling_apply(_lp(df), w, _katz)


@njit(cache=True)
def _petrosian(x):
    d = np.diff(x)
    nd = 0
    for i in range(1, len(d)):
        if d[i] * d[i - 1] < 0:
            nd += 1
    n = len(x)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nd)))


@feature("F.FR.PETROSIAN", F, "Petrosian FD from number of sign changes of the derivative", params={"w": 100}, space={"w": [50, 100, 250]})
def f_petrosian(df, w=100):
    return rolling_apply(_lp(df), w, _petrosian)


@njit(cache=True)
def _sevcik(x):
    n = len(x)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.nan
    y = (x - mn) / (mx - mn)
    L = 0.0
    for i in range(1, n):
        L += np.sqrt((y[i] - y[i - 1]) ** 2 + (1.0 / (n - 1)) ** 2)
    return 1 + np.log(L) / np.log(2 * (n - 1))


@feature("F.FR.SEVCIK", F, "Sevcik fractal dimension 1 + ln L / ln(2(N-1)) of curve normalised to the unit square",
         params={"w": 100}, space={"w": [50, 100, 250]})
def f_sevcik(df, w=100):
    return rolling_apply(_lp(df), w, _sevcik)


@njit(cache=True)
def _boxcount(x):
    n = len(x)
    mn, mx = x.min(), x.max()
    if mx <= mn:
        return np.nan
    y = (x - mn) / (mx - mn)
    t = np.arange(n) / (n - 1.0)
    grids = np.array([4, 8, 16, 32])
    ls = np.empty(4); lc = np.empty(4)
    for gi in range(4):
        g = grids[gi]
        occ = np.zeros((g, g), np.bool_)
        for i in range(n - 1):
            for s in range(8):  # sample each segment densely so it marks every box it crosses
                u = s / 8.0
                tx = t[i] + u * (t[i + 1] - t[i]); ty = y[i] + u * (y[i + 1] - y[i])
                a = min(int(tx * g), g - 1); b = min(int(ty * g), g - 1)
                occ[a, b] = True
        ls[gi] = np.log(g); lc[gi] = np.log(occ.sum())
    return _ols_slope(ls, lc)


@feature("F.FR.BOX_COUNT", F, "box-counting dimension of the normalised price path (grids 4..32)", params={"w": 128, "step": 1},
         space={"w": [64, 128, 256]})
def f_boxcount(df, w=128, step=1):
    return rolling_apply(_lp(df), w, _boxcount, step=step)


@njit(cache=True)
def _rqa(r):
    m = 3
    x = (r - r.mean()) / (r.std() + 1e-12)
    n = len(x) - m + 1
    eps = 0.5
    R = np.zeros((n, n), np.bool_)
    rec = 0
    for i in range(n):
        for j in range(n):
            d = 0.0
            for k in range(m):
                d = max(d, abs(x[i + k] - x[j + k]))
            if d < eps:
                R[i, j] = True
                if i != j:
                    rec += 1
    total = n * (n - 1)
    rr = rec / total
    diag = 0
    for off in range(1, n):
        run = 0
        for i in range(n - off):
            if R[i, i + off]:
                run += 1
            else:
                if run >= 2:
                    diag += run
                run = 0
        if run >= 2:
            diag += run
    det = 2 * diag / rec if rec > 0 else 0.0
    vert = 0
    for j in range(n):
        run = 0
        for i in range(n):
            if R[i, j] and i != j:
                run += 1
            else:
                if run >= 2:
                    vert += run
                run = 0
        if run >= 2:
            vert += run
    lam = vert / rec if rec > 0 else 0.0
    return rr, det, lam


@njit(cache=True)
def _rqa_rr(r):
    return _rqa(r)[0]


@njit(cache=True)
def _rqa_det(r):
    return _rqa(r)[1]


@njit(cache=True)
def _rqa_lam(r):
    return _rqa(r)[2]


@feature("F.FR.RQA", F, "recurrence quantification of standardised returns (embedding m=3, Chebyshev eps=0.5): "
         "recurrence rate RR, determinism DET (diagonal lines >=2), laminarity LAM (vertical lines >=2)",
         params={"w": 100, "step": 5}, space={"w": [64, 100, 150], "step": [1, 5, 10]}, complexity="O(n·w^2/step)")
def f_rqa(df, w=100, step=5):
    lr = _lr(df)
    return pd.DataFrame({"rr": rolling_apply(lr, w, _rqa_rr, step=step), "det": rolling_apply(lr, w, _rqa_det, step=step),
                         "lam": rolling_apply(lr, w, _rqa_lam, step=step)})


@njit(cache=True)
def _lyap(r):
    m = 3; sep = 5; K = 6
    n = len(r) - m + 1
    x = (r - r.mean()) / (r.std() + 1e-12)
    div = np.zeros(K); cnt = np.zeros(K)
    for i in range(n - K):
        best = np.inf; bj = -1
        for j in range(n - K):
            if abs(i - j) <= sep:
                continue
            d = 0.0
            for k in range(m):
                d += (x[i + k] - x[j + k]) ** 2
            if d < best:
                best = d; bj = j
        if bj < 0:
            continue
        for k in range(K):
            d = 0.0
            for q in range(m):
                d += (x[i + k + q] - x[bj + k + q]) ** 2
            if d > 0:
                div[k] += 0.5 * np.log(d); cnt[k] += 1
    ks = []; ds = []
    for k in range(K):
        if cnt[k] > 0:
            ks.append(float(k)); ds.append(div[k] / cnt[k])
    if len(ks) < 3:
        return np.nan
    return _ols_slope(np.array(ks), np.array(ds))


@feature("F.FR.LYAPUNOV", F, "largest Lyapunov exponent proxy (Rosenstein): slope of mean log divergence of nearest "
         "neighbours (m=3, Theiler window 5) over 0..5 steps", params={"w": 128, "step": 5},
         space={"w": [100, 128, 200], "step": [5, 10]}, complexity="O(n·w^2/step)")
def f_lyapunov(df, w=128, step=5):
    return rolling_apply(_lr(df), w, _lyap, step=step)


@njit(cache=True)
def _corrdim(r):
    m = 4
    x = (r - r.mean()) / (r.std() + 1e-12)
    n = len(x) - m + 1
    r1, r2 = 0.5, 1.0
    c1 = 0; c2 = 0; tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(m):
                d = max(d, abs(x[i + k] - x[j + k]))
            tot += 1
            if d < r1:
                c1 += 1
            if d < r2:
                c2 += 1
    if c1 == 0 or c2 == 0:
        return np.nan
    return (np.log(c2 / tot) - np.log(c1 / tot)) / (np.log(r2) - np.log(r1))


@feature("F.FR.CORR_DIM", F, "Grassberger-Procaccia correlation-dimension slope between radii 0.5 and 1.0 std "
         "(embedding m=4, Chebyshev)", params={"w": 128, "step": 5}, space={"w": [128, 256], "step": [5, 10]},
         complexity="O(n·w^2/step)")
def f_corrdim(df, w=128, step=5):
    return rolling_apply(_lr(df), w, _corrdim, step=step)


@njit(cache=True)
def _mfdfa_width(r):
    n = len(r)
    y = np.cumsum(r - r.mean())
    qs = np.array([-3.0, 3.0])
    hq = np.zeros(2)
    for qi in range(2):
        q = qs[qi]
        xs = []; ys = []
        for s in (8, 16, 32, 64):
            if s > n // 4:
                break
            m = n // s
            t = np.arange(s).astype(np.float64); tm = t.mean(); stt = ((t - tm) ** 2).sum()
            acc = 0.0
            for j in range(m):
                seg = y[j * s:(j + 1) * s]
                b = ((t - tm) * (seg - seg.mean())).sum() / stt
                f2 = ((seg - (seg.mean() + b * (t - tm))) ** 2).mean()
                acc += (f2 + 1e-300) ** (q / 2)
            xs.append(np.log(s)); ys.append(np.log(acc / m) / q)
        if len(xs) < 3:
            return np.nan
        hq[qi] = _ols_slope(np.array(xs), np.array(ys))
    return hq[0] - hq[1]


@feature("F.FR.MFDFA_WIDTH", F, "multifractal DFA: generalised Hurst h(q) for q=-3 and q=+3; width h(-3)-h(3)",
         params={"w": 256, "step": 1}, space={"w": [256, 512]}, complexity="O(n·w)")
def f_mfdfa(df, w=256, step=1):
    return rolling_apply(_lr(df), w, _mfdfa_width, step=step)


@njit(cache=True)
def _ghe(x, q):
    taus = np.arange(1, 11)
    ls = np.zeros(10); lk = np.zeros(10)
    for k in range(10):
        t = taus[k]
        d = np.abs(x[t:] - x[:-t]) ** q
        lk[k] = np.log(d.mean() + 1e-300); ls[k] = np.log(t)
    return _ols_slope(ls, lk) / q


@njit(cache=True)
def _ghe1(x):
    return _ghe(x, 1.0)


@njit(cache=True)
def _ghe2(x):
    return _ghe(x, 2.0)


@feature("F.FR.GEN_HURST", F, "generalised Hurst exponent H(q) from q-order structure functions E|x(t+tau)-x(t)|^q ~ "
         "tau^{qH(q)}, tau=1..10, q in {1,2}; also H(1)-H(2) (multiscaling)", params={"w": 200}, space={"w": [100, 200, 500]})
def f_ghe(df, w=200):
    lp = _lp(df)
    h1, h2 = rolling_apply(lp, w, _ghe1), rolling_apply(lp, w, _ghe2)
    return pd.DataFrame({"h1": h1, "h2": h2, "multiscale": h1 - h2})


@njit(cache=True)
def _simplex_skill(r):
    m = 3; kk = 4
    n = len(r)
    preds = []; acts = []
    for tgt in range(n - 21, n - 1):
        best_d = np.full(kk, np.inf); best_j = np.full(kk, -1)
        for j in range(m - 1, tgt - 1):
            d = 0.0
            for q in range(m):
                d += (r[tgt - q] - r[j - q]) ** 2
            # insert into k-best
            w = np.argmax(best_d)
            if d < best_d[w]:
                best_d[w] = d; best_j[w] = j
        if best_j.min() < 0:
            continue
        wt = np.exp(-np.sqrt(best_d) / (np.sqrt(best_d.min()) + 1e-12))
        p = 0.0
        for q in range(kk):
            p += wt[q] * r[best_j[q] + 1]
        preds.append(p / wt.sum()); acts.append(r[tgt + 1])
    if len(preds) < 10:
        return np.nan
    a = np.array(preds); b = np.array(acts)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)


@feature("F.FR.SIMPLEX_SKILL", F, "nonlinear state-space predictability: correlation between simplex-projection (m=3, k=4 "
         "nearest neighbours) one-step forecasts and realised returns for the last 20 in-window targets",
         params={"w": 200, "step": 5}, space={"w": [150, 200, 300], "step": [5, 10]}, complexity="O(n·w·20/step)")
def f_simplex(df, w=200, step=5):
    return rolling_apply(_lr(df), w, _simplex_skill, step=step)
