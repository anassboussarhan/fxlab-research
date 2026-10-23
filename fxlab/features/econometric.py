"""7.12 Econometric features. Model-based features (GARCH family, HAR) are
re-estimated on a fixed schedule anchored to absolute bar positions using only
the trailing estimation window, then filtered forward with frozen parameters;
this makes every value reproducible from the prefix of data it depends on."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, rolling_apply, bars_per_day

F = "econometrics"


def _lr(df):
    return np.log(df["close"]).diff()


@feature("F.EC.AUTOCORR", F, "rolling lag-k autocorrelation of log returns", params={"n": 100, "k": 1},
         space={"n": [50, 100, 250], "k": [1, 2, 5]})
def f_autocorr(df, n=100, k=1):
    lr = _lr(df)
    return lr.rolling(n, min_periods=n).corr(lr.shift(k))


@feature("F.EC.PACF2", F, "lag-2 partial autocorrelation via Yule-Walker (rho2 - rho1^2)/(1 - rho1^2)", params={"n": 100},
         space={"n": [100, 250]})
def f_pacf2(df, n=100):
    lr = _lr(df)
    r1 = lr.rolling(n, min_periods=n).corr(lr.shift(1))
    r2 = lr.rolling(n, min_periods=n).corr(lr.shift(2))
    return (r2 - r1 ** 2) / (1 - r1 ** 2 + EPS)


@njit(cache=True)
def _ar_resid_z(x):
    p = 3
    n = len(x)
    m = n - p
    X = np.ones((m, p + 1))
    y = np.empty(m)
    for i in range(m):
        y[i] = x[i + p]
        for k in range(p):
            X[i, k + 1] = x[i + p - 1 - k]
    beta = np.linalg.lstsq(X, y)[0]
    res = y - X @ beta
    sd = res[:-1].std()
    return res[-1] / sd if sd > 0 else 0.0


@feature("F.EC.AR_RESID", F, "standardised last residual of an AR(3) OLS fit on the trailing window of returns",
         params={"n": 200}, space={"n": [100, 200, 500]}, complexity="O(n·w)")
def f_ar_resid(df, n=200):
    return rolling_apply(_lr(df), n, _ar_resid_z)


# ----------------------------------------------------------------------------- GARCH family
@njit(cache=True)
def _filter_garch(e, omega, alpha, beta, gamma, kind, s0):
    n = len(e)
    s2 = np.full(n, np.nan)
    cur = s0
    for t in range(n):
        if np.isnan(e[t]):
            s2[t] = cur
            continue
        if kind == 2:  # EGARCH: ln s2_{t+1} = w + a(|z|-sqrt(2/pi)) + g z + b ln s2_t
            z = e[t] / np.sqrt(max(cur, 1e-12))
            lv = omega + alpha * (abs(z) - np.sqrt(2 / np.pi)) + gamma * z + beta * np.log(max(cur, 1e-12))
            cur = np.exp(min(max(lv, -30.0), 30.0))
        else:
            cur = max(omega + alpha * e[t] ** 2 + beta * cur + (gamma * e[t] ** 2 if (kind == 1 and e[t] < 0) else 0.0), 1e-12)
        s2[t] = cur  # forecast of variance for t+1, known at t
    return s2


def _garch_family(df, kind: str, window: int, refit: int):
    from arch import arch_model
    e = (_lr(df) * 100).to_numpy(float)
    n = len(e)
    out = np.full(n, np.nan)
    kmap = {"garch": 0, "gjr": 1, "egarch": 2}
    params = None
    start = window
    for i in range(start, n, refit):
        seg = e[i - window:i]
        seg = seg[~np.isnan(seg)]
        if len(seg) < window // 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if kind == "egarch":
                    am = arch_model(seg, mean="Zero", vol="EGARCH", p=1, o=1, q=1, rescale=False)
                elif kind == "gjr":
                    am = arch_model(seg, mean="Zero", vol="GARCH", p=1, o=1, q=1, rescale=False)
                else:
                    am = arch_model(seg, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
                res = am.fit(disp="off", options={"maxiter": 200})
                pr = res.params
                if kind == "garch":
                    params = (pr["omega"], pr["alpha[1]"], pr["beta[1]"], 0.0)
                elif kind == "gjr":
                    params = (pr["omega"], pr["alpha[1]"], pr["beta[1]"], pr["gamma[1]"])
                else:
                    params = (pr["omega"], pr["alpha[1]"], pr["beta[1]"], pr["gamma[1]"])
                s0 = float(res.conditional_volatility[-1] ** 2)
            except Exception:
                if params is None:
                    continue
                s0 = float(np.nanvar(seg))
        end = min(n, i + refit)
        # filter from the start of the estimation window so the state at i reflects the new params
        s2 = _filter_garch(e[i - window:end], *params, kmap[kind], float(np.nanvar(seg)))
        out[i:end] = s2[window:]
    return pd.Series(np.sqrt(out) / 100, index=df.index)


def _vol_ratio(df, fc, n=20):
    rv = _lr(df).rolling(n, min_periods=n).std()
    return pd.DataFrame({"forecast": fc, "fc_over_rv": np.log(fc / (rv + EPS))})


GS = {"window": [1000, 2000], "refit": [250, 500]}


@feature("F.EC.GARCH", F, "GARCH(1,1) one-step conditional vol forecast; params refit every `refit` bars on trailing "
         "`window` returns (arch MLE), filtered forward with frozen params", params={"window": 1000, "refit": 500},
         space=GS, requires=("arch",), complexity="O(n + (n/refit)·fit)")
def f_garch(df, window=1000, refit=500):
    return _vol_ratio(df, _garch_family(df, "garch", window, refit))


@feature("F.EC.GJR_GARCH", F, "GJR-GARCH(1,1,1) forecast (leverage term gamma on negative shocks), same schedule",
         params={"window": 1000, "refit": 500}, space=GS, requires=("arch",))
def f_gjr(df, window=1000, refit=500):
    return _vol_ratio(df, _garch_family(df, "gjr", window, refit))


@feature("F.EC.EGARCH", F, "EGARCH(1,1,1) log-variance forecast, same schedule", params={"window": 1000, "refit": 500},
         space=GS, requires=("arch",))
def f_egarch(df, window=1000, refit=500):
    return _vol_ratio(df, _garch_family(df, "egarch", window, refit))


# ----------------------------------------------------------------------------- Kalman
@njit(cache=True)
def _kalman_level(y, q, r):
    n = len(y)
    lvl = np.full(n, np.nan); z = np.full(n, np.nan)
    x = y[0]; P = 1.0
    for t in range(n):
        if np.isnan(y[t]):
            continue
        P = P + q
        S = P + r
        v = y[t] - x
        z[t] = v / np.sqrt(S)
        K = P / S
        x = x + K * v
        P = (1 - K) * P
        lvl[t] = x
    return lvl, z


@feature("F.EC.KALMAN_LEVEL", F, "local-level Kalman filter on ln C (state noise q, obs noise r; q/r = ratio): filtered "
         "level deviation (ln C - level)/sigma_r and standardised innovation", params={"ratio": 0.01},
         space={"ratio": [0.001, 0.01, 0.1]})
def f_kalman_level(df, ratio=0.01):
    y = np.log(df["close"]).to_numpy(float)
    r = np.nanvar(np.diff(y[:200])) if len(y) > 10 else 1e-8  # uses only the first 200 bars (fixed, causal for t>=200)
    lvl, z = _kalman_level(y, ratio * r, r)
    dev = (y - lvl) / np.sqrt(r)
    s = pd.DataFrame({"dev": dev, "innov": z}, index=df.index)
    s.iloc[:200] = np.nan
    return s


@njit(cache=True)
def _kalman_trend(y, q1, q2, r):
    n = len(y)
    slope = np.full(n, np.nan)
    x0 = y[0]; x1 = 0.0
    P00 = 1.0; P01 = 0.0; P11 = 1.0
    for t in range(n):
        if np.isnan(y[t]):
            continue
        # predict: x0 += x1
        x0 = x0 + x1
        P00 = P00 + 2 * P01 + P11 + q1
        P01 = P01 + P11
        P11 = P11 + q2
        S = P00 + r
        v = y[t] - x0
        K0 = P00 / S; K1 = P01 / S
        x0 += K0 * v; x1 += K1 * v
        P11 = P11 - K1 * P01
        P01 = P01 - K0 * P01
        P00 = P00 - K0 * P00
        slope[t] = x1
    return slope


@feature("F.EC.KALMAN_TREND", F, "local-linear-trend Kalman filter on ln C: filtered slope state / sqrt(obs var)",
         params={"q_level": 0.01, "q_slope": 1e-4}, space={"q_level": [0.01, 0.1], "q_slope": [1e-5, 1e-4, 1e-3]})
def f_kalman_trend(df, q_level=0.01, q_slope=1e-4):
    y = np.log(df["close"]).to_numpy(float)
    r = np.nanvar(np.diff(y[:200])) if len(y) > 10 else 1e-8
    s = pd.Series(_kalman_trend(y, q_level * r, q_slope * r, r) / np.sqrt(r), index=df.index)
    s.iloc[:200] = np.nan
    return s


@njit(cache=True)
def _ou(x):
    n = len(x)
    a = x[:-1]; b = x[1:]
    am = a.mean(); bm = b.mean()
    sxx = ((a - am) ** 2).sum()
    if sxx == 0:
        return np.nan, np.nan, np.nan
    phi = ((a - am) * (b - bm)).sum() / sxx
    c = bm - phi * am
    res = b - (c + phi * a)
    if phi <= 0 or phi >= 1:
        return np.nan, np.nan, np.nan
    mu = c / (1 - phi)
    sig_eq = res.std() / np.sqrt(1 - phi * phi)
    hl = np.log(2) / -np.log(phi)
    return hl, (x[-1] - mu) / sig_eq, -np.log(phi)


@njit(cache=True)
def _ou_hl(x):
    return _ou(x)[0]


@njit(cache=True)
def _ou_z(x):
    return _ou(x)[1]


@feature("F.EC.OU", F, "Ornstein-Uhlenbeck fit by AR(1) on trailing ln C: x_{t+1} = c + phi x_t + e; theta=-ln phi, "
         "half-life ln2/theta, equilibrium z = (x_t - mu)/(sigma_e/sqrt(1-phi^2)) (NaN when phi not in (0,1))",
         params={"n": 200}, space={"n": [100, 200, 500]})
def f_ou(df, n=200):
    lp = np.log(df["close"])
    return pd.DataFrame({"half_life": rolling_apply(lp, n, _ou_hl), "z": rolling_apply(lp, n, _ou_z)})


@feature("F.EC.VARIANCE_RATIO", F, "Lo-MacKinlay variance ratio VR(q) = Var(q-bar ret)/(q Var(1-bar ret)) with "
         "homoskedastic z-statistic", params={"n": 250, "q": 5}, space={"n": [250, 500], "q": [2, 5, 10]})
def f_vr(df, n=250, q=5):
    lr = _lr(df)
    v1 = lr.rolling(n, min_periods=n).var()
    vq = lr.rolling(q, min_periods=q).sum().rolling(n, min_periods=n).var()
    vr = vq / (q * v1 + EPS)
    se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    return pd.DataFrame({"vr": vr, "z": (vr - 1) / se})


@njit(cache=True)
def _adf_t(x):
    n = len(x)
    dy = x[1:] - x[:-1]
    m = n - 2
    X = np.ones((m, 3))
    y = np.empty(m)
    for i in range(m):
        y[i] = dy[i + 1]
        X[i, 1] = x[i + 1]
        X[i, 2] = dy[i]
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    res = y - X @ beta
    s2 = (res ** 2).sum() / (m - 3)
    cov = s2 * np.linalg.inv(XtX)
    return beta[1] / np.sqrt(cov[1, 1])


@feature("F.EC.ADF", F, "rolling augmented Dickey-Fuller t-statistic (constant, 1 lag) on ln C", params={"n": 250},
         space={"n": [100, 250, 500]})
def f_adf(df, n=250):
    return rolling_apply(np.log(df["close"]), n, _adf_t)


@njit(cache=True)
def _cusum(x):
    s = (x - x.mean()) / (x.std() + 1e-12)
    c = np.cumsum(s)
    return np.abs(c).max() / np.sqrt(len(x))


@feature("F.EC.CUSUM", F, "OLS-CUSUM (mean model) structural-break statistic max_k |sum_{i<=k} z_i|/sqrt(n) of trailing returns",
         params={"n": 250}, space={"n": [100, 250, 500]})
def f_cusum(df, n=250):
    return rolling_apply(_lr(df), n, _cusum)


@njit(cache=True)
def _supf(x):
    n = len(x)
    tot = ((x - x.mean()) ** 2).sum()
    best = 0.0
    lo = int(0.15 * n); hi = int(0.85 * n)
    for k in range(lo, hi):
        a = x[:k]; b = x[k:]
        ssr = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        f = (tot - ssr) / (ssr / (n - 2))
        if f > best:
            best = f
    return best


@feature("F.EC.SUP_F", F, "Andrews sup-F statistic for a single mean break in trailing returns (15% trimming)",
         params={"n": 250}, space={"n": [100, 250, 500]}, complexity="O(n·w^2)")
def f_supf(df, n=250):
    return rolling_apply(_lr(df), n, _supf)


@njit(cache=True)
def _ljung_box(x):
    n = len(x)
    xm = x - x.mean()
    d = (xm ** 2).sum()
    q = 0.0
    for k in range(1, 11):
        r = (xm[k:] * xm[:-k]).sum() / d
        q += r * r / (n - k)
    return n * (n + 2) * q


@feature("F.EC.LJUNG_BOX", F, "Ljung-Box Q(10) of trailing returns", params={"n": 250}, space={"n": [250, 500]})
def f_lb(df, n=250):
    return rolling_apply(_lr(df), n, _ljung_box)


@njit(cache=True)
def _arch_lm(x):
    p = 5
    e2 = (x - x.mean()) ** 2
    n = len(e2) - p
    X = np.ones((n, p + 1)); y = np.empty(n)
    for i in range(n):
        y[i] = e2[i + p]
        for k in range(p):
            X[i, k + 1] = e2[i + p - 1 - k]
    beta = np.linalg.lstsq(X, y)[0]
    res = y - X @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return n * (1 - (res ** 2).sum() / ss) if ss > 0 else 0.0


@feature("F.EC.ARCH_LM", F, "Engle ARCH-LM statistic n R^2 from regressing e_t^2 on 5 lags (trailing window)",
         params={"n": 250}, space={"n": [250, 500]})
def f_arch_lm(df, n=250):
    return rolling_apply(_lr(df), n, _arch_lm)


@feature("F.EC.HAR_RV", F, "Corsi HAR: OLS of next-day RV on daily/weekly/monthly RV components (in bars), fitted on the "
         "trailing window at a fixed schedule; outputs ln(forecast RV / current daily RV)",
         params={"window_days": 250, "refit_days": 20}, space={"window_days": [250, 500]}, frequencies="M5..H4")
def f_har(df, window_days=250, refit_days=20):
    bpd = int(round(bars_per_day(df.index)))
    lr2 = _lr(df) ** 2
    rv_d = lr2.rolling(bpd, min_periods=bpd).sum()
    rv_w = rv_d.rolling(5 * bpd, min_periods=5 * bpd).mean()
    rv_m = rv_d.rolling(22 * bpd, min_periods=22 * bpd).mean()
    target = rv_d.shift(-bpd)  # next-day RV, used ONLY inside the fit window where it is fully observed
    X = np.column_stack([np.ones(len(df)), rv_d, rv_w, rv_m])
    out = np.full(len(df), np.nan)
    W = window_days * bpd
    R = refit_days * bpd
    beta = None
    for i in range(W + bpd, len(df), R):
        lo, hi = i - W, i - bpd  # target at row j needs data up to j+bpd <= i
        Xi, yi = X[lo:hi], target.to_numpy()[lo:hi]
        ok = ~np.isnan(Xi).any(1) & ~np.isnan(yi)
        if ok.sum() > 50:
            beta = np.linalg.lstsq(Xi[ok], yi[ok], rcond=None)[0]
        if beta is not None:
            end = min(len(df), i + R)
            out[i:end] = X[i:end] @ beta
    fc = pd.Series(out, index=df.index)
    return np.log(fc.clip(lower=1e-14) / (rv_d + 1e-14))
