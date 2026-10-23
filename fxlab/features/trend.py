"""7.2 Trend analysis. Moving-average features are reported as ln(C_t / MA_t)
(scale-free); each MA entry is a genuinely different weighting/adaptation rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import (EPS, ema, sma, wma, rma, conv_causal, atr, true_range,
                                  rolling_slope, as_series, rolling_apply)

F = "trend"
OHLC = ("open", "high", "low", "close")
NS = {"n": [10, 20, 50, 100, 200]}


def _rel(c, ma):
    return np.log(c / ma)


@feature("F.TR.SMA", F, "ln(C_t / SMA_n), SMA_n = (1/n) sum_{i<n} C_{t-i}", params={"n": 50}, space=NS)
def f_sma(df, n=50):
    return _rel(df["close"], sma(df["close"], n))


@feature("F.TR.EMA", F, "ln(C/EMA_n), EMA_t = a C_t + (1-a) EMA_{t-1}, a=2/(n+1)", params={"n": 50}, space=NS)
def f_ema(df, n=50):
    return _rel(df["close"], ema(df["close"], n))


@feature("F.TR.WMA", F, "ln(C/WMA_n), linearly increasing weights 1..n", params={"n": 50}, space=NS)
def f_wma(df, n=50):
    return _rel(df["close"], wma(df["close"], n))


@feature("F.TR.TRIMA", F, "ln(C/TRIMA_n), TRIMA = SMA_{ceil(n/2)}(SMA_{floor(n/2)+1}) (triangular weights)",
         params={"n": 50}, space=NS)
def f_trima(df, n=50):
    a = n // 2 + 1
    b = (n + 1) // 2
    return _rel(df["close"], sma(sma(df["close"], a), b))


@feature("F.TR.HMA", F, "ln(C/HMA_n), HMA = WMA_{sqrt n}(2 WMA_{n/2} - WMA_n)", params={"n": 50}, space=NS)
def f_hma(df, n=50):
    c = df["close"]
    raw = 2 * wma(c, max(2, n // 2)) - wma(c, n)
    return _rel(c, wma(raw, max(2, int(np.sqrt(n)))))


@feature("F.TR.DEMA", F, "ln(C/DEMA), DEMA = 2 E1 - E2, E2 = EMA(E1)", params={"n": 50}, space=NS)
def f_dema(df, n=50):
    e1 = ema(df["close"], n)
    return _rel(df["close"], 2 * e1 - ema(e1, n))


@feature("F.TR.TEMA", F, "ln(C/TEMA), TEMA = 3E1 - 3E2 + E3", params={"n": 50}, space=NS)
def f_tema(df, n=50):
    e1 = ema(df["close"], n)
    e2 = ema(e1, n)
    return _rel(df["close"], 3 * e1 - 3 * e2 + ema(e2, n))


@njit(cache=True)
def _kama(c, n, fast, slow):
    out = np.full(len(c), np.nan)
    fsc, ssc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    for i in range(n, len(c)):
        ch = abs(c[i] - c[i - n])
        vol = 0.0
        for j in range(i - n + 1, i + 1):
            vol += abs(c[j] - c[j - 1])
        er = ch / vol if vol > 0 else 0.0
        sc = (er * (fsc - ssc) + ssc) ** 2
        prev = out[i - 1] if not np.isnan(out[i - 1]) else c[i - 1]
        out[i] = prev + sc * (c[i] - prev)
    return out


@feature("F.TR.KAMA", F, "Kaufman adaptive MA: sc=(ER(2/(f+1)-2/(s+1))+2/(s+1))^2, K_t=K_{t-1}+sc(C_t-K_{t-1})",
         params={"n": 10, "fast": 2, "slow": 30}, space={"n": [10, 20], "fast": [2, 3], "slow": [20, 30, 50]})
def f_kama(df, n=10, fast=2, slow=30):
    c = df["close"].to_numpy(float)
    return _rel(df["close"], as_series(_kama(c, n, fast, slow), df.index))


@njit(cache=True)
def _vidya(c, n, m):
    out = np.full(len(c), np.nan)
    a = 2.0 / (n + 1)
    for i in range(m, len(c)):
        up = 0.0
        dn = 0.0
        for j in range(i - m + 1, i + 1):
            d = c[j] - c[j - 1]
            if d > 0:
                up += d
            else:
                dn -= d
        cmo = abs(up - dn) / (up + dn) if up + dn > 0 else 0.0
        prev = out[i - 1] if not np.isnan(out[i - 1]) else c[i - 1]
        out[i] = prev + a * cmo * (c[i] - prev)
    return out


@feature("F.TR.VIDYA", F, "Chande VIDYA: V_t = V_{t-1} + (2/(n+1))|CMO_m|(C_t - V_{t-1})",
         params={"n": 20, "m": 9}, space={"n": [10, 20, 50], "m": [9, 14]})
def f_vidya(df, n=20, m=9):
    return _rel(df["close"], as_series(_vidya(df["close"].to_numpy(float), n, m), df.index))


@feature("F.TR.ALMA", F, "Arnaud Legoux MA: Gaussian weights w_i=exp(-(i-off(n-1))^2/(2(n/sigma)^2))",
         params={"n": 50, "offset": 0.85, "sigma": 6.0}, space={"n": [20, 50, 100], "offset": [0.85, 0.9], "sigma": [6.0]})
def f_alma(df, n=50, offset=0.85, sigma=6.0):
    i = np.arange(n)
    m = offset * (n - 1)
    s = n / sigma
    w = np.exp(-((i - m) ** 2) / (2 * s * s))
    return _rel(df["close"], conv_causal(df["close"], w / w.sum()))


@feature("F.TR.ZLEMA", F, "zero-lag EMA: EMA_n(C_t + (C_t - C_{t-L})), L=(n-1)/2", params={"n": 50}, space=NS)
def f_zlema(df, n=50):
    c = df["close"]
    lag = (n - 1) // 2
    return _rel(c, ema(c + (c - c.shift(lag)), n))


@feature("F.TR.T3", F, "Tillson T3: GD applied 3x, GD(x)=EMA(x)(1+v)-EMA(EMA(x))v, v=0.7",
         params={"n": 10, "v": 0.7}, space={"n": [5, 10, 20], "v": [0.5, 0.7, 0.9]})
def f_t3(df, n=10, v=0.7):
    def gd(x):
        e = ema(x, n)
        return e * (1 + v) - ema(e, n) * v
    return _rel(df["close"], gd(gd(gd(df["close"]))))


@njit(cache=True)
def _frama(h, l, c, n):
    out = np.full(len(c), np.nan)
    half = n // 2
    for i in range(n - 1, len(c)):
        h1 = h[i - n + 1:i - half + 1].max(); l1 = l[i - n + 1:i - half + 1].min()
        h2 = h[i - half + 1:i + 1].max(); l2 = l[i - half + 1:i + 1].min()
        h3 = h[i - n + 1:i + 1].max(); l3 = l[i - n + 1:i + 1].min()
        n1 = (h1 - l1) / half; n2 = (h2 - l2) / half; n3 = (h3 - l3) / n
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            d = 1.0
        a = np.exp(-4.6 * (d - 1.0))
        a = min(1.0, max(0.01, a))
        prev = out[i - 1] if not np.isnan(out[i - 1]) else c[i]
        out[i] = a * c[i] + (1 - a) * prev
    return out


@feature("F.TR.FRAMA", F, "Ehlers fractal adaptive MA: alpha=exp(-4.6(D-1)), D=(ln(N1+N2)-ln N3)/ln2",
         params={"n": 16}, space={"n": [16, 32, 64]}, inputs=OHLC)
def f_frama(df, n=16):
    n = n + (n % 2)
    return _rel(df["close"], as_series(_frama(df["high"].to_numpy(float), df["low"].to_numpy(float),
                                              df["close"].to_numpy(float), n), df.index))


@njit(cache=True)
def _mcginley(c, n, k):
    out = np.full(len(c), np.nan)
    md = c[0]
    for i in range(len(c)):
        if np.isnan(c[i]):
            continue
        if np.isnan(md):
            md = c[i]
        md = md + (c[i] - md) / (k * n * (c[i] / md) ** 4)
        if i >= n:
            out[i] = md
    return out


@feature("F.TR.MCGINLEY", F, "McGinley dynamic MD_t = MD_{t-1} + (C-MD)/(k n (C/MD)^4)",
         params={"n": 20, "k": 0.6}, space={"n": [10, 20, 50]})
def f_mcginley(df, n=20, k=0.6):
    return _rel(df["close"], as_series(_mcginley(df["close"].to_numpy(float), n, k), df.index))


@njit(cache=True)
def _laguerre(c, g):
    n = len(c)
    out = np.full(n, np.nan)
    l0 = l1 = l2 = l3 = c[0]
    for i in range(n):
        p0, p1, p2 = l0, l1, l2
        l0 = (1 - g) * c[i] + g * p0
        l1 = -g * l0 + p0 + g * p1
        l2 = -g * l1 + p1 + g * p2
        l3 = -g * l2 + p2 + g * l3
        if i >= 4:
            out[i] = (l0 + 2 * l1 + 2 * l2 + l3) / 6.0
    return out


@feature("F.TR.LAGUERRE", F, "Ehlers 4-element Laguerre filter with damping gamma",
         params={"gamma": 0.8}, space={"gamma": [0.5, 0.7, 0.8, 0.9]})
def f_laguerre(df, gamma=0.8):
    return _rel(df["close"], as_series(_laguerre(df["close"].to_numpy(float), gamma), df.index))


@njit(cache=True)
def _mama(price, fast, slow):
    n = len(price)
    mama = np.full(n, np.nan); fama = np.full(n, np.nan)
    smooth = np.zeros(n); det = np.zeros(n); i1 = np.zeros(n); q1 = np.zeros(n)
    i2 = np.zeros(n); q2 = np.zeros(n); re = np.zeros(n); im = np.zeros(n)
    period = np.zeros(n); phase = np.zeros(n)
    m_prev = price[0]; f_prev = price[0]
    for t in range(6, n):
        smooth[t] = (4 * price[t] + 3 * price[t - 1] + 2 * price[t - 2] + price[t - 3]) / 10.0
        adj = 0.075 * period[t - 1] + 0.54
        det[t] = (0.0962 * smooth[t] + 0.5769 * smooth[t - 2] - 0.5769 * smooth[t - 4] - 0.0962 * smooth[t - 6]) * adj
        q1[t] = (0.0962 * det[t] + 0.5769 * det[t - 2] - 0.5769 * det[t - 4] - 0.0962 * det[t - 6]) * adj
        i1[t] = det[t - 3]
        ji = (0.0962 * i1[t] + 0.5769 * i1[t - 2] - 0.5769 * i1[t - 4] - 0.0962 * i1[t - 6]) * adj
        jq = (0.0962 * q1[t] + 0.5769 * q1[t - 2] - 0.5769 * q1[t - 4] - 0.0962 * q1[t - 6]) * adj
        i2[t] = i1[t] - jq; q2[t] = q1[t] + ji
        i2[t] = 0.2 * i2[t] + 0.8 * i2[t - 1]; q2[t] = 0.2 * q2[t] + 0.8 * q2[t - 1]
        re[t] = i2[t] * i2[t - 1] + q2[t] * q2[t - 1]; im[t] = i2[t] * q2[t - 1] - q2[t] * i2[t - 1]
        re[t] = 0.2 * re[t] + 0.8 * re[t - 1]; im[t] = 0.2 * im[t] + 0.8 * im[t - 1]
        p = period[t - 1]
        if im[t] != 0 and re[t] != 0:
            p = 2 * np.pi / np.arctan(im[t] / re[t])
        p = min(p, 1.5 * period[t - 1]) if period[t - 1] > 0 else p
        p = max(p, 0.67 * period[t - 1])
        p = min(max(p, 6.0), 50.0)
        period[t] = 0.2 * p + 0.8 * period[t - 1]
        phase[t] = np.degrees(np.arctan(q1[t] / i1[t])) if i1[t] != 0 else phase[t - 1]
        dphase = max(phase[t - 1] - phase[t], 1.0)
        alpha = min(max(fast / dphase, slow), fast)
        m_prev = alpha * price[t] + (1 - alpha) * m_prev
        f_prev = 0.5 * alpha * m_prev + (1 - 0.5 * alpha) * f_prev
        if t >= 40:
            mama[t] = m_prev; fama[t] = f_prev
    return mama, fama


@feature("F.TR.MAMA", F, "Ehlers MESA adaptive MA (homodyne phase-rate alpha in [slow,fast]); outputs ln(C/MAMA), ln(MAMA/FAMA)",
         params={"fast": 0.5, "slow": 0.05}, space={"fast": [0.3, 0.5], "slow": [0.02, 0.05]})
def f_mama(df, fast=0.5, slow=0.05):
    m, f = _mama(df["close"].to_numpy(float), fast, slow)
    m, f = as_series(m, df.index), as_series(f, df.index)
    return pd.DataFrame({"c_mama": _rel(df["close"], m), "mama_fama": np.log(m / f)})


@feature("F.TR.REG_SLOPE", F, "OLS slope of ln C on time over n bars, scaled by 1/sigma_n(l) (t-like) plus R^2",
         params={"n": 50}, space=NS, complexity="O(n·w)")
def f_reg_slope(df, n=50):
    lc = np.log(df["close"])
    b, r2 = rolling_slope(lc, n)
    vol = lc.diff().rolling(n, min_periods=n).std()
    return pd.DataFrame({"slope": b / (vol + EPS), "r2": r2})


@njit(cache=True)
def _reg_channel(y, w):
    n = len(y)
    pos = np.full(n, np.nan)
    xs = np.arange(w).astype(np.float64)
    xm = xs.mean()
    sxx = ((xs - xm) ** 2).sum()
    for i in range(w - 1, n):
        win = y[i - w + 1:i + 1]
        if np.isnan(win).any():
            continue
        ym = win.mean()
        b = ((xs - xm) * (win - ym)).sum() / sxx
        a = ym - b * xm
        res = win - (a + b * xs)
        sd = np.sqrt((res ** 2).sum() / (w - 2))
        pos[i] = res[-1] / sd if sd > 0 else 0.0
    return pos


@feature("F.TR.REG_CHANNEL", F, "regression channel position: residual of C_t vs OLS line over n / residual std",
         params={"n": 50}, space=NS, complexity="O(n·w)")
def f_reg_channel(df, n=50):
    return as_series(_reg_channel(np.log(df["close"]).to_numpy(float), n), df.index)


@njit(cache=True)
def _quad_curv(y, w):
    n = len(y)
    out = np.full(n, np.nan)
    x = (np.arange(w) - (w - 1) / 2.0) / w
    X = np.empty((w, 3))
    X[:, 0] = 1.0; X[:, 1] = x; X[:, 2] = x * x
    XtX_inv = np.linalg.inv(X.T @ X)
    P = XtX_inv @ X.T
    for i in range(w - 1, n):
        win = y[i - w + 1:i + 1]
        if np.isnan(win).any():
            continue
        out[i] = (P[2] * win).sum()
    return out


@feature("F.TR.CURVATURE", F, "trend curvature: quadratic coefficient c of OLS fit ln C = a + b x + c x^2 over n bars (x scaled to [-.5,.5])",
         params={"n": 50}, space=NS, complexity="O(n·w)")
def f_curvature(df, n=50):
    lc = np.log(df["close"])
    vol = lc.diff().rolling(n, min_periods=n).std()
    return as_series(_quad_curv(lc.to_numpy(float), n), df.index) / (vol * np.sqrt(n) + EPS)


@feature("F.TR.ADX", F, "Wilder ADX/DMI: +DI, -DI, ADX = RMA_n(|+DI - -DI|/(+DI + -DI))",
         params={"n": 14}, space={"n": [14, 20, 28]}, inputs=OHLC)
def f_adx(df, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = rma(true_range(df), n)
    pdi = 100 * rma(pdm, n) / (tr + EPS)
    mdi = 100 * rma(mdm, n) / (tr + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    return pd.DataFrame({"pdi": pdi, "mdi": mdi, "adx": rma(dx, n)})


@feature("F.TR.AROON", F, "Aroon up/down = 100 (n - bars since n-high/low)/n; oscillator = up - down",
         params={"n": 25}, space={"n": [14, 25, 50]}, inputs=OHLC)
def f_aroon(df, n=25):
    hi = df["high"].rolling(n + 1, min_periods=n + 1).apply(lambda x: n - (len(x) - 1 - np.argmax(x)), raw=True)
    lo = df["low"].rolling(n + 1, min_periods=n + 1).apply(lambda x: n - (len(x) - 1 - np.argmin(x)), raw=True)
    return pd.DataFrame({"up": 100 * hi / n, "down": 100 * lo / n, "osc": 100 * (hi - lo) / n})


@njit(cache=True)
def _supertrend(h, l, c, a, mult):
    n = len(c)
    st = np.full(n, np.nan)
    d = np.zeros(n)
    ub_prev = np.nan; lb_prev = np.nan; dir_prev = 1.0
    for i in range(n):
        if np.isnan(a[i]):
            continue
        mid = (h[i] + l[i]) / 2
        ub = mid + mult * a[i]; lb = mid - mult * a[i]
        if not np.isnan(ub_prev):
            if not (ub < ub_prev or c[i - 1] > ub_prev):
                ub = ub_prev
            if not (lb > lb_prev or c[i - 1] < lb_prev):
                lb = lb_prev
        if np.isnan(ub_prev):
            dr = 1.0
        elif dir_prev > 0 and c[i] < lb_prev:
            dr = -1.0
        elif dir_prev < 0 and c[i] > ub_prev:
            dr = 1.0
        else:
            dr = dir_prev
        st[i] = lb if dr > 0 else ub
        d[i] = dr
        ub_prev, lb_prev, dir_prev = ub, lb, dr
    return st, d


@feature("F.TR.SUPERTREND", F, "Supertrend: ATR bands around (H+L)/2 with ratchet; outputs direction and (C-ST)/ATR",
         params={"n": 10, "mult": 3.0}, space={"n": [10, 14], "mult": [2.0, 3.0, 4.0]}, inputs=OHLC)
def f_supertrend(df, n=10, mult=3.0):
    a = atr(df, n)
    st, d = _supertrend(df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float),
                        a.to_numpy(float), mult)
    st = as_series(st, df.index)
    return pd.DataFrame({"dir": as_series(d, df.index).where(st.notna()), "dist": (df["close"] - st) / (a + EPS)})


@njit(cache=True)
def _psar(h, l, af0, afstep, afmax):
    n = len(h)
    out = np.full(n, np.nan)
    d = np.zeros(n)
    if n < 2:
        return out, d
    up = True
    sar = l[0]; ep = h[0]; af = af0
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < sar:
                up = False; sar = ep; ep = l[i]; af = af0
            elif h[i] > ep:
                ep = h[i]; af = min(af + afstep, afmax)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:
                up = True; sar = ep; ep = h[i]; af = af0
            elif l[i] < ep:
                ep = l[i]; af = min(af + afstep, afmax)
        out[i] = sar
        d[i] = 1.0 if up else -1.0
    return out, d


@feature("F.TR.PSAR", F, "Wilder parabolic SAR (af0, step, max); outputs direction and (C-SAR)/ATR",
         params={"af0": 0.02, "step": 0.02, "afmax": 0.2}, space={"step": [0.01, 0.02, 0.03], "afmax": [0.1, 0.2]},
         inputs=OHLC)
def f_psar(df, af0=0.02, step=0.02, afmax=0.2):
    s, d = _psar(df["high"].to_numpy(float), df["low"].to_numpy(float), af0, step, afmax)
    a = atr(df, 14)
    return pd.DataFrame({"dir": as_series(d, df.index), "dist": (df["close"] - as_series(s, df.index)) / (a + EPS)})


@feature("F.TR.ICHIMOKU", F, "Ichimoku (causal parts only; chikou excluded as it uses future): "
         "tenkan=(maxH+minL)/2 over a, kijun over b, cloud = spans shifted forward b bars (i.e. spans computed b bars ago)",
         params={"a": 9, "b": 26, "c": 52}, space={"a": [9], "b": [26], "c": [52]}, inputs=OHLC)
def f_ichimoku(df, a=9, b=26, c=52):
    mid = lambda n: (df["high"].rolling(n, min_periods=n).max() + df["low"].rolling(n, min_periods=n).min()) / 2
    tenkan, kijun = mid(a), mid(b)
    span_a = ((tenkan + kijun) / 2).shift(b)   # value plotted at t was computed at t-b
    span_b = mid(c).shift(b)
    a_ = atr(df, 14)
    top = np.maximum(span_a, span_b); bot = np.minimum(span_a, span_b)
    return pd.DataFrame({"tk": (tenkan - kijun) / (a_ + EPS),
                         "c_kijun": (df["close"] - kijun) / (a_ + EPS),
                         "cloud_pos": np.where(df["close"] > top, (df["close"] - top) / (a_ + EPS),
                                               np.where(df["close"] < bot, (df["close"] - bot) / (a_ + EPS), 0.0)),
                         "cloud_thick": (span_a - span_b) / (a_ + EPS)}, index=df.index)


@feature("F.TR.VORTEX", F, "Vortex VI+ = sum|H_t - L_{t-1}|/sum TR, VI- = sum|L_t - H_{t-1}|/sum TR",
         params={"n": 14}, space={"n": [14, 21, 28]}, inputs=OHLC)
def f_vortex(df, n=14):
    vp = (df["high"] - df["low"].shift(1)).abs().rolling(n, min_periods=n).sum()
    vm = (df["low"] - df["high"].shift(1)).abs().rolling(n, min_periods=n).sum()
    tr = true_range(df).rolling(n, min_periods=n).sum()
    return pd.DataFrame({"vip": vp / (tr + EPS), "vim": vm / (tr + EPS), "diff": (vp - vm) / (tr + EPS)})


@feature("F.TR.RIBBON", F, "MA ribbon alignment: mean sign of consecutive EMA differences over periods (8..89 Fibonacci) "
         "and ribbon dispersion std(ln EMA_k)/ATR%", params={}, space={})
def f_ribbon(df):
    ps = [8, 13, 21, 34, 55, 89]
    es = [np.log(ema(df["close"], p)) for p in ps]
    E = pd.concat(es, axis=1)
    align = sum(np.sign(es[i] - es[i + 1]) for i in range(len(es) - 1)) / (len(es) - 1)
    vol = np.log(df["close"]).diff().rolling(89, min_periods=89).std()
    return pd.DataFrame({"align": align, "disp": E.std(axis=1) / (vol + EPS)})


@njit(cache=True)
def _mann_kendall(x):
    n = len(x)
    s = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = x[j] - x[i]
            s += 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        return (s - 1) / np.sqrt(var)
    elif s < 0:
        return (s + 1) / np.sqrt(var)
    return 0.0


@feature("F.TR.MANN_KENDALL", F, "rolling Mann-Kendall trend Z statistic of C over n bars (non-parametric trend)",
         params={"n": 30}, space={"n": [20, 30, 60]}, complexity="O(n·w^2)")
def f_mann_kendall(df, n=30):
    return rolling_apply(df["close"], n, _mann_kendall)
