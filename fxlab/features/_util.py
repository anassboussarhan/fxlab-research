"""Shared causal numerical primitives for the feature catalog (numba-accelerated)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

EPS = 1e-12


def as_series(x, index) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float), index=index)


def logret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(x: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing."""
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def wma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return pd.Series(_conv_causal(x.to_numpy(float), w), index=x.index)


@njit(cache=True)
def _conv_causal(x, w):
    n, m = len(x), len(w)
    out = np.full(n, np.nan)
    for i in range(m - 1, n):
        s = 0.0
        ok = True
        for j in range(m):
            v = x[i - m + 1 + j]
            if np.isnan(v):
                ok = False
                break
            s += w[j] * v
        if ok:
            out[i] = s
    return out


def conv_causal(x: pd.Series, w: np.ndarray) -> pd.Series:
    """Weighted window with weights w[0]=oldest ... w[-1]=newest."""
    return pd.Series(_conv_causal(x.to_numpy(float), np.asarray(w, float)), index=x.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n, min_periods=n).mean()
    s = x.rolling(n, min_periods=n).std()
    return (x - m) / (s + EPS)


@njit(cache=False)  # takes a jitted function as argument: numba's disk cache is unreliable for such signatures
def _rolling_apply(x, w, fn, minp):
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        cnt = 0
        for v in win:
            if not np.isnan(v):
                cnt += 1
        if cnt >= minp:
            clean = win[~np.isnan(win)]
            out[i] = fn(clean)
    return out


def rolling_apply(x: pd.Series, w: int, fn, minp: int | None = None, step: int = 1) -> pd.Series:
    """Apply a numba-jitted scalar function to each trailing window.

    step>1 evaluates every ``step`` bars and carries the last value forward
    (the carried value was computed from data available at its own bar, so
    this remains causal).
    """
    arr = x.to_numpy(float)
    mp = w if minp is None else minp
    if step == 1:
        return pd.Series(_rolling_apply(arr, w, fn, mp), index=x.index)
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr), step):
        win = arr[i - w + 1:i + 1]
        win = win[~np.isnan(win)]
        if len(win) >= mp:
            out[i] = fn(win)
    return pd.Series(out, index=x.index).ffill()


def rolling_apply_py(x: pd.Series, w: int, fn, minp: int | None = None, step: int = 1) -> pd.Series:
    """Pure-python trailing-window apply for functions that cannot be jitted
    (e.g. pywt/scipy). Same causality contract as rolling_apply."""
    arr = x.to_numpy(float)
    mp = w if minp is None else minp
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr), step):
        win = arr[i - w + 1:i + 1]
        win = win[~np.isnan(win)]
        if len(win) >= mp:
            out[i] = fn(win)
    s = pd.Series(out, index=x.index)
    return s.ffill() if step > 1 else s


@njit(cache=True)
def _rolling_ols_slope(y, w):
    n = len(y)
    out = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    xs = np.arange(w).astype(np.float64)
    xm = xs.mean()
    sxx = ((xs - xm) ** 2).sum()
    for i in range(w - 1, n):
        win = y[i - w + 1:i + 1]
        if np.isnan(win).any():
            continue
        ym = win.mean()
        sxy = ((xs - xm) * (win - ym)).sum()
        b = sxy / sxx
        out[i] = b
        syy = ((win - ym) ** 2).sum()
        r2[i] = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return out, r2


def rolling_slope(y: pd.Series, w: int) -> tuple[pd.Series, pd.Series]:
    b, r2 = _rolling_ols_slope(y.to_numpy(float), w)
    return pd.Series(b, index=y.index), pd.Series(r2, index=y.index)


@njit(cache=True)
def _causal_pivots(h, l, k):
    """Swing highs/lows confirmed k bars after the extreme (no lookahead):
    at bar i we may declare bar i-k a swing high if it is the max of
    [i-2k, i]. Returns arrays (level, index-of-pivot) carried forward."""
    n = len(h)
    sh_lvl = np.full(n, np.nan)
    sl_lvl = np.full(n, np.nan)
    sh_idx = np.full(n, -1)
    sl_idx = np.full(n, -1)
    last_h, last_l, last_hi, last_li = np.nan, np.nan, -1, -1
    for i in range(n):
        c = i - k
        if c - k >= 0:
            is_h = True
            is_l = True
            for j in range(c - k, i + 1):
                if h[j] > h[c]:
                    is_h = False
                if l[j] < l[c]:
                    is_l = False
            if is_h:
                last_h, last_hi = h[c], c
            if is_l:
                last_l, last_li = l[c], c
        sh_lvl[i], sl_lvl[i], sh_idx[i], sl_idx[i] = last_h, last_l, last_hi, last_li
    return sh_lvl, sl_lvl, sh_idx, sl_idx


def causal_pivots(df: pd.DataFrame, k: int = 3):
    return _causal_pivots(df["high"].to_numpy(float), df["low"].to_numpy(float), k)


@njit(cache=True)
def _zigzag(close, thr):
    """Causal percentage zigzag. A pivot is confirmed only when price has
    reversed by ``thr`` (fraction) from the running extreme. Returns for every
    bar the last 4 confirmed pivot prices/indices (oldest..newest) and the
    current direction (+1 up leg, -1 down leg)."""
    n = len(close)
    piv_p = np.full((n, 4), np.nan)
    piv_i = np.full((n, 4), -1)
    direction = np.zeros(n)
    pp = np.full(4, np.nan)
    pi = np.full(4, -1)
    d = 0
    ext = close[0]
    ext_i = 0
    for i in range(n):
        p = close[i]
        if np.isnan(p):
            piv_p[i] = pp
            piv_i[i] = pi
            direction[i] = d
            continue
        if np.isnan(ext):
            ext, ext_i = p, i
        if d >= 0:
            if p > ext:
                ext, ext_i = p, i
            elif p < ext * (1 - thr):
                # confirm swing high at ext
                pp[:3] = pp[1:].copy(); pi[:3] = pi[1:].copy()
                pp[3], pi[3] = ext, ext_i
                d = -1
                ext, ext_i = p, i
        if d < 0:
            if p < ext:
                ext, ext_i = p, i
            elif p > ext * (1 + thr):
                pp[:3] = pp[1:].copy(); pi[:3] = pi[1:].copy()
                pp[3], pi[3] = ext, ext_i
                d = 1
                ext, ext_i = p, i
        piv_p[i] = pp
        piv_i[i] = pi
        direction[i] = d
    return piv_p, piv_i, direction


def zigzag(close: pd.Series, thr: float):
    return _zigzag(close.to_numpy(float), float(thr))


def bars_per_day(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 1440.0
    dt = pd.Series(index).diff().dropna().median()
    return max(1.0, pd.Timedelta(days=1) / dt)


def periods_per_year(index: pd.DatetimeIndex) -> float:
    return bars_per_day(index) * 260.0
