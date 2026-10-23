"""7.13 Cross-asset relationships. Multi-asset features receive ``panel``: a
DataFrame of mid closes for other instruments aligned to the target's index
(NaN where an instrument had no bar), and ``symbol``: the target's name.
Explanatory instruments are never traded by these features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, rolling_apply
from fxlab.data.instruments import CURRENCIES

F = "cross_asset"
KW = dict(multi_asset=True, instruments="panel")


def _rets(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log(panel).diff()


def _with_target(df, panel, symbol):
    p = panel.drop(columns=[symbol], errors="ignore").copy()
    p[symbol] = df["close"]
    return p


def mp(n):
    return max(2, int(0.8 * n))


def _clean_window(W: np.ndarray, max_nan_share: float = 0.2) -> np.ndarray:
    """Drop instruments (columns) with too many gaps inside the window, then incomplete rows."""
    keep = np.isnan(W).mean(0) <= max_nan_share
    W = W[:, keep]
    return W[~np.isnan(W).any(1)] if W.shape[1] else W


def currency_strength(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per-currency strength: mean over pairs containing the currency of signed n-bar log return."""
    r = np.log(panel) - np.log(panel.shift(n))
    out = {}
    for ccy in CURRENCIES + ["XAU", "XAG"]:
        cols = []
        for s in panel.columns:
            if s[:3] == ccy:
                cols.append(r[s])
            elif s[3:] == ccy:
                cols.append(-r[s])
        if cols:
            m = pd.concat(cols, axis=1)
            out[ccy] = m.mean(axis=1).where(m.notna().sum(axis=1) >= max(1, (m.shape[1] + 1) // 2))
    return pd.DataFrame(out)


@feature("F.CA.CURRENCY_STRENGTH", F, "currency strength S_c = mean over pairs (c as base: +r, as quote: -r) of n-bar log "
         "returns; outputs S_base, S_quote and S_base - S_quote for the target", params={"n": 20},
         space={"n": [5, 20, 60]}, **KW)
def f_strength(df, panel, symbol, n=20):
    S = currency_strength(_with_target(df, panel, symbol), n)
    b, q = symbol[:3], symbol[3:]
    sb = S[b] if b in S else pd.Series(np.nan, index=df.index)
    sq = S[q] if q in S else pd.Series(np.nan, index=df.index)
    return pd.DataFrame({"base": sb, "quote": sq, "diff": sb - sq}, index=df.index)


def _triangle(panel, symbol):
    b, q = symbol[:3], symbol[3:]
    cols = set(panel.columns)
    for z in CURRENCIES:
        if z in (b, q):
            continue
        # implied ln(b/q) = ln(b/z) - ln(q/z)
        legs = []
        for x in (b, q):
            if f"{x}{z}" in cols:
                legs.append((np.log(panel[f"{x}{z}"]), 1))
            elif f"{z}{x}" in cols:
                legs.append((-np.log(panel[f"{z}{x}"]), 1))
            else:
                legs = None
                break
        if legs:
            return legs[0][0] - legs[1][0], z
    return None, None


@feature("F.CA.TRIANGULAR", F, "triangular parity deviation d = ln C_XY - (ln XZ - ln YZ) in bps for the first available "
         "third currency Z, and its z-score over n", params={"n": 200}, space={"n": [100, 200, 1000]}, **KW)
def f_triangular(df, panel, symbol, n=200):
    implied, _ = _triangle(panel, symbol)
    if implied is None:
        return pd.DataFrame({"bps": np.nan, "z": np.nan}, index=df.index)
    d = 1e4 * (np.log(df["close"]) - implied)
    return pd.DataFrame({"bps": d, "z": (d - d.rolling(n, min_periods=mp(n)).mean()) / (d.rolling(n, min_periods=mp(n)).std() + EPS)})


def _ref(panel, symbol, ref):
    if ref in panel.columns and ref != symbol:
        return panel[ref]
    for alt in ("XAUUSD", "EURUSD", "USDJPY", "GBPUSD"):
        if alt in panel.columns and alt != symbol:
            return panel[alt]
    return None


@feature("F.CA.ROLLING_CORR", F, "rolling Pearson correlation of target returns with a reference instrument's returns",
         params={"ref": "XAUUSD", "n": 100}, space={"ref": ["XAUUSD", "EURUSD", "USDJPY"], "n": [50, 100, 500]}, **KW)
def f_corr(df, panel, symbol, ref="XAUUSD", n=100):
    r = _ref(panel, symbol, ref)
    if r is None:
        return pd.Series(np.nan, index=df.index)
    a, b = np.log(df["close"]).diff(), np.log(r).diff()
    return a.rolling(n, min_periods=mp(n)).corr(b)


@feature("F.CA.LEAD_LAG", F, "lead-lag: max over k=1..L of corr(r_target,t , r_ref,t-k) over n bars, the arg-max lag, "
         "and the contemporaneous correlation for contrast", params={"ref": "EURUSD", "n": 200, "L": 5},
         space={"n": [200, 500], "L": [3, 5, 10]}, **KW)
def f_lead_lag(df, panel, symbol, ref="EURUSD", n=200, L=5):
    r = _ref(panel, symbol, ref)
    if r is None:
        return pd.DataFrame({"max_corr": np.nan, "lag": np.nan, "corr0": np.nan}, index=df.index)
    a, b = np.log(df["close"]).diff(), np.log(r).diff()
    cs = pd.concat([a.rolling(n, min_periods=mp(n)).corr(b.shift(k)) for k in range(1, L + 1)], axis=1)
    absmax = cs.abs().to_numpy()
    idx = np.where(np.isnan(absmax).all(1), -1, np.nanargmax(np.nan_to_num(absmax, nan=-1), axis=1))
    mc = np.where(idx >= 0, cs.to_numpy()[np.arange(len(cs)), np.clip(idx, 0, None)], np.nan)
    return pd.DataFrame({"max_corr": mc, "lag": np.where(idx >= 0, idx + 1, np.nan),
                         "corr0": a.rolling(n, min_periods=mp(n)).corr(b)}, index=df.index)


def usd_factor(panel: pd.DataFrame) -> pd.Series:
    r = _rets(panel)
    cols = []
    for s in panel.columns:
        if s.endswith("USD") and s[:3] in CURRENCIES:
            cols.append(-r[s])
        elif s.startswith("USD") and s[3:] in CURRENCIES:
            cols.append(r[s])
    return pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(np.nan, index=panel.index)


@feature("F.CA.USD_BETA", F, "rolling OLS beta of target returns on the USD factor (mean USD-signed return across USD pairs)",
         params={"n": 250}, space={"n": [100, 250, 1000]}, **KW)
def f_usd_beta(df, panel, symbol, n=250):
    f = usd_factor(panel.drop(columns=[symbol], errors="ignore"))
    a = np.log(df["close"]).diff()
    return a.rolling(n, min_periods=mp(n)).cov(f) / (f.rolling(n, min_periods=mp(n)).var() + EPS)


@feature("F.CA.RV_SPREAD_Z", F, "relative-value spread z-score of ln(C_target / C_ref) over n (e.g. gold/silver ratio)",
         params={"ref": "XAGUSD", "n": 500}, space={"ref": ["XAGUSD", "XAUUSD", "EURUSD"], "n": [200, 500, 2000]}, **KW)
def f_rv_spread(df, panel, symbol, ref="XAGUSD", n=500):
    r = _ref(panel, symbol, ref)
    if r is None:
        return pd.Series(np.nan, index=df.index)
    s = np.log(df["close"]) - np.log(r)
    return (s - s.rolling(n, min_periods=mp(n)).mean()) / (s.rolling(n, min_periods=mp(n)).std() + EPS)


@njit(cache=True)
def _eg(y, x):
    n = len(y)
    xm, ym = x.mean(), y.mean()
    b = ((x - xm) * (y - ym)).sum() / (((x - xm) ** 2).sum() + 1e-300)
    a = ym - b * xm
    e = y - a - b * x
    # ADF(0) on residual
    de = e[1:] - e[:-1]
    el = e[:-1]
    g = (el * de).sum() / ((el ** 2).sum() + 1e-300)
    res = de - g * el
    se = np.sqrt((res ** 2).sum() / (n - 2) / ((el ** 2).sum() + 1e-300))
    return e[-1] / (e.std() + 1e-300), g / se


@feature("F.CA.COINTEGRATION", F, "rolling Engle-Granger: OLS ln C_target = a + b ln C_ref on trailing n; residual z-score and "
         "Dickey-Fuller t on the residual", params={"ref": "EURUSD", "n": 500}, space={"n": [250, 500, 1000]}, **KW)
def f_coint(df, panel, symbol, ref="EURUSD", n=500):
    r = _ref(panel, symbol, ref)
    if r is None:
        return pd.DataFrame({"z": np.nan, "adf_t": np.nan}, index=df.index)
    y, x = np.log(df["close"]).to_numpy(float), np.log(r).to_numpy(float)
    z = np.full(len(y), np.nan); t = np.full(len(y), np.nan)
    for i in range(n - 1, len(y)):
        yy, xx = y[i - n + 1:i + 1], x[i - n + 1:i + 1]
        ok = ~(np.isnan(yy) | np.isnan(xx))
        if ok.sum() >= n * 0.8:
            z[i], t[i] = _eg(yy[ok], xx[ok])
    return pd.DataFrame({"z": z, "adf_t": t}, index=df.index)


@njit(cache=True)
def _te(xy):
    # xy: concatenated [target(0:m), source(m:2m)] -> TE source->target with 3-state symbols, lag 1
    m = len(xy) // 2
    a = xy[:m]; b = xy[m:]
    def sym(v):
        s = (v - v.mean()) / (v.std() + 1e-12)
        o = np.zeros(len(s), np.int64)
        for i in range(len(s)):
            o[i] = 0 if s[i] < -0.5 else (2 if s[i] > 0.5 else 1)
        return o
    sa = sym(a); sb = sym(b)
    c3 = np.zeros((3, 3, 3)); c2 = np.zeros((3, 3)); c2b = np.zeros((3, 3)); c1 = np.zeros(3)
    for t in range(1, m):
        c3[sa[t], sa[t - 1], sb[t - 1]] += 1
        c2[sa[t], sa[t - 1]] += 1
        c2b[sa[t - 1], sb[t - 1]] += 1
        c1[sa[t - 1]] += 1
    N = m - 1
    te = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if c3[i, j, k] > 0:
                    te += c3[i, j, k] / N * np.log((c3[i, j, k] * c1[j]) / (c2b[j, k] * c2[i, j]))
    return te


@feature("F.CA.TRANSFER_ENTROPY", F, "transfer entropy TE_{ref->target} with lag-1 histories and 3-state symbols (±0.5 std) "
         "over trailing n bars", params={"ref": "EURUSD", "n": 500}, space={"n": [300, 500, 1000]}, **KW)
def f_te(df, panel, symbol, ref="EURUSD", n=500):
    r = _ref(panel, symbol, ref)
    if r is None:
        return pd.Series(np.nan, index=df.index)
    a = np.log(df["close"]).diff().to_numpy(float)
    b = np.log(r).diff().to_numpy(float)
    out = np.full(len(a), np.nan)
    for i in range(n, len(a)):
        wa, wb = a[i - n + 1:i + 1], b[i - n + 1:i + 1]
        ok = ~(np.isnan(wa) | np.isnan(wb))
        if ok.sum() > n * 0.8:
            out[i] = _te(np.concatenate((wa[ok], wb[ok])))
    return pd.Series(out, index=df.index)


@feature("F.CA.PCA_FACTOR", F, "rolling PCA of panel returns (window n, refit every `refit` bars): target loading on PC1 "
         "and PC1 explained-variance share (market-factor strength)", params={"n": 500, "refit": 100},
         space={"n": [250, 500, 1000], "refit": [50, 100]}, **KW)
def f_pca(df, panel, symbol, n=500, refit=100):
    R = _rets(pd.concat([df["close"].rename(symbol), panel.drop(columns=[symbol], errors="ignore")], axis=1))
    X = R.to_numpy(float)
    load = np.full(len(X), np.nan); share = np.full(len(X), np.nan)
    cur = (np.nan, np.nan)
    for i in range(n, len(X)):
        if (i - n) % refit == 0:
            Wf = X[i - n + 1:i + 1]
            if np.isnan(Wf[:, 0]).mean() > 0.2:
                load[i], share[i] = cur
                continue
            keep = np.isnan(Wf).mean(0) <= 0.2
            keep[0] = True
            W = Wf[:, keep]
            W = W[~np.isnan(W).any(1)]
            if len(W) > n * 0.6 and W.shape[1] > 1:
                Z = (W - W.mean(0)) / (W.std(0) + EPS)
                C = np.cov(Z.T)
                ev, V = np.linalg.eigh(C)
                v = V[:, -1] * np.sign(V[:, -1].sum() + EPS)
                cur = (v[0], ev[-1] / ev.sum())
        load[i], share[i] = cur
    return pd.DataFrame({"loading": load, "share": share}, index=df.index)


@feature("F.CA.DISPERSION", F, "cross-sectional dispersion: std across panel instruments of n-bar log returns, divided by "
         "the mean absolute n-bar return", params={"n": 20}, space={"n": [5, 20, 60]}, **KW)
def f_dispersion(df, panel, symbol, n=20):
    r = np.log(panel) - np.log(panel.shift(n))
    return r.std(axis=1) / (r.abs().mean(axis=1) + EPS)


@feature("F.CA.USD_BREADTH", F, "USD breadth: fraction of USD pairs in which USD appreciated over n bars", params={"n": 20},
         space={"n": [5, 20, 60]}, **KW)
def f_usd_breadth(df, panel, symbol, n=20):
    p = _with_target(df, panel, symbol)
    r = np.log(p) - np.log(p.shift(n))
    cols = []
    for s in p.columns:
        if s.endswith("USD") and s[:3] in CURRENCIES:
            cols.append((r[s] < 0).astype(float).where(r[s].notna()))
        elif s.startswith("USD") and s[3:] in CURRENCIES:
            cols.append((r[s] > 0).astype(float).where(r[s].notna()))
    if not cols:
        return pd.Series(np.nan, index=df.index)
    m = pd.concat(cols, axis=1)
    return m.mean(axis=1).where(m.notna().sum(axis=1) >= 1)


@feature("F.CA.CORR_LEVEL", F, "average pairwise correlation of panel returns over n bars (correlation regime)",
         params={"n": 250, "step": 10}, space={"n": [100, 250]}, **KW)
def f_corr_level(df, panel, symbol, n=250, step=10):
    X = _rets(panel).to_numpy(float)
    out = np.full(len(X), np.nan)
    for i in range(n, len(X), step):
        W = _clean_window(X[i - n + 1:i + 1])
        if len(W) > 0.6 * n and W.shape[1] > 1:
            C = np.corrcoef(W.T)
            k = C.shape[0]
            out[i] = (C.sum() - k) / (k * (k - 1))
    return pd.Series(out, index=df.index).ffill()


@feature("F.CA.GRANGER", F, "rolling Granger causality F-statistic: does adding 2 lags of ref returns to an AR(2) of "
         "target returns reduce SSR (trailing n)", params={"ref": "EURUSD", "n": 500, "step": 5},
         space={"n": [250, 500]}, **KW)
def f_granger(df, panel, symbol, ref="EURUSD", n=500, step=5):
    r = _ref(panel, symbol, ref)
    if r is None:
        return pd.Series(np.nan, index=df.index)
    a = np.log(df["close"]).diff()
    b = np.log(r).diff()
    M = pd.concat([a, a.shift(1), a.shift(2), b.shift(1), b.shift(2)], axis=1).to_numpy(float)
    out = np.full(len(M), np.nan)
    for i in range(n, len(M), step):
        W = M[i - n + 1:i + 1]
        W = W[~np.isnan(W).any(1)]
        if len(W) < 0.8 * n:
            continue
        y = W[:, 0]
        Xr = np.column_stack([np.ones(len(W)), W[:, 1:3]])
        Xu = np.column_stack([Xr, W[:, 3:5]])
        ssr_r = ((y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]) ** 2).sum()
        ssr_u = ((y - Xu @ np.linalg.lstsq(Xu, y, rcond=None)[0]) ** 2).sum()
        out[i] = ((ssr_r - ssr_u) / 2) / (ssr_u / (len(W) - 5))
    return pd.Series(out, index=df.index).ffill()
