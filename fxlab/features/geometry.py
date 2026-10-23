"""7.8 Fibonacci, harmonic and geometric price-time constructs, all anchored on
causally confirmed zigzag pivots (no repainting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, atr, zigzag, causal_pivots

F = "fibonacci_geometry"
TS = {"thr": [0.002, 0.005, 0.01]}
FIB_RETR = np.array([0.236, 0.382, 0.5, 0.618, 0.786])
FIB_EXT = np.array([1.272, 1.618, 2.0, 2.618])
FIBS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377])
PHI = (1 + 5 ** 0.5) / 2


def _atr_or_proxy(df):
    if {"high", "low"} <= set(df.columns):
        return atr(df, 14)
    return df["close"].diff().abs().ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()


@feature("F.FG.FIB_RETRACE", F, "retracement ratio rho=(P_last - C)/(P_last - P_prev) of the last confirmed zigzag leg; "
         "outputs rho and distance (in ATR) to the nearest Fibonacci retracement level {.236,.382,.5,.618,.786}",
         params={"thr": 0.005}, space=TS)
def f_fib_retrace(df, thr=0.005):
    pp, _, _ = zigzag(df["close"], thr)
    c = df["close"].to_numpy(float)
    leg = pp[:, 3] - pp[:, 2]
    rho = (pp[:, 3] - c) / np.where(np.abs(leg) > 0, leg, np.nan)
    levels = pp[:, 3][:, None] - FIB_RETR[None, :] * leg[:, None]
    a = _atr_or_proxy(df).to_numpy(float)
    near = np.nanmin(np.abs(c[:, None] - levels), axis=1) / (a + EPS) if len(c) else np.array([])
    return pd.DataFrame({"rho": rho, "nearest_level_atr": near}, index=df.index)


@feature("F.FG.FIB_EXTENSION", F, "projection of the prior leg from the last pivot: eta=(C - P_last)/(P_prev - P_prev2) "
         "(ABC -> D extension) and distance to nearest extension level {1.272,1.618,2,2.618}", params={"thr": 0.005}, space=TS)
def f_fib_ext(df, thr=0.005):
    pp, _, _ = zigzag(df["close"], thr)
    c = df["close"].to_numpy(float)
    ab = pp[:, 2] - pp[:, 1]  # previous same-direction leg
    eta = (c - pp[:, 3]) / np.where(np.abs(ab) > 0, ab, np.nan)
    levels = pp[:, 3][:, None] + FIB_EXT[None, :] * ab[:, None]
    a = _atr_or_proxy(df).to_numpy(float)
    near = np.nanmin(np.abs(c[:, None] - levels), axis=1) / (a + EPS)
    return pd.DataFrame({"eta": eta, "nearest_ext_atr": near}, index=df.index)


HARMONICS = {  # (AB/XA, BC/AB range, CD/BC range, AD/XA target)
    "gartley": ((0.618, 0.618), (0.382, 0.886), (1.272, 1.618), 0.786),
    "bat": ((0.382, 0.5), (0.382, 0.886), (1.618, 2.618), 0.886),
    "butterfly": ((0.786, 0.786), (0.382, 0.886), (1.618, 2.24), 1.27),
    "crab": ((0.382, 0.618), (0.382, 0.886), (2.24, 3.618), 1.618),
}


@feature("F.FG.HARMONIC", F, "XABCD harmonic score: X,A,B,C = last 4 confirmed pivots, D = current price; for Gartley/Bat/"
         "Butterfly/Crab, score = -sum of log-distance of ratios AB/XA, BC/AB to their bands and of AD/XA to target; "
         "outputs best score and its direction (+1 bullish D below C)", params={"thr": 0.005}, space=TS)
def f_harmonic(df, thr=0.005):
    pp, _, _ = zigzag(df["close"], thr)
    c = df["close"].to_numpy(float)
    X, A, B, C = pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3]
    XA, AB, BC = A - X, B - A, C - B
    CD = c - C
    with np.errstate(divide="ignore", invalid="ignore"):
        r1, r2, r4 = np.abs(AB / XA), np.abs(BC / AB), np.abs((c - A) / XA)
        r3 = np.abs(CD / BC)

    def band_pen(r, lo, hi):
        return np.where(r < lo, np.log(lo / r), np.where(r > hi, np.log(r / hi), 0.0))

    best = np.full(len(c), -np.inf)
    for (b1, b2, b3, t) in HARMONICS.values():
        pen = band_pen(r1, *b1) + band_pen(r2, *b2) + band_pen(r3, *b3) + np.abs(np.log(r4 / t))
        best = np.maximum(best, -pen)
    best = np.where(np.isfinite(best), best, np.nan)
    direction = np.sign(-XA)  # XA up => bearish D (sell), XA down => bullish
    return pd.DataFrame({"score": best, "dir": direction}, index=df.index)


@feature("F.FG.FIB_TIME", F, "Fibonacci time zones: bars since last confirmed pivot, and log-distance to the nearest "
         "Fibonacci number", params={"thr": 0.005}, space=TS)
def f_fib_time(df, thr=0.005):
    _, pi, _ = zigzag(df["close"], thr)
    t = np.arange(len(df)) - pi[:, 3]
    t = np.where(pi[:, 3] >= 0, t, np.nan).astype(float)
    d = np.nanmin(np.abs(np.log((t[:, None] + 1) / (FIBS[None, :] + 1))), axis=1)
    return pd.DataFrame({"bars_since": t, "fib_dist": d}, index=df.index)


@feature("F.FG.TIME_PRICE_SYMMETRY", F, "AB=CD symmetry: ratios (current leg price / previous same-direction leg price) and "
         "(current leg duration / previous leg duration)", params={"thr": 0.005}, space=TS)
def f_symmetry(df, thr=0.005):
    pp, pi, _ = zigzag(df["close"], thr)
    c = df["close"].to_numpy(float)
    i = np.arange(len(c))
    p_ratio = (c - pp[:, 3]) / np.where(np.abs(pp[:, 2] - pp[:, 1]) > 0, pp[:, 2] - pp[:, 1], np.nan)
    t_ratio = (i - pi[:, 3]) / np.where(pi[:, 2] - pi[:, 1] > 0, pi[:, 2] - pi[:, 1], np.nan)
    ok = pi[:, 1] >= 0
    return pd.DataFrame({"price_ratio": np.where(ok, p_ratio, np.nan), "time_ratio": np.where(ok, t_ratio, np.nan)},
                        index=df.index)


@feature("F.FG.GANN_ANGLE", F, "Gann 1x1 from last confirmed pivot with unit = ATR per bar measured at the pivot: "
         "deviation (C - (P + dir*unit*bars))/ATR and angle atan2(ΔP/ATR, Δbars) in degrees",
         params={"thr": 0.005}, space=TS)
def f_gann(df, thr=0.005):
    pp, pi, d = zigzag(df["close"], thr)
    a = _atr_or_proxy(df).to_numpy(float)
    c = df["close"].to_numpy(float)
    i = np.arange(len(c))
    piv_i = pi[:, 3]
    unit = np.where(piv_i >= 0, a[np.clip(piv_i, 0, None)], np.nan)
    bars = i - piv_i
    dirn = np.where(c >= pp[:, 3], 1.0, -1.0)
    dev = (c - (pp[:, 3] + dirn * unit * bars)) / (a + EPS)
    ang = np.degrees(np.arctan2((c - pp[:, 3]) / (unit + EPS), np.maximum(bars, 1)))
    ok = piv_i >= 0
    return pd.DataFrame({"dev_1x1": np.where(ok, dev, np.nan), "angle": np.where(ok, ang, np.nan)}, index=df.index)


@feature("F.FG.SQUARE_OF_9", F, "Gann square-of-nine: frac(sqrt(P_scaled)/2) phase of price on the spiral "
         "(P scaled to integer pips), encoded as sin/cos; levels at 45-degree increments", params={}, space={})
def f_sq9(df):
    c = df["close"]
    scale = 100.0 if c.median() > 20 else 10000.0
    th = 2 * np.pi * ((np.sqrt(c * scale) / 2) % 1.0)
    return pd.DataFrame({"sin": np.sin(th), "cos": np.cos(th)})


@njit(cache=True)
def _pitchfork(c, pp, pi, a):
    n = len(c)
    out = np.full(n, np.nan)
    for i in range(n):
        if pi[i, 1] < 0 or np.isnan(a[i]) or a[i] <= 0:
            continue
        # pivots: P1=pp[,1] (handle), P2=pp[,2], P3=pp[,3]
        m_t = (pi[i, 2] + pi[i, 3]) / 2.0
        m_p = (pp[i, 2] + pp[i, 3]) / 2.0
        if m_t == pi[i, 1]:
            continue
        slope = (m_p - pp[i, 1]) / (m_t - pi[i, 1])
        median_line = pp[i, 1] + slope * (i - pi[i, 1])
        out[i] = (c[i] - median_line) / a[i]
    return out


@feature("F.FG.PITCHFORK", F, "Andrews pitchfork: median line from pivot P1 through midpoint of (P2,P3) (last 3 confirmed "
         "pivots); signed distance of C to the median line in ATR", params={"thr": 0.005}, space=TS)
def f_pitchfork(df, thr=0.005):
    pp, pi, _ = zigzag(df["close"], thr)
    return pd.Series(_pitchfork(df["close"].to_numpy(float), pp, pi.astype(np.float64),
                                _atr_or_proxy(df).to_numpy(float)), index=df.index)


@feature("F.FG.GOLDEN_SWING", F, "golden-ratio swing amplitude: ln(|leg_k| / |leg_{k-1}|) - ln(phi) for the last two "
         "completed zigzag legs, and its absolute value", params={"thr": 0.005}, space=TS)
def f_golden_swing(df, thr=0.005):
    pp, _, _ = zigzag(df["close"], thr)
    l1 = np.abs(pp[:, 3] - pp[:, 2]); l0 = np.abs(pp[:, 2] - pp[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(l1 / l0)
    dev = np.abs(np.abs(r) - np.log(PHI))
    return pd.DataFrame({"log_ratio": r, "phi_dev": dev}, index=df.index).replace([np.inf, -np.inf], np.nan)


@feature("F.FG.MURREY", F, "Murrey math octave: position of C in the 1/8 grid of the trailing n-bar range snapped to a "
         "power-of-two frame; outputs octave fraction in [0,1] and distance to nearest 1/8 line",
         params={"n": 64}, space={"n": [32, 64, 128]})
def f_murrey(df, n=64):
    hi = df["high"].rolling(n, min_periods=n).max() if "high" in df else df["close"].rolling(n, min_periods=n).max()
    lo = df["low"].rolling(n, min_periods=n).min() if "low" in df else df["close"].rolling(n, min_periods=n).min()
    rng = hi - lo
    frame = 2.0 ** np.ceil(np.log2(rng.where(rng > 0)))
    base = np.floor(lo / (frame / 8)) * (frame / 8)
    pos = (df["close"] - base) / frame
    eighth = pos * 8
    return pd.DataFrame({"octave_pos": pos, "dist_eighth": np.abs(eighth - np.round(eighth))})


@feature("F.FG.LEG_VECTOR", F, "geometric leg vector in (bars, ATR) space from last confirmed pivot: magnitude "
         "sqrt(Δt^2 + (ΔP/ATR)^2) and slope ΔP/(ATR Δt)", params={"thr": 0.005}, space=TS)
def f_leg_vector(df, thr=0.005):
    pp, pi, _ = zigzag(df["close"], thr)
    c = df["close"].to_numpy(float)
    a = _atr_or_proxy(df).to_numpy(float)
    dt = (np.arange(len(c)) - pi[:, 3]).astype(float)
    dp = (c - pp[:, 3]) / (a + EPS)
    ok = pi[:, 3] >= 0
    return pd.DataFrame({"mag": np.where(ok, np.sqrt(dt ** 2 + dp ** 2), np.nan),
                         "slope": np.where(ok, dp / np.maximum(dt, 1), np.nan)}, index=df.index)
