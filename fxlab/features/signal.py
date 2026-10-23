"""7.9 Signal processing. Window-based transforms are applied to each TRAILING
window and only the window's last sample (or a window-level statistic) is
kept, so values are causal by construction (end effects are accepted, never
repaired with future data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view

from fxlab.registry import feature
from fxlab.features._util import EPS, ema, as_series, bars_per_day

F = "signal_processing"
WS = {"w": [64, 128, 256]}


def batch_windows(x: np.ndarray, w: int, func, k: int, chunk: int = 20000) -> np.ndarray:
    """Apply func((m,w) array) -> (m,k) to every trailing window; NaN-containing windows -> NaN."""
    n = len(x)
    out = np.full((n, k), np.nan)
    if n < w:
        return out
    V = sliding_window_view(x, w)  # row j = x[j:j+w], ends at j+w-1
    for s in range(0, len(V), chunk):
        blk = V[s:s + chunk]
        ok = ~np.isnan(blk).any(axis=1)
        if ok.any():
            res = func(blk[ok])
            tmp = np.full((len(blk), k), np.nan)
            tmp[ok] = res.reshape(len(res), k)
            out[s + w - 1:s + w - 1 + len(blk)] = tmp
    return out


def _detrended_logprice(df):
    return np.log(df["close"]).to_numpy(float)


def _prep(blk):
    t = np.arange(blk.shape[1], dtype=float)
    t = t - t.mean()
    b = (blk - blk.mean(1, keepdims=True)) @ t / (t @ t)
    res = blk - blk.mean(1, keepdims=True) - b[:, None] * t[None, :]
    return res * np.hanning(blk.shape[1])[None, :]


def _power(blk):
    P = np.abs(np.fft.rfft(_prep(blk), axis=1)) ** 2
    return P[:, 1:]  # drop DC


@feature("F.SP.FFT_DOMINANT", F, "dominant period of linearly detrended, Hann-windowed ln C over trailing w bars: "
         "argmax_f |FFT|^2 -> period w/f, and share of power at that frequency", params={"w": 128}, space=WS,
         complexity="O(n·w log w)")
def f_fft_dom(df, w=128):
    def fn(blk):
        P = _power(blk)
        k = P.argmax(1)
        return np.column_stack([w / (k + 1.0), P[np.arange(len(P)), k] / (P.sum(1) + EPS)])
    o = batch_windows(_detrended_logprice(df), w, fn, 2)
    return pd.DataFrame({"period": o[:, 0], "power_share": o[:, 1]}, index=df.index)


@feature("F.SP.SPECTRAL_ENTROPY", F, "normalised Shannon entropy of the power spectrum of trailing w log-returns "
         "-sum p_f ln p_f / ln(#f)", params={"w": 128}, space=WS, complexity="O(n·w log w)")
def f_spec_ent(df, w=128):
    r = np.log(df["close"]).diff().to_numpy(float)
    def fn(blk):
        P = np.abs(np.fft.rfft(blk - blk.mean(1, keepdims=True), axis=1))[:, 1:] ** 2
        p = P / (P.sum(1, keepdims=True) + EPS)
        return -(p * np.log(p + EPS)).sum(1) / np.log(p.shape[1])
    return pd.Series(batch_windows(r, w, fn, 1)[:, 0], index=df.index)


@feature("F.SP.SPECTRAL_BANDS", F, "power share of trailing-window log-return spectrum in low (<w/8 bars^-1... periods > w/4), "
         "mid and high (periods < 4 bars) bands", params={"w": 128}, space=WS)
def f_spec_bands(df, w=128):
    r = np.log(df["close"]).diff().to_numpy(float)
    def fn(blk):
        P = np.abs(np.fft.rfft(blk - blk.mean(1, keepdims=True), axis=1))[:, 1:] ** 2
        f = np.arange(1, P.shape[1] + 1) / w  # cycles per bar
        tot = P.sum(1) + EPS
        lo = P[:, f < 4.0 / w].sum(1) / tot
        hi = P[:, f > 0.25].sum(1) / tot
        return np.column_stack([lo, hi, 1 - lo - hi])
    o = batch_windows(r, w, fn, 3)
    return pd.DataFrame({"low": o[:, 0], "high": o[:, 1], "mid": o[:, 2]}, index=df.index)


@feature("F.SP.SPECTRAL_SLOPE", F, "spectral exponent beta: OLS slope of ln P(f) on ln f for trailing log returns "
         "(beta~0 white noise, <0 persistent low-frequency dominance)", params={"w": 128}, space=WS)
def f_spec_slope(df, w=128):
    r = np.log(df["close"]).diff().to_numpy(float)
    def fn(blk):
        P = np.abs(np.fft.rfft(blk - blk.mean(1, keepdims=True), axis=1))[:, 1:] ** 2
        lf = np.log(np.arange(1, P.shape[1] + 1))
        lf = lf - lf.mean()
        lp = np.log(P + EPS)
        return (lp - lp.mean(1, keepdims=True)) @ lf / (lf @ lf)
    return pd.Series(batch_windows(r, w, fn, 1)[:, 0], index=df.index)


@feature("F.SP.HILBERT", F, "analytic signal (scipy Hilbert) of the detrended trailing window; instantaneous phase "
         "(sin, cos) and amplitude (relative to window std) at the window's last sample", params={"w": 64}, space={"w": [32, 64, 128]})
def f_hilbert(df, w=64):
    from scipy.signal import hilbert
    def fn(blk):
        t = np.arange(w, dtype=float); t -= t.mean()
        b = (blk - blk.mean(1, keepdims=True)) @ t / (t @ t)
        res = blk - blk.mean(1, keepdims=True) - b[:, None] * t
        z = hilbert(res, axis=1)[:, -1]
        sd = res.std(1) + EPS
        ph = np.angle(z)
        return np.column_stack([np.sin(ph), np.cos(ph), np.abs(z) / sd])
    o = batch_windows(_detrended_logprice(df), w, fn, 3)
    return pd.DataFrame({"sin": o[:, 0], "cos": o[:, 1], "amp": o[:, 2]}, index=df.index)


@njit(cache=True)
def _homodyne(price):
    n = len(price)
    smooth = np.zeros(n); det = np.zeros(n); i1 = np.zeros(n); q1 = np.zeros(n)
    i2 = np.zeros(n); q2 = np.zeros(n); re = np.zeros(n); im = np.zeros(n)
    period = np.zeros(n); sp = np.zeros(n); out = np.full(n, np.nan)
    for t in range(6, n):
        smooth[t] = (4 * price[t] + 3 * price[t - 1] + 2 * price[t - 2] + price[t - 3]) / 10.0
        adj = 0.075 * period[t - 1] + 0.54
        det[t] = (0.0962 * smooth[t] + 0.5769 * smooth[t - 2] - 0.5769 * smooth[t - 4] - 0.0962 * smooth[t - 6]) * adj
        q1[t] = (0.0962 * det[t] + 0.5769 * det[t - 2] - 0.5769 * det[t - 4] - 0.0962 * det[t - 6]) * adj
        i1[t] = det[t - 3]
        ji = (0.0962 * i1[t] + 0.5769 * i1[t - 2] - 0.5769 * i1[t - 4] - 0.0962 * i1[t - 6]) * adj
        jq = (0.0962 * q1[t] + 0.5769 * q1[t - 2] - 0.5769 * q1[t - 4] - 0.0962 * q1[t - 6]) * adj
        i2[t] = 0.2 * (i1[t] - jq) + 0.8 * i2[t - 1]
        q2[t] = 0.2 * (q1[t] + ji) + 0.8 * q2[t - 1]
        re[t] = 0.2 * (i2[t] * i2[t - 1] + q2[t] * q2[t - 1]) + 0.8 * re[t - 1]
        im[t] = 0.2 * (i2[t] * q2[t - 1] - q2[t] * i2[t - 1]) + 0.8 * im[t - 1]
        p = period[t - 1]
        if im[t] != 0 and re[t] != 0:
            p = 2 * np.pi / np.arctan(im[t] / re[t])
        if period[t - 1] > 0:
            p = min(max(p, 0.67 * period[t - 1]), 1.5 * period[t - 1])
        p = min(max(p, 6.0), 50.0)
        period[t] = 0.2 * p + 0.8 * period[t - 1]
        sp[t] = 0.33 * period[t] + 0.67 * sp[t - 1]
        if t >= 50:
            out[t] = sp[t]
    return out


@feature("F.SP.HOMODYNE_CYCLE", F, "Ehlers homodyne-discriminator dominant cycle period (6..50 bars), recursive and causal")
def f_homodyne(df):
    return as_series(_homodyne(df["close"].to_numpy(float)), df.index)


@njit(cache=True)
def _sinewave(price, period):
    n = len(price)
    s = np.full(n, np.nan); ls = np.full(n, np.nan)
    for t in range(60, n):
        p = period[t]
        if np.isnan(p):
            continue
        L = int(p + 0.5)
        if L < 6 or t - L < 0:
            continue
        rp = 0.0; ip = 0.0
        for k in range(L):
            rp += np.sin(2 * np.pi * k / L) * price[t - k]
            ip += np.cos(2 * np.pi * k / L) * price[t - k]
        ph = np.arctan2(rp, ip) + np.pi / 2
        s[t] = np.sin(ph); ls[t] = np.sin(ph + np.pi / 4)
    return s, ls


@feature("F.SP.SINEWAVE", F, "Ehlers sinewave indicator: DFT phase of the last homodyne dominant-cycle length of price; "
         "outputs sin(phase) and lead sin(phase+45deg)")
def f_sinewave(df):
    p = _homodyne(df["close"].to_numpy(float))
    s, l = _sinewave(df["close"].to_numpy(float), p)
    return pd.DataFrame({"sine": s, "lead": l}, index=df.index)


@njit(cache=True)
def _bandpass(x, period, delta):
    n = len(x)
    out = np.full(n, np.nan)
    beta = np.cos(2 * np.pi / period)
    gamma = 1 / np.cos(4 * np.pi * delta / period)
    alpha = gamma - np.sqrt(gamma * gamma - 1)
    bp1 = 0.0; bp2 = 0.0
    for t in range(2, n):
        bp = 0.5 * (1 - alpha) * (x[t] - x[t - 2]) + beta * (1 + alpha) * bp1 - alpha * bp2
        bp2, bp1 = bp1, bp
        if t > 3 * period:
            out[t] = bp
    return out


@feature("F.SP.BANDPASS", F, "Ehlers 2-pole band-pass filter of ln C centred at period P with fractional bandwidth delta, "
         "normalised by trailing std of log returns", params={"period": 20, "delta": 0.3},
         space={"period": [10, 20, 40, 60], "delta": [0.1, 0.3]})
def f_bandpass(df, period=20, delta=0.3):
    lc = np.log(df["close"])
    bp = as_series(_bandpass(lc.to_numpy(float), float(period), float(delta)), df.index)
    return bp / (lc.diff().rolling(period * 3, min_periods=period).std() + EPS)


@njit(cache=True)
def _supersmoother(x, period):
    n = len(x)
    out = np.full(n, np.nan)
    a = np.exp(-1.414 * np.pi / period)
    b = 2 * a * np.cos(1.414 * np.pi / period)
    c2, c3 = b, -a * a
    c1 = 1 - c2 - c3
    f1 = x[0]; f2 = x[0]
    for t in range(1, n):
        f = c1 * (x[t] + x[t - 1]) / 2 + c2 * f1 + c3 * f2
        f2, f1 = f1, f
        if t > period:
            out[t] = f
    return out


@feature("F.SP.SUPER_SMOOTHER", F, "Ehlers 2-pole super smoother (Butterworth-derived) of ln C; ln C - SS",
         params={"period": 20}, space={"period": [10, 20, 40]})
def f_supersmoother(df, period=20):
    lc = np.log(df["close"])
    return lc - as_series(_supersmoother(lc.to_numpy(float), float(period)), df.index)


@njit(cache=True)
def _roofing(x, hp_period, ss_period):
    n = len(x)
    out = np.full(n, np.nan)
    a1 = (np.cos(0.707 * 2 * np.pi / hp_period) + np.sin(0.707 * 2 * np.pi / hp_period) - 1) / np.cos(0.707 * 2 * np.pi / hp_period)
    hp1 = 0.0; hp2 = 0.0
    hp = np.zeros(n)
    for t in range(2, n):
        h = (1 - a1 / 2) ** 2 * (x[t] - 2 * x[t - 1] + x[t - 2]) + 2 * (1 - a1) * hp1 - (1 - a1) ** 2 * hp2
        hp2, hp1 = hp1, h
        hp[t] = h
    a = np.exp(-1.414 * np.pi / ss_period)
    c2 = 2 * a * np.cos(1.414 * np.pi / ss_period); c3 = -a * a; c1 = 1 - c2 - c3
    f1 = 0.0; f2 = 0.0
    for t in range(1, n):
        f = c1 * (hp[t] + hp[t - 1]) / 2 + c2 * f1 + c3 * f2
        f2, f1 = f1, f
        if t > hp_period:
            out[t] = f
    return out


@feature("F.SP.ROOFING", F, "Ehlers roofing filter: 2-pole high-pass (period hp) followed by super smoother (ss); "
         "normalised by trailing return std", params={"hp": 48, "ss": 10}, space={"hp": [48, 96], "ss": [10, 20]})
def f_roofing(df, hp=48, ss=10):
    lc = np.log(df["close"])
    r = as_series(_roofing(lc.to_numpy(float), float(hp), float(ss)), df.index)
    return r / (lc.diff().rolling(hp, min_periods=hp).std() + EPS)


@feature("F.SP.WAVELET_ENERGY", F, "relative energy per scale of a db4 DWT (level L) of trailing-window log returns",
         params={"w": 128, "level": 4}, space={"w": [64, 128, 256], "level": [3, 4]}, requires=("pywt",),
         complexity="O(n·w)")
def f_wavelet_energy(df, w=128, level=4):
    import pywt
    r = np.log(df["close"]).diff().to_numpy(float)
    def fn(blk):
        coeffs = pywt.wavedec(blk, "db4", level=level, axis=1, mode="periodization")
        E = np.column_stack([(c ** 2).sum(1) for c in coeffs])
        return E / (E.sum(1, keepdims=True) + EPS)
    o = batch_windows(r, w, fn, level + 1)
    return pd.DataFrame(o, index=df.index, columns=[f"e{i}" for i in range(level + 1)])


@feature("F.SP.WAVELET_DENOISE", F, "soft-threshold (universal threshold sigma sqrt(2 ln w), sigma=MAD/0.6745 of finest "
         "detail) db4 denoising of the trailing window of ln C; output (ln C_t - denoised_t)/std of returns",
         params={"w": 128}, space={"w": [64, 128, 256]}, requires=("pywt",))
def f_wavelet_denoise(df, w=128):
    import pywt
    lc = np.log(df["close"])
    def fn(blk):
        coeffs = pywt.wavedec(blk, "db4", level=4, axis=1, mode="periodization")
        sig = np.median(np.abs(coeffs[-1]), axis=1) / 0.6745
        thr = sig * np.sqrt(2 * np.log(w))
        new = [coeffs[0]] + [np.sign(c) * np.maximum(np.abs(c) - thr[:, None], 0) for c in coeffs[1:]]
        rec = pywt.waverec(new, "db4", axis=1, mode="periodization")[:, :w]
        return blk[:, -1] - rec[:, -1]
    d = pd.Series(batch_windows(lc.to_numpy(float), w, fn, 1)[:, 0], index=df.index)
    return d / (lc.diff().rolling(w, min_periods=w).std() + EPS)


@feature("F.SP.GOERTZEL", F, "Goertzel power at a fixed calendar period (default 1 trading day in bars) in trailing "
         "log returns, relative to total window power", params={"cycles_per_day": 1.0, "days": 5},
         space={"cycles_per_day": [1.0, 2.0, 3.0], "days": [5, 10]}, frequencies="M5..H4")
def f_goertzel(df, cycles_per_day=1.0, days=5):
    bpd = bars_per_day(df.index)
    w = int(bpd * days)
    period = bpd / cycles_per_day
    r = np.log(df["close"]).diff().to_numpy(float)
    k = np.arange(w)
    cs = np.cos(2 * np.pi * k / period); sn = np.sin(2 * np.pi * k / period)
    def fn(blk):
        x = blk - blk.mean(1, keepdims=True)
        p = (x @ cs) ** 2 + (x @ sn) ** 2
        return p / ((x ** 2).sum(1) * w / 2 + EPS)
    return pd.Series(batch_windows(r, w, fn, 1, chunk=5000)[:, 0], index=df.index)


@feature("F.SP.ACF_PERIODOGRAM", F, "Ehlers autocorrelation periodogram: DFT of the autocorrelation function (lags 1..48) "
         "of roofing-filtered price; dominant period 10..48 bars", params={"w": 48}, space={"w": [48]})
def f_acf_periodogram(df, w=48):
    lc = np.log(df["close"]).to_numpy(float)
    filt = _roofing(lc, 48.0, 10.0)
    L = w
    periods = np.arange(10, 49)
    def fn(blk):
        x = blk - blk.mean(1, keepdims=True)
        sd = x.std(1) + EPS
        acf = np.column_stack([(x[:, lag:] * x[:, :-lag]).mean(1) / sd ** 2 for lag in range(1, L)])
        lags = np.arange(1, L)
        P = np.column_stack([(acf @ np.cos(2 * np.pi * lags / p)) ** 2 + (acf @ np.sin(2 * np.pi * lags / p)) ** 2
                             for p in periods])
        return periods[P.argmax(1)].astype(float)
    return pd.Series(batch_windows(filt, 2 * L, fn, 1)[:, 0], index=df.index)


@feature("F.SP.CEPSTRUM", F, "real cepstrum of the trailing-window return spectrum: quefrency (bars) and height of the "
         "largest peak at quefrency >= 2 (echo / periodicity detector)", params={"w": 128}, space=WS)
def f_cepstrum(df, w=128):
    r = np.log(df["close"]).diff().to_numpy(float)
    def fn(blk):
        S = np.abs(np.fft.fft(blk - blk.mean(1, keepdims=True), axis=1)) ** 2
        c = np.real(np.fft.ifft(np.log(S + EPS), axis=1))[:, 2:w // 2]
        k = c.argmax(1)
        return np.column_stack([k + 2.0, c[np.arange(len(c)), k]])
    o = batch_windows(r, w, fn, 2)
    return pd.DataFrame({"quefrency": o[:, 0], "peak": o[:, 1]}, index=df.index)


@feature("F.SP.ZERO_CROSSING", F, "zero-crossing rate of (ln C - EMA_m ln C) over trailing w bars (cycle frequency proxy)",
         params={"w": 100, "m": 20}, space={"w": [50, 100], "m": [10, 20]})
def f_zero_cross(df, w=100, m=20):
    lc = np.log(df["close"])
    d = lc - ema(lc, m)
    s = np.sign(d)
    cross = (s != s.shift(1)).astype(float).where(d.notna() & d.shift(1).notna())
    return cross.rolling(w, min_periods=w).mean()


def _emd_imfs(x, n_imfs=2, max_sift=10):
    """Basic EMD sifting with cubic-spline envelopes (end points mirrored). Returns n_imfs IMFs."""
    from scipy.interpolate import CubicSpline
    t = np.arange(len(x), dtype=float)
    res = x.astype(float).copy()
    imfs = []
    for _ in range(n_imfs):
        h = res.copy()
        for _ in range(max_sift):
            d = np.diff(h)
            mx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
            mn = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
            if len(mx) < 2 or len(mn) < 2:
                break
            mx = np.concatenate([[0], mx, [len(h) - 1]]); mn = np.concatenate([[0], mn, [len(h) - 1]])
            up = CubicSpline(t[mx], h[mx])(t); lo = CubicSpline(t[mn], h[mn])(t)
            h = h - (up + lo) / 2
        imfs.append(h)
        res = res - h
    return np.array(imfs)


@feature("F.SP.EMD", F, "empirical mode decomposition (Huang) of the trailing window of ln C via spline-envelope sifting; "
         "last-sample values of IMF1 and IMF2 (normalised by return std) - adaptive multiresolution decomposition",
         params={"w": 128, "step": 1}, space={"w": [64, 128, 256], "step": [1, 5]}, complexity="O(n·w·sift/step)")
def f_emd(df, w=128, step=1):
    lc = np.log(df["close"]).to_numpy(float)
    n = len(lc)
    o = np.full((n, 2), np.nan)
    for i in range(w - 1, n, step):
        win = lc[i - w + 1:i + 1]
        if np.isnan(win).any():
            continue
        imf = _emd_imfs(win - win.mean())
        o[i] = imf[:, -1]
    out = pd.DataFrame(o, index=df.index, columns=["imf1", "imf2"])
    if step > 1:
        out = out.ffill()
    sd = np.log(df["close"]).diff().rolling(w, min_periods=w).std()
    return out.div(sd + EPS, axis=0)


@feature("F.SP.SSA", F, "singular spectrum analysis of the trailing window of ln C (embedding L=w/4): share of variance in "
         "the leading eigentriple and deviation of ln C_t from its rank-1 (trend) reconstruction, in return-std units",
         params={"w": 128}, space={"w": [64, 128, 256]}, complexity="O(n·w·L^2)")
def f_ssa(df, w=128):
    L = w // 4
    lc = np.log(df["close"])
    def fn(blk):
        out = np.empty((len(blk), 2))
        for r in range(len(blk)):
            x = blk[r] - blk[r].mean()
            X = sliding_window_view(x, L).T  # L x K trajectory matrix
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            X1 = s[0] * np.outer(U[:, 0], Vt[0])
            out[r, 0] = s[0] ** 2 / (s ** 2).sum()
            out[r, 1] = x[-1] - X1[-1, -1]  # anti-diagonal of length 1 at the final sample
        return out
    o = batch_windows(lc.to_numpy(float), w, fn, 2, chunk=2000)
    sd = lc.diff().rolling(w, min_periods=w).std()
    return pd.DataFrame({"lead_share": o[:, 0], "trend_dev": o[:, 1] / (sd + EPS).to_numpy()}, index=df.index)
