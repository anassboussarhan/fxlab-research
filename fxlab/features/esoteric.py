"""7.16 Unconventional / esoteric hypotheses translated into explicit, causal,
reproducible rules. These are SPECULATIVE: the validation layer applies the
same multiple-testing corrections as everything else, and the experiment
manifest tags them (tag 'speculative') so they can be reported separately."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, zigzag, as_series
from fxlab.data.instruments import INSTRUMENTS

F = "esoteric"
SPEC = dict(tags=("speculative",))
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                   101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199])
FIBS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
PHI = (1 + 5 ** 0.5) / 2


def _pip(df, symbol=None):
    if symbol and symbol in INSTRUMENTS:
        return INSTRUMENTS[symbol].pip
    return 0.01 if df["close"].median() > 20 else 0.0001


@feature("F.ES.ROUND_NUMBER", F, "psychological levels: signed distance (pips) of C to the nearest 'big figure' (100 pips) "
         "and 'half figure' (50 pips), and to the nearest 00-level, scaled by ATR in pips", params={}, space={}, **SPEC)
def f_round(df):
    pip = _pip(df)
    p = df["close"] / pip
    d100 = p - np.round(p / 100) * 100
    d50 = p - np.round(p / 50) * 50
    a = (df["high"] - df["low"]).rolling(14, min_periods=14).mean() / pip if "high" in df else p.diff().abs().rolling(14).mean()
    return pd.DataFrame({"d100": d100, "d50": d50, "d100_atr": d100 / (a + EPS)})


@feature("F.ES.DIGIT_PATTERN", F, "digit pattern: last whole-pip digit of C (one-hot not used; encoded as sin/cos of "
         "2 pi d/10) and chi-square deviation from uniform of that digit over trailing n closes",
         params={"n": 500}, space={"n": [200, 500]}, **SPEC)
def f_digits(df, n=500):
    pip = _pip(df)
    d = np.floor(df["close"] / pip + 1e-9).astype("int64") % 10
    oh = pd.get_dummies(d).reindex(columns=range(10), fill_value=0).astype(float)
    cnt = oh.rolling(n, min_periods=n).sum()
    chi = ((cnt - n / 10) ** 2 / (n / 10)).sum(axis=1).where(cnt.notna().all(axis=1))
    return pd.DataFrame({"sin": np.sin(2 * np.pi * d / 10), "cos": np.cos(2 * np.pi * d / 10), "chi2": chi})


@feature("F.ES.TIME_MODULAR", F, "modular time: phase of minute-of-day modulo k (sin/cos 2 pi (m mod k)/k) for k in "
         "{15, 60, 90} (cyclical time structures)", params={"k": 90}, space={"k": [15, 60, 90, 144]}, inputs=(),
         frequencies="M1..M30", **SPEC)
def f_time_mod(df, k=90):
    m = df.index.hour * 60 + df.index.minute
    ph = (m % k) / k
    return pd.DataFrame({"sin": np.sin(2 * np.pi * ph), "cos": np.cos(2 * np.pi * ph)}, index=df.index)


@feature("F.ES.PRIME_INTERVAL", F, "prime-number intervals: 1 if bars since the last confirmed zigzag pivot is prime, "
         "and price in pips modulo 7 and 11 (residue classes)", params={"thr": 0.005}, space={"thr": [0.002, 0.005]}, **SPEC)
def f_prime(df, thr=0.005):
    _, pi, _ = zigzag(df["close"], thr)
    t = np.arange(len(df)) - pi[:, 3]
    isprime = np.isin(t, PRIMES).astype(float)
    pips = np.floor(df["close"].to_numpy() / _pip(df)).astype(np.int64)
    return pd.DataFrame({"since_pivot_prime": np.where(pi[:, 3] >= 0, isprime, np.nan),
                         "mod7": pips % 7, "mod11": pips % 11}, index=df.index)


@feature("F.ES.BENFORD", F, "Benford deviation: KL divergence of first significant digits of tick counts over trailing n "
         "bars from Benford's law log10(1+1/d)", params={"n": 500}, space={"n": [200, 500]}, inputs=("n_ticks",), **SPEC)
def f_benford(df, n=500):
    v = df["n_ticks"].astype(float).clip(lower=1)
    fd = np.floor(v / 10 ** np.floor(np.log10(v))).astype(int)
    oh = pd.get_dummies(fd).reindex(columns=range(1, 10), fill_value=0).astype(float)
    cnt = oh.rolling(n, min_periods=n).sum() + 0.5
    p = cnt.div(cnt.sum(axis=1), axis=0)
    b = np.log10(1 + 1 / np.arange(1, 10))
    return (p * np.log(p / b)).sum(axis=1).where(cnt.notna().all(axis=1))


SYNODIC = 29.530588853
NEW_MOON_REF = pd.Timestamp("2000-01-06 18:14", tz="UTC")


@feature("F.ES.LUNAR_PHASE", F, "lunar phase from the mean synodic month (29.530588853 d) referenced to the 2000-01-06 "
         "18:14 UTC new moon: sin/cos of phase and days to full moon", inputs=(), **SPEC)
def f_lunar(df):
    days = (df.index - NEW_MOON_REF).total_seconds() / 86400.0
    ph = (days / SYNODIC) % 1.0
    return pd.DataFrame({"sin": np.sin(2 * np.pi * ph), "cos": np.cos(2 * np.pi * ph),
                         "days_to_full": ((0.5 - ph) % 1.0) * SYNODIC}, index=df.index)


@feature("F.ES.FIB_SESSION_TIME", F, "bars elapsed since the London session open (08:00 local) belongs to the Fibonacci "
         "set {1,2,3,5,8,13,...}; plus log-distance to nearest Fibonacci number", inputs=(), frequencies="M1..H1", **SPEC)
def f_fib_session(df):
    l = df.index.tz_convert("Europe/London")
    mins = (l.hour - 8) * 60 + l.minute
    step = max(1, int(round(pd.Series(df.index).diff().median().total_seconds() / 60))) if len(df) > 2 else 1
    bars = np.asarray(mins // step)
    ok = bars >= 0
    isfib = np.isin(bars, FIBS).astype(float)
    dist = np.min(np.abs(np.log((np.maximum(bars, 0)[:, None] + 1) / (FIBS[None, :] + 1))), axis=1)
    return pd.DataFrame({"is_fib": np.where(ok, isfib, np.nan), "dist": np.where(ok, dist, np.nan)}, index=df.index)


@njit(cache=True)
def _sax_next_up(r, wlen, alpha_edges):
    n = len(r)
    out = np.full(n, np.nan)
    # symbols per bar (computed on a trailing standardisation)
    for t in range(500, n):
        hist = r[t - 500:t + 1]
        mu = hist.mean(); sd = hist.std() + 1e-12
        sy = np.zeros(len(hist), np.int64)
        for i in range(len(hist)):
            z = (hist[i] - mu) / sd
            k = 0
            for e in alpha_edges:
                if z > e:
                    k += 1
            sy[i] = k
        cur = sy[-wlen:]
        hits = 0.0; ups = 0.0
        for i in range(wlen - 1, len(hist) - 1):
            same = True
            for q in range(wlen):
                if sy[i - wlen + 1 + q] != cur[q]:
                    same = False
                    break
            if same:
                hits += 1
                if hist[i + 1] > 0:
                    ups += 1
        out[t] = (ups + 1) / (hits + 2)
    return out


@feature("F.ES.SAX_RECURRENCE", F, "symbolic dynamics (SAX): 4-letter alphabet on standardised returns of the trailing 500 "
         "bars; Laplace-smoothed empirical P(next return > 0) after the current length-w word in that history",
         params={"w": 3}, space={"w": [2, 3, 4]}, complexity="O(n·500·w)", **SPEC)
def f_sax(df, w=3):
    r = np.log(df["close"]).diff().fillna(0).to_numpy(float)
    return as_series(_sax_next_up(r, w, np.array([-0.67, 0.0, 0.67])), df.index)


@njit(cache=True)
def _ordinal_markov(r, m, hist_len):
    """O(n): pattern codes computed once; transition counts maintained on a sliding window."""
    n = len(r)
    out = np.full(n, np.nan)
    codes = np.full(n, -1, np.int64)
    for i in range(m - 1, n):
        idx = np.argsort(r[i - m + 1:i + 1])
        c = 0
        for q in range(m):
            c = c * m + idx[q]
        codes[i] = c
    K = m ** m
    hits = np.zeros(K); ups = np.zeros(K)
    lo = m - 1  # first usable i (needs a code)
    for t in range(m, n):
        i_new = t - 1  # transition i -> i+1 completes at t
        if codes[i_new] >= 0:
            hits[codes[i_new]] += 1
            if r[i_new + 1] > 0:
                ups[codes[i_new]] += 1
        i_old = t - 1 - hist_len
        if i_old >= lo and codes[i_old] >= 0:
            hits[codes[i_old]] -= 1
            if r[i_old + 1] > 0:
                ups[codes[i_old]] -= 1
        if t >= hist_len:
            c = codes[t]
            out[t] = (ups[c] + 1) / (hits[c] + 2)
    return out


@feature("F.ES.ORDINAL_MARKOV", F, "ordinal-pattern Markov predictor: Laplace-smoothed empirical P(next return > 0) given "
         "the current order-m ordinal pattern, counted over the trailing history", params={"m": 3, "hist": 500},
         space={"m": [3, 4], "hist": [250, 500]}, complexity="O(n·hist·m log m)", **SPEC)
def f_ordinal_markov(df, m=3, hist=500):
    r = np.log(df["close"]).diff().fillna(0).to_numpy(float)
    return as_series(_ordinal_markov(r, m, hist), df.index)


@feature("F.ES.GOLDEN_GRID", F, "golden-ratio log grid: phase of ln C modulo ln(phi)/k (sin/cos), k scales the grid",
         params={"k": 100}, space={"k": [50, 100, 200]}, **SPEC)
def f_golden_grid(df, k=100):
    ph = (np.log(df["close"]) % (np.log(PHI) / k)) / (np.log(PHI) / k)
    return pd.DataFrame({"sin": np.sin(2 * np.pi * ph), "cos": np.cos(2 * np.pi * ph)})


@feature("F.ES.DATE_NUMEROLOGY", F, "date numerology: digital root (1..9) of the UTC calendar date digits YYYYMMDD, "
         "encoded as sin/cos of 2 pi root/9", inputs=(), **SPEC)
def f_numerology(df):
    s = df.index.year * 10000 + df.index.month * 100 + df.index.day
    root = 1 + (np.asarray(s) - 1) % 9
    return pd.DataFrame({"sin": np.sin(2 * np.pi * root / 9), "cos": np.cos(2 * np.pi * root / 9)}, index=df.index)
