"""7.3 Momentum and oscillators. Williams %R (= %K - 100), ROC (monotone in
F.PR.LOGRET_N) and PPO (MACD rescaled by EMA_s instead of C) are transforms of
registered entries and are therefore not counted separately."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, ema, sma, rma, true_range, as_series, rolling_apply

F = "momentum"
OHLC = ("open", "high", "low", "close")


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / (dn + EPS))


@feature("F.MO.RSI", F, "Wilder RSI = 100 - 100/(1 + RMA_n(up)/RMA_n(down))", params={"n": 14}, space={"n": [2, 7, 14, 21]})
def f_rsi(df, n=14):
    return _rsi(df["close"], n)


@feature("F.MO.STOCH", F, "%K = 100(C - min_n L)/(max_n H - min_n L), %D = SMA_d(%K)",
         params={"n": 14, "d": 3}, space={"n": [5, 14, 21], "d": [3]}, inputs=OHLC)
def f_stoch(df, n=14, d=3):
    lo = df["low"].rolling(n, min_periods=n).min()
    hi = df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - lo) / (hi - lo + EPS)
    return pd.DataFrame({"k": k, "d": sma(k, d)})


@feature("F.MO.STOCHRSI", F, "StochRSI = (RSI - min_m RSI)/(max_m RSI - min_m RSI)",
         params={"n": 14, "m": 14}, space={"n": [14], "m": [14, 21]})
def f_stochrsi(df, n=14, m=14):
    r = _rsi(df["close"], n)
    lo, hi = r.rolling(m, min_periods=m).min(), r.rolling(m, min_periods=m).max()
    return (r - lo) / (hi - lo + EPS)


@feature("F.MO.MACD", F, "MACD = EMA_f - EMA_s, signal = EMA_g(MACD), hist; all divided by C",
         params={"f": 12, "s": 26, "g": 9}, space={"f": [8, 12], "s": [21, 26, 50], "g": [9]})
def f_macd(df, f=12, s=26, g=9):
    c = df["close"]
    m = (ema(c, f) - ema(c, s)) / c
    sg = ema(m, g)
    return pd.DataFrame({"macd": m, "signal": sg, "hist": m - sg})


@feature("F.MO.CMO", F, "Chande momentum oscillator 100(SU-SD)/(SU+SD) over n", params={"n": 14}, space={"n": [9, 14, 20]})
def f_cmo(df, n=14):
    d = df["close"].diff()
    su = d.clip(lower=0).rolling(n, min_periods=n).sum()
    sd = (-d).clip(lower=0).rolling(n, min_periods=n).sum()
    return 100 * (su - sd) / (su + sd + EPS)


@feature("F.MO.CCI", F, "CCI = (TP - SMA_n TP)/(0.015 MAD_n(TP)), TP=(H+L+C)/3", params={"n": 20}, space={"n": [14, 20, 50]},
         inputs=OHLC)
def f_cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = sma(tp, n)
    mad = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * mad + EPS)


@feature("F.MO.TRIX", F, "TRIX = 1-bar % change of triple-smoothed EMA of ln C", params={"n": 15}, space={"n": [9, 15, 30]})
def f_trix(df, n=15):
    e = ema(ema(ema(np.log(df["close"]), n), n), n)
    return 1e4 * e.diff()


@feature("F.MO.TSI", F, "true strength index 100 EMA_s(EMA_r(ΔC))/EMA_s(EMA_r(|ΔC|))",
         params={"r": 25, "s": 13}, space={"r": [25], "s": [13]})
def f_tsi(df, r=25, s=13):
    d = df["close"].diff()
    return 100 * ema(ema(d, r), s) / (ema(ema(d.abs(), r), s) + EPS)


@njit(cache=True)
def _fisher(x):
    n = len(x)
    v = np.zeros(n); f = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(x[i]):
            v[i] = v[i - 1]
            continue
        v[i] = 0.33 * 2 * (x[i] - 0.5) + 0.67 * v[i - 1]
        v[i] = min(max(v[i], -0.999), 0.999)
        prev = f[i - 1] if not np.isnan(f[i - 1]) else 0.0
        f[i] = 0.5 * np.log((1 + v[i]) / (1 - v[i])) + 0.5 * prev
    return f


@feature("F.MO.FISHER", F, "Ehlers Fisher transform of normalised median price position in n-bar range",
         params={"n": 10}, space={"n": [10, 20]}, inputs=OHLC)
def f_fisher(df, n=10):
    m = (df["high"] + df["low"]) / 2
    lo, hi = m.rolling(n, min_periods=n).min(), m.rolling(n, min_periods=n).max()
    x = ((m - lo) / (hi - lo + EPS)).to_numpy(float)
    return as_series(_fisher(x), df.index).where(lo.notna())


@feature("F.MO.ULTIMATE", F, "Ultimate oscillator: weighted BP/TR averages over (7,14,28)", inputs=OHLC)
def f_ultimate(df):
    pc = df["close"].shift(1)
    bp = df["close"] - np.minimum(df["low"], pc)
    tr = true_range(df)
    a = [bp.rolling(n, min_periods=n).sum() / (tr.rolling(n, min_periods=n).sum() + EPS) for n in (7, 14, 28)]
    return 100 * (4 * a[0] + 2 * a[1] + a[2]) / 7


@feature("F.MO.AWESOME", F, "Awesome oscillator SMA_5(median) - SMA_34(median), divided by C", inputs=OHLC)
def f_awesome(df):
    m = (df["high"] + df["low"]) / 2
    return (sma(m, 5) - sma(m, 34)) / df["close"]


@feature("F.MO.KST", F, "Know-Sure-Thing: weighted sum of SMA-smoothed ROCs (10,15,20,30) and its signal")
def f_kst(df):
    c = df["close"]
    roc = lambda n: 100 * (c / c.shift(n) - 1)
    k = sma(roc(10), 10) + 2 * sma(roc(15), 10) + 3 * sma(roc(20), 10) + 4 * sma(roc(30), 15)
    return pd.DataFrame({"kst": k, "signal": sma(k, 9)})


@feature("F.MO.COPPOCK", F, "Coppock curve WMA_10(ROC_14 + ROC_11)")
def f_coppock(df):
    from fxlab.features._util import wma
    c = df["close"]
    return wma(100 * (c / c.shift(14) - 1) + 100 * (c / c.shift(11) - 1), 10)


@feature("F.MO.DPO", F, "causal detrended price oscillator C_{t-(n/2+1)} - SMA_n(t), / C",
         params={"n": 20}, space={"n": [14, 20, 40]})
def f_dpo(df, n=20):
    c = df["close"]
    return (c.shift(n // 2 + 1) - sma(c, n)) / c


@feature("F.MO.RSI_DIVERGENCE", F, "divergence score: +1 if price makes n-bar low below previous n-bar low while RSI does not "
         "(bullish), -1 mirrored (bearish), 0 otherwise",
         params={"n": 20, "rsi_n": 14}, space={"n": [10, 20, 40]})
def f_rsi_div(df, n=20, rsi_n=14):
    c = df["close"]
    r = _rsi(c, rsi_n)
    lo_now, lo_prev = c.rolling(n, min_periods=n).min(), c.shift(n).rolling(n, min_periods=n).min()
    rl_now, rl_prev = r.rolling(n, min_periods=n).min(), r.shift(n).rolling(n, min_periods=n).min()
    hi_now, hi_prev = c.rolling(n, min_periods=n).max(), c.shift(n).rolling(n, min_periods=n).max()
    rh_now, rh_prev = r.rolling(n, min_periods=n).max(), r.shift(n).rolling(n, min_periods=n).max()
    bull = (lo_now < lo_prev) & (rl_now > rl_prev)
    bear = (hi_now > hi_prev) & (rh_now < rh_prev)
    out = bull.astype(float) - bear.astype(float)
    return out.where(rh_prev.notna())


@feature("F.MO.CONNORS_RSI", F, "Connors RSI = mean(RSI_3(C), RSI_2(streak), pct-rank_100 of 1-bar ROC)")
def f_connors(df):
    from fxlab.features.price import streak
    c = df["close"]
    s = streak(df)
    pr = (c / c.shift(1) - 1).rolling(100, min_periods=100).rank(pct=True) * 100
    return (_rsi(c, 3) + _rsi(s, 2) + pr) / 3


@feature("F.MO.RVI", F, "relative vigor index: SWMA(C-O)/SWMA(H-L) summed over n, with 4-bar symmetric weights",
         params={"n": 10}, space={"n": [10, 20]}, inputs=OHLC)
def f_rvi(df, n=10):
    from fxlab.features._util import conv_causal
    w = np.array([1, 2, 2, 1], float) / 6
    num = conv_causal(df["close"] - df["open"], w).rolling(n, min_periods=n).sum()
    den = conv_causal(df["high"] - df["low"], w).rolling(n, min_periods=n).sum()
    rvi = num / (den + EPS)
    return pd.DataFrame({"rvi": rvi, "signal": conv_causal(rvi, w)})


@feature("F.MO.SCHAFF", F, "Schaff trend cycle: double stochastic of MACD(23,50) with factor 0.5",
         params={"cycle": 10}, space={"cycle": [10, 20]})
def f_schaff(df, cycle=10):
    c = df["close"]
    m = ema(c, 23) - ema(c, 50)
    def stoch_smooth(x):
        lo, hi = x.rolling(cycle, min_periods=cycle).min(), x.rolling(cycle, min_periods=cycle).max()
        k = 100 * (x - lo) / (hi - lo + EPS)
        return k.ewm(alpha=0.5, adjust=False).mean()
    return stoch_smooth(stoch_smooth(m))


@feature("F.MO.ELDER_RAY", F, "Elder ray: bull power H - EMA_n, bear power L - EMA_n, divided by ATR",
         params={"n": 13}, space={"n": [13, 26]}, inputs=OHLC)
def f_elder(df, n=13):
    from fxlab.features._util import atr
    e = ema(df["close"], n)
    a = atr(df, 14)
    return pd.DataFrame({"bull": (df["high"] - e) / (a + EPS), "bear": (df["low"] - e) / (a + EPS)})


@feature("F.MO.MASS_INDEX", F, "Mass index: sum_25 EMA_9(H-L)/EMA_9(EMA_9(H-L)) (range-expansion reversal)", inputs=OHLC)
def f_mass(df):
    r = df["high"] - df["low"]
    e1 = ema(r, 9)
    return (e1 / (ema(e1, 9) + EPS)).rolling(25, min_periods=25).sum()


@feature("F.MO.CFO", F, "Chande forecast oscillator 100 (C - TSF_n)/C, TSF = OLS forecast at t from last n closes",
         params={"n": 14}, space={"n": [14, 28]})
def f_cfo(df, n=14):
    from fxlab.features._util import rolling_slope
    c = df["close"]
    b, _ = rolling_slope(c, n)
    mean = sma(c, n)
    tsf = mean + b * (n - 1) / 2.0
    return 100 * (c - tsf) / c


@feature("F.MO.RWI", F, "random walk index: max_k (H_t - L_{t-k})/(ATR_k sqrt k) and (H_{t-k} - L_t)/(ATR sqrt k), k=2..n",
         params={"n": 14}, space={"n": [8, 14]}, inputs=OHLC)
def f_rwi(df, n=14):
    tr = true_range(df)
    hi_best = pd.Series(-np.inf, index=df.index)
    lo_best = pd.Series(-np.inf, index=df.index)
    for k in range(2, n + 1):
        a = tr.rolling(k, min_periods=k).mean() * np.sqrt(k)
        hi_best = np.maximum(hi_best, (df["high"] - df["low"].shift(k)) / (a + EPS))
        lo_best = np.maximum(lo_best, (df["high"].shift(k) - df["low"]) / (a + EPS))
    ok = tr.rolling(n, min_periods=n).mean().notna()
    return pd.DataFrame({"rwi_hi": hi_best.where(ok), "rwi_lo": lo_best.where(ok)})
