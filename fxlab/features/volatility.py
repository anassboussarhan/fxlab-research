"""7.4 Volatility estimators. Range estimators use mid OHLC; the intrabar
realised measures use the M1 sub-bars recorded at aggregation time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, atr, ema, sma, logret, true_range, as_series, rolling_apply

F = "volatility"
OHLC = ("open", "high", "low", "close")
NS = {"n": [10, 20, 50, 100]}


@feature("F.VO.ATR", F, "Wilder ATR_n / C (normalised average true range)", params={"n": 14}, space={"n": [7, 14, 50]}, inputs=OHLC)
def f_atr(df, n=14):
    return atr(df, n) / df["close"]


@feature("F.VO.HV", F, "close-to-close volatility std_n(l)", params={"n": 20}, space=NS)
def f_hv(df, n=20):
    return logret(df["close"]).rolling(n, min_periods=n).std()


@feature("F.VO.RV_INTRABAR", F, "intrabar realised variance RV_t = sum of squared M1 log returns inside bar t, "
         "averaged over n bars (sqrt)", params={"n": 1}, space={"n": [1, 5, 20]}, inputs=("rv_intrabar",),
         frequencies="M2..D1")
def f_rv_intrabar(df, n=1):
    return np.sqrt(df["rv_intrabar"].rolling(n, min_periods=n).mean())


@feature("F.VO.PARKINSON", F, "Parkinson sigma^2 = (1/(4 ln2)) mean_n (ln H/L)^2", params={"n": 20}, space=NS, inputs=OHLC)
def f_parkinson(df, n=20):
    return np.sqrt((np.log(df["high"] / df["low"]) ** 2).rolling(n, min_periods=n).mean() / (4 * np.log(2)))


@feature("F.VO.GARMAN_KLASS", F, "GK sigma^2 = mean_n[0.5(ln H/L)^2 - (2ln2-1)(ln C/O)^2]", params={"n": 20}, space=NS, inputs=OHLC)
def f_gk(df, n=20):
    v = 0.5 * np.log(df["high"] / df["low"]) ** 2 - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
    return np.sqrt(v.rolling(n, min_periods=n).mean().clip(lower=0))


def _rs_terms(df):
    h, l, o, c = (np.log(df[k]) for k in ("high", "low", "open", "close"))
    return (h - c) * (h - o) + (l - c) * (l - o)


@feature("F.VO.ROGERS_SATCHELL", F, "RS sigma^2 = mean_n[ln(H/C)ln(H/O) + ln(L/C)ln(L/O)] (drift-robust)",
         params={"n": 20}, space=NS, inputs=OHLC)
def f_rs(df, n=20):
    return np.sqrt(_rs_terms(df).rolling(n, min_periods=n).mean().clip(lower=0))


@feature("F.VO.YANG_ZHANG", F, "Yang-Zhang sigma^2 = s_o^2 + k s_c^2 + (1-k) s_RS^2, k = 0.34/(1.34+(n+1)/(n-1))",
         params={"n": 20}, space=NS, inputs=OHLC)
def f_yz(df, n=20):
    o = np.log(df["open"] / df["close"].shift(1))
    c = np.log(df["close"] / df["open"])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v = o.rolling(n, min_periods=n).var() + k * c.rolling(n, min_periods=n).var() + \
        (1 - k) * _rs_terms(df).rolling(n, min_periods=n).mean()
    return np.sqrt(v.clip(lower=0))


@feature("F.VO.BIPOWER", F, "bipower variation BV = (pi/2) sum_n |l_t||l_{t-1}| (jump-robust)", params={"n": 20}, space=NS)
def f_bipower(df, n=20):
    a = logret(df["close"]).abs()
    return np.sqrt((np.pi / 2) * (a * a.shift(1)).rolling(n, min_periods=n).mean())


@feature("F.VO.SEMIVAR", F, "realised semivariances RS- = sum l^2 1[l<0], RS+ = sum l^2 1[l>0]; outputs sqrt and signed jump RS+-RS-",
         params={"n": 20}, space=NS)
def f_semivar(df, n=20):
    lr = logret(df["close"])
    dn = (lr.clip(upper=0) ** 2).rolling(n, min_periods=n).sum()
    up = (lr.clip(lower=0) ** 2).rolling(n, min_periods=n).sum()
    return pd.DataFrame({"down": np.sqrt(dn), "up": np.sqrt(up), "signed_jump": (up - dn) / (up + dn + EPS)})


@feature("F.VO.VOL_OF_VOL", F, "std_m of log(HV_n) (volatility of volatility)", params={"n": 20, "m": 50},
         space={"n": [10, 20], "m": [50, 100]})
def f_volvol(df, n=20, m=50):
    hv = logret(df["close"]).rolling(n, min_periods=n).std()
    return np.log(hv + EPS).rolling(m, min_periods=m).std()


@feature("F.VO.JUMP_RATIO", F, "BNS relative jump RJ = (RV - BV)/RV over n bars (share of variance from jumps)",
         params={"n": 50}, space={"n": [20, 50, 100]})
def f_jump_ratio(df, n=50):
    lr = logret(df["close"])
    rv = (lr ** 2).rolling(n, min_periods=n).sum()
    bv = (np.pi / 2) * (lr.abs() * lr.abs().shift(1)).rolling(n, min_periods=n).sum()
    return ((rv - bv) / (rv + EPS)).clip(lower=0)


@feature("F.VO.LEE_MYKLAND", F, "Lee-Mykland jump statistic L_t = l_t / sqrt(BV_{t-1}/ (n-2)) using bipower over the previous n bars",
         params={"n": 50}, space={"n": [50, 100]})
def f_lee_mykland(df, n=50):
    lr = logret(df["close"])
    bpv = (lr.abs() * lr.abs().shift(1)).rolling(n - 1, min_periods=n - 1).mean().shift(1)
    return lr / (np.sqrt(bpv) + EPS)


@feature("F.VO.CLUSTERING", F, "volatility clustering: rolling autocorrelation of |l| at lag 1 over n bars",
         params={"n": 100}, space={"n": [50, 100, 250]})
def f_clustering(df, n=100):
    a = logret(df["close"]).abs()
    return a.rolling(n, min_periods=n).corr(a.shift(1))


@feature("F.VO.EWMA", F, "RiskMetrics EWMA volatility sigma_t^2 = lambda sigma_{t-1}^2 + (1-lambda) l_t^2",
         params={"lam": 0.94}, space={"lam": [0.9, 0.94, 0.97, 0.99]})
def f_ewma_vol(df, lam=0.94):
    lr = logret(df["close"])
    return np.sqrt((lr ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=int(1 / (1 - lam))).mean())


@feature("F.VO.BB_WIDTH", F, "Bollinger bandwidth 2k std_n(C)/SMA_n(C) and %B position", params={"n": 20, "k": 2.0},
         space={"n": [20, 50], "k": [2.0]})
def f_bbw(df, n=20, k=2.0):
    m = sma(df["close"], n)
    s = df["close"].rolling(n, min_periods=n).std()
    return pd.DataFrame({"width": 2 * k * s / m, "pctb": (df["close"] - (m - k * s)) / (2 * k * s + EPS)})


@feature("F.VO.KELTNER", F, "Keltner position (C - EMA_n)/(k ATR_n)", params={"n": 20, "k": 2.0}, space={"n": [20, 50], "k": [1.5, 2.0]},
         inputs=OHLC)
def f_keltner(df, n=20, k=2.0):
    return (df["close"] - ema(df["close"], n)) / (k * atr(df, n) + EPS)


@feature("F.VO.CHAIKIN_VOL", F, "Chaikin volatility: % change over m bars of EMA_n(H-L)", params={"n": 10, "m": 10},
         space={"n": [10], "m": [10, 20]}, inputs=OHLC)
def f_chaikin_vol(df, n=10, m=10):
    e = ema(df["high"] - df["low"], n)
    return e / e.shift(m) - 1


@feature("F.VO.ULCER", F, "Ulcer index sqrt(mean_n DD_t^2), DD = 100 (C/max_n C - 1)", params={"n": 14}, space={"n": [14, 50]})
def f_ulcer(df, n=14):
    c = df["close"]
    dd = 100 * (c / c.rolling(n, min_periods=n).max() - 1)
    return np.sqrt((dd ** 2).rolling(n, min_periods=n).mean())


@feature("F.VO.ATR_RATIO", F, "short/long ATR ratio ATR_s/ATR_l (volatility term structure)",
         params={"s": 5, "l": 50}, space={"s": [5, 10], "l": [50, 100]}, inputs=OHLC)
def f_atr_ratio(df, s=5, l=50):
    return atr(df, s) / (atr(df, l) + EPS)


@feature("F.VO.QUARTICITY", F, "realised quarticity RQ = (n/3) sum l^4, reported as RQ / RV^2 (vol-of-variance proxy)",
         params={"n": 50}, space={"n": [20, 50]})
def f_quarticity(df, n=50):
    lr = logret(df["close"])
    rq = (n / 3) * (lr ** 4).rolling(n, min_periods=n).sum()
    rv = (lr ** 2).rolling(n, min_periods=n).sum()
    return rq / (rv ** 2 + EPS)


@feature("F.VO.DONCHIAN_WIDTH", F, "Donchian width ln(max_n H / min_n L)", params={"n": 20}, space=NS, inputs=OHLC)
def f_donchian_width(df, n=20):
    return np.log(df["high"].rolling(n, min_periods=n).max() / df["low"].rolling(n, min_periods=n).min())


@feature("F.VO.MEDRV", F, "MedRV (Andersen-Dobrev-Schaumburg): pi/(6-4sqrt3+pi) n/(n-2) sum med(|l_{t-1}|,|l_t|,|l_{t+1}|)^2 "
         "computed on the trailing window (the median uses only returns inside the window)", params={"n": 50}, space={"n": [20, 50]})
def f_medrv(df, n=50):
    a = logret(df["close"]).abs()
    med = pd.concat([a.shift(2), a.shift(1), a], axis=1).median(axis=1, skipna=False)
    c = np.pi / (6 - 4 * np.sqrt(3) + np.pi)
    return np.sqrt(c * (n / (n - 2)) * (med ** 2).rolling(n - 2, min_periods=n - 2).sum() / n)
