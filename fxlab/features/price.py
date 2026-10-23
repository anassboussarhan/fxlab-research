"""7.1 Price and return dynamics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.registry import feature
from fxlab.features._util import EPS, logret, true_range, as_series

F = "price_returns"
OHLC = ("open", "high", "low", "close")


@feature("F.PR.LOGRET_N", F, "n-bar log return / multi-horizon momentum M_t(n) = ln C_t - ln C_{t-n} = sum_{i<n} l_{t-i} (n=1 is the 1-bar log return; simple return is a monotone transform and is not counted separately)",
         params={"n": 20}, space={"n": [1, 2, 5, 10, 20, 50, 100, 200]})
def cumulative_logret(df, n=20):
    return logret(df["close"]).rolling(n, min_periods=n).sum()


@feature("F.PR.UP_FRACTION", F, "directional persistence p_t = mean(1[l_{t-i}>0], i<n)",
         params={"n": 20}, space={"n": [10, 20, 50]})
def up_fraction(df, n=20):
    lr = logret(df["close"])
    return (lr > 0).astype(float).where(lr.notna()).rolling(n, min_periods=n).mean()


@njit(cache=True)
def _streak(x):
    out = np.zeros(len(x))
    s = 0.0
    for i in range(len(x)):
        v = x[i]
        if np.isnan(v) or v == 0:
            s = 0.0
        elif v > 0:
            s = s + 1 if s > 0 else 1.0
        else:
            s = s - 1 if s < 0 else -1.0
        out[i] = s
    return out


@feature("F.PR.STREAK", F, "signed length of the current run of same-sign returns")
def streak(df):
    return as_series(_streak(df["close"].diff().to_numpy(float)), df.index)


@feature("F.PR.ACCEL", F, "a_t = M_t(n) - M_{t-n}(n), change in n-bar log momentum",
         params={"n": 10}, space={"n": [5, 10, 20]})
def acceleration(df, n=10):
    m = logret(df["close"]).rolling(n, min_periods=n).sum()
    return m - m.shift(n)


@feature("F.PR.JERK", F, "third difference of log price Δ³ ln C_t, smoothed by n-bar mean",
         params={"n": 5}, space={"n": [1, 5, 10]})
def jerk(df, n=5):
    return np.log(df["close"]).diff().diff().diff().rolling(n, min_periods=n).mean()


@feature("F.PR.EFFICIENCY", F, "Kaufman ER = |C_t - C_{t-n}| / sum |ΔC|", params={"n": 20}, space={"n": [10, 20, 50]})
def efficiency_ratio(df, n=20):
    c = df["close"]
    return (c - c.shift(n)).abs() / (c.diff().abs().rolling(n, min_periods=n).sum() + EPS)


@feature("F.PR.PRICE_Z", F, "z_t = (C_t - mean_n C) / std_n C", params={"n": 50}, space={"n": [20, 50, 100, 200]})
def price_zscore(df, n=50):
    c = df["close"]
    return (c - c.rolling(n, min_periods=n).mean()) / (c.rolling(n, min_periods=n).std() + EPS)


@feature("F.PR.PCT_RANK", F, "percentile rank of C_t within last n closes (rank normalisation)",
         params={"n": 100}, space={"n": [50, 100, 250]})
def percentile_rank(df, n=100):
    return df["close"].rolling(n, min_periods=n).rank(pct=True)


@feature("F.PR.CANDLE_ANATOMY", F, "candle structure: body=|C-O|/(H-L), upper=(H-max(O,C))/(H-L), lower=(min(O,C)-L)/(H-L)", inputs=OHLC)
def candle_anatomy(df):
    rng = df["high"] - df["low"] + EPS
    return pd.DataFrame({"body": (df["close"] - df["open"]).abs() / rng,
                         "upper": (df["high"] - df[["open", "close"]].max(axis=1)) / rng,
                         "lower": (df[["open", "close"]].min(axis=1) - df["low"]) / rng})


@feature("F.PR.CLV", F, "close location value ((C-L)-(H-C))/(H-L)", inputs=OHLC)
def close_location_value(df):
    return ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + EPS)


@feature("F.PR.EXTREMA_DIST", F, "ln(C/max_n H) and ln(C/min_n L): distance to rolling extrema",
         params={"n": 50}, space={"n": [20, 50, 100, 250]}, inputs=OHLC)
def extrema_distance(df, n=50):
    return pd.DataFrame({"to_max": np.log(df["close"] / df["high"].rolling(n, min_periods=n).max()),
                         "to_min": np.log(df["close"] / df["low"].rolling(n, min_periods=n).min())})


@feature("F.PR.GAP", F, "gap g_t = (O_t - C_{t-1}) / ATR_n(t-1)", params={"n": 14}, space={"n": [14, 50]}, inputs=OHLC)
def gap(df, n=14):
    a = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().shift(1)
    return (df["open"] - df["close"].shift(1)) / (a + EPS)


@feature("F.PR.RANGE_RATIO", F, "(H_t-L_t) / mean_n(H-L)", params={"n": 20}, space={"n": [10, 20, 50]}, inputs=OHLC)
def range_ratio(df, n=20):
    r = df["high"] - df["low"]
    return r / (r.rolling(n, min_periods=n).mean() + EPS)


@feature("F.PR.INTRABAR_RET", F, "ln(C_t/O_t)", inputs=("open", "close"))
def intrabar_return(df):
    return np.log(df["close"] / df["open"])


@feature("F.PR.RET_TSTAT", F, "t-stat of mean log return: mean_n(l)/std_n(l)*sqrt(n)",
         params={"n": 50}, space={"n": [20, 50, 100]})
def return_tstat(df, n=50):
    lr = logret(df["close"])
    return lr.rolling(n, min_periods=n).mean() / (lr.rolling(n, min_periods=n).std() + EPS) * np.sqrt(n)


def _ffd_weights(d, thr=1e-4, max_len=500):
    w = [1.0]
    k = 1
    while k < max_len:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thr:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])  # oldest..newest


@feature("F.PR.FRACDIFF", F, "fixed-width-window fractional difference of ln C (López de Prado): "
         "x_t = sum_k w_k ln C_{t-k}, w_k = -w_{k-1}(d-k+1)/k, truncated at |w|<1e-4",
         params={"d": 0.4}, space={"d": [0.2, 0.3, 0.4, 0.5, 0.6]}, complexity="O(n·L)")
def frac_diff(df, d=0.4):
    from fxlab.features._util import conv_causal
    return conv_causal(np.log(df["close"]), _ffd_weights(d))

