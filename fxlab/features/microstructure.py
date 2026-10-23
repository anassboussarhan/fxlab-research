"""7.6 Market microstructure from real Dukascopy quote data.

Available per bar: bid/ask OHLC, quote count (n_ticks), up/down mid-quote
ticks, mean/max/min quoted spread, and Dukascopy quoted liquidity (bid_liq,
ask_liq). There is NO order book depth beyond top of book and NO transaction
prints in OTC FX; trade-flow features below are explicitly *quote-based proxies*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.registry import feature
from fxlab.features._util import EPS, atr, logret, rolling_slope, sma

F = "microstructure"
Q = ("spread_mean",)
NS = {"n": [10, 20, 50]}


@feature("F.MS.SPREAD_COST", F, "quoted spread relative to volatility: spread_mean / ATR_14 and spread_mean / C (bps)",
         inputs=("spread_mean", "high", "low", "close"))
def f_spread_cost(df):
    return pd.DataFrame({"to_atr": df["spread_mean"] / (atr(df, 14) + EPS), "bps": 1e4 * df["spread_mean"] / df["close"]})


@feature("F.MS.SPREAD_Z", F, "spread variability: z-score of ln spread_mean vs trailing n", params={"n": 100},
         space={"n": [50, 100, 500]}, inputs=Q)
def f_spread_z(df, n=100):
    ls = np.log(df["spread_mean"] + EPS)
    return (ls - ls.rolling(n, min_periods=n).mean()) / (ls.rolling(n, min_periods=n).std() + EPS)


@feature("F.MS.TICK_IMBALANCE", F, "signed quote flow sum(up - down)/sum(up + down) over n bars", params={"n": 1},
         space={"n": [1, 5, 20]}, inputs=("up_ticks", "down_ticks"))
def f_tick_imb(df, n=1):
    u = df["up_ticks"].rolling(n, min_periods=n).sum()
    d = df["down_ticks"].rolling(n, min_periods=n).sum()
    return (u - d) / (u + d + EPS)


@feature("F.MS.QUOTE_INTENSITY", F, "quote arrival intensity ln(n_ticks / median_n n_ticks)", params={"n": 100},
         space={"n": [50, 100, 500]}, inputs=("n_ticks",))
def f_intensity(df, n=100):
    t = df["n_ticks"].astype(float)
    return np.log((t + 1) / (t.rolling(n, min_periods=n).median() + 1))


@feature("F.MS.LIQ_IMBALANCE", F, "quoted liquidity imbalance (bid_liq - ask_liq)/(bid_liq + ask_liq), mean over n",
         params={"n": 5}, space={"n": [1, 5, 20]}, inputs=("bid_liq", "ask_liq"))
def f_liq_imb(df, n=5):
    b, a = df["bid_liq"].astype(float), df["ask_liq"].astype(float)
    return ((b - a) / (b + a + EPS)).rolling(n, min_periods=n).mean()


@feature("F.MS.AMIHUD", F, "Amihud-style illiquidity mean_n(|l| / n_ticks) * 1e6", params={"n": 20}, space=NS,
         inputs=("close", "n_ticks"))
def f_amihud(df, n=20):
    return (logret(df["close"]).abs() / (df["n_ticks"] + 1)).rolling(n, min_periods=n).mean() * 1e6


@feature("F.MS.ROLL_SPREAD", F, "Roll (1984) implied spread 2 sqrt(-cov(ΔC_t, ΔC_{t-1})) (0 if cov>0), / C",
         params={"n": 50}, space={"n": [20, 50, 100]})
def f_roll(df, n=50):
    d = df["close"].diff()
    cov = d.rolling(n, min_periods=n).cov(d.shift(1))
    return 2 * np.sqrt((-cov).clip(lower=0)) / df["close"]


@feature("F.MS.CORWIN_SCHULTZ", F, "Corwin-Schultz (2012) high-low spread estimator S = 2(e^a - 1)/(1 + e^a), averaged over n",
         params={"n": 20}, space=NS, inputs=("high", "low"))
def f_corwin(df, n=20):
    h, l = df["high"], df["low"]
    beta = np.log(h / l) ** 2 + np.log(h.shift(1) / l.shift(1)) ** 2
    gamma = np.log(np.maximum(h, h.shift(1)) / np.minimum(l, l.shift(1))) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0).rolling(n, min_periods=n).mean()


@feature("F.MS.ABDI_RANALDO", F, "Abdi-Ranaldo (2017) CHL spread: s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})] using "
         "c=ln C, eta=ln((H+L)/2); trailing version pairs (c_{t-1}, eta_{t-1}, eta_t)", params={"n": 20}, space=NS,
         inputs=("high", "low", "close"))
def f_abdi(df, n=20):
    c = np.log(df["close"])
    eta = np.log((df["high"] + df["low"]) / 2)
    term = (c.shift(1) - eta.shift(1)) * (c.shift(1) - eta)
    return np.sqrt((4 * term.rolling(n, min_periods=n).mean()).clip(lower=0))


@feature("F.MS.NOISE_RATIO", F, "microstructure-noise signature ratio: mean_n RV_intrabar(M1) / mean_n l_bar^2 "
         "(>1 indicates noise / mean-reversion at M1, <1 momentum)", params={"n": 50}, space={"n": [20, 50]},
         inputs=("rv_intrabar", "close"), frequencies="M5..D1")
def f_noise_ratio(df, n=50):
    lr2 = logret(df["close"]) ** 2
    return df["rv_intrabar"].rolling(n, min_periods=n).mean() / (lr2.rolling(n, min_periods=n).mean() + EPS)


@feature("F.MS.KYLE_LAMBDA", F, "Kyle lambda proxy: rolling OLS slope of l_t on signed quote flow (up-down)/sqrt(n_ticks)",
         params={"n": 100}, space={"n": [50, 100, 250]}, inputs=("close", "up_ticks", "down_ticks", "n_ticks"))
def f_kyle(df, n=100):
    x = (df["up_ticks"] - df["down_ticks"]) / np.sqrt(df["n_ticks"] + 1)
    y = logret(df["close"])
    cov = y.rolling(n, min_periods=n).cov(x)
    var = x.rolling(n, min_periods=n).var()
    return 1e4 * cov / (var + EPS)


@feature("F.MS.VPIN", F, "VPIN with bulk-volume classification on tick volume: V_buy = V Phi(Δp/sigma); "
         "VPIN = sum|V_buy - V_sell| / sum V over n bars", params={"n": 50}, space={"n": [20, 50]},
         inputs=("close", "n_ticks"))
def f_vpin(df, n=50):
    from scipy.stats import norm
    dp = df["close"].diff()
    sig = dp.rolling(n, min_periods=n).std().shift(1)
    v = df["n_ticks"].astype(float)
    vb = v * pd.Series(norm.cdf((dp / (sig + EPS)).to_numpy(float)), index=df.index)
    imb = (2 * vb - v).abs()
    return (imb.rolling(n, min_periods=n).sum() / (v.rolling(n, min_periods=n).sum() + EPS)).where(sig.notna())


@feature("F.MS.SPREAD_TO_RANGE", F, "spread_mean / (H - L) (fraction of bar range eaten by spread)",
         inputs=("spread_mean", "high", "low"))
def f_spread_range(df):
    return df["spread_mean"] / (df["high"] - df["low"] + EPS)


@feature("F.MS.SPREAD_SPIKE", F, "intrabar spread widening ln(spread_max / spread_mean)", inputs=("spread_max", "spread_mean"))
def f_spread_spike(df):
    return np.log((df["spread_max"] + EPS) / (df["spread_mean"] + EPS))


@feature("F.MS.MFI_TICK", F, "money flow index with tick volume: 100 - 100/(1 + PMF/NMF) on typical price",
         params={"n": 14}, space={"n": [14, 28]}, inputs=("high", "low", "close", "n_ticks"))
def f_mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["n_ticks"]
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n).sum()
    return 100 - 100 / (1 + pos / (neg + EPS))


@feature("F.MS.OBV_TICK", F, "on-balance tick volume: OBV_t = OBV_{t-1} + sign(ΔC) V_t; feature = n-bar change / sum V",
         params={"n": 20}, space=NS, inputs=("close", "n_ticks"))
def f_obv(df, n=20):
    v = df["n_ticks"].astype(float)
    signed = np.sign(df["close"].diff()).fillna(0) * v
    return signed.rolling(n, min_periods=n).sum() / (v.rolling(n, min_periods=n).sum() + EPS)


@feature("F.MS.CHAIKIN_AD", F, "Chaikin oscillator EMA_3(AD) - EMA_10(AD), AD = cumsum(CLV * V), normalised by EMA_10(V)",
         inputs=("high", "low", "close", "n_ticks"))
def f_chaikin_ad(df):
    from fxlab.features._util import ema
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + EPS)
    v = df["n_ticks"].astype(float)
    mfv = clv * v
    # oscillator of the cumulative line equals oscillator of its increments' EMA difference; use increments to stay prefix-stable
    return (ema(mfv, 3) - ema(mfv, 10)) / (ema(v, 10) + EPS)


@feature("F.MS.FORCE_TICK", F, "Elder force index EMA_n(ΔC * V) / (ATR * EMA_n(V))", params={"n": 13}, space={"n": [2, 13]},
         inputs=("high", "low", "close", "n_ticks"))
def f_force(df, n=13):
    from fxlab.features._util import ema
    v = df["n_ticks"].astype(float)
    return ema(df["close"].diff() * v, n) / (atr(df, 14) * ema(v, n) + EPS)


@feature("F.MS.EOM_TICK", F, "ease of movement: SMA_n(Δmidpoint * (H-L) / V), scaled", params={"n": 14}, space={"n": [14]},
         inputs=("high", "low", "n_ticks"))
def f_eom(df, n=14):
    mid = (df["high"] + df["low"]) / 2
    v = df["n_ticks"].astype(float) + 1
    e = mid.diff() * (df["high"] - df["low"]) / v
    return sma(e, n) / (sma((df["high"] - df["low"]) ** 2 / v, n) + EPS)


@feature("F.MS.TWAP_TICK", F, "distance of C to tick-weighted average price over n bars, in ATR",
         params={"n": 20}, space={"n": [20, 50, 100]}, inputs=("high", "low", "close", "n_ticks"))
def f_twap(df, n=20):
    v = df["n_ticks"].astype(float)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vw = (tp * v).rolling(n, min_periods=n).sum() / (v.rolling(n, min_periods=n).sum() + EPS)
    return (df["close"] - vw) / (atr(df, 14) + EPS)


@feature("F.MS.QUOTE_STALENESS", F, "quote staleness: fraction of the last n bars with no mid-quote change "
         "(up_ticks + down_ticks == 0) and mean zero-change share of ticks (1 - (up+down)/n_ticks)",
         params={"n": 30}, space={"n": [10, 30, 100]}, inputs=("up_ticks", "down_ticks", "n_ticks"))
def f_staleness(df, n=30):
    moves = df["up_ticks"] + df["down_ticks"]
    stale_bar = (moves == 0).astype(float)
    zero_share = 1 - moves / (df["n_ticks"].clip(lower=1))
    return pd.DataFrame({"stale_bars": stale_bar.rolling(n, min_periods=n).mean(),
                         "zero_change_share": zero_share.rolling(n, min_periods=n).mean()})
