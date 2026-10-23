"""Rule-based strategy families (trend, mean reversion, breakout, structure,
geometry, sessions/seasonality, volatility, regimes, cross-asset, signal
processing, microstructure). Every family has symmetric long/short logic,
explicit entry and exit rules, a parameter space, and a default RiskConfig
(stop / target / sizing) applied by the evaluation layer unless the strategy
sets its own levels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.registry import strategy
from fxlab.strategies.base import latch, edge, out, compute, session_hours, local_day
from fxlab.features._util import atr, ema, sma, EPS

ST = {"mode": "state"}
TR = {"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine",
                                "target_params": {"a": 0.0, "b": 1.5}, "max_hold": 48}}
TR_NOTP = {"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 2.5}, "target": "none",
                                     "target_params": {}, "trail_atr": 3.0, "max_hold": 240}}
PANEL = dict(multi_asset=True, instruments="panel")


def _c(df):
    return df["close"]


# =============================================================================== trend / momentum
@strategy("S.TREND.REGRESSION", "trend", "trend following on OLS trend t-stat: long if slope-t > thr, short if < -thr; "
          "exit when |t| falls below thr/2", params={"n": 100, "thr": 2.0}, space={"n": [50, 100, 200], "thr": [1.5, 2.0, 3.0]},
          extra=ST)
def s_trend_regression(df, n=100, thr=2.0):
    t = compute(df, "F.TR.REG_SLOPE", n=n, col="slope") * np.sqrt(n)
    return out(latch(t > thr, t < -thr, t < thr / 2, t > -thr / 2))


@strategy("S.TREND.MA_CROSS", "trend", "EMA(f) vs EMA(s) crossover: position = sign(EMA_f - EMA_s)",
          params={"f": 20, "s": 100}, space={"f": [10, 20, 50], "s": [50, 100, 200]}, extra=ST)
def s_ma_cross(df, f=20, s=100):
    return out(np.sign(ema(_c(df), f) - ema(_c(df), s)))


@strategy("S.TREND.ADAPTIVE", "trend", "adaptive trend: KAMA slope direction, only while efficiency ratio > e_min",
          params={"n": 10, "e_min": 0.3}, space={"n": [10, 20], "e_min": [0.2, 0.3, 0.4]}, extra=ST)
def s_adaptive(df, n=10, e_min=0.3):
    from fxlab.features.trend import _kama
    k = pd.Series(_kama(_c(df).to_numpy(float), n, 2, 30), index=df.index)
    er = compute(df, "F.PR.EFFICIENCY", n=n)
    return out(np.sign(k.diff()) * (er > e_min))


@strategy("S.TREND.TSMOM", "trend", "time-series momentum (Moskowitz et al.): sign of n-bar return, scaled to target "
          "volatility (position in [-1, 1])", params={"n": 120, "vol_n": 100}, space={"n": [24, 120, 480]}, extra=ST)
def s_tsmom(df, n=120, vol_n=100):
    lr = np.log(_c(df)).diff()
    sig = np.sign(lr.rolling(n, min_periods=n).sum())
    vol = lr.rolling(vol_n, min_periods=vol_n).std()
    scale = (vol.rolling(2000, min_periods=200).median() / (vol + EPS)).clip(0, 1)
    return out(sig * scale)


@strategy("S.TREND.XS_MOMENTUM", "trend", "cross-sectional momentum: rank currencies by n-bar strength across the panel; "
          "long the pair if base is in the top third and quote in the bottom third (short mirrored)",
          params={"n": 120}, space={"n": [24, 120, 480]}, extra=ST, **PANEL)
def s_xs_mom(df, panel, symbol, n=120):
    from fxlab.features.cross_asset import currency_strength, _with_target
    S = currency_strength(_with_target(df, panel, symbol), n)
    S = S[[c for c in S.columns if c not in ("XAU", "XAG")]]
    rk = S.rank(axis=1, pct=True)
    b, q = symbol[:3], symbol[3:]
    if b not in rk or q not in rk:
        return out(pd.Series(0.0, index=df.index))
    long = (rk[b] > 2 / 3) & (rk[q] < 1 / 3)
    short = (rk[b] < 1 / 3) & (rk[q] > 2 / 3)
    return out(long.astype(float) - short.astype(float))


@strategy("S.TREND.DONCHIAN", "breakout", "Donchian/Turtle breakout: long on close above prior n-bar high, short below "
          "prior n-bar low; exit on opposite m-bar channel", params={"n": 55, "m": 20},
          space={"n": [20, 55, 100], "m": [10, 20]}, extra=ST)
def s_donchian(df, n=55, m=20):
    hh = df["high"].shift(1).rolling(n, min_periods=n).max(); ll = df["low"].shift(1).rolling(n, min_periods=n).min()
    xh = df["high"].shift(1).rolling(m, min_periods=m).max(); xl = df["low"].shift(1).rolling(m, min_periods=m).min()
    c = _c(df)
    return out(latch(c > hh, c < ll, c < xl, c > xh))


@strategy("S.TREND.KALMAN", "signal_processing", "Kalman local-linear-trend: follow the sign of the filtered slope when "
          "|slope| exceeds thr (in obs-noise units)", params={"thr": 0.05, "q_slope": 1e-4},
          space={"thr": [0.02, 0.05, 0.1], "q_slope": [1e-5, 1e-4]}, extra=ST)
def s_kalman(df, thr=0.05, q_slope=1e-4):
    s = compute(df, "F.EC.KALMAN_TREND", q_slope=q_slope)
    return out(np.sign(s) * (s.abs() > thr))


# =============================================================================== mean reversion
@strategy("S.MR.SHORT_REVERSAL", "mean_reversion", "short-horizon reversal: fade the k-bar return when its z-score "
          "(vs trailing n) exceeds thr; exit after h bars", params={"k": 3, "n": 500, "thr": 2.0, "h": 6},
          space={"k": [1, 3, 6], "thr": [1.5, 2.0, 2.5], "h": [3, 6, 12]},
          extra={"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine",
                                           "target_params": {"a": 0, "b": 1.0}, "max_hold": 6}})
def s_short_reversal(df, k=3, n=500, thr=2.0, h=6):
    r = np.log(_c(df)).diff(k)
    z = (r - r.rolling(n, min_periods=n // 2).mean()) / (r.rolling(n, min_periods=n // 2).std() + EPS)
    side = pd.Series(np.where(z > thr, -1.0, np.where(z < -thr, 1.0, 0.0)), index=df.index)
    return out(edge(side), max_hold=float(h))


@strategy("S.MR.BOLLINGER", "mean_reversion", "Bollinger reversion: long when C < lower band, short when C > upper band; "
          "exit at the middle band", params={"n": 20, "k": 2.0}, space={"n": [20, 50], "k": [2.0, 2.5]}, extra=ST)
def s_bollinger(df, n=20, k=2.0):
    m = sma(_c(df), n); s = _c(df).rolling(n, min_periods=n).std(); c = _c(df)
    return out(latch(c < m - k * s, c > m + k * s, c >= m, c <= m))


@strategy("S.MR.RSI_REVERSAL", "mean_reversion", "RSI reversal: long below lo, short above hi, exit at 50",
          params={"n": 14, "lo": 30, "hi": 70}, space={"n": [2, 7, 14], "lo": [10, 20, 30]}, extra=ST)
def s_rsi_rev(df, n=14, lo=30, hi=70):
    r = compute(df, "F.MO.RSI", n=n)
    return out(latch(r < lo, r > hi, r >= 50, r <= 50))


@strategy("S.MR.STOCH_REVERSAL", "mean_reversion", "stochastic reversal: %K crosses above %D below lo -> long, crosses "
          "below %D above hi -> short; exit at opposite zone or after max_hold", params={"n": 14, "lo": 20, "hi": 80},
          space={"n": [5, 14], "lo": [10, 20]}, extra=TR)
def s_stoch_rev(df, n=14, lo=20, hi=80):
    s = compute(df, "F.MO.STOCH", n=n, col="k"); d = compute(df, "F.MO.STOCH", n=n, col="d")
    up = (s > d) & (s.shift(1) <= d.shift(1)) & (s < lo)
    dn = (s < d) & (s.shift(1) >= d.shift(1)) & (s > hi)
    return out(up.astype(float) - dn.astype(float))


@strategy("S.MR.ZSCORE", "mean_reversion", "z-score reversion of C vs SMA_n: enter at |z| > k, exit at z crossing 0",
          params={"n": 100, "k": 2.0}, space={"n": [50, 100, 200], "k": [1.5, 2.0, 2.5]}, extra=ST)
def s_zscore(df, n=100, k=2.0):
    z = compute(df, "F.PR.PRICE_Z", n=n)
    return out(latch(z < -k, z > k, z >= 0, z <= 0))


@strategy("S.MR.OU", "mean_reversion", "Ornstein-Uhlenbeck trading: trade the equilibrium z only when the fitted half-life "
          "is within [hl_min, hl_max] bars; enter |z|>k, exit |z|<0.25", params={"n": 200, "k": 1.5, "hl_min": 5, "hl_max": 100},
          space={"n": [100, 200], "k": [1.0, 1.5, 2.0]}, extra=ST)
def s_ou(df, n=200, k=1.5, hl_min=5, hl_max=100):
    o = compute(df, "F.EC.OU", n=n, col="z"); hl = compute(df, "F.EC.OU", n=n, col="half_life")
    ok = (hl > hl_min) & (hl < hl_max)
    return out(latch((o < -k) & ok, (o > k) & ok, (o > -0.25) | ~ok, (o < 0.25) | ~ok))


@strategy("S.MR.TWAP_REVERSION", "mean_reversion", "intraday reversion to the tick-weighted average price: fade "
          "deviations > k ATR from TWAP_n, exit on return to TWAP", params={"n": 24, "k": 1.5},
          space={"n": [12, 24, 48], "k": [1.0, 1.5, 2.0]}, extra=ST)
def s_twap(df, n=24, k=1.5):
    d = compute(df, "F.MS.TWAP_TICK", n=n)
    return out(latch(d < -k, d > k, d >= 0, d <= 0))


# =============================================================================== breakout / range
@strategy("S.BO.OPENING_RANGE", "breakout", "opening-range breakout: London first-hour range; long on close above the "
          "range high, short below the low; stop at the opposite side of the range; exit at day end (max_hold)",
          params={"max_hold": 8}, space={"max_hold": [4, 8, 12]}, extra={"mode": "trade"}, frequencies="M5..H1")
def s_orb(df, max_hold=8):
    sr = compute(df, "F.TS.SESSION_RANGE", window="london_open", col="breakout")
    rng = compute(df, "F.TS.SESSION_RANGE", window="london_open", col="range_bps") * df["close"] / 1e4
    first = sr.ne(0) & sr.notna()
    day = local_day(df.index, "Europe/London")
    first_bo = first & ~first.groupby(day).shift(1, fill_value=False).groupby(day).cummax()
    side = np.sign(sr).where(first_bo, 0.0)
    return out(side, stop_dist=rng.clip(lower=1e-9) * 1.0, tp_dist=rng * 1.5, max_hold=float(max_hold))


@strategy("S.BO.VOL_BREAKOUT", "breakout", "volatility breakout (Williams): long when C exceeds the FX-day open by k*ATR, "
          "short when below by k*ATR; one trade per day", params={"k": 0.6}, space={"k": [0.4, 0.6, 1.0]},
          extra=TR_NOTP, frequencies="M5..H4")
def s_vol_bo(df, k=0.6):
    from fxlab.features.structure import fx_day_key
    key = fx_day_key(df.index)
    dopen = df["open"].groupby(key.values).transform("first")
    a = atr(df, 14) * np.sqrt(24)
    up = df["close"] > dopen + k * a; dn = df["close"] < dopen - k * a
    sig = up.astype(float) - dn.astype(float)
    first = sig.ne(0).groupby(key.values).cumsum().eq(1) & sig.ne(0)
    return out(sig.where(first, 0.0))


@strategy("S.BO.RANGE_EXPANSION", "breakout", "range expansion: bar range > k x mean range_n and close in the outer "
          "20% of the bar -> follow the bar direction", params={"k": 2.0, "n": 20}, space={"k": [1.5, 2.0, 3.0]}, extra=TR)
def s_range_exp(df, k=2.0, n=20):
    rr = compute(df, "F.PR.RANGE_RATIO", n=n)
    clv = compute(df, "F.PR.CLV")
    return out(((rr > k) & (clv > 0.6)).astype(float) - ((rr > k) & (clv < -0.6)).astype(float))


@strategy("S.BO.FALSE_BREAKOUT", "breakout", "fade failed breakouts (probe beyond prior n-bar extreme, close back inside)",
          params={"n": 20, "m": 3}, space={"n": [20, 55], "m": [1, 3]}, extra=TR)
def s_false_bo(df, n=20, m=3):
    return out(compute(df, "F.ST.FALSE_BREAKOUT", n=n, m=m).fillna(0))


@strategy("S.BO.RETEST", "breakout", "breakout-retest: after a close beyond the prior n-bar extreme, enter when price "
          "pulls back to within tol*ATR of the broken level within m bars (in breakout direction)",
          params={"n": 20, "m": 10, "tol": 0.3}, space={"n": [20, 55], "m": [5, 10], "tol": [0.2, 0.5]}, extra=TR)
def s_retest(df, n=20, m=10, tol=0.3):
    hh = df["high"].shift(1).rolling(n, min_periods=n).max(); ll = df["low"].shift(1).rolling(n, min_periods=n).min()
    c, a = _c(df), atr(df, 14)
    bo_up = (c > hh); bo_dn = (c < ll)
    lvl_up = hh.where(bo_up).ffill(limit=m); lvl_dn = ll.where(bo_dn).ffill(limit=m)
    recent_up = bo_up.rolling(m, min_periods=1).max().astype(bool) & ~bo_up
    recent_dn = bo_dn.rolling(m, min_periods=1).max().astype(bool) & ~bo_dn
    long = recent_up & ((df["low"] - lvl_up).abs() < tol * a) & (c > lvl_up)
    short = recent_dn & ((df["high"] - lvl_dn).abs() < tol * a) & (c < lvl_dn)
    return out(edge(long.astype(float) - short.astype(float)))


@strategy("S.ST.SR_REVERSAL", "market_structure", "support/resistance reversal: at a level with >= t touches (within 0.5 "
          "ATR) and a rejection wick (CLV), trade away from the level", params={"t": 2, "k": 3},
          space={"t": [2, 3], "k": [3, 5]}, extra=TR)
def s_sr(df, t=2, k=3):
    tc = compute(df, "F.ST.SR_LEVELS", k=k, col="touches")
    near = compute(df, "F.ST.SR_LEVELS", k=k, col="nearest") < 0.5
    clv = compute(df, "F.PR.CLV")
    sw = compute(df, "F.ST.SWING_DIST", k=k, col="to_low"); sh = compute(df, "F.ST.SWING_DIST", k=k, col="to_high")
    at_sup = (sw.abs() < sh.abs())
    ok = (tc >= t) & near
    return out((ok & at_sup & (clv > 0.3)).astype(float) - (ok & ~at_sup & (clv < -0.3)).astype(float))


@strategy("S.ST.TRENDLINE_BREAK", "market_structure", "trendline breakout: close crosses below the support trendline -> "
          "short, above the resistance trendline -> long", params={"k": 5}, space={"k": [3, 5, 8]}, extra=TR)
def s_trendline(df, k=5):
    s = compute(df, "F.ST.TRENDLINES", k=k, col="support"); r = compute(df, "F.ST.TRENDLINES", k=k, col="resistance")
    up = (r > 0) & (r.shift(1) <= 0); dn = (s < 0) & (s.shift(1) >= 0)
    return out(up.astype(float) - dn.astype(float))


@strategy("S.ST.CHANNEL", "market_structure", "regression-channel trading: in an up-sloping channel buy at the lower "
          "band (pos < -k), in a down-sloping channel sell at the upper band; exit at the regression line",
          params={"n": 100, "k": 1.5}, space={"n": [50, 100], "k": [1.0, 1.5, 2.0]}, extra=ST)
def s_channel(df, n=100, k=1.5):
    pos = compute(df, "F.TR.REG_CHANNEL", n=n); sl = compute(df, "F.TR.REG_SLOPE", n=n)
    return out(latch((sl > 0) & (pos < -k), (sl < 0) & (pos > k), pos >= 0, pos <= 0))


@strategy("S.ST.SWING_STRUCTURE", "market_structure", "trade with swing structure (HH/HL = +1) on pullbacks to within "
          "1 ATR of the last swing low (long) / high (short)", params={"k": 3}, space={"k": [2, 3, 5]}, extra=TR)
def s_swing(df, k=3):
    st = compute(df, "F.ST.STRUCTURE_STATE", k=k)
    tl = compute(df, "F.ST.SWING_DIST", k=k, col="to_low"); th = compute(df, "F.ST.SWING_DIST", k=k, col="to_high")
    return out(edge(((st > 0.5) & (tl < 1)).astype(float) - ((st < -0.5) & (th > -1)).astype(float)))


@strategy("S.ST.PIVOT_POINTS", "market_structure", "floor-pivot reversal: fade touches of R1 (short) / S1 (long) from the "
          "previous FX day; target the pivot P", params={}, space={}, extra=TR, frequencies="M5..H1")
def s_pivots(df):
    pos = compute(df, "F.ST.FLOOR_PIVOTS", col="pos")
    return out((pos < 0).astype(float).where(pos.shift(1) >= 0, 0) - (pos > 1).astype(float).where(pos.shift(1) <= 1, 0))


@strategy("S.ST.CANDLESTICK", "market_structure", "candlestick reversal patterns in context: bullish engulfing / hammer "
          "after an n-bar decline -> long; bearish engulfing / shooting star after an advance -> short",
          params={"n": 10}, space={"n": [5, 10, 20]}, extra=TR)
def s_candles(df, n=10):
    e = compute(df, "F.ST.CANDLE_2BAR", col="engulfing").fillna(0)
    h = compute(df, "F.ST.CANDLE_1BAR", col="hammer"); s = compute(df, "F.ST.CANDLE_1BAR", col="shooting_star")
    mom = np.log(_c(df)).diff(n)
    long = ((e > 0) | (h > 0)) & (mom < 0)
    short = ((e < 0) | (s > 0)) & (mom > 0)
    return out(long.astype(float) - short.astype(float))


@strategy("S.GEO.HARMONIC", "geometry", "harmonic pattern completion: when the XABCD score exceeds thr, enter in the "
          "pattern's reversal direction", params={"thr": -0.3, "zz": 0.005}, space={"thr": [-0.5, -0.3, -0.1]}, extra=TR)
def s_harmonic(df, thr=-0.3, zz=0.005):
    sc = compute(df, "F.FG.HARMONIC", thr=zz, col="score"); d = compute(df, "F.FG.HARMONIC", thr=zz, col="dir")
    return out(edge((d * (sc > thr)).fillna(0)))


@strategy("S.GEO.FIB_RETRACE", "geometry", "Fibonacci retracement entries: in the direction of the last completed leg, "
          "enter when price retraces into [0.5, 0.618] of that leg", params={"zz": 0.005}, space={"zz": [0.003, 0.005, 0.01]},
          extra=TR)
def s_fib_retrace(df, zz=0.005):
    rho = compute(df, "F.FG.FIB_RETRACE", thr=zz, col="rho")
    leg = compute(df, "F.ST.ZIGZAG_LEG", thr=zz, col="prev_leg")
    zone = (rho >= 0.5) & (rho <= 0.618)
    return out(edge((np.sign(leg) * zone).fillna(0)))


@strategy("S.GEO.FIB_EXTENSION", "geometry", "Fibonacci extension exhaustion: fade the move when the current leg reaches "
          "1.618 x the prior same-direction leg (eta in [1.5, 1.75])", params={"zz": 0.005}, space={"zz": [0.003, 0.005]},
          extra=TR)
def s_fib_ext(df, zz=0.005):
    eta = compute(df, "F.FG.FIB_EXTENSION", thr=zz, col="eta")
    leg = compute(df, "F.ST.ZIGZAG_LEG", thr=zz, col="leg")
    zone = (eta.abs() >= 1.5) & (eta.abs() <= 1.75)
    return out(edge((-np.sign(leg) * zone).fillna(0)))


@strategy("S.GEO.GANN", "geometry", "geometric price-time: trade crosses of the Gann 1x1 line drawn from the last pivot "
          "(cross above -> long, below -> short)", params={"zz": 0.005}, space={"zz": [0.003, 0.005, 0.01]}, extra=TR)
def s_gann(df, zz=0.005):
    d = compute(df, "F.FG.GANN_ANGLE", thr=zz, col="dev_1x1")
    return out(((d > 0) & (d.shift(1) <= 0)).astype(float) - ((d < 0) & (d.shift(1) >= 0)).astype(float))


# =============================================================================== sessions / seasonality
@strategy("S.SES.SESSION_MOMENTUM", "session", "session momentum: direction of the first `h` hours of the London session "
          "is held until the London close", params={"h": 2}, space={"h": [1, 2, 3]}, extra=ST, frequencies="M5..H1")
def s_session_mom(df, h=2):
    hrs = session_hours(df.index, "Europe/London")
    day = local_day(df.index, "Europe/London")
    open_px = df["open"].where((hrs >= 8) & (hrs < 8 + 1e-9 + 1)).groupby(day).transform("first")
    ref = df["close"].where((hrs >= 8 + h - 1) & (hrs < 8 + h)).groupby(day).transform("last")
    d = np.sign(ref - open_px)
    active = (hrs >= 8 + h) & (hrs < 16)
    return out((d * active).fillna(0))


@strategy("S.SES.LONDON_BREAKOUT", "session", "London breakout of the Asian range (00-07 London time): first close beyond "
          "the range after 07:00 -> trade; stop at range midpoint; flat by 16:00", params={}, space={},
          extra={"mode": "trade"}, frequencies="M5..H1")
def s_london_bo(df):
    bo = compute(df, "F.TS.SESSION_RANGE", window="asia", col="breakout")
    rng = compute(df, "F.TS.SESSION_RANGE", window="asia", col="range_bps") * df["close"] / 1e4
    hrs = session_hours(df.index, "Europe/London")
    day = local_day(df.index, "Europe/London")
    sig = np.sign(bo).where((hrs >= 7) & (hrs < 12), 0).fillna(0)
    first = sig.ne(0) & ~sig.ne(0).groupby(day).shift(1, fill_value=False).groupby(day).cummax()
    return out(sig.where(first, 0), stop_dist=rng / 2 + 1e-9, tp_dist=rng, max_hold=float(9 * 60 // max(1, int(
        pd.Series(df.index).diff().median().total_seconds() // 60))))


@strategy("S.SES.NY_OPEN_REVERSAL", "session", "NY-open reversal: at 08:00-09:00 New York, fade the London-session move "
          "(08:00 London open to NY open) when it exceeds k ATR; flat by 12:00 NY", params={"k": 1.0},
          space={"k": [0.5, 1.0, 1.5]}, extra=ST, frequencies="M5..H1")
def s_ny_open(df, k=1.0):
    hl = session_hours(df.index, "Europe/London"); hn = session_hours(df.index, "America/New_York")
    day = local_day(df.index, "Europe/London")
    lo = df["open"].where((hl >= 8) & (hl < 9)).groupby(day).transform("first")
    move = (df["close"] - lo) / (atr(df, 14) * np.sqrt(8) + EPS)
    trigger = (hn >= 8) & (hn < 9) & (move.abs() > k)
    side = (-np.sign(move)).where(trigger)
    side = side.groupby(day).ffill()
    return out(side.where((hn >= 8) & (hn < 12), 0).fillna(0))


@strategy("S.SES.ASIAN_RANGE", "session", "Asian-session range fade: during 03:00-07:00 London time, fade moves to the "
          "outer 20% of the developing Asian range back toward its middle", params={}, space={}, extra=ST,
          frequencies="M5..H1")
def s_asian(df):
    hl = session_hours(df.index, "Europe/London")
    day = local_day(df.index, "Europe/London")
    m = (hl >= 0) & (hl < 7)
    hi = df["high"].where(m).groupby(day).cummax(); lo = df["low"].where(m).groupby(day).cummin()
    pos = (df["close"] - lo) / (hi - lo + EPS)
    act = (hl >= 3) & (hl < 7)
    return out(latch(act & (pos < 0.2), act & (pos > 0.8), (pos > 0.5) | ~act, (pos < 0.5) | ~act))


@strategy("S.SES.OVERLAP_MOMENTUM", "session", "London-NY overlap momentum: follow the sign of the trailing n-bar return "
          "only while both sessions are open", params={"n": 3}, space={"n": [1, 3, 6]}, extra=ST, frequencies="M5..H1")
def s_overlap(df, n=3):
    o = compute(df, "F.TS.SESSION_CLOCK", col="ldn_ny")
    return out(np.sign(np.log(_c(df)).diff(n)) * o)


@strategy("S.SEAS.INTRADAY", "seasonality", "intraday seasonality: for the NEXT bar's UTC time slot, the t-stat of its "
          "mean return over the previous D days (all realised before the decision); trade its sign when |t| > thr",
          params={"D": 60, "thr": 2.0}, space={"D": [20, 60, 120], "thr": [1.5, 2.0]}, extra=ST, frequencies="M5..H4")
def s_intraday(df, D=60, thr=2.0):
    lr = np.log(_c(df)).diff()
    day = df.index.tz_localize(None).normalize()
    slot = df.index.hour * 60 + df.index.minute
    P = pd.DataFrame({"d": day, "s": slot, "r": lr.values}).pivot_table(index="d", columns="s", values="r", aggfunc="last")
    # M[d, s]: stats over days < d ; M1[d, s]: stats over days <= d
    mean_lt, sd_lt = P.shift(1).rolling(D, min_periods=D // 2).mean(), P.shift(1).rolling(D, min_periods=D // 2).std()
    mean_le, sd_le = P.rolling(D, min_periods=D // 2).mean(), P.rolling(D, min_periods=D // 2).std()
    step = int(round(pd.Series(df.index).diff().median().total_seconds() / 60)) or 1
    nxt_slot = (slot + step) % 1440
    same_day = (slot + step) < 1440
    t = np.full(len(df), np.nan)
    for i, (d, s2, sd_) in enumerate(zip(day, nxt_slot, same_day)):
        M, S = (mean_lt, sd_lt) if sd_ else (mean_le, sd_le)  # next-day slot: today's occurrence is already realised
        if s2 in M.columns and d in M.index:
            m, v = M.at[d, s2], S.at[d, s2]
            t[i] = m / (v + EPS) * np.sqrt(D) if np.isfinite(m) and np.isfinite(v) else np.nan
    t = pd.Series(t, index=df.index)
    return out(np.sign(t) * (t.abs() > thr))


@strategy("S.SEAS.WEEKDAY", "seasonality", "weekday-hour effect: trade the sign of the trailing mean return of the same "
          "weekday & hour over the last W weeks when |t| > thr", params={"W": 52, "thr": 2.0}, space={"W": [26, 52, 104]},
          extra=ST, frequencies="H1..H4")
def s_weekday(df, W=52, thr=2.0):
    lr = np.log(_c(df)).diff().shift(-1)  # return of the NEXT bar, realised at t+1
    key = df.index.dayofweek * 24 + df.index.hour
    # statistics use only past occurrences whose next-bar return is already realised (shift 1 occurrence)
    g = lr.groupby(key)
    m = g.transform(lambda s: s.shift(1).rolling(W, min_periods=W // 2).mean())
    sd = g.transform(lambda s: s.shift(1).rolling(W, min_periods=W // 2).std())
    t = m / (sd + EPS) * np.sqrt(W)
    return out(np.sign(t) * (t.abs() > thr))


@strategy("S.SEAS.WEEKEND_GAP_FADE", "seasonality", "weekend gap fade: if Monday's first bar gaps more than k ATR from "
          "Friday's close, trade toward the gap fill; exit at fill or after max_hold", params={"k": 0.5},
          space={"k": [0.3, 0.5, 1.0]}, extra={"mode": "trade"})
def s_gap(df, k=0.5):
    g = compute(df, "F.PR.GAP")
    gap_px = (df["open"] - df["close"].shift(1)).abs()
    weekend = df.index.to_series().diff() > pd.Timedelta(hours=24)
    sig = (-np.sign(g)).where(weekend & (g.abs() > k), 0)
    return out(sig, stop_dist=gap_px * 1.5, tp_dist=gap_px, max_hold=24.0)


@strategy("S.SEAS.FIX_REVERSAL", "seasonality", "London 4pm fix reversal: fade the move from 15:00 London to the fix "
          "into 16:00-17:00", params={"k": 0.5}, space={"k": [0.3, 0.5, 1.0]}, extra=ST, frequencies="M5..H1")
def s_fix(df, k=0.5):
    hl = session_hours(df.index, "Europe/London")
    day = local_day(df.index, "Europe/London")
    p15 = df["open"].where((hl >= 15) & (hl < 15 + 1)).groupby(day).transform("first")
    move = (df["close"] - p15) / (atr(df, 14) + EPS)
    trig = (hl >= 15.9) & (hl < 16.1) & (move.abs() > k)
    side = (-np.sign(move)).where(trig).groupby(day).ffill()
    return out(side.where((hl >= 16) & (hl < 17), 0).fillna(0))


@strategy("S.SEAS.TURN_OF_MONTH", "seasonality", "turn-of-month flow: hold the sign of the trailing mean return observed "
          "in previous turn-of-month windows (last 2 / first 3 business days), estimated over the last Y months",
          params={"Y": 24}, space={"Y": [12, 24, 48]}, extra=ST)
def s_tom(df, Y=24):
    tom = compute(df, "F.TS.MONTH_CALENDAR", col="turn_of_month")
    lr = np.log(_c(df)).diff()
    month = df.index.tz_localize(None).to_period("M")
    tom_ret = lr.where(tom > 0).groupby(month).sum(min_count=1)
    hist = tom_ret.shift(1).rolling(Y, min_periods=Y // 2).mean()  # completed months only
    sig = np.sign(pd.Series(hist.reindex(month).values, index=df.index))
    return out((sig * tom).fillna(0))


# =============================================================================== volatility & regimes
@strategy("S.VOL.COMPRESSION", "volatility", "volatility compression breakout: after Bollinger width falls into its "
          "bottom q percentile (squeeze), trade the first close outside the bands", params={"n": 20, "q": 0.1},
          space={"q": [0.05, 0.1, 0.2]}, extra=TR)
def s_compression(df, n=20, q=0.1):
    w = compute(df, "F.VO.BB_WIDTH", n=n, col="width"); pb = compute(df, "F.VO.BB_WIDTH", n=n, col="pctb")
    sq = w.rolling(500, min_periods=200).rank(pct=True) < q
    armed = sq.rolling(10, min_periods=1).max().astype(bool)
    return out(edge(((pb > 1) & armed).astype(float) - ((pb < 0) & armed).astype(float)))


@strategy("S.VOL.EXPANSION", "volatility", "volatility expansion follow-through: when short/long ATR ratio jumps above k, "
          "follow the direction of the last s bars", params={"k": 1.8, "s": 3}, space={"k": [1.5, 1.8, 2.5]}, extra=TR)
def s_expansion(df, k=1.8, s=3):
    r = compute(df, "F.VO.ATR_RATIO", s=5, l=50)
    trig = (r > k) & (r.shift(1) <= k)
    return out((np.sign(np.log(_c(df)).diff(s)) * trig).fillna(0))


@strategy("S.VOL.REGIME_SWITCH", "regime", "volatility-regime switching: trend-follow (EMA 20/100 sign) when the vol "
          "percentile > hi, mean-revert (Bollinger) when < lo, flat otherwise", params={"lo": 0.3, "hi": 0.7},
          space={"lo": [0.2, 0.3], "hi": [0.7, 0.8]}, extra=ST)
def s_vol_regime(df, lo=0.3, hi=0.7):
    v = compute(df, "F.RG.VOL_PERCENTILE")
    tr = np.sign(ema(_c(df), 20) - ema(_c(df), 100))
    mr = s_bollinger(df)["side"]
    return out(pd.Series(np.where(v > hi, tr, np.where(v < lo, mr, 0.0)), index=df.index))


@strategy("S.RG.HMM", "regime", "HMM regime trading: momentum (sign of 24-bar return) when the filtered calm-state "
          "probability > p, flat otherwise", params={"p": 0.7}, space={"p": [0.6, 0.7, 0.8]}, extra=ST)
def s_hmm(df, p=0.7):
    calm = compute(df, "F.RG.HMM", col="p0")
    return out(np.sign(np.log(_c(df)).diff(24)) * (calm > p))


@strategy("S.RG.CHANGE_POINT", "regime", "change-point trading: when BOCPD indicates a fresh regime (P(run<5) > p), trade "
          "the drift since the change for up to max_hold bars", params={"p": 0.5}, space={"p": [0.3, 0.5]},
          extra={"mode": "trade", "risk": {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine",
                                           "target_params": {"a": 0, "b": 2.0}, "max_hold": 24}})
def s_cp(df, p=0.5):
    pc = compute(df, "F.RG.BOCPD", col="p_recent_cp")
    trig = (pc > p) & (pc.shift(1) <= p)
    return out((np.sign(np.log(_c(df)).diff(3)) * trig).fillna(0))


@strategy("S.RG.FRACTAL", "regime", "fractal-regime switching: Hurst > h_hi -> trend-follow 24-bar momentum; Hurst < "
          "h_lo -> fade it", params={"h_lo": 0.45, "h_hi": 0.55}, space={"h_lo": [0.4, 0.45], "h_hi": [0.55, 0.6]}, extra=ST)
def s_fractal(df, h_lo=0.45, h_hi=0.55):
    h = compute(df, "F.FR.HURST_RS", w=256, step=4)
    m = np.sign(np.log(_c(df)).diff(24))
    return out(pd.Series(np.where(h > h_hi, m, np.where(h < h_lo, -m, 0.0)), index=df.index))


@strategy("S.RG.ENTROPY", "regime", "entropy-conditioned momentum: trade 12-bar momentum only when permutation entropy is "
          "in its bottom q percentile (structured, less random market)", params={"q": 0.3}, space={"q": [0.2, 0.3, 0.5]},
          extra=ST)
def s_entropy(df, q=0.3):
    pe = compute(df, "F.IT.PERMUTATION", w=200)
    low = pe.rolling(2000, min_periods=500).rank(pct=True) < q
    return out(np.sign(np.log(_c(df)).diff(12)) * low)


@strategy("S.RG.VOL_FORECAST", "volatility", "volatility-forecast conditioned: when GARCH forecast vol exceeds realised "
          "vol by a margin (expected expansion) follow 24-bar breakouts; when below, fade 2-sigma extremes",
          params={"m": 0.15}, space={"m": [0.1, 0.15, 0.3]}, extra=ST)
def s_volfc(df, m=0.15):
    r = compute(df, "F.EC.GARCH", col="fc_over_rv")
    bo = compute(df, "F.ST.BREAKOUT", n=24)
    z = compute(df, "F.PR.PRICE_Z", n=50)
    return out(pd.Series(np.where(r > m, np.sign(bo), np.where(r < -m, -np.sign(z) * (z.abs() > 2), 0.0)), index=df.index))


# =============================================================================== cross-asset
@strategy("S.XA.PAIRS", "cross_asset", "pairs trading on the log-price spread with a correlated reference (z of "
          "ln(C/C_ref)): trade the target leg against the spread, exit at z=0", params={"ref": "GBPUSD", "n": 500, "k": 2.0},
          space={"ref": ["GBPUSD", "EURGBP", "USDCHF"], "k": [1.5, 2.0, 2.5]}, extra=ST, **PANEL)
def s_pairs(df, panel, symbol, ref="GBPUSD", n=500, k=2.0):
    z = compute_panel(df, panel, symbol, "F.CA.RV_SPREAD_Z", ref=ref, n=n)
    return out(latch(z < -k, z > k, z >= 0, z <= 0))


@strategy("S.XA.COINTEGRATION", "cross_asset", "Engle-Granger cointegration: trade the residual z only when the residual "
          "ADF t-stat < crit (cointegrated window)", params={"ref": "GBPUSD", "n": 500, "k": 2.0, "crit": -3.0},
          space={"k": [1.5, 2.0], "crit": [-3.4, -3.0]}, extra=ST, **PANEL)
def s_coint(df, panel, symbol, ref="GBPUSD", n=500, k=2.0, crit=-3.0):
    z = compute_panel(df, panel, symbol, "F.CA.COINTEGRATION", ref=ref, n=n, col="z")
    t = compute_panel(df, panel, symbol, "F.CA.COINTEGRATION", ref=ref, n=n, col="adf_t")
    ok = t < crit
    return out(latch((z < -k) & ok, (z > k) & ok, (z >= 0) | ~ok, (z <= 0) | ~ok))


@strategy("S.XA.RELATIVE_VALUE", "cross_asset", "relative value vs a fundamentally linked asset (e.g. gold vs silver): "
          "fade extreme ratio z-scores with a long lookback", params={"ref": "XAGUSD", "n": 2000, "k": 2.0},
          space={"n": [1000, 2000], "k": [2.0, 2.5]}, extra=ST, **PANEL)
def s_rv(df, panel, symbol, ref="XAGUSD", n=2000, k=2.0):
    z = compute_panel(df, panel, symbol, "F.CA.RV_SPREAD_Z", ref=ref, n=n)
    return out(latch(z < -k, z > k, z >= -0.5, z <= 0.5))


@strategy("S.XA.CURRENCY_STRENGTH", "cross_asset", "currency-strength trading: long when base-minus-quote strength > thr "
          "(in trailing std units), short when < -thr", params={"n": 24, "thr": 1.0}, space={"n": [12, 24, 120]},
          extra=ST, **PANEL)
def s_strength(df, panel, symbol, n=24, thr=1.0):
    d = compute_panel(df, panel, symbol, "F.CA.CURRENCY_STRENGTH", n=n, col="diff")
    z = d / (d.rolling(500, min_periods=100).std() + EPS)
    return out(np.sign(z) * (z.abs() > thr))


@strategy("S.XA.LEAD_LAG", "cross_asset", "lead-lag: when the reference's lagged returns correlate with the target "
          "(|max corr| > c), trade the target in the direction implied by the reference's last return",
          params={"ref": "EURUSD", "c": 0.1}, space={"c": [0.05, 0.1, 0.2]}, extra=ST, **PANEL)
def s_leadlag(df, panel, symbol, ref="EURUSD", c=0.1):
    mc = compute_panel(df, panel, symbol, "F.CA.LEAD_LAG", ref=ref, col="max_corr")
    lag = compute_panel(df, panel, symbol, "F.CA.LEAD_LAG", ref=ref, col="lag")
    from fxlab.features.cross_asset import _ref
    r = _ref(panel, symbol, ref)
    if r is None:
        return out(pd.Series(0.0, index=df.index))
    rr = np.log(r).diff()
    # the reference's return that will be "lag" bars old at the next bar: lag-1 bars ago
    lagged = pd.Series(np.nan, index=df.index)
    for L in range(1, 11):
        lagged = lagged.where(lag != L, rr.shift(L - 1))
    return out((np.sign(mc) * np.sign(lagged) * (mc.abs() > c)).fillna(0))


@strategy("S.XA.TRIANGULAR", "cross_asset", "triangular-parity reversion: fade deviations of the cross from its implied "
          "rate (z > k), exit at parity", params={"n": 200, "k": 2.5}, space={"k": [2.0, 2.5, 3.0]}, extra=ST, **PANEL)
def s_triangular(df, panel, symbol, n=200, k=2.5):
    z = compute_panel(df, panel, symbol, "F.CA.TRIANGULAR", n=n, col="z")
    return out(latch(z < -k, z > k, z >= 0, z <= 0))


@strategy("S.XA.GOLD_FX", "cross_asset", "gold/commodity-FX relative value: z-score of ln(XAUUSD) - beta ln(AUDUSD) "
          "(rolling beta); trade AUDUSD/XAUUSD leg toward equilibrium", params={"n": 1000, "k": 2.0},
          space={"k": [1.5, 2.0]}, extra=ST, **PANEL)
def s_gold_fx(df, panel, symbol, n=1000, k=2.0):
    ref = "AUDUSD" if symbol == "XAUUSD" else "XAUUSD"
    if ref not in panel:
        return out(pd.Series(0.0, index=df.index))
    a, b = np.log(df["close"]), np.log(panel[ref])
    beta = (a.diff().rolling(n, min_periods=n // 2).cov(b.diff()) / (b.diff().rolling(n, min_periods=n // 2).var() + EPS))
    s = a - beta * b
    z = (s - s.rolling(n, min_periods=n // 2).mean()) / (s.rolling(n, min_periods=n // 2).std() + EPS)
    return out(latch(z < -k, z > k, z >= 0, z <= 0))


def _carry(df, symbol):
    from fxlab.data.rates import rate_diff
    return rate_diff(symbol, df.index).fillna(0.0)


CARRY_NOTE = ("Point-in-time 3-month interbank rate differential (OECD/FRED monthly averages, used from the "
              "month after). Run with financing='rates' so overnight swaps follow the same differential.")


@strategy("S.XA.CARRY_FILTER", "carry", "carry filter: trend signal (EMA 20/100) taken only in the direction of "
          "positive carry (base - quote rate > thr for longs, < -thr for shorts). " + CARRY_NOTE,
          params={"thr": 0.0}, space={"thr": [0.0, 0.005, 0.01]}, requires=("fxlab.data.rates",), extra=ST, **PANEL)
def s_carry(df, panel=None, symbol="EURUSD", thr=0.0):
    tr = np.sign(ema(_c(df), 20) - ema(_c(df), 100))
    d = _carry(df, symbol)
    cd = np.where(d > thr, 1.0, np.where(d < -thr, -1.0, 0.0))
    return out(tr.where(tr == cd, 0.0))


@strategy("S.XA.CARRY", "carry", "pure carry: hold the higher-yielding currency of the pair while the rate "
          "differential exceeds thr (Lustig-Verdelhan style, pair version). " + CARRY_NOTE,
          params={"thr": 0.0}, space={"thr": [0.0, 0.005, 0.01, 0.02]}, requires=("fxlab.data.rates",), extra=ST,
          **PANEL)
def s_carry_pure(df, panel=None, symbol="EURUSD", thr=0.0):
    d = _carry(df, symbol)
    return out(pd.Series(np.where(d > thr, 1.0, np.where(d < -thr, -1.0, 0.0)), index=df.index))


@strategy("S.XA.CARRY_VOLGATE", "carry", "carry with crash protection: pure carry, flat while realised volatility "
          "(20 bars) is above its trailing q-quantile (500 bars), i.e. in carry-unwind regimes. " + CARRY_NOTE,
          params={"thr": 0.0, "q": 0.8}, space={"thr": [0.0, 0.01], "q": [0.7, 0.8, 0.9]},
          requires=("fxlab.data.rates",), extra=ST, **PANEL)
def s_carry_vol(df, panel=None, symbol="EURUSD", thr=0.0, q=0.8):
    d = _carry(df, symbol)
    side = pd.Series(np.where(d > thr, 1.0, np.where(d < -thr, -1.0, 0.0)), index=df.index)
    rv = np.log(_c(df)).diff().rolling(20, min_periods=10).std()
    lim = rv.rolling(500, min_periods=100).quantile(q)
    return out(side.where(~(rv > lim), 0.0))


@strategy("S.XA.CARRY_MOM", "carry", "carry with momentum confirmation (Menkhoff et al.): hold the carry side only "
          "while the n-bar return of the pair agrees with it. " + CARRY_NOTE,
          params={"thr": 0.0, "n": 63}, space={"thr": [0.0, 0.01], "n": [21, 63, 126, 252]},
          requires=("fxlab.data.rates",), extra=ST, **PANEL)
def s_carry_mom(df, panel=None, symbol="EURUSD", thr=0.0, n=63):
    d = _carry(df, symbol)
    side = pd.Series(np.where(d > thr, 1.0, np.where(d < -thr, -1.0, 0.0)), index=df.index)
    mom = np.sign(np.log(_c(df)).diff(n))
    return out(side.where(mom == side, 0.0))


@strategy("S.XA.STAT_ARB", "cross_asset", "statistical arbitrage (Avellaneda-Lee): regress target returns on panel "
          "principal-component factors over a trailing window; trade the z-score of the cumulative idiosyncratic "
          "residual (s-score) toward zero", params={"n": 500, "k": 1.5, "refit": 100}, space={"k": [1.25, 1.5, 2.0]},
          extra=ST, **PANEL)
def s_statarb(df, panel, symbol, n=500, k=1.5, refit=100):
    R = np.log(pd.concat([df["close"].rename(symbol), panel.drop(columns=[symbol], errors="ignore")], axis=1)).diff()
    X = R.to_numpy(float)
    resid = np.full(len(X), np.nan)
    beta = None; mu = None
    for i in range(n, len(X)):
        if (i - n) % refit == 0:
            W = X[i - n:i]
            keep = np.isnan(W).mean(0) <= 0.2; keep[0] = True
            W2 = W[:, keep]; W2 = W2[~np.isnan(W2).any(1)]
            if len(W2) > 0.6 * n and W2.shape[1] > 2:
                others = W2[:, 1:]
                mu = others.mean(0)
                U, S_, Vt = np.linalg.svd(others - mu, full_matrices=False)
                F = (others - mu) @ Vt[:2].T
                beta = (np.linalg.lstsq(np.c_[np.ones(len(F)), F], W2[:, 0], rcond=None)[0], Vt[:2], keep)
        if beta is not None and not np.isnan(X[i]).any():
            coef, V, keep = beta
            row = X[i][keep]
            f = (row[1:] - mu) @ V.T
            resid[i] = row[0] - coef[0] - f @ coef[1:]
    e = pd.Series(resid, index=df.index).fillna(0)
    cs = e.rolling(60, min_periods=30).sum()
    z = (cs - cs.rolling(n, min_periods=n // 2).mean()) / (cs.rolling(n, min_periods=n // 2).std() + EPS)
    return out(latch(z < -k, z > k, z >= -0.5, z <= 0.5))


# =============================================================================== signal processing / microstructure
@strategy("S.SP.SPECTRAL_CYCLE", "signal_processing", "Ehlers sinewave cycle trading: long when sine crosses above lead "
          "sine, short when below, only while a cycle dominates (FFT power share > s)", params={"s": 0.15},
          space={"s": [0.1, 0.15, 0.25]}, extra=ST)
def s_cycle(df, s=0.15):
    sw = compute(df, "F.SP.SINEWAVE", col="sine"); ld = compute(df, "F.SP.SINEWAVE", col="lead")
    ps = compute(df, "F.SP.FFT_DOMINANT", col="power_share")
    up = (ld > sw) & (ld.shift(1) <= sw.shift(1)); dn = (ld < sw) & (ld.shift(1) >= sw.shift(1))
    cyc = ps > s
    return out(latch(up & cyc, dn & cyc, ~cyc, ~cyc))


@strategy("S.SP.WAVELET", "signal_processing", "wavelet denoising reversion: fade deviations of price from its "
          "wavelet-denoised value beyond k return-std", params={"w": 128, "k": 1.5}, space={"k": [1.0, 1.5, 2.0]}, extra=ST)
def s_wavelet(df, w=128, k=1.5):
    d = compute(df, "F.SP.WAVELET_DENOISE", w=w)
    return out(latch(d < -k, d > k, d >= 0, d <= 0))


@strategy("S.MS.ORDER_FLOW", "microstructure", "order-flow proxy: follow cumulative signed tick-volume flow (OBV over n) "
          "when confirmed by the same-signed quoted-liquidity imbalance", params={"n": 12, "thr": 0.1},
          space={"n": [6, 12, 24], "thr": [0.05, 0.1, 0.2]}, extra=ST)
def s_orderflow(df, n=12, thr=0.1):
    ob = compute(df, "F.MS.OBV_TICK", n=n)
    li = compute(df, "F.MS.LIQ_IMBALANCE", n=n)
    return out(np.sign(ob) * (ob.abs() > thr) * (np.sign(ob) == np.sign(li)))


@strategy("S.MS.TICK_IMBALANCE", "microstructure", "tick-imbalance momentum: trade the sign of the n-bar up/down quote "
          "imbalance when |imbalance| > thr", params={"n": 3, "thr": 0.1}, space={"n": [1, 3, 6], "thr": [0.05, 0.1, 0.2]},
          extra=ST)
def s_tickimb(df, n=3, thr=0.1):
    t = compute(df, "F.MS.TICK_IMBALANCE", n=n)
    return out(np.sign(t) * (t.abs() > thr))


@strategy("S.MS.LIQUIDITY_FILTERED", "microstructure", "liquidity-conditioned momentum: 24-bar momentum taken only when "
          "the expected move covers > m spreads and the liquidity regime is above its median", params={"m": 20},
          space={"m": [10, 20, 40]}, extra=ST)
def s_liq(df, m=20):
    mc = compute(df, "F.IN.MOVE_TO_COST", h=5)
    lr = compute(df, "F.RG.LIQUIDITY_REGIME")
    return out(np.sign(np.log(_c(df)).diff(24)) * ((mc > m) & (lr > 0.5)))


def compute_panel(df, panel, symbol, fid, col=None, **params):
    from fxlab.registry import CATALOG
    e = CATALOG[fid]
    o = e.fn(df, panel=panel, symbol=symbol, **(e.params | params))
    if isinstance(o, pd.DataFrame):
        o = o[col] if col else o.iloc[:, 0]
    return o.astype(float)
