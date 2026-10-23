"""Risk management: stop/target distance models, target functions of the stop
(linear and nonlinear), position sizing and portfolio-level controls.
Every quantity is computed from information available at the decision bar."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxlab.features._util import atr, causal_pivots, EPS


# ------------------------------------------------------------------------------------------- stop distances
def stop_fixed(bars, pips: float, pip: float) -> pd.Series:
    return pd.Series(pips * pip, index=bars.index)


def stop_atr(bars, mult: float = 2.0, n: int = 14) -> pd.Series:
    return mult * atr(bars, n)


def stop_realized_vol(bars, mult: float = 2.0, n: int = 50, horizon: int = 10) -> pd.Series:
    """mult * sigma_n(log returns) * sqrt(horizon) * price."""
    lr = np.log(bars["close"]).diff()
    return mult * lr.rolling(n, min_periods=n).std() * np.sqrt(horizon) * bars["close"]


def stop_structural(bars, side: pd.Series, k: int = 3, buffer_atr: float = 0.2) -> pd.Series:
    """Beyond the last confirmed swing low (longs) / swing high (shorts) plus a buffer."""
    sh, sl, _, _ = causal_pivots(bars, k)
    a = atr(bars, 14).to_numpy(float)
    c = bars["close"].to_numpy(float)
    s = side.reindex(bars.index).fillna(0).to_numpy()
    d = np.where(s > 0, c - sl + buffer_atr * a, np.where(s < 0, sh - c + buffer_atr * a, np.nan))
    d = np.where(d > 0.2 * a, d, np.nan)  # invalid if the structure is on the wrong side / too close
    return pd.Series(d, index=bars.index)


def stop_quantile(bars, q: float = 0.95, horizon: int = 10, n: int = 2000) -> pd.Series:
    """Empirical q-quantile of |h-bar adverse excursion| over the trailing n bars
    (excursions that completed before t only)."""
    lo = bars["low"].rolling(horizon, min_periods=horizon).min()
    hi = bars["high"].rolling(horizon, min_periods=horizon).max()
    start = bars["close"].shift(horizon)
    adverse = pd.concat([(start - lo), (hi - start)], axis=1).max(axis=1)  # completed at t, began at t-h
    return adverse.rolling(n, min_periods=n // 4).quantile(q)


STOP_MODELS = {"fixed": stop_fixed, "atr": stop_atr, "realized_vol": stop_realized_vol,
               "structural": stop_structural, "quantile": stop_quantile}


# --------------------------------------------------------------------------------- target as f(stop distance)
def target_from_stop(stop: pd.Series, kind: str = "affine", a: float = 0.0, b: float = 1.5, gamma: float = 1.0,
                     knots: tuple = (), unit: pd.Series | None = None) -> pd.Series:
    """Take-profit distance as a function of the stop distance s (in ATR units if unit given):
    affine      tp = a + b s
    polynomial  tp = a + b s^2
    log         tp = a + b ln(1+s)
    power       tp = b s^gamma
    piecewise   tp = b_i s on segments defined by knots=((s1,b1),(s2,b2),...)
    none        no target (NaN)"""
    u = unit if unit is not None else pd.Series(1.0, index=stop.index)
    s = stop / u
    if kind == "affine":
        tp = a + b * s
    elif kind == "polynomial":
        tp = a + b * s ** 2
    elif kind == "log":
        tp = a + b * np.log1p(s)
    elif kind == "power":
        tp = b * s ** gamma
    elif kind == "piecewise":
        tp = pd.Series(np.nan, index=s.index)
        prev = -np.inf
        for edge, slope in knots:
            m = (s > prev) & (s <= edge)
            tp[m] = slope * s[m]
            prev = edge
        tp[s > prev] = knots[-1][1] * s[s > prev] if knots else np.nan
    elif kind == "none":
        return pd.Series(np.nan, index=stop.index)
    else:
        raise ValueError(kind)
    return (tp * u).where(tp > 0)


# ------------------------------------------------------------------------------------------------ sizing
def size_fixed_fractional(stop_dist: pd.Series, price: pd.Series, risk: float = 0.005, max_lev: float = 10.0):
    """Notional fraction of equity so that a stop-out loses `risk` of equity."""
    return (risk * price / stop_dist).clip(upper=max_lev)


def size_vol_target(bars: pd.DataFrame, target_ann_vol: float = 0.10, n: int = 100, bars_per_year: float = 6240,
                    max_lev: float = 10.0) -> pd.Series:
    vol = np.log(bars["close"]).diff().rolling(n, min_periods=n).std() * np.sqrt(bars_per_year)
    return (target_ann_vol / (vol + EPS)).clip(upper=max_lev)


def kelly_fraction(trade_returns: np.ndarray, fraction: float = 0.25, cap: float = 5.0) -> float:
    """Fractional Kelly for the per-trade return distribution: f* = mu / var (continuous approx)."""
    r = np.asarray(trade_returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 20 or r.var() <= 0:
        return 0.0
    return float(np.clip(fraction * r.mean() / r.var(), 0.0, cap))


def drawdown_scaler(equity: pd.Series, dd_start: float = 0.05, dd_stop: float = 0.20) -> pd.Series:
    """Linear de-risking between dd_start and dd_stop drawdown (1 -> 0), using equity up to t-1."""
    eq = equity.shift(1)
    dd = 1 - eq / eq.cummax()
    return (1 - (dd - dd_start) / (dd_stop - dd_start)).clip(0, 1).fillna(1.0)


# ------------------------------------------------------------------------------------------- portfolio
def currency_exposures(positions: pd.DataFrame) -> pd.DataFrame:
    """Net exposure per currency (long base, short quote) from notional positions per pair."""
    out = {}
    for s in positions.columns:
        b, q = s[:3], s[3:]
        out[b] = out.get(b, 0) + positions[s]
        out[q] = out.get(q, 0) - positions[s]
    return pd.DataFrame(out)


def net_currency_exposure(positions: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    """Scale all positions down whenever any single-currency net exposure exceeds cap."""
    ex = currency_exposures(positions).abs().max(axis=1)
    scale = (cap / ex.where(ex > cap)).fillna(1.0)
    return positions.mul(scale, axis=0)


def correlation_aware_weights(returns: pd.DataFrame, n: int = 500, shrink: float = 0.5) -> pd.DataFrame:
    """Risk-parity-style weights from a shrunk trailing covariance (inverse-vol scaled by
    1/(1 + avg correlation with others)); recomputed each bar with data to t-1."""
    w = {}
    vol = returns.rolling(n, min_periods=n // 2).std().shift(1)
    corr_avg = {}
    for c in returns.columns:
        others = [o for o in returns.columns if o != c]
        if others:
            cs = pd.concat([returns[c].rolling(n, min_periods=n // 2).corr(returns[o]) for o in others], axis=1)
            corr_avg[c] = cs.mean(axis=1).shift(1)
        else:
            corr_avg[c] = pd.Series(0.0, index=returns.index)
    for c in returns.columns:
        w[c] = 1 / (vol[c] + EPS) / (1 + shrink * corr_avg[c].clip(lower=0))
    W = pd.DataFrame(w)
    return W.div(W.sum(axis=1), axis=0)


def min_variance_hedge_ratio(target_ret: pd.Series, hedge_ret: pd.Series, n: int = 500) -> pd.Series:
    """Rolling minimum-variance hedge ratio beta = cov/var (trailing, shifted: known before t)."""
    return (target_ret.rolling(n, min_periods=n // 2).cov(hedge_ret) /
            (hedge_ret.rolling(n, min_periods=n // 2).var() + EPS)).shift(1)


@dataclass
class RiskConfig:
    stop: str = "atr"
    stop_params: dict = field(default_factory=lambda: {"mult": 2.0})
    target: str = "affine"
    target_params: dict = field(default_factory=lambda: {"a": 0.0, "b": 1.5})
    sizing: str = "fixed_fractional"
    risk_per_trade: float = 0.005
    max_hold: int | None = None
    trail_atr: float | None = None
    max_leverage: float = 10.0
    breakeven_r: float | None = None     # move the stop to entry after a favourable move of breakeven_r x stop
    partial_r: float | None = None       # take partial_frac of the position off at partial_r x stop distance
    partial_frac: float = 0.5


def build_orders(bars: pd.DataFrame, side: pd.Series, rc: RiskConfig, pip: float) -> pd.DataFrame:
    """Turn a decided side series into simulator orders using a RiskConfig."""
    if rc.stop == "fixed":
        sd = stop_fixed(bars, pip=pip, **rc.stop_params)
    elif rc.stop == "structural":
        sd = stop_structural(bars, side, **rc.stop_params)
    else:
        sd = STOP_MODELS[rc.stop](bars, **rc.stop_params)
    unit = atr(bars, 14)
    tp = target_from_stop(sd, rc.target, unit=unit, **rc.target_params)
    if rc.sizing == "fixed_fractional":
        size = size_fixed_fractional(sd, bars["close"], rc.risk_per_trade, rc.max_leverage)
    elif rc.sizing == "vol_target":
        size = size_vol_target(bars, max_lev=rc.max_leverage)
    else:
        size = pd.Series(1.0, index=bars.index)
    out = pd.DataFrame({"side": side.reindex(bars.index).fillna(0), "stop_dist": sd, "tp_dist": tp, "size": size},
                       index=bars.index)
    out["max_hold"] = float(rc.max_hold) if rc.max_hold else np.nan
    out["trail_dist"] = rc.trail_atr * unit if rc.trail_atr else np.nan
    out["be_mult"] = float(rc.breakeven_r) if rc.breakeven_r else np.nan
    out["part_mult"] = float(rc.partial_r) if rc.partial_r else np.nan
    out["part_frac"] = float(rc.partial_frac) if rc.partial_r else 0.0
    return out
