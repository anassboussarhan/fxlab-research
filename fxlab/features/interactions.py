"""7.18 Feature interactions and transforms. Generic operators take other
catalog features by id (first output column unless ``col`` is given), so their
causality is inherited from (and tested together with) their inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.registry import feature, CATALOG
from fxlab.features._util import EPS, atr

F = "interactions"


_MEMO: "OrderedDict[tuple, pd.DataFrame | pd.Series]" = None  # type: ignore


def _memo_key(df, fid, params):
    import json
    return (id(df), len(df), str(df.index[0]) if len(df) else "", str(df.index[-1]) if len(df) else "",
            float(df["close"].iloc[-1]) if len(df) and "close" in df else 0.0, fid,
            json.dumps(params, sort_keys=True, default=str))


def compute(df, fid: str, col: str | None = None, **params) -> pd.Series:
    """Compute one catalog feature (first or named output column). Results are memoised per process for the
    same bars object + parameters, so strategy candidates that differ only in thresholds reuse features."""
    from collections import OrderedDict
    global _MEMO
    if _MEMO is None:
        _MEMO = OrderedDict()
    e = CATALOG[fid]
    p = e.params | params
    k = _memo_key(df, fid, p)
    out = _MEMO.get(k)
    if out is None:
        out = e.fn(df, **p)
        _MEMO[k] = out
        if len(_MEMO) > 256:
            _MEMO.popitem(last=False)
    else:
        _MEMO.move_to_end(k)
    if isinstance(out, pd.DataFrame):
        out = out[col] if col else out.iloc[:, 0]
    return out.astype(float)


def pit_zscore(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=n // 2).mean()) / (x.rolling(n, min_periods=n // 2).std() + EPS)


@feature("F.IN.PAIRWISE_OPS", F, "pairwise interaction of two catalog features a, b after point-in-time z-scoring over n: "
         "op in {product, ratio=a/(|b|+1), difference, max, min}", params={"a": "F.MO.RSI", "b": "F.RG.VOL_PERCENTILE",
         "op": "product", "n": 500}, space={"op": ["product", "ratio", "difference", "max", "min"]},
         inputs=("open", "high", "low", "close"))
def f_pairwise(df, a="F.MO.RSI", b="F.RG.VOL_PERCENTILE", op="product", n=500):
    x, y = pit_zscore(compute(df, a), n), pit_zscore(compute(df, b), n)
    return {"product": x * y, "ratio": x / (y.abs() + 1), "difference": x - y,
            "max": np.maximum(x, y), "min": np.minimum(x, y)}[op]


UNARY = {"tanh": np.tanh, "sign": np.sign, "abs": np.abs, "sq": np.square, "id": lambda v: v,
         "log1p_abs": lambda v: np.sign(v) * np.log1p(np.abs(v))}
BINARY = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": lambda a, b: a / (np.abs(b) + 1.0),
          "max": np.maximum, "min": np.minimum}


def eval_expr(expr, leaves: dict[str, pd.Series]) -> pd.Series:
    """expr: nested tuples ('op', child[, child]) or a leaf name."""
    if isinstance(expr, str):
        return leaves[expr]
    op = expr[0]
    if op in UNARY:
        return UNARY[op](eval_expr(expr[1], leaves))
    return BINARY[op](eval_expr(expr[1], leaves), eval_expr(expr[2], leaves))


@feature("F.IN.SYMBOLIC_COMPOSE", F, "symbolic composition: evaluate an expression tree over PIT-z-scored catalog features "
         "(unary tanh/sign/abs/sq/log1p_abs; binary + - * protected-/ max min). Used by genetic programming to build "
         "candidate features", params={"expr": ("*", ("tanh", "F.TR.REG_SLOPE"), ("sign", "F.MS.TICK_IMBALANCE")), "n": 500},
         space={"expr": "grammar"}, inputs=("open", "high", "low", "close", "up_ticks", "down_ticks"))
def f_symbolic(df, expr=("*", ("tanh", "F.TR.REG_SLOPE"), ("sign", "F.MS.TICK_IMBALANCE")), n=500):
    leaves = {}

    def collect(e):
        if isinstance(e, str):
            leaves[e] = pit_zscore(compute(df, e), n)
        else:
            for c in e[1:]:
                collect(c)
    collect(expr)
    return eval_expr(expr, leaves)


@feature("F.IN.REGIME_GATE", F, "regime-conditioned indicator: feature x gated by a regime feature g: x * 1[g in band] "
         "(bands on g's percentile rank over n), else 0", params={"x": "F.MO.RSI", "g": "F.RG.VOL_PERCENTILE",
         "lo": 0.0, "hi": 0.5, "n": 1000}, space={"lo": [0.0, 0.5], "hi": [0.5, 1.0]}, inputs=("open", "high", "low", "close"))
def f_regime_gate(df, x="F.MO.RSI", g="F.RG.VOL_PERCENTILE", lo=0.0, hi=0.5, n=1000):
    xv = compute(df, x)
    gr = compute(df, g).rolling(n, min_periods=n // 2).rank(pct=True)
    return (xv * ((gr >= lo) & (gr <= hi)).astype(float)).where(gr.notna())


@feature("F.IN.VOL_NORMALIZE", F, "volatility-adjusted indicator: feature x divided by trailing return volatility "
         "(or by ATR/C when x is price-scaled) - turns level-dependent indicators into regime-comparable units",
         params={"x": "F.MO.MACD", "n": 100}, space={"n": [50, 100, 500]}, inputs=("open", "high", "low", "close"))
def f_vol_norm(df, x="F.MO.MACD", n=100):
    vol = np.log(df["close"]).diff().rolling(n, min_periods=n).std()
    return compute(df, x) / (vol + EPS)


@feature("F.IN.PIT_RANK", F, "point-in-time rank transform of feature x: percentile rank within its trailing n values "
         "(non-parametric, outlier-robust normalisation)", params={"x": "F.VO.ATR", "n": 500},
         space={"n": [250, 500, 2000]}, inputs=("open", "high", "low", "close"))
def f_pit_rank(df, x="F.VO.ATR", n=500):
    return compute(df, x).rolling(n, min_periods=n // 2).rank(pct=True)


@feature("F.IN.MOVE_TO_COST", F, "economic interaction: expected move over h bars (ATR_14 sqrt h) divided by the quoted "
         "spread - how many spreads a typical move covers", params={"h": 5}, space={"h": [1, 5, 20]},
         inputs=("high", "low", "close", "spread_mean"))
def f_move_cost(df, h=5):
    return atr(df, 14) * np.sqrt(h) / (df["spread_mean"].rolling(20, min_periods=1).mean() + EPS)


@feature("F.IN.MTF_ALIGNMENT", F, "multi-timeframe alignment: mean of sign(n-bar log return) for n in (k, 4k, 16k) "
         "and their product (all-aligned flag)", params={"k": 5}, space={"k": [3, 5, 12]})
def f_mtf(df, k=5):
    lc = np.log(df["close"])
    s = [np.sign(lc - lc.shift(m)) for m in (k, 4 * k, 16 * k)]
    return pd.DataFrame({"mean_sign": (s[0] + s[1] + s[2]) / 3, "all_aligned": s[0] * s[1] * s[2] * ((s[0] == s[1]) & (s[1] == s[2]))})
