"""Strategy helpers.

A strategy is ``fn(bars, panel=None, symbol=None, **params) -> DataFrame`` indexed like
bars with column ``side`` in {-1, 0, +1}: the position desired after the CLOSE of
each bar (executed by the simulator at the next open). Strategies may add
``stop_dist`` / ``tp_dist`` / ``max_hold`` (price units / bars) when their logic
defines natural levels (e.g. an opening-range stop); otherwise the evaluation
layer derives them from the strategy's RiskConfig (extra['risk']).

Execution mode (extra['mode']):
  'state' - continuous target-position strategy -> vector engine (costs on every change)
  'trade' - discrete trades with stops/targets   -> event engine
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from fxlab.features.interactions import compute  # noqa: F401  (re-export)


@njit(cache=True)
def _latch(el, es, xl, xs):
    n = len(el)
    out = np.zeros(n)
    s = 0.0
    for i in range(n):
        if s > 0 and xl[i]:
            s = 0.0
        elif s < 0 and xs[i]:
            s = 0.0
        if el[i] and not es[i]:
            s = 1.0
        elif es[i] and not el[i]:
            s = -1.0
        out[i] = s
    return out


def latch(entry_long, entry_short, exit_long, exit_short) -> pd.Series:
    """Stateful position from entry/exit conditions (entries also reverse an opposite position)."""
    f = lambda s: np.asarray(pd.Series(s).fillna(False).astype(bool))
    idx = entry_long.index
    return pd.Series(_latch(f(entry_long), f(entry_short), f(exit_long), f(exit_short)), index=idx)


def edge(side: pd.Series) -> pd.Series:
    """Keep the side only on bars where it changes (entry triggers) - for trade-mode strategies."""
    return side.where(side != side.shift(1), 0.0)


def out(side, **cols) -> pd.DataFrame:
    if not isinstance(side, pd.Series):
        raise TypeError("strategy side must be a pandas Series aligned to the bars (got bare array)")
    d = pd.DataFrame({"side": side.astype(float).fillna(0.0)})
    for k, v in cols.items():
        d[k] = v
    return d


def session_hours(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    l = index.tz_convert(tz)
    return l.hour.to_numpy() + l.minute.to_numpy() / 60.0


def local_day(index: pd.DatetimeIndex, tz: str):
    return index.tz_convert(tz).tz_localize(None).normalize()
