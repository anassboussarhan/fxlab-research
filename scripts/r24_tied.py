"""Round 24 R2 / R3: library rules rewritten with their constants tied by role (docs/round24_preregistration.md).

In round 23 every numeric literal became a separate parameter. For these rules that broke them:
- the two session rules looked ahead when their hour constants separated;
- TTM Squeeze failed on every draw, because its regression vector and its rolling window had different lengths.

Here each role has one parameter:
- the session rules: one shared shift s added to every hour constant (the reference stays before the window);
- TTM: one length L for every 20-bar window, (L - 1) / 2 for the regression centre, b for the Bollinger
  multiplier and q for the falling / rising run.

At their defaults (s = 0; L = 20, b = 2, q = 2) each function reproduces the library rule exactly. TIED maps the
rule id to (function, parameter specs in the LW spec format).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.features._util import EPS, atr, sma
from fxlab.strategies.base import latch, local_day, out, session_hours


def session_mom_tied(df, h=2, s=0):
    hrs = session_hours(df.index, "Europe/London")
    day = local_day(df.index, "Europe/London")
    o = 8 + s
    open_px = df["open"].where((hrs >= o) & (hrs < o + 1e-9 + 1)).groupby(day).transform("first")
    ref = df["close"].where((hrs >= o + h - 1) & (hrs < o + h)).groupby(day).transform("last")
    d = np.sign(ref - open_px)
    active = (hrs >= o + h) & (hrs < 16 + s)
    return out((d * active).fillna(0))


def ny_open_tied(df, k=1.0, s=0, atr_n=14, scale=8):
    hl = session_hours(df.index, "Europe/London")
    hn = session_hours(df.index, "America/New_York")
    day = local_day(df.index, "Europe/London")
    lo = df["open"].where((hl >= 8 + s) & (hl < 9 + s)).groupby(day).transform("first")
    move = (df["close"] - lo) / (atr(df, atr_n) * np.sqrt(scale) + EPS)
    trigger = (hn >= 8 + s) & (hn < 9 + s) & (move.abs() > k)
    side = (-np.sign(move)).where(trigger)
    side = side.groupby(day).ffill()
    return out(side.where((hn >= 8 + s) & (hn < 12 + s), 0).fillna(0))


def ttm_tied(df, k=1.5, m=6, L=20, b=2.0, q=2):
    c = df["close"]
    mid = sma(c, L)
    sd = c.rolling(L, min_periods=L).std()
    a = atr(df, L)
    on = ((mid + b * sd) < (mid + k * a)) & ((mid - b * sd) > (mid - k * a))
    run = on.astype(int).groupby((~on).cumsum()).cumsum()
    fired = (~on) & (run.shift(1) >= m)
    base = (df["high"].rolling(L, min_periods=L).max() + df["low"].rolling(L, min_periods=L).min()) / 2
    x = c - (base + mid) / 2
    cen = (L - 1) / 2.0
    xs = np.arange(L, dtype=float) - cen
    mom = x.rolling(L, min_periods=L).apply(lambda w: float(np.dot(xs, w) / np.dot(xs, xs) * cen + w.mean()), raw=True)
    falling = (mom < mom.shift(1)) & (mom.shift(1) < mom.shift(q))
    rising = (mom > mom.shift(1)) & (mom.shift(1) > mom.shift(q))
    return out(latch(fired & (mom > 0), fired & (mom < 0), falling, rising))


def _i(name, lo, hi, log=False):
    return {"name": name, "kind": "int", "lo": lo, "hi": hi, "log": log}


def _f(name, lo, hi):
    return {"name": name, "kind": "float", "lo": lo, "hi": hi, "log": False}


TIED = {
    "S.SES.SESSION_MOMENTUM": (session_mom_tied, [_i("h", 1, 6), _i("s", -8, 8)]),
    "S.SES.NY_OPEN_REVERSAL": (ny_open_tied, [_f("k", 0.25, 3.0), _i("s", -8, 8), _i("atr_n", 4, 42, True),
                                              _i("scale", 2, 24, True)]),
    "S.FAMOUS.TTM_SQUEEZE": (ttm_tied, [_f("k", 0.75, 4.0), _i("m", 1, 24), _i("L", 6, 60, True), _f("b", 1.0, 3.5),
                                        _i("q", 2, 6)]),
}
DEFAULTS = {"S.SES.SESSION_MOMENTUM": {"h": 2, "s": 0}, "S.SES.NY_OPEN_REVERSAL": {"k": 1.0, "s": 0, "atr_n": 14, "scale": 8},
            "S.FAMOUS.TTM_SQUEEZE": {"k": 1.5, "m": 6, "L": 20, "b": 2.0, "q": 2}}
