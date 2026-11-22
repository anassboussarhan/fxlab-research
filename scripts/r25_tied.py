"""Round 25: three more library rules with their constants tied by role (round 25 addendum 1).

Round 23 drew each numeric literal separately. Its unit statistics show that this broke three more rules besides TTM
Squeeze (round 24), each because a rolling window and its min_periods were drawn apart:
- S.DAILY.HIGH52: 88% of round-23 draws failed (four 252s: two windows and two min_periods);
- S.DAILY.VOL_TARGET: 57% (the 20-day window and its min_periods);
- S.VOL.COMPRESSION: 22% (the 500-bar rank window and its 200-bar min_periods).

Here each role has one parameter:
- HIGH52: L for the 52-week window (both the high and the low, window = min_periods);
- VOL_TARGET: n for the realised-vol window (= min_periods), ann for the annualisation;
- COMPRESSION: W for the rank window, f for its min_periods as a share of W (default 0.4 = 200 / 500), a for the
  arming window.

At their defaults each function reproduces the library rule exactly (checked in main()). Ranges as round 23: rule
parameters from half the smallest to twice the largest grid value, constants x1/3 to x3, shares 0.1-1.
TIED25 maps the rule id to (function, specs); DEFAULTS25 gives the library defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.features._util import EPS, sma
from fxlab.strategies.base import compute, edge, latch, out


def _none(c):
    return pd.Series(False, index=c.index)


def high52_tied(df, tol=0.0, m=50, short=0, L=252):
    c = df["close"]
    L = int(L)
    hi, lo = c.rolling(L, min_periods=L).max(), c.rolling(L, min_periods=L).min()
    sm = sma(c, int(m))
    el = c >= (1 - tol) * hi
    es = (c <= (1 + tol) * lo) if short else _none(c)
    return out(latch(el, es, c < sm, c > sm))


def voltarget_tied(df, tv=0.15, n=20, ann=252):
    c = df["close"]
    n = int(n)
    rv = np.log(c).diff().rolling(n, min_periods=n).std() * np.sqrt(ann)
    w = (tv / (rv + EPS)).clip(upper=1.0)
    return out(w.fillna(0.0))


def compression_tied(df, n=20, q=0.1, W=500, f=0.4, a=10):
    W, a = int(W), int(a)
    w = compute(df, "F.VO.BB_WIDTH", n=int(n), col="width")
    pb = compute(df, "F.VO.BB_WIDTH", n=int(n), col="pctb")
    sq = w.rolling(W, min_periods=max(1, int(round(f * W)))).rank(pct=True) < q
    armed = sq.rolling(a, min_periods=1).max().astype(bool)
    return out(edge(((pb > 1) & armed).astype(float) - ((pb < 0) & armed).astype(float)))


def _i(name, lo, hi, log=True):
    return {"name": name, "kind": "int", "lo": lo, "hi": hi, "log": log}


def _f(name, lo, hi):
    return {"name": name, "kind": "float", "lo": lo, "hi": hi, "log": False}


TIED25 = {
    "S.DAILY.HIGH52": (high52_tied, [_f("tol", 0.0, 0.1), _i("m", 25, 200), {"name": "short", "kind": "cat",
                                                                             "choices": [0, 1]}, _i("L", 84, 756)]),
    "S.DAILY.VOL_TARGET": (voltarget_tied, [_f("tv", 0.05, 0.6), _i("n", 7, 60), _i("ann", 84, 756)]),
    "S.VOL.COMPRESSION": (compression_tied, [_f("q", 0.025, 0.4), _i("n", 7, 60), _i("W", 167, 1500), _f("f", 0.1, 1.0),
                                             _i("a", 3, 30)]),
}
DEFAULTS25 = {"S.DAILY.HIGH52": {"tol": 0.0, "m": 50, "short": 0, "L": 252},
              "S.DAILY.VOL_TARGET": {"tv": 0.15, "n": 20, "ann": 252},
              "S.VOL.COMPRESSION": {"n": 20, "q": 0.1, "W": 500, "f": 0.4, "a": 10}}


def main():
    """At the defaults the tied functions reproduce the library rules (synthetic H1 and D1 bars)."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import intraday17 as I17
    from r25_lookahead import synth
    S = {e.id: e for e in I17.strategies()}
    for tf in ("H4", "D1"):
        b = synth(tf)
        for k, (fn, _) in TIED25.items():
            lib = S[k].fn(b.copy(), **dict(S[k].params or {}))["side"].to_numpy(float)
            new = fn(b.copy(), **DEFAULTS25[k])["side"].to_numpy(float)
            same = np.allclose(np.nan_to_num(lib, nan=9e9), np.nan_to_num(new, nan=9e9))
            print(tf, k, "identical at defaults:", same, "active", round(float(np.mean(new != 0)), 3))


if __name__ == "__main__":
    main()
