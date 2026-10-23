"""Entry-timing transforms applied to a target-position series BEFORE execution (exits are never delayed).

confirm_entries : a new entry (or reversal) is only taken once a later bar CLOSES beyond the signal bar's close in
                  the signal's direction, within k bars; otherwise the signal is skipped until it changes.
session_entries : new entries only when the next bar starts inside a local-time window (e.g. London 07-17);
                  exits and position reductions are always allowed; a persisting signal enters at the window open.
Both are causal: the output at t uses side[0..t] and close[0..t] only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=False)
def _confirm(side, close, k):
    n = len(side)
    out = np.zeros(n)
    o = 0.0; pend = 0.0; ref = 0.0; cnt = 0; skip = 0.0
    for t in range(n):
        s = side[t]
        if s == 0.0:
            o = 0.0; pend = 0.0; skip = 0.0
        elif s == o:
            pass
        else:
            if o != 0.0 and s * o < 0:          # reversal: exit now, re-entry needs confirmation
                o = 0.0
            if s == skip:
                pass                            # this signal already failed confirmation
            elif pend == s:
                if (close[t] - ref) * s > 0:
                    o = s; pend = 0.0
                else:
                    cnt -= 1
                    if cnt <= 0:
                        pend = 0.0; skip = s
            else:
                pend = s; ref = close[t]; cnt = k; skip = 0.0
        out[t] = o
    return out


def confirm_entries(side: pd.Series, close: pd.Series, k: int = 1) -> pd.Series:
    return pd.Series(_confirm(side.fillna(0).to_numpy(float), close.to_numpy(float), int(k)), index=side.index)


@njit(cache=False)
def _session(side, allowed):
    n = len(side)
    out = np.zeros(n)
    o = 0.0
    for t in range(n):
        s = side[t]
        if s == o:
            pass
        elif s == 0.0 or (o != 0.0 and abs(s) < abs(o) and s * o > 0):
            o = s                                   # exits / reductions always allowed
        elif allowed[t]:
            o = s
        elif o != 0.0 and s * o < 0:
            o = 0.0                                 # reversal outside the window: exit, wait for the window
        out[t] = o
    return out


def session_entries(side: pd.Series, tz: str = "Europe/London", start: float = 7.0, end: float = 17.0) -> pd.Series:
    idx = side.index
    step = pd.Series(idx[: min(len(idx), 5000)]).diff().dropna().median() if len(idx) > 2 else pd.Timedelta(hours=1)
    nxt = (idx + step).tz_convert(tz)
    h = nxt.hour + nxt.minute / 60.0
    allowed = np.asarray((h >= start) & (h < end) & (nxt.dayofweek < 5))
    return pd.Series(_session(side.fillna(0).to_numpy(float), allowed), index=idx)
