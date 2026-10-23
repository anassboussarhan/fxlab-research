"""Prediction targets. Labels necessarily look forward; each function returns
the label AND ``label_end`` (integer position of the last bar the label uses)
so validation can purge overlapping samples and embargo the test boundary.

All executable-return labels use the SAME timing as the simulator: decision at
close t, entry at open t+1 (ask for longs / bid for shorts), exit at open
t+1+h (bid for longs / ask for shorts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def _end(n: int, h: int) -> np.ndarray:
    return np.minimum(np.arange(n) + h + 1, n - 1)


def forward_mid_return(bars: pd.DataFrame, h: int) -> tuple[pd.Series, np.ndarray]:
    """ln(mid_open[t+1+h] / mid_open[t+1])."""
    mo = (bars["bid_o"] + bars["ask_o"]) / 2
    y = np.log(mo.shift(-(1 + h)) / mo.shift(-1))
    return y, _end(len(bars), h)


def forward_net_return(bars: pd.DataFrame, h: int, side: int, cost_spread_mult: float = 1.0) -> pd.Series:
    """Executable round-trip return of a long (side=1) or short (side=-1) entered at open t+1, exited at open t+1+h."""
    if side > 0:
        entry = bars["ask_o"].shift(-1); exit_ = bars["bid_o"].shift(-(1 + h))
        return exit_ / entry - 1
    entry = bars["bid_o"].shift(-1); exit_ = bars["ask_o"].shift(-(1 + h))
    return entry / exit_ - 1


def direction_label(bars: pd.DataFrame, h: int) -> tuple[pd.Series, np.ndarray]:
    """+1 if a long round trip is profitable after spread, -1 if a short is, 0 if neither (inside the spread)."""
    L, S = forward_net_return(bars, h, 1), forward_net_return(bars, h, -1)
    y = pd.Series(np.where(L > 0, 1.0, np.where(S > 0, -1.0, 0.0)), index=bars.index).where(L.notna())
    return y, _end(len(bars), h)


def volatility_label(bars: pd.DataFrame, h: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Forward realised vol, downside semivol and large-move flag (|ret| > 2 trailing sigma) over h bars."""
    lr = np.log(bars["close"]).diff()
    fwd = pd.concat([lr.shift(-k) for k in range(1, h + 1)], axis=1)
    rv = np.sqrt((fwd ** 2).sum(axis=1, min_count=h))
    dn = np.sqrt((fwd.clip(upper=0) ** 2).sum(axis=1, min_count=h))
    sig = lr.rolling(500, min_periods=100).std() * np.sqrt(h)
    big = (fwd.sum(axis=1, min_count=h).abs() > 2 * sig).astype(float).where(rv.notna() & sig.notna())
    return pd.DataFrame({"rv": rv, "down_rv": dn, "large_move": big}), np.minimum(np.arange(len(bars)) + h, len(bars) - 1)


@njit(cache=True)
def _triple_barrier(bo, bh, bl, ao, ah, al, up_d, dn_d, h, side):
    """side=+1 long: entry ask_o[t+1]; profit barrier on bid high, stop on bid low (stop first if both).
    Returns label (+1 target, -1 stop, 0 time), realised return, end position."""
    n = len(bo)
    lab = np.full(n, np.nan); ret = np.full(n, np.nan); end = np.full(n, n - 1)
    for t in range(n - 2):
        if np.isnan(up_d[t]) or np.isnan(dn_d[t]):
            continue
        j0 = t + 1
        if side > 0:
            e = ao[j0]; tp = e + up_d[t]; sl = e - dn_d[t]
        else:
            e = bo[j0]; tp = e - up_d[t]; sl = e + dn_d[t]
        last = min(n - 1, j0 + h - 1)
        res = 0.0; r = np.nan; k = last
        for j in range(j0, last + 1):
            if side > 0:
                hit_sl = bl[j] <= sl; hit_tp = bh[j] >= tp
            else:
                hit_sl = ah[j] >= sl; hit_tp = al[j] <= tp
            if hit_sl:
                res = -1.0; r = side * (sl - e) / e; k = j
                break
            if hit_tp:
                res = 1.0; r = side * (tp - e) / e; k = j
                break
        if res == 0.0:
            if j0 + h >= n:
                continue  # time barrier beyond data: no label
            x = bo[j0 + h] if side > 0 else ao[j0 + h]
            r = side * (x - e) / e
            k = j0 + h
        lab[t] = res; ret[t] = r; end[t] = k
    return lab, ret, end


def triple_barrier(bars: pd.DataFrame, up: pd.Series, dn: pd.Series, h: int, side: int = 1):
    """Triple-barrier labels (López de Prado) with executable prices and conservative
    stop-first ordering. up/dn: barrier distances in price units decided at t."""
    a = [bars[c].to_numpy(np.float64) for c in ("bid_o", "bid_h", "bid_l", "ask_o", "ask_h", "ask_l")]
    lab, ret, end = _triple_barrier(*a, up.to_numpy(np.float64), dn.to_numpy(np.float64), int(h), int(side))
    return pd.DataFrame({"label": lab, "ret": ret}, index=bars.index), end


def trade_utility(bars: pd.DataFrame, h: int):
    """Expected-utility targets for a long and a short over h bars: net return, MAE (downside risk)."""
    out = {}
    for side, name in ((1, "long"), (-1, "short")):
        out[f"{name}_net"] = forward_net_return(bars, h, side)
        if side > 0:
            lo = pd.concat([bars["bid_l"].shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1)
            out[f"{name}_mae"] = lo / bars["ask_o"].shift(-1) - 1
        else:
            hi = pd.concat([bars["ask_h"].shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
            out[f"{name}_mae"] = bars["bid_o"].shift(-1) / hi - 1
    return pd.DataFrame(out), _end(len(bars), h)


def time_horizon_bars(index: pd.DatetimeIndex, minutes: int) -> int:
    """Convert a fixed elapsed-time horizon to bars for this index (median bar spacing)."""
    dt = pd.Series(index).diff().median()
    return max(1, int(round(pd.Timedelta(minutes=minutes) / dt)))


HORIZONS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
