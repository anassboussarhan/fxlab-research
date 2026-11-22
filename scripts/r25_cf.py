"""Round 25 part CF, confluence (docs/round25_preregistration.md).

Members: the library rules at their defaults (the tied defaults for the three of round 24). On one instrument and
timeframe, each member's side and daily returns (market entry, its own mode) are computed once.
Each January, members are ranked by their score (t or mean) over the training window before it. The top n with a
positive score are taken, and the side is the sign of their summed opinions when |sum| >= k x (members taken), else
flat. Trade-mode members' opinions are their entry signals held h bars. Then part P's execution space; with sizing
on, the vote share is the strength.
Placebo: the same with members ranked by random numbers.
"""
from __future__ import annotations

import signal
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit

import intraday17 as I17
import r25_common as C
import r25_exec as X

CF_SPECS = 5                                                               # n, k, window, score, hold


@njit(cache=True)
def _hold(trig, h):
    n = len(trig)
    out = np.zeros(n)
    last = 0.0; age = h + 1
    for t in range(n):
        if trig[t] != 0.0:
            last = np.sign(trig[t]); age = 0
        else:
            age += 1
        out[t] = last if age < h else 0.0
    return out


def members(rules, bars, sym, day, dd, financing=None, member_cap=300):
    """Default side and daily returns of every member: {key: (side array, trade_mode, daily returns)}."""
    import r24_tied
    from fxlab.evaluate import evaluate
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, C._alarm)
    out = {}
    for key, e, fn, sps in rules:
        try:
            signal.alarm(member_cap)
            if key in r24_tied.TIED:
                sig = fn(bars.copy(), **r24_tied.DEFAULTS[key])
            else:
                sig = e.fn(bars.copy(), **dict(e.params or {}))
            r = evaluate(key, bars, sym, signals=sig, financing=financing)["returns"]
            signal.alarm(0)
            side = sig["side"].reindex(bars.index).fillna(0.0).to_numpy(float)
            out[key] = (side, e.extra.get("mode", "state") == "trade", I17.daily_sum(r, day, dd).astype(np.float64))
        except Exception:                                                  # noqa: BLE001  (C.Timeout included)
            signal.alarm(0)
            continue
    return out


def cf_config(u):
    win = [None, 12, 24, 36, 60][min(int(u[2] * 5), 4)]
    return {"n": X._ilog(u[0], 3, 30), "k": 0.5 + 0.5 * u[1], "win": win, "score": "t" if u[3] < 0.5 else "mean",
            "hold": X._ilog(u[4], 1, 48)}


def cf_side(mem, bars_year, years, dd_index, c, rng=None):
    """The confluence side and vote share on the bars, re-ranked each January."""
    keys = list(mem)
    R = np.vstack([mem[k][2] for k in keys])                               # members x days
    dyear = dd_index.year.to_numpy()
    dmon = (dd_index.year * 12 + dd_index.month - 1).to_numpy()
    ops = []
    for k in keys:
        side, trade, _ = mem[k]
        ops.append(_hold(np.sign(side), int(c["hold"])) if trade else np.sign(side))
    O = np.vstack(ops)                                                     # members x bars
    side = np.zeros(O.shape[1])
    share = np.zeros(O.shape[1])
    for Y in sorted(set(years)):
        tr = dyear < Y
        if c["win"] is not None:
            tr &= dmon >= (Y * 12 - c["win"])
        if tr.sum() < 250:                                                 # less than a year of training: flat
            continue
        x = R[:, tr]
        nd = x.shape[1]
        mu = x.mean(1)
        sd = x.std(1, ddof=1)
        act = (x != 0).sum(1)
        sc = mu / (sd / np.sqrt(nd) + 1e-300) if c["score"] == "t" else mu
        if rng is not None:
            sc = rng.normal(size=len(sc))
        sc = np.where((act >= 30) & np.isfinite(sc), sc, -np.inf)
        order = [i for i in np.argsort(-sc, kind="stable")[: c["n"]] if sc[i] > 0]
        if not order:
            continue
        bmask = years == Y
        v = O[order][:, bmask].sum(0)
        m = len(order)
        side[bmask] = np.where(np.abs(v) >= c["k"] * m, np.sign(v), 0.0)
        share[bmask] = np.abs(v) / m
    return side, share


def cf_matrix(rules, bars, sym, tf, n, day, dd, unit_cap, financing=None, crypto=False, placebo=False):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    mem = members(rules, bars, sym, day, dd, financing)
    key = "CF_PLACEBO" if placebo else "CF"
    U = C.design(key, tf, CF_SPECS, n)
    seed = C.seed_of(key, tf)
    dd_index = pd.DatetimeIndex(pd.to_datetime(dd))
    years = pd.DatetimeIndex(pd.to_datetime(day)).year.to_numpy()          # trading-day year of each bar
    M = np.full((n, len(dd)), np.nan, np.float32)
    errors = timeouts = n_run = 0
    signal.signal(signal.SIGALRM, C._alarm)
    cache = {}
    for i, u in enumerate(U):
        if time.time() - t0 > unit_cap or len(mem) < 3:
            break
        n_run += 1
        c = cf_config(u[:CF_SPECS])
        cfg = X.config(u[CF_SPECS:], tf, crypto)
        try:
            signal.alarm(300)
            side, share = cf_side(mem, bars, years, dd_index, c, np.random.default_rng(seed + i) if placebo else None)
            r = X.evaluate25(None, None, bars, sym, {}, cfg, placebo_side=side, financing=financing, cache=cache,
                             strength=share)
            signal.alarm(0)
            M[i] = I17.daily_sum(r, day, dd)
        except C.Timeout:
            signal.alarm(0)
            timeouts += 1
            errors += 1
        except Exception:                                                  # noqa: BLE001
            signal.alarm(0)
            errors += 1
    return M, U, {"n_run": n_run, "errors": errors, "timeouts": timeouts, "members": len(mem),
                  "seconds": round(time.time() - t0, 1)}
