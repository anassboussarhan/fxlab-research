"""Round 25 shared pieces (docs/round25_preregistration.md): the rule set with identical Sobol spaces in every job,
the per-unit configuration loop, and the selection outputs of one unit.

Rule exposure (round 23's exposed constants) is verified on a fixed synthetic probe, not on the instrument, so that a
rule has the same parameter list, and configuration i the same values, on every instrument and in every job (panel
selection depends on it).
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import intraday17 as I17  # noqa: E402
import r23_LW as LW  # noqa: E402
import r25_exec as X  # noqa: E402
import wf25  # noqa: E402

N_PLACEBO = 10
DEV_END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
PLACEBO_SPECS = [{"name": "persist", "kind": "float", "lo": 0.9, "hi": 0.99}]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def probe_bars():
    from r25_engine_check import synth_bars
    return synth_bars(seed=3, tf="H1").iloc[-3000:]


def prepare_rules(tf, only=None):
    """[(key, e, fn, specs)] for the round-25 rule set at timeframe tf."""
    import r23_expose as XP
    import r24_tied
    import r25_tied
    probe = probe_bars()
    out = []
    for e in I17.strategies():
        if only and e.id not in only:
            continue
        if e.id == "S.RG.HMM" and tf == "H1":                             # stuck below H2 (round 23)
            continue
        tied = r24_tied.TIED.get(e.id) or r25_tied.TIED25.get(e.id)       # constants tied by role (rounds 24, 25)
        if tied is not None:
            fn, sps = tied
            out.append((e.id, e, fn, list(sps)))
            continue
        fn, consts = XP.expose(e)
        ok = bool(consts) and XP.verified(e, fn, probe)
        if not ok:
            fn, consts = e.fn, []
        out.append((e.id, e, fn, LW.strategy_specs(e, consts)))
    return out


def seed_of(key, tf, tag="LW25"):
    return zlib.crc32(f"{tag}|{key}|{tf}".encode()) & 0x7FFFFFFF


def design(key, tf, n_specs, n, extra=0, tag="LW25"):
    return qmc.Sobol(n_specs + extra + X.N_EXEC, scramble=True, seed=seed_of(key, tf, tag)).random(n)


def placebo_side(seed, i, persist, nb):
    rng = np.random.default_rng(seed + i)
    sw = rng.random(nb) > persist
    drawv = rng.integers(-1, 2, nb).astype(float)
    ix = np.maximum.accumulate(np.where(sw, np.arange(nb), 0))
    ps = drawv[ix]
    ps[: int(np.argmax(sw))] = 0.0
    return ps


def unit_matrix(key, e, fn, sps, bars, sym, tf, n, day, dd, unit_cap, financing=None, crypto=False):
    """n x days matrix of daily net returns of a P unit (NaN = configuration not run); with counters."""
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    plc = key.startswith("PLACEBO")
    if plc:
        sps = PLACEBO_SPECS
    seed = seed_of(key, tf)
    U = design(key, tf, len(sps), n)
    M = np.full((n, len(dd)), np.nan, np.float32)
    t0 = time.time()
    errors = timeouts = n_run = 0
    cache = {}
    for i, u in enumerate(U):
        if time.time() - t0 > unit_cap:
            break
        n_run += 1
        params = {s["name"]: LW.draw(s, u[j]) for j, s in enumerate(sps)}
        cfg = X.config(u[len(sps):], tf, crypto)
        ps = None
        if plc:
            ps = placebo_side(seed, i, params["persist"], len(bars))
        try:
            signal.alarm(300)
            r = X.evaluate25(None if plc else e, None if plc else fn, bars, sym, {} if plc else params, cfg,
                             sps=None if plc else sps, u_params=None if plc else u[:len(sps)],
                             seed=seed + 7919 * i, placebo_side=ps, financing=financing, cache=cache)
            signal.alarm(0)
            M[i] = I17.daily_sum(r, day, dd)
        except Timeout:
            signal.alarm(0)
            timeouts += 1
            errors += 1
        except Exception:                                                  # noqa: BLE001
            signal.alarm(0)
            errors += 1
    return M, U, {"n_run": n_run, "errors": errors, "timeouts": timeouts, "seconds": round(time.time() - t0, 1)}


def unit_select(M, U, dd, member=None):
    """Per-instrument streams of a unit: counted (default, nested_plus) and diagnostics (nested23, nbr, share,
    decay24). member: optional boolean function of a DatetimeIndex (part ALT); streams are kept on member days."""
    cal = pd.DatetimeIndex(pd.to_datetime(dd))
    first = max(2010, int(cal.year.min()) + 3)
    ag = wf25.Agg25(cal, M.astype(np.float64), U)
    kw = {"min_active": 30, "min_valid": 50}
    out = {}
    s, _ = wf25.walk_forward(ag, first, wf25.wf23.DEFAULT, **kw)
    out["default"] = s
    out["nested_plus"], ch = wf25.nested(ag, first, wf25.SETTINGS_PLUS, **kw)
    out["nested23"], _ = wf25.nested(ag, first, wf25.SETTINGS23, **kw)
    for k, st in wf25.FIXED_DIAG.items():
        out[k], _ = wf25.walk_forward(ag, first, st, **kw)
    if member is not None:
        out = {k: v[member(v.index)] for k, v in out.items()}
    return out, ag, {"nested_plus_last": {str(k): v for k, v in list(ch.items())[-2:]}}


COUNTED = ("default", "nested_plus")


def rows_of(streams, base, counted=COUNTED, use_p1=True):
    rows = []
    for kind, s in streams.items():
        st = wf25.stats(s)
        rows.append({**base, "selection": kind, "counted_kind": kind in counted, **st,
                     "own": wf25.counts_rule(st, use_p1) if len(s) else False})
    return rows


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=0, default=str)
