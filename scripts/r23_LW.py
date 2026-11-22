"""Round 23 part LW (docs/round23_preregistration.md): the 105 single-asset catalog strategies with every parameter
free (grid parameters widened, keyword defaults and the constants inside the function exposed), drawn jointly with
continuous entry, exit and direction settings (Sobol 128 per strategy x timeframe), at M15 ... W1, default and nested
walk-forward per unit; 10 random-state placebo units per timeframe through the same execution settings.
Usage: python scripts/r23_LW.py --symbol SYM --out DIR [--n 128] [--budget 19000] [--procs 4]
"""
from __future__ import annotations

import argparse
import faulthandler
import inspect
import resource
import json
import math
import multiprocessing as mp
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
import r23_expose as XP  # noqa: E402
import wf23  # noqa: E402

TFS = ("W1", "D2", "D1", "H12", "H8", "H4", "H2", "H1", "M30", "M15")    # cheapest first: a budget cut drops M15 first
N_EXEC = 19
N_PLACEBO = 10
KILL_MARGIN = 900                                                          # s past the unit cap before a stuck worker is killed
DEV_END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
G = {}                                                                     # bars per timeframe etc. (inherited by fork)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# ------------------------------------------------------------------------------------------- parameter ranges
def spec_values(name, vals):
    vals = list(vals)
    if any(isinstance(v, (str, bool)) for v in vals) or set(vals) <= {0, 1}:
        return {"name": name, "kind": "cat", "choices": vals}
    lo, hi = min(vals), max(vals)
    if all(isinstance(v, (int, np.integer)) for v in vals):
        a, b = max(1, int(math.floor(lo / 2))), max(2, int(math.ceil(hi * 2)))
        return {"name": name, "kind": "int", "lo": a, "hi": b, "log": a >= 2}
    a = lo / 2 if lo > 0 else lo * 2
    b = hi * 2 if hi > 0 else hi / 2
    return {"name": name, "kind": "float", "lo": a, "hi": b, "log": False}


def spec_default(name, v):
    if isinstance(v, bool):
        return {"name": name, "kind": "cat", "choices": [v, not v]}
    if isinstance(v, str) or v is None:
        return None
    if isinstance(v, (int, np.integer)):
        if v in (0, 1):
            return {"name": name, "kind": "cat", "choices": [0, 1]}
        if v >= 2:
            return {"name": name, "kind": "int", "lo": max(2, v // 3), "hi": 3 * v, "log": True}
        return {"name": name, "kind": "int", "lo": 3 * v, "hi": min(-1, v // 3), "log": False}
    if isinstance(v, float):
        if v == 0:
            return None
        a, b = sorted((v / 3, 3 * v))
        return {"name": name, "kind": "float", "lo": a, "hi": b, "log": False}
    return None


def draw(sp, u):
    if sp["kind"] == "cat":
        return sp["choices"][min(int(u * len(sp["choices"])), len(sp["choices"]) - 1)]
    lo, hi = sp["lo"], sp["hi"]
    if sp["kind"] == "int":
        if sp.get("log") and lo > 0:
            return int(round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))))
        return int(round(lo + u * (hi - lo)))
    return float(lo + u * (hi - lo))


def strategy_specs(e, consts):
    sps, seen = [], set()
    for k, vals in (e.space or {}).items():
        sps.append(spec_values(k, vals))
        seen.add(k)
    try:
        for p, v in inspect.signature(e.fn).parameters.items():
            if p in seen or v.default is inspect._empty:
                continue
            s = spec_default(p, e.params.get(p, v.default))
            if s:
                sps.append(s)
                seen.add(p)
    except (TypeError, ValueError):
        pass
    for name, v in consts:
        s = spec_default(name, v)
        if s:
            sps.append(s)
    return sps


def execution(u):
    """Map the 19 execution dimensions to (direction, mode, entry config, risk override)."""
    d = -1 if u[0] < 0.5 else 1
    mode = "native" if u[1] < 0.5 else "custom"
    et = ["market", "pullback", "stop", "confirm", "session"][min(4, int(u[2] * 5))]
    if et == "market":
        ent = None
    elif et == "pullback":
        ent = {"type": "order", "entry": "limit_pullback", "k": 0.05 + u[3] * 1.45, "ttl": int(1 + u[4] * 23.999)}
    elif et == "stop":
        ent = {"type": "order", "entry": "stop_breakout", "k": 0.05 + u[3] * 1.45, "ttl": int(1 + u[4] * 23.999)}
    elif et == "confirm":
        ent = {"type": "confirm", "k": int(1 + u[3] * 4.999)}
    else:
        st = int(u[3] * 23.999)
        ent = {"type": "session", "tz": "Europe/London", "start": float(st), "end": float(min(24, st + 4 + int(u[4] * 20.999)))}
    sk = ["atr", "realized_vol", "structural", "none"][min(3, int(u[5] * 4))]
    if sk == "atr":
        stop = ("atr", {"mult": 0.5 + u[6] * 5.5})
    elif sk == "realized_vol":
        stop = ("realized_vol", {"mult": 0.5 + u[6] * 5.5})
    elif sk == "structural":
        stop = ("structural", {"k": int(2 + u[6] * 8.999), "buffer_atr": u[7]})
    else:
        stop = ("atr", {"mult": 100.0})
    risk = {"stop": stop[0], "stop_params": stop[1], "sizing": "none",
            "target": "affine" if u[8] >= 0.5 else "none",
            "target_params": {"a": 0.0, "b": 0.5 + u[9] * 5.5} if u[8] >= 0.5 else {},
            "trail_atr": (0.5 + u[11] * 5.5) if u[10] >= 0.5 else None,
            "breakeven_r": (0.3 + u[13] * 2.7) if u[12] >= 0.5 else None,
            "partial_r": (0.5 + u[15] * 3.5) if u[14] >= 0.5 else None, "partial_frac": 0.2 + u[16] * 0.6,
            "max_hold": int(round(math.exp(math.log(2) + u[18] * (math.log(500) - math.log(2))))) if u[17] >= 0.5 else None}
    return d, mode, ent, risk


def evaluate_one(e, fn, bars, sym, params, u_exec, placebo_side=None):
    from fxlab.evaluate import evaluate
    d, mode, ent, risk = execution(u_exec)
    if placebo_side is None:
        sig = fn(bars.copy(), **params)
    else:
        sig = pd.DataFrame({"side": placebo_side}, index=bars.index)
    sig = sig.copy()
    sig["side"] = sig["side"] * d
    fin = G.get("financing")                                                # round 24 R1: "rates_oecd"
    if mode == "native":
        r = evaluate(e.id, bars, sym, signals=sig, exec_cfg=ent, financing=fin) if e is not None else \
            evaluate("S.TREND.MA_CROSS", bars, sym, signals=sig, exec_cfg=ent, mode_override="state", financing=fin)
    else:
        r = evaluate(e.id if e is not None else "S.TREND.MA_CROSS", bars, sym, signals=sig[["side"]], mode_override="trade",
                     risk_override=risk, exec_cfg=ent, financing=fin)
    return r["returns"]


def _proc_kb(path, field):
    """A kB field of /proc/self/status or /proc/meminfo."""
    try:
        for line in open(path):
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def run_unit(args):
    """One unit: strategy (or placebo k) x timeframe. Returns rows, pooled daily sums and the time taken."""
    key, tf, n = args
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    if G.get("mem_allow") and not G.get("_lim"):                           # once per worker: a runaway allocation
        try:                                                               # raises MemoryError in its configuration
            lim = _proc_kb("/proc/self/status", "VmData") * 1024 + G["mem_allow"]   # instead of the kernel killing
            resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))           # the worker. The limit is on top of what
            G["_lim"] = lim                                                # the worker inherited from the parent.
        except (ValueError, OSError, TypeError):
            G["_lim"] = -1
    G["q"].put((key, tf, os.getpid(), time.time()))
    bars, day, dd, sym = G["bars"][tf], G["day"][tf], G["dd"][tf], G["sym"]
    t0 = time.time()
    seed = zlib.crc32(f"LW|{key}|{tf}".encode()) & 0x7FFFFFFF
    if key.startswith("PLACEBO"):
        e, fn, sps = None, None, [{"name": "persist", "kind": "float", "lo": 0.9, "hi": 0.99}]
    else:
        e, fn, sps = G["cat"][key], G["fn"][key], G["specs"][key]
    U = qmc.Sobol(len(sps) + N_EXEC, scramble=True, seed=seed).random(n)
    M = np.full((n, len(dd)), np.nan, np.float32)
    errors, timeouts, n_run = 0, 0, 0
    for i, u in enumerate(U):
        if time.time() - t0 > G["unit_cap"]:                               # unit cap: the remaining configurations
            break                                                          # are no data
        n_run += 1
        params = {s["name"]: draw(s, u[j]) for j, s in enumerate(sps)}
        ps = None
        if e is None:
            rng = np.random.default_rng(seed + i)
            nb = len(bars)
            sw = rng.random(nb) > params["persist"]
            drawv = rng.integers(-1, 2, nb).astype(float)
            ix = np.maximum.accumulate(np.where(sw, np.arange(nb), 0))
            ps = drawv[ix]
            ps[: int(np.argmax(sw))] = 0.0
            params = {}
        try:
            signal.alarm(300)
            faulthandler.dump_traceback_later(600, exit=False)              # a configuration stuck past its alarm
            r = evaluate_one(e, fn, bars, sym, params, u[len(sps):], ps)
            signal.alarm(0)
            M[i] = I17.daily_sum(r, day, dd)
        except Timeout:
            signal.alarm(0)
            timeouts += 1
            errors += 1
        except Exception:                                                  # noqa: BLE001
            signal.alarm(0)
            errors += 1
        finally:
            faulthandler.cancel_dump_traceback_later()
    sec = time.time() - t0
    if sec > 600 or n_run < n:
        print("SLOW", key, tf, "seconds", round(sec), "configs", n_run, "of", n, "timeouts", timeouts, "errors", errors, flush=True)
    cal = pd.DatetimeIndex(pd.to_datetime(dd))
    first = max(2010, int(cal.year.min()) + 3)
    res = wf23.both(wf23.Agg(cal, P=M.astype(np.float64)), first, min_active=30, min_valid=50)
    rows, pooled = [], {}
    for kind, (s, st, own, ch) in res.items():
        rows.append({"part": "LW", "family": f"LW-{key}", "unit": f"{key}|{tf}", "tf": tf, "symbol": sym,
                     "placebo": key.startswith("PLACEBO"), "selection": kind, "n_cfg": n, "n_run": n_run, "errors": errors, "timeouts": timeouts, **st, "own": own,
                     "n_params": len(sps), "choices": json.dumps({str(k): v for k, v in list(ch.items())[-1:]})[:600]})
        pooled[kind] = s.astype(np.float32)
    return key, tf, rows, pooled, sec


def higher(d1, kind):
    """D2 (pairs of D1 bars) or W1 (Monday-Friday weeks) from D1 bars."""
    if kind == "D2":
        kv = np.arange(len(d1)) // 2
    else:
        iso = d1.index.tz_convert("America/New_York").isocalendar()
        kv = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    agg = {"bid_o": "first", "bid_h": "max", "bid_l": "min", "bid_c": "last", "ask_o": "first", "ask_h": "max",
           "ask_l": "min", "ask_c": "last", "n_ticks": "sum", "spread_mean": "mean", "spread_max": "max", "spread_min": "min",
           "rv_intrabar": "sum", "n_m1": "sum", "bid_liq": "sum", "ask_liq": "sum", "up_ticks": "sum", "down_ticks": "sum"}
    agg = {c: f for c, f in agg.items() if c in d1.columns}
    b = d1.groupby(kv, sort=True).agg(agg)
    starts = np.r_[0, np.flatnonzero(kv[1:] != kv[:-1]) + 1]
    b.index = d1.index[starts]
    from fxlab.data.store import add_mid
    return add_mid(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--budget", type=float, default=19000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--unit-cap", type=float, default=1800, help="seconds; a unit stops drawing configurations after this")
    ap.add_argument("--tied", action="store_true", help="round 24: use scripts/r24_tied.py for the rules it lists")
    ap.add_argument("--placebos", type=int, default=None, help="number of placebo units (default N_PLACEBO)")
    ap.add_argument("--financing", default=None, help="rates_oecd: rate-differential financing in every engine (round 24)")
    ap.add_argument("--tfs", default=",".join(TFS))
    ap.add_argument("--only", default="", help="test runs: comma-separated strategy ids")
    ap.add_argument("--save-bars", default="", help="round 25: write the H1-W1 bars used here to this directory")
    ap.add_argument("--save-series", action="store_true", help="round 25 part HAC: keep every unit's streams")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_job = time.time()
    from fxlab.data.instruments import get as instrument
    from fxlab.data.store import load_m1, resample
    sym = a.symbol
    m1 = load_m1(sym)
    m1 = m1[m1.index <= DEV_END]
    try:
        instrument(sym)
    except KeyError:                                                       # index / commodity CFDs: about 1 bp of price
        from fxlab.data.instruments import INSTRUMENTS, Instrument
        pip = float(1e-4 * np.median(0.5 * (m1.bid_c + m1.ask_c)))
        INSTRUMENTS[sym] = Instrument(sym, sym[:-3], sym[-3:], pip, 1.0, "index")
    G["sym"] = sym
    G["financing"] = a.financing
    if a.financing == "rates_oecd":
        from fxlab.data import rates_oecd
        print("carry", sym, "has rates", rates_oecd.has_rates(sym), flush=True)
    G["bars"], G["day"], G["dd"] = {}, {}, {}
    tfs = [t for t in a.tfs.split(",") if t]
    d1 = None
    for tf in tfs:
        if tf in ("D2", "W1"):
            d1 = d1 if d1 is not None else resample(m1, "D1")
            b = higher(d1, tf)
        else:
            b = resample(m1, tf)
            if tf == "D1":
                d1 = b
        b = b[b.index <= DEV_END]
        G["bars"][tf] = b
        G["day"][tf] = I17.trading_day(b.index)
        G["dd"][tf] = pd.Index(sorted(set(G["day"][tf])))
        print(sym, tf, "bars", len(b), flush=True)
        if a.save_bars and tf in ("H1", "H2", "H4", "H8", "H12", "D1", "D2", "W1"):
            os.makedirs(a.save_bars, exist_ok=True)
            b.to_parquet(os.path.join(a.save_bars, f"{sym}_{tf}.parquet"), compression="zstd")
    del m1
    S = I17.strategies()
    if a.only:
        S = [e for e in S if e.id in a.only.split(",")]
    probe = G["bars"]["H1"] if "H1" in G["bars"] else G["bars"][tfs[0]]
    G["cat"], G["fn"], G["specs"], expo = {}, {}, {}, {}
    TIED = {}
    if a.tied:                                                             # round 24 R2 / R3: constants tied by role
        import r24_tied
        TIED = r24_tied.TIED
    for e in S:
        if e.id in TIED:
            fn, sps = TIED[e.id]
            G["cat"][e.id], G["fn"][e.id], G["specs"][e.id] = e, fn, list(sps)
            expo[e.id] = {"tied": True, "params": [s["name"] for s in sps]}
            continue
        fn, consts = XP.expose(e)
        ok = bool(consts) and XP.verified(e, fn, probe.iloc[-3000:])
        if not ok:
            fn, consts = e.fn, []
        G["cat"][e.id], G["fn"][e.id] = e, fn
        G["specs"][e.id] = strategy_specs(e, consts)
        expo[e.id] = {"constants_exposed": len(consts), "verified": ok, "params": [s["name"] for s in G["specs"][e.id]]}
    npl = N_PLACEBO if a.placebos is None else a.placebos
    units = [(e.id, tf, a.n) for tf in tfs for e in S] + [(f"PLACEBO_{k:02d}", tf, a.n) for tf in tfs for k in range(npl)]
    rows, done, skipped, timing = [], set(), [], {}
    pooled = {}
    unit_series = {}

    def write(final):
        """Write rows, pooled sums and meta (also as a checkpoint, so a job killed at its time limit keeps its work)."""
        sk = [f"{k}|{t}" for k, t, _ in units if (k, t) not in done]
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"LW_rows_{sym}.csv"), index=False)
        P = {}
        for (key, kind), (sm, ct) in pooled.items():
            P[f"{key}|{kind}|sum"] = sm.astype(np.float32)
            P[f"{key}|{kind}|n"] = ct.astype(np.float32)
        if P:
            pd.DataFrame(P).to_parquet(os.path.join(a.out, f"LW_pooled_{sym}.parquet"))
        if unit_series:
            pd.DataFrame(unit_series).to_parquet(os.path.join(a.out, f"LW_series_{sym}.parquet"))
        json.dump({"symbol": sym, "tfs": tfs, "units": len(units), "done": len(done), "skipped": sk, "final": final,
                   "seconds": round(time.time() - t_job), "lost": G.get("lost", []), "unit_cap": a.unit_cap,
                   "exposure": expo, "unit_seconds": timing},
                  open(os.path.join(a.out, f"LW_meta_{sym}.json"), "w"), indent=0)
        return sk
    deadline = t_job + a.budget
    t_ck = time.time()
    ctx = mp.get_context("fork")
    G["q"], G["unit_cap"] = ctx.Queue(), a.unit_cap
    try:                                                                   # new memory per worker: 80% of the
        G["mem_allow"] = int(0.8 * _proc_kb("/proc/meminfo", "MemAvailable") * 1024 / a.procs)   # available / procs
    except (ValueError, OSError, TypeError):
        G["mem_allow"] = None
    print("memory allowance per worker GB", round((G["mem_allow"] or 0) / 1e9, 2), "parent VmData GB",
          round(_proc_kb("/proc/self/status", "VmData") * 1024 / 1e9, 2), flush=True)
    pool = ctx.Pool(a.procs, maxtasksperchild=20)
    pend = {(k, t): pool.apply_async(run_unit, ((k, t, n),)) for k, t, n in units}
    t_pool = time.time()
    started, lost, sus = {}, [], {}
    while pend and time.time() < deadline:
        time.sleep(2)
        while True:
            try:
                k_, t_, pid, ts = G["q"].get_nowait()
                started[(k_, t_)] = (pid, ts)
            except Exception:                                              # noqa: BLE001  (queue empty)
                break
        alive = {w.pid for w in pool._pool if w.is_alive()}
        cur = {}                                                           # the unit each worker is running now
        for u_, (pid, ts) in started.items():
            if u_ in pend and (pid not in cur or ts > cur[pid][1]):
                cur[pid] = (u_, ts)
        for pid, (u_, ts) in cur.items():                                  # a configuration stuck where its alarm
            if pid in alive and time.time() - ts > a.unit_cap + KILL_MARGIN and not pend[u_].ready():   # cannot reach it:
                print("KILL (stuck unit)", u_, "pid", pid, flush=True)     # kill the worker; the unit is then lost
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        for u_, r_ in list(pend.items()):
            if r_.ready():
                del pend[u_]
                try:
                    key, tf, rr, pl, sec = r_.get()
                except Exception as ex:                                    # noqa: BLE001
                    print("UNIT ERROR", u_, repr(ex)[:200], flush=True)
                    continue
                rows.extend(rr)
                done.add((key, tf))
                timing[f"{key}|{tf}"] = round(sec, 1)
                if a.save_series:
                    for kind, s_ in pl.items():
                        unit_series[f"{key}|{tf}|{kind}"] = s_.astype(np.float32)
                for kind, s_ in pl.items():
                    p_ = pooled.setdefault((key, kind), [pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)])
                    p_[0] = p_[0].add(s_.astype(np.float64), fill_value=0.0)
                    p_[1] = p_[1].add(pd.Series(1.0, index=s_.index), fill_value=0.0)
            elif u_ in started and started[u_][0] not in alive:                # its worker is gone: if the result
                sus.setdefault(u_, time.time())                                # has not arrived 30 s later, the
                if time.time() - sus[u_] > 30:                                 # worker died and it never will
                    del pend[u_]
                    lost.append(f"{u_[0]}|{u_[1]}")
                    print("LOST (worker died)", u_, flush=True)
        if not started and not done and time.time() - t_pool > 900:        # no worker has reported in: fail loudly
            print("NO UNIT STARTED in 900 s: the workers are not running", flush=True)
            pool.terminate()
            write(True)
            sys.exit(3)
        if time.time() - t_ck > 900:
            write(False)
            t_ck = time.time()
    pool.terminate()
    G["lost"] = lost
    skipped = write(True)
    slow = sorted(timing.items(), key=lambda kv: -kv[1])[:10]
    print("slowest units", slow, flush=True)
    print("skipped units", skipped[:30], "lost", lost[:30], flush=True)
    print(sym, "units", len(units), "done", len(done), "skipped", len(skipped), "seconds", round(time.time() - t_job), flush=True)


if __name__ == "__main__":
    main()
