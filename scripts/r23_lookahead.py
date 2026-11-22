"""Round 23 look-ahead audit of part LW (docs/round23_results.md): truncation test of every LW strategy.

For each strategy and timeframe, the first K configurations of its LW Sobol sequence (rule parameters, exposed
constants and execution settings, exactly as scripts/r23_LW.py draws them) are run on a full synthetic series and on
the same series cut at several points. A causal rule gives the same signal on every bar before the cut, and the
engine the same per-bar return on every bar before the cut except the last two (where an open trade is closed).
Any difference is a look-ahead. Synthetic data: a random walk on M1 with a bid / ask spread, 2023-2024.
Usage: python scripts/r23_lookahead.py --out DIR [--k 6] [--tfs M15,H1,H4,D1] [--procs 2] [--engine]
"""
from __future__ import annotations

import argparse
import json
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
import r23_LW as LW  # noqa: E402

G = {}
SYM = "TSTUSD"
TASK_KILL = 1500                                                           # s before a stuck task's worker is killed


def synth_m1(path):
    rng = np.random.default_rng(7)
    ts = pd.date_range("2023-01-02", "2024-12-31 23:59", freq="min", tz="UTC")
    ts = ts[ts.dayofweek < 5]
    p = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.00012, len(ts))))
    o = np.r_[p[0], p[:-1]]
    w = np.abs(rng.normal(0, 0.00006, len(ts)))
    h, l = np.maximum(o, p) * (1 + w), np.minimum(o, p) * (1 - w)
    sp = 1.5e-5
    df = pd.DataFrame({"ts": ts, "bid_o": o, "bid_h": h, "bid_l": l, "bid_c": p, "ask_o": o + sp, "ask_h": h + sp,
                       "ask_l": l + sp, "ask_c": p + sp, "n_ticks": 10, "spread_mean": sp})
    os.makedirs(path, exist_ok=True)
    for y in (2023, 2024):
        df[df.ts.dt.year == y].to_parquet(os.path.join(path, f"{SYM}_{y}.parquet"), index=False)


def _alarm(signum, frame):
    raise TimeoutError()


def diff_frame(a, b, n_cmp):
    """Columns of signal frames a (full) and b (truncated) that differ on the first n_cmp bars."""
    out = []
    for c in b.columns:
        if c not in a:
            continue
        x = pd.to_numeric(a[c].iloc[:n_cmp], errors="coerce").to_numpy(float)
        y = pd.to_numeric(b[c].iloc[:n_cmp], errors="coerce").to_numpy(float)
        bad = ~(np.isclose(x, y, rtol=1e-9, atol=1e-12) | (np.isnan(x) & np.isnan(y)))
        if bad.any():
            out.append({"col": c, "n_bad": int(bad.sum()), "first_bad": str(a.index[int(np.argmax(bad))])})
    return out


def task(args):
    key, tf, k, engine = args
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    G["q"].put((key, tf, os.getpid(), time.time()))
    bars = G["bars"][tf]
    e, fn, sps = G["cat"][key], G["fn"][key], G["specs"][key]
    seed = zlib.crc32(f"LW|{key}|{tf}".encode()) & 0x7FFFFFFF
    U = qmc.Sobol(len(sps) + LW.N_EXEC, scramble=True, seed=seed).random(max(k, 1))
    n = len(bars)
    # signal cuts: 48 points in the last 40% of the series, spread over every hour of the day (an intraday leak
    # shows only when the cut falls inside the leaking window); engine cuts: 4 of them
    rs = np.random.default_rng(seed)
    pos = np.sort(rs.choice(np.arange(int(0.6 * n), n - 2), size=min(48, max(1, n // 3)), replace=False))
    hrs = bars.index[pos].hour
    order = np.lexsort((pos, hrs))                                         # interleave the hours
    cuts = [int(c) for c in pos[order]]
    ecuts = set(cuts[:: max(1, len(cuts) // 4)][:4])
    res = {"key": key, "tf": tf, "configs": 0, "signal_lookahead": [], "engine_lookahead": [], "errors": 0,
           "last_bar_only": 0}
    t0 = time.time()
    for i, u in enumerate(U):
        params = {s["name"]: LW.draw(s, u[j]) for j, s in enumerate(sps)}
        u_exec = u[len(sps):]
        try:
            signal.alarm(240)
            full = fn(bars.copy(), **params)
            rf = LW.evaluate_one(e, fn, bars, SYM, params, u_exec) if engine else None
            for c in cuts:
                tb = bars.iloc[:c]
                part = fn(tb.copy(), **params)
                d = diff_frame(full, part, c - 1)                          # every bar before the cut but the last
                if d:
                    res["signal_lookahead"].append({"config": i, "cut": str(bars.index[c]), "diff": d[:3],
                                                    "params": {kk: (round(v, 4) if isinstance(v, float) else v)
                                                               for kk, v in params.items()}})
                elif diff_frame(full, part, c):                            # only the last bar differs: an end-of-day
                    res["last_bar_only"] += 1                              # or end-of-data rule (the series ends there)
                if engine and c in ecuts:
                    rp = LW.evaluate_one(e, fn, tb, SYM, params, u_exec)
                    m = max(0, c - 2)
                    x, y = np.asarray(rf.iloc[:m], float), np.asarray(rp.iloc[:m], float)
                    bad = ~(np.isclose(x, y, rtol=1e-9, atol=1e-12) | (np.isnan(x) & np.isnan(y)))
                    if bad.any():
                        res["engine_lookahead"].append({"config": i, "cut": str(bars.index[c]), "n_bad": int(bad.sum()),
                                                        "first_bad": str(rf.index[int(np.argmax(bad))]),
                                                        "exec": [round(float(v), 3) for v in u_exec]})
            signal.alarm(0)
            res["configs"] += 1
        except Exception as ex:                                            # noqa: BLE001
            signal.alarm(0)
            res["errors"] += 1
            res.setdefault("error", repr(ex)[:160])
        if time.time() - t0 > 900:
            break
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tfs", default="M15,H1,H4,D1")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--tied", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "m1", SYM)
    if not os.path.exists(os.path.join(root, f"{SYM}_2024.parquet")):
        synth_m1(root)
    from fxlab.data.instruments import INSTRUMENTS, Instrument
    from fxlab.data.store import load_m1, resample
    INSTRUMENTS[SYM] = Instrument(SYM, "TST", "USD", 1e-4, 1.0, "fx")
    m1 = load_m1(SYM)
    G["bars"] = {tf: resample(m1, tf) for tf in a.tfs.split(",")}
    del m1
    S = I17.strategies()
    if a.only:
        S = [e for e in S if e.id in a.only.split(",")]
    probe = G["bars"]["H1"] if "H1" in G["bars"] else next(iter(G["bars"].values()))
    G["cat"], G["fn"], G["specs"] = {}, {}, {}
    TIED = {}
    if a.tied:
        import r24_tied
        TIED = r24_tied.TIED
    for e in S:
        if e.id in TIED:
            fn, sps = TIED[e.id]
            G["cat"][e.id], G["fn"][e.id], G["specs"][e.id] = e, fn, list(sps)
            continue
        fn, consts = XP.expose(e)
        if not (consts and XP.verified(e, fn, probe.iloc[-3000:])):
            fn, consts = e.fn, []
        G["cat"][e.id], G["fn"][e.id], G["specs"][e.id] = e, fn, LW.strategy_specs(e, consts)
    jobs = [(e.id, tf, a.k, a.engine) for tf in a.tfs.split(",") for e in S]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    G["q"] = ctx.Queue()
    pool = ctx.Pool(a.procs)
    pend = {(j[0], j[1]): pool.apply_async(task, (j,)) for j in jobs}
    started, out, killed, t_last = {}, [], [], time.time()
    while pend:
        time.sleep(2)
        while True:
            try:
                k_, t_, pid, ts = G["q"].get_nowait()
                started[(k_, t_)] = (pid, ts)
            except Exception:                                              # noqa: BLE001  (queue empty)
                break
        alive = {w.pid for w in pool._pool if w.is_alive()}
        cur = {}
        for u_, (pid, ts) in started.items():
            if u_ in pend and (pid not in cur or ts > cur[pid][1]):
                cur[pid] = (u_, ts)
        for u_, r_ in list(pend.items()):
            if r_.ready():
                del pend[u_]
                try:
                    r = r_.get()
                except Exception as ex:                                    # noqa: BLE001
                    print("TASK ERROR", u_, repr(ex)[:200], flush=True)
                    continue
                out.append(r)
                if r["signal_lookahead"] or r["engine_lookahead"]:
                    print("LOOKAHEAD", r["key"], r["tf"], "configs", r["configs"], "signal", len(r["signal_lookahead"]),
                          "engine", len(r["engine_lookahead"]), flush=True)
                    for x in (r["signal_lookahead"][:2] + r["engine_lookahead"][:2]):
                        print("   ", json.dumps(x, default=str)[:400], flush=True)
            elif u_ in started and started[u_][0] not in alive and time.time() - started[u_][1] > 30:
                del pend[u_]
                killed.append(f"{u_[0]}|{u_[1]}")
        for pid, (u_, ts) in cur.items():                                  # a task stuck where its alarm cannot reach
            if pid in alive and time.time() - ts > TASK_KILL and u_ in pend:   # it: kill the worker
                print("KILL (stuck task)", u_, flush=True)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if time.time() - t_last > 600:
            print("progress", len(out), "of", len(jobs), "done; running", [u for u, _ in cur.values()][:6], flush=True)
            t_last = time.time()
    pool.terminate()
    print("killed (stuck) tasks", killed, flush=True)
    json.dump(out, open(os.path.join(a.out, "lookahead.json"), "w"), indent=1, default=str)
    bad = sorted({r["key"] for r in out if r["signal_lookahead"] or r["engine_lookahead"]})
    print("strategies", len(S), "tasks", len(out), "with look-ahead", len(bad), bad, flush=True)
    print("tasks with errors", [(r["key"], r["tf"], r["errors"], r.get("error")) for r in out if r["errors"]][:60], flush=True)
    print("tasks with fewer than k configurations", [(r["key"], r["tf"], r["configs"]) for r in out if r["configs"] < a.k], flush=True)
    print("last-bar-only differences (end-of-day rules)", sorted({r["key"] for r in out if r["last_bar_only"]}), flush=True)
    print("configurations tested", sum(r["configs"] for r in out), flush=True)


if __name__ == "__main__":
    main()
