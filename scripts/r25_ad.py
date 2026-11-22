"""Round 25 part AD, adaptive search (docs/round25_preregistration.md).

Unit: rule x instrument x timeframe (the 20 rules of ci/r25_ad_rules.json; the 28 G10 pairs, XAU, XAG; H1, H4, D1).
Search space: part P's (the rule's parameters and every execution dimension), as the unit cube that part P draws by
Sobol; the mapping to values is part P's (r23_LW.draw, r25_exec.config).

Procedure:
- the first 64 configurations of part P's Sobol design for the rule and timeframe;
- each January Y from the data start + 3 years: a multivariate TPE sampler (Optuna) is fitted on every earlier trial,
  each scored by its t over all days before Y; it proposes 32 new trials, which are evaluated and scored the same way;
- the top 5 trials by that score (old and new) are traded during Y, equal weight.
A trial's returns on a day depend only on data up to that day (the round-25 path passes the truncation test), so
scoring it on the days before Y uses nothing from Y on.
Counted: that stream on 2019-2025 only (P1 chose the rules). Placebo: the same on one random-state unit.
Usage: python scripts/r25_ad.py --tf H1 --rules A,B --insts X,Y --bars DIR --out DIR
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import intraday17 as I17  # noqa: E402
import r23_LW as LW  # noqa: E402
import r25_common as C  # noqa: E402
import r25_exec as X  # noqa: E402
import wf25  # noqa: E402

N0, N_YEAR, TOP = 64, 32, 5
MIN_ACTIVE, MIN_VALID = 30, 50
G = {}
KILL_MARGIN = 900


def tstat(x):
    x = x[np.isfinite(x)]
    if len(x) < MIN_VALID or (x != 0).sum() < MIN_ACTIVE:
        return -np.inf
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else -np.inf


def evaluate(key, e, fn, sps, u, bars, sym, tf, day, dd, i, plc, seed, cache):
    params = {s["name"]: LW.draw(s, u[j]) for j, s in enumerate(sps)}
    cfg = X.config(u[len(sps):], tf, False)
    ps = C.placebo_side(seed, i, params["persist"], len(bars)) if plc else None
    r = X.evaluate25(None if plc else e, None if plc else fn, bars, sym, {} if plc else params, cfg,
                     sps=None if plc else sps, u_params=None if plc else u[:len(sps)], seed=seed + 7919 * i,
                     placebo_side=ps, financing=G["financing"], cache=cache)              # per unit (instrument)
    return I17.daily_sum(r, day, dd).astype(np.float64)


def ad_unit(key, e, fn, sps, bars, sym, tf, day, dd, unit_cap):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings("ignore")
    plc = key.startswith("PLACEBO")
    if plc:
        sps = C.PLACEBO_SPECS
    dim = len(sps) + X.N_EXEC
    seed = C.seed_of(key, tf)
    U0 = C.design(key, tf, len(sps), G.get("n0", N0))
    dist = {f"u{j}": optuna.distributions.FloatDistribution(0.0, 1.0 - 1e-9) for j in range(dim)}
    cal = pd.DatetimeIndex(pd.to_datetime(dd))
    yrs = cal.year.to_numpy()
    first = max(2010, int(yrs.min()) + 3)
    t0 = time.time()
    trials, R = [], []                                                     # unit-cube points, daily returns
    errors = 0
    cache = {}
    signal.signal(signal.SIGALRM, C._alarm)

    def run(u, i):
        nonlocal errors
        try:
            signal.alarm(300)
            r = evaluate(key, e, fn, sps, u, bars, sym, tf, day, dd, i, plc, seed, cache)
            signal.alarm(0)
        except Exception:                                                  # noqa: BLE001  (C.Timeout included)
            signal.alarm(0)
            errors += 1
            r = np.full(len(dd), np.nan)
        trials.append(np.asarray(u, float))
        R.append(r)

    for i, u in enumerate(U0):
        if time.time() - t0 > unit_cap:
            break
        run(u, i)
    out = np.full(len(dd), np.nan)
    picks = {}
    stopped = None
    for Y in range(first, 2026):
        if time.time() - t0 > unit_cap:
            stopped = Y
            break
        tr = yrs < Y
        scores = [tstat(r[tr]) for r in R]
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed + Y, n_startup_trials=0)
        st = optuna.create_study(direction="maximize", sampler=sampler)
        st.add_trials([optuna.trial.create_trial(params={f"u{j}": float(min(max(v, 0.0), 1.0 - 1e-9))
                                                         for j, v in enumerate(u)}, distributions=dist,
                                                 value=float(sc) if np.isfinite(sc) else -50.0)
                       for u, sc in zip(trials, scores)])
        for k in range(G.get("n_year", N_YEAR)):
            tq = st.ask(dist)
            u = np.array([tq.params[f"u{j}"] for j in range(dim)])
            run(u, len(trials))
            sc = tstat(R[-1][tr])
            st.tell(tq, float(sc) if np.isfinite(sc) else -50.0)
            scores.append(sc)
        order = [i for i in np.argsort(-np.asarray(scores), kind="stable") if np.isfinite(scores[i])][:TOP]
        te = yrs == Y
        if order:
            out[te] = np.nanmean(np.vstack([R[i][te] for i in order]), axis=0)
        else:
            out[te] = 0.0
        picks[Y] = [int(i) for i in order]
    s = pd.Series(out, index=cal).dropna()
    meta = {"trials": len(trials), "errors": errors, "seconds": round(time.time() - t0, 1), "stopped_at": stopped,
            "last_picks": {str(k): v for k, v in list(picks.items())[-2:]}}
    return s, meta


def run_unit(args):
    key, sym = args
    warnings.filterwarnings("ignore")
    G["q"].put((key, sym, os.getpid(), time.time()))
    e, fn, sps = G["rules"].get(key, (None, None, []))
    bars, day, dd = G["bars"][sym], G["day"][sym], G["dd"][sym]
    s, meta = ad_unit(key, e, fn, sps, bars, sym, G["tf"], day, dd, G["unit_cap"])
    st = wf25.stats(s)
    row = {"part": "AD", "family": f"AD-{key}", "unit": f"{key}|{G['tf']}", "tf": G["tf"], "symbol": sym,
           "klass": G["klass"].get(sym, ""), "placebo": key.startswith("PLACEBO"), "selection": "ad_top5",
           "counted_kind": True, **{k: v for k, v in meta.items() if k != "last_picks"}, **st,
           "own": wf25.counts_rule(st, use_p1=False) if len(s) else False}
    return key, sym, row, s.astype(np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--insts", required=True)
    ap.add_argument("--placebo", type=int, default=1)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget", type=float, default=19500)
    ap.add_argument("--unit-cap", type=float, default=3600)
    ap.add_argument("--financing", default="rates_oecd")
    ap.add_argument("--groups", default="ci/r25_groups.json")
    ap.add_argument("--n0", type=int, default=N0)
    ap.add_argument("--n-year", type=int, default=N_YEAR)
    a = ap.parse_args()
    G["n0"], G["n_year"] = a.n0, a.n_year
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_job = time.time()
    tag = a.tag or f"AD-{a.tf}"
    GR = json.load(open(a.groups)) if os.path.exists(a.groups) else {"class": {}}
    G["klass"] = GR.get("class", {})
    G["tf"], G["unit_cap"], G["financing"] = a.tf, a.unit_cap, a.financing
    from r25_P import ensure_instrument, load_bars
    G["bars"], G["day"], G["dd"] = {}, {}, {}
    syms = [s for s in a.insts.split(",") if s]
    missing = []
    for s in syms:
        try:
            b = load_bars(a.bars, s, a.tf)
        except (FileNotFoundError, OSError):
            missing.append(s)
            continue
        ensure_instrument(s, b)
        G["bars"][s], G["day"][s] = b, I17.trading_day(b.index)
        G["dd"][s] = pd.Index(sorted(set(G["day"][s])))
    syms = [s for s in syms if s in G["bars"]]
    want = [k for k in a.rules.split(",") if k]
    allr = C.prepare_rules(a.tf, set(want))
    G["rules"] = {k: (e, fn, sps) for k, e, fn, sps in allr}
    keys = [k for k in want if k in G["rules"]] + (["PLACEBO_AD"] if a.placebo else [])
    units = [(k, s) for k in keys for s in syms]
    print("job", tag, "units", len(units), "missing bars", missing, flush=True)
    rows, series, metas = [], {}, {}
    ctx = mp.get_context("fork")
    G["q"] = ctx.Queue()
    pool = ctx.Pool(a.procs, maxtasksperchild=4)
    pend = {u: pool.apply_async(run_unit, (u,)) for u in units}
    started, lost, sus = {}, [], {}
    deadline, t_ck = t_job + a.budget, time.time()

    def write(final):
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"AD_rows_{tag}.csv"), index=False)
        if series:
            pd.DataFrame(series).to_parquet(os.path.join(a.out, f"AD_series_{tag}.parquet"))
        C.write_json(os.path.join(a.out, f"AD_meta_{tag}.json"),
                     {"tf": a.tf, "units": len(units), "done": len(rows), "final": final, "missing_bars": missing,
                      "lost": lost + ([f"{k}|{s}" for k, s in pend] if final else []), "unit_meta": metas,
                      "seconds": round(time.time() - t_job)})

    while pend and time.time() < deadline:
        time.sleep(2)
        while True:
            try:
                k_, s_, pid, ts = G["q"].get_nowait()
                started[(k_, s_)] = (pid, ts)
            except Exception:                                              # noqa: BLE001
                break
        alive = {w.pid for w in pool._pool if w.is_alive()}
        for u_, (pid, ts) in list(started.items()):
            if u_ in pend and pid in alive and time.time() - ts > a.unit_cap + KILL_MARGIN and not pend[u_].ready():
                print("KILL (stuck unit)", u_, flush=True)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        for u_, r_ in list(pend.items()):
            if r_.ready():
                del pend[u_]
                try:
                    k, s, row, ser, meta = r_.get()
                except Exception as ex:                                    # noqa: BLE001
                    print("UNIT ERROR", u_, repr(ex)[:300], flush=True)
                    lost.append(f"{u_[0]}|{u_[1]}")
                    continue
                rows.append(row)
                metas[f"{k}|{s}"] = meta
                if row["own"]:
                    series[f"{k}|{a.tf}|{s}|ad_top5"] = ser
            elif u_ in started and started[u_][0] not in alive:
                sus.setdefault(u_, time.time())
                if time.time() - sus[u_] > 30:
                    del pend[u_]
                    lost.append(f"{u_[0]}|{u_[1]}")
                    print("LOST (worker died)", u_, flush=True)
        if time.time() - t_ck > 900:
            write(False)
            t_ck = time.time()
            print("progress", len(rows), "of", len(units), round(time.time() - t_job), "s", flush=True)
    pool.terminate()
    write(True)
    print("job", tag, "done", len(rows), "of", len(units), "lost", len(lost) + len(pend), flush=True)


if __name__ == "__main__":
    main()
