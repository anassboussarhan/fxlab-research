"""Round 25 look-ahead gate (docs/round25_preregistration.md, sanity checks): the truncation test of the complete
round-25 signal path.

For each rule (the round-25 rule set of the timeframe) the first `--per-rule` configurations of its part-P Sobol design
are taken (alternately with the crypto weekday masks, as part ALT draws them). The executed target (side and size)
on a synthetic random walk is compared with the target computed on the walk truncated at each of `--cuts` points: it
must be identical on every bar before the last bar of the truncated sample. Differences on that last bar alone are
counted apart (rules that act on a bar's own close), as in scripts/r23_lookahead.py.
The path covers the rule with its categorical swaps, the neighbourhood sizing, the confirmation, the gate (session,
weekday, regime, higher timeframe) and the volatility sizing. With --cf, part CF's confluence side and vote share are
tested the same way (members on the truncated walk).
Usage: python scripts/r25_lookahead.py --tf H1 [--per-rule 2] [--cuts 48] [--rules A,B] [--cf] --out DIR
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
import r25_common as C  # noqa: E402
import r25_exec as X  # noqa: E402
import r23_LW as LW  # noqa: E402

YEARS = {"H1": 5, "H4": 10, "D1": 18}
G = {}
TASK_KILL = 1500


def synth(tf, seed=11):
    """Hourly random walk (weekdays), resampled to tf as the store's bars are."""
    from fxlab.data.store import resample
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2006-01-02", periods=int(YEARS[tf] * 365.25 * 24), freq="h", tz="UTC")
    ts = ts[ts.dayofweek < 5]
    p = 1.2 * np.exp(np.cumsum(rng.normal(0, 0.0012, len(ts))))
    o = np.r_[p[0], p[:-1]]
    w = np.abs(rng.normal(0, 0.0006, len(ts)))
    h, l = np.maximum(o, p) * (1 + w), np.minimum(o, p) * (1 - w)
    sp = 1.5e-5
    m = pd.DataFrame({"bid_o": o, "bid_h": h, "bid_l": l, "bid_c": p, "ask_o": o + sp, "ask_h": h + sp,
                      "ask_l": l + sp, "ask_c": p + sp, "n_ticks": 10.0, "spread_mean": sp}, index=ts)
    m.index.name = "ts"
    if tf == "H1":
        return resample(m, "H1")
    return resample(m, tf)


def compare(full, part, c):
    """(mismatch before the last bar, mismatch on the last bar only) between two (side, size) targets."""
    bad_pre = bad_last = False
    for a, b in zip(full, part):
        a, b = np.asarray(a, float)[:c], np.asarray(b, float)
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        if not same[: c - 1].all():
            bad_pre = True
        elif not same[c - 1]:
            bad_last = True
    return bad_pre, bad_last


def target(key, e, fn, sps, u, bars, crypto):
    params = {s["name"]: LW.draw(s, u[j]) for j, s in enumerate(sps)}
    cfg = X.config(u[len(sps):], G["tf"], crypto)
    seed = C.seed_of(key, G["tf"]) + 7919
    return X.evaluate25(e, fn, bars, "EURUSD", params, cfg, sps=sps, u_params=u[:len(sps)], seed=seed,
                        return_target=True), cfg


def run_task(args):
    key, j = args
    warnings.filterwarnings("ignore")
    e, fn, sps = G["rules"][key]
    bars = G["bars"]
    n = len(bars)
    U = C.design(key, G["tf"], len(sps), max(G["per_rule"], 2))
    u = U[j]
    crypto = bool(j % 2)
    t0 = time.time()
    try:
        full, cfg = target(key, e, fn, sps, u, bars, crypto)
    except Exception as ex:                                                # noqa: BLE001
        return {"key": key, "j": j, "error": repr(ex)[:200]}
    pre = last = 0
    first_bad = None
    for c in G["cuts"]:
        try:
            part, _ = target(key, e, fn, sps, u, bars.iloc[:c], crypto)
        except Exception as ex:                                            # noqa: BLE001
            return {"key": key, "j": j, "error": f"cut {c}: {ex!r}"[:200]}
        bp, bl = compare(full, part, c)
        pre += bp
        last += bl
        if bp and first_bad is None:
            first_bad = int(c)
    act = {k: (v is not None and v is not False and v != 0) for k, v in cfg.items()
           if k in ("conf", "sess", "day", "reg", "htf", "size", "vt")}
    act.update({"psrc": cfg["psrc"], "ma": cfg["ma"], "vol": cfg["vol"][0], "mode": cfg["mode"], "otype": cfg["otype"]})
    return {"key": key, "j": j, "n": n, "cuts": len(G["cuts"]), "lookahead_cuts": int(pre), "last_bar_only_cuts": int(last),
            "first_bad_cut": first_bad, "active": float(np.mean(np.asarray(full[0]) != 0)), "cfg": act,
            "seconds": round(time.time() - t0, 1)}


def run_cf(j):
    import r25_cf as CF
    warnings.filterwarnings("ignore")
    bars = G["bars"]
    U = C.design("CF", G["tf"], CF.CF_SPECS, 16)
    u = U[j]
    c_ = CF.cf_config(u[:CF.CF_SPECS])

    def side_of(b):
        day = I17.trading_day(b.index)
        dd = pd.Index(sorted(set(day)))
        mem = G["mem_cache"].get(len(b))
        if mem is None:
            mem = CF.members(G["allr"], b, "EURUSD", day, dd)
            G["mem_cache"][len(b)] = mem
        years = pd.DatetimeIndex(pd.to_datetime(day)).year.to_numpy()
        return CF.cf_side(mem, b, years, pd.DatetimeIndex(pd.to_datetime(dd)), c_)

    full = side_of(bars)
    pre = last = 0
    for c in G["cf_cuts"]:
        bp, bl = compare(full, side_of(bars.iloc[:c]), c)
        pre += bp
        last += bl
    return {"key": "CF", "j": j, "cuts": len(G["cf_cuts"]), "lookahead_cuts": int(pre), "last_bar_only_cuts": int(last),
            "cfg": {k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in c_.items()},
            "active": float(np.mean(full[0] != 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True, choices=list(YEARS))
    ap.add_argument("--per-rule", type=int, default=2)
    ap.add_argument("--cuts", type=int, default=48)
    ap.add_argument("--rules", default="")
    ap.add_argument("--cf", action="store_true")
    ap.add_argument("--cf-configs", type=int, default=16)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")
    G["tf"], G["per_rule"] = a.tf, a.per_rule
    bars = synth(a.tf)
    G["bars"] = bars
    n = len(bars)
    G["cuts"] = sorted({int(x) for x in np.linspace(int(0.4 * n), n - 1, a.cuts)})
    allr = C.prepare_rules(a.tf, set(a.rules.split(",")) if a.rules else None)
    G["allr"] = allr
    G["rules"] = {k: (e, fn, sps) for k, e, fn, sps in allr}
    print("tf", a.tf, "bars", n, "rules", len(allr), "cuts", len(G["cuts"]), flush=True)
    out = []
    if a.cf:
        G["mem_cache"] = {}
        G["cf_cuts"] = sorted({int(x) for x in np.linspace(int(0.5 * n), n - 1, min(a.cuts, 12))})
        for j in range(a.cf_configs):
            r = run_cf(j)
            out.append(r)
            print(json.dumps(r, default=str), flush=True)
    tasks = [(k, j) for k in G["rules"] for j in range(a.per_rule)]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs, maxtasksperchild=4) as pool:
        pend = {t: pool.apply_async(run_task, (t,)) for t in tasks}
        t0 = {t: time.time() for t in tasks}
        while pend:
            time.sleep(2)
            for t, r in list(pend.items()):
                if r.ready():
                    del pend[t]
                    try:
                        res = r.get()
                    except Exception as ex:                                # noqa: BLE001
                        res = {"key": t[0], "j": t[1], "error": repr(ex)[:200]}
                    out.append(res)
                    print(json.dumps(res, default=str), flush=True)
            if pend and all(time.time() - t0[t] > TASK_KILL for t in pend) and len(pend) <= a.procs:
                for t in pend:
                    out.append({"key": t[0], "j": t[1], "error": "killed after TASK_KILL seconds"})
                print("KILL remaining tasks", list(pend)[:8], flush=True)
                pool.terminate()
                break
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(a.out, f"r25_lookahead_{a.tf}{'_cf' if a.cf else ''}.csv"), index=False)
    ok = R[~R.get("error", pd.Series(index=R.index, dtype=object)).notna()] if "error" in R else R
    summ = {"tf": a.tf, "tasks": int(len(R)), "errors": int(R["error"].notna().sum()) if "error" in R else 0,
            "with_lookahead": int((ok.lookahead_cuts > 0).sum()), "last_bar_only": int(((ok.lookahead_cuts == 0)
                                                                                         & (ok.last_bar_only_cuts > 0)).sum()),
            "lookahead_keys": sorted(set(ok.loc[ok.lookahead_cuts > 0, "key"]))}
    json.dump(summ, open(os.path.join(a.out, f"r25_lookahead_{a.tf}{'_cf' if a.cf else ''}.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
