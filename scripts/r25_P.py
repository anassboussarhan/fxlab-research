"""Round 25 part P (and CF, ALT) job (docs/round25_preregistration.md): one class group x timeframe x rule chunk,
all of the group's instruments in one job so that panel selection runs here.

Units are (rule, instrument). Each worker evaluates a unit's 128 configurations and makes its per-instrument streams
(default and nested-plus counted; nested23, nbr, share, decay24 as diagnostics). The parent then makes the panel
streams of every rule x class: panel default and panel nested, pooled, counted; per-instrument panel streams as
diagnostics.
With --insts (a part of a group, for the few rule x timeframe cells too large for one job), each unit's aggregates are
saved instead (P_pack_*.npz) and the combine makes the panel streams.
Part ALT (--group ALT): coins from the point-in-time universe file, funding as carry, streams kept on member years.
Usage: python scripts/r25_P.py --group FX --tf H1 --rules A,B|all [--placebos 10] [--cf] --bars DIR --out DIR
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                                    # one thread per process: forked workers
    # otherwise deadlock in OpenMP (the HMM rule's k-means) once the parent has started its thread pool

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
import wf25  # noqa: E402

G = {}
KILL_MARGIN = 900


def load_bars(path, sym, tf):
    b = pd.read_parquet(os.path.join(path, f"{sym}_{tf}.parquet"))
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.DatetimeIndex(b["ts"] if "ts" in b else b.index)
    if b.index.tz is None:
        b.index = b.index.tz_localize("UTC")
    return b[b.index <= C.DEV_END]


def ensure_instrument(sym, b):
    from fxlab.data.instruments import INSTRUMENTS, Instrument, get
    try:
        get(sym)
    except KeyError:                                                       # CFDs and coins: about 1 bp of price
        pip = float(1e-4 * np.median(0.5 * (b.bid_c + b.ask_c)))
        INSTRUMENTS[sym] = Instrument(sym, sym[:-3], sym[-3:], pip, 1.0, "index")


# ------------------------------------------------------------------------------------- part ALT: funding as carry
def install_funding(fund):
    """fund: {sym: Series of funding rates at their timestamps}. Longs pay a positive rate at each stamp."""
    import fxlab.data.rates as RT
    import fxlab.data.rates_oecd as RO
    import fxlab.evaluate as EV
    RT.SWAP_MARKUP_PA = 0.0
    G["fund"] = fund

    def _per_bar(index):
        sym = G.get("cur_sym")
        f = G["fund"].get(sym)
        idx = pd.DatetimeIndex(index).as_unit("ns")
        if f is None or not len(f):
            return np.zeros(len(idx)), np.zeros(len(idx))
        step = pd.Series(idx[: min(len(idx), 200)]).diff().dropna().median() if len(idx) > 2 else pd.Timedelta(hours=1)
        edges = np.r_[idx.asi8, (idx[-1] + step).as_unit("ns").value]
        pos = np.searchsorted(edges, pd.DatetimeIndex(f.index).as_unit("ns").asi8, side="right") - 1
        ok = (pos >= 0) & (pos < len(idx))
        cnt = np.bincount(pos[ok], minlength=len(idx)).astype(float)
        tot = np.bincount(pos[ok], weights=f.to_numpy(float)[ok], minlength=len(idx))
        return cnt, tot

    def rollover_counts(index):
        return _per_bar(index)[0]

    def has_rates(symbol):
        return symbol in G["fund"]

    def rate_diff(symbol, index):
        cnt, tot = _per_bar(index)
        cr = np.where(cnt > 0, -tot / np.maximum(cnt, 1) * 360.0, 0.0)    # fin = cnt/360 x (-held x cr) = held x sum
        return pd.Series(cr, index=index)

    EV.rollover_counts = rollover_counts
    RO.has_rates = has_rates
    RO.rate_diff = rate_diff


# ---------------------------------------------------------------------------------------------------- worker
def run_unit(args):
    key, sym = args
    warnings.filterwarnings("ignore")
    G["q"].put((key, sym, os.getpid(), time.time()))
    G["cur_sym"] = sym
    bars, day, dd = G["bars"][sym], G["day"][sym], G["dd"][sym]
    tf, n = G["tf"], G["n"]
    fin = G["financing"]
    crypto = G["crypto"].get(sym, False)
    t0 = time.time()
    if key in ("CF", "CF_PLACEBO"):
        import r25_cf
        M, U, meta = r25_cf.cf_matrix(G["all_rules"], bars, sym, tf, n, day, dd, G["unit_cap"], fin, crypto,
                                      placebo=key == "CF_PLACEBO")
    else:
        e, fn, sps = G["rules"].get(key, (None, None, []))
        M, U, meta = C.unit_matrix(key, e, fn, sps, bars, sym, tf, n, day, dd, G["unit_cap"], fin, crypto)
    member = G["member"].get(sym)
    mfun = (lambda ix: np.isin(ix.year, member)) if member is not None else None
    streams, ag, extra = C.unit_select(M, U, dd, mfun)
    base = {"part": G["part"], "family": f"{G['part']}-{key}", "unit": f"{key}|{tf}", "tf": tf, "symbol": sym,
            "klass": G["klass"][sym], "placebo": "PLACEBO" in key, "n_cfg": n, **meta}
    rows = C.rows_of(streams, base)
    keep = {k: v.astype(np.float32) for k, v in streams.items() if k in C.COUNTED}
    ag.rows = None                                                         # not picklable; streams use ag.X
    ag.X = ag.X.astype(np.float32)
    meta["select_seconds"] = round(time.time() - t0 - meta["seconds"], 1)
    return key, sym, rows, keep, ag, meta


# ---------------------------------------------------------------------------------------------------- parent
def panel_rows(key, aggs_by_class, tf, members):
    rows, ser = [], {}
    for klass, items in aggs_by_class.items():
        syms = [s for s, _ in items]
        aggs = [a for _, a in items]
        first = max(2010, min(int(a.dates.year.min()) for a in aggs) + 3)
        kw = {"min_active": 30, "min_valid": 50}
        if members:                                                        # part ALT: point-in-time panel of members
            mem = [list(members.get(sy, [])) for sy in syms]
            pool_d, per_d, _ = wf25.panel_pit(aggs, mem, first, wf25.wf23.DEFAULT, **kw)
            pool_n, ch = wf25.panel_pit_nested(aggs, mem, first, wf25.SETTINGS23, **kw)
        else:
            pool_d, per_d, _ = wf25.panel(aggs, first, wf25.wf23.DEFAULT, **kw)
            pool_n, ch = wf25.panel_nested(aggs, first, wf25.SETTINGS23, **kw)
        base = {"part": G["part"], "family": f"{G['part']}-{key}", "unit": f"{key}|{tf}|{klass}", "tf": tf,
                "symbol": f"PANEL:{klass}", "klass": klass, "placebo": "PLACEBO" in key, "n_cfg": G["n"],
                "n_inst": len(aggs)}
        for kind, s in (("panel_default", pool_d), ("panel_nested", pool_n)):
            st = wf25.stats(s)
            rows.append({**base, "selection": kind, "counted_kind": True, **st,
                         "own": wf25.counts_rule(st) if len(s) else False})
            ser[f"{key}|{tf}|{klass}|{kind}"] = s.astype(np.float32)
        for i, s in per_d.items():                                         # diagnostics
            st = wf25.stats(s)
            rows.append({**base, "symbol": syms[i], "selection": "panel_inst", "counted_kind": False, **st,
                         "own": wf25.counts_rule(st) if len(s) else False})
    return rows, ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--rules", default="all")
    ap.add_argument("--insts", default="", help="a part of the group (split cells): comma-separated symbols")
    ap.add_argument("--placebos", type=int, default=0)
    ap.add_argument("--cf", action="store_true")
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--budget", type=float, default=19500)
    ap.add_argument("--unit-cap", type=float, default=3600)
    ap.add_argument("--financing", default="rates_oecd")
    ap.add_argument("--groups", default="ci/r25_groups.json")
    ap.add_argument("--universe", default="", help="part ALT: universe file (members by year, funding dir)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_job = time.time()
    tag = a.tag or f"{a.group}-{a.tf}"
    GR = json.load(open(a.groups))
    G["part"] = "ALT" if a.group == "ALT" else "P"
    G["member"], G["crypto"] = {}, {}
    if a.group == "ALT":
        UNI = json.load(open(a.universe))
        syms = sorted(UNI["members"])
        G["klass"] = {s: "ALT" for s in syms}
        G["member"] = {s: list(map(int, ys)) for s, ys in UNI["members"].items()}
        G["crypto"] = {s: True for s in syms}
        fund = {}
        for s in syms:
            f = os.path.join(UNI.get("funding_dir", a.bars), f"funding_{s}.csv")
            if os.path.exists(f):
                F = pd.read_csv(f, parse_dates=["ts"])
                fund[s] = pd.Series(F["rate"].to_numpy(float), index=pd.DatetimeIndex(F["ts"], tz="UTC")
                                    if F["ts"].dt.tz is None else pd.DatetimeIndex(F["ts"])).sort_index()
        install_funding(fund)
        G["financing"] = "rates_oecd"
        print("ALT coins", len(syms), "with funding", len(fund), flush=True)
    else:
        syms = list(GR["groups"][a.group])
        G["klass"] = {s: GR["class"][s] for s in syms}
        G["crypto"] = {s: GR["class"][s] == "crypto" for s in syms}
        G["financing"] = a.financing
    full = not a.insts
    if a.insts:
        syms = [s for s in a.insts.split(",") if s]
    G["tf"], G["n"], G["unit_cap"] = a.tf, a.n, a.unit_cap
    G["bars"], G["day"], G["dd"] = {}, {}, {}
    missing = []
    for s in syms:
        try:
            b = load_bars(a.bars, s, a.tf)
        except (FileNotFoundError, OSError) as ex:
            missing.append(s)
            print("NO BARS", s, a.tf, repr(ex)[:100], flush=True)
            continue
        ensure_instrument(s, b)
        G["bars"][s] = b
        G["day"][s] = I17.trading_day(b.index)
        G["dd"][s] = pd.Index(sorted(set(G["day"][s])))
        print(s, a.tf, "bars", len(b), b.index.min(), b.index.max(), flush=True)
    syms = [s for s in syms if s in G["bars"]]
    if G["financing"] == "rates_oecd" and a.group != "ALT":
        from fxlab.data import rates_oecd
        print("carry:", {s: rates_oecd.has_rates(s) for s in syms}, flush=True)
    allr = C.prepare_rules(a.tf)
    G["all_rules"] = allr
    want = None if a.rules == "all" else set(a.rules.split(","))
    G["rules"] = {k: (e, fn, sps) for k, e, fn, sps in allr if want is None or k in want}
    keys = [k for k, _, _, _ in allr if k in G["rules"]]
    if want is not None:                                                   # the matrix lists rules longest first
        order = {k: i for i, k in enumerate(a.rules.split(","))}
        keys.sort(key=lambda k: order.get(k, 1e9))
    keys += [f"PLACEBO_{i:02d}" for i in range(a.placebos)]
    if a.cf:
        keys += ["CF", "CF_PLACEBO"]
    units = [(k, s) for k in keys for s in syms]
    print("job", tag, "units", len(units), "keys", len(keys), "instruments", len(syms), "full group", full, flush=True)
    rows, series, done, meta_u = [], {}, set(), {}
    aggs = {}

    def write(final):
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"P_rows_{tag}.csv"), index=False)
        if series:
            pd.DataFrame(series).to_parquet(os.path.join(a.out, f"P_series_{tag}.parquet"))
        C.write_json(os.path.join(a.out, f"P_meta_{tag}.json"),
                     {"group": a.group, "tf": a.tf, "tag": tag, "units": len(units), "done": len(done), "final": final,
                      "full_group": full, "missing_bars": missing, "seconds": round(time.time() - t_job),
                      "lost": G.get("lost", []), "incomplete_keys": G.get("incomplete", []), "unit_meta": meta_u})

    broken = set()

    def finish_key(k):
        """All instruments of key k are in: panel streams (full group) or packs (part of a group). A key with a failed
        or lost unit gets no panel streams (reported in the meta)."""
        items = aggs.pop(k, {})
        if not items or (k in broken and full):
            return
        if full:
            by = {}
            for s, ag in items.items():
                by.setdefault(G["klass"][s], []).append((s, ag))
            pr, ps = panel_rows(k, by, a.tf, G["member"])
            rows.extend(pr)
            series.update(ps)
        else:
            for s, ag in items.items():
                np.savez_compressed(os.path.join(a.out, f"P_pack_{k}_{a.tf}_{s}.npz"), X=ag.X, S=ag.S, Q=ag.Q, A=ag.A,
                                    V=ag.V, dates=ag.dates.as_unit("ns").asi8, nbr=ag.nbr if ag.nbr is not None else np.zeros(0))

    remaining = {k: set(syms) for k in keys}
    ctx = mp.get_context("fork")
    G["q"] = ctx.Queue()
    pool = ctx.Pool(a.procs, maxtasksperchild=10)
    pend = {u: pool.apply_async(run_unit, (u,)) for u in units}
    started, lost, sus = {}, [], {}
    deadline, t_ck = t_job + a.budget, time.time()
    while pend and time.time() < deadline:
        time.sleep(2)
        while True:
            try:
                k_, s_, pid, ts = G["q"].get_nowait()
                started[(k_, s_)] = (pid, ts)
            except Exception:                                              # noqa: BLE001
                break
        alive = {w.pid for w in pool._pool if w.is_alive()}
        cur = {}
        for u_, (pid, ts) in started.items():
            if u_ in pend and (pid not in cur or ts > cur[pid][1]):
                cur[pid] = (u_, ts)
        for pid, (u_, ts) in cur.items():
            if pid in alive and time.time() - ts > a.unit_cap + KILL_MARGIN and not pend[u_].ready():
                print("KILL (stuck unit)", u_, flush=True)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        for u_, r_ in list(pend.items()):
            if r_.ready():
                del pend[u_]
                try:
                    k, s, rr, keep, ag, meta = r_.get()
                except Exception as ex:                                    # noqa: BLE001
                    print("UNIT ERROR", u_, repr(ex)[:300], flush=True)
                    broken.add(u_[0])
                    remaining[u_[0]].discard(u_[1])
                    if not remaining[u_[0]]:
                        finish_key(u_[0])
                    continue
                rows.extend(rr)
                for kind, sr in keep.items():
                    st = [r for r in rr if r["selection"] == kind]
                    if st and st[0]["own"]:
                        series[f"{k}|{a.tf}|{s}|{kind}"] = sr
                aggs.setdefault(k, {})[s] = ag
                meta_u[f"{k}|{s}"] = meta
                done.add(u_)
                remaining[k].discard(s)
                if not remaining[k]:
                    finish_key(k)
            elif u_ in started and started[u_][0] not in alive:
                sus.setdefault(u_, time.time())
                if time.time() - sus[u_] > 30:
                    del pend[u_]
                    lost.append(f"{u_[0]}|{u_[1]}")
                    print("LOST (worker died)", u_, flush=True)
                    broken.add(u_[0])
                    remaining[u_[0]].discard(u_[1])
                    if not remaining[u_[0]]:
                        finish_key(u_[0])
        if not started and not done and time.time() - t_job > 1200:
            print("NO UNIT STARTED in 1200 s", flush=True)
            pool.terminate()
            write(True)
            sys.exit(3)
        if time.time() - t_ck > 900:
            write(False)
            t_ck = time.time()
            print("progress", len(done), "of", len(units), "seconds", round(time.time() - t_job), flush=True)
    pool.terminate()
    G["lost"] = lost + [f"{k}|{s}" for k, s in pend]                       # units not finished in the budget
    incomplete = sorted({k for k, s in pend} | {x.split("|")[0] for x in lost} | broken)
    G["incomplete"] = incomplete
    for k in list(aggs):                                                   # keys with an unfinished instrument: no panel
        if k in incomplete:
            aggs.pop(k)
    write(True)
    print("job", tag, "done", len(done), "of", len(units), "incomplete keys", incomplete[:20],
          "seconds", round(time.time() - t_job), flush=True)


if __name__ == "__main__":
    main()
