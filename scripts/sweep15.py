"""Round 15 parts A-B (docs/round15_preregistration.md): every single-instrument rule (catalog, book systems, round-12
state systems) at D1, W1 and MN, over its full parameter grid and in both directions, plus random-signal placebos;
part C: every feature column as a tercile rule at D1, W1, MN. Priced with the round-13 engine (long_lab: M1 fills
at 08:00 London, stops on the M1 path, swaps). Output per instrument: monthly P&L and trade counts per
(unit, configuration, direction). Usage: python scripts/sweep15.py --symbol EURUSD --out DIR [--parts AB,C]"""
from __future__ import annotations

import argparse
import inspect
import itertools
import json
import os
import signal
import sys
import time
import traceback
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import long_lab as LL                                                       # noqa: E402

EXCL_FAM = {"machine_learning", "retail_mm"}
TIME_OF_DAY = {"S.BO.OPENING_RANGE", "S.SES.ASIAN_RANGE", "S.SES.LONDON_BREAKOUT", "S.SES.NY_OPEN_REVERSAL",
               "S.SES.OVERLAP_MOMENTUM", "S.SES.SESSION_MOMENTUM", "S.SEAS.FIX_REVERSAL", "S.SEAS.GOTOBI",
               "S.SEAS.MONTH_END_FIX", "S.SEAS.INTRADAY", "S.FORUM.NEWS_STRADDLE", "S.SMC.SILVER_BULLET",
               "S.ST.PIVOT_POINTS", "S.FAMOUS.CRABEL_NR", "S.FAMOUS.WILLIAMS_VBO", "S.BO.VOL_BREAKOUT"}
D1_ONLY = {"S.SEAS.WEEKDAY"}
CL_STATE = ("CL3_ABERRATION", "CL3_KING_KELTNER", "CL4_THERMOSTAT", "CL4_BOLLINGER_BANDIT")
STEP = {"D1": 1, "W1": 5, "MN": 21}
N_PLACEBO = 20
PERSIST = (0.8, 0.9, 0.95, 0.98)
LOG = {}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# ------------------------------------------------------------------------------------------------ bars per timeframe
def period_bars(M: LL.Mkt, kind: str):
    """W1 (Monday-Friday weeks) or MN (calendar months by trading date) bars from the D1 bars, with, for each D1 bar,
    the index of the last completed period (-1 before) and the D1 bars that end a period."""
    d1 = M.d1
    if kind == "W1":
        iso = M.tday.isocalendar()
        kv = (iso.year.to_numpy() * 100 + iso.week.to_numpy())
    else:
        kv = M.tday.year.to_numpy() * 100 + M.tday.month.to_numpy()
    agg = {"bid_o": "first", "bid_h": "max", "bid_l": "min", "bid_c": "last", "ask_o": "first", "ask_h": "max",
           "ask_l": "min", "ask_c": "last", "n_ticks": "sum", "spread_mean": "mean", "rv_intrabar": "sum", "n_m1": "sum"}
    agg = {c: f for c, f in agg.items() if c in d1.columns}
    for c in d1.columns:
        if c not in agg and c not in {"open", "high", "low", "close", "spread", "volume"} and pd.api.types.is_numeric_dtype(d1[c]):
            agg[c] = "sum"
    b = d1.groupby(kv, sort=True).agg(agg)
    starts = np.r_[0, np.flatnonzero(kv[1:] != kv[:-1]) + 1]
    assert len(starts) == len(b) and (np.diff(kv[starts]) > 0).all()
    b.index = d1.index[starts]
    from fxlab.data.store import add_mid
    b = add_mid(b)
    pend = np.r_[kv[1:] != kv[:-1], True]
    cw = np.cumsum(pend)
    last_done = np.r_[-1, cw[:-1] - 1].astype(np.int64)
    return b, last_done, pend


def grids(M: LL.Mkt):
    from fxlab.features._util import atr
    d1 = M.d1
    nb = len(d1)
    G = {"D1": dict(bars=d1, N=LL.known(LL.turtle_n(d1)), atr=LL.known(atr(d1, 14)), upd=np.ones(nb, bool),
                    map=None, fresh=None)}
    for tf in ("W1", "MN"):
        b, ld, pend = period_bars(M, tf)
        fresh = np.r_[False, ld[1:] != ld[:-1]]
        G[tf] = dict(bars=b, N=LL.map_weekly(LL.turtle_n(b).to_numpy(), ld), atr=LL.map_weekly(atr(b, 14).to_numpy(), ld),
                     upd=pend, map=ld, fresh=fresh)
    return G


def to_d1(x: np.ndarray, G: dict, fresh_only: bool):
    """A per-period-bar series (known at the period's close) as a per-D1-bar array usable during each D1 bar."""
    x = np.asarray(x, float)
    if G["map"] is None:
        return np.r_[np.nan, x[:-1]]
    y = LL.map_weekly(x, G["map"])
    if fresh_only:
        y[~G["fresh"]] = np.nan
    return y


# ------------------------------------------------------------------------------------------------ rule list
def rule_list():
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    R = []
    for e in sorted((e for e in CATALOG.values() if e.kind == "strategy"), key=lambda e: e.id):
        if e.family in EXCL_FAM or e.multi_asset or e.id in TIME_OF_DAY:
            continue
        R.append(("cat", e.id, e))
    import anatomy_systems as AS
    for k in CL_STATE:
        R.append(("cl", k, AS.SYSTEMS[k][0]))
    return R


def configs(e):
    sp = dict(e.space or {})
    keys = sorted(sp)
    if not keys:
        return [dict(e.params)]
    out = []
    for vals in itertools.product(*[sp[k] for k in keys]):
        p = dict(e.params)
        p.update(dict(zip(keys, vals)))
        out.append(p)
    return out


def cfg_name(p: dict):
    return ",".join(f"{k}={p[k]}" for k in sorted(p)) or "default"


# ------------------------------------------------------------------------------------------------ pricing one signal
def price(M, G, tf, sd, sig, mode, risk):
    """Run the engine for one side series (per period bar) in its own exit mode; returns monthly frame."""
    nb = len(M.d1)
    if mode == "state":
        kw = dict(mode=0, tgt=to_d1(sd, G, False), stop_mult=0.0)
    else:
        prev = np.r_[0.0, sd[:-1]]
        fr = np.where((sd != 0) & (sd != prev), sd, 0.0)
        ent = to_d1(fr, G, True)
        sl_bar = sig["stop_dist"].to_numpy(float) if (sig is not None and "stop_dist" in sig) else np.full(len(sd), np.nan)
        sl = to_d1(sl_bar, G, False)
        mult = (risk.get("stop_params") or {}).get("mult", 2.0)
        sl = np.where(np.isfinite(sl) & (sl > 0), sl, mult * np.asarray(G["atr"]))
        tpp = risk.get("target_params") or {"a": 0.0, "b": 2.0}
        if sig is not None and "tp_dist" in sig:
            tp = to_d1(sig["tp_dist"].to_numpy(float), G, False)
            tp = np.where(np.isfinite(tp) & (tp > 0), tp, tpp.get("a", 0.0) + tpp.get("b", 2.0) * sl)
        elif risk.get("target", "affine") == "none":
            tp = np.full(nb, np.nan)
        else:
            tp = tpp.get("a", 0.0) + tpp.get("b", 2.0) * sl
        hold = float(risk.get("max_hold", 48)) * STEP[tf]
        kw = dict(ent=ent, sld=sl, tpd=tp, hld=np.full(nb, hold), stop_mult=2.0)
    kw["upd"] = G["upd"]
    pnl, gp, sw, T = LL.run_ev(M, G["N"], **kw)
    return LL.to_monthly(M, pnl, gp, sw, T)[["pnl", "gpnl", "n"]]


def run_rules(M, G, sym, t_job, budget, only=None):
    rows = []
    for src, rid, obj in rule_list():
        if only and rid not in only:
            continue
        for tf in ("D1", "W1", "MN"):
            if rid in D1_ONLY and tf != "D1":
                continue
            key = f"{rid}|{tf}"
            if time.time() - t_job > budget:
                LOG[key] = {"error": "budget"}
                continue
            t0 = time.time()
            try:
                signal.alarm(600)
                bars = G[tf]["bars"]
                if src == "cl":
                    plist = [{}]
                    mode, risk = "state", {}
                else:
                    plist = configs(obj)
                    mode = obj.extra.get("mode", "state")
                    risk = obj.extra.get("risk") or {}
                nconf = 0
                for p in plist:
                    try:
                        if src == "cl":
                            kind, side = obj(bars.copy(), tf)
                            sig = None
                        else:
                            kw = dict(p)
                            if "symbol" in inspect.signature(obj.fn).parameters:
                                kw["symbol"] = sym
                            sig = obj.fn(bars.copy(), **kw)
                            side = sig["side"]
                        sd0 = np.sign(np.nan_to_num(np.asarray(side, float)))
                    except Timeout:
                        raise
                    except Exception as ex:                                # noqa: BLE001
                        LOG.setdefault(key, {})["cfg_error"] = repr(ex)[:160]
                        continue
                    for d in (1, -1):
                        g = price(M, G[tf], tf, sd0 * d, sig, mode, risk)
                        g["unit"], g["cfg"], g["dir"] = key, cfg_name(p), d
                        rows.append(g.reset_index())
                        nconf += 1
                signal.alarm(0)
                LOG[key] = {**LOG.get(key, {}), "s": round(time.time() - t0, 1), "configs": nconf}
            except Timeout:
                LOG[key] = {"error": "timeout"}
            except Exception as ex:                                            # noqa: BLE001
                signal.alarm(0)
                LOG[key] = {"error": repr(ex)[:200]}
                traceback.print_exc(limit=1)
    # placebos: random long / flat / short states with persistence p
    for tf in ("D1", "W1", "MN"):
        nbars = len(G[tf]["bars"])
        for k in range(N_PLACEBO):
            for p in PERSIST:
                rng = np.random.default_rng(zlib.crc32(f"P{k}|{p}|{sym}|{tf}".encode()))
                sw = rng.random(nbars) > p
                draw = rng.integers(-1, 2, nbars).astype(float)
                sd = np.zeros(nbars)
                s = 0.0
                for i in range(nbars):
                    if sw[i]:
                        s = draw[i]
                    sd[i] = s
                for d in (1, -1):
                    g = price(M, G[tf], tf, sd * d, None, "state", {})
                    g["unit"], g["cfg"], g["dir"] = f"PLACEBO_{k:02d}|{tf}", f"p={p}", d
                    rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ part C: features
def feature_columns():
    import fxlab.features  # noqa: F401
    from fxlab.registry import CATALOG
    return sorted((e for e in CATALOG.values() if e.kind == "feature" and not e.multi_asset and e.family != "learned"),
                  key=lambda e: e.id)


def run_features(M, G, sym, t_job, budget):
    rows = []
    fit_end = pd.Timestamp("2018-12-31 23:59", tz="UTC")
    c = M.d1["close"]
    vol = np.sqrt((np.log(c / c.shift(1)) ** 2).ewm(span=36, min_periods=36).mean() * 256)
    size = (0.20 / vol).shift(1).to_numpy()                                  # known before the day's fill
    nb = len(M.d1)
    zero = np.zeros(nb)

    def trade(vals, tf, name):
        bars = G[tf]["bars"]
        v = pd.Series(np.asarray(vals, float), index=bars.index)
        f = v[(v.index <= fit_end) & np.isfinite(v)]
        if len(f) < 30 or f.nunique() < 3:
            return
        lo, hi = f.quantile(1 / 3), f.quantile(2 / 3)
        if not hi > lo:
            return
        pos = np.where(v > hi, 1.0, np.where(v < lo, -1.0, 0.0))
        pos[~np.isfinite(v.to_numpy())] = np.nan
        p_d1 = to_d1(pos, G[tf], False)
        for d in (1, -1):
            ptg = p_d1 * d * size
            pnl, gp, sw, _ = LL.cont_run(M, ptg, zero)
            act = (np.nan_to_num(ptg) != 0).astype(np.int64)                # days with a position
            g = pd.DataFrame({"pnl": pnl, "gpnl": gp, "n": act, "month": M.tday.to_period("M").astype(str)}).groupby("month").sum()
            g["unit"], g["cfg"], g["dir"] = f"{name}|{tf}", "terciles", d
            rows.append(g.reset_index())

    fl = feature_columns()
    if os.environ.get("SWEEP15_MAXFEAT"):
        fl = fl[: int(os.environ["SWEEP15_MAXFEAT"])]
    for e in fl:
        for tf in ("D1", "W1", "MN"):
            key = f"{e.id}|{tf}"
            if time.time() - t_job > budget:
                LOG[key] = {"error": "budget"}
                continue
            t0 = time.time()
            try:
                signal.alarm(300)
                out = e.fn(G[tf]["bars"].copy(), **e.params)
                cols = {e.id: out} if isinstance(out, pd.Series) else {f"{e.id}:{k}": out[k] for k in out.columns}
                for name, s in cols.items():
                    if pd.api.types.is_numeric_dtype(s):
                        trade(s.to_numpy(float), tf, "FEAT:" + name)
                signal.alarm(0)
                LOG[key] = {"s": round(time.time() - t0, 1), "cols": len(cols)}
            except Timeout:
                LOG[key] = {"error": "timeout"}
            except Exception as ex:                                            # noqa: BLE001
                signal.alarm(0)
                LOG[key] = {"error": repr(ex)[:200]}
    # noise placebos: random walks and white noise
    for tf in ("D1", "W1", "MN"):
        nbars = len(G[tf]["bars"])
        for k in range(30):
            rng = np.random.default_rng(zlib.crc32(f"N{k}|{sym}|{tf}".encode()))
            x = rng.standard_normal(nbars)
            vals = np.cumsum(x) if k % 2 == 0 else x
            trade(vals, tf, f"PLACEBOF_{k:02d}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="AB,C")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_job = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    if os.environ.get("SWEEP15_SYNTH"):
        import pickle
        m1, pip = pickle.load(open(os.environ["SWEEP15_SYNTH"], "rb"))
        M = LL.Mkt(m1, a.symbol, pip)
    else:
        import edges_lab2 as E2
        bk = E2.book(a.symbol)
        if bk is None:
            print("no data", a.symbol)
            return
        M = LL.Mkt(bk.m, a.symbol, bk.pip)
    G = grids(M)
    print(a.symbol, "D1", len(M.d1), "W1", len(G["W1"]["bars"]), "MN", len(G["MN"]["bars"]), flush=True)
    parts = set(a.parts.split(","))
    only = set(a.only.split(",")) if a.only else None
    for part, fn, budget in (("AB", run_rules, 9000), ("C", run_features, 15500)):
        if part not in parts:
            continue
        df = fn(M, G, a.symbol, t_job, budget) if part == "C" else fn(M, G, a.symbol, t_job, budget, only)
        if len(df):
            df["symbol"] = a.symbol
            for col in ("pnl", "gpnl"):
                df[col] = df[col].astype(np.float32)
            df["n"] = df["n"].astype(np.int16)
            df["dir"] = df["dir"].astype(np.int8)
            for col in ("unit", "cfg", "symbol", "month"):
                df[col] = df[col].astype("category")
            df.to_parquet(os.path.join(a.out, f"S{part}_{a.symbol}.parquet"), index=False)
        print(a.symbol, part, "rows", len(df), round(time.time() - t_job), "s", flush=True)
    json.dump(LOG, open(os.path.join(a.out, f"log15_{a.symbol}.json"), "w"), indent=0, default=str)
    print("done", a.symbol, round(time.time() - t_job), "s", flush=True)


if __name__ == "__main__":
    main()
