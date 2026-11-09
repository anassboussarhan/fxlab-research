"""Round 17 intraday library (docs/round17_preregistration.md, section 1): every single-instrument catalog strategy
outside the ML and money-management families, including the time-of-day rules, at H4, H1, M30 and M15, over its
full parameter grid and in both directions, through the project's bar engine (fxlab.evaluate: base cost model,
flat financing, stop first). Random-state placebos per timeframe. P&L is summed by trading day (17:00 New York).
Also the instrument's daily intraday state features from 5-minute returns: RV_5M, JUMP, AC_5M, ASIA_SHARE.

Output: daily17_SYM.npz (unit, cfg, dir, days, pnl[config x day]) and feats17_SYM.parquet.
Usage: python scripts/intraday17.py --symbol EURUSD --out DIR [--tfs H4,H1,M30,M15] [--max-settings N]"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
EXCL_FAM = {"machine_learning", "retail_mm"}
PLACEBO_PERSIST = (0.9, 0.98, 0.995)
N_PLACEBO = 10
LOG = {}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def strategies():
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    return sorted((e for e in CATALOG.values() if e.kind == "strategy" and e.family not in EXCL_FAM and not e.multi_asset),
                  key=lambda e: e.id)


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


def trading_day(index: pd.DatetimeIndex) -> np.ndarray:
    """The trading day of each bar: the day that ends at 17:00 New York (bars opening at or after 17:00 belong to the
    next day)."""
    ny = index.tz_convert("America/New_York").tz_localize(None)
    return np.asarray((ny + pd.Timedelta(hours=7)).floor("D").strftime("%Y-%m-%d"))


def daily_sum(r: pd.Series, day: np.ndarray, days: pd.Index) -> np.ndarray:
    s = pd.Series(np.nan_to_num(np.asarray(r, float)), index=day).groupby(level=0).sum()
    return s.reindex(days).fillna(0.0).to_numpy(np.float32)


def intraday_features(m1: pd.DataFrame) -> pd.DataFrame:
    from fxlab.data.store import resample
    b = resample(m1, "M5")
    mid = (b["bid_c"] + b["ask_c"]) / 2
    r = np.log(mid).diff()
    day = trading_day(b.index)
    same = pd.Series(day, index=b.index).eq(pd.Series(day, index=b.index).shift(1)).to_numpy()
    r = r.where(same)
    df = pd.DataFrame({"r": r.to_numpy(), "day": day, "hour": b.index.hour,
                       "hi": ((b["bid_h"] + b["ask_h"]) / 2).to_numpy(), "lo": ((b["bid_l"] + b["ask_l"]) / 2).to_numpy()})
    df["r1"] = df.groupby("day")["r"].shift(1)
    g = df.groupby("day")
    rv2 = g["r"].apply(lambda x: float(np.nansum(x * x)))
    bv = g.apply(lambda x: float(np.pi / 2 * np.nansum(np.abs(x.r) * np.abs(x.r1))), include_groups=False)
    ac = g.apply(lambda x: x.r.corr(x.r1) if x.r.notna().sum() > 20 else np.nan, include_groups=False)
    rng = g["hi"].max() - g["lo"].min()
    asia = df[df.hour < 7].groupby("day")
    arng = asia["hi"].max() - asia["lo"].min()
    out = pd.DataFrame({"RV_5M": np.sqrt(rv2 * 252), "JUMP": (1 - bv / rv2).clip(lower=0, upper=1),
                        "AC_5M": ac, "ASIA_SHARE": (arng.reindex(rng.index) / rng).clip(0, 1)})
    out.index.name = "day"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tfs", default="H4,H1,M30,M15")
    ap.add_argument("--max-settings", type=int, default=0, help="test runs only: cap the settings per strategy")
    ap.add_argument("--budget", type=float, default=17000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_job = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    from fxlab.data.instruments import get as instrument
    from fxlab.data.store import load_m1, resample
    from fxlab.evaluate import evaluate, run_signals
    from fxlab.execution.costs import BASE
    from fxlab.execution.simulator import rollover_counts, vector_backtest
    sym = a.symbol
    pip = instrument(sym).pip
    m1 = load_m1(sym)
    m1 = m1[m1.index <= DEV_END]
    print(sym, "M1 rows", len(m1), m1.index[0], m1.index[-1], flush=True)
    try:
        F = intraday_features(m1)
        F.to_parquet(os.path.join(a.out, f"feats17_{sym}.parquet"))
        print(sym, "intraday features", len(F), flush=True)
    except Exception as ex:                                                # noqa: BLE001
        LOG["features"] = repr(ex)[:300]
        print("features failed", repr(ex)[:300], flush=True)
    days = None
    rows, keys = [], []
    S = strategies()
    for tf in a.tfs.split(","):
        bars = resample(m1, tf)
        bars = bars[bars.index <= DEV_END]
        day = trading_day(bars.index)
        dd = pd.Index(sorted(set(day)))
        days = dd if days is None else days.union(dd)
        print(sym, tf, "bars", len(bars), round(time.time() - t_job), "s", flush=True)
        tf_rows = []
        for e in S:
            key = f"{e.id}|{tf}"
            if time.time() - t_job > a.budget:
                LOG[key] = {"error": "budget"}
                continue
            t0 = time.time()
            plist = configs(e)
            if a.max_settings:
                plist = plist[: a.max_settings]
            n = 0
            for p in plist:
                try:
                    signal.alarm(400)
                    sig = run_signals(e.id, bars, symbol=sym, params=p)
                    for d in (1, -1):
                        s2 = sig.copy()
                        s2["side"] = sig["side"] * d
                        r = evaluate(e.id, bars, sym, signals=s2)["returns"]
                        tf_rows.append((key, cfg_name(p), d, daily_sum(r, day, dd), dd))
                        n += 1
                    signal.alarm(0)
                except Timeout:
                    LOG.setdefault(key, {})["timeout"] = LOG.get(key, {}).get("timeout", 0) + 1
                except Exception as ex:                                    # noqa: BLE001
                    signal.alarm(0)
                    LOG.setdefault(key, {})["error"] = repr(ex)[:200]
            LOG.setdefault(key, {}).update({"configs": n, "s": round(time.time() - t0, 1)})
        # placebos: random long / flat / short states with persistence p per bar
        roll = rollover_counts(bars.index)
        nb = len(bars)
        for k in range(N_PLACEBO):
            for pp in PLACEBO_PERSIST:
                rng = np.random.default_rng(zlib.crc32(f"P17|{k}|{pp}|{sym}|{tf}".encode()))
                sw = rng.random(nb) > pp
                draw = rng.integers(-1, 2, nb).astype(float)
                idx = np.maximum.accumulate(np.where(sw, np.arange(nb), 0))
                sd = draw[idx]
                sd[: int(np.argmax(sw))] = 0.0
                for d in (1, -1):
                    bt = vector_backtest(bars, pd.Series(sd * d, index=bars.index), pip, BASE, roll)
                    tf_rows.append((f"PLACEBO17_{k:02d}|{tf}", f"p={pp}", d, daily_sum(bt["net"], day, dd), dd))
        rows.extend(tf_rows)
        print(sym, tf, "configs", len(tf_rows), round(time.time() - t_job), "s", flush=True)
        del bars
    # align every row on the union of trading days
    mat = np.zeros((len(rows), len(days)), np.float32)
    pos = {d: i for i, d in enumerate(days)}
    for i, (_, _, _, v, dd) in enumerate(rows):
        mat[i, [pos[x] for x in dd]] = v
    np.savez_compressed(os.path.join(a.out, f"daily17_{sym}.npz"), unit=np.array([r[0] for r in rows]),
                        cfg=np.array([r[1] for r in rows]), dir=np.array([r[2] for r in rows], np.int8),
                        days=np.array(list(days)), pnl=mat)
    json.dump(LOG, open(os.path.join(a.out, f"log17_{sym}.json"), "w"), indent=0, default=str)
    print("done", sym, "rows", len(rows), "days", len(days), round(time.time() - t_job), "s", flush=True)


if __name__ == "__main__":
    main()
