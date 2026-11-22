"""Round 25 engine regression check (docs/round25_preregistration.md, "Engine"): with the new dimensions off, the
changed engine must reproduce the round-24 engine bit for bit.

Runs the round-23 LW evaluation (scripts/r23_LW.evaluate_one) for the first K Sobol configurations of a set of
strategies on synthetic H1 bars, and writes a digest of every per-bar return series. Run once on the old code
(`--save`), once on the new (`--compare`).
Usage: python scripts/r25_engine_check.py --save FILE | --compare FILE [--k 24]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import intraday17 as I17  # noqa: E402
import r23_LW as LW  # noqa: E402

RULES = ["S.TREND.MA_CROSS", "S.MR.BOLLINGER", "S.BO.VOL_BREAKOUT", "S.TREND.DONCHIAN", "S.BOOK.CONNORS_R3",
         "S.MR.RSI_REVERSAL", "S.SMC.FVG_RETRACE", "S.FAMOUS.TURTLE", "S.DAILY.ABS_MOM", "S.BO.RETEST",
         "S.ST.PIVOT_POINTS", "S.BOOK.HEAD_SHOULDERS"]


def synth_bars(seed=3, tf="H1"):
    from fxlab.data.store import resample
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2021-01-04", "2022-12-30 23:59", freq="min", tz="UTC")
    ts = ts[ts.dayofweek < 5]
    p = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.00012, len(ts))))
    o = np.r_[p[0], p[:-1]]
    w = np.abs(rng.normal(0, 0.00006, len(ts)))
    h, l = np.maximum(o, p) * (1 + w), np.minimum(o, p) * (1 - w)
    sp = 1.5e-5
    m1 = pd.DataFrame({"bid_o": o, "bid_h": h, "bid_l": l, "bid_c": p, "ask_o": o + sp, "ask_h": h + sp,
                       "ask_l": l + sp, "ask_c": p + sp, "n_ticks": 10, "spread_mean": sp}, index=ts)
    m1.index.name = "ts"
    return resample(m1, tf)


def run(k):
    warnings.filterwarnings("ignore")
    bars = synth_bars()
    S = {e.id: e for e in I17.strategies()}
    import r23_expose as XP
    out = {}
    for fin in (None, "rates_oecd"):
        LW.G["financing"] = fin
        for rid in RULES:
            e = S[rid]
            fn, consts = XP.expose(e)
            if not (consts and XP.verified(e, fn, bars.iloc[-3000:])):
                fn, consts = e.fn, []
            sps = LW.strategy_specs(e, consts)
            seed = zlib.crc32(f"LW|{rid}|H1".encode()) & 0x7FFFFFFF
            U = qmc.Sobol(len(sps) + LW.N_EXEC, scramble=True, seed=seed).random(k)
            for i, u in enumerate(U):
                params = {s["name"]: LW.draw(s, u[j]) for j, s in enumerate(sps)}
                try:
                    r = LW.evaluate_one(e, fn, bars, "EURUSD", params, u[len(sps):])
                    v = np.nan_to_num(r.to_numpy(np.float64), nan=-9.0)
                    out[f"{fin}|{rid}|{i}"] = hashlib.sha1(v.tobytes()).hexdigest()[:16] + f"|{v.sum():.12e}"
                except Exception as ex:                                    # noqa: BLE001
                    out[f"{fin}|{rid}|{i}"] = "ERR " + type(ex).__name__
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--k", type=int, default=24)
    a = ap.parse_args()
    out = run(a.k)
    if a.save:
        json.dump(out, open(a.save, "w"), indent=0)
        print("saved", len(out), "digests;", sum(v.startswith("ERR") for v in out.values()), "errors")
    if a.compare:
        ref = json.load(open(a.compare))
        bad = [k for k in ref if ref[k] != out.get(k)]
        print("compared", len(ref), "differ", len(bad))
        for k in bad[:20]:
            print(" ", k, ref[k], "->", out.get(k))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
