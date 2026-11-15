"""Round 22 robustness check of the fair-value-gap family (I-FVG), which met the counting rule on NAS100 and pooled.
Development data only. Re-runs the same 128 configurations and walk-forward under harsher execution:
  V0 baseline (reproduces round 22)
  V1 CFD bid / ask costs instead of futures costs (indices)
  V2 a limit fills only if price trades through it by one tick (NQ / ES 0.25 point; FX 0.1 pip; gold 0.1)
  V3 one tick of slippage on every stop and twice the futures cost
  V4 V2 and V3 together
and reports, for V0, the chosen configurations and the trades by year, side and hour.
Usage: python scripts/r22_fvg_check.py --symbol SYM --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r22_I as I  # noqa: E402
import wf22  # noqa: E402

KNOB = {"through": 0.0, "stop_slip": 0.0, "cost_mult": 1.0, "cfd": False}
ORIG_COST = I.Bars.cost


def cost(self, i, px):
    if KNOB["cfd"] or not self.fut:
        from fxlab.execution.costs import BASE
        s = self.spr[min(i, self.n - 1)]
        return (0.5 * s + BASE.slip_spread_frac * s + BASE.slip_fixed_pips * self.pip) / px + BASE.commission_bps_per_side * 1e-4
    return KNOB["cost_mult"] * self.fut / px


def trade(B, i0, d, entry, stop, tgt, max_hold, limit=None, expiry=0, cancel=None, t_exit=None):
    o, h, l, c = B.o, B.h, B.l, B.c
    n = B.n
    th, ss = KNOB["through"], KNOB["stop_slip"]
    j = None
    for k in range(i0, min(n, i0 + expiry + 1)):
        if cancel is not None and ((d > 0 and h[k] >= cancel) or (d < 0 and l[k] <= cancel)):
            return None
        if (d > 0 and l[k] <= limit - th) or (d < 0 and h[k] >= limit + th):
            j, E = k, (min(o[k], limit) if d > 0 else max(o[k], limit))
            break
    if j is None or stop is None or not np.isfinite(stop) or (d > 0 and stop >= E) or (d < 0 and stop <= E):
        return None
    X, kx = None, None
    for k in range(j, min(n, j + max_hold + 1)):
        if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
            base = stop if k == j else (min(o[k], stop) if d > 0 else max(o[k], stop))
            X, kx = base - d * ss, k
            break
        if k > j and tgt is not None and ((d > 0 and h[k] >= tgt) or (d < 0 and l[k] <= tgt)):
            X, kx = (max(o[k], tgt) if d > 0 else min(o[k], tgt)), k
            break
    if X is None:
        kx = min(n - 1, j + max_hold)
        X = c[kx]
    g = d * (X / E - 1)
    return j, g - B.cost(j, E) - B.cost(kx, X), g, kx, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mid, pip = I.load(a.symbol)
    index = "IDX" in a.symbol
    tfs = ("5min", "15min") if index else ("15min", "1h")
    tick = 0.25 if index else (0.1 if a.symbol.startswith("XAU") else pip * 0.1)
    I.Bars.cost = cost
    I.trade = trade
    BB = {tf: I.Bars(I.bars(mid, tf, index), a.symbol, pip) for tf in tfs}
    cal = pd.DatetimeIndex(sorted(set(BB[tfs[-1]].day)))
    U = qmc.Sobol(I.FAMS["FVG"] + 1, scramble=True, seed=22 + len("FVG")).random(128)
    first = max(2010, int(cal.year.min()) + 3)
    variants = {"V0": dict(through=0.0, stop_slip=0.0, cost_mult=1.0, cfd=False),
                "V1": dict(through=0.0, stop_slip=0.0, cost_mult=1.0, cfd=True),
                "V2": dict(through=tick, stop_slip=0.0, cost_mult=1.0, cfd=False),
                "V3": dict(through=0.0, stop_slip=tick, cost_mult=2.0, cfd=False),
                "V4": dict(through=tick, stop_slip=tick, cost_mult=2.0, cfd=False)}
    rows, trades0 = [], {}
    for vn, kn in variants.items():
        if vn == "V1" and not index:
            continue
        KNOB.update(kn)
        M = []
        for ci, u in enumerate(U):
            tf = I.pick(u[-1], list(tfs))
            tr = I.fam_FVG(BB[tf], u)
            M.append(I.daily(tr, BB[tf], cal).to_numpy())
            if vn == "V0":
                trades0[ci] = (tf, tr)
        M = np.vstack(M)
        s5, s1, ch = wf22.walk_forward(M, cal, first, min_active=20)
        st = wf22.stream_stats(s5)
        yr = s5.groupby(s5.index.year).agg(["mean", "count"])
        rows.append({"variant": vn, **kn, **st, "own": wf22.counts_rule(st),
                     "by_year_bp": {int(y): round(float(v) * 1e4, 2) for y, v in yr["mean"].items()},
                     "median_cfg_mean_bp": float(np.nanmedian(M.mean(axis=1)) * 1e4),
                     "share_cfg_positive_2019_2025": float((M[:, cal >= "2019-01-01"].mean(axis=1) > 0).mean())})
        if vn == "V0":
            chosen = ch
        print(vn, json.dumps({k: rows[-1][k] for k in ("p1_mean", "p2_mean", "p2_t", "own")}), flush=True)
    # V0 diagnostics: the chosen configurations and their trades in their traded years
    diag = []
    for y, v in chosen.items():
        for i in v["cfg"]:
            tf, tr = trades0[i]
            Bx = BB[tf]
            for t in tr:
                dday = Bx.day[t[0]]
                if dday.year != y:
                    continue
                diag.append({"year": y, "cfg": i, "tf": tf, "day": dday, "hour_ny": round(float(Bx.nyh[t[0]]), 2), "side": t[4],
                             "net": t[1], "gross": t[2], "bars": t[3] - t[0]})
    D = pd.DataFrame(diag)
    out = {"symbol": a.symbol, "variants": rows,
           "chosen": {str(y): [{"cfg": i, **{k: (round(float(x), 3) if isinstance(x, float) else x) for k, x in
                                            zip(["gmin", "with", "entry", "stop_atr", "R", "window", "expiry", "max_hold", "tf"],
                                                [U[i][0] * 1.5, bool(U[i][1] < 0.5), I.pick(U[i][2], ["near", "mid", "far"]),
                                                 U[i][3], 1 + 4 * U[i][4], I.pick(U[i][5], ["any", "london", "nyam", "sb"]),
                                                 int(3 + 45 * U[i][6]), int(6 + 90 * U[i][7]), I.pick(U[i][8], list(tfs))])}}
                                    for i in v["cfg"]] for y, v in chosen.items()}}
    if len(D):
        out["trades"] = {"n": int(len(D)), "win": float((D.net > 0).mean()), "mean_net_bp": float(D.net.mean() * 1e4),
                         "mean_gross_bp": float(D.gross.mean() * 1e4), "median_bars": float(D.bars.median()),
                         "by_side": D.groupby("side").net.agg(["count", "mean"]).assign(mean=lambda x: x["mean"] * 1e4).round(3).to_dict("index"),
                         "by_year": D.groupby("year").net.agg(["count", "mean"]).assign(mean=lambda x: x["mean"] * 1e4).round(3).to_dict("index"),
                         "by_hour": D.groupby(D.hour_ny.astype(int)).net.agg(["count", "mean"]).assign(mean=lambda x: x["mean"] * 1e4).round(3).to_dict("index"),
                         "top_days_share": float(D.groupby("day").net.sum().sort_values(ascending=False).head(20).sum() / D.net.sum())
                         if D.net.sum() > 0 else None}
        D.to_csv(os.path.join(a.out, f"fvg_trades_{a.symbol}.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, f"fvg_check_{a.symbol}.json"), "w"), indent=1, default=str)
    print(json.dumps(out.get("trades", {}), indent=1, default=str))


if __name__ == "__main__":
    main()
