"""Random entries, mean-reversion harvest with POOLED, ROBUST parameter choice (run 15).

Runs 11-14 found one property of the price path that management of random entries can harvest: daily mean
reversion (D1 averaging grid gross +0.23; settings fitted to 2010-2018 keep +0.37 with fresh random entries), but
re-tuning every pair every six months picked noise (training +0.84, test +0.10). This script attacks the tuning
noise instead of adding levers. Entries stay strictly random (coin-flip direction, random time).

A SHARED, deterministic candidate set (the same for every pair and timeframe; parameters in ATR units):
  C0       a-priori grid, fixed before any result of this run: 0.5 entries/day, add every 1.0 ATR, 5 levels x1.5,
           take-profit 0.5 ATR beyond the average, stop 3 ATR beyond the last level, no gate
  grid     224 random draws: rate 0.1-2/day, step 0.3-3 ATR, 2-10 levels, size x1-2, take-profit 0.1-3 ATR,
           stop 0.5-8 ATR
  bracket  96 random draws (mean-reversion and other brackets): rate, stop 0.5-8 ATR, target none / 0.1-3 R,
           trailing none / 0.5-8 ATR, time exit none / 1-200 bars
  gates    each draw is ungated, gated to calm markets (ATR14/ATR100 <= q, q 0.7-1.2) or to range-bound
           markets (|close - SMA100| <= m ATR, m 1-5), all measured at the close of the signal bar
For every walk-forward fold (3-year training / 6-month test from 2010, aligned by test-start date across pairs)
and every candidate this job stores:
  - the mean daily net Sharpe over 8 training seeds on the training window (and trades);
  - the out-of-sample expected-value stream (mean over 16 NEW seeds) as daily returns, at full costs and at half
    costs (half spread, half fixed slippage, commission and swap), and gross (zero cost, mid prices, 8 seeds);
  - per-seed daily sums and sums of squares per fold (single-trader statistics of any selection sequence).
  - the pair's variance ratios VR(5), VR(20) of bar returns on the training window (mean-reversion measure).
The selection rules (per-pair, pooled across pairs, top-10 ensemble, pooled + pair filters, a-priori C0) are
applied afterwards by scripts/random_entry_pool_report.py, each using training-window information only.
Usage: python scripts/random_entry_pool.py --pair EURUSD --tf D1 [--out DIR]"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import random_entry_opt as RO  # noqa: E402

TRAIN_SEEDS = int(os.environ.get("REP_TRAIN_SEEDS", 8))
TEST_SEEDS = int(os.environ.get("REP_TEST_SEEDS", 16))
GROSS_SEEDS = 8
CAND_SEED = 20261025


def candidates():
    """The shared candidate list (deterministic, independent of pair and timeframe)."""
    rng = np.random.default_rng(CAND_SEED)
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))       # noqa: E731

    def gate():
        u = rng.random()
        if u < 0.4:
            return {"gate": "none"}
        if u < 0.7:
            return {"gate": "calm", "q": float(rng.uniform(0.7, 1.2))}
        return {"gate": "range", "m": float(rng.uniform(1.0, 5.0))}
    C = [{"id": 0, "fam": "grid", "P": {"rate": 0.5, "step": 1.0, "N": 5, "g": 1.5, "tp": 0.5, "sl": 3.0},
          "gate": {"gate": "none"}, "apriori": True}]
    for _ in range(224):
        C.append({"id": len(C), "fam": "grid", "P": {"rate": lu(0.1, 2.0), "step": lu(0.3, 3.0),
                                                     "N": int(rng.integers(2, 11)), "g": float(rng.uniform(1.0, 2.0)),
                                                     "tp": lu(0.1, 3.0), "sl": lu(0.5, 8.0)}, "gate": gate()})
    for _ in range(96):
        P = {"rate": lu(0.1, 2.0), "sl": lu(0.5, 8.0), "size": "notional"}
        if rng.random() < 0.8:
            P["tp"] = lu(0.1, 3.0)
        if rng.random() < 0.3:
            P["trail"] = lu(0.5, 8.0)
        if rng.random() < 0.5:
            P["tmax"] = int(round(lu(1.0, 200.0)))
        C.append({"id": len(C), "fam": "bracket", "P": P, "gate": gate()})
    return C


class Pool:
    """Per-pair data plus the half-cost variant and the gate inputs."""

    def __init__(self, bars, pip, m1):
        self.D = RO.Data(bars, pip, m1)
        D = self.D
        H = copy.copy(D)                         # half spread: bid' = (3 bid + ask) / 4, ask' = (bid + 3 ask) / 4
        bo, bh, bl, bc, ao, ah, al, ac = D.A
        H.A = tuple((3 * b + a) / 4 for b, a in ((bo, ao), (bh, ah), (bl, al), (bc, ac))) + \
            tuple((b + 3 * a) / 4 for b, a in ((bo, ao), (bh, ah), (bl, al), (bc, ac)))
        if D.use_m1:
            H.mbh, H.mah = (3 * D.mbh + D.mah) / 4, (D.mbh + 3 * D.mah) / 4
            H.mbl, H.mal = (3 * D.mbl + D.mal) / 4, (D.mbl + 3 * D.mal) / 4
        H.spr = D.spr / 2
        self.H = H
        mc = pd.Series(D.mc)
        self.vr = np.r_[D.vr_prev[1:], np.nan]                       # ATR14 / ATR100 at each bar's close
        self.trend = (np.abs(mc - mc.rolling(100, min_periods=100).mean()) / D.atr).to_numpy()
        from fxlab.execution.costs import BASE
        self.half_costs = (BASE.slip_spread_frac, 0.5 * BASE.slip_fixed_pips,
                           0.5 * BASE.commission_bps_per_side * 1e-4, 0.5 * BASE.swap_pips_per_night)

    def gate(self, c, w0, i1):
        g = c["gate"]
        if g["gate"] == "calm":
            return self.vr[w0:i1] <= g["q"]                     # NaN (warm-up) -> False
        if g["gate"] == "range":
            return self.trend[w0:i1] <= g["m"]
        return None

    def run(self, c, i0, i1, seeds, bpd, mode="net"):
        """Bar returns on [i0, i1) per seed and trades, for one candidate."""
        P = c["P"]
        p = min(1.0, P["rate"] / bpd)
        w0 = max(0, i0 - RO.WARM)
        g = self.gate(c, w0, i1)
        out = []
        for s in seeds:
            e = RO.entries(p, s, w0, i0, i1)
            if g is not None:
                e[~g] = 0
            if mode == "half":
                b, ntr = RO.run_family(c["fam"], P, self.H, e, i0, i1, costs=self.half_costs)
            else:
                b, ntr = RO.run_family(c["fam"], P, self.D, e, i0, i1, gross=(mode == "gross"))
            out.append((b, ntr))
        return out


def daily(b, D, i0):
    dd = D.day[i0:i0 + len(b)] - D.day[i0]
    return np.bincount(dd, weights=b)


def sharpe(s):
    return float(s.mean() / s.std() * math.sqrt(260)) if len(s) >= 20 and s.std() > 0 else 0.0


def variance_ratio(c, q):
    r = np.diff(np.log(c))
    r = r[np.isfinite(r)]
    if len(r) < 10 * q:
        return float("nan")
    v1 = r.var()
    s = np.convolve(r, np.ones(q), "valid")
    return float(s.var() / (q * v1)) if v1 > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "random_entry_pool"))
    ap.add_argument("--limit", type=int, default=0, help="first N candidates only (tests)")
    a = ap.parse_args()
    import feature_scan as FS
    from fxlab.data.store import load_m1, M1_PATH_COLUMNS, TF_MINUTES
    from fxlab.data.instruments import get as instrument
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.tf == "M1":                             # the bars are the M1 path itself: no intrabar replay below M1
        bars = load_m1(a.pair, a.start, a.end, columns=["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"])
        m1 = None
    else:
        bars = FS.cached_bars(a.pair, a.tf, a.start, a.end)
        m1 = load_m1(a.pair, a.start, a.end, columns=M1_PATH_COLUMNS)
    X = Pool(bars, instrument(a.pair).pip, m1)
    del m1
    D = X.D
    bpd = 1440.0 / TF_MINUTES.get(a.tf, 60) if a.tf in TF_MINUTES else 24.0
    C = candidates()
    if a.limit:
        C = C[:a.limit]
    tr_lo, tr_hi, te_hi = FS.folds(bars.index)
    F = len(tr_lo)
    nC = len(C)
    train_sh = np.full((nC, F), np.nan, np.float32); train_tr = np.zeros((nC, F), np.float32)
    gross_sh = np.full((nC, F), np.nan, np.float32)
    seed_sum = np.zeros((nC, F, TEST_SEEDS), np.float64); seed_sq = np.zeros((nC, F, TEST_SEEDS), np.float64)
    ndays = np.zeros(F, np.int32)
    fold_info, net_parts, half_parts, gross_parts = [], [], [], []
    for f in range(F):
        i0, i1, i2 = int(tr_lo[f]), int(tr_hi[f]), int(te_hi[f])
        d0 = int(D.day[i1]); nd = int(D.day[i2 - 1] - D.day[i1] + 1)
        dates = pd.DatetimeIndex(D.days[d0:d0 + nd])
        ndays[f] = nd
        fold_info.append({"fold": f, "train_start": str(bars.index[i0]), "test_start": str(bars.index[i1]),
                          "test_end": str(bars.index[i2 - 1]), "test_days": nd,
                          "vr5": variance_ratio(D.mc[i0:i1], 5), "vr20": variance_ratio(D.mc[i0:i1], 20)})
        tseeds = [10 ** 6 + 1000 * f + k for k in range(TEST_SEEDS)]
        net = np.zeros((nd, nC), np.float32); half = np.zeros((nd, nC), np.float32); gro = np.zeros((nd, nC), np.float32)
        for ci, c in enumerate(C):
            tr = X.run(c, i0, i1, [1000 * f + k for k in range(TRAIN_SEEDS)], bpd)
            train_sh[ci, f] = np.mean([sharpe(daily(b, D, i0)) for b, _ in tr])
            train_tr[ci, f] = np.mean([n for _, n in tr])
            te = X.run(c, i1, i2, tseeds, bpd)
            ds = np.array([daily(b, D, i1) for b, _ in te])                      # seeds x days
            seed_sum[ci, f] = ds.sum(1); seed_sq[ci, f] = (ds ** 2).sum(1)
            net[:, ci] = ds.mean(0)
            th = X.run(c, i1, i2, tseeds, bpd, "half")
            half[:, ci] = np.mean([daily(b, D, i1) for b, _ in th], axis=0)
            tg = X.run(c, i1, i2, tseeds[:GROSS_SEEDS], bpd, "gross")
            gro[:, ci] = np.mean([daily(b, D, i1) for b, _ in tg], axis=0)
            gross_sh[ci, f] = sharpe(gro[:, ci])
        net_parts.append(pd.DataFrame(net, index=dates)); half_parts.append(pd.DataFrame(half, index=dates))
        gross_parts.append(pd.DataFrame(gro, index=dates))
        print(f"{a.pair} {a.tf} fold {f + 1}/{F} {fold_info[-1]['test_start'][:10]} done "
              f"({time.time() - t0:.0f}s)", flush=True)
    tag = f"{a.pair}_{a.tf}"
    # daily expected-value streams in basis points as float16 (relative precision ~1e-3; keeps artifacts small)
    day = np.concatenate([p.index.values.astype("datetime64[ns]").astype(np.int64) for p in net_parts])
    fold = np.concatenate([np.full(len(p), i, np.int16) for i, p in enumerate(net_parts)])
    net_bp = np.concatenate([p.to_numpy() for p in net_parts]) * 1e4
    half_bp = np.concatenate([p.to_numpy() for p in half_parts]) * 1e4
    gross_bp = np.concatenate([p.to_numpy() for p in gross_parts]) * 1e4
    np.savez_compressed(os.path.join(a.out, f"{tag}.npz"), train_sh=train_sh, train_tr=train_tr, gross_sh=gross_sh,
                        seed_sum=seed_sum.astype(np.float32), seed_sq=seed_sq.astype(np.float32), ndays=ndays,
                        day=day, fold=fold, net_bp=net_bp.astype(np.float16), half_bp=half_bp.astype(np.float16),
                        gross_bp=gross_bp.astype(np.float16))
    json.dump({"pair": a.pair, "tf": a.tf, "bars": D.n, "folds": fold_info, "candidates": nC,
               "candidate_seed": CAND_SEED, "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
               "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    print(f"done {a.pair} {a.tf}: {nC} candidates x {F} folds in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
