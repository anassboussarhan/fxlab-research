"""Round 18, part R0 (docs/round18_preregistration.md): is there anything for a regime detector to find?

Feature-free tests of persistent states in strategy performance, against a null that permutes the time order (the
same permutation for every strategy: averages and the cross-section are kept, persistence is removed).
- Variance ratio VR(K) of each strategy's scaled return (median over strategies).
- Mean cross-sectional rank correlation between the trailing mean and the next-period return.

Usage: python scripts/r18_regime0.py --r16 R16DATA --r17 R17DATA --out DIR [--perms 300]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def clean(z, min_share=0.8):
    """Rows from the first period in which at least half of the strategies have a scaled return; strategies with at
    least min_share finite values there; remaining gaps set to 0."""
    fin = np.isfinite(z)
    first = int(np.argmax(fin.mean(1) >= 0.5))
    z = z[first:]
    keep = np.isfinite(z).mean(0) >= min_share
    return np.nan_to_num(z[:, keep]).astype(np.float64), first, int(keep.sum())


def vr_median(x, K):
    T = (x.shape[0] // K) * K
    if T < 4 * K:
        return np.nan
    s = x[:T].reshape(T // K, K, -1).sum(1)
    v1 = x[:T].var(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vr = s.var(0) / (K * v1)
    return float(np.nanmedian(vr[np.isfinite(vr)]))


def rank_ic(x, L, step):
    """Mean over decision periods of the cross-sectional Spearman correlation between the mean of the previous L
    periods and the next period's value."""
    cs = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, 0)])
    out = []
    for j in range(L, x.shape[0], step):
        a = cs[j] - cs[j - L]
        b = x[j]
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        c = np.corrcoef(ra, rb)[0, 1]
        if np.isfinite(c):
            out.append(c)
    return float(np.mean(out)) if out else np.nan


def test(name, z, Ks, L, step, perms, rng, log):
    x, first, n = clean(z)
    T = x.shape[0]
    res = {"library": name, "strategies": n, "periods": T}
    real = {K: vr_median(x, K) for K in Ks}
    ic = rank_ic(x, L, step)
    null_vr = {K: [] for K in Ks}
    null_ic = []
    for p in range(perms):
        xp = x[rng.permutation(T)]
        for K in Ks:
            null_vr[K].append(vr_median(xp, K))
        if p < min(perms, 200):
            null_ic.append(rank_ic(xp, L, step))
    for K in Ks:
        nv = np.array(null_vr[K])
        res[f"VR{K}"] = {"real": real[K], "null_mean": float(np.nanmean(nv)), "null_p95": float(np.nanquantile(nv, 0.95)),
                         "p": float((1 + (nv >= real[K]).sum()) / (1 + len(nv)))}
    ni = np.array(null_ic)
    res["rank_ic"] = {"L": L, "real": ic, "null_mean": float(np.nanmean(ni)), "null_p95": float(np.nanquantile(ni, 0.95)),
                      "p": float((1 + (ni >= ic).sum()) / (1 + len(ni)))}
    res["horizons_above_p95"] = int(sum(res[f"VR{K}"]["real"] > res[f"VR{K}"]["null_p95"] for K in Ks))
    log(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r16", default="")
    ap.add_argument("--r17", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)       # noqa: E731
    rng = np.random.default_rng(18)
    out = []
    if a.r16:
        import regime16 as G16
        D = G16.load(a.r16)
        KM = (3, 6, 12, 24)
        out.append(test("monthly UNIT (round 16)", D.tiers["UNIT"].z, KM, 12, 1, a.perms, rng, log))
        out.append(test("monthly UNIT placebo", D.ptiers["UNIT"].z, KM, 12, 1, a.perms, rng, log))
        out.append(test("monthly FACTOR, 9 styles", D.tiers["FACTOR"].z, KM, 12, 1, a.perms, rng, log))
    if a.r17:
        import regime17 as G17
        KD = (5, 21, 63, 126, 252)
        D = G17.load(a.r17, "BOOK-UNIT", "D")
        out.append(test("daily BOOK-UNIT (round 17)", D.tier.z, KD, 63, 21, a.perms, rng, log))
        out.append(test("daily BOOK-UNIT placebo", D.ptier.z, KD, 63, 21, a.perms, rng, log))
        # by engine: the D1 / W1 / MN library and the intraday library separately
        eng = D.tier.desc["engine"].to_numpy()
        for e, lab in (("D", "daily engine units"), ("I", "intraday units")):
            out.append(test(f"daily BOOK-UNIT, {lab}", D.tier.z[:, eng == e], KD, 63, 21, a.perms, rng, log))
        del D
        D = G17.load(a.r17, "FACTOR", "D")
        out.append(test("daily FACTOR, 9 styles", D.tier.z, KD, 63, 21, a.perms, rng, log))
    json.dump(out, open(os.path.join(a.out, "regime0.json"), "w"), indent=1)
    rows = []
    for r in out:
        for k, v in r.items():
            if k.startswith("VR") or k == "rank_ic":
                rows.append({"library": r["library"], "strategies": r["strategies"], "periods": r["periods"], "stat": k, **v})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "regime0.csv"), index=False)


if __name__ == "__main__":
    main()
