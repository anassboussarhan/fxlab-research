"""Second audit of the round-22 fair-value-gap result (docs/fvg_audit_preregistration.md). Development data only.

Re-implements the I-FVG family in numba and runs the same 128 Sobol configurations and the same walk-forward under:
  A0 legacy   the round-22 logic exactly (checked against scripts/r22_I.py: must match to 1e-12). Each gap's
              limit order is resolved to the end before the next gap is looked at, so whether an earlier order
              fills later decides which trade is taken: a look-ahead. A fill at an open beyond the stop is dropped.
  A1 causal   event-driven: every gap places a resting limit order; one position at a time; when an order fills,
              the others are cancelled; a fill at an open beyond the stop is kept and closed at that open.
              Orders and stops work in the cash session only, so a gap through a level fills at the open.
  A2 flat     A1, flat every day: the three candles on the same day, orders expire at the session end, positions
              closed at the close of the last session bar (16:00 New York). Primary variant: it needs no
              overnight model and is what futures prop firms allow.
  A3 harsh    A2, and a limit fills only if price trades through it by one tick, one tick of slippage on every
              stop, twice the futures cost.
Sources: the Dukascopy NAS100 CFD store (2012-2025); NQ futures minute bars from trades (Databento GLBX.MDP3,
front contract by volume, 2010-06 to 2025; each contract simulated separately so no trade spans a roll); or
random walks (null: no costs, the gross result should be about zero).
Usage: python scripts/fvg_audit.py --source cfd|nq|null [--nq FILE] --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numba as nb
import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r22_I as I  # noqa: E402
import wf22  # noqa: E402

SYM = "USATECHIDXUSD"
TICK = 0.25
FROZEN = [64, 56, 8, 40, 88]
VARIANTS = {"A0": dict(causal=False, discard=True, improve=True, flat=False, through=0.0, ss=0.0, cm=1.0),
            "A1": dict(causal=True, discard=False, improve=True, flat=False, through=0.0, ss=0.0, cm=1.0),
            "A2": dict(causal=True, discard=False, improve=True, flat=True, through=0.0, ss=0.0, cm=1.0),
            "A3": dict(causal=True, discard=False, improve=True, flat=True, through=TICK, ss=TICK, cm=2.0)}


@nb.njit(cache=True)
def _exit(o, h, l, c, dayid, j, d, stop, tgt, mh, ss, improve, flat, n):
    end = min(n - 1, j + mh)
    for k in range(j, end + 1):
        if (d > 0 and l[k] <= stop) or (d < 0 and h[k] >= stop):
            if k == j:
                X = stop
            elif d > 0:
                X = min(o[k], stop)
            else:
                X = max(o[k], stop)
            return k, X - d * ss
        if k > j and ((d > 0 and h[k] >= tgt) or (d < 0 and l[k] <= tgt)):
            if improve:
                X = max(o[k], tgt) if d > 0 else min(o[k], tgt)
            else:
                X = tgt
            return k, X
        if flat and (k == n - 1 or dayid[k + 1] != dayid[k]):
            return k, c[k]
    return end, c[end]


@nb.njit(cache=True)
def _gap(h, l, atr, k, gmin, withd, ent, se, R):
    a = atr[k]
    if not np.isfinite(a) or a <= 0:
        return 0, 0.0, 0.0, 0.0
    if l[k] > h[k - 2] and l[k] - h[k - 2] >= gmin * a:
        top, bot, gd, c1 = l[k], h[k - 2], 1, l[k - 2]
    elif h[k] < l[k - 2] and l[k - 2] - h[k] >= gmin * a:
        top, bot, gd, c1 = l[k - 2], h[k], -1, h[k - 2]
    else:
        return 0, 0.0, 0.0, 0.0
    d = gd if withd else -gd
    if ent == 0:
        lvl = top if gd > 0 else bot
    elif ent == 1:
        lvl = 0.5 * (top + bot)
    else:
        lvl = bot if gd > 0 else top
    if d == gd:
        stop = c1 - gd * se * a
    else:
        stop = (top if gd > 0 else bot) + gd * (se * a + 1e-12)
    if (d > 0 and stop >= lvl) or (d < 0 and stop <= lvl):
        return 0, 0.0, 0.0, 0.0
    risk = abs(lvl - stop)
    return d, lvl, stop, lvl + d * R * risk


@nb.njit(cache=True)
def sim(o, h, l, c, atr, dayid, okwin, gmin, withd, ent, se, R, exp, mh, causal, discard, improve, flat, through, ss,
        fut, cm):
    n = len(o)
    J = np.empty(n, np.int64)
    NET = np.empty(n)
    cnt = 0
    if not causal:                                                         # round-22 sequencing
        busy = -1
        for k in range(22, n - 2):
            if k <= busy or not okwin[k]:
                continue
            if flat and dayid[k - 2] != dayid[k]:
                continue
            d, lvl, stop, tgt = _gap(h, l, atr, k, gmin, withd, ent, se, R)
            if d == 0:
                continue
            j = -1
            E = 0.0
            for kk in range(k + 1, min(n, k + 1 + exp + 1)):
                if flat and dayid[kk] != dayid[k]:
                    break
                if (d > 0 and l[kk] <= lvl - through) or (d < 0 and h[kk] >= lvl + through):
                    j = kk
                    if improve:
                        E = min(o[kk], lvl) if d > 0 else max(o[kk], lvl)
                    else:
                        E = lvl
                    break
            if j < 0:
                continue
            if (d > 0 and stop >= E) or (d < 0 and stop <= E):
                if discard:
                    continue
                kx, X = j, o[j] - d * ss
            else:
                kx, X = _exit(o, h, l, c, dayid, j, d, stop, tgt, mh, ss, improve, flat, n)
            g = d * (X / E - 1.0)
            J[cnt] = j
            NET[cnt] = g - cm * fut / E - cm * fut / X
            cnt += 1
            busy = kx
        return J[:cnt], NET[:cnt]
    # event-driven: resting orders, one position at a time
    P = 512
    pk = np.empty(P, np.int64)
    pd_ = np.empty(P, np.int64)
    pl = np.empty(P)
    ps = np.empty(P)
    pt = np.empty(P)
    npd = 0
    t = 22
    while t < n:
        # fills during bar t, oldest active order first
        m = 0
        fill = -1
        for q in range(npd):
            kq = pk[q]
            if t > kq + 1 + exp or (flat and dayid[t] != dayid[kq]):
                continue                                                   # expired
            pk[m] = pk[q]
            pd_[m] = pd_[q]
            pl[m] = pl[q]
            ps[m] = ps[q]
            pt[m] = pt[q]
            if fill < 0 and t >= kq + 1:
                d = pd_[m]
                if (d > 0 and l[t] <= pl[m] - through) or (d < 0 and h[t] >= pl[m] + through):
                    fill = m
            m += 1
        npd = m
        if fill >= 0:
            d, lvl, stop, tgt = pd_[fill], pl[fill], ps[fill], pt[fill]
            if improve:
                E = min(o[t], lvl) if d > 0 else max(o[t], lvl)
            else:
                E = lvl
            if (d > 0 and stop >= E) or (d < 0 and stop <= E):
                kx, X = t, o[t] - d * ss
            else:
                kx, X = _exit(o, h, l, c, dayid, t, d, stop, tgt, mh, ss, improve, flat, n)
            g = d * (X / E - 1.0)
            J[cnt] = t
            NET[cnt] = g - cm * fut / E - cm * fut / X
            cnt += 1
            npd = 0
            t = kx + 1
            continue
        # a gap completed by bar t places an order from bar t+1
        if t <= n - 3 and okwin[t] and not (flat and dayid[t - 2] != dayid[t]):
            d, lvl, stop, tgt = _gap(h, l, atr, t, gmin, withd, ent, se, R)
            if d != 0 and npd < P:
                pk[npd] = t
                pd_[npd] = d
                pl[npd] = lvl
                ps[npd] = stop
                pt[npd] = tgt
                npd += 1
        t += 1
    return J[:cnt], NET[:cnt]


def params(u):
    return dict(gmin=I.rng(u[0], 0, 1.5), withd=bool(u[1] < 0.5), ent=["near", "mid", "far"].index(I.pick(u[2], ["near", "mid", "far"])),
                se=I.rng(u[3], 0, 1), R=I.rng(u[4], 1, 5), win=I.pick(u[5], ["any", "london", "nyam", "sb"]),
                exp=int(I.rng(u[6], 3, 48)), mh=int(I.rng(u[7], 6, 96)))


class Seg:
    """One stretch of bars (a futures contract, or the whole CFD history)."""

    def __init__(self, mid, tf):
        self.B = I.Bars(I.bars(mid, tf, True), SYM, 1.0)
        B = self.B
        self.dayid = pd.factorize(pd.Index(B.day))[0].astype(np.int64)
        self.win = {w: np.array([I.window_ok(B, i, w) for i in range(B.n)], bool) for w in ("any", "london", "nyam", "sb")}


def run_cfg(segs, u, v):
    p = params(u)
    tf = I.pick(u[-1], ["5min", "15min"])
    days, nets = [], []
    for S in segs[tf]:
        B = S.B
        if B.n < 30:
            continue
        J, NET = sim(B.o, B.h, B.l, B.c, B.atr, S.dayid, S.win[p["win"]], p["gmin"], p["withd"], p["ent"], p["se"], p["R"],
                     p["exp"], p["mh"], v["causal"], v["discard"], v["improve"], v["flat"], v["through"], v["ss"],
                     I.FUT_PTS[SYM], v["cm"])
        days.append(np.asarray(B.day)[J])
        nets.append(NET)
    if not days:
        return pd.Series(dtype=float), 0
    d = np.concatenate(days)
    x = np.concatenate(nets)
    return pd.Series(x, index=pd.DatetimeIndex(d)).groupby(level=0).sum(), len(x)


def choose_next(M, cal, top=5, min_active=20, min_valid=50):
    """The configurations the walk-forward would trade in 2026: same scoring as wf22.walk_forward, on all of
    2010-2025 (no 2026 data is used)."""
    X = np.where(np.isfinite(M), M, 0.0)
    m = np.asarray(cal <= wf22.DEV_END)
    X = X[:, m]
    n = np.maximum(np.isfinite(M[:, m]).sum(1), 1)
    act = (X != 0).sum(1)
    mu = X.sum(1) / n
    var = np.maximum((X ** 2).sum(1) / n - mu ** 2, 1e-30) * n / np.maximum(n - 1, 1)
    t = mu / np.sqrt(var / n)
    t[(act < min_active) | (n < min_valid)] = -np.inf
    o = np.argsort(-t)
    return [int(i) for i in o[:top] if np.isfinite(t[i])], [float(t[i]) for i in o[:top] if np.isfinite(t[i])]


def null_report(out, seeds=6):
    U = qmc.Sobol(I.FAMS["FVG"] + 1, scramble=True, seed=22 + len("FVG")).random(128)
    res = {}
    for sd in range(seeds):
        rg = np.random.default_rng(100 + sd)
        sec = pd.date_range("2024-01-01", "2024-04-30", freq="5s", tz="UTC")
        s = pd.Series(20000 * np.exp(np.cumsum(rg.normal(0, 0.6e-4, len(sec)))), index=sec)
        g = s.resample("1min")
        mid = pd.DataFrame({"o": g.first(), "h": g.max(), "l": g.min(), "c": g.last()})
        mid["spr"] = 1.0
        segs = {tf: [Seg(mid, tf)] for tf in ("5min", "15min")}
        for vn, v in VARIANTS.items():
            v2 = dict(v, cm=0.0)
            tot = {"all": [0.0, 0], "with_gap": [0.0, 0]}
            for u in U:
                x, k = run_cfg(segs, u, v2)
                for key in (["all", "with_gap"] if u[1] < 0.5 else ["all"]):
                    tot[key][0] += float(x.sum())
                    tot[key][1] += k
            for key, (a, b) in tot.items():
                res.setdefault(vn, {}).setdefault(key, []).append(a / max(b, 1) * 1e4)
    rep = {vn: {key: {"gross_bp_per_trade_by_seed": [round(x, 3) for x in xs], "mean": float(np.mean(xs))}
                for key, xs in d.items()} for vn, d in res.items()}
    json.dump(rep, open(os.path.join(out, "fvg_audit_null.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def load_nq(path):
    x = pd.read_parquet(path)
    x.index = pd.DatetimeIndex(x.index)
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    x = x[x.index < pd.Timestamp("2026-01-01", tz="UTC")].sort_index()
    out = []
    for iid, g in x.groupby((x.instrument_id != x.instrument_id.shift()).cumsum()):
        if len(g) < 2000:
            continue
        mid = pd.DataFrame({"o": g.open, "h": g.high, "l": g.low, "c": g.close, "spr": TICK}, index=g.index)
        out.append(mid[np.isfinite(mid.o) & (mid.o > 0)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["cfd", "nq", "null"])
    ap.add_argument("--nq", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.source == "null":
        null_report(a.out)
        return
    if a.source == "cfd":
        mid, _ = I.load(SYM)
        pieces = [mid]
    else:
        pieces = load_nq(a.nq)
    segs = {tf: [Seg(m, tf) for m in pieces] for tf in ("5min", "15min")}
    cal = pd.DatetimeIndex(sorted(set().union(*[set(S.B.day) for S in segs["15min"]])))
    U = qmc.Sobol(I.FAMS["FVG"] + 1, scramble=True, seed=22 + len("FVG")).random(128)
    rep = {"source": a.source, "segments": len(pieces), "days": int(len(cal)), "first": str(cal.min().date()),
           "last": str(cal.max().date()), "variants": {}}
    # A0 must reproduce round 22 exactly (CFD only: round 22 ran on the CFD store)
    if a.source == "cfd":
        diffs = []
        for i in range(0, 128, 8):
            u = U[i]
            tf = I.pick(u[-1], ["5min", "15min"])
            ref = I.daily(I.fam_FVG(segs[tf][0].B, u), segs[tf][0].B, cal)
            mine, _ = run_cfg(segs, u, VARIANTS["A0"])
            diffs.append(float((ref - mine.reindex(cal).fillna(0.0)).abs().max()))
        rep["A0_vs_round22_max_abs_diff"] = max(diffs)
        print("A0 vs round 22, max abs diff over 16 configurations:", max(diffs), flush=True)
    series = {}
    for vn, v in VARIANTS.items():
        M, ntr = [], []
        for u in U:
            s, k = run_cfg(segs, u, v)
            M.append(s.reindex(cal).fillna(0.0).to_numpy())
            ntr.append(k)
        M = np.vstack(M)
        first = max(2010, int(cal.year.min()) + 3)
        s5, s1, ch = wf22.walk_forward(M, cal, first, min_active=20)
        st = wf22.stream_stats(s5)
        fz = pd.Series(M[FROZEN].mean(axis=0), index=cal)
        fz2 = fz[fz.index >= "2019-01-01"]
        rep["variants"][vn] = {**v, **st, "own": wf22.counts_rule(st),
                               "by_year_bp": {int(y): round(float(x) * 1e4, 2) for y, x in s5.groupby(s5.index.year).mean().items()},
                               "median_cfg_2019_2025_bp": float(np.median(M[:, cal >= "2019-01-01"].mean(axis=1)) * 1e4),
                               "trades_per_cfg_median": float(np.median(ntr)),
                               "frozen5_2019_2025": {"mean_bp": float(fz2.mean() * 1e4), "t": wf22.tstat(fz2)},
                               "chosen_2025": ch.get(2025, {}),
                               "chosen_2026_from_dev": dict(zip(["cfg", "train_t"], choose_next(M, cal)))}
        series[vn] = s5
        series[vn + "_frozen5"] = fz
        r = rep["variants"][vn]
        print(vn, json.dumps({k: (round(r[k] * 1e4, 2) if "mean" in k and isinstance(r[k], float) else r[k])
                              for k in ("p1_mean", "p2_mean", "p2_t", "own")}), "frozen5", r["frozen5_2019_2025"], flush=True)
    json.dump(rep, open(os.path.join(a.out, f"fvg_audit_{a.source}.json"), "w"), indent=1, default=str)
    pd.DataFrame(series).to_parquet(os.path.join(a.out, f"fvg_audit_{a.source}_series.parquet"))


if __name__ == "__main__":
    main()
