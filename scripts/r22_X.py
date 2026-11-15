"""Round 22 part X: index session rules with every parameter optimised walk-forward
(docs/round22_preregistration.md): noise-area breakout, opening-range breakout, VWAP trend.
US500 / NAS100 at CME futures costs; the European and Asian index CFDs at CFD costs. 128 Sobol configurations.
Usage: python scripts/r22_X.py --symbol SYM --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402

COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
SESS = {"USA500IDXUSD": ("America/New_York", 570, 390), "USATECHIDXUSD": ("America/New_York", 570, 390),
        "DEUIDXEUR": ("Europe/Berlin", 540, 510), "FRAIDXEUR": ("Europe/Paris", 540, 510),
        "EUSIDXEUR": ("Europe/Berlin", 540, 510), "CHEIDXCHF": ("Europe/Zurich", 540, 510),
        "GBRIDXGBP": ("Europe/London", 480, 510), "HKGIDXHKD": ("Asia/Hong_Kong", 570, 390),
        "JPNIDXJPY": ("Asia/Tokyo", 540, 360)}
FUT_PTS = {"USA500IDXUSD": 0.25 + 2.5 / 50, "USATECHIDXUSD": 0.25 + 2.5 / 20}


def days_of(sym):
    from fxlab.data.store import load_m1
    from fxlab.execution.costs import BASE
    try:
        m = load_m1(sym, "2007-01-01", "2025-12-31", columns=COLS + ["n_ticks"])
    except Exception:                                                      # noqa: BLE001
        m = load_m1(sym, "2007-01-01", "2025-12-31", columns=COLS)
    tz, op, L = SESS[sym]
    loc = m.index.tz_convert(tz)
    k = (loc.hour * 60 + loc.minute).to_numpy() - op
    keep = (k >= 0) & (k < L) & (loc.dayofweek < 5)
    x = m[keep]
    kk = k[keep]
    day = loc[keep].normalize().tz_localize(None)
    pip = float(1e-4 * np.nanmedian(0.5 * (m.bid_c + m.ask_c)))
    sf, sp, cm = BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4
    D = {}
    pos = pd.Series(np.arange(len(x)), index=day)
    for d, idx in pos.groupby(level=0):
        ii = idx.to_numpy()
        q = kk[ii]
        if len(ii) < 0.75 * L or q[0] > 5:
            continue
        g = x.iloc[ii]
        A = {}
        for nm in "ohlc":
            v = np.full(L, np.nan)
            v[q] = 0.5 * (g[f"bid_{nm}"].to_numpy() + g[f"ask_{nm}"].to_numpy())
            A[nm] = v
        c = pd.Series(A["c"]).ffill().bfill().to_numpy()
        for nm in "ohl":
            A[nm] = np.where(np.isfinite(A[nm]), A[nm], c)
        A["c"] = c
        s = np.full(L, np.nan)
        s[q] = (g.ask_o - g.bid_o).to_numpy()
        s = pd.Series(s).ffill().bfill().to_numpy()
        w = np.ones(L)
        if "n_ticks" in g and g.n_ticks.notna().mean() > 0.9:
            w = np.ones(L)
            w[q] = np.maximum(g.n_ticks.to_numpy(float), 1.0)
        A["w"] = w
        if sym in FUT_PTS:
            A["cost"] = FUT_PTS[sym] / A["o"]
        else:
            A["cost"] = (0.5 * s + sf * s + sp * pip) / A["o"] + cm
        D[d] = A
    return D, L


def pick(u, opts):
    return opts[min(int(u * len(opts)), len(opts) - 1)]


def rng(u, lo, hi):
    return lo + u * (hi - lo)


def run_trades(A, trades):
    """trades: list of (dir, k_in, k_out) at minute opens (k_out may be L meaning the last close)."""
    L = len(A["o"])
    tot = 0.0
    for d, ki, ko in trades:
        E = A["o"][ki]
        X = A["c"][-1] if ko >= L else A["o"][ko]
        cx = A["cost"][min(ko, L - 1)]
        tot += d * (X / E - 1) - A["cost"][ki] - cx
    return tot


def nb(D, days, L, u):
    vm = rng(u[0], 0.5, 2.5)
    lb = int(rng(u[1], 5, 30))
    step = pick(u[2], [5, 15, 30, 60])
    first = int(rng(u[3], 30, 120))
    trail = pick(u[4], ["vwap", "band", "none"])
    gap = u[5] < 0.5
    out = np.zeros(len(days))
    moves, closes = [], []
    for j, d in enumerate(days):
        A = D[d]
        if len(moves) >= lb and closes:
            sig = np.mean(np.vstack(moves[-lb:]), axis=0)
            op = A["o"][0]
            ub0, lb0 = (max(op, closes[-1]), min(op, closes[-1])) if gap else (op, op)
            vw = np.cumsum((A["h"] + A["l"] + A["c"]) / 3 * A["w"]) / np.cumsum(A["w"])
            pos, ent, tr = 0, None, []
            for m in range(first, L, step):
                p = A["c"][m - 1]
                ub, lbd = ub0 * (1 + vm * sig[m - 1]), lb0 * (1 - vm * sig[m - 1])
                tgt = pos
                if p > ub:
                    tgt = 1
                elif p < lbd:
                    tgt = -1
                elif pos > 0 and ((trail == "vwap" and p < max(ub, vw[m - 1])) or (trail == "band" and p < ub)):
                    tgt = 0
                elif pos < 0 and ((trail == "vwap" and p > min(lbd, vw[m - 1])) or (trail == "band" and p > lbd)):
                    tgt = 0
                if tgt != pos:
                    if pos != 0:
                        tr.append((pos, ent, m))
                    pos = tgt
                    if pos != 0:
                        ent = m
            if pos != 0:
                tr.append((pos, ent, L))
            out[j] = run_trades(A, tr)
        moves.append(np.abs(A["c"] / A["o"][0] - 1.0))
        closes.append(A["c"][-1])
    return out


def orb(D, days, L, u):
    rm = int(rng(u[0], 5, 60))
    mode = pick(u[1], ["candle", "either"])
    stop_k = pick(u[2], ["opp", 0.5, 1.0, 2.0])
    R = pick(u[3], [1, 2, 3, 5, 10, None])
    ex = int(rng(u[4], 0.4, 1.0) * L)
    out = np.zeros(len(days))
    for j, d in enumerate(days):
        A = D[d]
        hi, lo = A["h"][:rm].max(), A["l"][:rm].min()
        rgn = hi - lo
        if rgn <= 0:
            continue
        if mode == "candle":
            dd = np.sign(A["c"][rm - 1] - A["o"][0])
            if dd == 0:
                continue
            ki = rm
        else:
            ki, dd = None, 0
            for k in range(rm, ex):
                if A["h"][k] > hi:
                    ki, dd = k + 1, 1
                    break
                if A["l"][k] < lo:
                    ki, dd = k + 1, -1
                    break
            if ki is None or ki >= L:
                continue
        E = A["o"][ki]
        stop = (lo if dd > 0 else hi) if stop_k == "opp" else E - dd * stop_k * rgn
        if (dd > 0 and stop >= E) or (dd < 0 and stop <= E):
            continue
        tgt = None if R is None else E + dd * R * abs(E - stop)
        X, ko = None, None
        for k in range(ki, min(ex, L)):
            if (dd > 0 and A["l"][k] <= stop) or (dd < 0 and A["h"][k] >= stop):
                X = stop if k == ki else (min(A["o"][k], stop) if dd > 0 else max(A["o"][k], stop))
                ko = k
                break
            if tgt is not None and k > ki and ((dd > 0 and A["h"][k] >= tgt) or (dd < 0 and A["l"][k] <= tgt)):
                X, ko = (max(A["o"][k], tgt) if dd > 0 else min(A["o"][k], tgt)), k
                break
        if X is None:
            ko = min(ex, L - 1)
            X = A["o"][ko] if ex < L else A["c"][-1]
        out[j] = dd * (X / E - 1) - A["cost"][ki] - A["cost"][ko]
    return out


def vwap(D, days, L, u):
    step = max(1, int(rng(u[0], 1, 60)))
    start = int(rng(u[1], 0, 120))
    band = rng(u[2], 0, 0.003)
    out = np.zeros(len(days))
    k = np.arange(L)
    for j, d in enumerate(days):
        A = D[d]
        vw = np.cumsum((A["h"] + A["l"] + A["c"]) / 3 * A["w"]) / np.cumsum(A["w"])
        sig = np.where(A["c"] > vw * (1 + band), 1.0, np.where(A["c"] < vw * (1 - band), -1.0, 0.0))
        dec = (k >= start) & ((k - start) % step == 0)
        held = pd.Series(np.where(dec, sig, np.nan)).ffill().fillna(0.0).to_numpy()
        pos = np.zeros(L)
        pos[1:] = held[:-1]
        pos[-1] = 0.0
        ret = A["o"][1:] / A["o"][:-1] - 1
        dpos = np.abs(np.diff(np.r_[0.0, pos]))
        out[j] = float((pos[:-1] * ret).sum() - (dpos * A["cost"]).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    D, L = days_of(a.symbol)
    days = pd.DatetimeIndex(sorted(D))
    print(a.symbol, "sessions", len(days), days.min(), days.max(), flush=True)
    rows, ser = [], {}
    for fam, fn, dims in (("NB", nb, 6), ("ORB", orb, 5), ("VWAP", vwap, 3)):
        U = qmc.Sobol(dims, scramble=True, seed=40 + dims).random(a.n)
        M = np.vstack([fn(D, days, L, u) for u in U])
        first = max(2010, int(days.year.min()) + 3)
        s5, s1, ch = wf22.walk_forward(M, days, first, min_active=20)
        st, st1 = wf22.stream_stats(s5), wf22.stream_stats(s1)
        rows.append({"part": "X", "family": f"X-{fam}", "symbol": a.symbol, "placebo": False, "n_cfg": a.n, **st,
                     "top1_p2_mean": st1["p2_mean"], "top1_p2_t": st1["p2_t"], "own": wf22.counts_rule(st),
                     "choices": json.dumps({str(k): [[round(float(x), 3) for x in U[i]] for i in v["cfg"]] for k, v in ch.items()})})
        ser[f"X-{fam}"] = s5
        print(a.symbol, fam, round(time.time() - t0), "s", json.dumps({k: rows[-1][k] for k in ("p1_mean", "p2_mean", "p2_t", "own")}), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"X_rows_{a.symbol}.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, f"X_series_{a.symbol}.parquet"))


if __name__ == "__main__":
    main()
