"""FS1, FS2, FS4 and FS5 (information part) of docs/edges_batch_preregistration.md (addenda 4 and 5). Bars
2007-2025 only (nothing from 2026).

Engine (all four): walk-forward folds of 3 training years and 6 test months, first test 2010 (weekly bars: the same
calendar folds). In each fold a candidate's sign is learned from its training IC (uncentred correlation of the
training-standardised, 1%/99%-winsorised value with the next-h-bar log return, as scripts/feature_ic_lab.py) and its
signed test IC is recorded. The null repeats EVERYTHING, including any selection among candidates, on 20 circularly
shifted targets; z = (IC - mean null) / sd null, n-weighted over the folds of a period (2010-2018 and 2019-2025).
Economic test: in each test window, bars with the candidate at or above its training 90th percentile take the sign
of the training top-minus-bottom-decile spread for h bars, bars at or below the 10th the opposite sign; net = signed
next-h-bar mid return minus the round-trip cost at the bar (2 x (spread x (1/2 + 0.25) + 0.1 pip) / mid + 0.7 bp).
Overlapping signals are all kept; t is clustered by calendar month. Counts (FS1, FS2, FS4): net > 0 and t >= 1.645 on
2019-2025, and the 2019-2025 IC z significant after Benjamini-Hochberg (10%) over the whole batch (--combine).

FS1  every registered feature with a search space, every setting (cartesian product of its space) x 5 transforms
     (raw, minus 20-bar EMA, percentile rank over 250 bars, z over 100 bars, 1-bar change); in each fold the setting
     and transform with the largest |training IC| is chosen per (feature, output column). H8, D1, W1 (W1 = D1 bars
     to Friday's New York close); h = 6, 24 (W1: 1, 4).
FS4  the 20 features with the highest median IC in docs/fx_feature_predictability.csv, defaults, first output
     column; the 190 products of two of them, each factor standardised with the fold's training statistics.
     H4 and D1, h = 6, 24.
FS5  the same 20 features: IC with |next-h-bar return| and with the indicator |return| > 2 x cost (sign learned in
     training). Information only. H4 and D1, h = 6, 24.
FS2  the 68 features rated "predictive" in the same file, defaults, all output columns; M30, H1, H2, H4; h = 6, 24.
     Six cells: Tokyo 00-07, London 07-13, New York 13-21 UTC, and the terciles of 20-bar realised volatility
     (cut-offs from 2007-2018). On 2007-2018 the cell with the largest |IC| (standardised and signed in sample) is
     chosen; on 2019-2025 only that cell is traded (deciles and sign from 2007-2018, in that cell), after costs.
Usage: python scripts/fs_lab.py --mode fs1|fs2|fs4|fs5 --sym EURUSD --tf D1 --out DIR
       python scripts/fs_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K_NULL = 20
PRED = os.path.join(ROOT, "docs", "fx_feature_predictability.csv")
DEV_END = "2025-12-31"


# ------------------------------------------------------------------------------------------------ data
def bars_of(sym, tf):
    from fxlab.orchestration.runners import cached_bars
    if tf != "W1":
        return cached_bars(sym, tf, "2007-01-01", DEV_END)
    b = cached_bars(sym, "D1", "2007-01-01", DEV_END)
    # D1 bars open at 17:00 New York; a week = the bars opening Sunday..Thursday, labelled by its first bar
    wk = (b.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)).to_period("W-FRI")
    agg = {c: ("first" if c.endswith("_o") or c == "open" else "max" if c.endswith("_h") or c in ("high", "spread_max")
               else "min" if c.endswith("_l") or c in ("low", "spread_min") else "last" if c.endswith("_c") or c in ("close", "spread")
               else "sum" if c in ("volume", "n_ticks", "n_m1", "up_ticks", "down_ticks", "bid_liq", "ask_liq") else "mean")
           for c in b.columns}
    g = b.groupby(wk)
    w = g.agg(agg)
    w.index = g.apply(lambda x: x.index[0])
    w.index.name = "ts"
    return w


def panel_of(sym, tf, index):
    import feature_scan as FS
    from fxlab.orchestration.runners import load_panel_closes
    others = [s for s in FS.ALL if s != sym]
    if tf != "W1":
        return load_panel_closes(others, tf, index, "2007-01-01", DEV_END)
    cols = {}
    for s in others:
        try:
            cols[s] = bars_of(s, "W1")["close"]
        except Exception:                                                   # noqa: BLE001
            pass
    return pd.DataFrame(cols).reindex(index)


def costs(bars, sym):
    from fxlab.data.instruments import get as instrument
    try:
        pip = instrument(sym).pip
    except Exception:                                                       # noqa: BLE001
        pip = 0.01 if "JPY" in sym else 1e-4
    mid = ((bars.bid_c + bars.ask_c) / 2).to_numpy(float)
    sp = (bars.ask_c - bars.bid_c).to_numpy(float)
    return mid, 2 * ((0.5 + 0.25) * sp + 0.1 * pip) / mid + 2 * 0.35e-4


def folds(index):
    t0 = pd.Timestamp("2010-01-01", tz="UTC")
    end = pd.Timestamp(DEV_END, tz="UTC") + pd.Timedelta(days=1)
    out = []
    while t0 < end:
        t1 = min(t0 + pd.DateOffset(months=6), end)
        a, m, b = index.searchsorted(t0 - pd.DateOffset(years=3)), index.searchsorted(t0), index.searchsorted(t1)
        if m - a > 100 and b > m:
            out.append((a, m, b, t0.year))
        t0 = t1
    return out


def targets(mid, hs, seed=777):
    n = len(mid)
    lr = np.r_[np.diff(np.log(mid)), 0.0]
    cs = np.r_[0.0, np.cumsum(lr)]
    ys = {}
    for h in hs:
        y = np.full(n, np.nan)
        y[: n - h] = cs[h + 1: n + 1] - cs[1: n - h + 1]
        ys[h] = y
    shifts = np.random.default_rng(seed).integers(int(0.1 * n), int(0.9 * n), size=K_NULL)
    return ys, shifts


# ------------------------------------------------------------------------------------------------ engine
class Acc:
    """Per (group, h, period) accumulators: IC (real and nulls, n-weighted) and month-clustered trade nets."""

    def __init__(self, groups, hs, labels=("2010-2018", "2019-2025"), split=2019):
        self.groups, self.hs = list(groups), list(hs)
        self.labels, self.split = labels, split
        G, H = len(self.groups), len(hs)
        self.ic = {p: np.zeros((G, H, K_NULL + 1)) for p in labels}
        self.w = {p: np.zeros(H) for p in self.ic}
        self.tr = {p: [[{} for _ in range(H)] for _ in range(G)] for p in self.ic}  # month -> [sum, count, sumsq]

    def add_trades(self, per, g, hi, months, net):
        d = self.tr[per][g][hi]
        if not len(net):
            return
        s = pd.DataFrame({"m": months, "x": net}).groupby("m").x.agg(["sum", "count"])
        for m, r in s.iterrows():
            v = d.setdefault(m, [0.0, 0])
            v[0] += r["sum"]
            v[1] += int(r["count"])

    def rows(self, meta):
        out = []
        for per in self.ic:
            for gi, g in enumerate(self.groups):
                for hi, h in enumerate(self.hs):
                    w = self.w[per][hi]
                    if w <= 0:
                        continue
                    v = self.ic[per][gi, hi] / w
                    z = (v[0] - v[1:].mean()) / (v[1:].std() + 1e-12)
                    d = self.tr[per][gi][hi]
                    S = np.array([x[0] for x in d.values()]) if d else np.zeros(0)
                    C = np.array([x[1] for x in d.values()]) if d else np.zeros(0)
                    N = C.sum()
                    mean = S.sum() / N if N else np.nan
                    if N >= 10 and len(S) > 1:
                        e = S - C * mean
                        se = math.sqrt((e ** 2).sum()) / N * math.sqrt(len(S) / (len(S) - 1))
                        t = mean / se if se > 0 else np.nan
                    else:
                        t = np.nan
                    out.append({**meta, "group": g, "h": h, "period": per, "ic": v[0], "null_mean": v[1:].mean(),
                                "null_sd": v[1:].std(), "z": z, "trades": int(N), "net_bp": mean * 1e4, "t": t})
        return out


def standardise(xtr, xte):
    q = np.nanpercentile(xtr, [1, 10, 90, 99], axis=0)
    lo, q10, q90, hi = q
    xtr_c, xte_c = np.clip(xtr, lo, hi), np.clip(xte, lo, hi)
    mu, sd = np.nanmean(xtr_c, 0), np.nanstd(xtr_c, 0) + 1e-15
    return np.nan_to_num((xtr_c - mu) / sd), np.nan_to_num((xte_c - mu) / sd), q10, q90


def run_engine(acc, gid, fold_mats, ys, shifts, c_rt, index, hs, target_fn=None):
    """gid: group index of every candidate column. fold_mats(a, m, b) -> (xtr, xte) for the candidates."""
    n = len(index)
    G = len(acc.groups)
    month = index.tz_convert(None).to_period("M").astype(str).to_numpy() if index.tz is not None else \
        index.to_period("M").astype(str).to_numpy()
    T = {h: (target_fn(ys[h]) if target_fn else ys[h]) for h in hs}
    Y = np.column_stack([np.roll(T[h], int(s)) if k else T[h] for h in hs for k, s in enumerate([0] + list(shifts))])
    order = np.argsort(gid, kind="stable")
    gs = gid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    for a, m, b, yr in FOLDS_CACHE:
        per = acc.labels[1] if yr >= acc.split else acc.labels[0]
        tr = np.arange(a, max(a, m - max(hs)))
        te = np.arange(m, min(b, n - max(hs)))
        if len(tr) < 100 or len(te) < 10:
            continue
        xtr, xte = fold_mats(tr, te)
        ztr, zte, q10, q90 = standardise(xtr, xte)
        Ytr, Yte = np.nan_to_num(Y[tr]), np.nan_to_num(Y[te])
        sy_tr = np.sqrt((Ytr ** 2).mean(0)) + 1e-15
        sy_te = np.sqrt((Yte ** 2).mean(0)) + 1e-15
        ic_tr = (ztr.T @ Ytr) / len(tr) / sy_tr
        ic_te = (zte.T @ Yte) / len(te) / (np.sqrt((zte ** 2).mean(0))[:, None] + 1e-15) / sy_te
        A = np.abs(ic_tr)[order]                                            # (cands sorted by group, H*G)
        best = np.maximum.reduceat(A, starts, axis=0)                       # (groups, H*G)
        # index of the first candidate reaching the max within each group
        hit = A == np.repeat(best, np.diff(np.r_[starts, len(gs)]), axis=0)
        pos = np.arange(len(gs))[:, None] * np.ones((1, A.shape[1]), int)
        pos = np.where(hit, pos, len(gs) + 1)
        choose = order[np.minimum.reduceat(pos, starts, axis=0)]            # candidate column per (group, H*G)
        val = np.take_along_axis(np.sign(ic_tr) * ic_te, choose, axis=0)    # (groups, H*G)
        grp_ids = gs[starts]
        for hi, h in enumerate(hs):
            cols = slice(hi * (K_NULL + 1), (hi + 1) * (K_NULL + 1))
            acc.ic[per][grp_ids, hi] += val[:, cols] * len(te)
            acc.w[per][hi] += len(te)
            # economics for the real target's chosen candidate
            j0 = hi * (K_NULL + 1)
            ytr_h, yte_h = ys[h][tr], ys[h][te]
            for k, g in enumerate(grp_ids):
                c = choose[k, j0]
                x_tr, x_te = xtr[:, c], xte[:, c]
                top, bot = x_tr >= q90[c], x_tr <= q10[c]
                if top.sum() < 5 or bot.sum() < 5:
                    continue
                s = np.sign(np.nanmean(ytr_h[top]) - np.nanmean(ytr_h[bot]))
                if s == 0 or not np.isfinite(s):
                    continue
                pz = np.where(x_te >= q90[c], s, np.where(x_te <= q10[c], -s, 0.0))
                k_ = np.flatnonzero((pz != 0) & np.isfinite(yte_h))
                if len(k_):
                    acc.add_trades(per, g, hi, month[te][k_], pz[k_] * yte_h[k_] - c_rt[te][k_])
    return acc


FOLDS_CACHE: list = []


# ------------------------------------------------------------------------------------------------ candidate sets
def feature_cols(bars, sym, tf, panel, fid, params=None):
    from fxlab.feature_store import compute_feature
    return compute_feature(bars, fid, sym, tf, panel, params, use_cache=False)


def transforms(X):
    out = {"raw": X, "d20": X - X.ewm(span=20, adjust=False, min_periods=20).mean(),
           "rank250": X.rolling(250, min_periods=100).rank(pct=True),
           "z100": (X - X.rolling(100, min_periods=50).mean()) / (X.rolling(100, min_periods=50).std() + 1e-12),
           "chg1": X.diff()}
    return out


def fs1_candidates(bars, sym, tf, panel, budget_s):
    import json as _j
    from fxlab.config import RESULTS_DIR
    from fxlab.registry import by_kind, load_all
    load_all()
    try:
        bench = _j.loads((RESULTS_DIR / "benchmark.json").read_text())["features"]
    except Exception:                                                       # noqa: BLE001
        bench = {}
    scale = len(bars) / bench.get("bars", 12501)
    mats, gid, groups, log = [], [], [], {"skipped": {}, "errors": {}}
    t0 = time.time()
    for e in sorted((e for e in by_kind("feature") if e.space), key=lambda e: e.id):
        keys = list(e.space)
        combos = list(itertools.product(*[(e.space[k] if isinstance(e.space[k], (list, tuple)) else [e.space[k]]) for k in keys]))
        est = bench.get("per_feature", {}).get(e.id, 0.0) * scale * len(combos)
        if est > 900:
            log["skipped"][e.id] = round(est)
            continue
        if time.time() - t0 > budget_s:
            log["skipped"][e.id] = "time budget"
            continue
        per_col = {}
        for cb in combos:
            p = dict(e.params) | dict(zip(keys, cb))
            try:
                X = feature_cols(bars, sym, tf, panel, e.id, p).astype("float64")
            except Exception as ex:                                         # noqa: BLE001
                log["errors"][f"{e.id} {p}"] = repr(ex)[:120]
                continue
            for tn, Xt in transforms(X).items():
                for c in Xt.columns:
                    per_col.setdefault(c, []).append(Xt[c].to_numpy(np.float32))
        for c, arrs in per_col.items():
            gi = len(groups)
            groups.append(c)
            for x in arrs:
                mats.append(x)
                gid.append(gi)
    M = np.column_stack(mats) if mats else np.zeros((len(bars), 0), np.float32)
    M[~np.isfinite(M)] = np.nan
    log["groups"], log["candidates"], log["seconds"] = len(groups), int(M.shape[1]), round(time.time() - t0)
    return M, np.array(gid), groups, log


def top_features(k=20):
    P = pd.read_csv(PRED)
    return list(P.sort_values("median_ic", ascending=False).feature[:k])


def default_first_cols(bars, sym, tf, panel, fids):
    cols = {}
    for f in fids:
        try:
            X = feature_cols(bars, sym, tf, panel, f)
            cols[f] = X.iloc[:, 0].to_numpy(np.float64)
        except Exception as ex:                                             # noqa: BLE001
            print("feature", f, repr(ex)[:100], flush=True)
    return cols


# ------------------------------------------------------------------------------------------------ modes
def mode_fs1(a, bars, panel, mid, c_rt):
    hs = (1, 4) if a.tf == "W1" else (6, 24)
    M, gid, groups, log = fs1_candidates(bars, a.sym, a.tf, panel, a.budget)
    print(a.sym, a.tf, log["groups"], "groups", log["candidates"], "candidates", log["seconds"], "s", flush=True)
    ys, sh = targets(mid, hs)
    acc = Acc(groups, hs)
    run_engine(acc, gid, lambda tr, te: (M[tr].astype(np.float64), M[te].astype(np.float64)), ys, sh, c_rt,
               bars.index, hs)
    return acc.rows({"mode": "fs1", "sym": a.sym, "tf": a.tf}), log


def mode_fs4(a, bars, panel, mid, c_rt):
    hs = (6, 24)
    fids = top_features(20)
    cols = default_first_cols(bars, a.sym, a.tf, panel, fids)
    names = [f for f in fids if f in cols]
    pairs = list(itertools.combinations(names, 2))
    B = np.column_stack([cols[f] for f in names])

    def mats(tr, te):
        mu, sd = np.nanmean(B[tr], 0), np.nanstd(B[tr], 0) + 1e-12
        Z = (B - mu) / sd
        P = np.column_stack([Z[:, names.index(x)] * Z[:, names.index(y)] for x, y in pairs])
        return P[tr], P[te]
    ys, sh = targets(mid, hs)
    acc = Acc([f"{x} x {y}" for x, y in pairs], hs)
    run_engine(acc, np.arange(len(pairs)), mats, ys, sh, c_rt, bars.index, hs)
    return acc.rows({"mode": "fs4", "sym": a.sym, "tf": a.tf}), {"features": names, "pairs": len(pairs)}


def mode_fs5(a, bars, panel, mid, c_rt):
    hs = (6, 24)
    fids = top_features(20)
    cols = default_first_cols(bars, a.sym, a.tf, panel, fids)
    names = [f for f in fids if f in cols]
    B = np.column_stack([cols[f] for f in names])
    ys, sh = targets(mid, hs)
    rows = []
    med_c = float(np.nanmedian(c_rt))
    for tgt, fn in (("abs_return", np.abs), ("large_move", lambda y: (np.abs(y) > 2 * med_c).astype(float) -
                                                                      np.nanmean(np.abs(y) > 2 * med_c))):
        acc = Acc(names, hs)
        def f2(y, fn=fn):
            out = fn(y)
            out = np.where(np.isfinite(y), out, np.nan)
            return out - np.nanmean(out) if tgt == "abs_return" else out
        run_engine(acc, np.arange(len(names)), lambda tr, te: (B[tr], B[te]), ys, sh, c_rt, bars.index, hs,
                   target_fn=f2)
        for r in acc.rows({"mode": "fs5", "sym": a.sym, "tf": a.tf, "target": tgt}):
            r.pop("trades"), r.pop("net_bp"), r.pop("t")
            rows.append(r)
    return rows, {"features": names}


def mode_fs2(a, bars, panel, mid, c_rt):
    hs = (6, 24)
    P = pd.read_csv(PRED)
    fids = list(P[P.verdict.astype(str).str.startswith("predictive")].feature)
    ys, _ = targets(mid, hs)
    idx = bars.index
    hr = idx.hour.to_numpy()
    lr = np.r_[np.nan, np.diff(np.log(mid))]
    rv = pd.Series(lr).rolling(20, min_periods=15).std().to_numpy()
    fit = (idx <= pd.Timestamp("2018-12-31 23:59", tz="UTC"))
    chk = ~fit
    v1, v2 = np.nanquantile(rv[fit], [1 / 3, 2 / 3])
    cells = {"Tokyo 00-07": (hr >= 0) & (hr < 7), "London 07-13": (hr >= 7) & (hr < 13),
             "New York 13-21": (hr >= 13) & (hr < 21), "vol low": rv <= v1, "vol mid": (rv > v1) & (rv <= v2),
             "vol high": rv > v2}
    month = idx.tz_convert(None).to_period("M").astype(str).to_numpy()
    rows = []
    from fxlab.feature_store import compute_feature
    for f in fids:
        try:
            X = compute_feature(bars, f, a.sym, a.tf, panel, None, use_cache=False)
        except Exception as ex:                                             # noqa: BLE001
            print("feature", f, repr(ex)[:100], flush=True)
            continue
        for col in X.columns:
            x = X[col].to_numpy(np.float64)
            for h in hs:
                y = ys[h]
                ok = np.isfinite(x) & np.isfinite(y)
                best = None
                for cn, cm in cells.items():
                    m = ok & fit & cm
                    if m.sum() < 300:
                        continue
                    xs = (x[m] - x[m].mean()) / (x[m].std() + 1e-12)
                    ic = float((xs * y[m]).mean() / (np.sqrt((y[m] ** 2).mean()) + 1e-15))
                    if best is None or abs(ic) > abs(best[1]):
                        best = (cn, ic)
                if best is None:
                    continue
                cm = cells[best[0]]
                mf = ok & fit & cm
                q10, q90 = np.quantile(x[mf], [0.1, 0.9])
                s = np.sign(y[mf & (x >= q90)].mean() - y[mf & (x <= q10)].mean())
                mc = ok & chk & cm
                pz = np.where(x >= q90, s, np.where(x <= q10, -s, 0.0))
                k = np.flatnonzero(mc & (pz != 0))
                net = pz[k] * y[k] - c_rt[k]
                xs = x[ok & chk & cm]
                ic_c = float(np.sign(best[1]) * ((xs - x[mf].mean()) / (x[mf].std() + 1e-12) * y[ok & chk & cm]).mean() /
                             (np.sqrt((y[ok & chk & cm] ** 2).mean()) + 1e-15)) if (ok & chk & cm).sum() > 50 else np.nan
                t = np.nan
                if len(k) >= 10:
                    S = pd.Series(net).groupby(month[k]).agg(["sum", "count"])
                    mean = net.mean()
                    e = S["sum"] - S["count"] * mean
                    se = math.sqrt((e ** 2).sum()) / len(net) * math.sqrt(len(S) / max(len(S) - 1, 1))
                    t = mean / se if se > 0 else np.nan
                rows.append({"mode": "fs2", "sym": a.sym, "tf": a.tf, "group": col, "h": h, "cell": best[0],
                             "fit_ic": best[1], "check_ic": ic_c, "period": "2019-2025", "trades": int(len(k)),
                             "net_bp": float(net.mean() * 1e4) if len(k) else np.nan, "t": t})
    return rows, {"features": len(fids)}


CROSS = ("USTBONDTRUSD", "BUNDTREUR", "USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY", "BRENTCMDUSD")


def raw_glob(raw, *parts):
    return sorted(glob.glob(os.path.join(raw, "**", *parts), recursive=True))


def event_clock(index, raw):
    """Minutes to the next / since the last event of each schedule (2017-2023 for the calendar dataset)."""
    import flows_lab as FL
    ev = {}
    us = []
    for f in raw_glob(raw, "news_data", "1*_US_*.csv"):
        try:
            us += list(pd.to_datetime(pd.read_csv(f, usecols=["Timestamp"]).Timestamp))
        except Exception:                                                   # noqa: BLE001
            pass
    ev["us_release"] = us
    cb = []
    for f in raw_glob(raw, "news_data", "*_Fed_Interest_Rate_Decision.csv") + \
            raw_glob(raw, "news_data", "*_Eurozone_Interest_Rate_Decision.csv"):
        cb += list(pd.to_datetime(pd.read_csv(f, usecols=["Timestamp"]).Timestamp))
    ev["cb_decision"] = cb
    fs = raw_glob(raw, "treasury_auctions.json")
    if fs:
        import edges_lab2 as E
        E.RAW = os.path.dirname(fs[0])
        ev["treasury_auction"] = [c.tz_convert(None) for _, _, c in E.auctions()]
    years = range(2006, 2027)
    hol = FL.jp_holidays(years)
    G = FL.gotobi_days("2006-12-01", "2026-01-31", hol)
    ev["gotobi_fix"] = [FL.local_ts(d, "Asia/Tokyo", 9, 55).tz_convert(None) for d in G]
    ev["london_fix"] = [FL.local_ts(d, "Europe/London", 16, 0).tz_convert(None) for d in pd.bdate_range("2006-12-01", "2026-01-31")]
    t = index.tz_convert(None)
    tv = t.values.astype("datetime64[ns]").astype(np.int64)
    out = {}
    for k, v in ev.items():
        if len(v) < 20:
            continue
        e = np.unique(pd.DatetimeIndex(v).values.astype("datetime64[ns]").astype(np.int64))
        j = np.searchsorted(e, tv, side="right")
        nxt = np.where(j < len(e), (e[np.minimum(j, len(e) - 1)] - tv) / 6e10, np.nan)
        prv = np.where(j > 0, (tv - e[np.maximum(j - 1, 0)]) / 6e10, np.nan)
        lo, hi = pd.DatetimeIndex(v).min(), pd.DatetimeIndex(v).max()
        ok = (t >= lo) & (t <= hi)
        out[f"ev:{k}_to_next_min"] = np.where(ok, np.minimum(nxt, 60 * 24 * 14), np.nan)
        out[f"ev:{k}_since_last_min"] = np.where(ok, np.minimum(prv, 60 * 24 * 14), np.nan)
    return out


def mode_fs3(a, bars, panel, mid, c_rt):
    from fxlab.orchestration.runners import cached_bars
    hs = (6, 24)
    idx = bars.index
    F = {}
    for s in CROSS:
        try:
            b = cached_bars(s, a.tf, "2007-01-01", DEV_END)
        except Exception as ex:                                             # noqa: BLE001
            print("cross", s, repr(ex)[:100], flush=True)
            continue
        c = np.log(((b.bid_c + b.ask_c) / 2))
        c = c.reindex(c.index.union(idx)).ffill().reindex(idx)
        for k in (1, 4, 24):
            F[f"x:{s}_r{k}"] = (c - c.shift(k)).to_numpy()
    F.update(event_clock(idx, a.raw))
    ccy = {"USDJPY": "JPY", "EURUSD": "EUR"}.get(a.sym)
    if ccy:
        import ml_weekly_lab as W
        Z = W.tff(a.raw).get(ccy)
        if Z is not None:
            pub = pd.DatetimeIndex(Z.index + pd.Timedelta(days=3, hours=20, minutes=30)).tz_localize("UTC")
            zz = pd.Series(Z.values, index=pub).sort_index()
            F["tff:lev_z"] = zz.reindex(zz.index.union(idx)).ffill().reindex(idx).to_numpy()
        from fxlab.data.rates import has_rates, rate_diff
        if has_rates(a.sym):
            days = pd.date_range(idx[0].floor("D"), idx[-1].floor("D"), freq="D")
            rd = rate_diff(a.sym, days)
            ch = (rd - rd.shift(91)).reindex(idx.floor("D")).to_numpy()
            F["rate:gap_chg_3m"] = ch
    names = list(F)
    B = np.column_stack([np.asarray(F[k], float) for k in names])
    ys, sh = targets(mid, hs)
    rows = []
    # split: 2007-2018 / 2019-2025, or 60/40 of the feature's own rows if it starts after 2008
    groups = {}
    for j, k in enumerate(names):
        v = np.isfinite(B[:, j])
        if v.sum() < 300:
            continue
        first = idx[v][0]
        split = 2019 if first.year <= 2008 else int(idx[v][int(0.6 * v.sum())].year)
        groups.setdefault(split, []).append(j)
    for split, js in groups.items():
        acc = Acc([names[j] for j in js], hs, labels=(f"fit (before {split})", f"check (from {split})"), split=split)
        Bs = B[:, js]
        run_engine(acc, np.arange(len(js)), lambda tr, te: (Bs[tr], Bs[te]), ys, sh, c_rt, idx, hs)
        rows += acc.rows({"mode": "fs3", "sym": a.sym, "tf": a.tf})
    return rows, {"features": names, "splits": {str(k): len(v) for k, v in groups.items()}}


def combine(a):
    from scipy import stats as st
    fs = glob.glob(os.path.join(a.combine, "**", "fs_*.csv"), recursive=True)
    R = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 5], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    summ = {}
    for mode, g in R.groupby("mode"):
        c = g[g.period.astype(str).isin(["2019-2025"]) | g.period.astype(str).str.startswith("check")].copy()
        if mode == "fs2":
            c["p"] = st.norm.sf(c.t.fillna(-9))
        else:
            c["p"] = st.norm.sf(c.z.fillna(-9))
        c = c.sort_values("p").reset_index(drop=True)
        m = len(c)
        thr = 0.10 * (np.arange(1, m + 1)) / m
        ok = c.p.to_numpy() <= thr
        kmax = np.flatnonzero(ok).max() + 1 if ok.any() else 0
        c["bh10"] = np.arange(m) < kmax
        if mode == "fs5":
            c["counts"] = False
            cnt = None
        else:
            c["counts"] = c.bh10 & (c.net_bp > 0) & (c.t >= 1.645)
            cnt = int(c.counts.sum())
        c.to_csv(os.path.join(a.out, f"{mode}_check.csv.gz"), index=False)
        top = c.head(15)
        summ[mode] = {"tests_2019_2025": m, "bh10_significant": int(kmax), "count": cnt,
                      "net_positive_t1645": int(((c.net_bp > 0) & (c.t >= 1.645)).sum()) if "net_bp" in c else None,
                      "median_z": float(c.z.median()) if "z" in c and c.z.notna().any() else None,
                      "top": top.drop(columns=[x for x in ("null_mean", "null_sd") if x in top]).round(4).to_dict("records")}
    json.dump(summ, open(os.path.join(a.out, "fs_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "top"} for k, v in summ.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="")
    ap.add_argument("--sym", default="")
    ap.add_argument("--tf", default="D1")
    ap.add_argument("--budget", type=float, default=9000)
    ap.add_argument("--combine", default="")
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(a.out, exist_ok=True)
    if a.combine:
        return combine(a)
    t0 = time.time()
    bars = bars_of(a.sym, a.tf)
    bars = bars[bars.index <= pd.Timestamp(DEV_END + " 23:59", tz="UTC")]
    panel = panel_of(a.sym, a.tf, bars.index)
    mid, c_rt = costs(bars, a.sym)
    FOLDS_CACHE[:] = folds(bars.index)
    fn = {"fs1": mode_fs1, "fs2": mode_fs2, "fs3": mode_fs3, "fs4": mode_fs4, "fs5": mode_fs5}[a.mode]
    rows, log = fn(a, bars, panel, mid, c_rt)
    tag = f"{a.mode}_{a.sym}_{a.tf}"
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"fs_{tag}.csv"), index=False)
    log["seconds_total"] = round(time.time() - t0)
    json.dump(log, open(os.path.join(a.out, f"log_{tag}.json"), "w"), indent=1, default=str)
    R = pd.DataFrame(rows)
    print(tag, len(R), "rows", round(time.time() - t0), "s", flush=True)
    if len(R) and "z" in R:
        c = R[R.period.astype(str).isin(["2019-2025"]) | R.period.astype(str).str.startswith("check")]
        print(c.sort_values("z", ascending=False).head(8).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
