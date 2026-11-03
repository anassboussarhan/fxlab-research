"""M3, M4 and the FS5 trading rule of docs/edges_batch_preregistration.md (addenda 3 and 5).
Stage 1 (--bars SYM --tf TF): per market, the bar table of the fixed-barrier ML lab (8.5 features minus tick counts;
B1 barrier labels) with, for EVERY bar, the long and short trade outcome opened at the next bar (net with costs and
swap, exit time on the M1 path), the round-trip cost, and the 12-bar forward mid return. For --cross, the 1- and
4-bar returns of the cross instruments over their 20-bar volatility are added (M3).
Stage 2 (--pool TF): M4. One LightGBM model across the 12 markets (market id as a feature), walk-forward by year as
8.5 (model A before the previous year sets per-market cut-offs on the previous year; model B on all earlier years);
rules C90 / C95 / C98 / C99 (one position at a time per market). FS5: a second pooled model predicts
P(|12-bar return| > 2 x cost); the C90 trades are kept only when it is in its fit-year top tercile.
Stage 3 (--m3 SYM): M3. The single-market C90 rule of E20 with and without the cross-asset features, bars from
2012, split 60/40 by years.
Usage: python scripts/ml_pooled_lab.py --bars EURUSD --tf H1 --out DIR [--cross]
       python scripts/ml_pooled_lab.py --pool H1 --tables DIR --out DIR
       python scripts/ml_pooled_lab.py --m3 USDJPY --tables DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_triple_barrier_lab as TB  # noqa: E402

UK, DK, HZ = TB.BARRIERS["B1"]
CUTS = (0.90, 0.95, 0.98, 0.99)
CROSS = ("USTBONDTRUSD", "BUNDTREUR", "USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY", "BRENTCMDUSD")
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]


def h_closes(sym, tf, index):
    """1- and 4-bar returns of `sym` over its 20-bar volatility, aligned to bars opening at `index` (same open time:
    both bars close together, so the value is known when the target bar closes)."""
    from fxlab.data.store import load_m1, resample
    try:
        m1 = load_m1(sym, "2007-01-01", TB.END)
    except FileNotFoundError:
        return None
    m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)]
    b = resample(m1, tf)
    c = 0.5 * (b.bid_c + b.ask_c)
    lr = np.log(c).diff()
    v = lr.rolling(20, min_periods=15).std()
    f = pd.DataFrame({f"x_{sym}_r1": lr / (v + 1e-12), f"x_{sym}_r4": np.log(c / c.shift(4)) / (v * 2 + 1e-12)})
    return f.reindex(f.index.union(index)).ffill().reindex(index)


def bars_stage(a):
    import flows_lab as FL
    from fxlab.data.store import load_m1, resample
    t0 = time.time()
    m1 = load_m1(a.bars, "2007-01-01", TB.END)
    m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)].dropna(subset=COLS)
    bars = resample(m1, a.tf).dropna(subset=["bid_o", "ask_o", "bid_h", "ask_l"])
    pip = TB.PIPS.get(a.bars, 1e-4)
    bk = FL.Book(m1[COLS], pip, a.bars if len(a.bars) == 6 and a.bars not in TB.CFD else None)
    del m1
    first_m1 = bk.idx.searchsorted(bars.index)
    F, at, o, h, l = TB.features(bars)
    F = F.drop(columns=[c for c in ("ticks_z",) if c in F.columns])
    y = TB.tb_labels(o, h, l, at, UK, DK, HZ)
    usd = TB.cfd_rates(bk.idx) if a.bars in TB.CFD else None
    n = len(bars)
    out = np.full((n, 4), np.nan)
    for t in range(n - 2):
        if not (np.isfinite(at[t]) and at[t] > 0):
            continue
        for k, side in ((0, 1), (1, -1)):
            r, jx = TB.trade(bk, first_m1, t, side, at[t], UK, DK, HZ, a.bars, usd)
            if r is not None:
                out[t, k] = r[0]
                out[t, 2 + k] = bk.idx[jx].as_unit("ns").value
    mid_c = 0.5 * (bars.bid_c + bars.ask_c).to_numpy()
    fwd = np.full(n, np.nan)
    fwd[:n - HZ - 1] = np.log(o[1 + HZ:n] / o[1:n - HZ])
    cost = (1.5 * (bars.ask_c - bars.bid_c).to_numpy() + 0.2 * pip) / mid_c + 0.7e-4
    T = F.copy()
    T["label"], T["long_net"], T["short_net"] = y, out[:, 0], out[:, 1]
    T["long_exit_ns"], T["short_exit_ns"] = out[:, 2], out[:, 3]
    T["entry_ns"] = np.r_[bk.idx[np.clip(first_m1[1:], 0, len(bk.idx) - 1)].as_unit("ns").asi8, -1].astype("float64")
    T["cost"], T["fwd"], T["market"] = cost, fwd, a.bars
    if a.cross:
        for s in CROSS:
            x = h_closes(s, a.tf, bars.index)
            if x is not None:
                T = T.join(x, how="left")
    os.makedirs(a.out, exist_ok=True)
    T.reset_index().to_parquet(os.path.join(a.out, f"table_{a.bars}_{a.tf}.parquet"))
    print(a.bars, a.tf, T.shape, f"{time.time() - t0:.0f}s", flush=True)


def simulate(D, side, rule_name, rows, trades):
    """One position at a time per market; D sorted by time; side array (+1 / -1 / 0)."""
    side = np.asarray(side)
    for mkt in D.market.unique():
        idx = np.flatnonzero((D.market.to_numpy() == mkt) & (side != 0))
        busy = -1.0
        en, ln, sn = D.entry_ns.to_numpy(), D.long_net.to_numpy(), D.short_net.to_numpy()
        lx, sx, ts = D.long_exit_ns.to_numpy(), D.short_exit_ns.to_numpy(), D.ts.to_numpy()
        for i in idx:
            s = side[i]
            if en[i] <= busy:
                continue
            net, ex = (ln[i], lx[i]) if s > 0 else (sn[i], sx[i])
            if not np.isfinite(net):
                continue
            trades.append({"rule": rule_name, "market": mkt, "time": ts[i], "side": int(s), "net": float(net)})
            busy = ex


def summarise(T):
    import ml_confidence_lab as MC
    rows = []
    for (rule, mkt), g in T.groupby(["rule", "market"]):
        for row in MC.summarise(mkt, "pooled", rule, g[["time", "side", "net"]].assign(gross=np.nan)):
            rows.append(row)
    return pd.DataFrame(rows)


def pool_stage(a):
    import lightgbm as lgb
    fs = sorted(glob.glob(os.path.join(a.tables, "**", f"table_*_{a.pool}.parquet"), recursive=True))
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    D = D.sort_values(["ts", "market"]).reset_index(drop=True)
    D["mkt_id"] = D.market.astype("category").cat.codes
    feats = [c for c in D.columns if c not in ("ts", "label", "long_net", "short_net", "long_exit_ns", "short_exit_ns",
                                               "entry_ns", "cost", "fwd", "market") and not c.startswith("x_")]
    yrs = D.ts.dt.year.to_numpy()
    step = pd.Timedelta({"H1": "1h", "H4": "4h", "D1": "1D"}[a.pool]) * (HZ + 1)
    lab_end_year = (D.ts + step).dt.year.to_numpy()                       # labels must end before the test year
    ok = ((D.label >= 0) & D.r200.notna()).to_numpy()
    big = (D.fwd.abs() > 2 * D.cost).astype(int).to_numpy()
    hasf = D.fwd.notna().to_numpy()
    X = D[feats].to_numpy(np.float32)
    lab = D.label.to_numpy()
    BIN = {k: v for k, v in TB.LGB.items() if k not in ("objective", "num_class")}
    side_all = {f"C{int(c * 100)}": np.zeros(len(D), int) for c in CUTS}
    side_all["FS5 C90 + large-move filter"] = np.zeros(len(D), int)
    years = sorted(set(yrs))
    for Y in years[3:]:
        trB = ok & (lab_end_year < Y)
        trA = ok & (lab_end_year < Y - 1)
        cal = ok & (yrs == Y - 1)
        te = np.flatnonzero(yrs == Y)
        if trA.sum() < 5000 or not len(te):
            continue
        mA = lgb.LGBMClassifier(**TB.LGB).fit(X[trA], lab[trA])
        pc = mA.predict_proba(X[cal])
        mB = lgb.LGBMClassifier(**TB.LGB).fit(X[trB], lab[trB])
        p = mB.predict_proba(X[te])
        sl, ss = p[:, 2] - p[:, 0], p[:, 0] - p[:, 2]
        vA = lgb.LGBMClassifier(**BIN).fit(X[trA & hasf], big[trA & hasf])
        vcut = float(np.quantile(vA.predict_proba(X[cal])[:, 1], 2 / 3))
        vB = lgb.LGBMClassifier(**BIN).fit(X[trB & hasf], big[trB & hasf])
        pv = vB.predict_proba(X[te])[:, 1]
        mk_cal = D.market[cal].to_numpy()
        mk_te = D.market.iloc[te].to_numpy()
        for c in CUTS:
            thl = {m: np.quantile((pc[:, 2] - pc[:, 0])[mk_cal == m], c) for m in set(mk_cal)}
            ths = {m: np.quantile((pc[:, 0] - pc[:, 2])[mk_cal == m], c) for m in set(mk_cal)}
            l_ = np.array([thl.get(m, np.inf) for m in mk_te])
            s_ = np.array([ths.get(m, np.inf) for m in mk_te])
            sd = np.where((sl > l_) & (sl > 0), 1, np.where((ss > s_) & (ss > 0), -1, 0))
            side_all[f"C{int(c * 100)}"][te] = sd
            if c == 0.90:
                side_all["FS5 C90 + large-move filter"][te] = np.where(pv > vcut, sd, 0)
        print(a.pool, Y, {k: int((v[te] != 0).sum()) for k, v in side_all.items()}, flush=True)
    trades = []
    for rule, sd in side_all.items():
        simulate(D, sd, rule, None, trades)
    T = pd.DataFrame(trades)
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, f"pooled_trades_{a.pool}.csv.gz"), index=False)
    S = summarise(T)
    S.to_csv(os.path.join(a.out, f"pooled_tests_{a.pool}.csv"), index=False)
    pd.set_option("display.width", 220)
    if "net_bp" in S:
        print(S[S.period != "all"].pivot_table(index=["symbol"], columns=["rule", "period"], values="net_bp").round(2).to_string())


def m3_stage(a):
    import lightgbm as lgb
    import ml_confidence_lab as MC
    f = glob.glob(os.path.join(a.tables, "**", f"table_{a.m3}_{a.tf}.parquet"), recursive=True)
    D = pd.read_parquet(f[0])
    D = D[D.ts >= "2012-01-01"].reset_index(drop=True)
    xc = [c for c in D.columns if c.startswith("x_")]
    base = [c for c in D.columns if c not in ("ts", "label", "long_net", "short_net", "long_exit_ns", "short_exit_ns",
                                              "entry_ns", "cost", "fwd", "market") and not c.startswith("x_")]
    yrs = D.ts.dt.year.to_numpy()
    ok = ((D.label >= 0) & D.r200.notna()).to_numpy()
    lab = D.label.to_numpy()
    years = sorted(set(yrs))
    cut_year = years[int(0.6 * len(years))]
    rows = []
    for name, feats in (("M3 without cross-asset", base), ("M3 with cross-asset", base + xc)):
        sd_all = np.zeros(len(D), int)
        X = D[feats].to_numpy(np.float32)
        for Y in years[2:]:
            trA, cal, trB = ok & (yrs < Y - 1), ok & (yrs == Y - 1), ok & (yrs < Y)
            te = np.flatnonzero(yrs == Y)
            if trA.sum() < 2000:
                continue
            mA = lgb.LGBMClassifier(**TB.LGB).fit(X[trA], lab[trA])
            pc = mA.predict_proba(X[cal])
            mB = lgb.LGBMClassifier(**TB.LGB).fit(X[trB], lab[trB])
            p = mB.predict_proba(X[te])
            sl, ss = p[:, 2] - p[:, 0], p[:, 0] - p[:, 2]
            thl, ths = np.quantile(pc[:, 2] - pc[:, 0], 0.9), np.quantile(pc[:, 0] - pc[:, 2], 0.9)
            sd_all[te] = np.where((sl > thl) & (sl > 0), 1, np.where((ss > ths) & (ss > 0), -1, 0))
        trades = []
        simulate(D, sd_all, name, None, trades)
        T = pd.DataFrame(trades)
        T["time"] = pd.to_datetime(T.time, utc=True)
        for per, m in (("fit (before %d)" % cut_year, T.time.dt.year < cut_year), ("check (from %d)" % cut_year, T.time.dt.year >= cut_year)):
            s = T[m]
            t = TB.month_t(s.set_index("time").net) if len(s) > 10 else np.nan
            rows.append({"test": name, "symbol": a.m3, "period": per, "trades": int(len(s)),
                         "net_bp": float(s.net.mean() * 1e4) if len(s) else np.nan, "t": t,
                         "cross_features": len(xc)})
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, f"m3_{a.m3}.csv"), index=False)
    print(R.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default="")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--pool", default="")
    ap.add_argument("--m3", default="")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.bars:
        bars_stage(a)
    elif a.pool:
        pool_stage(a)
    elif a.m3:
        m3_stage(a)


if __name__ == "__main__":
    main()
