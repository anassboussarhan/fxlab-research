"""E20 and E22 of docs/edges_batch_preregistration.md: the confidence curve of the fixed-barrier ML entries.
Rerun of scripts/ml_triple_barrier_lab.py (8.5) with its symmetric barrier B1 (take-profit +1.5 ATR, stop -1.5 ATR,
12 bars) on M15, M30, H1 and H4, 11 FX pairs and gold, 2007-2025 (the 2019-2025 panel files added). Same features
(minus the tick-count feature, which the 2019-2025 candle files cannot supply), model, walk-forward, fills and
costs as 8.5; nothing tuned.

Rules (fixed before the run):
  C90 / C95 / C98 / C99: long when P(up) - P(down) is above the q-th percentile of model A's calibration-year
    scores (and above 0), short by the mirror rule, q = 90, 95, 98, 99; one position at a time.
  Q25: a LightGBM quantile model (alpha 0.25 and 0.75, same trees and leaves) of the 12-bar mid return from the
    next bar's open; long when the 25% quantile is above the round-trip cost of that bar (spread x 1.5 + 0.2 pip +
    0.7 bp), short when the 75% quantile is below minus that cost; the same B1 barriers.
  Every test-year bar's score is saved for E22 (agreement across timeframes), which runs in --combine: an H1 C90
    trade is kept only when the latest closed M15 and H4 bars' scores (P(up) - P(down)) have the same sign.
Counts (E20): C98 or C99 net > 0 with month-clustered t >= 1.645 over the walk-forward test years, and the net
  mean rising C90 -> C95 -> C98. 2019-2025 is reported separately.
Usage: python scripts/ml_confidence_lab.py --symbol EURUSD --tf H1 --out DIR
       python scripts/ml_confidence_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_triple_barrier_lab as TB  # noqa: E402

CUTS = (0.90, 0.95, 0.98, 0.99)
UK, DK, HZ = TB.BARRIERS["B1"]


def run(a):
    import lightgbm as lgb
    import flows_lab as FL
    from fxlab.data.store import load_m1, resample
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    m1 = load_m1(a.symbol, "2007-01-01", TB.END)
    m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)].dropna(subset=["bid_o", "ask_o", "bid_c", "ask_c"])
    bars = resample(m1, a.tf).dropna(subset=["bid_o", "ask_o", "bid_h", "ask_l"])
    pip = TB.PIPS.get(a.symbol, 1e-4)
    bk = FL.Book(m1[["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]], pip,
                 a.symbol if len(a.symbol) == 6 and a.symbol not in TB.CFD else None)
    del m1
    first_m1 = bk.idx.searchsorted(bars.index)
    F, at, o, h, l = TB.features(bars)
    F = F.drop(columns=[c for c in ("ticks_z",) if c in F.columns])
    years = bars.index.year.to_numpy()
    yrs = sorted(set(years))
    per_year = len(bars) / max(len(yrs), 1)
    usd = TB.cfd_rates(bk.idx) if a.symbol in TB.CFD else None
    mid_o = o
    fwd = np.full(len(bars), np.nan)
    fwd[:-HZ - 1] = np.log(mid_o[1 + HZ:][: len(bars) - HZ - 1] / mid_o[1:len(bars) - HZ])
    spr = (bars.ask_c - bars.bid_c).to_numpy()
    cost = (1.5 * spr + 0.2 * pip) / bars[["bid_c", "ask_c"]].mean(axis=1).to_numpy() + 0.7e-4
    y = TB.tb_labels(o, h, l, at, UK, DK, HZ)
    ok = (y >= 0) & np.isfinite(F["r200"].to_numpy()) & np.isfinite(at)
    step = HZ if per_year > 20000 else 1
    tidx = np.arange(len(bars))
    lab_end_year = np.r_[years[HZ:], np.repeat(years[-1] + 1, HZ)]
    trades = {r: [] for r in [f"C{int(c * 100)}" for c in CUTS] + ["Q25"]}
    scores = []
    Q = dict(objective="quantile", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=4)
    for Y in yrs[3:]:
        trB = ok & (lab_end_year < Y) & (tidx % step == 0) & (years < Y)
        trA = ok & (lab_end_year < Y - 1) & (tidx % step == 0) & (years < Y - 1)
        cal = ok & (years == Y - 1)
        te = np.flatnonzero(years == Y)
        if trA.sum() < 1000 or cal.sum() < 200 or not len(te):
            continue
        mA = lgb.LGBMClassifier(**TB.LGB).fit(F.values[trA], y[trA])
        pc = mA.predict_proba(F.values[cal])
        cl, cs = pc[:, 2] - pc[:, 0], pc[:, 0] - pc[:, 2]
        th = {c: (float(np.quantile(cl, c)), float(np.quantile(cs, c))) for c in CUTS}
        mB = lgb.LGBMClassifier(**TB.LGB).fit(F.values[trB], y[trB])
        p = mB.predict_proba(F.values[te])
        sl, ss = p[:, 2] - p[:, 0], p[:, 0] - p[:, 2]
        fq = trB & np.isfinite(fwd)
        q25 = lgb.LGBMRegressor(alpha=0.25, **Q).fit(F.values[fq], fwd[fq]).predict(F.values[te])
        q75 = lgb.LGBMRegressor(alpha=0.75, **Q).fit(F.values[fq], fwd[fq]).predict(F.values[te])
        scores.append(pd.DataFrame({"time": bars.index[te], "score": sl}))
        sides = {}
        for c in CUTS:
            thl, ths = th[c]
            sides[f"C{int(c * 100)}"] = np.where((sl > thl) & (sl > 0), 1, np.where((ss > ths) & (ss > 0), -1, 0))
        sides["Q25"] = np.where(q25 > cost[te], 1, np.where(q75 < -cost[te], -1, 0))
        for rule, sd in sides.items():
            busy = -1
            for k, t in enumerate(te):
                if sd[k] == 0 or t + 1 >= len(bars) or not np.isfinite(at[t]) or at[t] <= 0:
                    continue
                i = int(first_m1[t + 1])
                if i <= busy or i >= len(bk.idx):
                    continue
                r, jx = TB.trade(bk, first_m1, t, int(sd[k]), at[t], UK, DK, HZ, a.symbol, usd)
                if r is None:
                    continue
                trades[rule].append({"time": bk.idx[i], "side": int(sd[k]), "net": r[0], "gross": r[1]})
                busy = jx
        print(a.symbol, a.tf, Y, {k: len(v) for k, v in trades.items()}, f"{time.time() - t0:.0f}s", flush=True)
    tag = f"{a.symbol}_{a.tf}"
    pd.concat(scores).to_csv(os.path.join(a.out, f"scores_{tag}.csv.gz"), index=False)
    rows = []
    for rule, T in trades.items():
        T = pd.DataFrame(T)
        T.to_csv(os.path.join(a.out, f"trades_{tag}_{rule}.csv.gz"), index=False)
        rows += summarise(a.symbol, a.tf, rule, T)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"tests_{tag}.csv"), index=False)


def summarise(sym, tf, rule, T):
    out = []
    if not len(T):
        return [{"symbol": sym, "tf": tf, "rule": rule, "period": "all", "trades": 0}]
    T = T.copy()
    T["time"] = pd.to_datetime(T["time"], utc=True)
    x = T.set_index("time").sort_index()
    for per, m in (("all", np.ones(len(x), bool)), ("2010-2018", x.index.year <= 2018), ("2019-2025", x.index.year >= 2019)):
        s = x[m]
        if len(s) < 10:
            out.append({"symbol": sym, "tf": tf, "rule": rule, "period": per, "trades": int(len(s))})
            continue
        t = TB.month_t(s.net)
        out.append({"symbol": sym, "tf": tf, "rule": rule, "period": per, "trades": int(len(s)),
                    "net_bp": float(s.net.mean() * 1e4), "t": t, "gross_bp": float(s.gross.mean() * 1e4),
                    "win": float((s.net > 0).mean())})
    return out


def combine(a):
    from scipy import stats as st
    os.makedirs(a.out, exist_ok=True)
    R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.combine, "**", "tests_*.csv"), recursive=True)
                   if os.path.getsize(f) > 5], ignore_index=True)
    # E22: H1 C90 trades kept when the latest closed M15 and H4 scores agree in sign
    rows = []
    for f in glob.glob(os.path.join(a.combine, "**", "trades_*_H1_C90.csv.gz"), recursive=True):
        sym = os.path.basename(f).split("_")[1]
        T = pd.read_csv(f)
        if not len(T):
            continue
        T["time"] = pd.to_datetime(T["time"], utc=True)
        ok = np.ones(len(T), bool)
        for tf, dt in (("M15", pd.Timedelta(minutes=15)), ("H4", pd.Timedelta(hours=4))):
            g = glob.glob(os.path.join(a.combine, "**", f"scores_{sym}_{tf}.csv.gz"), recursive=True)
            if not g:
                ok[:] = False
                continue
            S = pd.read_csv(g[0])
            S["time"] = pd.to_datetime(S["time"], utc=True) + dt           # a bar's score is known at its close
            S = S.sort_values("time")
            j = pd.merge_asof(T[["time"]].sort_values("time"), S, on="time", direction="backward")
            ok &= (np.sign(j.score.to_numpy()) == T.side.to_numpy())
        rows += summarise(sym, "H1", "E22 agree", T[ok])
    if rows:
        R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    allp = R[R.period == "all"].copy()
    allp["p_one_sided"] = st.norm.sf(allp.t.astype(float))
    from statsmodels.stats.multitest import multipletests
    m = allp.p_one_sided.notna()
    allp.loc[m, "bh_q"] = multipletests(allp.p_one_sided[m], method="fdr_bh")[1]
    piv = R[R.period == "all"].pivot_table(index=["symbol", "tf"], columns="rule", values="net_bp")
    mono = piv.get("C90") is not None and piv.get("C98") is not None
    allp.to_csv(os.path.join(a.out, "tests_all.csv"), index=False)
    R.to_csv(os.path.join(a.out, "tests_by_period.csv"), index=False)
    piv.to_csv(os.path.join(a.out, "net_bp_by_rule.csv"))
    med = R[R.period == "all"].groupby(["tf", "rule"]).net_bp.median().unstack()
    med.to_csv(os.path.join(a.out, "median_net_bp_tf_rule.csv"))
    medg = R[R.period == "all"].groupby(["tf", "rule"]).gross_bp.median().unstack()
    medg.to_csv(os.path.join(a.out, "median_gross_bp_tf_rule.csv"))
    counts = []
    if mono:
        for (s, tf), r in piv.iterrows():
            hi = allp[(allp.symbol == s) & (allp.tf == tf) & allp.rule.isin(["C98", "C99"])]
            okc = ((hi.net_bp > 0) & (hi.t >= 1.645)).any() and r.get("C90", np.nan) < r.get("C95", np.nan) < r.get("C98", np.nan)
            counts.append({"symbol": s, "tf": tf, "counts": bool(okc)})
    pd.DataFrame(counts).to_csv(os.path.join(a.out, "e20_counts.csv"), index=False)
    pd.set_option("display.width", 220)
    print("median net bp by timeframe and rule\n", med.round(2).to_string())
    print("median gross bp\n", medg.round(2).to_string())
    print("best by t\n", allp.sort_values("t", ascending=False).head(15).round(3).to_string())
    print("E20 counts:", [c for c in counts if c["counts"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", default="")
    a = ap.parse_args()
    if a.combine:
        combine(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
