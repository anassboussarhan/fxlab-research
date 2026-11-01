"""Gotobi on every yen pair, optimised walk-forward. The Tokyo-fix flow (importers buying dollars into the 09:55
fix on Gotobi days) is the one edge that has survived costs here; this lab asks which yen pairs carry it, which
entry / exit / stop settings work best, and whether choosing them on past years improves the next year.
Development data only (to 2025-12-31). M1 bid/ask, costs as in the flows lab, swap included.

FIXED BEFORE ANY RESULT:
  Pairs: USDJPY, EURJPY, AUDJPY, GBPJPY, CHFJPY, CADJPY, NZDJPY (long the pair = short the yen).
  Settings grid (72): entry 23:00 Tokyo the evening before, or 01:00, 03:00, 05:00, 07:00, 08:00 Tokyo on the day;
    exit 09:50, 09:53 or 09:55 Tokyo (the 2007-2018 mirror has no 00:00-01:00 UTC bars, so those exits accept the
    01:00 UTC bar, as in every earlier Gotobi test); no stop, or a stop at the mean high-low range of the
    23:00-09:53 window over the previous 20 Gotobi days; Monday Gotobi days traded or skipped.
  Walk-forward: for each test year from 2011, the setting with the highest t of net returns over all earlier years
    (at least 100 trades) is traded in that year, per pair. Pair selection: a pair enters the basket in year Y if
    its chosen setting's t on the earlier years is above 1.5; the basket is the equal-weight average of the
    selected pairs that trade that day.
  Baseline: the fixed I3 setting (23:00 -> 09:53, no stop here; Mondays traded) over the same test years.
  Open loss: every trade's worst open loss (lowest bid before the exit) is recorded for the risk simulations.
PRE-REGISTERED TESTS (one-sided): walk-forward out-of-sample mean net > 0 per pair and for the basket;
  Benjamini-Hochberg over the pairs and the basket.
Usage: python scripts/gotobi_opt_lab.py --out DIR"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

PAIRS = ("USDJPY", "EURJPY", "AUDJPY", "GBPJPY", "CHFJPY", "CADJPY", "NZDJPY")
U.PIPS.update({p: 0.01 for p in PAIRS})
ENTRIES = (("prev", 23), ("day", 1), ("day", 3), ("day", 5), ("day", 7), ("day", 8))
EXITS = (50, 53, 55)
STOPS = ("none", "window")
MONDAYS = (True, False)
GRID = list(itertools.product(ENTRIES, EXITS, STOPS, MONDAYS))
BASE = (("prev", 23), 53, "none", True)
DAY = pd.Timedelta(days=1)


def tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def all_settings(bk, G):
    """{setting: DataFrame[net, gross, mae] indexed by Gotobi day} for every setting of the grid."""
    bk._hl()
    # stop distance: mean range of the 23:00 -> 09:53 window over the previous 20 Gotobi days
    wr = {}
    for d in G:
        i, j = F.window(bk, d)
        if i >= 0:
            wr[d] = float(np.max(bk.mh[i:j + 1]) - np.min(bk.ml[i:j + 1]))
    wr = pd.Series(wr).sort_index()
    stop_d = wr.rolling(20, min_periods=10).mean().shift(1)
    out = {}
    for g in GRID:
        (when, h), m, stop, mon = g
        rows = {}
        for d in G:
            if not mon and d.weekday() == 0:
                continue
            t0 = U.jst(d - DAY, h, 0) if when == "prev" else U.jst(d, h, 0)
            t1 = U.jst(d, 9, m)
            i, j = bk.at(t0, 10), bk.at(t1, 65)
            if i < 0 or j <= i:
                continue
            sl = np.inf
            if stop == "window":
                sl = float(stop_d.get(d, np.nan))
                if not np.isfinite(sl) or sl <= 0:
                    continue
                r = bk.bracket(i, j, +1, 1e9, sl)
            else:
                r = bk.trade_ij(i, j, +1, None, None)
            if r is None:
                continue
            net, gross = U.rec(r)
            rows[d] = (net, gross, F.open_loss(bk, i, j, sl, net))
        out[g] = pd.DataFrame.from_dict(rows, orient="index", columns=["net", "gross", "mae"]).sort_index()
    return out


def walk_forward(S, first=2011, min_n=100):
    """Per test year, the setting with the best t on all earlier years; returns OOS trades and the choices."""
    years = sorted({d.year for f in S.values() for d in f.index})
    oos, chosen = [], []
    for Y in years:
        if Y < first:
            continue
        best, bt = None, -np.inf
        for g, f in S.items():
            tr = f[f.index.year < Y]
            if len(tr) < min_n:
                continue
            t = tstat(tr.net)
            if np.isfinite(t) and t > bt:
                best, bt = g, t
        if best is None:
            continue
        te = S[best][S[best].index.year == Y]
        oos.append(te)
        chosen.append({"year": Y, "setting": str(best), "train_t": bt, "test_n": len(te),
                       "test_mean_bp": float(te.net.mean() * 1e4) if len(te) else np.nan})
    return (pd.concat(oos) if oos else pd.DataFrame(columns=["net", "gross", "mae"])), chosen


def summ(x):
    x = pd.Series(x, dtype=float).dropna()
    t = tstat(x)
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4) if len(x) else np.nan, "t": t,
            "p_one_sided": float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan,
            "hit": float((x > 0).mean()) if len(x) else np.nan,
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}",
            "mean_bp_2019_2025": float(x[x.index >= "2019-01-01"].mean() * 1e4) if len(x) else np.nan,
            "t_2019_2025": tstat(x[x.index >= "2019-01-01"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, choices, oos_all, insample = [], {}, {}, []
    for p in PAIRS:
        bk = U.book(p)
        if bk is None:
            print(p, "no data", flush=True)
            continue
        G, _, _ = F.calendar(bk)
        S = all_settings(bk, G)
        for g, f in S.items():                                   # full-sample table (for information only)
            insample.append({"pair": p, "setting": str(g), **summ(f.net)})
        oos, ch = walk_forward(S)
        oos_all[p], choices[p] = oos, ch
        base = S[BASE][S[BASE].index.year >= 2011]
        oos.to_csv(os.path.join(a.out, f"oos_{p}.csv"))
        rows.append({"pair": p, "series": "walk-forward", **summ(oos.net)})
        rows.append({"pair": p, "series": "fixed I3 (no stop)", **summ(base.net)})
        print(rows[-2], flush=True)
        print(rows[-1], flush=True)
    # basket with walk-forward pair selection
    net_d, mae_d = {}, {}
    for p, ch in choices.items():
        o = oos_all[p]
        for c in ch:
            if c["train_t"] > 1.5:
                y = o[o.index.year == c["year"]]
                for d, r in y.iterrows():
                    net_d.setdefault(d, []).append(r.net)
                    mae_d.setdefault(d, []).append(r.mae)
    B = pd.DataFrame({"net": pd.Series({d: np.mean(v) for d, v in net_d.items()}),
                      "mae": pd.Series({d: np.mean(v) for d, v in mae_d.items()}),
                      "pairs": pd.Series({d: len(v) for d, v in net_d.items()})}).sort_index()
    B.to_csv(os.path.join(a.out, "oos_basket.csv"))
    rows.append({"pair": "basket", "series": "walk-forward pairs and settings", **summ(B.net),
                 "mean_pairs": float(B.pairs.mean()) if len(B) else np.nan})
    R = pd.DataFrame(rows)
    from cross_market_ta_lab import bh_q
    wf = R.series.str.startswith("walk")
    R["q_bh"] = np.nan
    R.loc[wf, "q_bh"] = bh_q(R.loc[wf, "p_one_sided"].to_numpy())
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame(insample).to_csv(os.path.join(a.out, "all_settings_full_sample.csv"), index=False)
    json.dump(choices, open(os.path.join(a.out, "choices.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())
    I = pd.DataFrame(insample)
    print(I.sort_values("t", ascending=False).groupby("pair").head(3).round(2).to_string())


if __name__ == "__main__":
    main()
