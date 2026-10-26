"""Prop-firm simulation of the edges that a retail prop account can actually trade, alone and mixed, on their real
2019-2025 daily outcomes:
  G  Gotobi USD/JPY (docs/gotobi_oos.md, test F1, net of costs and swap): one trade on each Gotobi day;
  O  Nasdaq-100 overnight on an index CFD (QQQ close-to-open as the price proxy), Monday-Thursday nights only
     (FTMO's standard funded accounts must be flat over the weekend), minus 1 bp of spread and CFD financing of
     the T-bill rate + 2.5% a year per night (an assumption: providers charge a benchmark plus a markup);
  S  SVXY short volatility in VIX contango (docs/missed_edges.md, M8), as a HYPOTHETICAL: the common prop firms do
     not list VIX futures or volatility ETFs.
Rules: FTMO 2-step 100k as in scripts/prop_challenge.py (fee USD 585 refunded with the first payout; +10% then +5%;
5% daily and 10% maximum loss of the initial balance; 4 trading days a phase; 80% of profits every 10 trading days
for a year funded). Each simulated trader starts on a random day of 2019-2025 and walks forward through the real
calendar (wrapping around), so the strategies keep their true joint behaviour (March 2020, February 2018 for S).
Usage: python scripts/prop_mix_sim.py --gotobi series.csv --out DIR"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prop_challenge as PC  # noqa: E402

START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-31")


def yahoo(s):
    import yfinance as yf
    h = yf.download(s, start="2010-01-01", end="2026-01-01", progress=False, auto_adjust=False)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) else pd.to_datetime(h.index)
    return h[h.index <= END]


def build(gotobi_csv):
    G = pd.read_csv(gotobi_csv, parse_dates=["day"])
    G = G[(G.test == "F1") & G.net_sw.notna()].set_index("day").net_sw
    q, irx = yahoo("QQQ"), yahoo("^IRX")
    f = q["Adj Close"] / q["Close"]
    o, c = q["Open"] * f, q["Adj Close"]
    rf = (irx["Close"] / 100).reindex(q.index).ffill().fillna(0)
    night = o / c.shift(1) - 1                                         # realised at the open of day t
    prev = pd.Series(q.index, index=q.index).shift(1)
    weekday_night = (q.index.to_series().diff().dt.days == 1)          # Mon-Thu nights (no weekend, no holiday)
    O = (night - 1e-4 - (rf.shift(1) + 0.025) / 360)[weekday_night].dropna()
    s, v, v3 = yahoo("SVXY"), yahoo("^VIX"), yahoo("^VIX3M")
    j = pd.concat([v["Close"].rename("v"), v3["Close"].rename("v3")], axis=1).dropna()
    r = s["Adj Close"].pct_change()
    cont = (j.v < j.v3).reindex(r.index).ffill().shift(2).fillna(False).astype(bool)
    S = (r * cont - cont.astype(float).diff().abs().fillna(0) * 3e-4).dropna()
    days = pd.bdate_range(START, END)
    X = pd.DataFrame({"G": G.reindex(days), "O": O.reindex(days), "S": S.reindex(days)}).fillna(0.0)
    return X


def calendar_gen(X, lev):
    """Daily account returns: each trader walks the real calendar from a random start (one column per strategy,
    applied in sequence within the day for the daily-loss check)."""
    A = X.to_numpy() * np.asarray(lev)[None, :]
    n = len(A)

    def gen(rng, k):
        pos = rng.integers(0, n, size=k)
        while True:
            yield A[pos % n]
            pos = pos + 1
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gotobi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traders", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = build(a.gotobi)
    X.to_csv(os.path.join(a.out, "daily_inputs.csv"))
    desc = {}
    for c in X:
        x = X[c][X[c] != 0]
        desc[c] = {"events": int(len(x)), "mean_bp": float(x.mean() * 1e4), "sd_bp": float(x.std() * 1e4),
                   "worst_bp": float(x.min() * 1e4), "worst_day": str(x.idxmin().date()),
                   "ann_return_at_1x": float(X[c].sum() / 7.0)}
    corr = X.replace(0, np.nan).corr().round(3).to_dict()
    rows = []
    grid = [(g, o, s_) for g in (0, 3, 6, 10) for o in (0, 0.5, 1, 2, 3) for s_ in (0, 0.1, 0.25, 0.5)]
    for g, o, s_ in grid:
        if g == o == s_ == 0:
            continue
        res = PC.simulate(calendar_gen(X, (g, o, s_)), a.traders, seed=21)
        rows.append({"gotobi_lev": g, "overnight_lev": o, "svxy_lev": s_, **res})
        print(g, o, s_, {k: round(v, 3) for k, v in res.items()}, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "prop_mix.csv"), index=False)
    json.dump({"inputs": desc, "correlation_on_event_days": corr,
               "best_mean_net": R.sort_values("mean_net_usd", ascending=False).head(10).to_dict("records")},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps(desc, indent=1))
    print(R.sort_values("mean_net_usd", ascending=False).head(25).round(3).to_string())


if __name__ == "__main__":
    main()
