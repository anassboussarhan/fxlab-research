"""Round 23 part PF (docs/round23_preregistration.md): the three live candidates in one prop account. Reported, not
counted.

Daily 2019-2025 outcomes (fractions of the account at leverage 1):
  G  Gotobi B0: test F1 of the gotobi-oos branch, net of costs and swap (one trade per Gotobi day)
  D  D7-base: every open trade marked from fixing to fixing at 0.1 of the account each (ten open trades = 1x), 3 bp a
     round trip split between entry and exit
  N  the frozen NAS100 noise-area breakout of the forward test (scripts/nb_forward.py), net at NQ futures costs
Rules: FTMO 2-step 100k (scripts/prop_challenge.py), traders starting on random days and walking the real calendar.
Leverage per strategy 0, 1, 2, 3, 4 or 6, chosen on 2019-2022 by mean net USD per attempt, then run on 2023-2025.
A second version scales each strategy's edge to its 2026 mean (G +2.04 bp a trade, N +2.6 bp a day, D from part A).
Usage: python scripts/r23_PF.py --gotobi series.csv --a r23_A.json --out DIR [--ecb FILE]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forward_d7 as F  # noqa: E402
import prop_challenge as PC  # noqa: E402
import r23_A  # noqa: E402

START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-31")
LEV = (0, 1, 2, 3, 4, 6)


def d7_daily(ecb):
    X = F.load(ecb)
    X = X[X.index <= END]
    W = F.weekly(X)
    T = r23_A.base_trades(W)
    days = pd.bdate_range(START, END)
    out = pd.Series(0.0, index=days)
    for _, t in T.iterrows():
        px = (X[t.symbol[3:]] / X[t.symbol[:3]]).dropna()
        ex = t.exit if pd.notna(t.exit) else px.index.max()
        seg = px[(px.index >= t.entry) & (px.index <= ex)]
        if len(seg) < 2:
            continue
        r = seg.pct_change().dropna()
        r.iloc[0] -= F.COST / 2
        r.iloc[-1] -= F.COST / 2
        r = r[(r.index >= START) & (r.index <= END)]
        out = out.add(0.1 * r.reindex(days).fillna(0.0), fill_value=0.0)
    return out


def nb_daily():
    import nb_forward as NB
    import r19
    from fxlab.data.store import load_m1
    m1 = load_m1(NB.SYM, "2018-11-01", "2025-12-31", columns=r19.COLS)
    R = NB.run(m1)
    s = R.set_index(pd.to_datetime(R.day)).net_fut
    s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
    return s[(s.index >= START) & (s.index <= END)]


def gen(X, lev):
    A = X.to_numpy() * np.asarray(lev)[None, :]
    n = len(A)

    def g(rng, k):
        pos = rng.integers(0, n, size=k)
        while True:
            yield A[pos % n]
            pos = pos + 1
    return g


def scan(X, traders, seed):
    rows = []
    for lev in itertools.product(LEV, repeat=X.shape[1]):
        if not any(lev):
            continue
        r = PC.simulate(gen(X, lev), traders, seed=seed)
        rows.append({**{f"lev_{c}": l for c, l in zip(X.columns, lev)}, **r})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gotobi", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--ecb", default="")
    ap.add_argument("--traders", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    G = pd.read_csv(a.gotobi, parse_dates=["day"])
    G = G[(G.test == "F1") & G.net_sw.notna()].set_index("day").net_sw
    days = pd.bdate_range(START, END)
    X = pd.DataFrame({"G": G.reindex(days), "D": d7_daily(a.ecb).reindex(days), "N": nb_daily().reindex(days)}).fillna(0.0)
    X.to_csv(os.path.join(a.out, "PF_daily.csv"))
    A = json.load(open(a.a))
    d7 = A["D7-base"]
    ratio_d = (d7["2026"]["corrected"].get("mean_R", 0.0) / d7["2019_2025"]["corrected"]["mean_R"]
               if d7["2019_2025"]["corrected"].get("mean_R") else 0.0)
    act = {c: X[c][X[c] != 0] for c in X}
    ratios = {"G": 2.04e-4 / act["G"].mean(), "N": 2.6e-4 / act["N"].mean(), "D": ratio_d}
    X26 = X.copy()
    for c in X:                                                            # shift the mean on active days only
        m = X[c] != 0
        X26.loc[m, c] = X[c][m] - act[c].mean() * (1 - ratios[c])
    desc = {c: {"active_days": int(len(v)), "mean_bp": float(v.mean() * 1e4), "sd_bp": float(v.std() * 1e4),
                "edge_ratio_2026": float(ratios[c])} for c, v in act.items()}
    rep = {"inputs": desc, "corr": X.replace(0, np.nan).corr().round(3).to_dict()}
    for label, Z in (("as_2019_2025", X), ("at_2026_edges", X26)):
        tr, te = Z[Z.index < "2023-01-01"], Z[Z.index >= "2023-01-01"]
        S_tr = scan(tr, a.traders, 23)
        best = S_tr.sort_values("mean_net_usd", ascending=False).iloc[0]
        lev = tuple(int(best[f"lev_{c}"]) for c in Z.columns)
        test = PC.simulate(gen(te, lev), a.traders, seed=24)
        full = scan(Z, a.traders, 25)
        rep[label] = {"chosen_on_2019_2022": dict(zip(Z.columns, lev)), "train": best.to_dict(), "test_2023_2025": test,
                      "best_full_period_in_sample": full.sort_values("mean_net_usd", ascending=False).head(5).to_dict("records")}
        S_tr.to_csv(os.path.join(a.out, f"PF_scan_{label}.csv"), index=False)
        print(label, json.dumps(rep[label], default=str)[:1500], flush=True)
    json.dump(rep, open(os.path.join(a.out, "PF_summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
