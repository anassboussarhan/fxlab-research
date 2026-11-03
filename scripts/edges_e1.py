"""E1 of docs/edges_batch_preregistration.md: size each Gotobi night by the forward-v1 walk-forward probability
instead of trading or skipping it. Walk-forward exactly as scripts/gotobi_filter_wf.py (refit each year 2012-2025 on
all earlier nights, logistic L2 C=0.1, the lab's 24 features); weight = p / mean(p over that fit's own training
nights), capped at 2. Series (80 bp stop, from docs/gotobi_losses/trades_basket_stop_80_bp.csv): all nights,
skip-lowest-third (forward-v1 rule) and weighted. Counts if, on 2019-2025, FTMO 2-Step 5x P(paid) at the 2026-shifted
edge beats both others by >= 3 points AND the weighted night's t beats both.
Usage: python scripts/edges_e1.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gotobi_filter_wf as W  # noqa: E402
import prop_best as PB  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def walk_weights(X, y, first=2012):
    out = []
    for Y in range(first, int(y.index.year.max()) + 1):
        fit = np.asarray(y.index < pd.Timestamp(f"{Y}-01-01"))
        prd = np.asarray(y.index.year == Y)
        if prd.sum() == 0:
            continue
        p, c1, _ = W.fit_predict(X, y, fit, prd)
        p_fit, _, _ = W.fit_predict(X, y, fit, fit)
        out.append(pd.DataFrame({"p": p, "keep": p > c1, "w": np.minimum(p / p_fit.mean(), 2.0)}))
    return pd.concat(out).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = W.load(os.path.join(ROOT, "docs/gotobi_accuracy"), "features_basket.csv")
    y0 = W.load(os.path.join(ROOT, "docs/gotobi_accuracy"), "trades_basket_none.csv").net.reindex(X.index)
    ok = y0.notna()
    X, y0 = X[ok], y0[ok]
    G = walk_weights(X, y0)
    S = W.load(os.path.join(ROOT, "docs/gotobi_losses"), "trades_basket_stop_80_bp.csv")[["net", "mae"]]
    D = G.join(S, how="inner")
    D = D[(D.index >= "2019-01-01") & (D.index <= "2025-12-31")]
    res = {"nights": int(len(D)), "mean_weight": float(D.w.mean()), "share_kept": float(D.keep.mean())}
    shift = D.net.mean() - 2.04e-4
    series = {}
    for scen, d in (("2019-2025", 0.0), ("2026", shift)):
        net, mae = D.net - d, np.minimum(D.mae - d, np.minimum(D.net - d, 0.0))
        series[scen] = {"all": (net, mae), "skip": (net.where(D.keep, 0.0), mae.where(D.keep, 0.0)),
                        "weighted": (net * D.w, mae * D.w)}
    r = PB.P["FTMO 2-Step"]
    for scen, ss in series.items():
        for name, (net, mae) in ss.items():
            B = pd.DataFrame({"net": net, "mae": mae})
            Xd, Md = PB.daily(B)
            paid, npay, passed, first, fail = PB.simulate(Xd, Md, r, a.n, 1, 5, 5)
            x = net[net != 0] if name == "skip" else net
            res[f"{scen}|{name}"] = {"paid": float((npay > 0).mean()), "pass": float(passed.mean()),
                                     "mean_bp": float(x.mean() * 1e4),
                                     "t": float(x.mean() / x.std() * np.sqrt(len(x))),
                                     "per_night_bp": float(net.sum() / len(D) * 1e4)}
    p = {k: res[f"2026|{k}"]["paid"] for k in ("all", "skip", "weighted")}
    t = {k: res[f"2019-2025|{k}"]["t"] for k in ("all", "skip", "weighted")}
    res["counts"] = bool(p["weighted"] >= max(p["all"], p["skip"]) + 0.03 and t["weighted"] > max(t["all"], t["skip"]))
    json.dump(res, open(os.path.join(a.out, "e1.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
