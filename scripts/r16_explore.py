"""Round 16 exploratory follow-ups (written after the pre-registered results; nothing here counts):
X1. Part A2 with 2,000 time shifts for the features with the smallest p (200 shifts cannot give p below 0.005,
    which is above the first Benjamini-Hochberg threshold of 0.10 / 33 = 0.003).
X2. Part O3 with long-only tilts. The signed tilt can short a factor whose return is net of costs, and then earns
    those costs; the long-only version holds the factor at weight max(0, sign(b) x clipped feature) only."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regime16 as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = G.load(a.data)
rows = []
for tn, feats in (("UNIT", ["JPY_IV", "JPY_RR25", "USD3", "FX_VRP"]), ("FACTOR", ["DVIX", "VR5", "JPY_IV", "JPY_RR25"])):
    t = D.tiers[tn]
    for f in feats:
        x = D.FV[:, D.fcol[f]]
        rho = G.a2_rho(D, t, x)
        offs = G.offsets_for(D, 2000, f"X1|{tn}|{f}")
        null = np.array([G.a2_rho(D, t, np.roll(x, o)) for o in offs])
        null = null[np.isfinite(null)]
        rows.append({"tier": tn, "feature": f, "rho": rho, "draws": len(null), "null_p95": float(np.quantile(null, 0.95)),
                     "null_p99": float(np.quantile(null, 0.99)), "p": (1 + (null >= rho).sum()) / (1 + len(null))})
        print(rows[-1], flush=True)
X1 = pd.DataFrame(rows)
X1["bh_threshold_rank1_of_33"] = 0.10 / 33
X1.to_csv(os.path.join(a.out, "X1_A2_2000_shifts.csv"), index=False)
fac = D.tiers["FACTOR"]
rows = []
for f in D.feats:
    grp = "OPT" if f in G.OPT else "CME" if f in G.CME else "PRICE" if f in G.PRICE else "MACRO"
    xs, fit = G.lagged_feature(D, f, G.FIT_A)
    if xs is None:
        continue
    for i, fn in enumerate(fac.names):
        y = fac.zw[:, i]
        ok = fit & np.isfinite(xs) & np.isfinite(y)
        if ok.sum() < G.MIN_ROWS:
            continue
        b = np.polyfit(xs[ok], y[ok], 1)[0]
        w = np.maximum(0.0, np.sign(b) * np.clip(xs, -2, 2))
        u = w * fac.z[:, i]
        base = fac.z[:, i] * np.isfinite(xs)                                 # always holding the factor
        uc, bc = u[D.mask_check], base[D.mask_check]
        rows.append({"group": grp, "feature": f, "factor": fn, "fit_slope": float(b), "check_mean": float(np.nanmean(uc)),
                     "check_t": G.nw_t(uc), "check_minus_always_t": G.nw_t(uc - bc * np.nanmean(w[D.mask_check]))})
X2 = pd.DataFrame(rows)
X2["p"] = 1 - G.norm.cdf(X2.check_t.fillna(-9))
for grp in (("OPT", "CME"), ("PRICE", "MACRO")):
    m = X2.group.isin(grp)
    X2.loc[m, "bh_pass"] = G.bh(X2.loc[m, "p"].to_numpy())
X2.to_csv(os.path.join(a.out, "X2_O3_long_only.csv"), index=False)
print(X2.sort_values("p").head(15).round(3).to_string(), flush=True)
json.dump({"X1": X1.round(4).to_dict("records"), "X2_top": X2.sort_values("p").head(15).round(4).to_dict("records"),
           "X2_bh_pass": X2[X2.bh_pass == True].round(4).to_dict("records")},                    # noqa: E712
          open(os.path.join(a.out, "explore16.json"), "w"), indent=1, default=str)
