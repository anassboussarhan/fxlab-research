"""Round 17 exploratory follow-ups (written after the pre-registered results; nothing here counts). The closest
miss was the daily single-feature split on 5-day cross-sectional dispersion (DISP5, terciles) in both book tiers:
- what it holds in each regime (last refit), and how often it picks each timeframe / family on 2019-2025;
- the same setting with the feature lagged 1 and 2 extra days (an alignment check: a genuine effect should not
  vanish with one more day of lag, a look-ahead would);
- turnover and the cost drag; returns by year."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regime17 as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
res = {}
for tier, p, k in (("BOOK-CFG", {"f": "DISP5", "K": 3, "n0": 0, "window": "exp"}, 10),
                   ("BOOK-UNIT", {"f": "DISP5", "K": 3, "n0": 252, "window": "exp"}, 5)):
    D = G.load(a.data, tier, "D")
    t = D.tier
    out = {}
    for lag in (0, 1, 2):
        Dl = D.rolled(lag) if lag else D
        if lag:
            Dl.FV[:lag] = np.nan
        Sc = G.scores_for(Dl, t, "M3", p, G.Cache())
        R = G.evaluate(Dl, t, Sc, (k,))[k]
        st = G.stats_of(D, R)
        out[f"lag+{lag}"] = {kk: round(v, 3) for kk, v in st.items()}
        if lag == 0:
            # weights, turnover, picks
            W = np.zeros_like(t.z)
            for j in range(D.j0, D.T):
                s = Sc[j]
                ok = np.flatnonzero(np.isfinite(s) & t.elig[j] & (s > 0))
                if len(ok) > k:
                    ok = ok[np.argpartition(-s[ok], k - 1)[:k]]
                W[j, ok] = 1.0 / k
            turn = np.abs(np.diff(W, axis=0)).sum(1)
            gross = (np.nan_to_num(t.z) * W).sum(1)
            chk = D.mask_check
            out["turnover_per_day_check"] = round(float(turn[chk[1:]].mean()), 3)
            out["gross_mean_check"] = round(float(np.nanmean(gross[chk])), 4)
            out["cost_mean_check"] = round(float(G.FREQ["D"]["cost"] * turn[chk[1:]].mean()), 4)
            desc = t.desc
            tf = desc.unit.str.split("|").str[-1].to_numpy()
            fam = desc.family.to_numpy()
            wc = W[chk].sum(0)
            out["weight_by_tf_check"] = pd.Series(wc, index=tf).groupby(level=0).sum().div(wc.sum()).round(3).to_dict()
            out["weight_by_family_check"] = pd.Series(wc, index=fam).groupby(level=0).sum().div(wc.sum()).round(3).nlargest(10).to_dict()
            out["top_held_check"] = pd.Series(wc, index=t.names).nlargest(15).div(wc.sum()).round(3).to_dict()
            yr = pd.Series(R, index=D.ends.year)
            out["by_year_mean"] = yr[D.ends.year >= 2011].groupby(level=0).mean().round(4).to_dict()
            out["by_year_sharpe"] = yr[D.ends.year >= 2011].groupby(level=0).apply(lambda x: G.sharpe(x.to_numpy(), 252)).round(2).to_dict()
            # regime-wise mean of R on 2019-2025 (by the decision regime)
            per = G.split_periods(D, D.FV[:, D.fcol["DISP5"]], 3)
            lab = np.full(D.T, -1)
            for s_, e_, Gm, P in per:
                if P is not None:
                    lab[s_:e_] = np.nan_to_num(np.argmax(np.nan_to_num(P[s_:e_, :, 0]), axis=1), nan=-1)
            out["check_mean_by_regime"] = {int(r): round(float(np.nanmean(R[chk & (lab == r)])), 4) for r in range(3)}
            out["check_share_by_regime"] = {int(r): round(float(np.mean(lab[chk] == r)), 3) for r in range(3)}
            s_, e_, Gm, P = per[-1]
            x = D.FV[:, D.fcol["DISP5"]]
            out["last_refit_cuts"] = [round(float(c), 3) for c in np.quantile(x[:s_][np.isfinite(x[:s_])], [1 / 3, 2 / 3])]
            top = {}
            for r in range(3):
                w = Gm[:s_, r, 0]
                fin = np.isfinite(t.zw[:s_])
                nr = w @ fin
                with np.errstate(invalid="ignore", divide="ignore"):
                    mr = (w @ np.nan_to_num(t.zw[:s_])) / nr
                mr = np.where(nr >= 126, mr, np.nan)
                o = np.argsort(-np.nan_to_num(mr, nan=-9))[:k]
                top[r] = [(t.names[i], round(float(mr[i]), 3)) for i in o]
            out["last_refit_top_by_regime"] = top
    res[tier] = out
    print(tier, json.dumps({kk: v for kk, v in out.items() if kk.startswith("lag") or kk.startswith("check") or kk.startswith("turn")}), flush=True)
json.dump(res, open(os.path.join(a.out, "explore17.json"), "w"), indent=1, default=str)
