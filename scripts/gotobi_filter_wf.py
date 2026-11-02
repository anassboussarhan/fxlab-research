"""Follow-up to scripts/gotobi_accuracy_lab.py (run 41427843322): the logistic "winning night" filter, examined.
Reads the lab's published outputs (features_basket.csv, trades_basket_*.csv, features.csv); development data only.

1. The pre-registered model (logistic regression, L2, C = 0.1, the lab's feature set and preprocessing) refitted
   on 2007-2018 exactly as in the lab: its standardised coefficients, and its 2019-2025 numbers (a reproduction).
2. Walk-forward (not pre-registered; a robustness check of the same specification): for each year 2012-2025, fit
   on all earlier trades, predict that year, skip the nights in the lowest tercile of the training predictions.
3. A reduced model (not pre-registered): only the features with q <= 0.10 on 2007-2018 in the lab (chosen on train
   only), same fitting; reported on 2019-2025 and walk-forward.
4. Prop-firm and own-account simulations of the walk-forward filtered basket with the 100 bp stop against the
   unfiltered one, 2012-2025 trades, at the observed edge and with every night shifted down so that the unfiltered
   mean is the 2026 no-stop mean (+2.04 bp); for the filtered series also a scenario in which its mean shrinks in
   the same proportion (the filter's gain shrinking with the edge).
Usage: python scripts/gotobi_filter_wf.py --dir LAB_OUT --out DIR [--n 4000]"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TRAIN_END = pd.Timestamp("2018-12-31")


def load(d, name):
    f = pd.read_csv(os.path.join(d, name), index_col=0, parse_dates=True)
    f.index = pd.DatetimeIndex(f.index).normalize()
    return f.sort_index()


def fit_predict(X, y, fit_mask, pred_mask, cols=None):
    from sklearn.linear_model import LogisticRegression
    Xf = X[fit_mask]
    keep = cols if cols is not None else [c for c in X.columns if Xf[c].notna().mean() > 0.8]
    med, mu = Xf[keep].median(), Xf[keep].mean()
    sd = Xf[keep].std().replace(0, 1.0)
    Z = ((X[keep].fillna(med) - mu) / sd)
    m = LogisticRegression(C=0.1, max_iter=2000)
    yb = (y > 0).astype(int)
    m.fit(Z[fit_mask].to_numpy(), yb[fit_mask].to_numpy())
    p_fit = m.predict_proba(Z[fit_mask].to_numpy())[:, 1]
    p = pd.Series(m.predict_proba(Z[pred_mask].to_numpy())[:, 1], index=X.index[pred_mask])
    return p, np.quantile(p_fit, 1 / 3), pd.Series(m.coef_[0], index=keep)


def st(x, n_days=None, mae=None):
    x = x.dropna()
    out = {"n": int(len(x)), "mean_bp": x.mean() * 1e4, "t": float(x.mean() / x.std() * np.sqrt(len(x))),
           "win": float((x > 0).mean()), "worst_bp": x.min() * 1e4}
    if n_days:
        out["per_day_bp"] = x.sum() / n_days * 1e4
    if mae is not None:
        m = mae.reindex(x.index)
        q = m.quantile(0.05)
        out["cvar5_bp"] = m[m <= q].mean() * 1e4
    return out


def walk_forward(X, y, cols=None, first=2012):
    keep_all, pred = [], []
    for Y in range(first, int(y.index.year.max()) + 1):
        fit = np.asarray(y.index < pd.Timestamp(f"{Y}-01-01"))
        prd = np.asarray(y.index.year == Y)
        if prd.sum() == 0:
            continue
        p, c1, _ = fit_predict(X, y, fit, prd, cols)
        pred.append(pd.DataFrame({"p": p, "keep": p > c1}))
    return pd.concat(pred).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = load(a.dir, "features_basket.csv")
    T = {v: load(a.dir, f"trades_basket_{v}.csv") for v in ("none", "stop_100_bp")}
    y = T["none"].net.reindex(X.index)
    ok = y.notna()
    X, y = X[ok], y[ok]
    res = {}
    # 1. the pre-registered fit
    tr, te = (X.index <= TRAIN_END), (X.index > TRAIN_END)
    p, c1, coef = fit_predict(X, y, tr, te)
    from sklearn.metrics import roc_auc_score
    res["prereg"] = {"test_auc": roc_auc_score((y[te] > 0).astype(int), p), "test_all": st(y[te], te.sum()),
                     "test_kept": st(y[te][p > c1], te.sum()), "test_skipped": st(y[te][p <= c1]),
                     "coef_std": coef.sort_values().round(3).to_dict()}
    # 3. reduced model
    F = pd.read_csv(os.path.join(a.dir, "features.csv"))
    red = F[(F.series == "basket") & (F.train_q_bh <= 0.10)].feature.tolist()
    pr, c1r, coefr = fit_predict(X, y, tr, te, red)
    res["reduced"] = {"features": red, "test_auc": roc_auc_score((y[te] > 0).astype(int), pr),
                      "test_kept": st(y[te][pr > c1r], te.sum()), "test_skipped": st(y[te][pr <= c1r]),
                      "coef_std": coefr.sort_values().round(3).to_dict()}
    # 2. walk-forward, both specifications
    W = {"full": walk_forward(X, y), "reduced": walk_forward(X, y, red)}
    yrs = []
    for k, w in W.items():
        yy = y.reindex(w.index)
        for per, msk in (("2012-2025", np.ones(len(w), bool)), ("2012-2018", w.index <= TRAIN_END),
                         ("2019-2025", w.index > TRAIN_END)):
            res[f"wf_{k}_{per}"] = {"all": st(yy[msk], msk.sum()), "kept": st(yy[msk][w.keep[msk]], msk.sum()),
                                    "skipped": st(yy[msk][~w.keep[msk]]),
                                    "auc": roc_auc_score((yy[msk] > 0).astype(int), w.p[msk])}
        g = pd.DataFrame({"y": yy, "keep": w.keep})
        for Y, h in g.groupby(g.index.year):
            yrs.append({"model": k, "year": Y, "all_bp": h.y.mean() * 1e4, "kept_bp": h.y[h.keep].mean() * 1e4,
                        "skipped_bp": h.y[~h.keep].mean() * 1e4, "share_kept": h.keep.mean(),
                        "win_all": (h.y > 0).mean(), "win_kept": (h.y[h.keep] > 0).mean()})
    pd.DataFrame(yrs).to_csv(os.path.join(a.out, "walk_forward_years.csv"), index=False)
    # 4. simulations, walk-forward full model, with the 100 bp stop
    import final_sims as FS
    import gotobi_opt_sims as GS
    import prop_firms_ev as PF
    keep = W["full"].keep
    S = T["stop_100_bp"]
    S = S[S.index >= keep.index.min()]
    filt = S[S.index.isin(keep.index[keep])]
    shift = S.net.mean() - FS.TARGET_2026
    res["sim_inputs"] = {"unfiltered": st(S.net, len(S), S.mae), "filtered": st(filt.net, len(S), filt.mae),
                         "shift_bp": shift * 1e4}
    rows, acc = [], []
    ratio = FS.TARGET_2026 / S.net.mean()
    for name, g in (("unfiltered, stop 100 bp", S), ("filtered, stop 100 bp", filt)):
        scen_list = [("observed", g), ("2026 edge", FS.shifted(g, g.net.mean() - shift))]
        if g is filt:                                    # the filter's gain shrinks in proportion to the edge
            scen_list.append(("2026 edge, filter gain scaled", FS.shifted(g, g.net.mean() * ratio)))
        for scen, h in scen_list:
            Xd, Md = FS.daily(h)
            span = (S.index.max() - S.index.min()).days / 365.25
            for firm, rules in FS.FIRMS.items():
                for lp, lf in itertools.product((3, 5, 8), (3, 5, 8)):
                    r = PF.simulate(Xd, rules, n=a.n, seed=1, M=Md, scale_eval=lp, scale_funded=lf)
                    rows.append({"series": name, "scenario": scen, "firm": firm, "lev_pass": lp, "lev_funded": lf, **r})
            for L in (10, 15, 20, 30, 40):
                acc.append({"series": name, "scenario": scen, **GS.account(h.net.to_numpy(), h.mae.to_numpy(), L, 3,
                                                                           len(h) / span)})
            print(name, scen, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "prop.csv"), index=False)
    pd.DataFrame(acc).to_csv(os.path.join(a.out, "account.csv"), index=False)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
