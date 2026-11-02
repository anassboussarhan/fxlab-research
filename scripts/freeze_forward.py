"""Freeze the winning-night filter for the forward test (November 2026 on).

The model is the pre-registered specification of scripts/gotobi_accuracy_lab.py (logistic regression, L2, C = 0.1;
features kept when more than 80% present in the fit sample; missing values filled with the fit medians; standardised
with the fit means and standard deviations), now fitted once on every development night, 2007-2025 (USD/JPY + EUR/JPY
basket, no stop, net > 0 as the target). A night is kept when its predicted probability exceeds the lowest-tercile cut
of the fitted probabilities. The frozen parameters are written to JSON; `score` applies them to new feature rows, so
the forward test needs no refitting.
Usage: python scripts/freeze_forward.py --features docs/gotobi_accuracy/features_basket.csv \
           --trades docs/gotobi_accuracy/trades_basket_none.csv --out results/frozen_forward_v1.json
       python scripts/freeze_forward.py --check results/frozen_forward_v1.json --features ... --trades ..."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

END = pd.Timestamp("2025-12-31")


def load(features, trades, drop_prefix=""):
    X = pd.read_csv(features, index_col=0, parse_dates=True)
    if drop_prefix:
        X = X[[c for c in X.columns if not c.startswith(drop_prefix)]]
    T = pd.read_csv(trades, index_col=0, parse_dates=True)
    X.index, T.index = pd.DatetimeIndex(X.index).normalize(), pd.DatetimeIndex(T.index).normalize()
    y = T.net.reindex(X.index)
    ok = y.notna() & (X.index <= END)
    return X[ok], y[ok]


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    keep = [c for c in X.columns if X[c].notna().mean() > 0.8]
    med, mu = X[keep].median(), X[keep].mean()
    sd = X[keep].std().replace(0, 1.0)
    Z = (X[keep].fillna(med) - mu) / sd
    m = LogisticRegression(C=0.1, max_iter=2000).fit(Z.to_numpy(), (y > 0).astype(int).to_numpy())
    p = m.predict_proba(Z.to_numpy())[:, 1]
    return {"features": keep, "median": med.to_dict(), "mean": mu.to_dict(), "sd": sd.to_dict(),
            "coef": dict(zip(keep, m.coef_[0].tolist())), "intercept": float(m.intercept_[0]),
            "threshold": float(np.quantile(p, 1 / 3))}, p


def score(X, F):
    """Probability of a winning night and the keep decision for feature rows X (columns as in the lab)."""
    cols = F["features"]
    Z = (X.reindex(columns=cols).fillna(pd.Series(F["median"])) - pd.Series(F["mean"])) / pd.Series(F["sd"])
    lin = Z[cols].to_numpy() @ np.array([F["coef"][c] for c in cols]) + F["intercept"]
    p = 1.0 / (1.0 + np.exp(-lin))
    return pd.DataFrame({"p_win": p, "keep": p > F["threshold"]}, index=X.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="")
    ap.add_argument("--version", default="forward-v1")
    ap.add_argument("--feature-code", default="scripts/gotobi_accuracy_lab.py, function features(), commit b90884b")
    ap.add_argument("--drop-prefix", default="", help="columns not known at entry (forward-v2: night_)")
    a = ap.parse_args()
    X, y = load(a.features, a.trades, a.drop_prefix)
    if a.check:
        F = json.load(open(a.check))
        s = score(X, F)
        print("nights", len(s), "kept", int(s.keep.sum()), "kept mean bp", round(y[s.keep].mean() * 1e4, 2),
              "skipped mean bp", round(y[~s.keep].mean() * 1e4, 2))
        return
    F, p = fit(X, y)
    s = score(X, F)
    assert np.allclose(s.p_win.to_numpy(), p, atol=1e-9), "score() does not reproduce the fit"
    F.update({"version": a.version, "fitted_on": f"{X.index.min().date()} to {X.index.max().date()}",
              "n_fit": int(len(X)), "target": "basket no-stop net > 0",
              "feature_code": a.feature_code,
              "fit_sample_kept": {"n": int(s.keep.sum()), "mean_bp": float(y[s.keep].mean() * 1e4),
                                  "skipped_mean_bp": float(y[~s.keep].mean() * 1e4)}})
    txt = json.dumps(F, indent=1, sort_keys=True)
    open(a.out, "w").write(txt + "\n")
    print(txt[:1500])
    print("sha256", hashlib.sha256((txt + "\n").encode()).hexdigest())


if __name__ == "__main__":
    main()
