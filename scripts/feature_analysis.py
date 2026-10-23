"""Feature relevance, stability, redundancy and cost analysis on development data.

For every feature output column: per-year Spearman IC with forward h-bar mid returns (h = 1, 5, 20),
mean IC, t-stat across years, sign stability, mutual information with the forward direction, and a
comparison with PLACEBO features (pure noise and time-shuffled copies of real features) that go
through exactly the same pipeline. p-values are BH-adjusted across all columns x horizons.
Redundancy: Spearman correlation clustering at |rho| >= 0.9. Cost: seconds from results/benchmark.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

import fxlab.features  # noqa: E402,F401
from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.registry import by_kind, CATALOG  # noqa: E402
from fxlab.feature_store import build_matrix  # noqa: E402
from fxlab.labels import forward_mid_return  # noqa: E402
from fxlab.orchestration.runners import cached_bars, load_panel_closes  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg  # noqa: E402

PANEL = ["GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "XAUUSD", "XAGUSD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2018-12-31")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    out = RESULTS_DIR / "reports" / f"features_{a.symbol}_{a.tf}"
    out.mkdir(parents=True, exist_ok=True)
    bars = cached_bars(a.symbol, a.tf, a.start, a.end)
    panel = load_panel_closes([p for p in PANEL if p != a.symbol], a.tf, bars.index, a.start, a.end)
    fids = [e.id for e in sorted(by_kind("feature"), key=lambda e: e.id) if e.id not in a.skip]
    X, errors = build_matrix(bars, fids, a.symbol, a.tf, panel)
    # placebo features: noise and time-shuffled real features (same NaN pattern)
    rng = np.random.default_rng(0)
    for k in range(20):
        X[f"PLACEBO.noise:{k}"] = rng.normal(size=len(X)).astype("float32")
    for k, c in enumerate(rng.choice(X.columns[:-20], 20, replace=False)):
        v = X[c].to_numpy().copy()
        ok = np.isfinite(v)
        v[ok] = rng.permutation(v[ok])
        X[f"PLACEBO.shuffled:{k}"] = v
    rows = []
    years = bars.index.year
    for h in (1, 5, 20):
        y, _ = forward_mid_return(bars, h)
        yv = y.to_numpy()
        for c in X.columns:
            x = X[c].to_numpy(dtype=float)
            ics = []
            for Y in np.unique(years):
                m = (years == Y) & np.isfinite(x) & np.isfinite(yv)
                if m.sum() > 200 and np.nanstd(x[m]) > 0:
                    ics.append(st.spearmanr(x[m][:: max(1, h)], yv[m][:: max(1, h)]).statistic)  # non-overlapping
            ics = np.array([v for v in ics if np.isfinite(v)])
            if len(ics) < 3:
                continue
            t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std(ddof=1) > 0 else 0.0
            p = 2 * (1 - st.t.cdf(abs(t), len(ics) - 1))
            rows.append({"column": c, "feature": c.split(":")[0], "h": h, "mean_ic": ics.mean(), "ic_t": t, "p": p,
                         "years": len(ics), "sign_stability": float((np.sign(ics) == np.sign(ics.mean())).mean()),
                         "placebo": c.startswith("PLACEBO")})
    R = pd.DataFrame(rows)
    R["p_bh"] = benjamini_hochberg(R["p"].to_numpy())
    fam = {e.id: e.family for e in CATALOG.values()}
    R["family"] = R["feature"].map(fam).fillna("placebo")
    R["speculative"] = R["feature"].map(lambda f: "speculative" in CATALOG[f].tags if f in CATALOG else False)
    plac = R[R.placebo]
    R["beats_placebo_p95"] = R.apply(lambda r: abs(r["ic_t"]) > plac[plac.h == r["h"]]["ic_t"].abs().quantile(0.95), axis=1)
    R.sort_values("p").to_csv(out / "relevance.csv", index=False)
    # redundancy
    real = [c for c in X.columns if not c.startswith("PLACEBO")]
    S = X[real].iloc[::5].rank().corr(method="pearson", min_periods=500).abs()
    pairs = [(i, j, S.loc[i, j]) for ii, i in enumerate(real) for j in real[ii + 1:] if S.loc[i, j] >= 0.9]
    pd.DataFrame(pairs, columns=["a", "b", "abs_spearman"]).to_csv(out / "redundant_pairs.csv", index=False)
    bm = json.loads((RESULTS_DIR / "benchmark.json").read_text()) if (RESULTS_DIR / "benchmark.json").exists() else {}
    cost = bm.get("features", {}).get("per_feature", {})
    summ = {"symbol": a.symbol, "tf": a.tf, "bars": len(bars), "feature_columns": len(real), "feature_errors": errors,
            "significant_bh10": int(((R.p_bh < 0.10) & ~R.placebo).sum()),
            "placebo_significant_bh10": int(((R.p_bh < 0.10) & R.placebo).sum()),
            "top": R[~R.placebo].sort_values("p").head(25)[["column", "h", "mean_ic", "ic_t", "p_bh", "sign_stability",
                                                             "beats_placebo_p95"]].to_dict("records"),
            "redundant_pairs_rho_ge_0.9": len(pairs),
            "slowest_features_s": sorted(((v, k) for k, v in cost.items() if isinstance(v, (int, float))), reverse=True)[:10]}
    by_fam = R[~R.placebo].groupby(["family", "h"])["ic_t"].apply(lambda s: s.abs().median()).unstack()
    by_fam.to_csv(out / "family_relevance.csv")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
