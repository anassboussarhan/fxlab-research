"""Program-level analysis of the all-feature scan (scripts/feature_scan.py).

For each timeframe:
  * distribution of out-of-sample net Sharpe of real columns vs PLACEBO columns run through the same pipeline
  * one-sided t-test per (column, instrument) stream, Benjamini-Hochberg across ALL real streams of all
    timeframes, Holm (family-wise) as the strict version
  * Hansen SPA and White Reality Check per (instrument, timeframe) across all its real columns (H0: no column
    beats zero after costs)
  * cross-instrument version of every column (equal-weight average of its OOS daily returns on all instruments)
  * META walk-forward ("trailing winners"): every 6 months from 2013, take the K streams with the best OOS
    Sharpe over ALL PREVIOUS out-of-sample windows only, hold them equal-weighted for the next 6 months.
    This is the honest answer to "pick the best features and trade them" - the selection never sees the
    period it is scored on.
Outputs: results/reports/feature_scan/program_summary.json and per-timeframe CSVs."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg, holm, hansen_spa, white_reality_check, sharpe_ci  # noqa: E402

ROOT = RESULTS_DIR / "reports" / "feature_scan"


def sr(d):
    d = pd.Series(d).dropna()
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0


def meta_trailing(R: pd.DataFrame, k: int, start="2013-01-01", min_days=250):
    """R: daily OOS returns (columns = streams, NaN before a stream's first OOS day)."""
    t0 = pd.Timestamp(start)
    out, picks = [], []
    while t0 < R.index[-1]:
        t1 = t0 + pd.DateOffset(months=6)
        past = R[R.index < t0]
        ok = past.notna().sum() >= min_days
        score = (past.mean() / past.std() * np.sqrt(260))[ok].dropna()
        top = score.nlargest(k).index
        seg = R.loc[(R.index >= t0) & (R.index < t1), top].fillna(0.0)
        out.append(seg.mean(axis=1) if len(top) else pd.Series(0.0, index=seg.index))
        picks.append({"start": str(t0.date()), "top": list(top[:5]), "trailing_sharpe_top": float(score.max()) if len(score) else None})
        t0 = t1
    s = pd.concat(out)
    return s, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 20])
    a = ap.parse_args()
    S, R = [], {}
    for d in sorted(ROOT.glob("*_*")):
        if not (d / "summary.csv").exists():
            continue
        sym, tf = d.name.rsplit("_", 1)
        s = pd.read_csv(d / "summary.csv")
        s["symbol"], s["tf"] = sym, tf
        S.append(s)
        x = pd.read_parquet(d / "oos_daily.parquet")
        x.index = pd.to_datetime(x.index)
        R[(sym, tf)] = x
    S = pd.concat(S, ignore_index=True)
    S["p_one_sided"] = st.t.sf(S.t_stat, df=S.oos_days.clip(lower=2) - 1)
    real = S[~S.placebo].copy()
    real["p_bh"] = benjamini_hochberg(real.p_one_sided.to_numpy())
    real["p_holm"] = holm(real.p_one_sided.to_numpy())
    res = {"streams_real": int(len(real)), "streams_placebo": int(S.placebo.sum()),
           "instruments_by_tf": S.groupby("tf").symbol.nunique().to_dict(),
           "bh_significant_10pct": int((real.p_bh < 0.10).sum()), "bh_significant_5pct": int((real.p_bh < 0.05).sum()),
           "holm_significant_5pct": int((real.p_holm < 0.05).sum()), "by_tf": {}}
    for tf, g in S.groupby("tf"):
        rg, pg = g[~g.placebo], g[g.placebo]
        p95 = float(pg.oos_sharpe.quantile(0.95)) if len(pg) else np.nan
        row = {"real_columns": int(len(rg)), "real_median_sharpe": float(rg.oos_sharpe.median()),
               "real_share_positive": float((rg.oos_sharpe > 0).mean()), "real_best": float(rg.oos_sharpe.max()),
               "real_median_gross_sharpe": float(rg.oos_gross_sharpe.median()),
               "placebo_median_sharpe": float(pg.oos_sharpe.median()), "placebo_p95_sharpe": p95,
               "placebo_best": float(pg.oos_sharpe.max()),
               "share_real_above_placebo_p95": float((rg.oos_sharpe > p95).mean()),
               "median_share_folds_active": float(rg.share_folds_active.median())}
        # SPA / RC per instrument over its real columns
        spa = {}
        for (sym, t2), x in R.items():
            if t2 != tf:
                continue
            cols = [c for c in x.columns if not c.startswith("PLACEBO")]
            M = x[cols].dropna(how="all").fillna(0.0)
            M = M.loc[:, M.std() > 0]
            if M.shape[1] < 2:
                continue
            perf = M.to_numpy()                         # (T days, K columns)
            best = M.columns[int(np.argmax(M.mean() / M.std()))]
            spa[sym] = {"spa_p": hansen_spa(perf, n_boot=1000)["p_consistent"],
                        "rc_p": white_reality_check(perf, n_boot=1000)["p_value"],
                        "best_column": best, "best_sharpe": sr(M[best]), "K": int(M.shape[1])}
        row["spa_by_instrument"] = spa
        # cross-instrument average of each column
        xs = {}
        streams = {(sym, c): x[c] for (sym, t2), x in R.items() if t2 == tf for c in x.columns}
        cols = sorted({c for (_, c) in streams})
        for c in cols:
            parts = [v for (sym, cc), v in streams.items() if cc == c]
            if len(parts) >= 3:
                xs[c] = pd.concat(parts, axis=1).mean(axis=1)
        if xs:
            X = pd.DataFrame(xs)
            xsr = pd.Series({c: sr(X[c]) for c in X.columns})
            real_x = xsr[[c for c in xsr.index if not c.startswith("PLACEBO")]]
            plac_x = xsr[[c for c in xsr.index if c.startswith("PLACEBO")]]
            row["cross_instrument"] = {"columns": int(len(real_x)), "median_sharpe": float(real_x.median()),
                                       "share_positive": float((real_x > 0).mean()),
                                       "top5": real_x.nlargest(5).round(3).to_dict(),
                                       "placebo_best": float(plac_x.max()) if len(plac_x) else None}
            real_x.sort_values(ascending=False).to_csv(ROOT / f"cross_instrument_{tf}.csv")
        # meta walk-forward on real streams
        Rt = pd.concat({f"{sym}|{c}": x[c] for (sym, t2), x in R.items() if t2 == tf for c in x.columns
                        if not c.startswith("PLACEBO")}, axis=1).sort_index()
        meta = {}
        for k in a.k:
            s, picks = meta_trailing(Rt, k)
            _, lo, hi = sharpe_ci(s.to_numpy(), n_boot=1000)
            meta[f"top{k}"] = {"oos_sharpe": sr(s), "ci95": [lo, hi], "ann_return": float(s.mean() * 260),
                               "by_year": {int(y): float(v) for y, v in ((1 + s).groupby(s.index.year).prod() - 1).items()},
                               "first_picks": picks[:3], "last_picks": picks[-2:]}
        row["meta_trailing_winners"] = meta
        res["by_tf"][tf] = row
        g.sort_values("oos_sharpe", ascending=False).to_csv(ROOT / f"all_streams_{tf}.csv", index=False)
    top = real.sort_values("oos_sharpe", ascending=False).head(25)
    res["top25_streams"] = top[["column", "symbol", "tf", "oos_sharpe", "oos_gross_sharpe", "t_stat", "p_bh",
                                "turnover_per_year", "share_folds_active"]].round(4).to_dict("records")
    (ROOT / "program_summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "top25_streams"}, indent=1, default=str)[:6000])
    print(top[["column", "symbol", "tf", "oos_sharpe", "oos_gross_sharpe", "t_stat", "p_bh"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
