"""Selection rules and portfolio tests for scripts/random_entry_pool.py (run 15: random entries, pooled robust
choice of the mean-reversion harvest).

For every timeframe and every walk-forward fold (aligned across pairs by the half-year of the test start) a rule
picks candidates from the shared set using TRAINING-window information only; the pick then trades the 6-month test
window (expected-value stream = mean over 16 new random-entry seeds). Rules, declared before the run:
  apriori        candidate 0 (fixed grid), no selection at all
  per_pair       each pair takes its own best training candidate (the run 11-14 style)
  pooled         one candidate for all pairs: best mean training Sharpe across the pairs
  pooled_top10   the 10 best pooled candidates, averaged
  pooled_filter  the pooled candidate, traded only on pairs where it was profitable in training
  pooled_mr      pooled over, and traded on, the more mean-reverting half of the pairs (training VR(5) below the
                 cross-sectional median)
Candidates need >= 30 training trades per seed (pair level; pooled: on average). Tests: equal-weight portfolio
across the traded pairs, one-sided t of daily returns, Benjamini-Hochberg over all timeframe x rule tests at full
costs (primary) and separately at half costs. Also per-pair Sharpe of the expected stream, single-trader Sharpe
(from per-seed sums), stability of the pooled choice and year-by-year results.
Writes results/reports/random_entry_pool_summary/{rules.csv, pairs.csv, pooled_choices.csv, years.csv,
summary.json}."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg  # noqa: E402

IN = Path(os.environ.get("RP_IN", RESULTS_DIR / "reports" / "random_entry_pool"))
OUT = Path(os.environ.get("RP_OUT", RESULTS_DIR / "reports" / "random_entry_pool_summary"))
TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1", "TICK60", "VOLA60"]
RULES = ["apriori", "per_pair", "pooled", "pooled_top10", "pooled_filter", "pooled_mr"]
MIN_TRADES = 30


def half_year(ts):
    t = pd.Timestamp(ts)
    return f"{t.year}H{1 if t.month <= 6 else 2}"


def sharpe_t(x):
    x = pd.Series(x).dropna()
    if len(x) < 20 or x.std() == 0:
        return 0.0, 0.0
    return float(x.mean() / x.std() * np.sqrt(260)), float(x.mean() / x.std() * np.sqrt(len(x)))


def load(tf):
    cells = {}
    for f in sorted(glob.glob(str(IN / f"*_{tf}.json"))):
        J = json.load(open(f))
        if J.get("tf") != tf:
            continue
        base = f[:-5]
        if not os.path.exists(base + ".npz"):
            continue
        Z = np.load(base + ".npz")
        keys = [half_year(x["test_start"]) for x in J["folds"]]
        idx = pd.DatetimeIndex(Z["day"].astype("datetime64[ns]"), name="day")
        frames = {}
        for kind in ("net", "half", "gross"):
            X = pd.DataFrame(Z[f"{kind}_bp"].astype(np.float64) * 1e-4, index=idx,
                             columns=[str(i) for i in range(Z[f"{kind}_bp"].shape[1])])
            X["fold"] = Z["fold"]
            frames[kind] = X
        cells[J["pair"]] = {"keys": keys, "folds": J["folds"], "train": Z["train_sh"], "trades": Z["train_tr"],
                            "gross_sh": Z["gross_sh"], "ssum": Z["seed_sum"].astype(np.float64),
                            "ssq": Z["seed_sq"].astype(np.float64), "nd": Z["ndays"], **frames}
    return cells


def stream(cell, kind, picks):
    """Daily expected-value stream of one pair for a {fold index: [candidate ids]} selection (mean over the ids);
    folds not in picks are not traded (absent)."""
    D = cell[kind]
    parts = []
    for f, ids in picks.items():
        g = D[D["fold"] == f]
        if len(g) and ids:
            parts.append(g[[str(i) for i in ids]].mean(axis=1))
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)
    return s.groupby(level=0).sum()                       # a trading day split by a fold boundary


def single_trader(cell, picks):
    """Median over seeds of the whole-period Sharpe of one random trader following single-candidate picks."""
    S = Q = N = 0.0
    for f, ids in picks.items():
        if len(ids) != 1:
            return np.nan
        c = ids[0]
        S = S + cell["ssum"][c, f]; Q = Q + cell["ssq"][c, f]; N = N + float(cell["nd"][f])
    if not isinstance(S, np.ndarray) or N < 20:
        return np.nan
    m = S / N
    v = Q / N - m ** 2
    sh = np.where(v > 0, m / np.sqrt(np.maximum(v, 1e-30)) * np.sqrt(260), 0.0)
    return float(np.median(sh))


def select(cells):
    """{rule: {pair: {fold index: [candidate ids]}}} plus the pooled choices per half-year."""
    pairs = sorted(cells)
    nC = next(iter(cells.values()))["train"].shape[0]
    keys = sorted({k for p in pairs for k in cells[p]["keys"]})
    R = {r: {p: {} for p in pairs} for r in RULES}
    choices = []
    for k in keys:
        avail = [(p, cells[p]["keys"].index(k)) for p in pairs if k in cells[p]["keys"]]
        T = np.array([cells[p]["train"][:, f] for p, f in avail], float)            # pairs x candidates
        N = np.array([cells[p]["trades"][:, f] for p, f in avail], float)
        T_ok = np.where(N >= MIN_TRADES, T, np.nan)
        for (p, f), row in zip(avail, T_ok):
            R["apriori"][p][f] = [0]
            if np.isfinite(row).any():
                R["per_pair"][p][f] = [int(np.nanargmax(row))]

        def pooled(rows, nrows):
            m = np.nanmean(np.where(np.isfinite(rows), rows, np.nan), axis=0)
            m = np.where(np.nanmean(nrows, axis=0) >= MIN_TRADES, m, np.nan)
            return m
        m = pooled(T, N)
        if not np.isfinite(m).any():
            continue
        order = np.argsort(-np.nan_to_num(m, nan=-1e9))
        cstar = int(order[0])
        top = [int(x) for x in order[:10]]
        for (p, f), trow in zip(avail, T):
            R["pooled"][p][f] = [cstar]
            R["pooled_top10"][p][f] = top
            if trow[cstar] > 0:
                R["pooled_filter"][p][f] = [cstar]
        vr = np.array([cells[p]["folds"][f]["vr5"] for p, f in avail], float)
        med = np.nanmedian(vr)
        sel = np.isfinite(vr) & (vr < med)
        cmr = None
        if sel.sum() >= 3:
            mm = pooled(T[sel], N[sel])
            if np.isfinite(mm).any():
                cmr = int(np.nanargmax(mm))
                for (p, f), s_ in zip(avail, sel):
                    if s_:
                        R["pooled_mr"][p][f] = [cmr]
        choices.append({"half_year": k, "pairs": len(avail), "pooled": cstar, "pooled_train_mean": float(m[cstar]),
                        "pooled_mr": cmr, "top10": top})
    return R, choices, nC


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tfs = sorted({json.load(open(f))["tf"] for f in glob.glob(str(IN / "*.json"))},
                 key=lambda t: TF_ORDER.index(t) if t in TF_ORDER else 99)
    if not tfs:
        json.dump({"cells": 0}, open(OUT / "summary.json", "w"))
        print("no results")
        return
    from random_entry_pool import candidates
    CAND = candidates()
    rows, prow, crow, yrs = [], [], [], []
    for tf in tfs:
        cells = load(tf)
        if not cells:
            continue
        R, choices, nC = select(cells)
        for ch in choices:
            c = CAND[ch["pooled"]] if ch["pooled"] < len(CAND) else {}
            crow.append({"tf": tf, **{k: v for k, v in ch.items() if k != "top10"}, "family": c.get("fam"),
                         "gate": c.get("gate", {}).get("gate"), "params": json.dumps(c.get("P", {}))})
        for rule in RULES:
            per = {}
            for kind in ("net", "half", "gross"):
                S = {p: stream(cells[p], kind, R[rule][p]) for p in cells}
                S = {p: s for p, s in S.items() if len(s)}
                if not S:
                    continue
                M = pd.DataFrame(S).sort_index()
                port = M.mean(axis=1, skipna=True).dropna()
                sh, t = sharpe_t(port)
                per[kind] = (sh, t, port, M)
            if "net" not in per:
                continue
            sh, t, port, M = per["net"]
            pair_sh = {p: sharpe_t(M[p].dropna())[0] for p in M}
            st_med = np.nanmedian([single_trader(cells[p], R[rule][p]) for p in cells]) if rule not in ("pooled_top10",) else np.nan
            rows.append({"tf": tf, "rule": rule, "pairs": int(M.shape[1]), "days": int(len(port)),
                         "portfolio_sharpe": sh, "portfolio_t": t, "ann_return": float(port.mean() * 260),
                         "half_cost_sharpe": per.get("half", (np.nan, np.nan))[0],
                         "half_cost_t": per.get("half", (np.nan, np.nan))[1],
                         "median_pair_sharpe": float(np.median(list(pair_sh.values()))),
                         "share_pairs_positive": float(np.mean([v > 0 for v in pair_sh.values()])),
                         "median_single_trader": st_med,
                         "gross_sharpe": per.get("gross", (np.nan, np.nan))[0],
                         "gross_t": per.get("gross", (np.nan, np.nan))[1],
                         "ann_return_at_10pct_vol": 0.10 * sh,
                         "max_drawdown": float((port.cumsum() - port.cumsum().cummax().clip(lower=0)).min())})
            for p, v in pair_sh.items():
                prow.append({"tf": tf, "rule": rule, "pair": p, "sharpe": v})
            for y, g in port.groupby(port.index.year):
                yrs.append({"tf": tf, "rule": rule, "year": int(y), "return": float(g.sum()), "sharpe": sharpe_t(g)[0]})
    T = pd.DataFrame(rows)
    T["p"] = st.norm.sf(T.portfolio_t)
    T["p_bh"] = benjamini_hochberg(T.p.to_numpy())
    T["p_half"] = st.norm.sf(T.half_cost_t.fillna(0))
    T["p_half_bh"] = benjamini_hochberg(T.p_half.to_numpy())
    T.to_csv(OUT / "rules.csv", index=False)
    pd.DataFrame(prow).to_csv(OUT / "pairs.csv", index=False)
    PC = pd.DataFrame(crow)
    PC.to_csv(OUT / "pooled_choices.csv", index=False)
    pd.DataFrame(yrs).to_csv(OUT / "years.csv", index=False)
    stab = {}
    for tf, g in PC.groupby("tf"):
        vc = g.pooled.value_counts()
        stab[tf] = {"folds": int(len(g)), "distinct_pooled_choices": int(vc.size),
                    "most_common_share": float(vc.iloc[0] / len(g)),
                    "share_grid": float((g.family == "grid").mean())}
    best = T.sort_values("portfolio_t", ascending=False).head(5)
    summ = {"tests": int(len(T)), "tfs": tfs, "rules": RULES, "bh_survivors_q10": int((T.p_bh < 0.10).sum()),
            "bh_survivors_half_cost_q10": int((T.p_half_bh < 0.10).sum()),
            "best_by_t": best[["tf", "rule", "portfolio_sharpe", "portfolio_t", "p", "p_bh", "half_cost_sharpe",
                               "half_cost_t", "median_pair_sharpe", "share_pairs_positive"]].to_dict("records"),
            "pooled_choice_stability": stab}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
