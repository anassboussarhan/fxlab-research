"""Campaign-level scientific analysis (Phase 10 statistics). Reads the results DB and per-experiment
out-of-sample artifacts and writes results/reports/<name>/*.csv|json plus a Markdown summary.

Multiple-testing discipline:
  * Deflated Sharpe uses N = number of strategy configurations tried in the campaign (all candidates of all
    families/instruments) and the cross-sectional variance of their OOS Sharpe ratios;
  * White's Reality Check and Hansen's SPA test the best OOS strategy against a flat benchmark, across ALL
    experiments in the campaign simultaneously;
  * per-experiment one-sided t-test p-values are adjusted with Benjamini-Hochberg and Holm;
  * speculative (esoteric) features/strategies are flagged and never promoted without passing the same bar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.validation import stats  # noqa: E402


def load(name):
    con = connect()
    df = con.execute("SELECT id, kind, target, symbol, tf, status, seconds, error, metrics, artifacts, config "
                     "FROM experiments WHERE campaign = ?", [name]).fetchdf()
    con.close()
    df["metrics"] = df["metrics"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    df["artifacts"] = df["artifacts"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    return df


def strategy_report(name, df, out):
    ok = df[df.status == "ok"].copy()
    m = pd.json_normalize(ok["metrics"]).set_index(ok.index)
    keep = ["sharpe", "sortino", "calmar", "ann_return", "max_drawdown", "trades", "trades_per_year", "win_rate",
            "profit_factor", "expectancy", "payoff_ratio", "cvar_5_daily", "psr_0", "ttest_p", "param_stability",
            "stress.cost_x2.0.sharpe", "stress.cost_x3.0.sharpe", "stress.latency1.sharpe",
            "diagnostic_zero_cost_default_sharpe", "in_sample_default.sharpe", "n_candidates", "runtime_s"]
    tab = pd.concat([ok[["id", "target", "symbol", "tf"]], m[[c for c in keep if c in m]]], axis=1)
    tab["family"] = tab["target"].str.split(".").str[1]
    # daily OOS returns matrix for data-snooping tests
    rets = {}
    for _, r in ok.iterrows():
        p = r["artifacts"].get("oos_returns")
        if p and os.path.exists(p):
            s = pd.read_parquet(p)["ret"]
            from fxlab.validation.metrics import daily
            rets[r["id"]] = daily(s)
    R = pd.DataFrame(rets)
    res = {"n_ok": int(len(ok)), "n_failed": int((df.status == "failed").sum())}
    if len(R.columns) >= 2:
        Rf = R.fillna(0.0)
        n_trials = int(tab["n_candidates"].fillna(1).sum())
        srs = Rf.mean() / Rf.std()
        var_sr = float(srs.var())
        tab["dsr"] = [stats.dsr(Rf[i].to_numpy(), n_trials, var_sr) if i in Rf else np.nan for i in tab["id"]]
        res["n_trials_for_dsr"] = n_trials
        res["white_rc"] = stats.white_reality_check(Rf.to_numpy(), n_boot=1000)
        res["hansen_spa"] = stats.hansen_spa(Rf.to_numpy(), n_boot=1000)
        res["white_rc"]["best_id"] = R.columns[res["white_rc"]["best"]]
        p = tab["ttest_p"].fillna(1).to_numpy()
        tab["p_bh"] = stats.benjamini_hochberg(p)
        tab["p_holm"] = stats.holm(p)
    tab = tab.sort_values("sharpe", ascending=False)
    tab.to_csv(out / "strategy_results.csv", index=False)
    fam = tab.groupby("family").agg(n=("id", "count"), median_sharpe=("sharpe", "median"),
                                    best_sharpe=("sharpe", "max"), share_positive=("sharpe", lambda s: (s > 0).mean()),
                                    median_x2cost_sharpe=("stress.cost_x2.0.sharpe", "median"))
    fam.to_csv(out / "family_comparison.csv")
    inst = tab.groupby("symbol").agg(n=("id", "count"), median_sharpe=("sharpe", "median"), best_sharpe=("sharpe", "max"),
                                     share_positive=("sharpe", lambda s: (s > 0).mean()))
    inst.to_csv(out / "instrument_comparison.csv")
    tfc = tab.groupby("tf").agg(n=("id", "count"), median_sharpe=("sharpe", "median"), share_positive=("sharpe", lambda s: (s > 0).mean()))
    tfc.to_csv(out / "timeframe_comparison.csv")
    cost = tab[["id", "target", "symbol", "sharpe", "stress.cost_x2.0.sharpe", "stress.cost_x3.0.sharpe",
                "stress.latency1.sharpe", "diagnostic_zero_cost_default_sharpe"]]
    cost.to_csv(out / "cost_sensitivity.csv", index=False)
    # regime / session / year robustness (from per-experiment breakdowns)
    reg_rows = []
    for _, r in ok.iterrows():
        b = r["metrics"].get("breakdowns", {})
        row = {"id": r["id"], "target": r["target"], "symbol": r["symbol"]}
        if isinstance(b.get("by_vol_regime"), dict):
            for k, v in b["by_vol_regime"].items():
                row[f"vol:{k}"] = v.get("sum_ret")
        for k, v in (b.get("by_session") or {}).items():
            row[f"session:{k}"] = v
        yrs = b.get("by_year") or {}
        if yrs:
            row["years_positive_share"] = float(np.mean([v > 0 for v in yrs.values()]))
        reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(out / "regime_robustness.csv", index=False)
    _charts(tab, ok, out)
    # survivors: strict, pre-declared bar
    cond = (tab["sharpe"] > 0) & (tab.get("p_bh", 1) < 0.10) & (tab.get("dsr", 0) > 0.95) & \
           (tab["stress.cost_x2.0.sharpe"] > 0)
    surv = tab[cond]
    surv.to_csv(out / "survivors.csv", index=False)
    res["survivors"] = surv[["id", "target", "symbol", "tf", "sharpe"]].to_dict("records")
    res["families"] = fam.reset_index().to_dict("records")
    return tab, res


def _charts(tab, ok, out, top=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = out / "charts"
    ch.mkdir(exist_ok=True)
    arts = dict(zip(ok["id"], ok["artifacts"]))
    for _, r in tab.head(top).iterrows():
        p = arts.get(r["id"], {}).get("oos_returns")
        if not p or not os.path.exists(p):
            continue
        x = pd.read_parquet(p)["ret"]
        eq = (1 + x).cumprod()
        dd = eq / eq.cummax() - 1
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(eq.index, eq.values, lw=1)
        ax[0].set_title(f"{r['target']} {r['symbol']} {r['tf']} - OOS walk-forward equity (net of costs)", fontsize=9)
        ax[1].fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.5)
        ax[1].set_ylabel("drawdown")
        fig.tight_layout()
        fig.savefig(ch / f"{r['target']}_{r['symbol']}_{r['tf']}.png", dpi=90)
        plt.close(fig)


def model_report(name, df, out):
    ok = df[df.status == "ok"].copy()
    cfg = ok["config"].apply(json.loads)
    m = pd.json_normalize(ok["metrics"]).set_index(ok.index)
    tab = pd.concat([ok[["id", "target", "symbol", "tf"]], cfg.apply(pd.Series)[["task", "h"]],
                     m[[c for c in ("auc", "pr_auc", "brier", "ece", "accuracy", "base_rate", "log_loss",
                                    "sign_ret_mean_bps", "ic_spearman", "r2_oos", "rmse", "hit_rate", "n", "n_folds",
                                    "runtime_s") if c in m]]], axis=1)
    tab["family"] = tab["target"].str.split(".").str[1]
    tab = tab.sort_values(["task", "h", "auc" if "auc" in tab else "ic_spearman"], ascending=[True, True, False])
    tab.to_csv(out / "model_results.csv", index=False)
    res = {"n_ok": int(len(ok)), "n_failed": int((df.status == "failed").sum())}
    if "auc" in tab:
        # is the best AUC distinguishable from 0.5? binomial-approx SE per model, Holm across models
        se = np.sqrt(tab["auc"] * (1 - tab["auc"]) / tab["n"].clip(lower=1) * 2)
        z = (tab["auc"] - 0.5) / se
        from scipy.stats import norm
        p = 1 - norm.cdf(z.fillna(0))
        tab["p_auc_gt_half"] = p
        tab["p_holm"] = stats.holm(p)
        tab.to_csv(out / "model_results.csv", index=False)
        res["horizon_comparison"] = tab.groupby("h")["auc"].agg(["median", "max"]).reset_index().to_dict("records")
        res["family_comparison"] = tab.groupby("family")["auc"].agg(["median", "max"]).reset_index().to_dict("records")
    return tab, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = RESULTS_DIR / "reports" / a.name
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.name)
    failed = df[df.status == "failed"][["id", "target", "symbol", "tf", "error"]]
    failed.to_csv(out / "failures.csv", index=False)
    kinds = set(df["kind"])
    res = {"campaign": a.name, "experiments": int(len(df)), "failed": int(len(failed))}
    if "strategy" in kinds:
        _, r = strategy_report(a.name, df[df.kind == "strategy"], out)
        res["strategy"] = r
    if "model" in kinds:
        _, r = model_report(a.name, df[df.kind == "model"], out)
        res["model"] = r
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
