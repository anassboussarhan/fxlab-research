"""Report of the JOINT optimisation campaigns (strategy parameters x entry type x exit chosen per fold, per pair).

For every joint experiment (config.joint): OOS net Sharpe, return, drawdown, t-test p, 2x-cost Sharpe, the two
abstention diagnostics (best training Sharpe <= 0 / <= deflation benchmark -> flat), the number of distinct
candidates simulated, the configuration carried forward (best on the most recent 3-year dev window) and how often
each entry type / exit configuration was selected across folds. Compared with the standard walk-forward of the
same <family, pair, timeframe> (default entry, family exit, parameter search only) where that exists.
BH / Holm over the joint streams only; the program-wide correction is in program_report.py.
Output: results/reports/joint/{streams.csv, choice_frequency.csv, by_family.csv, by_pair.csv, summary.json}"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg, holm  # noqa: E402

OUT = RESULTS_DIR / "reports" / "joint"


def label(x):
    if x is None:
        return "default"
    if "entry" in x:
        return f"{x['entry']}" + (f"(k={x.get('k')},ttl={x.get('ttl')})" if x["entry"] != "market" else "")
    if "type" in x:
        return x["type"] + "".join(f",{k}={v}" for k, v in x.items() if k != "type")
    s = f"{x.get('stop')}{x.get('stop_params', {})}->{x.get('target')}{x.get('target_params', {})}"
    for k in ("trail_atr", "max_hold", "breakeven_r", "partial_r", "sizing"):
        if k in x:
            s += f",{k}={x[k]}"
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect(read_only=True)
    rows = con.execute("SELECT id, campaign, target, symbol, tf, config, metrics FROM experiments "
                       "WHERE status='ok' AND kind='strategy'").fetchall()
    con.close()
    J, base = [], {}
    choice = collections.Counter()
    for eid, camp, tgt, sym, tf, cfg, m in rows:
        cfg, m = json.loads(cfg), json.loads(m)
        if not cfg.get("joint"):
            if not cfg.get("exec_grid") and not cfg.get("exec") and not cfg.get("risk_grid"):
                k = (tgt, sym, tf)
                if m.get("sharpe") is not None:
                    base.setdefault(k, []).append(m["sharpe"])
            continue
        x2 = (m.get("stress") or {}).get("cost_x2.0") or {}
        lc = m.get("latest_choice") or {}
        av, dv = m.get("abstain_variant") or {}, m.get("deflated_variant") or {}
        for kind in ("exec", "risk"):
            for k, v in ((m.get("choice_counts") or {}).get(kind) or {}).items():
                choice[(kind, label(json.loads(k)))] += v
        J.append({"id": eid, "family": tgt, "symbol": sym, "tf": tf, "sharpe": m.get("sharpe"),
                  "ann_return": m.get("ann_return"), "max_drawdown": m.get("max_drawdown"), "p": m.get("ttest_p"),
                  "cost_x2_sharpe": x2.get("sharpe") if isinstance(x2, dict) else None,
                  "abstain_sharpe": av.get("sharpe"), "deflated_sharpe": dv.get("sharpe"),
                  "deflated_folds_active": dv.get("folds_active"), "folds": dv.get("folds"),
                  "n_candidates": m.get("n_candidates"), "space": (m.get("joint_space") or {}).get("size"),
                  "param_stability": m.get("param_stability"),
                  "latest_params": json.dumps(lc.get("params"), default=str), "latest_exec": label(lc.get("exec")),
                  "latest_risk": label(lc.get("risk")), "latest_train_sharpe": lc.get("train_sharpe"),
                  "latest_deflation_benchmark": lc.get("deflation_benchmark")})
    if not J:
        json.dump({"joint_streams": 0}, open(OUT / "summary.json", "w"))
        print("no joint experiments")
        return
    df = pd.DataFrame(J)
    df["standard_sharpe"] = [float(np.mean(base[k])) if (k := (r.family, r.symbol, r.tf)) in base else np.nan
                             for r in df.itertuples()]
    df["delta_vs_standard"] = df["sharpe"] - df["standard_sharpe"]
    p = df["p"].fillna(1.0).clip(0, 1).to_numpy()
    df["bh"] = benjamini_hochberg(p)
    df["holm"] = holm(p)
    df = df.sort_values("sharpe", ascending=False)
    df.to_csv(OUT / "streams.csv", index=False)
    cf = pd.DataFrame([{"kind": k, "choice": c, "folds_selected": n} for (k, c), n in choice.items()])
    cf["share"] = cf["folds_selected"] / cf.groupby("kind")["folds_selected"].transform("sum")
    cf.sort_values(["kind", "folds_selected"], ascending=[True, False]).to_csv(OUT / "choice_frequency.csv", index=False)
    agg = dict(n=("sharpe", "size"), median_sharpe=("sharpe", "median"), mean_sharpe=("sharpe", "mean"),
               share_positive=("sharpe", lambda s: float((s > 0).mean())),
               median_delta_vs_standard=("delta_vs_standard", "median"),
               median_cost_x2=("cost_x2_sharpe", "median"), best=("sharpe", "max"))
    df.groupby("family").agg(**agg).sort_values("median_sharpe", ascending=False).to_csv(OUT / "by_family.csv")
    df.groupby(["symbol", "tf"]).agg(**agg).sort_values("median_sharpe", ascending=False).to_csv(OUT / "by_pair.csv")
    summ = {
        "joint_streams": int(len(df)), "families": int(df.family.nunique()), "pairs": int(df.symbol.nunique()),
        "tfs": sorted(df.tf.unique().tolist()),
        "median_oos_sharpe": float(df.sharpe.median()), "share_positive": float((df.sharpe > 0).mean()),
        "median_delta_vs_standard": float(df.delta_vs_standard.median()) if df.delta_vs_standard.notna().any() else None,
        "share_better_than_standard": float((df.delta_vs_standard > 0).mean()) if df.delta_vs_standard.notna().any() else None,
        "median_abstain_sharpe": float(df.abstain_sharpe.median()),
        "deflated_share_of_folds_traded": float(df.deflated_folds_active.sum() / max(1, df.folds.sum())),
        "bh_survivors_q10": int((df.bh < 0.10).sum()), "holm_survivors_a05": int((df.holm < 0.05).sum()),
        "total_candidates_simulated": int(df.n_candidates.fillna(0).sum()),
        "top10": df.head(10)[["family", "symbol", "tf", "sharpe", "p", "bh", "cost_x2_sharpe", "deflated_sharpe",
                              "latest_exec", "latest_risk"]].to_dict("records"),
        "note": "top10 is a post-hoc ranking across all joint streams (selection bias); only BH/Holm survivors "
                "and pre-registered follow-ups count as evidence.",
    }
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "top10"}, indent=1, default=str))


if __name__ == "__main__":
    main()
