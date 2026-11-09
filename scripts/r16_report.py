"""Markdown tables for docs/round16_results.md from the round-16 outputs (branch round16). Usage:
python scripts/r16_report.py DIR > tables.md"""
import json
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
S = json.load(open(os.path.join(d, "summary16.json")))
C = pd.read_csv(os.path.join(d, "chosen.csv"))


def f(x, n=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def yn(x):
    if isinstance(x, float) and not np.isfinite(x):
        return "-"
    return "yes" if bool(x) else "no"


print("## chosen\n")
print("| Tier | Family | Chosen on 2011-2018 | 2011-18 t | 2019-25 mean | 2019-25 t | 2019-25 Sharpe | minus B1, t | placebo p95 Sharpe | P1 | P2 | P3 | P4 | Counts |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in C.iterrows():
    print(f"| {r.tier} | {r.family} | `{r.variant.split(':', 1)[1]}` | {f(r.inner_t)} | {f(r.check_mean, 3)} | {f(r.check_t)} | "
          f"{f(r.check_sharpe)} | {f(r.diff_check_t)} | {f(r.placebo_p95_sharpe)} | {yn(r.P1)} | {yn(r.P2)} | {yn(r.P3)} | "
          f"{yn(r.P4)} | {yn(r.counts)} |")
print("\noverall pick", json.dumps(S["overall_pick"], default=str))
print("\nA1", json.dumps(S["A1"], indent=1))
print("\nA2 top", json.dumps(S["A2_top"], indent=1))
print("\nA2 pass", S["A2_pass"])
kc = pd.read_csv(os.path.join(d, "k_curve.csv"))
print("\n## K curve\n")
print(kc.round(2).to_markdown(index=False))
O2 = pd.read_csv(os.path.join(d, "O2_options_in_regimes.csv"))
print("\n## O2\n")
print(O2.round(3).to_markdown(index=False))
O3 = pd.read_csv(os.path.join(d, "O3_options_timing.csv"))
print("\n## O3 top 15 (all groups)\n")
print(O3.sort_values("p").head(15).round(3).to_markdown(index=False))
print("\nO3 counts", int(O3.counts.sum()), "of", int(O3.group.isin(["OPT", "CME"]).sum()), "options tests")
print("O3 by group: share with check t >= 1.645:", O3.assign(hit=O3.check_t >= 1.645).groupby("group").hit.mean().round(3).to_dict())
D = pd.read_csv(os.path.join(d, "D_dip_filter.csv"))
print("\n## D\n")
print(D.sort_values("p").round(3).to_markdown(index=False))
print("\nBIC", json.dumps(S["bic_k"], default=str)[:2000])
print("\ndropped", S["dropped_features"], "tiers", S["tiers"], S["placebo_tiers"])
surf = pd.read_csv(os.path.join(d, "surface.csv.gz"))
print("\n## surface summary\n")
g = surf.groupby(["tier", "family"]).agg(settings=("variant", "size"), med_inner_t=("inner_t", "median"),
                                         med_check_t=("check_t", "median"),
                                         share_check_ge_1645=("check_t", lambda s: float((s >= 1.645).mean())),
                                         best_check_t=("check_t", "max"))
print(g.round(3).to_markdown())
maps = json.load(open(os.path.join(d, "regime_maps.json")))
print("\n## regime maps\n")
for k, m in maps.items():
    if not m:
        continue
    print(f"### {k} (refit {m['refit_month']}), last decision probabilities {m['last_decision_probs']}")
    for r in m["regimes"]:
        prof = ", ".join(f"{a} {b:+.2f}" for a, b in sorted(r["feature_profile_z"].items(), key=lambda x: -abs(x[1]))[:6])
        tops = "; ".join(f"{t['strategy']} ({t['cond_mean_z']:+.2f})" for t in r["top"][:3])
        print(f"- regime {r['regime']}: {r['training_months']} months; profile: {prof}; top: {tops}")
