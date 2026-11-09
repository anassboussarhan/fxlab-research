"""Markdown tables for docs/round17_results.md from the round-17 outputs (branch round17).
Usage: python scripts/r17_report.py DIR"""
import json
import os
import sys

import numpy as np
import pandas as pd

d = sys.argv[1]
S = json.load(open(os.path.join(d, "summary17.json")))
C = pd.read_csv(os.path.join(d, "chosen_all.csv"))


def f(x, n=2):
    try:
        return "-" if x is None or not np.isfinite(float(x)) else f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return "-"


def yn(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return "yes" if str(x) in ("True", "true", "1") or x is True else "no"


order = {"FACTOR": 0, "BOOK-UNIT": 1, "BOOK-CFG": 2, "PAIR-UNIT": 3}
C["o"] = C.tier.map(order)
C["fo"] = C.freq.map({"D": 0, "W": 1, "M": 2})
C = C.sort_values(["o", "fo", "family"])
print("| Tier | Freq | Family | Chosen on 2011-2018 | t 2011-18 | t 2019-25 | Sharpe 2019-25 | minus B1, t | p | BH | P3 | P4 | Counts |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in C.iterrows():
    v = r.variant.split(":", 1)[1]
    print(f"| {r.tier} | {r.freq} | {r.family} | `{v}` | {f(r.inner_t)} | {f(r.check_t)} | {f(r.check_sharpe)} | "
          f"{f(r.diff_check_t)} | {f(r.p, 4)} | {yn(r.bh_pass)} | {yn(r.P3)} | {yn(r.P4)} | {yn(r.counts)} |")
print("\nsummary", json.dumps({k: S[k] for k in ("tests", "counts", "p1_pre", "bh_pass")}))
print("\noverall picks", json.dumps(S["overall_pick"], indent=0, default=str))
B = C.drop_duplicates(["tier", "freq"])[["tier", "freq", "b1", "b1_check_t", "b1_check_sharpe"]]
print("\n## B1 benchmarks\n")
print(B.round(2).to_markdown(index=False))
ss = pd.read_csv(os.path.join(d, "surface_summary.csv"))
print("\n## surface\n")
print(ss.round(3).to_markdown(index=False))
kc = os.path.join(d, "k_curve.csv")
if os.path.exists(kc):
    K = pd.read_csv(kc)
    print("\n## K curve (check t of the best inner setting per K)\n")
    print(K.pivot_table(index=["tier", "freq", "family"], columns="K", values="check_t").round(2).to_markdown())
tf = os.path.join(d, "timing_long_only.csv")
if os.path.exists(tf):
    X = pd.read_csv(tf)
    print("\n## timing\n", len(X), "tests;", int(X.counts.sum()), "count; share diff t >= 1.645:",
          round(float((X.diff_t >= 1.645).mean()), 3))
    print(X.sort_values("p").head(12).round(3).to_markdown(index=False))
