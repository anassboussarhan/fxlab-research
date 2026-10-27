"""Benjamini-Hochberg over every pre-registered primary test of the untested batch (options mechanics, daily data,
intraday), from the published result branches. One-sided p-values as each lab reports them; tests whose data
were shown invalid are listed but left out of the correction (D5 FTSE: Yahoo opens equal the previous close 73%
of the time; D8: no venue answered).
Usage: python scripts/untested_batch_summary.py --options OPT_tests.csv --daily DAILY_tests.csv
                                                --intraday INTRA_tests_all.csv --out docs/untested_batch/summary.csv"""
import argparse
import os

import numpy as np
import pandas as pd

INVALID = {"D5 Overnight, other indices: ^FTSE": "Yahoo FTSE opens equal the previous close 73% of the time"}


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", required=True)
    ap.add_argument("--daily", required=True)
    ap.add_argument("--intraday", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    O = pd.read_csv(a.options)
    for _, r in O[O.primary == True].iterrows():  # noqa: E712
        rows.append({"lab": "options", "test": r.test, "family": r.family, "n": r.months, "effect": r.mean_month_pct,
                     "unit": "% a month (paired difference)" if r.family in ("T1", "T2", "T3", "T0") else "% a month",
                     "t": r.t, "p_one_sided": r.p_one_sided})
    D = pd.read_csv(a.daily)
    for _, r in D[D.primary == True].iterrows():  # noqa: E712
        rows.append({"lab": "daily", "test": r.test, "family": r.family, "n": r.n, "effect": r.mean_pct,
                     "unit": "% a trade / month / night", "t": r.t, "p_one_sided": r.p_one_sided})
    I = pd.read_csv(a.intraday)
    I = I[~I.test.str.contains("control")]
    for _, r in I.iterrows():
        rows.append({"lab": "intraday", "test": r.test, "family": r.family, "n": r.n, "effect": r.mean_bp,
                     "unit": "bp a trade, net", "t": r.t, "p_one_sided": r.p_one_sided,
                     "gross_bp": r.get("gross_mean_bp"), "gross_t": r.get("gross_t"),
                     "t_2007_2018": r.get("t_2007_2018"), "t_2019_2025": r.get("t_2019_2025")})
    T = pd.DataFrame(rows)
    T["excluded"] = T.test.map(INVALID).fillna("")
    T.loc[T.p_one_sided.isna() & (T.excluded == ""), "excluded"] = "too few trades / no data"
    m = T.excluded == ""
    T.loc[m, "q_bh"] = bh(T.loc[m, "p_one_sided"].to_numpy())
    T["survives_q10"] = T.q_bh <= 0.10
    T = T.sort_values(["q_bh", "p_one_sided"], na_position="last")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    T.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(f"{int(m.sum())} tests in the correction, {int(T.survives_q10.sum())} with q <= 0.10")
    print(T[["lab", "test", "n", "effect", "t", "p_one_sided", "q_bh", "excluded"]].round(4).to_string())


if __name__ == "__main__":
    main()
