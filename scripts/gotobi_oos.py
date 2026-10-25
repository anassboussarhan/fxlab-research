"""Gotobi out of sample, 2019-2025 (after the 2007-2018 tests of runs 17 / 19 and after the published samples).

Data: USDJPY ticks downloaded directly from Dukascopy on the GitHub runner (fxlab.data.acquire_dukascopy), built
into the same M1 bid/ask layout. Development data only: the download ends 2025-12-31 23:59 UTC and the loader
refuses 2026.

PRE-REGISTERED (fixed before the data were downloaded, rules unchanged from run 17 / scripts/flows_lab.py):
  primary   F1: long USDJPY 03:00 JST -> 09:55 JST (Tokyo fix) on Gotobi days (5th, 10th, 15th, 20th, 25th, last
            day of the month, moved back to the previous Tokyo business day), Mondays skipped; net of spread,
            slippage, commission and the rate-differential swap. Confirmed if the mean is > 0 with one-sided
            t >= 1.645; strong if t >= 2.
  secondary F1 with the exit at 10:00 JST (the exit forced by the mirror's missing 00:00-01:00 UTC hour in runs
            17 / 19, for comparability); F2, F3; random-date test against the same window on non-Gotobi days of
            the same year and weekday; random entry timing (B1, B2); cost levels B (raw ECN) and C (IBKR) as in
            docs/breakeven.md.
Usage: python scripts/gotobi_oos.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

import flows_lab as FL  # noqa: E402
from flows_report import permutation  # noqa: E402

START, END = "2019-01-01", "2025-12-31 23:59"


def stat(x, per_year=62.5):
    x = pd.Series(x).dropna()
    if len(x) < 10:
        return {"events": int(len(x))}
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    return {"events": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": float(x.mean() / x.std() * np.sqrt(len(x))),
            "sharpe": float(x.mean() / x.std() * np.sqrt(per_year)), "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}",
            "worst_bp": float(x.min() * 1e4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reports/gotobi_oos")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from fxlab.data.store import load_m1
    m1 = load_m1("USDJPY", a.start, a.end, columns=["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"])
    mid = 0.5 * (m1.bid_c + m1.ask_c)
    meta = {"rows": int(len(m1)), "first": str(m1.index[0]), "last": str(m1.index[-1]),
            "median_mid": float(mid.median()), "median_spread_pips": float(((m1.ask_c - m1.bid_c) / 0.01).median()),
            "bars_by_utc_hour": {int(k): int(v) for k, v in m1.groupby(m1.index.hour).size().items()},
            "years": sorted({int(y) for y in m1.index.year})}
    assert 70 < meta["median_mid"] < 200, f"implausible USDJPY level {meta['median_mid']}"
    book = FL.Book(m1, 0.01, "USDJPY")
    start, end = str(m1.index[0].date()), str(m1.index[-1].date())
    rows = FL.gotobi(book, start, end)
    # secondary: exit at 10:00 JST (as forced by the mirror data in runs 17 / 19)
    hol = FL.jp_holidays(range(m1.index[0].year, m1.index[-1].year + 1))
    G = set(FL.gotobi_days(start, end, hol))
    for d in pd.bdate_range(start, end):
        if d in hol or d.weekday() == 0:
            continue
        r = book.trade(FL.local_ts(d, "Asia/Tokyo", 3, 0), FL.local_ts(d, "Asia/Tokyo", 10, 0), +1, tol_min=65)
        if r is not None:
            rows.append(FL.rec("F1_1000" if d in G else "F1_1000_control", d, r))
    R = pd.DataFrame(rows)
    R["day"] = pd.to_datetime(R["day"])
    R["net_sw"] = R["net"] + R["swap_rate"].fillna(R["swap_flat"])
    R["net_swf"] = R["net"] + R["swap_flat"]
    pipbp = 0.01 / meta["median_mid"] * 1e4
    cost = R["gross"] - R["net"]
    spread = ((cost * 1e4 - 0.2 * pipbp - 0.7) / 1.5).clip(lower=0) / 1e4
    swap = R["net_sw"] - R["net"]
    R["net_B_ecn"] = R["gross"] + swap - (spread + 0.7e-4)
    R["net_C_ibkr"] = R["gross"] + swap - (spread + 0.4e-4)
    R.to_parquet(os.path.join(a.out, "trades.parquet"))
    out, series = [], {}
    for t in sorted(R.test.unique()):
        g = R[R.test == t]
        if t.startswith("B"):
            g = g.groupby("day")[["net", "gross", "net_sw", "net_swf", "net_B_ecn", "net_C_ibkr"]].mean()
        else:
            g = g.set_index("day")[["net", "gross", "net_sw", "net_swf", "net_B_ecn", "net_C_ibkr", "delay_out"]]
        series[t] = g
        r = {"test": t}
        for col in ("net_sw", "net", "net_swf", "gross", "net_B_ecn", "net_C_ibkr"):
            for k, v in stat(g[col]).items():
                r[f"{col}_{k}" if k != "events" else "events"] = v
        if "delay_out" in g:
            r["median_exit_delay_min"] = float(g["delay_out"].median())
        out.append(r)
    T = pd.DataFrame(out)
    T["p_one_sided_net_sw"] = st.norm.sf(T["net_sw_t"].astype(float).fillna(0))
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)

    def yw(ix):
        ix = pd.DatetimeIndex(ix)
        return np.array([f"{y}-{w}" for y, w in zip(ix.year, ix.weekday)])
    perm = []
    for ev, co in (("F1", "F1_control"), ("F1_1000", "F1_1000_control"), ("F2", "F2_control"), ("F3", "F3_control")):
        if ev in series and co in series:
            for col in ("net_sw", "gross"):
                p = permutation(series[ev][col], series[co][col], yw)
                wt = st.ttest_ind(series[ev][col].dropna(), series[co][col].dropna(), equal_var=False).statistic
                perm.append({"test": ev, "column": col, **p, "welch_t": float(wt)})
    pd.DataFrame(perm).to_csv(os.path.join(a.out, "permutation.csv"), index=False)
    years = []
    for t in ("F1", "F1_1000", "F1_control"):
        if t in series:
            y = series[t]["net_sw"].groupby(series[t].index.year).agg(["mean", "count", "sum"])
            for yr, v in y.iterrows():
                years.append({"test": t, "year": int(yr), "mean_bp": float(v["mean"] * 1e4), "events": int(v["count"]),
                              "sum_bp": float(v["sum"] * 1e4)})
    pd.DataFrame(years).to_csv(os.path.join(a.out, "by_year.csv"), index=False)
    pd.concat([s.assign(test=k).reset_index() for k, s in series.items()]).to_csv(os.path.join(a.out, "series.csv"),
                                                                                    index=False)
    f1 = T[T.test == "F1"].iloc[0].to_dict() if (T.test == "F1").any() else {}
    meta["verdict"] = ("confirmed (t >= 2)" if f1.get("net_sw_t", 0) >= 2 else
                       "confirmed (t >= 1.645)" if f1.get("net_sw_t", 0) >= 1.645 and f1.get("net_sw_mean_bp", 0) > 0 else
                       "not confirmed")
    meta["F1"] = {k: f1.get(k) for k in ("events", "net_sw_mean_bp", "net_sw_t", "net_sw_sharpe", "net_sw_years_positive",
                                         "gross_mean_bp", "gross_t", "net_B_ecn_mean_bp", "net_B_ecn_t")}
    json.dump(meta, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = ["test", "events", "net_sw_mean_bp", "net_sw_t", "net_sw_sharpe", "net_sw_years_positive", "gross_mean_bp",
            "gross_t", "net_B_ecn_mean_bp", "net_B_ecn_t", "net_C_ibkr_t"]
    print(T[[c for c in cols if c in T]].round(3).to_string())
    print(pd.DataFrame(perm).round(4).to_string())
    print(json.dumps({k: v for k, v in meta.items() if k != "bars_by_utc_hour"}, indent=1, default=str))


if __name__ == "__main__":
    main()
