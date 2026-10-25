"""Aggregate scripts/flows_lab.py (run 17): the 17 pre-registered flow / calendar / option-implied tests.

Each test becomes one return series per event (a portfolio return per event day for multi-currency tests, equal
weight over the currencies trading that day). Reported: events, mean per event (bp), t-statistic of the mean,
annualised Sharpe (mean / sd x sqrt(events per year)), share of years positive, net and gross, and the split at the
end of the original paper's sample where there is one (in-sample overlap vs after the paper). One-sided p-values
(positive = the effect as published), Benjamini-Hochberg over the 17 primary tests. Controls are reported alongside.
Writes results/reports/flows_summary/{tests.csv, by_currency.csv, summary.json}."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg  # noqa: E402

IN = Path(os.environ.get("FR_IN", RESULTS_DIR / "reports" / "flows"))
OUT = Path(os.environ.get("FR_OUT", RESULTS_DIR / "reports" / "flows_summary"))
PRIMARY = ["F1", "F2", "F3", "M1", "M2", "L1", "L2", "V1", "V2", "C1", "C2", "K1", "K2", "K3", "P1", "P2", "W1"]
DESC = {"F1": "Gotobi: long USDJPY 03:00 -> fix (10:00 JST in this data)",
        "F2": "Gotobi: long USDJPY 08:00 -> fix (10:00 JST in this data)",
        "F3": "Gotobi reversal: short USDJPY after the fix -> 12:00 JST",
        "M1": "month-end London fix, equity-hedging signal, 08:00 -> 16:00 London",
        "M2": "month-end London fix, equity-hedging signal, 24 h into the fix",
        "L1": "local trading hours: short each currency in its home session (8 currencies)",
        "L2": "local trading hours, EURUSD", "V1": "sell 1-month EURUSD variance at EVZ (0.5 vol pt cost)",
        "V2": "volatility-risk-premium direction signal, EURUSD", "C1": "carry, only when VIX is calm (7 pairs)",
        "C2": "carry, always (7 pairs)", "K1": "long foreign currencies in New York hours (9)",
        "K2": "short foreign currencies overnight (9)", "K3": "both legs, EUR / GBP / CHF",
        "P1": "FOMC days: long foreign currencies vs USD (9)", "P2": "FOMC days: 3 highest-yielding currencies",
        "W1": "fade the 15:30 -> 16:00 London move after the fix (9)"}
PAPER_END = {"F1": "2020-12-31", "F2": "2020-12-31", "F3": "2020-12-31", "L1": "2007-05-31", "L2": "2007-05-31",
             "K1": "2014-12-31", "K2": "2014-12-31", "K3": "2014-12-31", "P1": "2010-12-31", "P2": "2010-12-31"}
EVENTS_PER_YEAR = {"F": 70, "M": 12, "V": 12, "P": 8}


def stats(x, per_year):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 10 or x.std() == 0:
        return {"events": n}
    t = float(x.mean() / x.std() * np.sqrt(n))
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    return {"events": n, "mean_bp": float(x.mean() * 1e4), "t": t, "sharpe": float(x.mean() / x.std() * np.sqrt(per_year)),
            "share_years_positive": float((yrs > 0).mean()), "worst_event_bp": float(x.min() * 1e4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, metas = [], {}
    for f in sorted(glob.glob(str(IN / "*.parquet"))):
        frames.append(pd.read_parquet(f))
    for f in sorted(glob.glob(str(IN / "*.json"))):
        J = json.load(open(f))
        if not J.get("skipped"):
            metas[J["pair"]] = J
    if not frames:
        json.dump({"tests": 0, "meta": metas}, open(OUT / "summary.json", "w"), indent=1, default=str)
        print("no flows results")
        return
    R = pd.concat(frames, ignore_index=True)
    R["day"] = pd.to_datetime(R["day"])
    series = {}

    def port(test, pairs=None, ccys=None):
        g = R[R.test == test]
        if pairs is not None:
            g = g[g.pair.isin(pairs)]
        if ccys is not None and "ccy" in g:
            g = g[g.ccy.isin(ccys)]
        return g.groupby("day")[["net", "gross"]].mean()

    for t in ("F1", "F2", "F3"):
        series[t] = port(t)
        series[t + "_control"] = port(t + "_control")
    series["M1"], series["M2"] = port("M1"), port("M2")
    series["L1"], series["L2"] = port("L1"), port("L2")
    series["V1"], series["V2"] = port("V1"), port("V2")
    series["C1"], series["C2"] = port("C1"), port("C2")
    series["K1"], series["K2"] = port("K_ny"), port("K_on")
    k3 = R[R.test.isin(["K_ny", "K_on"]) & R.ccy.isin(["EUR", "GBP", "CHF"])] if "ccy" in R else R.iloc[:0]
    series["K3"] = k3.groupby(["day", "ccy"])[["net", "gross"]].sum().groupby("day").mean()
    series["P1"], series["P1_control"] = port("P_dol"), port("P_dol_control")
    pd_ = R[R.test == "P_dol"].dropna(subset=["rate_diff_foreign"]) if "rate_diff_foreign" in R else R.iloc[:0]
    if len(pd_):
        top = pd_.sort_values("rate_diff_foreign", ascending=False).groupby("day").head(3)
        series["P2"] = top.groupby("day")[["net", "gross"]].mean()
    series["W1"] = port("W1")
    rows = []
    for name, S in series.items():
        if S is None or not len(S):
            rows.append({"test": name, "events": 0})
            continue
        base = name.split("_")[0]
        per_year = EVENTS_PER_YEAR.get(base[0], 252) if base[0] in "FMVP" else 252
        r = {"test": name, "primary": name in PRIMARY, "description": DESC.get(base, base) + (" (control)" if "control" in name else ""),
             "first": str(S.index.min().date()), "last": str(S.index.max().date())}
        for k, v in stats(S["net"], per_year).items():
            r[f"net_{k}" if k != "events" else "events"] = v
        for k, v in stats(S["gross"], per_year).items():
            if k != "events":
                r[f"gross_{k}"] = v
        if base in PAPER_END:
            cut = pd.Timestamp(PAPER_END[base])
            a, b = S[S.index <= cut], S[S.index > cut]
            r["in_paper_sample_t"] = stats(a["net"], per_year).get("t") if len(a) else None
            r["after_paper_t"] = stats(b["net"], per_year).get("t") if len(b) else None
            r["after_paper_mean_bp"] = float(b["net"].mean() * 1e4) if len(b) else None
        rows.append(r)
    T = pd.DataFrame(rows)
    P = T[T.primary == True].copy()                                     # noqa: E712
    P["p"] = st.norm.sf(P.net_t.astype(float).fillna(0))
    P["p_bh"] = benjamini_hochberg(P.p.to_numpy())
    T = T.merge(P[["test", "p", "p_bh"]], on="test", how="left")
    T.to_csv(OUT / "tests.csv", index=False)
    # per-currency breakdown of the multi-currency tests
    bc = []
    for test in ("M1", "M2", "L1", "C1", "C2", "K_ny", "K_on", "P_dol", "W1"):
        for pair, g in R[R.test == test].groupby("pair"):
            s_ = stats(g.set_index("day")["net"], 12 if test.startswith("M") else (8 if test == "P_dol" else 252))
            bc.append({"test": test, "pair": pair, **s_})
    pd.DataFrame(bc).to_csv(OUT / "by_currency.csv", index=False)
    summ = {"primary_tests": int(len(P)), "bh_survivors_q10": int((P.p_bh < 0.10).sum()),
            "survivors": P[P.p_bh < 0.10][["test", "net_mean_bp", "net_t", "net_sharpe", "p_bh"]].to_dict("records"),
            "meta": {k: {kk: vv for kk, vv in v.items() if kk not in ("alt_files",)} for k, v in metas.items()}}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k != "meta"}, indent=1, default=str))


if __name__ == "__main__":
    main()
