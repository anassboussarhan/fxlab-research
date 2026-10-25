"""Aggregate scripts/flows_lab.py (run 17): the 17 pre-registered flow / calendar / option-implied tests.

Each test becomes one return series per event (a portfolio return per event day for multi-currency tests, equal
weight over the currencies trading that day). Reported: events, mean per event (bp), t-statistic of the mean,
annualised Sharpe (mean / sd x sqrt(events per year)), share of years positive, net and gross, and the split at the
end of the original paper's sample where there is one (in-sample overlap vs after the paper). One-sided p-values
(positive = the effect as published), Benjamini-Hochberg over the 17 primary tests. Controls are reported alongside.
Writes results/reports/flows_summary/{tests.csv, by_currency.csv, summary.json}.

Run 19 additions: (1) swap: 'net_sw' = net + the rate-differential swap (flat 0.5 pip per night where no rates),
'net_swf' = net + the flat swap; BH is repeated on net_sw. (2) random-date tests (permutation.csv): each event
test against the same rule on randomly drawn non-event days from the same stratum (year x weekday for Gotobi and
FOMC, year x month for the month-end rule), 10,000 draws; p = share of draws with a mean at least as high.
(3) random timing (random_timing.csv): B1-B6, random entry minutes inside the event window in the edge's direction
(16 seeds per event); statistics of the seed-average stream, the single-trader distribution over the 16 seeds,
coin-flip controls; BH over B1-B6 on net_sw."""
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


def permutation(event, control, strata, n_draw=10000, seed=19, min_pool=5):
    """Random-date test: the event series' mean vs the means of n_draw sets of control days, each set drawing as many
    days per stratum as there are events in it, with replacement (Gotobi Fridays outnumber the non-Gotobi Fridays
    of a year). Strata with fewer than min_pool control days are dropped with their events."""
    rng = np.random.default_rng(seed)
    ev, co = event.dropna(), control.dropna()
    se, sc = strata(ev.index), strata(co.index)
    total = np.zeros(n_draw)
    used, keep_keys = 0, []
    for key in pd.unique(se):
        k = int((se == key).sum())
        pool = co.to_numpy()[sc == key]
        if len(pool) < min_pool or k == 0:
            continue
        keep_keys.append(key)
        used += k
        total += pool[rng.integers(0, len(pool), (n_draw, k))].sum(axis=1)
    keep = np.isin(se, keep_keys)
    actual = float(ev.to_numpy()[keep].mean()) if used else np.nan
    draws = total / max(used, 1)
    p = float((1 + (draws >= actual).sum()) / (n_draw + 1)) if used else np.nan
    return {"events_used": used, "actual_bp": actual * 1e4, "draws_mean_bp": float(draws.mean() * 1e4),
            "draws_p5_bp": float(np.percentile(draws, 5) * 1e4), "draws_p95_bp": float(np.percentile(draws, 95) * 1e4),
            "p_random_dates": p}


def random_timing(R, out):
    """B1-B6: per seed, the equal-weight portfolio per event day; the seed-average stream and the 16 single
    traders."""
    per_year = {"B1": 70, "B2": 70, "B3": 12, "B4": 12, "B5": 8, "B6": 8}
    desc = {"B1": "Gotobi, random entry 03:00-09:00 JST, long USDJPY, exit at the fix",
            "B2": "Gotobi, random entry, long, random bracket (0.1/0.2/0.4 x daily range), exit at the fix",
            "B3": "month-end, random entry in the 24 h before the London fix, signal direction, exit at the fix",
            "B4": "month-end, random entry, signal direction, random bracket, exit at the fix",
            "B5": "FOMC, random entry 17:00 ET prev day -> 16:30 ET, long foreign currencies, exit 17:00 ET",
            "B6": "FOMC, random entry, long foreign currencies, random bracket, exit 17:00 ET"}
    rows = []
    for b in ("B1", "B2", "B3", "B4", "B5", "B6"):
        for name in (b, b + "_control"):
            g = R[R.test == name]
            if not len(g):
                rows.append({"test": name, "events": 0})
                continue
            S = g.groupby(["seed", "day"])[["net", "gross", "net_sw", "net_swf"]].mean()     # portfolio per seed-day
            ev = S.groupby("day").mean()                                                     # seed-average stream
            r = {"test": name, "primary": name == b, "description": desc[b] + (" (coin-flip direction)" if
                 "control" in name else ""), "first": str(ev.index.min().date()), "last": str(ev.index.max().date())}
            for col in ("net", "gross", "net_sw", "net_swf"):
                for k, v in stats(ev[col], per_year[b]).items():
                    r[f"{col}_{k}" if k != "events" else "events"] = v
            # single traders: each seed's own stream
            seed_stats = [stats(S.loc[sd]["net_sw"], per_year[b]) for sd in S.index.get_level_values(0).unique()]
            m = np.array([x.get("mean_bp", np.nan) for x in seed_stats])
            t = np.array([x.get("t", np.nan) for x in seed_stats])
            r.update({"seeds": int(len(m)), "single_median_mean_bp_sw": float(np.nanmedian(m)),
                      "single_worst_mean_bp_sw": float(np.nanmin(m)), "single_best_mean_bp_sw": float(np.nanmax(m)),
                      "single_median_t_sw": float(np.nanmedian(t)), "single_share_positive_sw": float(np.mean(m > 0)),
                      "single_share_t_above_2_sw": float(np.mean(t > 2))})
            rows.append(r)
    T = pd.DataFrame(rows)
    P = T[T.primary == True].copy()                                     # noqa: E712
    if len(P) and "net_sw_t" in P:
        P["p_sw"] = st.norm.sf(P.net_sw_t.astype(float).fillna(0))
        P["p_bh_sw"] = benjamini_hochberg(P.p_sw.to_numpy())
        T = T.merge(P[["test", "p_sw", "p_bh_sw"]], on="test", how="left")
    T.to_csv(out / "random_timing.csv", index=False)
    return T


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
    for c in ("swap_flat", "swap_rate"):
        if c not in R:
            R[c] = np.nan
    R["swap_flat"] = R["swap_flat"].fillna(0.0)                  # carry / variance rows carry no separate swap
    R["net_swf"] = R["net"] + R["swap_flat"]
    R["net_sw"] = R["net"] + R["swap_rate"].fillna(R["swap_flat"])
    COLS = ["net", "gross", "net_sw", "net_swf"]
    series = {}

    def port(test, pairs=None, ccys=None):
        g = R[R.test == test]
        if pairs is not None:
            g = g[g.pair.isin(pairs)]
        if ccys is not None and "ccy" in g:
            g = g[g.ccy.isin(ccys)]
        return g.groupby("day")[COLS].mean()

    for t in ("F1", "F2", "F3"):
        series[t] = port(t)
        series[t + "_control"] = port(t + "_control")
    series["M1"], series["M2"] = port("M1"), port("M2")
    series["L1"], series["L2"] = port("L1"), port("L2")
    series["V1"], series["V2"] = port("V1"), port("V2")
    series["C1"], series["C2"] = port("C1"), port("C2")
    series["K1"], series["K2"] = port("K_ny"), port("K_on")
    k3 = R[R.test.isin(["K_ny", "K_on"]) & R.ccy.isin(["EUR", "GBP", "CHF"])] if "ccy" in R else R.iloc[:0]
    series["K3"] = k3.groupby(["day", "ccy"])[COLS].sum().groupby("day").mean()
    series["P1"], series["P1_control"] = port("P_dol"), port("P_dol_control")
    pd_ = R[R.test == "P_dol"].dropna(subset=["rate_diff_foreign"]) if "rate_diff_foreign" in R else R.iloc[:0]
    if len(pd_):
        top = pd_.sort_values("rate_diff_foreign", ascending=False).groupby("day").head(3)
        series["P2"] = top.groupby("day")[COLS].mean()
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
        for col in ("gross", "net_sw", "net_swf"):
            for k, v in stats(S[col], per_year).items():
                if k != "events":
                    r[f"{col}_{k}"] = v
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
    P["p_sw"] = st.norm.sf(P.net_sw_t.astype(float).fillna(0))
    P["p_bh_sw"] = benjamini_hochberg(P.p_sw.to_numpy())
    T = T.merge(P[["test", "p", "p_bh", "p_sw", "p_bh_sw"]], on="test", how="left")
    T.to_csv(OUT / "tests.csv", index=False)
    # per-currency breakdown of the multi-currency tests
    bc = []
    for test in ("M1", "M2", "L1", "C1", "C2", "K_ny", "K_on", "P_dol", "W1"):
        for pair, g in R[R.test == test].groupby("pair"):
            s_ = stats(g.set_index("day")["net"], 12 if test.startswith("M") else (8 if test == "P_dol" else 252))
            bc.append({"test": test, "pair": pair, **s_})
    pd.DataFrame(bc).to_csv(OUT / "by_currency.csv", index=False)
    # ---- run 19: random-date (permutation) tests
    def yw(ix):
        ix = pd.DatetimeIndex(ix)
        return np.array([f"{y}-{w}" for y, w in zip(ix.year, ix.weekday)])

    def ym(ix):
        ix = pd.DatetimeIndex(ix)
        return np.array([f"{y}-{m}" for y, m in zip(ix.year, ix.month)])
    perm_rows = []
    pairs_ = [("F1", "F1_control", yw), ("F2", "F2_control", yw), ("F3", "F3_control", yw),
              ("P1", "P1_control", yw), ("M1", "M1_placebo", ym), ("M2", "M2_placebo", ym)]
    if "M1_placebo" in set(R.test):
        series["M1_placebo"], series["M2_placebo"] = port("M1_placebo"), port("M2_placebo")
    pc = R[R.test == "P_dol_control"].dropna(subset=["rate_diff_foreign"]) if "rate_diff_foreign" in R else R.iloc[:0]
    if len(pc):
        series["P2_control"] = pc.sort_values("rate_diff_foreign", ascending=False).groupby("day").head(3) \
            .groupby("day")[COLS].mean()
        pairs_.append(("P2", "P2_control", yw))
    for ev_name, co_name, strata in pairs_:
        if ev_name not in series or co_name not in series or not len(series[ev_name]) or not len(series[co_name]):
            perm_rows.append({"test": ev_name, "control": co_name, "events_used": 0})
            continue
        for col in ("net_sw", "gross"):
            r = permutation(series[ev_name][col], series[co_name][col], strata)
            perm_rows.append({"test": ev_name, "control": co_name, "column": col, **r})
    PM = pd.DataFrame(perm_rows)
    PM.to_csv(OUT / "permutation.csv", index=False)
    long = [S.assign(test=k).reset_index() for k, S in series.items() if S is not None and len(S)]
    if long:                                             # per-event portfolio returns, for plots and per-year checks
        pd.concat(long, ignore_index=True).round(8).to_csv(OUT / "series.csv", index=False)
    # ---- run 19: random timing inside the event windows
    RT = random_timing(R, OUT)
    summ = {"primary_tests": int(len(P)), "bh_survivors_q10": int((P.p_bh < 0.10).sum()),
            "survivors": P[P.p_bh < 0.10][["test", "net_mean_bp", "net_t", "net_sharpe", "p_bh"]].to_dict("records"),
            "bh_survivors_q10_with_swap": int((P.p_bh_sw < 0.10).sum()),
            "survivors_with_swap": P[P.p_bh_sw < 0.10][["test", "net_sw_mean_bp", "net_sw_t", "net_sw_sharpe",
                                                         "p_bh_sw"]].to_dict("records"),
            "permutation": PM.to_dict("records"),
            "random_timing": RT.to_dict("records"),
            "meta": {k: {kk: vv for kk, vv in v.items() if kk not in ("alt_files",)} for k, v in metas.items()}}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    print(PM.round(4).to_string())
    print(RT.round(3).to_string())
    print(json.dumps({k: v for k, v in summ.items() if k != "meta"}, indent=1, default=str))


if __name__ == "__main__":
    main()
