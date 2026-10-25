"""Aggregate scripts/lit2_lab.py (run 22): the 11 pre-registered round-2 literature tests and the diagnostics.

Metals tests (S1, G1-G3, A1, A2, SEAS): one return per event from the lab. FX tests (DC, FM1, FM2, WM) are built
here from the exported daily bid/ask closes of the 9 USD pairs:
  * daily excess return of each foreign currency vs USD = log mid change (+ = foreign appreciates) + the
    point-in-time 3-month rate differential (foreign - USD) / 365 per calendar day (SEK, NOK: no rate data, spot
    only); summed to months / weeks
  * costs = (half spread + 0.25 x spread slippage) x turnover + 0.35 bp commission per unit traded, spread = the
    median close spread of the last 5 days before the rebalancing date
  * DC, FM1, FM2 use the 7 currencies with rate data; WM uses all 9
Reported per test: events, mean (bp), t, Sharpe, share of years positive, gross; net with the rate swap where the
lab recorded one (net_sw). One-sided p-values (positive = the effect as published), BH over the 11.
Writes results/reports/lit2_summary/{tests.csv, by_year.csv, series.csv, summary.json}."""
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

IN = Path(os.environ.get("L2_IN", RESULTS_DIR / "reports" / "lit2"))
OUT = Path(os.environ.get("L2_OUT", RESULTS_DIR / "reports" / "lit2_summary"))
PRIMARY = ["S1", "G1", "G2", "G3", "A1", "A2", "SEAS", "DC", "FM1", "FM2", "WM"]
DESC = {"S1": "silver fix: short XAGUSD 11:45 -> 12:15 London",
        "G1": "gold PM fix: short XAUUSD 14:45 -> 15:15 London", "G2": "gold AM fix: short XAUUSD 10:15 -> 10:45 London",
        "G3": "gold PM fix: follow the 15:00 -> 15:03 move, 15:03 -> 15:15",
        "A1": "gold Asian hours: long 18:00 New York -> 08:00 London",
        "A2": "gold Western hours: short 08:00 London -> 16:55 New York",
        "SEAS": "gold long in September and November", "DC": "dollar carry (7 currencies), monthly",
        "FM1": "dollar-factor momentum (3-month), monthly", "FM2": "carry-factor momentum (3-month), monthly",
        "WM": "weekly momentum: 4-week lookback, 4-week hold, top/bottom 3 of 9",
        "A1_silver": "silver Asian hours (diagnostic)", "A2_silver": "silver Western hours (diagnostic)",
        "SEAS_silver": "silver September / November (diagnostic)", "carry_factor": "carry factor, always (diagnostic)",
        "PF_all": "long foreign currencies 2 days before every FOMC (diagnostic)",
        "PF_change_lookahead": "dollar 2 days before FOMC rate changes, REALISED decision (look-ahead diagnostic)"}
PER_YEAR = {"SEAS": 2, "SEAS_silver": 2, "DC": 12, "FM1": 12, "FM2": 12, "carry_factor": 12, "WM": 13,
            "PF_all": 8, "PF_change_lookahead": 2}
SPLIT = {"S1": ("2014-08-14", "silver fix replaced by the LBMA Silver Price auction (15 Aug 2014)"),
         "G1": ("2015-03-19", "gold fix replaced by the LBMA Gold Price auction (20 Mar 2015)"),
         "G2": ("2015-03-19", "same"), "G3": ("2015-03-19", "same")}
RATE_CCYS = ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
PAIR = {"EUR": "EURUSD", "GBP": "GBPUSD", "JPY": "USDJPY", "AUD": "AUDUSD", "CAD": "USDCAD", "CHF": "USDCHF",
        "SEK": "USDSEK", "NOK": "USDNOK", "NZD": "NZDUSD"}
SLIP, COMM = 0.25, 0.35e-4


def stats(x, per_year):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 6 or x.std() == 0:
        return {"events": n}
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    return {"events": n, "mean_bp": float(x.mean() * 1e4), "t": float(x.mean() / x.std() * np.sqrt(n)),
            "sharpe": float(x.mean() / x.std() * np.sqrt(per_year)), "share_years_positive": float((yrs > 0).mean()),
            "worst_bp": float(x.min() * 1e4)}


def fx_panel():
    """Daily excess returns, spreads and rate differentials of the foreign currencies vs USD."""
    from fxlab.data.rates import rate_diff
    xr, spot, spr, rdf = {}, {}, {}, {}
    for f in sorted(glob.glob(str(IN / "daily_*.csv"))):
        c = os.path.basename(f)[6:9]
        d = pd.read_csv(f, index_col=0)
        idx = pd.to_datetime(d.index, utc=True)
        d.index = idx.tz_convert(None).normalize()
        d = d[~d.index.duplicated(keep="last")]
        mid = 0.5 * (d.bid_c + d.ask_c)
        sign = 1.0 if PAIR[c].startswith(c) else -1.0
        r = sign * np.log(mid).diff()
        rd = sign * rate_diff(PAIR[c], pd.DatetimeIndex(d.index).tz_localize("UTC")).to_numpy()
        days = pd.Series(d.index, index=d.index).diff().dt.days.fillna(1).to_numpy()
        carry = pd.Series(np.nan_to_num(np.r_[np.nan, rd[:-1]]) * days / 365.0, index=d.index)   # rate known at t-1
        spot[c], xr[c] = r, r + carry
        spr[c] = ((d.ask_c - d.bid_c) / mid).rolling(5, min_periods=1).median()
        rdf[c] = pd.Series(rd, index=d.index)
    return pd.DataFrame(xr), pd.DataFrame(spot), pd.DataFrame(spr), pd.DataFrame(rdf)


def cost(turn, spr_row):
    return float((turn * ((0.5 + SLIP) * spr_row.reindex(turn.index).fillna(spr_row.median()) + COMM)).sum())


def monthly_tests(xr, spr, rdf):
    m = xr.index.to_period("M")
    XR = xr.groupby(m).sum(min_count=1)
    last = xr.groupby(m).apply(lambda g: g.index.max())
    SP = spr.reindex(last.to_numpy()).set_axis(XR.index)
    RD = rdf.reindex(last.to_numpy()).set_axis(XR.index)
    cc = [c for c in RATE_CCYS if c in XR]
    out = {k: {} for k in ("DC", "FM1", "FM2", "carry_factor")}
    gross = {k: {} for k in out}
    prev = {k: pd.Series(0.0, index=XR.columns) for k in out}
    months = list(XR.index)
    dol = XR[cc].mean(axis=1, skipna=False)
    hml = {}
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        avail = [c for c in cc if np.isfinite(XR.at[t1, c]) and np.isfinite(RD.at[t, c])]
        if len(avail) < 6:
            continue
        r1 = XR.loc[t1, avail]
        w_eq = pd.Series(1.0 / len(avail), index=avail)
        o = RD.loc[t, avail].sort_values()
        w_c = pd.Series(0.0, index=avail)
        w_c[o.index[-3:]], w_c[o.index[:3]] = 1 / 3, -1 / 3
        hml[t1] = float((w_c * r1).sum())
        past_dol = dol.loc[:t].tail(3)
        past_hml = pd.Series(hml).loc[:t].tail(3) if hml else pd.Series(dtype=float)
        pos = {"DC": 1.0 if RD.loc[t, avail].mean() > 0 else -1.0,
               "FM1": (1.0 if past_dol.sum() > 0 else -1.0) if past_dol.notna().sum() == 3 else None,
               "FM2": (1.0 if past_hml.sum() > 0 else -1.0) if len(past_hml.dropna()) == 3 and t in hml else None,
               "carry_factor": 1.0}
        for k, p_ in pos.items():
            if p_ is None:
                continue
            w = (w_eq if k in ("DC", "FM1") else w_c) * p_
            wf = w.reindex(XR.columns).fillna(0.0)
            c_ = cost((wf - prev[k]).abs(), SP.loc[t])
            g = float((w * r1).sum())
            gross[k][t1.to_timestamp()], out[k][t1.to_timestamp()] = g, g - c_
            prev[k] = wf
    return {k: (pd.Series(out[k]), pd.Series(gross[k])) for k in out}


def weekly_momentum(xr, spot, spr):
    wk = xr.index.to_period("W-FRI")
    XR, SPOT = xr.groupby(wk).sum(min_count=1), spot.groupby(wk).sum(min_count=1)
    last = xr.groupby(wk).apply(lambda g: g.index.max())
    SP = spr.reindex(last.to_numpy()).set_axis(XR.index)
    weeks = list(XR.index)
    net, gross = {}, {}
    prev = pd.Series(0.0, index=XR.columns)
    for i in range(4, len(weeks) - 4, 4):
        t = weeks[i]
        past = SPOT.iloc[i - 3:i + 1].sum(min_count=4)
        fut = XR.iloc[i + 1:i + 5].sum(min_count=4)
        ok = past.notna() & fut.notna()
        if ok.sum() < 6:
            continue
        o = past[ok].sort_values()
        w = pd.Series(0.0, index=o.index)
        w[o.index[-3:]], w[o.index[:3]] = 1 / 3, -1 / 3
        wf = w.reindex(XR.columns).fillna(0.0)
        c_ = cost((wf - prev).abs(), SP.loc[t])
        g = float((w * fut[o.index]).sum())
        end = weeks[i + 4].end_time.normalize()
        gross[end], net[end] = g, g - c_
        prev = wf
    return pd.Series(net), pd.Series(gross)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(IN / "*.parquet")))]
    metas = {}
    for f in sorted(glob.glob(str(IN / "*.json"))):
        J = json.load(open(f))
        if not J.get("skipped"):
            metas[J["pair"]] = J
    series = {}
    if frames:
        R = pd.concat(frames, ignore_index=True)
        R["day"] = pd.to_datetime(R["day"])
        R["swap_flat"] = R.get("swap_flat", 0.0).fillna(0.0)
        R["net_sw"] = R["net"] + R["swap_rate"].fillna(R["swap_flat"])
        for t in R.test.unique():
            g = R[R.test == t].groupby("day")[["net", "gross", "net_sw"]].mean()
            series[t] = g
    if glob.glob(str(IN / "daily_*.csv")):
        xr, spot, spr, rdf = fx_panel()
        for k, (n, g) in monthly_tests(xr, spr, rdf).items():
            series[k] = pd.DataFrame({"net": n, "gross": g, "net_sw": n})        # carry already in the returns
        n, g = weekly_momentum(xr, spot, spr)
        series["WM"] = pd.DataFrame({"net": n, "gross": g, "net_sw": n})
    rows, years = [], []
    for name, S in series.items():
        S = S.sort_index()
        py = PER_YEAR.get(name, 252)
        r = {"test": name, "primary": name in PRIMARY, "description": DESC.get(name, name)}
        if len(S):
            r["first"], r["last"] = str(S.index.min().date()), str(S.index.max().date())
        for col in ("net_sw", "net", "gross"):
            for k, v in stats(S[col], py).items():
                r[f"{col}_{k}" if k != "events" else "events"] = v
        if name in SPLIT:
            cut = pd.Timestamp(SPLIT[name][0])
            a, b = S[S.index <= cut]["net_sw"], S[S.index > cut]["net_sw"]
            r.update({"split": SPLIT[name][1], "before_t": stats(a, py).get("t"), "before_mean_bp": stats(a, py).get("mean_bp"),
                      "after_t": stats(b, py).get("t"), "after_mean_bp": stats(b, py).get("mean_bp"),
                      "before_gross_bp": float(S[S.index <= cut]["gross"].mean() * 1e4) if len(a) else None,
                      "after_gross_bp": float(S[S.index > cut]["gross"].mean() * 1e4) if len(b) else None})
        rows.append(r)
        y = S.groupby(S.index.year)["net_sw"].agg(["sum", "count"])
        for yr, v in y.iterrows():
            years.append({"test": name, "year": int(yr), "net_sw_sum_bp": float(v["sum"] * 1e4), "events": int(v["count"])})
    T = pd.DataFrame(rows)
    if len(T):
        P = T[T.primary == True].copy()                                 # noqa: E712
        P["p"] = st.norm.sf(P["net_sw_t"].astype(float).fillna(0)) if "net_sw_t" in P else 0.5
        P["p_bh"] = benjamini_hochberg(P.p.to_numpy())
        T = T.merge(P[["test", "p", "p_bh"]], on="test", how="left")
        T["order"] = T.test.map({k: i for i, k in enumerate(PRIMARY + list(DESC))})
        T = T.sort_values("order").drop(columns="order")
    T.to_csv(OUT / "tests.csv", index=False)
    pd.DataFrame(years).to_csv(OUT / "by_year.csv", index=False)
    long = [S.assign(test=k).rename_axis("day").reset_index() for k, S in series.items() if len(S)]
    if long:
        pd.concat(long, ignore_index=True).round(8).to_csv(OUT / "series.csv", index=False)
    surv = T[(T.get("p_bh", pd.Series(dtype=float)) < 0.10)] if len(T) and "p_bh" in T else T.iloc[:0]
    json.dump({"primary_tests": int(T.primary.sum()) if len(T) else 0, "bh_survivors_q10": int(len(surv)),
               "survivors": surv[["test", "net_sw_mean_bp", "net_sw_t", "p_bh"]].to_dict("records") if len(surv) else [],
               "meta": metas}, open(OUT / "summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = [c for c in ["test", "events", "net_sw_mean_bp", "net_sw_t", "net_sw_sharpe", "net_sw_share_years_positive",
                        "gross_mean_bp", "gross_t", "before_t", "after_t", "p_bh"] if c in T]
    print(T[cols].round(3).to_string())


if __name__ == "__main__":
    main()
