"""Front-running balanced-fund rebalancing (Harvey, Mazzoleni & Melone 2025, "The Unintended Consequences of
Rebalancing", NBER w33554): when stocks have outperformed bonds, 60/40 funds sell stocks and buy bonds near month
end (calendar rebalancing) or once the drift exceeds a band (threshold rebalancing), and equities underperform bonds
over the next days. The paper reports a front-running Sharpe of about 1.1 (1997-2023). Rules fixed before any result,
following the paper's construction; development data only (to 2025-12-31). Daily Yahoo closes.

FIXED BEFORE ANY RESULT:
  Proxies: S&P 500 = SPY (total return), 10-year Treasury = IEF (total return), 2003-2025; the spread return is
    SPY minus IEF. A simulated 60/40 portfolio drifts with the two returns.
  Calendar signal: rebalanced to 60/40 on the last trading day of each month; drift d = equity weight - 0.60 at the
    previous close. R1: in the last 4 trading days of the month hold -clip(d / 1.5%, -1, 1) units of the spread;
    on the first trading day of the next month hold +clip(d_me / 1.5%, -1, 1), d_me being the drift at the
    month-end close before rebalancing (the paper's reversal leg).
  Threshold signal: for each band delta = 0.1%, 0.2%, ..., 2.5%, a 60/40 portfolio rebalanced whenever |d| > delta;
    the signal is the average drift over the 25 bands at the previous close. R2: hold -clip(signal / 1.5%, -1, 1)
    units every day.
  R3: the average of the R1 and R2 positions (the paper's combined strategy).
  E1-E3: the same positions in the equity leg only (an S&P 500 index CFD at a prop firm), with CFD financing
    (USD rate + 2.5% a year on longs, rate - 2.5% on shorts) on the position held overnight.
  Costs: 1 bp per unit of position change per leg (ETF / futures spread); 2 bp for the CFD.
PRE-REGISTERED TESTS (one-sided): mean daily net return > 0 (Newey-West t, 5 lags) for R1-R3 and E1-E3, over
  2003-2025 and separately after the paper's sample (2023-03-18 onwards). Benjamini-Hochberg over the six.
Usage: python scripts/rebalance_lab.py --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

TARGET, SCALE = 0.60, 0.015
BANDS = np.arange(1, 26) / 1000.0
PAPER_END = pd.Timestamp("2023-03-18")


def yahoo(t):
    import yfinance as yf
    d = yf.download(t, start="2002-07-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    d = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    return d.dropna()


def drifts(re, rb):
    """Calendar drift (rebalanced at each month end) and the band-averaged threshold drift, at each close."""
    idx = re.index
    month = idx.to_period("M")
    last = pd.Series(month, index=idx) != pd.Series(month, index=idx).shift(-1)
    we = TARGET
    cal, cal_pre = np.empty(len(idx)), np.empty(len(idx))
    for k in range(len(idx)):
        a, b = we * (1 + re.iloc[k]), (1 - we) * (1 + rb.iloc[k])
        we = a / (a + b)
        cal_pre[k] = we - TARGET
        cal[k] = cal_pre[k]
        if last.iloc[k]:
            we = TARGET
    W = np.full(len(BANDS), TARGET)
    thr = np.empty(len(idx))
    for k in range(len(idx)):
        a, b = W * (1 + re.iloc[k]), (1 - W) * (1 + rb.iloc[k])
        W = a / (a + b)
        thr[k] = float(np.mean(W - TARGET))
        W = np.where(np.abs(W - TARGET) > BANDS, TARGET, W)
    return pd.Series(cal_pre, idx), pd.Series(thr, idx), last


def positions(re, rb):
    cal, thr, last = drifts(re, rb)
    idx = re.index
    month = pd.Series(idx.to_period("M"), index=idx)
    # trading-day rank from the end of the month (0 = last day)
    rank_end = month.groupby(month).cumcount(ascending=False)
    first = month != month.shift(1)
    c_prev = cal.shift(1)                                    # drift known at the previous close
    r1 = pd.Series(0.0, index=idx)
    win = rank_end <= 3
    r1[win] = -np.clip(c_prev[win] / SCALE, -1, 1)
    me_drift = cal.where(last).ffill().shift(1)              # drift at the last month-end close
    r1[first] = np.clip(me_drift[first] / SCALE, -1, 1)
    r2 = -np.clip(thr.shift(1) / SCALE, -1, 1)
    r3 = 0.5 * (r1 + r2)
    return pd.DataFrame({"R1": r1, "R2": r2, "R3": r3}).fillna(0.0)


def nw_t(x, lags=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def main():
    from scipy import stats as st
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spy, ief, irx = yahoo("SPY"), yahoo("IEF"), yahoo("^IRX")
    px = pd.concat([spy.rename("spy"), ief.rename("ief")], axis=1).dropna()
    px = px[px.index <= "2025-12-31"]
    re, rb = px.spy.pct_change().fillna(0), px.ief.pct_change().fillna(0)
    P = positions(re, rb)
    rf = (irx.reindex(px.index).ffill().fillna(0) / 100).shift(1).fillna(0)
    days = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1)
    spread = re - rb
    out, rows = {}, []
    for k in ("R1", "R2", "R3"):
        p = P[k]
        turn = p.diff().abs().fillna(p.abs())
        out[k] = p * spread - turn * 2 * 1e-4                      # two legs, 1 bp each
        held = p.shift(1).fillna(0)                               # CFD position carried over the night
        fin = np.where(held > 0, -(rf + 0.025), np.where(held < 0, rf - 0.025, 0.0)) * np.abs(held) * days / 360
        out["E" + k[1]] = p * re - turn * 2e-4 + pd.Series(fin, index=px.index) * (held != 0)
    for k, x in out.items():
        x = x[x.index >= "2003-01-01"]
        for per, y in (("2003-2025", x), ("after the paper (2023-03-18 on)", x[x.index >= PAPER_END])):
            t = nw_t(y)
            act = (P[k if k in P else "R" + k[1]].reindex(y.index) != 0)
            rows.append({"test": k, "period": per, "days": int(len(y)), "active_days": int(act.sum()),
                         "mean_bp_day": float(y.mean() * 1e4), "mean_bp_active_day": float(y[act].mean() * 1e4),
                         "ann_return": float(y.mean() * 252), "sharpe": float(y.mean() / y.std() * math.sqrt(252)),
                         "t": t, "p_one_sided": float(1 - st.norm.cdf(t)),
                         "years_positive": f"{int((y.groupby(y.index.year).sum() > 0).sum())}/{y.index.year.nunique()}"})
    R = pd.DataFrame(rows)
    from cross_market_ta_lab import bh_q
    full = R.period == "2003-2025"
    R["q_bh"] = np.nan
    R.loc[full, "q_bh"] = bh_q(R.loc[full, "p_one_sided"].to_numpy())
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.DataFrame(out).to_csv(os.path.join(a.out, "daily.csv"))
    pd.set_option("display.width", 220)
    print(R.round(3).to_string())
    json.dump({"rows": len(R)}, open(os.path.join(a.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()
