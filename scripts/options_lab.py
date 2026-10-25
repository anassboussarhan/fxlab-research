"""Can retail options SELLING be profitable? Free-data tests on the S&P 500 (the most liquid options market).

Data (Yahoo Finance via yfinance, fetched on the GitHub runner; cut at 2025-12-31, no 2026): S&P 500 (^GSPC) and
total return (^SP500TR), VIX (^VIX), 3-month VIX (^VIX3M), and the Cboe options-strategy benchmark indices where
Yahoo carries them: PUT (cash-secured monthly ATM put writing), BXM (buy-write, ATM covered call), BXMD (30-delta
buy-write), CNDR (iron condor), BFLY (iron butterfly), WPUT (weekly put write), PPUT (protective put), CLL (collar),
CMBO (combo). The Cboe indices are rules-based backtests of real SPX option prices; they do not deduct
commissions or the bid/ask spread beyond their pricing convention.

PRE-REGISTERED TESTS (one-sided, BH over O1-O5):
  O1  short S&P variance every month at the VIX (variance-swap P&L (K^2 - RV^2) / 2K in vol points, K = VIX at the
      previous month's last close, RV = realised volatility of the month's daily S&P returns), minus 1 vol point of
      cost; 1990-2025
  O2  O1 only when VIX > the realised volatility of the previous 21 trading days (the premium is visible)
  O3  O1 only when VIX < VIX3M (term structure in contango, a calm market), 2007-2025
  O4  PUT index excess return over the S&P 500 total return, risk-matched (PUT's returns vs beta x S&P): alpha > 0
  O5  CNDR index (iron condor) excess return over cash > 0
DIAGNOSTICS: statistics of every index (return, volatility, Sharpe, worst month, maximum drawdown, crisis years);
O1 at 0.5 vol point cost; the small-account question: an account selling variance at 0.1-1% of capital per vol
point, outcomes over every 5-year window (median growth, probability of losing half, of ruin).
Usage: python scripts/options_lab.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

END = "2025-12-31"
SYMS = {"GSPC": "^GSPC", "SP500TR": "^SP500TR", "VIX": "^VIX", "VIX3M": "^VIX3M", "PUT": "^PUT", "BXM": "^BXM",
        "BXMD": "^BXMD", "CNDR": "^CNDR", "BFLY": "^BFLY", "WPUT": "^WPUT", "PPUT": "^PPUT", "CLL": "^CLL",
        "CMBO": "^CMBO"}


def fetch(out):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False)
    import yfinance as yf
    data, avail = {}, {}
    for k, s in SYMS.items():
        try:
            h = yf.download(s, start="1986-01-01", end="2026-01-01", progress=False, auto_adjust=False)
            c = h["Close"]
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            c = c.dropna()
            c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.to_datetime(c.index)
            c = c[c.index <= END]
            if len(c) > 250:
                data[k] = c
                avail[k] = {"first": str(c.index[0].date()), "last": str(c.index[-1].date()), "n": int(len(c))}
            else:
                avail[k] = {"n": int(len(c))}
        except Exception as ex:                                             # noqa: BLE001
            avail[k] = {"error": str(ex)[:200]}
    D = pd.DataFrame(data)
    D.to_csv(os.path.join(out, "daily_closes.csv"))
    return D, avail


def stats_m(r, rf=None):
    r = r.dropna()
    if len(r) < 24:
        return {"months": int(len(r))}
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    yrs = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {"months": int(len(r)), "first": str(r.index[0].date()), "cagr": float(eq.iloc[-1] ** (12 / len(r)) - 1),
            "vol": float(r.std() * np.sqrt(12)), "sharpe_excess": float(ex.mean() / ex.std() * np.sqrt(12)),
            "t_excess": float(ex.mean() / ex.std() * np.sqrt(len(ex))), "worst_month": float(r.min()),
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "share_years_positive": float((yrs > 0).mean()),
            **{f"y{y}": float(yrs.get(y, np.nan)) for y in (2008, 2018, 2020, 2022)}}


def short_variance(D, cost_vp):
    """Monthly short-variance P&L in vol points and its daily mark-to-market path."""
    spx, vix = D["GSPC"].dropna(), D["VIX"].dropna()
    r = np.log(spx).diff().dropna()
    months = pd.period_range(max(r.index[0], vix.index[0]) + pd.Timedelta(days=40), END, freq="M")
    rows, daily = [], []
    rv21 = (r.rolling(21).apply(lambda x: np.sqrt(252 * np.mean(x ** 2)), raw=True) * 100)
    for m in months:
        prev = vix[vix.index < m.start_time]
        rm = r[(r.index >= m.start_time) & (r.index <= m.end_time)]
        if not len(prev) or len(rm) < 15:
            continue
        K = float(prev.iloc[-1])
        rv = float(np.sqrt(252 * np.mean(rm ** 2)) * 100)
        pnl = (K ** 2 - rv ** 2) / (2 * K)
        rvp = rv21[rv21.index < m.start_time]
        v3 = D["VIX3M"][D.index < m.start_time].dropna() if "VIX3M" in D else pd.Series(dtype=float)
        rows.append({"month": m.start_time, "K": K, "rv": rv, "gross_vp": pnl, "net_vp": pnl - cost_vp,
                     "rv_prev21": float(rvp.iloc[-1]) if len(rvp) else np.nan,
                     "vix3m": float(v3.iloc[-1]) if len(v3) else np.nan})
        # daily mark-to-market: V_t = (sum of 252 r^2 so far + days left x VIX_t^2) / N
        N = len(rm)
        cum = 0.0
        v_prev = K ** 2
        vx = vix.reindex(rm.index).ffill()
        for i, (dt, x) in enumerate(rm.items(), start=1):
            cum += 252 * (x * 100) ** 2
            left = N - i
            v_t = (cum + left * float(vx.loc[dt]) ** 2) / N if np.isfinite(vx.loc[dt]) else (cum + left * K ** 2) / N
            daily.append({"day": dt, "month": m.start_time, "pnl_vp": -(v_t - v_prev) / (2 * K)})
            v_prev = v_t
    return pd.DataFrame(rows).set_index("month"), pd.DataFrame(daily).set_index("day")


def small_account(net_vp, lam_list=(0.001, 0.0025, 0.005, 0.01), horizon=60):
    """Every rolling 5-year window: account returns = lam x vol-point P&L per month (lam = fraction of capital per
    vol point of variance notional); ruin = equity <= 0 (the strategy stops)."""
    out = []
    x = net_vp.to_numpy()
    for lam in lam_list:
        finals, half, ruin = [], 0, 0
        for s in range(0, len(x) - horizon + 1):
            eq, peak, lost_half, dead = 1.0, 1.0, False, False
            for v in x[s:s + horizon]:
                eq *= (1 + lam * v)
                if eq <= 0:
                    dead = True
                    eq = 0.0
                    break
                peak = max(peak, eq)
                lost_half |= eq <= 0.5
            finals.append(eq)
            half += lost_half
            ruin += dead
        f = np.array(finals)
        out.append({"lambda_per_vol_point": lam, "windows": int(len(f)), "median_5y_multiple": float(np.median(f)),
                    "p10_5y_multiple": float(np.percentile(f, 10)), "p90_5y_multiple": float(np.percentile(f, 90)),
                    "share_lost_half": float(half / len(f)), "share_ruined": float(ruin / len(f)),
                    "mean_monthly_return": float(lam * x.mean()), "worst_month_return": float(lam * x.min())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reports/options_lab")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D, avail = fetch(a.out)
    M = D.resample("ME").last()
    R = M.pct_change()
    rf = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from fxlab.data.rates import monthly
        rf = (monthly()["USD"] / 12).copy()
        rf.index = (rf.index.tz_convert(None) if rf.index.tz is not None else rf.index) + pd.offsets.MonthEnd(0)
    except Exception:                                                       # noqa: BLE001
        rf = None
    idx_rows = []
    for k in ["SP500TR", "GSPC", "PUT", "WPUT", "BXM", "BXMD", "CNDR", "BFLY", "PPUT", "CLL", "CMBO"]:
        if k in R:
            idx_rows.append({"index": k, **stats_m(R[k], rf)})
            if k != "SP500TR" and "SP500TR" in R:
                j = pd.concat([R[k], R["SP500TR"]], axis=1).dropna()
                b = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()
                al = j.iloc[:, 0] - b * j.iloc[:, 1]
                idx_rows[-1].update({"beta_to_sp": float(b), "alpha_ann": float(al.mean() * 12),
                                     "alpha_t": float(al.mean() / al.std() * np.sqrt(len(al)))})
    IX = pd.DataFrame(idx_rows)
    IX.to_csv(os.path.join(a.out, "indices.csv"), index=False)
    tests = []
    if {"GSPC", "VIX"} <= set(D):
        V, Dd = short_variance(D, 1.0)
        V.to_csv(os.path.join(a.out, "short_variance_monthly.csv"))
        Dd.to_csv(os.path.join(a.out, "short_variance_daily.csv"))

        def vstat(name, x, desc):
            x = x.dropna()
            yrs = x.groupby(x.index.year).sum()
            tests.append({"test": name, "description": desc, "months": int(len(x)), "first": str(x.index.min().date()),
                          "mean_vp": float(x.mean()), "t": float(x.mean() / x.std() * np.sqrt(len(x))),
                          "sharpe": float(x.mean() / x.std() * np.sqrt(12)), "share_months_positive": float((x > 0).mean()),
                          "share_years_positive": float((yrs > 0).mean()), "worst_month_vp": float(x.min()),
                          "worst_month": str(x.idxmin().date()), "sum_2008_vp": float(yrs.get(2008, np.nan)),
                          "sum_2020_vp": float(yrs.get(2020, np.nan))})
        vstat("O1", V.net_vp, "short S&P variance monthly at VIX, 1 vol pt cost")
        vstat("O1_cost05", V.gross_vp - 0.5, "same, 0.5 vol pt cost (diagnostic)")
        vstat("O1_gross", V.gross_vp, "same, no cost (diagnostic)")
        vstat("O2", V.net_vp.where(V.K > V.rv_prev21, 0.0), "only when VIX > previous 21-day realised vol")
        if V.vix3m.notna().sum() > 60:
            sub = V[V.vix3m.notna()]
            vstat("O3", sub.net_vp.where(sub.K < sub.vix3m, 0.0), "only when VIX < VIX3M (contango)")
            vstat("O1_since_2007", sub.net_vp, "O1 on the O3 sample (diagnostic)")
        SA = small_account(V.net_vp)
        SA.to_csv(os.path.join(a.out, "small_account.csv"), index=False)
        dd = Dd.pnl_vp
        json.dump({"worst_days_vp": {str(k.date()): float(v) for k, v in dd.nsmallest(10).items()}},
                  open(os.path.join(a.out, "worst_days.json"), "w"), indent=1)
    if "PUT" in R and "SP500TR" in R:
        row = IX[IX["index"] == "PUT"].iloc[0]
        tests.append({"test": "O4", "description": "PUT index alpha vs beta x S&P 500 TR", "months": int(row["months"]),
                      "mean_vp": np.nan, "t": float(row["alpha_t"]), "alpha_ann": float(row["alpha_ann"])})
    if "CNDR" in R:
        row = IX[IX["index"] == "CNDR"].iloc[0]
        tests.append({"test": "O5", "description": "CNDR iron condor index excess return over cash",
                      "months": int(row["months"]), "mean_vp": np.nan, "t": float(row["t_excess"])})
    T = pd.DataFrame(tests)
    if len(T):
        P = T[T.test.isin(["O1", "O2", "O3", "O4", "O5"])].copy()
        P["p"] = st.norm.sf(P.t.astype(float))
        from_bh = P.p.to_numpy()
        order = np.argsort(from_bh)
        q = np.empty_like(from_bh)
        m = len(from_bh)
        prev = 1.0
        for rank, i in enumerate(order[::-1]):
            k = m - rank
            prev = min(prev, from_bh[i] * m / k)
            q[i] = prev
        P["p_bh"] = q
        T = T.merge(P[["test", "p", "p_bh"]], on="test", how="left")
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump({"availability": avail}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(avail, indent=1))
    print(IX.round(3).to_string())
    print(T.round(3).to_string())
    if os.path.exists(os.path.join(a.out, "small_account.csv")):
        print(pd.read_csv(os.path.join(a.out, "small_account.csv")).round(3).to_string())


if __name__ == "__main__":
    main()
