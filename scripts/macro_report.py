"""Macro-fundamental currency strategies (run 18) from the series exported by scripts/macro_lab.py.

PRE-REGISTERED PRIMARY TESTS (monthly rebalancing at month-end, 9 currencies vs USD, long the 3 best / short the
3 worst currencies with weights +-1/3, excess return = spot return + carry from the point-in-time 3-month rate
differential where rates exist, costs = (half spread + slippage 0.25 x spread) x turnover + 0.35 bp commission per
unit traded; Benjamini-Hochberg over the four):
  EM   economic momentum (Dahlquist & Hasseltoft 2020): for industrial production, retail sales, unemployment
       (sign reversed), CPI and PPI, the change of each series over the past 12, 24 and 36 months (averaged),
       relative to the US, z-scored across currencies and averaged over the available series (>= 2)
  OG   output gap (Colacito, Riddiough & Sarno): log industrial production minus its one-sided HP trend
       (lambda 129,600, data available at the decision date only), relative to the US
  TR   Taylor-rule fundamentals (after Molodtsova & Papell): implied rate 1.5 x 12-month CPI inflation + 0.5 x output
       gap, relative to the US
  ML   elastic net (after Filippou, Rapach, Taylor & Zhou): panel regression of next month's excess return on the
       standardised signals above plus carry, 3- and 12-month FX momentum and 5-year value; expanding window,
       first forecast after 36 months; rank by forecast
Diagnostics (not in the BH family): carry, 12-month FX momentum, 5-year value, the output-gap time-series version.
Macro data are revised, not real-time vintages; each series is used only from 2 months (monthly) or 5 months
(quarterly) after its observation date.
Writes results/reports/macro_summary/{tests.csv, monthly.csv, coverage.csv, summary.json}."""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg  # noqa: E402

IN = Path(os.environ.get("MR_IN", RESULTS_DIR / "reports" / "macro"))
OUT = Path(os.environ.get("MR_OUT", RESULTS_DIR / "reports" / "macro_summary"))
VARS = ("ip", "cpi", "unemp", "retail", "ppi")
SLIP_FRAC, COMM = 0.25, 0.35e-4


def asof_monthly(s, months):
    """Monthly as-of series: value usable at month-end t = latest observation whose availability month <= t."""
    if s is None or not len(s):
        return None
    s = s.sort_index()
    freq_q = (pd.Series(s.index).diff().dt.days.median() or 30) > 45
    lag = 5 if freq_q else 2
    avail = pd.PeriodIndex(s.index, freq="M") + lag
    a = pd.Series(s.to_numpy(), index=avail).groupby(level=0).last()
    return a.reindex(months, method="ffill")


def hp_gap_one_sided(x, lam=129600.0, min_obs=60):
    """Gap at each date from an HP filter estimated on data up to that date only (log levels)."""
    from scipy.sparse import diags, identity
    from scipy.sparse.linalg import spsolve
    x = pd.Series(x).dropna()
    out = pd.Series(np.nan, index=x.index)
    v = x.to_numpy()
    for k in range(min_obs, len(v) + 1):
        y = v[max(0, k - 240):k]                                  # last 20 years
        n = len(y)
        D = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
        trend = spsolve((identity(n) + lam * D.T @ D).tocsc(), y)
        out.iloc[k - 1] = y[-1] - trend[-1]
    return out


def load():
    fx, macro, cov = {}, {}, []
    for f in sorted(glob.glob(str(IN / "fx_*.csv"))):
        c = os.path.basename(f)[3:6]
        fx[c] = pd.read_csv(f, index_col=0)
        fx[c].index = pd.PeriodIndex(fx[c].index, freq="M")
    for f in sorted(glob.glob(str(IN / "macro_*_*.csv"))):
        c, var = os.path.basename(f)[6:-4].split("_")
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        macro[(c, var)] = d["value"]
        cov.append({"ccy": c, "var": var, "sid": d["sid"].iloc[0], "first": str(d.index[0].date()),
                    "last": str(d.index[-1].date()), "n": len(d)})
    return fx, macro, pd.DataFrame(cov).drop_duplicates(["ccy", "var"])


def xs_z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def portfolio(signal, ret, spr, n_side=3, ts_mode=False):
    """Monthly long-short of the currencies ranked by signal at t, held over t+1. Returns net, gross series."""
    months = signal.index
    w_prev = pd.Series(0.0, index=signal.columns)
    net, gross = {}, {}
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        s = signal.loc[t].dropna()
        r = ret.loc[t1].reindex(s.index) if t1 in ret.index else None
        if r is None:
            continue
        s = s[r.notna()]
        if ts_mode:
            if len(s) < 3:
                continue
            w = pd.Series(np.where(s > 0, 1.0, -1.0) / len(s), index=s.index)
        else:
            if len(s) < 2 * n_side:
                continue
            o = s.sort_values()
            w = pd.Series(0.0, index=s.index)
            w[o.index[-n_side:]] = 1.0 / n_side
            w[o.index[:n_side]] = -1.0 / n_side
        w_full = w.reindex(signal.columns).fillna(0.0)
        turn = (w_full - w_prev).abs()
        cost = float((turn * ((0.5 + SLIP_FRAC) * spr.loc[t].reindex(signal.columns).fillna(spr.loc[t].median()) + COMM)).sum())
        g = float((w * ret.loc[t1].reindex(w.index)).sum())
        gross[t1], net[t1] = g, g - cost
        w_prev = w_full
    return pd.Series(net), pd.Series(gross)


def tstats(x):
    x = pd.Series(x).dropna()
    if len(x) < 12 or x.std() == 0:
        return {"months": len(x)}
    yrs = x.groupby(x.index.year).sum()
    return {"months": len(x), "mean_bp": float(x.mean() * 1e4), "sharpe": float(x.mean() / x.std() * np.sqrt(12)),
            "t": float(x.mean() / x.std() * np.sqrt(len(x))), "share_years_positive": float((yrs > 0).mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fx, macro, cov = load()
    cov.to_csv(OUT / "coverage.csv", index=False)
    if len(fx) < 6:
        json.dump({"tests": 0, "fx": list(fx)}, open(OUT / "summary.json", "w"))
        print("not enough currencies", list(fx))
        return
    ccys = sorted(fx)
    months = pd.period_range(min(f.index.min() for f in fx.values()), max(f.index.max() for f in fx.values()), freq="M")
    ret = pd.DataFrame({c: fx[c]["spot_ret"] + fx[c]["rate_diff"].shift(1).fillna(0) / 12 for c in ccys}).reindex(months)
    spot = pd.DataFrame({c: fx[c]["spot_ret"] for c in ccys}).reindex(months)
    spr = pd.DataFrame({c: fx[c]["spr"] for c in ccys}).reindex(months).ffill()
    carry = pd.DataFrame({c: fx[c]["rate_diff"] for c in ccys}).reindex(months)
    long_m = pd.period_range("1990-01", months[-1], freq="M")
    A = {(c, v): asof_monthly(macro.get((c, v)), long_m) for c in ccys + ["USD"] for v in VARS}
    # economic momentum
    zs = []
    for v in VARS:
        rows = {}
        for c in ccys:
            x, u = A.get((c, v)), A.get(("USD", v))
            if x is None or u is None:
                continue
            def mom(s):
                if v == "unemp":
                    return -sum(s - s.shift(k) for k in (12, 24, 36)) / 3
                ls = np.log(s.where(s > 0))
                return sum(ls - ls.shift(k) for k in (12, 24, 36)) / 3
            rows[c] = mom(x) - mom(u)
        if len(rows) >= 6:
            zs.append(xs_z(pd.DataFrame(rows)))
    if zs:
        Z = pd.concat(zs, keys=range(len(zs)))
        cnt = Z.notna().groupby(level=1).sum()
        EM = Z.groupby(level=1).mean().where(cnt >= 2)
    else:
        EM = pd.DataFrame(columns=ccys)
    EM = EM.reindex(months)
    # output gap and Taylor rule
    gaps, infl = {}, {}
    for c in ccys + ["USD"]:
        ip = A.get((c, "ip"))
        if ip is not None:
            g = hp_gap_one_sided(np.log(ip.where(ip > 0)).loc[:months[-1]])
            gaps[c] = g.reindex(long_m)
        cp = A.get((c, "cpi"))
        if cp is not None:
            infl[c] = np.log(cp.where(cp > 0)).diff(12)
    OG = pd.DataFrame({c: gaps[c] - gaps["USD"] for c in ccys if c in gaps and "USD" in gaps}).reindex(months)
    TR = pd.DataFrame({c: (1.5 * infl[c] + 0.5 * gaps[c]) - (1.5 * infl["USD"] + 0.5 * gaps["USD"])
                       for c in ccys if c in gaps and c in infl and "USD" in gaps and "USD" in infl}).reindex(months)
    MOM12 = spot.rolling(12).sum()
    MOM3 = spot.rolling(3).sum()
    VAL = pd.DataFrame({c: -(spot[c].rolling(60, min_periods=36).sum() +
                             (infl[c] - infl["USD"]).reindex(months).rolling(60, min_periods=36).mean() * 5)
                        for c in ccys if c in infl and "USD" in infl}).reindex(months)
    # elastic net, expanding window
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    feats = {"em": EM, "og": OG, "tr": TR, "carry": carry, "mom3": MOM3, "mom12": MOM12, "val": VAL}
    ML = pd.DataFrame(np.nan, index=months, columns=ccys)
    panel = []
    for t in months:
        for c in ccys:
            row = {k: (f.at[t, c] if (c in f.columns and t in f.index) else np.nan) for k, f in feats.items()}
            row.update({"t": t, "c": c})
            panel.append(row)
    Pn = pd.DataFrame(panel)
    nxt = ret.shift(-1)
    Pn["y"] = [nxt.at[t, c] if t in nxt.index else np.nan for t, c in zip(Pn.t, Pn.c)]
    fcols = list(feats)
    Pn[fcols] = Pn.groupby("t")[fcols].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0)
    Pn[fcols] = Pn[fcols].fillna(0.0)
    first = months[0] + 36
    for t in months:
        if t < first:
            continue
        tr = Pn[(Pn.t < t) & Pn.y.notna()]
        if len(tr) < 200:
            continue
        sc = StandardScaler().fit(tr[fcols])
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, alphas=30, max_iter=5000).fit(sc.transform(tr[fcols]), tr.y)
        te = Pn[Pn.t == t]
        ML.loc[t, te.c.to_numpy()] = m.predict(sc.transform(te[fcols]))
    tests = {"EM": EM, "OG": OG, "TR": TR, "ML": ML}
    diag = {"carry": carry, "fx_momentum_12m": MOM12, "value_5y": VAL}
    rows, monthly = [], {}
    for name, S in list(tests.items()) + list(diag.items()):
        n, g = portfolio(S.reindex(columns=ccys), ret, spr)
        monthly[name] = n
        r = {"test": name, "primary": name in tests}
        r.update({f"net_{k}" if k != "months" else "months": v for k, v in tstats(n).items()})
        r.update({f"gross_{k}": v for k, v in tstats(g).items() if k != "months"})
        if len(n):
            r["first"], r["last"] = str(n.index.min()), str(n.index.max())
        rows.append(r)
    n, g = portfolio(OG.reindex(columns=ccys), ret, spr, ts_mode=True)
    monthly["OG_timeseries"] = n
    rows.append({"test": "OG_timeseries", "primary": False, **{f"net_{k}" if k != "months" else "months": v for k, v in tstats(n).items()},
                 **{f"gross_{k}": v for k, v in tstats(g).items() if k != "months"}})
    T = pd.DataFrame(rows)
    P = T[T.primary].copy()
    P["p"] = st.norm.sf(P.net_t.astype(float).fillna(0))
    P["p_bh"] = benjamini_hochberg(P.p.to_numpy())
    T = T.merge(P[["test", "p", "p_bh"]], on="test", how="left")
    T.to_csv(OUT / "tests.csv", index=False)
    M = pd.DataFrame(monthly)
    M.index = M.index.astype(str)
    M.to_csv(OUT / "monthly.csv")
    summ = {"currencies": ccys, "months": [str(months[0]), str(months[-1])], "bh_survivors_q10": int((P.p_bh < 0.10).sum()),
            "tests": T.to_dict("records"), "coverage_rows": int(len(cov))}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    print(cov.to_string())


if __name__ == "__main__":
    main()
