"""Round 13 combine (docs/long_preregistration.md): portfolios, Newey-West t of monthly returns, Benjamini-Hochberg,
the long-only benchmark and the placebos for part A; exit-variant selection on 2008-2018 and one check on
2019-2025 for part B. Usage: python scripts/long_combine.py --inp DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_lab as LL                                                       # noqa: E402

FIT = ("2008-01", "2018-12")
CHK = ("2019-01", "2025-12")
SYSTEMS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12"]
CONTROLS = ["PL1", "PL2", "BH"]
NAMES = {"L1": "Turtle System 2 (55/20, pyramids)", "L2": "Turtle System 1 (20/10, pyramids)",
         "L3": "Clenow core trend (50-day high, 3N trail)", "L4": "Donchian 100/50", "L5": "MA 50/200 always in",
         "L6": "Time-series momentum 12m", "L7": "Time-series momentum 1/3/12m", "L8": "Faber 10-month average",
         "L9": "EWMAC (Carver)", "L10": "Squeeze release, 3N trail", "L11": "Weekly breakout 20/10",
         "L12": "EWMAC with carry filter", "PL1": "Coin-flip entries, 3N trail", "PL2": "Random-sign TSMOM",
         "BH": "Long-only benchmark"}


def nw_t(x, lags=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12 or x.std() == 0:
        return np.nan
    e = x - x.mean()
    lrv = (e @ e) / n
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(lrv / n)) if lrv > 0 else np.nan


def months(a, b):
    return pd.period_range(a, b, freq="M").astype(str)


def split(s: pd.Series):
    s = s.reindex(months(FIT[0], CHK[1])).fillna(0.0)
    return s[s.index <= FIT[1]], s[s.index >= CHK[0]]


def bh(p, q=0.10):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if not ok.any():
        return out
    pv = p[ok]
    m = len(pv)
    o = np.argsort(pv)
    thr = q * np.arange(1, m + 1) / m
    passed = pv[o] <= thr
    kmax = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    sel = np.zeros(m, bool)
    sel[o[:kmax]] = True
    out[np.flatnonzero(ok)] = sel
    return out


def row_stats(net: pd.Series, gross: pd.Series, swp: pd.Series):
    f, c = split(net)
    sd = f.std()
    sc = 0.10 / (sd * math.sqrt(12)) if sd > 0 else np.nan
    gf, gc = split(gross)
    wf, wc = split(swp)
    x = pd.concat([f, c]) * sc
    eq = x.cumsum()
    r = {"fit_mean_pct": f.mean() * sc * 100, "fit_t": nw_t(f), "check_mean_pct": c.mean() * sc * 100, "check_t": nw_t(c),
         "fit_sharpe": f.mean() / f.std() * math.sqrt(12) if f.std() > 0 else np.nan,
         "check_sharpe": c.mean() / c.std() * math.sqrt(12) if c.std() > 0 else np.nan,
         "check_gross_mean_pct": gc.mean() * sc * 100, "check_gross_t": nw_t(gc),
         "fit_gross_mean_pct": gf.mean() * sc * 100,
         "check_swap_mean_pct": wc.mean() * sc * 100, "fit_swap_mean_pct": wf.mean() * sc * 100,
         "max_dd_pct_at_10vol": float((eq - eq.cummax()).min() * 100),
         "check_ann_return_pct_at_10vol": c.mean() * sc * 1200,
         "by_year_pct_at_10vol": {int(y): round(float(v * sc * 100), 1) for y, v in
                                  x.groupby(x.index.str[:4]).sum().items()}}
    r["p"] = float(1 - st.norm.cdf(r["check_t"])) if np.isfinite(r["check_t"]) else np.nan
    return r, sc


def trade_stats(T: pd.DataFrame, fx: bool):
    if T is None or not len(T):
        return {}
    T = T.copy()
    T["period"] = np.where(pd.to_datetime(T.t_out, utc=True) <= pd.Timestamp("2018-12-31 23:59", tz="UTC"), "fit", "check")
    out = {}
    for per, g in T.groupby("period"):
        if not len(g):
            continue
        win = g[g.net > 0]
        los = g[g.net <= 0]
        gs = g.gross.sort_values(ascending=False)
        top = gs.iloc[:max(1, int(round(0.1 * len(gs))))].sum()
        cost = (g.gross + g.swap - g.net).sum()
        out[per] = {"trades": int(len(g)), "median_hold_days": float(g.hold.median()), "win_rate": float(len(win) / len(g)),
                    "avg_win_pips": float(win.pips.mean()) if len(win) else np.nan,
                    "avg_loss_pips": float(los.pips.mean()) if len(los) else np.nan,
                    "avg_win_R": float(win.net.mean() / LL.RISK) if len(win) else np.nan,
                    "avg_loss_R": float(los.net.mean() / LL.RISK) if len(los) else np.nan,
                    "payoff": float(win.net.mean() / -los.net.mean()) if len(win) and len(los) and los.net.mean() < 0 else np.nan,
                    "mean_R": float(g.net.mean() / LL.RISK), "median_abs_pips": float(g.pips.abs().median()),
                    "top10_share_of_gross": float(top / g.gross.sum()) if g.gross.sum() != 0 else np.nan,
                    "gross_sum_R": float(g.gross.sum() / LL.RISK), "cost_sum_R": float(cost / LL.RISK),
                    "swap_sum_R": float(g.swap.sum() / LL.RISK), "net_sum_R": float(g.net.sum() / LL.RISK)}
    return out


def part_A(inp, out):
    D = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(inp, "**", "A_daily_*.parquet"), recursive=True))],
                  ignore_index=True)
    Tf = sorted(glob.glob(os.path.join(inp, "**", "A_trades_*.parquet"), recursive=True))
    T = pd.concat([pd.read_parquet(f) for f in Tf], ignore_index=True) if Tf else pd.DataFrame()
    D["month"] = pd.to_datetime(D.date).dt.to_period("M").astype(str)
    Mo = D.groupby(["system", "symbol", "month"])[["pnl", "gpnl", "swp"]].sum().reset_index()
    Mo.to_csv(os.path.join(out, "A_monthly_by_symbol.csv"), index=False)
    groups = {"FX": LL.FX_SYMBOLS, "NONFX": list(LL.NONFX)}
    cover = {g: sorted(set(Mo.symbol) & set(s)) for g, s in groups.items()}
    rows, series = [], {}
    for g, syms in groups.items():
        for sysn in SYSTEMS + CONTROLS:
            if sysn == "L12" and g == "NONFX":
                continue
            x = Mo[(Mo.system == sysn) & Mo.symbol.isin(syms)].groupby("month")[["pnl", "gpnl", "swp"]].sum()
            if not len(x):
                continue
            r, sc = row_stats(x.pnl, x.gpnl, x.swp)
            series[(sysn, g)] = (x.pnl, sc)
            tt = trade_stats(T[(T.system == sysn) & T.symbol.isin(syms)] if len(T) else None, g == "FX")
            rows.append({"system": sysn, "name": NAMES[sysn], "row": g, "kind": "control" if sysn in CONTROLS else "system",
                         "instruments": len(set(Mo[(Mo.system == sysn) & Mo.symbol.isin(syms)].symbol)), **r,
                         **{f"{per}_{k}": v for per, d in tt.items() for k, v in d.items()}})
    R = pd.DataFrame(rows)
    # excess over the long-only benchmark for non-FX rows
    R["excess_check_t"] = np.nan
    R["excess_check_mean_pct"] = np.nan
    if ("BH", "NONFX") in series:
        b, sb = series[("BH", "NONFX")]
        for i, r in R[R.row == "NONFX"].iterrows():
            s, ss = series[(r.system, "NONFX")]
            ex = s * ss - b.reindex(s.index).fillna(0) * sb
            _, c = split(ex)
            R.loc[i, "excess_check_t"] = nw_t(c)
            R.loc[i, "excess_check_mean_pct"] = c.mean() * 100
    prim = R.kind == "system"
    R["bh_pass"] = False
    R.loc[prim, "bh_pass"] = bh(R.loc[prim, "p"].to_numpy())
    rule = (R.check_mean_pct > 0) & (R.check_t >= 1.645) & (R.fit_mean_pct > 0)
    rule &= (R.row == "FX") | ((R.excess_check_mean_pct > 0) & (R.excess_check_t >= 1.645))
    R["meets_rule"] = rule
    pl = R[(R.system.isin(["PL1", "PL2"]))]
    placebo_pass = bool(((pl.check_mean_pct > 0) & (pl.check_t >= 1.645) & (pl.fit_mean_pct > 0)).any())
    R["counts"] = prim & R.meets_rule & R.bh_pass & (not placebo_pass)
    R.to_csv(os.path.join(out, "partA.csv"), index=False)
    # the combined 33-instrument book (information only)
    allr = []
    for sysn in SYSTEMS + CONTROLS:
        x = Mo[Mo.system == sysn].groupby("month")[["pnl", "gpnl", "swp"]].sum()
        if len(x):
            r, _ = row_stats(x.pnl, x.gpnl, x.swp)
            allr.append({"system": sysn, "name": NAMES[sysn], **r})
    pd.DataFrame(allr).to_csv(os.path.join(out, "partA_all33_info.csv"), index=False)
    # per instrument (information only)
    per = []
    for (sysn, sym), x in Mo.groupby(["system", "symbol"]):
        f, c = split(x.set_index("month").pnl)
        per.append({"system": sysn, "symbol": sym, "fit_t": nw_t(f), "check_t": nw_t(c),
                    "fit_sum": f.sum(), "check_sum": c.sum()})
    pd.DataFrame(per).to_csv(os.path.join(out, "partA_by_symbol.csv"), index=False)
    summ = {"rows": int(prim.sum()), "counts": int(R.counts.sum()), "meets_rule": int((prim & R.meets_rule).sum()),
            "check_t_ge_1645": int((prim & (R.check_t >= 1.645)).sum()), "placebo_pass": placebo_pass,
            "coverage": cover}
    return R, summ


def part_B(inp, out):
    fs = sorted(glob.glob(os.path.join(inp, "**", "B_monthly_*.parquet"), recursive=True))
    if not fs:
        return None, {}
    B = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for c in B.columns:
        if B[c].dtype == np.float32:
            B[c] = B[c].astype(float)
    if "month" not in B.columns:
        B = B.rename(columns={"ts": "month", "index": "month"})
    B["uid"] = B.unit + "|" + B.tf
    P = B.groupby(["uid", "variant", "month"])[["pnl", "gpnl", "swp", "n", "win", "sw", "sl", "hold", "pips", "apips"]].sum()
    nsym = B.groupby("uid").symbol.nunique()
    rows = []
    for (uid, v), g in P.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        f, c = split(g.pnl)
        gf, gc = split(g.gpnl)
        fm = g[g.index <= FIT[1]]
        cm = g[g.index >= CHK[0]]
        rows.append({"uid": uid, "variant": v, "fit_t": nw_t(f), "check_t": nw_t(c), "fit_mean": f.mean(),
                     "check_mean": c.mean(), "fit_sd": f.std(), "check_gross_t": nw_t(gc), "fit_gross_t": nw_t(gf),
                     "fit_trades": fm.n.sum(), "check_trades": cm.n.sum(),
                     "fit_hold": fm.hold.sum() / max(fm.n.sum(), 1), "check_hold": cm.hold.sum() / max(cm.n.sum(), 1),
                     "fit_win": fm.win.sum() / max(fm.n.sum(), 1),
                     "fit_avg_win_R": fm.sw.sum() / max(fm.win.sum(), 1) / LL.RISK,
                     "fit_avg_loss_R": fm.sl.sum() / max(fm.n.sum() - fm.win.sum(), 1) / LL.RISK,
                     "fit_abs_pips": fm.apips.sum() / max(fm.n.sum(), 1)})
    V = pd.DataFrame(rows)
    V["kind"] = np.where(V.uid.str.startswith("PB_RANDOM"), "placebo", "unit")
    V["instruments"] = V.uid.map(nsym)
    V.to_csv(os.path.join(out, "partB_variants.csv"), index=False)
    # selection: highest fit t per unit (at least 30 fit trades), then the check once
    ok = V[(V.fit_trades >= 30) & np.isfinite(V.fit_t)]
    sel = ok.loc[ok.groupby("uid").fit_t.idxmax()].copy()
    sel["p"] = 1 - st.norm.cdf(sel.check_t.fillna(-9))
    sel["bh_pass"] = False
    u = sel.kind == "unit"
    sel.loc[u, "bh_pass"] = bh(sel.loc[u, "p"].to_numpy())
    sel["meets_rule"] = (sel.check_mean > 0) & (sel.check_t >= 1.645)
    pls = sel[~u]
    pl_rate = float(pls.meets_rule.mean()) if len(pls) else np.nan
    un_rate = float(sel[u].meets_rule.mean()) if u.any() else np.nan
    sel["counts"] = u & sel.meets_rule & sel.bh_pass & (not (np.isfinite(pl_rate) and pl_rate >= un_rate))
    sel.to_csv(os.path.join(out, "partB_selected.csv"), index=False)
    # do long exits beat the strategy's own (information only)
    piv_f = V[V.kind == "unit"].pivot(index="uid", columns="variant", values="fit_t")
    piv_c = V[V.kind == "unit"].pivot(index="uid", columns="variant", values="check_t")
    cmp = {}
    for v in ("T1", "T2", "T3", "T4"):
        if v in piv_f and "T0" in piv_f:
            d_f, d_c = (piv_f[v] - piv_f["T0"]).dropna(), (piv_c[v] - piv_c["T0"]).dropna()
            cmp[v] = {"units": int(len(d_f)), "share_better_fit": float((d_f > 0).mean()),
                      "share_better_check": float((d_c > 0).mean()), "median_dt_fit": float(d_f.median()),
                      "median_dt_check": float(d_c.median())}
    byv = V[V.kind == "unit"].groupby("variant").agg(units=("uid", "size"), med_fit_t=("fit_t", "median"),
                                                     med_check_t=("check_t", "median"),
                                                     share_check_t_pos=("check_t", lambda s: float((s > 0).mean())),
                                                     med_hold_days=("fit_hold", "median"), med_win=("fit_win", "median"),
                                                     med_avg_win_R=("fit_avg_win_R", "median"),
                                                     med_avg_loss_R=("fit_avg_loss_R", "median"),
                                                     med_abs_pips=("fit_abs_pips", "median"))
    byv.to_csv(os.path.join(out, "partB_by_variant.csv"))
    summ = {"units": int(u.sum()), "counts": int(sel.counts.sum()), "meets_rule": int(sel[u].meets_rule.sum()),
            "share_meets_rule": un_rate, "placebo_units": int((~u).sum()), "placebo_meets_rule": int(pls.meets_rule.sum()),
            "placebo_share": pl_rate, "sel_fit_t_median": float(sel[u].fit_t.median()),
            "sel_check_t_median": float(sel[u].check_t.median()), "variant_chosen": sel[u].variant.value_counts().to_dict(),
            "long_exit_vs_T0": cmp}
    return sel, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R, sA = part_A(a.inp, a.out)
    sel, sB = part_B(a.inp, a.out)
    summ = {"part_A": sA, "part_B": sB}
    json.dump(summ, open(os.path.join(a.out, "long_summary.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    cols = ["system", "row", "instruments", "fit_mean_pct", "fit_t", "check_mean_pct", "check_t", "check_gross_t",
            "excess_check_t", "check_sharpe", "max_dd_pct_at_10vol", "bh_pass", "meets_rule", "counts"]
    print(R[cols].round(2).to_string())
    print(json.dumps(summ, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
