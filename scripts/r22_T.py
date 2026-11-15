"""Round 22 part T: every time window, walk-forward (docs/round22_preregistration.md).

Grid per instrument: entry hour 0-23 UTC x holding {1,2,3,4,6,8,12,16,24} h x sign {long, short, fade 1h, fade 4h,
follow 1h, follow 4h} x day filter {all, Mon..Fri, Gotobi, last business day, first business day, FOMC, payroll
Friday} = 14,256 configurations. Entry and exit at the hour's first-minute mid; costs from that minute's spread.
Selection: yearly expanding-window top-5 by training t (scripts/wf22.py logic, on per-year aggregates).
Usage: python scripts/r22_T.py --symbol SYM --out DIR [--files DIR]   (--files: parquet M1 files of the symbol)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402

HOLD = (1, 2, 3, 4, 6, 8, 12, 16, 24)
SIGNS = ("long", "short", "fade1", "fade4", "follow1", "follow4")
FILTERS = ("all", "mon", "tue", "wed", "thu", "fri", "gotobi", "month_end", "month_start", "fomc", "nfp")
COLS = ["bid_o", "ask_o"]


def load_mid(sym, files):
    if files:
        fs = sorted(glob.glob(os.path.join(files, "**", "*.parquet"), recursive=True))
        parts = []
        for f in fs:
            x = pd.read_parquet(f, columns=["ts", "bid_o", "ask_o"] if "ts" in pd.read_parquet(f).columns else None)
            if "ts" in x.columns:
                x = x.set_index("ts")
            parts.append(x[["bid_o", "ask_o"]])
        m = pd.concat(parts).sort_index()
        m.index = pd.DatetimeIndex(m.index)
        if m.index.tz is None:
            m.index = m.index.tz_localize("UTC")
        m = m[~m.index.duplicated(keep="last")]
        m = m[m.index <= pd.Timestamp("2025-12-31 23:59", tz="UTC")]
    else:
        from fxlab.data.store import load_m1
        m = load_m1(sym, "2007-01-01", "2025-12-31", columns=COLS)
    mid = 0.5 * (m.bid_o + m.ask_o)
    spr = (m.ask_o - m.bid_o).clip(lower=0)
    ok = np.isfinite(mid) & (mid > 0)
    return mid[ok], spr[ok]


def pip_of(sym, px):
    if "IDX" in sym or sym.endswith("USDT") or "CMD" in sym or "TR" in sym[-6:-3]:
        return float(1e-4 * np.nanmedian(px))
    if sym.startswith("XAU"):
        return 0.1
    if sym.startswith("XAG"):
        return 0.01
    return 0.01 if sym.endswith("JPY") else 1e-4


def side_cost(px, spr, pip):
    from fxlab.execution.costs import BASE
    return (0.5 * spr + BASE.slip_spread_frac * spr + BASE.slip_fixed_pips * pip) / px + BASE.commission_bps_per_side * 1e-4


def day_masks(days):
    import flows_lab as FL
    yrs = sorted(set(days.year))
    jp = FL.jp_holidays(list(range(min(yrs) - 1, max(yrs) + 2)))
    got = set(pd.DatetimeIndex(FL.gotobi_days(days.min(), days.max(), jp)))
    me = set(pd.DatetimeIndex(FL.month_end_days(days.min(), days.max())))
    ms = set()
    for m in pd.period_range(days.min(), days.max(), freq="M"):
        d = m.start_time.normalize()
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        ms.add(d)
    try:
        fo, rep = FL.fomc_dates(yrs)
        fomc = set(pd.DatetimeIndex(fo))
    except Exception as ex:                                                # noqa: BLE001
        fomc, rep = set(), {"error": repr(ex)[:100]}
    nfp = set()
    for m in pd.period_range(days.min(), days.max(), freq="M"):
        d = m.start_time.normalize()
        while d.weekday() != 4:
            d += pd.Timedelta(days=1)
        nfp.add(d)
    wd = days.weekday
    M = {"all": np.ones(len(days), bool), "mon": wd == 0, "tue": wd == 1, "wed": wd == 2, "thu": wd == 3, "fri": wd == 4,
         "gotobi": days.isin(got), "month_end": days.isin(me), "month_start": days.isin(ms), "fomc": days.isin(fomc),
         "nfp": days.isin(nfp)}
    return M, {"fomc_days": len(fomc), "fomc_report": {str(k): v for k, v in list(rep.items())[:30]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--files", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mid, spr = load_mid(a.symbol, a.files)
    pip = pip_of(a.symbol, mid.to_numpy())
    hidx = pd.date_range(mid.index.min().floor("h"), mid.index.max().ceil("h"), freq="h")
    # the first minute within the first 5 minutes of each hour
    hm = mid.groupby(mid.index.floor("h")).first()
    first_min = pd.Series(mid.index, index=mid.index).groupby(mid.index.floor("h")).first()
    okh = (first_min - first_min.index).dt.total_seconds() <= 300
    P = hm[okh].reindex(hidx).to_numpy(float)
    Sp = spr.groupby(spr.index.floor("h")).first()[okh].reindex(hidx).to_numpy(float)
    C = side_cost(P, Sp, pip)
    hour = hidx.hour.to_numpy()
    dates = hidx.tz_localize(None).normalize()
    has = pd.Series(np.isfinite(P), index=dates).groupby(level=0).any()
    days = pd.DatetimeIndex(has.index[has.to_numpy()])
    days = days[days <= wf22.DEV_END]
    dpos = {d: i for i, d in enumerate(days)}
    masks, mrep = day_masks(days)
    yrs = days.year.to_numpy()
    years = sorted(set(yrs))
    first = max(2010, int(years[0]) + 3)
    n_cfg = 24 * len(HOLD) * len(SIGNS) * len(FILTERS)
    # per-year aggregates for every configuration
    Ys = {y: np.flatnonzero(yrs == y) for y in years}
    agg = {y: np.zeros((n_cfg, 4)) for y in years}                         # sum, sumsq, active, valid
    cfgs = []

    def series(h, k, sgn):
        t = np.flatnonzero(hour == h)
        t = t[t + k < len(P)]
        g = P[t + k] / P[t] - 1
        cost = C[t] + C[t + k]
        if sgn == "long":
            s = np.ones(len(t))
        elif sgn == "short":
            s = -np.ones(len(t))
        else:
            L = 1 if sgn.endswith("1") else 4
            prev = np.full(len(t), np.nan)
            ok = t - L >= 0
            prev[ok] = P[t[ok]] / P[t[ok] - L] - 1
            s = np.sign(prev) * (-1 if sgn.startswith("fade") else 1)
        net = s * g - np.abs(s) * cost
        out = np.full(len(days), np.nan)
        dd = dates[t]
        idx = np.array([dpos.get(d, -1) for d in dd])
        keep = idx >= 0
        out[idx[keep]] = net[keep]
        return out

    ci = 0
    for h in range(24):
        for k in HOLD:
            for sgn in SIGNS:
                base = series(h, k, sgn)
                for fl in FILTERS:
                    x = np.where(masks[fl], base, np.where(np.isfinite(base), 0.0, np.nan))
                    xv = np.where(np.isfinite(x), x, 0.0)
                    for y, ii in Ys.items():
                        v = x[ii]
                        fin = np.isfinite(v)
                        agg[y][ci] = (xv[ii].sum(), (xv[ii] ** 2).sum(), float(((v != 0) & fin).sum()), float(fin.sum()))
                    cfgs.append((h, k, sgn, fl))
                    ci += 1
    # walk-forward on the aggregates
    cs = np.zeros((n_cfg, 4))
    out5 = pd.Series(np.nan, index=days)
    out1 = pd.Series(np.nan, index=days)
    choices = {}
    cache = {}
    for y in years:
        if y >= first:
            n = np.maximum(cs[:, 3], 1)
            mu = cs[:, 0] / n
            var = np.maximum(cs[:, 1] / n - mu ** 2, 1e-30) * n / np.maximum(n - 1, 1)
            t = mu / np.sqrt(var / n)
            t[(cs[:, 2] < 30) | (cs[:, 3] < 50)] = -np.inf
            pick = [int(i) for i in np.argsort(-t)[:5] if np.isfinite(t[i])]
            ii = Ys[y]
            vals = []
            for i in pick:
                if i not in cache:
                    h, k, sgn, fl = cfgs[i]
                    base = series(h, k, sgn)
                    cache[i] = np.where(masks[fl], base, np.where(np.isfinite(base), 0.0, np.nan))
                vals.append(np.where(np.isfinite(cache[i][ii]), cache[i][ii], 0.0))
            if vals:
                out5.iloc[ii] = np.mean(vals, axis=0)
                out1.iloc[ii] = vals[0]
            else:
                out5.iloc[ii] = 0.0
                out1.iloc[ii] = 0.0
            choices[str(y)] = [{"cfg": "|".join(map(str, cfgs[i])), "train_t": round(float(t[i]), 2)} for i in pick]
        cs += agg[y]
    s5, s1 = out5.dropna(), out1.dropna()
    st, st1 = wf22.stream_stats(s5), wf22.stream_stats(s1)
    row = {"part": "T", "family": "T_time_windows", "symbol": a.symbol, "placebo": False, "n_cfg": n_cfg, **st,
           "top1_p2_mean": st1["p2_mean"], "top1_p2_t": st1["p2_t"], "own": wf22.counts_rule(st),
           "choices": json.dumps({str(k): [c["cfg"] for c in v] for k, v in choices.items()})[:3000]}
    pd.DataFrame([row]).to_csv(os.path.join(a.out, f"T_row_{a.symbol}.csv"), index=False)
    pd.DataFrame({"T": s5, "T_top1": s1}).to_parquet(os.path.join(a.out, f"T_series_{a.symbol}.parquet"))
    json.dump({"symbol": a.symbol, "days": len(days), "first": str(days.min().date()), "last": str(days.max().date()),
               "choices": choices, "masks": {k: int(v.sum()) for k, v in masks.items()}, **mrep},
              open(os.path.join(a.out, f"T_meta_{a.symbol}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: row[k] for k in ("symbol", "oos_first", "p1_mean", "p2_mean", "p2_t", "own")}, default=str))


if __name__ == "__main__":
    main()
