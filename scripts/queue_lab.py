"""Queue batch Q of docs/queue_preregistration.md. 2007-2025 only.

--part-a  per-instrument M1 passes: HAR volatility forecasts; VS1 (Gotobi sizing for a constant breach risk, prop
          sims), VS2 (variance-ratio gamma choice), VS4 (cost-to-volatility gate), VS5 (volatility jumps and the
          carry crosses).
--part-b  weekly tests: T1 (slow trend on prop CFDs), BS1 (rate-expectation carry), C5E (euro COT before 2004), and
          P1 completed with Australia (via push_lab). Needs --store (the flow store).
Usage: python scripts/queue_lab.py --part-a --out OUT
       python scripts/queue_lab.py --part-b --store flow_store --out OUT"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = pd.Timestamp("2025-12-31")
FIT_END = pd.Timestamp("2018-12-31")
LOG = {}
FX7 = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")


def month_t(x):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 10:
        return np.nan
    idx = pd.DatetimeIndex(x.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    e = x.values - x.mean()
    g = pd.Series(e, index=idx).groupby(idx.to_period("M")).sum()
    m = len(g)
    se = math.sqrt((g ** 2).sum()) / len(x) * math.sqrt(m / max(m - 1, 1))
    return float(x.mean() / se) if se > 0 else np.nan


def tiid(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def naive(ix):
    ix = pd.DatetimeIndex(ix)
    return ix.tz_convert("UTC").tz_localize(None) if ix.tz is not None else ix


def summ(x, per_year=None):
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return {"n": 0}
    x.index = naive(x.index)
    out = {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": month_t(x), "win": float((x > 0).mean())}
    if per_year:
        sd = x.std()
        out["sharpe"] = float(x.mean() / sd * math.sqrt(per_year)) if sd > 0 else np.nan
        out["ann_return"] = float(x.mean() * per_year)
        eq = x.cumsum()
        out["max_dd"] = float((eq - eq.cummax()).min())
    out["by_year_bp"] = {int(k): round(float(v * 1e4), 2) for k, v in x.groupby(x.index.year).mean().items()}
    return out


def periods(x, per_year=None):
    x = pd.Series(x, dtype=float).dropna()
    x.index = naive(x.index)
    return {"all": summ(x, per_year), "2007-2018": summ(x[x.index <= FIT_END], per_year),
            "2019-2025": summ(x[x.index > FIT_END], per_year)}


# ------------------------------------------------------------------------------------------- HAR
def daily_rv(bk):
    m = bk.m
    mid = 0.5 * (m["bid_c"] + m["ask_c"])
    mid.index = naive(mid.index)
    p5 = mid.resample("5min").last().dropna()
    r = np.log(p5).diff()
    same_day = p5.index.normalize() == pd.Series(p5.index, index=p5.index).shift(1).dt.normalize()
    r = r[same_day.values]
    rv = (r ** 2).groupby(r.index.normalize()).sum()
    n = r.groupby(r.index.normalize()).size()
    rv = rv[(n >= 60) & (rv > 0)]
    return rv[rv.index <= DEV_END]


def har(rv):
    """sigma_hat indexed by the day it forecasts (made at the end of the previous RV day); coefficients fitted on
    2007-2018 and fixed."""
    L = np.log(rv)
    f = pd.DataFrame({"y": L.shift(-1), "d": L, "w": np.log(rv.rolling(5).mean()), "m": np.log(rv.rolling(22).mean())})
    f["target_day"] = pd.Series(rv.index, index=rv.index).shift(-1)
    fit = f.dropna()
    fit = fit[fit.index <= FIT_END]
    X = np.c_[np.ones(len(fit)), fit[["d", "w", "m"]].values]
    beta = np.linalg.lstsq(X, fit.y.values, rcond=None)[0]
    g = f.dropna(subset=["d", "w", "m"]).copy()
    pred = np.c_[np.ones(len(g)), g[["d", "w", "m"]].values] @ beta
    nxt = g.target_day.where(g.target_day.notna(), g.index + pd.offsets.BDay(1))
    s = pd.Series(np.sqrt(np.exp(pred)), index=pd.DatetimeIndex(nxt))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    resid = fit.y.values - X @ beta
    return s, {"beta": beta.round(4).tolist(), "r2_fit": float(1 - resid.var() / fit.y.var())}


def sig_on(s, d):
    d = pd.Timestamp(d).normalize()
    if d.tzinfo is not None:
        d = d.tz_convert("UTC").tz_localize(None)
    v = s.asof(d) if len(s) else np.nan
    return float(v) if v is not None and np.isfinite(v) else np.nan


def rel_spread(bk, i):
    return (bk.ao[i] - bk.bo[i]) / (0.5 * (bk.ao[i] + bk.bo[i]))


# ------------------------------------------------------------------------------------------- part A
def b0_nights(bk, sym):
    import flow_edges_lab as F
    import prop_checks as P
    G, _, _ = F.calendar(bk)
    rows = {}
    for d in G:
        if pd.Timestamp(d) > DEV_END:
            continue
        i, j = F.window(bk, d)
        if i < 0 or not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all():
            continue
        path, X = P.leg(bk, i, j)
        n, flat, rate = bk.swaps(i, j, 1, 0.5 * (bk.bo[i] + bk.ao[i]))
        net = X - 2 * bk.comm + (rate if np.isfinite(rate) else flat)
        rows[pd.Timestamp(d)] = {"net": net, "mae": min(float(path.min()), net, 0.0),
                                 "entry_day": naive([bk.idx[i]])[0].normalize()}
    return pd.DataFrame.from_dict(rows, orient="index")


def vs2_days(bk):
    """For each weekday: VR from the previous 60 weekdays, the previous 24-hour return, and the long / short nets of
    the 12:00 -> 12:00 UTC trade."""
    import untested_intraday_lab as U
    m = bk.m
    mid = 0.5 * (m["bid_o"] + m["ask_o"])
    mid.index = naive(mid.index)
    h = mid.resample("1h").first().dropna()
    hr = np.log(h).diff()
    hr = hr[(pd.Series(h.index, index=h.index).diff() == pd.Timedelta(hours=1)).values]
    days = [d for d in pd.bdate_range("2007-01-01", DEV_END)]
    d12 = {}
    for d in days:
        t = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=12)
        i = bk.at(t, 5)
        if i >= 0:
            d12[d] = (i, 0.5 * (bk.bo[i] + bk.ao[i]))
    keys = sorted(d12)
    rows = {}
    for k in range(61, len(keys) - 1):
        d, nxt = keys[k], keys[k + 1]
        if (nxt - d).days > 4:
            continue
        win = keys[k - 60:k + 1]
        lp = np.log([d12[x][1] for x in win])
        rd = np.diff(lp)
        hh = hr[(hr.index > win[0] + pd.Timedelta(hours=12)) & (hr.index <= d + pd.Timedelta(hours=12))]
        if len(hh) < 500 or rd.std() == 0:
            continue
        vr = rd.var() / (24 * hh.var())
        prev = lp[-1] - lp[-2]
        i, j = d12[d][0], d12[nxt][0]
        L = bk.trade_ij(i, j, 1)
        S = bk.trade_ij(i, j, -1)
        if L is None or S is None or prev == 0:
            continue
        rows[d] = {"vr": vr, "prev": prev, "long": U.net_of(L), "short": U.net_of(S)}
    return pd.DataFrame.from_dict(rows, orient="index")


def eod_trades(bk, tz, close_hm, signal_hm):
    import flows_lab as FL
    import untested_intraday_lab as U
    out = {}
    prev_close = None
    for d in pd.bdate_range(str(bk.idx[0].date()), str(min(bk.idx[-1], pd.Timestamp(DEV_END, tz="UTC")).date())):
        tc, ts = FL.local_ts(d, tz, *close_hm), FL.local_ts(d, tz, *signal_hm)
        ic, isg = bk.at(tc, 3), bk.at(ts, 3)
        if prev_close is not None and isg >= 0 and ic > isg:
            r = 0.5 * (bk.bo[isg] + bk.ao[isg]) / (0.5 * (bk.bo[prev_close] + bk.ao[prev_close])) - 1
            if r != 0:
                tr = bk.trade_ij(isg, ic, 1 if r > 0 else -1, ts, tc)
                if tr is not None:
                    out[pd.Timestamp(d)] = (U.net_of(tr), float(tr[1]), rel_spread(bk, isg))
        prev_close = ic if ic >= 0 else None
    return pd.DataFrame.from_dict(out, orient="index", columns=["net", "gross", "spread"])


def timed_trades(bk, days, fn, side, tol=10):
    import untested_intraday_lab as U
    out = {}
    for d in days:
        t0, t1 = fn(d)
        i, j = bk.at(t0, tol), bk.at(t1, tol)
        if i < 0 or j <= i:
            continue
        r = bk.trade_ij(i, j, side, t0, t1)
        if r is not None:
            out[pd.Timestamp(d)] = (U.net_of(r), float(r[1]), rel_spread(bk, i))
    return pd.DataFrame.from_dict(out, orient="index", columns=["net", "gross", "spread"])


def part_a(a):
    import edges_lab2 as E
    import flow_edges_lab as F
    import flows_lab as FL
    import untested_intraday_lab as U
    syms = list(FX7) + ["EURJPY", "XAUUSD", "AUDJPY", "NZDJPY"]
    SIG, NIGHTS, VS2, EFF, VS5 = {}, {}, {}, {}, {}
    for sym in syms:
        bk = E.book(sym)
        if bk is None:
            LOG.setdefault("missing", []).append(sym)
            continue
        rv = daily_rv(bk)
        s, info = har(rv)
        SIG[sym] = s
        LOG.setdefault("har", {})[sym] = {**info, "days": int(len(rv)), "first": str(rv.index.min().date())}
        # out-of-sample check of the forecast on 2019-2025
        chk = pd.concat([np.log(s), np.log(np.sqrt(rv))], axis=1, keys=["f", "a"]).dropna()
        chk = chk[chk.index > FIT_END]
        LOG["har"][sym]["oos_corr_2019_2025"] = float(chk.f.corr(chk.a)) if len(chk) > 50 else None
        if sym in ("USDJPY", "EURJPY"):
            NIGHTS[sym] = b0_nights(bk, sym)
            days = F.calendar(bk)[0]
            EFF.setdefault("PF", {})[sym] = timed_trades(
                bk, [d for d in days if pd.Timestamp(d) <= DEV_END],
                lambda d: (U.jst(pd.Timestamp(d), 9, 55), U.jst(pd.Timestamp(d), 10, 20)), -1, tol=5)
        if sym in FX7 or sym == "XAUUSD":
            VS2[sym] = vs2_days(bk)
        if sym in FX7:
            EFF.setdefault("E3", {})[sym] = eod_trades(bk, "America/New_York", (15, 0), (14, 30))
        if sym == "XAUUSD":
            EFF.setdefault("E2", {})[sym] = eod_trades(bk, "America/New_York", (13, 30), (13, 0))
        if sym in ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"):
            usd_long = 1 if sym.startswith("USD") else -1
            days = [d for d in pd.bdate_range("2007-01-01", DEV_END)]
            k1 = timed_trades(bk, days, lambda d: (FL.local_ts(d, "Europe/Berlin", 13, 15),
                                                   FL.local_ts(d, "Europe/Berlin", 14, 15)), usd_long)
            k2 = timed_trades(bk, days, lambda d: (FL.local_ts(d, "Europe/Berlin", 14, 15),
                                                   FL.local_ts(d, "Europe/Berlin", 15, 15)), -usd_long)
            ix = k1.index.intersection(k2.index)
            EFF.setdefault("K3", {})[sym] = pd.DataFrame({"net": k1.net[ix] + k2.net[ix], "gross": k1.gross[ix] + k2.gross[ix],
                                                          "spread": k1.spread[ix]})
        if sym in ("AUDJPY", "NZDJPY", "AUDUSD", "NZDUSD"):
            VS5[sym] = vs5_trades(bk, s)
        E.BOOKS.clear()
        del bk
        print("done", sym, flush=True)
    res = {"har": LOG.get("har"), "missing": LOG.get("missing", [])}
    res["VS1"] = vs1(NIGHTS, SIG, a)
    res["VS2"] = vs2_eval(VS2)
    res["VS4"] = vs4_eval(EFF, SIG)
    res["VS5"] = vs5_eval(VS5)
    json.dump(res, open(os.path.join(a.out, "part_a.json"), "w"), indent=1, default=str)
    return res


def vs1(NIGHTS, SIG, a):
    import final_sims as FS
    import prop_firms_ev as PF
    if len(NIGHTS) < 2:
        return {"status": "data missing"}
    A, B = NIGHTS["USDJPY"], NIGHTS["EURJPY"]
    ix = A.index.intersection(B.index)
    bas = pd.DataFrame({"net": (A.net[ix] + B.net[ix]) / 2, "mae": (A.mae[ix] + B.mae[ix]) / 2})
    bas["mae"] = np.minimum(bas.mae, np.minimum(bas.net, 0.0))
    bas["sig"] = [np.nanmean([sig_on(SIG["USDJPY"], A.entry_day[d]), sig_on(SIG["EURJPY"], B.entry_day[d])]) for d in ix]
    bas = bas.dropna()
    bas.index = naive(bas.index).normalize()
    bas.to_csv(os.path.join(a.out, "vs1_nights.csv"))
    fit = bas[bas.index <= FIT_END]
    k = 4.0 / float(np.mean(1.0 / fit.sig))
    for _ in range(30):                                                  # the clip changes the mean: iterate k
        m = float(np.mean(np.clip(k / fit.sig, 1, 10)))
        k *= 4.0 / m
    bas["L"] = np.clip(k / bas.sig, 1, 10)
    chk = bas[bas.index > FIT_END]
    out = {"nights": int(len(bas)), "k": k, "mean_L_fit": float(bas.L[bas.index <= FIT_END].mean()),
           "mean_L_check": float(chk.L.mean()),
           "corr_sigma_abs_net_check": float(chk.sig.corr(chk.net.abs())),
           "corr_sigma_mae_check": float(chk.sig.corr(-chk.mae)),
           "check_mean_net_bp": float(chk.net.mean() * 1e4)}
    rows = []
    for scen in ("dev", "2026"):
        base = chk[["net", "mae"]] if scen == "dev" else FS.shifted(chk[["net", "mae"]], FS.TARGET_2026)
        for rule in ("VS1", "fixed_3", "fixed_4", "fixed_5", "fixed_matched"):
            # fixed_matched (information only): a fixed leverage equal to VS1's average on the check nights
            L = chk.L.values if rule == "VS1" else (float(chk.L.mean()) if rule == "fixed_matched"
                                                    else float(rule.split("_")[1]))
            g = base.copy()
            g["net"] = g.net * L
            g["mae"] = g.mae * L
            X, M = FS.daily(g)
            for firm, rules in FS.FIRMS.items():
                r = PF.simulate(X, rules, n=8000, seed=1, M=M)
                rows.append({"scenario": scen, "rule": rule, "firm": firm, **r})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "vs1_prop.csv"), index=False)
    avg = R.groupby(["scenario", "rule"])[["paid", "pass", "fail_daily", "fail_max", "mean_net"]].mean()
    out["five_firm_average"] = {f"{s}|{r}": v for (s, r), v in avg.round(4).to_dict("index").items()}
    ok = True
    for s in ("dev", "2026"):
        ok &= avg.loc[(s, "VS1"), "paid"] >= avg.loc[(s, "fixed_4"), "paid"] + 0.03
        ok &= avg.loc[(s, "VS1"), "fail_daily"] <= avg.loc[(s, "fixed_4"), "fail_daily"]
    out["use_it"] = bool(ok)
    return out


def vs2_eval(VS2):
    rule, fol, fad = {}, {}, {}
    for sym, f in VS2.items():
        if not len(f):
            continue
        sgn = np.sign(f.prev)
        trend = f.vr > 1
        pos = np.where(trend, sgn, -sgn)
        rule[sym] = pd.Series(np.where(pos > 0, f.long, f.short), index=f.index)
        fol[sym] = pd.Series(np.where(sgn > 0, f.long, f.short), index=f.index)
        fad[sym] = pd.Series(np.where(sgn > 0, f.short, f.long), index=f.index)
        LOG.setdefault("vs2_vr", {})[sym] = {"median_vr": float(f.vr.median()), "share_trend": float(trend.mean())}
    if not rule:
        return {"status": "data missing"}
    pool = pd.concat(rule, axis=1).mean(axis=1)
    out = {"pooled": periods(pool), "always_follow": periods(pd.concat(fol, axis=1).mean(axis=1)),
           "always_fade": periods(pd.concat(fad, axis=1).mean(axis=1)), "vr": LOG.get("vs2_vr"),
           "per_instrument": {k: periods(v)["2019-2025"] for k, v in rule.items()}}
    c = out["pooled"]
    out["counts_own_rule"] = bool(c["2019-2025"].get("mean_bp", -1) > 0 and (c["2019-2025"].get("t") or 0) >= 1.645
                                  and c["2007-2018"].get("mean_bp", -1) > 0)
    return out


def vs4_eval(EFF, SIG):
    out = {}
    for eff, per in EFF.items():
        gated, ungated = {}, {}
        for sym, f in per.items():
            if not len(f):
                continue
            f = f.copy()
            f.index = naive(f.index).normalize()
            f["sig"] = [sig_on(SIG[sym], d) for d in f.index]
            f["ratio"] = f.sig / f.spread
            f = f.dropna()
            cut = f.ratio[f.index <= FIT_END].quantile(0.70) if (f.index <= FIT_END).sum() > 30 else np.nan
            if not np.isfinite(cut):
                continue
            ungated[sym] = f.net
            gated[sym] = f.net[f.ratio > cut]
            LOG.setdefault("vs4_cut", {})[f"{eff}:{sym}"] = {"cut": float(cut), "fit_trades": int((f.index <= FIT_END).sum())}
        if not gated:
            out[eff] = {"status": "data missing"}
            continue
        G = pd.concat(gated, axis=1).mean(axis=1)
        Ug = pd.concat(ungated, axis=1).mean(axis=1)
        out[eff] = {"gated": periods(G), "ungated": periods(Ug)}
        c = out[eff]["gated"]["2019-2025"]
        out[eff]["counts_own_rule"] = bool(c.get("mean_bp", -1) > 0 and (c.get("t") or 0) >= 1.645)
    return out


def vs5_trades(bk, s):
    import flow_volume_lab as V
    lj = np.log(s)
    jump = lj.diff()
    z = (jump - jump.rolling(250, min_periods=150).mean().shift(1)) / jump.rolling(250, min_periods=150).std().shift(1)
    out = {}
    sym_trade = None
    for d, zz in z[z >= 2].items():
        if d > DEV_END or d < pd.Timestamp("2007-01-01"):
            continue
        dd = d - pd.offsets.BDay(1)                                      # the forecast for d is known at the end of d-1
        t_in, t_out = V.ny(V.bday_after(dd, 1)), V.ny(V.bday_after(dd, 2))
        r = bk.trade(t_in, t_out, -1, tol_min=30)
        if r is not None:
            import untested_intraday_lab as U
            out[pd.Timestamp(d)] = U.net_of(r)
    return pd.Series(out, dtype=float)


def vs5_eval(VS5):
    use = {k: v for k, v in VS5.items() if k in ("AUDJPY", "NZDJPY") and len(v)}
    if len(use) < 2:
        use.update({k: v for k, v in VS5.items() if k in ("AUDUSD", "NZDUSD") and len(v)})
    if not use:
        return {"status": "data missing"}
    pool = pd.concat(use, axis=1).mean(axis=1)
    out = {"pairs": list(use), "pooled": periods(pool)}
    c = out["pooled"]
    out["counts_own_rule"] = bool(c["all"].get("mean_bp", -1) > 0 and (c["all"].get("t") or 0) >= 1.645
                                  and c["2007-2018"].get("mean_bp", -1) > 0 and c["2019-2025"].get("mean_bp", -1) > 0)
    return out


# ------------------------------------------------------------------------------------------- part B
T1_SYMS = list(FX7) + ["EURJPY", "XAUUSD", "XAGUSD", "USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY", "BRENTCMDUSD",
                       "USTBONDTRUSD", "BUNDTREUR"]
CLASS = {**{s: "FX" for s in list(FX7) + ["EURJPY"]}, "XAUUSD": "metals", "XAGUSD": "metals",
         "USA500IDXUSD": "indices", "USATECHIDXUSD": "indices", "JPNIDXJPY": "indices", "BRENTCMDUSD": "energy",
         "USTBONDTRUSD": "bonds", "BUNDTREUR": "bonds"}


def weekly(bk, sym):
    """Friday 20:00 UTC (the last bar within the previous 2 hours) mid, relative spread, and for FX the point-in-time
    rate differential (base - quote, a year)."""
    rows = {}
    for d in pd.date_range("2007-01-05", DEV_END, freq="W-FRI"):
        t = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=20)
        i = int(bk.idx.searchsorted(t, side="right")) - 1
        if i < 0 or (t - bk.idx[i]) > pd.Timedelta(hours=2):
            continue
        mid = 0.5 * (bk.bo[i] + bk.ao[i])
        rd = np.nan
        if bk.rd is not None:
            v = bk.rd.asof(bk.idx[i].floor("D"))
            rd = float(v) if v is not None and np.isfinite(v) else np.nan
        rows[pd.Timestamp(d)] = {"mid": mid, "spread": (bk.ao[i] - bk.bo[i]) / mid, "rd": rd}
    return pd.DataFrame.from_dict(rows, orient="index")


def book_weekly(signal_pos, W, fx):
    """Weekly net of holding pos (set at week t, held to t+1): pos x return - costs on the change - financing."""
    r = np.log(W.mid).diff().shift(-1)                                   # return from t to t+1
    pos = signal_pos.reindex(W.index).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (0.5 * W.spread * 1.5 + 0.35e-4)
    if fx:
        fin = (pos * W.rd.fillna(0.0) - 0.0075 * pos.abs()) / 52
    else:
        fin = -0.03 * pos.abs() / 52
    net = pos * (np.exp(r) - 1) - cost + fin
    return net.reindex(signal_pos.index).dropna()


def t1(a):
    import edges_lab2 as E
    nets, cov = {}, {}
    for sym in T1_SYMS:
        bk = E.book(sym)
        if bk is None:
            cov[sym] = "missing"
            continue
        W = weekly(bk, sym)
        E.BOOKS.clear()
        del bk
        if len(W) < 120:
            cov[sym] = f"only {len(W)} weeks"
            continue
        first_full = pd.Timestamp(year=W.index.min().year + (0 if W.index.min().month == 1 else 1), month=1, day=1)
        W = W[W.index >= first_full]
        lp = np.log(W.mid)
        r52 = lp - lp.shift(52)
        ew = lp.ewm(span=10).mean() - lp.ewm(span=40).mean()
        sig = 0.5 * np.sign(r52) + 0.5 * np.sign(ew)
        vol = np.log(W.mid).diff().rolling(26).std() * math.sqrt(52)
        pos = (sig * 0.10 / vol).clip(-3, 3)
        pos[r52.isna() | vol.isna()] = np.nan
        nets[sym] = book_weekly(pos.dropna(), W, CLASS[sym] == "FX")
        cov[sym] = [str(W.index.min().date()), str(W.index.max().date()), int(len(W))]
        print("T1", sym, cov[sym], flush=True)
    LOG["t1_coverage"] = cov
    if not nets:
        return {"status": "data missing"}
    N = pd.concat(nets, axis=1)
    port = N.mean(axis=1, skipna=True)
    out = {"portfolio": periods(port, 52), "coverage": cov,
           "by_class": {c: periods(N[[s for s in N.columns if CLASS[s] == c]].mean(axis=1), 52)["all"]
                        for c in sorted(set(CLASS[s] for s in N.columns))},
           "per_instrument": {s: periods(N[s], 52)["all"] for s in N.columns}}
    p = out["portfolio"]
    out["counts_own_rule"] = bool(p["all"].get("mean_bp", -1) > 0 and (p["all"].get("t") or 0) >= 1.645
                                  and p["2019-2025"].get("mean_bp", -1) > 0)
    port.to_csv(os.path.join(a.out, "t1_weekly.csv"))
    return out


def bs1(a):
    import edges_lab2 as E
    import flow_volume_lab as V
    import push_lab as PL
    Y = V.yields(a.store)
    F2 = PL.foreign_2y(a.store)
    US = Y.get("US")
    other = {"EU": Y.get("EU"), "JP": Y.get("JP"), "UK": F2.get("UK"), "CA": F2.get("CA"), "CH": F2.get("CH"),
             "AU": F2.get("AU")}
    pairs = {"EU": ("EURUSD", 1), "JP": ("USDJPY", -1), "UK": ("GBPUSD", 1), "CA": ("USDCAD", -1),
             "CH": ("USDCHF", -1), "AU": ("AUDUSD", 1)}
    if US is None:
        return {"status": "US yields missing"}
    res = {}
    for ma in (60, 20, 120):
        nets = {}
        for c, (sym, sg) in pairs.items():
            if other.get(c) is None:
                continue
            D = (other[c] - US).dropna()
            above = np.sign(D - D.rolling(ma).mean().shift(1))           # vs the previous ma trading days
            bk = E.book(sym)
            if bk is None:
                continue
            W = weekly(bk, sym)
            E.BOOKS.clear()
            del bk
            s = pd.Series([above.asof(d) for d in W.index], index=W.index, dtype=float)
            pos = (s * sg).where(s.notna())
            nets[c] = book_weekly(pos.dropna(), W, True)
        if not nets:
            res[f"ma{ma}"] = {"status": "data missing"}
            continue
        port = pd.concat(nets, axis=1).mean(axis=1)
        res[f"ma{ma}"] = {"currencies": list(nets), "portfolio": periods(port, 52)}
    p = res.get("ma60", {}).get("portfolio")
    if p:
        res["counts_own_rule"] = bool(p["all"].get("mean_bp", -1) > 0 and (p["all"].get("t") or 0) >= 1.645
                                      and p["2007-2018"].get("mean_bp", -1) > 0 and p["2019-2025"].get("mean_bp", -1) > 0)
    return res


def c5e(a):
    import requests
    CODES = {"EUR": "099741", "JPY": "097741", "GBP": "096742", "CHF": "092741", "CAD": "090741", "AUD": "232741",
             "NZD": "112741"}
    d = os.path.join(a.out, "cot_legacy")
    os.makedirs(d, exist_ok=True)
    try:
        r = requests.get("https://www.cftc.gov/files/dea/history/deacot1986_2016.zip",
                         headers={"User-Agent": "fxlab-research/1.0 (research data collection)"}, timeout=300)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        nm = z.namelist()[0]
        T = pd.read_csv(io.BytesIO(z.read(nm)), low_memory=False)
    except Exception as ex:                                             # noqa: BLE001
        return {"status": f"data missing: {ex!r}"[:200]}
    cols = {c.lower(): c for c in T.columns}
    date_c = next(c for k, c in cols.items() if "yyyy-mm-dd" in k)
    code_c = next(c for k, c in cols.items() if "cftc contract market code" in k)
    oi_c = next(c for k, c in cols.items() if k.startswith("open interest (all)"))
    nl_c = next(c for k, c in cols.items() if k.startswith("noncommercial positions-long (all)"))
    ns_c = next(c for k, c in cols.items() if k.startswith("noncommercial positions-short (all)"))
    T["date"] = pd.to_datetime(T[date_c])
    T["code"] = T[code_c].astype(str).str.strip().str.zfill(6)
    out, rets = {}, {}
    for ccy, code in CODES.items():
        g = T[T.code == code].sort_values("date").drop_duplicates("date")
        if len(g) < 60:
            continue
        net = (g[nl_c] - g[ns_c]) / g[oi_c]
        net.index = g.date
        x = net - net.shift(1).rolling(52, min_periods=40).mean()
        fut = glob.glob(os.path.join(a.store, "**", "futures", f"{ccy}.csv"), recursive=True)
        if not fut:
            continue
        px = pd.read_csv(fut[0])
        px.index = pd.to_datetime(px[px.columns[0]], errors="coerce").dt.tz_localize(None).dt.normalize()
        px = pd.to_numeric(px["Close"], errors="coerce").dropna()
        px = px[px.index.notna()]
        months = pd.period_range("2000-10", "2003-12", freq="M")
        rows, prev = {}, 0
        for mth in months:
            start = px[px.index >= mth.start_time]
            end = px[px.index >= (mth + 1).start_time]
            if not len(start) or not len(end):
                continue
            t0, t1 = start.index[0], end.index[0]
            pub = x[x.index + pd.Timedelta(days=3) < t0].dropna()
            if not len(pub):
                continue
            pos = int(np.sign(pub.iloc[-1]))
            ret = pos * (px[t1] / px[t0] - 1) - (1e-4 if ccy != "NZD" else 1.5e-4) * abs(pos - prev)
            prev = pos
            rows[t0] = ret
        s = pd.Series(rows, dtype=float)
        rets[ccy] = s
        out[ccy] = {"n_months": int(len(s)), "mean_bp": float(s.mean() * 1e4) if len(s) else None, "t_iid": tiid(s),
                    "win": float((s > 0).mean()) if len(s) else None}
    res = {"per_currency": out}
    if "EUR" in rets and len(rets["EUR"]) > 5:
        e = out["EUR"]
        res["counts_own_rule"] = bool(e["mean_bp"] > 0 and (e["t_iid"] or 0) >= 1.645)
    if rets:
        pool = pd.concat(rets, axis=1).mean(axis=1)
        res["pooled_7"] = {"n_months": int(len(pool)), "mean_bp": float(pool.mean() * 1e4), "t_iid": tiid(pool)}
    return res


def part_b(a):
    import push_lab as PL
    res = {}
    for nm, fn in (("T1", t1), ("BS1", bs1), ("C5E", c5e)):
        try:
            res[nm] = fn(a)
        except Exception as ex:                                         # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[nm] = {"status": f"error {ex!r}"[:300]}
        print(nm, json.dumps(res[nm], default=str)[:1500], flush=True)
    try:
        pa = argparse.Namespace(store=a.store, out=a.out)
        res["P1_with_AU"] = PL.p1(pa)
        res["P1_with_AU_log"] = PL.LOG.get("foreign_2y"), PL.LOG.get("foreign_2y_errors"), PL.LOG.get("rba_ids")
    except Exception as ex:                                             # noqa: BLE001
        res["P1_with_AU"] = {"status": f"error {ex!r}"[:300]}
    json.dump(res, open(os.path.join(a.out, "part_b.json"), "w"), indent=1, default=str)
    return res


def combine(a):
    from scipy import stats as st
    rows = []
    A = json.load(open(os.path.join(a.out, "part_a.json"))) if os.path.exists(os.path.join(a.out, "part_a.json")) else {}
    B = json.load(open(os.path.join(a.out, "part_b.json"))) if os.path.exists(os.path.join(a.out, "part_b.json")) else {}

    def add(name, x, tkey="t"):
        if not isinstance(x, dict) or not x.get("n"):
            rows.append({"test": name, "status": "missing"})
            return
        rows.append({"test": name, "n": x.get("n"), "mean_bp": x.get("mean_bp"), "t": x.get(tkey)})
    add("VS2 pooled 2019-2025", A.get("VS2", {}).get("pooled", {}).get("2019-2025"))
    for eff in ("E2", "E3", "K3", "PF"):
        add(f"VS4 {eff} gated 2019-2025", A.get("VS4", {}).get(eff, {}).get("gated", {}).get("2019-2025"))
    add("VS5 pooled 2007-2025", A.get("VS5", {}).get("pooled", {}).get("all"))
    add("T1 portfolio 2007-2025 (weekly)", B.get("T1", {}).get("portfolio", {}).get("all"))
    add("BS1 ma60 2007-2025 (weekly)", B.get("BS1", {}).get("ma60", {}).get("portfolio", {}).get("all"))
    e = B.get("C5E", {}).get("per_currency", {}).get("EUR")
    add("C5E euro 2000-2003 (monthly)", {"n": e.get("n_months"), "mean_bp": e.get("mean_bp"), "t_iid": e.get("t_iid")}
        if e else None, "t_iid")
    f3 = os.path.join(a.out, "vs3.json")
    vs3 = json.load(open(f3)) if os.path.exists(f3) else None
    if vs3:
        add("VS3 gold VRP check", vs3.get("check"), "t_iid")
    R = pd.DataFrame(rows)
    m = R.get("t", pd.Series(dtype=float)).notna()
    p = pd.Series(np.nan, index=R.index)
    p[m] = st.norm.sf(R.t[m].astype(float))
    k = int(m.sum())
    if k:
        idx = p[m].sort_values().index
        q = p[idx].values * k / np.arange(1, k + 1)
        R.loc[idx, "bh_q"] = np.minimum.accumulate(q[::-1])[::-1]
    R["counts"] = m & (R.get("mean_bp", 0).astype(float) > 0) & (R.t.astype(float) >= 1.645) & (R.get("bh_q", 1.0) <= 0.10)
    R.to_csv(os.path.join(a.out, "q_combined.csv"), index=False)
    print(R.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-a", action="store_true")
    ap.add_argument("--part-b", action="store_true")
    ap.add_argument("--vs3", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--store", default="flow_store")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.part_a:
        r = part_a(a)
        print(json.dumps(r, indent=1, default=str)[:12000], flush=True)
    if a.part_b:
        r = part_b(a)
    if a.vs3:
        r = vs3(a)
        json.dump(r, open(os.path.join(a.out, "vs3.json"), "w"), indent=1, default=str)
        print("VS3", json.dumps(r, indent=1, default=str)[:4000], flush=True)
    if a.combine:
        combine(a)
    json.dump(LOG, open(os.path.join(a.out, "log_q.json"), "w"), indent=1, default=str)


# ------------------------------------------------------------------------------------------- VS3
def vs3(a):
    import edges_lab2 as E
    import untested_intraday_lab as U
    f = glob.glob(os.path.join(a.store, "**", "implied_vol", "cboe_GVZ.csv"), recursive=True)
    if not f:
        return {"status": "GVZ missing"}
    g = pd.read_csv(f[0])
    g.columns = [c.strip().upper() for c in g.columns]
    vc = "CLOSE" if "CLOSE" in g.columns else g.columns[-1]
    gvz = pd.Series(pd.to_numeric(g[vc], errors="coerce").values, index=pd.to_datetime(g[g.columns[0]], errors="coerce")).dropna()
    LOG["gvz"] = [str(gvz.index.min().date()), str(gvz.index.max().date()), int(len(gvz)), vc]
    bk = E.book("XAUUSD")
    if bk is None:
        return {"status": "gold M1 missing"}
    d15 = {}
    for d in pd.bdate_range("2008-01-01", DEV_END):
        t = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=15)
        i = bk.at(t, 10)
        if i >= 0:
            d15[pd.Timestamp(d)] = (i, 0.5 * (bk.bo[i] + bk.ao[i]))
    px = pd.Series({k: v[1] for k, v in d15.items()})
    rv21 = np.log(px).diff().rolling(21).std() * math.sqrt(252) * 100
    firsts = px.groupby(px.index.to_period("M")).apply(lambda s: s.index[0])
    rows = {}
    vrp_hist = {}
    keys = list(firsts.values)
    for k in range(len(keys) - 1):
        t0, t1 = pd.Timestamp(keys[k]), pd.Timestamp(keys[k + 1])
        prev_day = px.index[px.index < t0]
        gv = gvz[gvz.index < t0]
        if not len(prev_day) or not len(gv) or not np.isfinite(rv21.get(prev_day[-1], np.nan)):
            continue
        vrp = float(gv.iloc[-1] - rv21[prev_day[-1]])
        vrp_hist[t0] = vrp
        past = [v for kk, v in vrp_hist.items() if kk < t0][-36:]
        if len(past) < 24:
            continue
        s = int(np.sign(vrp - np.median(past)))
        if s == 0:
            continue
        i, j = d15[t0][0], d15[t1][0]
        L, S = bk.trade_ij(i, j, 1), bk.trade_ij(i, j, -1)
        if L is None or S is None:
            continue
        rows[t0] = {"signal": s, "follow": U.net_of(L) if s > 0 else U.net_of(S),
                    "fade": U.net_of(S) if s > 0 else U.net_of(L),
                    "gross_follow": float(L[1]) if s > 0 else float(S[1])}
    F = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    if len(F) < 30:
        return {"status": f"too few months ({len(F)})"}
    k = int(0.6 * len(F))
    fit, chk = F.iloc[:k], F.iloc[k:]
    d = "follow" if fit.gross_follow.mean() > 0 else "fade"
    c = chk[d]
    res = {"months": int(len(F)), "fit_months": int(k), "check_from": str(chk.index[0].date()), "direction": d,
           "fit_gross_follow_bp": float(fit.gross_follow.mean() * 1e4),
           "check": {"n": int(len(c)), "mean_bp": float(c.mean() * 1e4), "t_iid": tiid(c), "win": float((c > 0).mean())},
           "note": "signal +1 = implied volatility rich relative to its 36-month median"}
    res["counts_own_rule"] = bool(res["check"]["mean_bp"] > 0 and (res["check"]["t_iid"] or 0) >= 1.645)
    F.to_csv(os.path.join(a.out, "vs3_months.csv"))
    return res


if __name__ == "__main__":
    main()
