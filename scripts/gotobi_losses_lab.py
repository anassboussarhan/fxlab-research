"""What goes with LOSING Gotobi nights. The trade: long USD/JPY and EUR/JPY 23:00 Tokyo the evening before -> 09:53
Tokyo on Gotobi days, no stop (and the 80 bp stop). Development data only (to 2025-12-31); 2026 is not loaded.
Chosen / significant on 2007-2018 ("train"), checked on 2019-2025 ("test").

OUTCOMES per night: lose (net < 0); big loss (net below -1 W, W = the mean 23:00-09:53 range of the previous 20
Gotobi days); deep open loss (worst mark-to-market below -1 W); and the net return itself (Spearman).
FEATURES KNOWN AT ENTRY (23:00 Tokyo = 14:00 UTC on the day before the Gotobi day; US / European closes up to two
days before, Asian closes up to the day before):
  the 24 of scripts/gotobi_accuracy_lab.py; and
  markets: dollar index, gold, oil, copper, Nasdaq, DAX, EUR/USD, AUD/JPY, bitcoin, Hang Seng, KOSPI, ASX 200,
    Shanghai (1-day returns; dollar index, gold, AUD/JPY, S&P 500 and Nikkei also 1-week), Nikkei and S&P 500 vs
    their 50-day means, VIX / VIX3M, US 5- and 10-year yield 1-day changes, 13-week bill 1-week change, 10-year
    minus 13-week level;
  the pair before entry: London-session move (07:00 -> 14:00 UTC), last-hour move, Tokyo-session move (01:00 ->
    06:00 UTC), realised volatility 08:00-14:00 UTC / W, position in the range since 21:00 UTC, distance to the
    nearest whole yen and to the next whole yen above (in W);
  rates: USD (EUR) minus JPY rate differential and its 3-month change;
  calendar: US daylight time, US market holiday the day before, Japanese holiday the day after, each month,
    year-end (Dec 20-31), Golden Week (Apr 27-May 7), Obon (Aug 10-18), the Gotobi's rank in the month, the 5th,
    quarter-start month, an FOMC statement during the night, a Japanese MoF intervention in the previous 30 days.
FEATURES OF THE NIGHT ITSELF (not known at entry; descriptive, to explain losses): the pair's New York leg (entry
  -> 21:00 UTC), Asia leg (21:00 -> 23:00 UTC), fix leg (23:00 UTC -> exit); S&P 500, VIX, US 10-year, gold and
  dollar index changes on the day of the night; the Nikkei's opening gap on the Gotobi day.
TESTS: per feature and outcome, top vs bottom tercile (or 1 vs 0) loss rate, two-proportion z; Spearman with the net;
  Benjamini-Hochberg over the features per series and outcome on train; "confirmed" when q <= 0.10 on train and
  the same sign with p <= 0.10 on test.
MODELS (features known at entry): logistic L2 C = 0.1 for a big loss and for a win, fitted on train (test AUC, the
  effect of skipping the riskiest fifth / the lowest-rated third), and refitted each year 2012-2025 (walk-forward).
  forward-v2 rule, fixed now: the extended win model replaces nothing; it is frozen as forward-v2 only if its
  walk-forward kept-minus-skipped gap (2012-2025) and its 2019-2025 AUC both beat forward-v1's (+9.2 bp, 0.58).
DESCRIPTIVE: the 25 worst nights with their legs and markets; per-year results against the USD/JPY yearly move and
  the rate differential; losing streaks against shuffled order.
Usage: python scripts/gotobi_losses_lab.py --out DIR [--data-cot DIR]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import gotobi_accuracy_lab as A  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

PAIRS = ("USDJPY", "EURJPY")
DAY = pd.Timedelta(days=1)
TRAIN_END = pd.Timestamp("2018-12-31")
NY = ZoneInfo("America/New_York")
INTERVENTIONS = [pd.Timestamp(x) for x in ("2010-09-15", "2011-03-18", "2011-08-04", "2011-10-31", "2022-09-22",
                                           "2022-10-21", "2022-10-24", "2024-04-29", "2024-05-01", "2024-07-11",
                                           "2024-07-12")]
# (name, ticker, Asian close usable on the day before?, kind)
MKT = [("dxy_ret_1d", "DX-Y.NYB", False, "ret1"), ("dxy_ret_1w", "DX-Y.NYB", False, "ret5"),
       ("gold_ret_1d", "GC=F", False, "ret1"), ("gold_ret_1w", "GC=F", False, "ret5"),
       ("oil_ret_1d", "CL=F", False, "ret1"), ("copper_ret_1d", "HG=F", False, "ret1"),
       ("nasdaq_ret_1d", "^IXIC", False, "ret1"), ("dax_ret_1d", "^GDAXI", False, "ret1"),
       ("eurusd_ret_1d", "EURUSD=X", False, "ret1"), ("audjpy_ret_1d", "AUDJPY=X", False, "ret1"),
       ("audjpy_ret_1w", "AUDJPY=X", False, "ret5"), ("btc_ret_1d", "BTC-USD", False, "ret1"),
       ("hsi_ret_1d", "^HSI", True, "ret1"), ("kospi_ret_1d", "^KS11", True, "ret1"),
       ("asx_ret_1d", "^AXJO", True, "ret1"), ("shanghai_ret_1d", "000001.SS", True, "ret1"),
       ("n225_ret_1w", "^N225", True, "ret5"), ("n225_vs_50d", "^N225", True, "vs50"),
       ("spx_ret_1w", "^GSPC", False, "ret5"), ("spx_vs_50d", "^GSPC", False, "vs50"),
       ("us5y_chg_1d", "^FVX", False, "chg1"), ("us10y_chg_1d", "^TNX", False, "chg1"),
       ("bill3m_chg_1w", "^IRX", False, "chg5")]
NIGHT = [("night_spx_ret", "^GSPC", "ret1"), ("night_vix_chg", "^VIX", "chg1"), ("night_us10y_chg", "^TNX", "chg1"),
         ("night_gold_ret", "GC=F", "ret1"), ("night_dxy_ret", "DX-Y.NYB", "ret1")]


def closes(tickers):
    out, errors = {}, {}
    for t in sorted(set(tickers)):
        try:
            import yfinance as yf
            h = yf.download(t, start="2005-01-01", end="2026-01-01", auto_adjust=True, progress=False)
            c, o = h["Close"], h["Open"]
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            o = o.iloc[:, 0] if isinstance(o, pd.DataFrame) else o
            c.index = o.index = pd.DatetimeIndex(c.index).tz_localize(None)
            out[t] = pd.DataFrame({"close": c, "open": o})
            out[t] = out[t][(out[t].index <= "2025-12-31") & (out[t].close > 0)].dropna(subset=["close"])
        except Exception as ex:                                             # noqa: BLE001
            errors[t] = repr(ex)[:200]
    return out, errors


def mkt_value(df, prev_day, same_day_ok, kind):
    c = df.close
    s = c[c.index <= prev_day] if same_day_ok else c[c.index < prev_day]
    if len(s) < 55:
        return np.nan
    if kind == "ret1":
        return float(s.iloc[-1] / s.iloc[-2] - 1.0)
    if kind == "ret5":
        return float(s.iloc[-1] / s.iloc[-6] - 1.0)
    if kind == "chg1":
        return float(s.iloc[-1] - s.iloc[-2])
    if kind == "chg5":
        return float(s.iloc[-1] - s.iloc[-6])
    if kind == "vs50":
        return float(s.iloc[-1] / s.iloc[-50:].mean() - 1.0)
    return float(s.iloc[-1])


def night_value(df, day, kind):
    """Change on US date `day` (the session inside the night); NaN if no close that day."""
    c = df.close
    if day not in c.index:
        return np.nan
    k = c.index.get_loc(day)
    if k < 1:
        return np.nan
    return float(c.iloc[k] / c.iloc[k - 1] - 1.0) if kind == "ret1" else float(c.iloc[k] - c.iloc[k - 1])


def more_features(bk, G, W, base, hol, data, us_hol, fomc):
    lab = A.nominal_map(G, hol)
    rank = {}
    for d in sorted(G):
        rank[d] = sum(1 for x in G if x.year == d.year and x.month == d.month and x <= d)
    rows = {}
    for d in base.index:
        d = pd.Timestamp(d)
        i, j = F.window(bk, d)
        if i < 0:
            continue
        w = float(W.get(d, np.nan))
        m0, t0 = bk.mo[i], bk.idx[i]
        prev = (d - DAY).normalize()

        def mo_at(ts, tol=30):
            k = int(bk.idx.searchsorted(ts, side="right")) - 1
            return bk.mo[k] if k >= 0 and (ts - bk.idx[k]) <= pd.Timedelta(minutes=tol) else np.nan
        r = {}
        for name, tick, same, kind in MKT:
            r[name] = mkt_value(data[tick], prev, same, kind) if tick in data else np.nan
        if "^VIX" in data and "^VIX3M" in data:
            a, b = mkt_value(data["^VIX"], prev, False, "level"), mkt_value(data["^VIX3M"], prev, False, "level")
            r["vix_term"] = a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan
        if "^TNX" in data and "^IRX" in data:
            r["curve_10y_3m"] = mkt_value(data["^TNX"], prev, False, "level") - mkt_value(data["^IRX"], prev, False, "level")
        day0 = pd.Timestamp(t0.date()).tz_localize("UTC")
        for name, h0 in (("london_move_W", 7), ("last_hour_move_W", 13)):
            p = mo_at(day0 + pd.Timedelta(hours=h0))
            r[name] = (m0 - p) / w if np.isfinite(p) else np.nan
        a6, a1 = mo_at(day0 + pd.Timedelta(hours=6)), mo_at(day0 + pd.Timedelta(hours=1))
        r["tokyo_move_W"] = (a6 - a1) / w if np.isfinite(a6) and np.isfinite(a1) else np.nan
        k8 = int(bk.idx.searchsorted(day0 + pd.Timedelta(hours=8)))
        seg = np.diff(np.log(bk.mo[k8:i])) if i - k8 > 200 else np.array([])
        r["rv_6h_vs_W"] = float(seg.std() * np.sqrt(360) / (w / m0)) if len(seg) else np.nan
        k21 = int(bk.idx.searchsorted(day0 - pd.Timedelta(hours=3)))
        if i - k21 > 200:
            hi, lo = np.max(bk.mh[k21:i]), np.min(bk.ml[k21:i])
            r["range_position"] = (m0 - lo) / (hi - lo) if hi > lo else np.nan
        else:
            r["range_position"] = np.nan
        r["dist_whole_yen_W"] = abs(m0 - round(m0)) / w
        r["to_next_yen_above_W"] = (np.floor(m0) + 1 - m0) / w
        if bk.rd is not None:
            v0 = bk.rd.asof(t0.floor("D"))
            v1 = bk.rd.asof((t0 - 91 * DAY).floor("D"))
            r["rate_diff"] = float(v0) if v0 is not None and np.isfinite(v0) else np.nan
            r["rate_diff_chg_3m"] = float(v0 - v1) if (v0 is not None and v1 is not None and np.isfinite(v0)
                                                       and np.isfinite(v1)) else np.nan
        r["us_dst"] = float(t0.tz_convert(NY).dst() != pd.Timedelta(0))
        r["us_holiday_before"] = float(prev in us_hol)
        nxt = d + DAY
        r["jp_holiday_after"] = float(nxt.weekday() < 5 and nxt in hol)
        for mth in range(1, 13):
            r[f"month_{mth:02d}"] = float(d.month == mth)
        r["year_end"] = float(d.month == 12 and d.day >= 20)
        r["golden_week"] = float((d.month == 4 and d.day >= 27) or (d.month == 5 and d.day <= 7))
        r["obon"] = float(d.month == 8 and 10 <= d.day <= 18)
        r["gotobi_rank"] = float(rank.get(d, np.nan))
        r["day_5"] = float(5 in lab.get(d, {"labels": set()})["labels"])
        r["quarter_start_month"] = float(d.month in (1, 4, 7, 10))
        r["fomc_night"] = float(prev in fomc)
        r["intervention_30d"] = float(any(0 <= (d - x).days <= 30 for x in INTERVENTIONS))
        # the night itself (descriptive)
        k21b = int(bk.idx.searchsorted(day0 + pd.Timedelta(hours=21)))
        k23 = int(bk.idx.searchsorted(day0 + pd.Timedelta(hours=23)))
        if i < k21b < k23 < j:
            r["night_ny_leg_W"] = (bk.mo[k21b] - m0) / w
            r["night_asia_leg_W"] = (bk.mo[k23] - bk.mo[k21b]) / w
            r["night_fix_leg_W"] = (bk.mo[j] - bk.mo[k23]) / w
        for name, tick, kind in NIGHT:
            r[name] = night_value(data[tick], prev, kind) if tick in data else np.nan
        if "^N225" in data:
            n = data["^N225"]
            if d in n.index:
                k = n.index.get_loc(d)
                r["night_n225_gap"] = float(n.open.iloc[k] / n.close.iloc[k - 1] - 1.0) if k >= 1 else np.nan
        rows[d] = r
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def outcomes(base, W_ret):
    y = base.net
    return {"lose": (y < 0).astype(float), "big_loss": (y < -W_ret.reindex(y.index)).astype(float).where(W_ret.reindex(y.index).notna()),
            "deep_open_loss": (base.mae < -W_ret.reindex(y.index)).astype(float).where(W_ret.reindex(y.index).notna())}


def prop_z(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 10 or n2 < 10:
        return np.nan, np.nan
    p1, p2 = a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return p1 - p2, (p1 - p2) / se if se > 0 else np.nan


def feature_tests(series, X, net, outs, rows):
    tr = np.asarray(net.index <= TRAIN_END)
    te = ~tr
    X = X.reindex(net.index)
    recs = []
    for c in X.columns:
        x = X[c]
        group = "night (descriptive)" if c.startswith("night_") else "known at entry"
        okt = tr & x.notna().to_numpy()
        if okt.sum() < 40 or x[okt].nunique() < 2:
            continue
        if x[okt].nunique() <= 2:
            v0, v1 = x[okt].min(), x[okt].max()
            hi_m, lo_m = (lambda s, v1=v1: s == v1), (lambda s, v0=v0: s == v0)
            kind = "1 vs 0"
        else:
            c1, c2 = x[okt].quantile([1 / 3, 2 / 3])
            hi_m, lo_m = (lambda s, c2=c2: s > c2), (lambda s, c1=c1: s <= c1)
            kind = "top vs bottom tercile"
        for oname, o in [("net", net)] + list(outs.items()):
            r = {"series": series, "feature": c, "group": group, "outcome": oname, "contrast": kind}
            for per, msk in (("train", tr), ("test", te)):
                ok = msk & x.notna().to_numpy() & o.notna().to_numpy()
                if ok.sum() < 30:
                    continue
                if oname == "net":
                    rho, p = st.spearmanr(x[ok], o[ok])
                    r[f"{per}_stat"], r[f"{per}_p"], r[f"{per}_n"] = float(rho), float(p), int(ok.sum())
                else:
                    xx, oo = x[ok], o[ok]
                    a, b = oo[hi_m(xx).to_numpy()], oo[lo_m(xx).to_numpy()]
                    diff, z = prop_z(a, b)
                    r[f"{per}_stat"], r[f"{per}_p"] = diff, float(2 * st.norm.sf(abs(z))) if np.isfinite(z) else np.nan
                    r[f"{per}_rate_high"], r[f"{per}_rate_low"], r[f"{per}_n"] = float(a.mean()), float(b.mean()), int(ok.sum())
            recs.append(r)
    D = pd.DataFrame(recs)
    if not len(D):
        return
    D["train_q_bh"] = np.nan
    for (g, o), idx in D.groupby(["group", "outcome"]).groups.items():
        D.loc[idx, "train_q_bh"] = A.bh(D.loc[idx, "train_p"].to_numpy(float))
    D["confirmed"] = ((D.train_q_bh <= 0.10) & (np.sign(D.train_stat) == np.sign(D.get("test_stat")))
                      & (D.get("test_p") <= 0.10))
    rows.extend(D.to_dict("records"))


def fit_lr(Xf, yf, Xp, cols=None):
    from sklearn.linear_model import LogisticRegression
    keep = [c for c in cols if c in Xf.columns] if cols is not None else [c for c in Xf.columns if Xf[c].notna().mean() > 0.8]
    med, mu = Xf[keep].median(), Xf[keep].mean()
    sd = Xf[keep].std().replace(0, 1.0)
    Zf, Zp = (Xf[keep].fillna(med) - mu) / sd, (Xp[keep].fillna(med) - mu) / sd
    m = LogisticRegression(C=0.1, max_iter=3000).fit(Zf.to_numpy(), yf.to_numpy().astype(int))
    return m.predict_proba(Zf.to_numpy())[:, 1], pd.Series(m.predict_proba(Zp.to_numpy())[:, 1], index=Xp.index), \
        pd.Series(m.coef_[0], index=keep)


def sstat(x, n_days, mae=None):
    x = x.dropna()
    out = {"n": int(len(x)), "mean_bp": x.mean() * 1e4, "t": float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 else np.nan,
           "win": float((x > 0).mean()), "worst_bp": x.min() * 1e4, "per_day_bp": x.sum() / max(n_days, 1) * 1e4}
    if mae is not None and len(x) > 20:
        m = mae.reindex(x.index)
        q = m.quantile(0.05)
        out["cvar5_bp"] = m[m <= q].mean() * 1e4
    return out


def model_tests(series, Xpre, base, outs, X_v1_cols, rows, coefs, wf_rows):
    from sklearn.metrics import roc_auc_score
    y = base.net
    Xpre = Xpre.reindex(y.index)
    tr, te = np.asarray(y.index <= TRAIN_END), np.asarray(y.index > TRAIN_END)
    specs = [("big-loss model, extended features", outs["big_loss"], None, "risk", 0.2),
             ("win model, extended features", (y > 0).astype(float), None, "win", 1 / 3),
             ("win model, forward-v1 features", (y > 0).astype(float), X_v1_cols, "win", 1 / 3)]
    for name, target, cols, kind, frac in specs:
        ok = target.notna().to_numpy()
        ftr, fte = tr & ok, te & ok
        p_fit, p_te, coef = fit_lr(Xpre[ftr], target[ftr], Xpre[fte], cols)
        coefs[f"{series} | {name}"] = coef.sort_values().round(3).to_dict()
        yt = y[fte]
        if kind == "risk":
            cut = np.quantile(p_fit, 1 - frac)
            skip = p_te > cut
        else:
            cut = np.quantile(p_fit, frac)
            skip = p_te <= cut
        r = {"series": series, "model": name, "features": len(coef), "test_auc": roc_auc_score(target[fte], p_te),
             "skip_share": float(skip.mean())}
        r.update({f"all_{k}": v for k, v in sstat(yt, len(yt), base.mae).items()})
        r.update({f"kept_{k}": v for k, v in sstat(yt[~skip.to_numpy()], len(yt), base.mae).items()})
        r.update({f"skipped_{k}": v for k, v in sstat(yt[skip.to_numpy()], len(yt)).items()})
        rows.append(r)
        # walk-forward, yearly refit from 2012
        kept, skipped, ps, tg = [], [], [], []
        for Y in range(2012, 2026):
            f = np.asarray(y.index < pd.Timestamp(f"{Y}-01-01")) & ok
            pmask = np.asarray(y.index.year == Y) & ok
            if pmask.sum() == 0:
                continue
            pf, pp, _ = fit_lr(Xpre[f], target[f], Xpre[pmask], cols)
            sk = (pp > np.quantile(pf, 1 - frac)) if kind == "risk" else (pp <= np.quantile(pf, frac))
            yy = y[pmask]
            kept.append(yy[~sk.to_numpy()])
            skipped.append(yy[sk.to_numpy()])
            ps.append(pp)
            tg.append(target[pmask])
            wf_rows.append({"series": series, "model": name, "year": Y, "kept_bp": yy[~sk.to_numpy()].mean() * 1e4,
                            "skipped_bp": yy[sk.to_numpy()].mean() * 1e4, "share_skipped": float(sk.mean())})
        K, Sk = pd.concat(kept), pd.concat(skipped)
        rows[-1].update({"wf_kept_bp": K.mean() * 1e4, "wf_kept_t": float(K.mean() / K.std() * np.sqrt(len(K))),
                         "wf_skipped_bp": Sk.mean() * 1e4, "wf_gap_bp": (K.mean() - Sk.mean()) * 1e4,
                         "wf_gap_t": float(st.ttest_ind(K, Sk, equal_var=False).statistic),
                         "wf_auc": roc_auc_score(pd.concat(tg), pd.concat(ps)),
                         "wf_kept_win": float((K > 0).mean()), "wf_kept_worst_bp": K.min() * 1e4})


def describe(series, base, X, W_ret, bk, out):
    y = base.net
    # worst nights
    cols = [c for c in X.columns if c.startswith("night_")] + ["n225_ret_1d", "us10y_chg_1w", "vix", "fomc_night",
                                                               "intervention_30d", "W_vs_median"]
    worst = y.nsmallest(25).to_frame("net_bp")
    worst["net_bp"] *= 1e4
    worst["W_bp"] = (W_ret.reindex(worst.index) * 1e4)
    worst = worst.join(X.reindex(worst.index)[[c for c in cols if c in X.columns]])
    worst.to_csv(os.path.join(out, f"worst_nights_{series}.csv"))
    # legs: winners vs losers
    legs = [c for c in ("night_ny_leg_W", "night_asia_leg_W", "night_fix_leg_W") if c in X]
    L = X.reindex(y.index)[legs].assign(net=y, lose=(y < 0))
    leg_tab = L.groupby("lose")[legs].mean().T
    leg_tab["var_share"] = [L[c].var() / (L[legs].sum(axis=1)).var() for c in legs]
    leg_tab["corr_with_net"] = [L[c].corr(L.net) for c in legs]
    leg_tab.to_csv(os.path.join(out, f"legs_{series}.csv"))
    # years
    yr = y.groupby(y.index.year).agg(["size", "mean", lambda s: (s > 0).mean(), "min"])
    yr.columns = ["n", "mean_bp", "win", "worst_bp"]
    yr[["mean_bp", "worst_bp"]] *= 1e4
    px = pd.Series(bk.mo, index=bk.idx)
    yr["pair_year_return"] = px.groupby(px.index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1.0).reindex(yr.index)
    if "rate_diff" in X:
        yr["rate_diff_mean"] = X.rate_diff.groupby(X.index.year).mean().reindex(yr.index)
    yr.to_csv(os.path.join(out, f"years_{series}.csv"))
    # losing streaks vs shuffled
    lose = (y < 0).to_numpy()

    def longest(a):
        best = cur = 0
        for v in a:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    def runs(a):
        return 1 + int(np.sum(a[1:] != a[:-1]))
    rng = np.random.default_rng(0)
    sh = [rng.permutation(lose) for _ in range(2000)]
    return {"longest_losing_streak": longest(lose), "shuffled_p95_longest": float(np.quantile([longest(s) for s in sh], 0.95)),
            "runs": runs(lose), "shuffled_runs_mean": float(np.mean([runs(s) for s in sh])),
            "runs_p_low": float(np.mean([runs(s) <= runs(lose) for s in sh])),
            "year_corr_mean_vs_pair_return": float(yr["mean_bp"].corr(yr["pair_year_return"])),
            "year_corr_mean_vs_rate_diff": float(yr["mean_bp"].corr(yr["rate_diff_mean"])) if "rate_diff_mean" in yr else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-cot", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    errors, info = {}, {}
    tick = {t for _, t, _, _ in MKT} | {t for _, t, _ in NIGHT} | {"^VIX", "^VIX3M", "^TNX", "^IRX", "^N225", "^GSPC"}
    data, e = closes(tick)
    errors.update({f"yahoo {k}": v for k, v in e.items()})
    info["yahoo"] = {k: [str(v.index.min().date()), str(v.index.max().date()), int(len(v))] for k, v in data.items()}
    mkt_v1 = [(n, data[t].close, s, k) for n, t, s, k in (("vix", "^VIX", False, "level"), ("vix_chg_1w", "^VIX", False, "chg5"),
                                                          ("spx_ret_1d", "^GSPC", False, "ret1"), ("n225_ret_1d", "^N225", True, "ret1"),
                                                          ("us10y_chg_1w", "^TNX", False, "chg5")) if t in data]
    try:
        cot = A.load_cot(a.data_cot)
    except Exception as ex:                                                 # noqa: BLE001
        cot, errors["cot"] = None, repr(ex)[:300]
    FL.jp_holidays([2020])                                                  # installs `holidays` if missing
    import holidays
    us_hol = {pd.Timestamp(d) for d in holidays.financial_holidays("NYSE", years=range(2006, 2027))}
    fomc = set()
    for y, s in FL.FOMC_FALLBACK.items():
        fomc |= {pd.Timestamp(f"{y}-{x}") for x in s.split()}
    frames, Xs, Wr, books = {}, {}, {}, {}
    for p in PAIRS:
        bk = U.book(p)
        if bk is None:
            errors[p] = "no data"
            continue
        bk._hl()
        G, _, hol = F.calendar(bk)
        W = R.window_units(bk, G)
        out = {"none": {}, "stop 80 bp": {}}
        wr = {}
        for d in G:
            w = float(W.get(d, np.nan))
            i, j = F.window(bk, d)
            if not np.isfinite(w) or w <= 0 or i < 0:
                continue
            if not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all():
                continue
            H = A.hours_map(bk, d, i, j)
            out["none"][d] = A.sim(bk, i, j, w, {}, H)
            out["stop 80 bp"][d] = A.sim(bk, i, j, w, {"sl_bp": 80}, H)
            wr[d] = w / bk.mo[i]
        fr = {k: pd.DataFrame.from_dict(v, orient="index").sort_index() for k, v in out.items()}
        for f in fr.values():
            f.index = pd.DatetimeIndex(f.index)
        frames[p], Wr[p], books[p] = fr, pd.Series(wr).sort_index(), bk
        base = fr["none"]
        try:
            X1 = A.features(bk, G, W, base, hol, mkt_v1, cot)
            X2 = more_features(bk, G, W, base, hol, data, us_hol, fomc)
            Xs[p] = X1.join(X2, how="outer")
        except Exception:                                                   # noqa: BLE001
            import traceback
            errors[f"features {p}"] = traceback.format_exc()[-1200:]
        info[p] = {"trades": int(len(base)), "first": str(base.index.min().date()), "last": str(base.index.max().date())}
        print(p, info[p], flush=True)
    if all(p in frames for p in PAIRS):
        iu, ie = frames["USDJPY"]["none"].index, frames["EURJPY"]["none"].index
        frames["basket"] = {k: A.basket(frames["USDJPY"][k], frames["EURJPY"][k], iu, ie) for k in ("none", "stop 80 bp")}
        Wr["basket"] = pd.concat([Wr["USDJPY"], Wr["EURJPY"]], axis=1).mean(axis=1)
        if "USDJPY" in Xs:
            Xs["basket"] = Xs["USDJPY"]
            books["basket"] = books["USDJPY"]
    v1_cols = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "results", "frozen_forward_v1.json")))["features"]
    ftests, mrows, coefs, wf, desc = [], [], {}, [], {}
    for s in ("basket", "USDJPY", "EURJPY"):
        if s not in Xs or s not in frames:
            continue
        base = frames[s]["none"]
        X = Xs[s]
        outs = outcomes(base, Wr[s])
        try:
            feature_tests(s, X, base.net, outs, ftests)
            pre = X[[c for c in X.columns if not c.startswith("night_")]]
            model_tests(s, pre, base, outs, v1_cols, mrows, coefs, wf)
            desc[s] = describe(s, base, X, Wr[s], books[s], a.out)
        except Exception:                                                   # noqa: BLE001
            import traceback
            errors[f"analysis {s}"] = traceback.format_exc()[-1500:]
        X.to_csv(os.path.join(a.out, f"features_{s}.csv"))
        for k, f in frames[s].items():
            f.to_csv(os.path.join(a.out, f"trades_{s}_{k.replace(' ', '_')}.csv"))
    pd.DataFrame(ftests).to_csv(os.path.join(a.out, "feature_tests.csv"), index=False)
    pd.DataFrame(mrows).to_csv(os.path.join(a.out, "models.csv"), index=False)
    pd.DataFrame(wf).to_csv(os.path.join(a.out, "models_walk_forward_years.csv"), index=False)
    v2 = {}
    M = pd.DataFrame(mrows)
    if len(M):
        b = M[(M.series == "basket")].set_index("model")
        if {"win model, extended features", "win model, forward-v1 features"} <= set(b.index):
            ext = b.loc["win model, extended features"]
            v2 = {"extended_wf_gap_bp": ext.wf_gap_bp, "extended_test_auc": ext.test_auc,
                  "qualifies_for_forward_v2": bool(ext.wf_gap_bp > 9.2 and ext.test_auc > 0.58)}
    json.dump({"info": info, "errors": errors, "describe": desc, "coefficients": coefs, "forward_v2_rule": v2},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_rows", 600)
    pd.set_option("display.max_columns", 40)
    T = pd.DataFrame(ftests)
    if len(T):
        print(T[T.confirmed.astype(bool)].round(3).to_string())
    print(M.round(3).T.to_string() if len(M) else "no models")
    print(json.dumps({"errors": errors, "describe": desc, "forward_v2_rule": v2}, indent=1, default=str))


if __name__ == "__main__":
    main()
