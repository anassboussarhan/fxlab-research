"""Round 18, part L (docs/round18_preregistration.md): regimes on a long history, 1975-2025, daily data.

Data: Bank of England daily spot rates (8 currencies, 28 crosses; the Deutsche Mark / 1.95583 for the euro before
1999) and OECD monthly 3-month rates from DBnomics (lagged one month). Styles F1 CARRY, F2 TREND, F5 REV1W,
F7 DIP_W on the 28 crosses; the regime detector (round 16's walk-forward machinery) chooses among them; settings are
chosen on 1999-2025 and verified once on 1981-1998 (never examined before). L3: global FX volatility and the dollar.

Usage: python scripts/r18_long.py --out DIR [--placebos 200]"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
CUR8 = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")
FX = sorted({a + b for i, a in enumerate(CUR8) for b in CUR8[i + 1:]})
BOE = {"USD": "XUDLUSS", "JPY": "XUDLJYS", "DEM": "XUDLDMS", "EUR": "XUDLERS", "CHF": "XUDLSFS", "CAD": "XUDLCDS",
       "AUD": "XUDLADS", "NZD": "XUDLNDS"}
OECD = {"USD": "USA", "JPY": "JPN", "GBP": "GBR", "DEM": "DEU", "EUR": "EA19", "CHF": "CHE", "CAD": "CAN", "AUD": "AUS",
        "NZD": "NZL"}
RULES = ("S.DAILY.DOUBLE_SEVEN", "S.DAILY.CUM_RSI", "S.FAMOUS.CONNORS_RSI2", "S.BOOK.CONNORS_R3", "S.BOOK.CONNORS_PCTB",
         "S.BOOK.CONNORS_CRSI", "S.DAILY.THREE_DOWN")
COST = 1.5e-4
END = "2025-12-31"
STYLES = ["F1", "F2", "F5", "F7"]
STYLE_NAME = {"F1": "CARRY", "F2": "TREND", "F5": "REV1W", "F7": "DIP_W"}


def get(url, tries=4, timeout=120):
    import requests
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 500:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as ex:                                            # noqa: BLE001
            last = repr(ex)[:160]
        time.sleep(4 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def boe_rates():
    url = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1975"
           f"&Dateto=31/Dec/2025&SeriesCodes={','.join(BOE.values())}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
    d = pd.read_csv(io.StringIO(get(url).text))
    d["DATE"] = pd.to_datetime(d["DATE"], format="%d %b %Y")
    d = d.set_index("DATE").sort_index().apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame({c: d[code] for c, code in BOE.items()})
    X["EUR"] = X["EUR"].where(X["EUR"].notna(), X["DEM"] / 1.95583)
    X["GBP"] = 1.0
    X = X[list(CUR8)]
    return X[X.index <= END].dropna(how="all")


def oecd_rates(months):
    """Monthly 3-month interbank rates (fallback: the short-term call rate), per cent a year, by currency."""
    out, src = {}, {}
    for c, cc in OECD.items():
        s = None
        for subj in ("IR3TIB01", "IRSTCI01"):
            try:
                doc = get(f"https://api.db.nomics.world/v22/series/OECD/MEI/{cc}.{subj}.ST.M?observations=1").json()
                doc = doc["series"]["docs"][0]
                v = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").to_numpy(),
                              index=pd.PeriodIndex(doc["period"], freq="M")).dropna()
                if s is None:
                    s, src[c] = v, f"{subj} from {v.index.min()}"
                else:                                                      # fill earlier years from the call rate
                    add = v[v.index < s.index.min()]
                    if len(add):
                        s = pd.concat([add, s])
                        src[c] += f"; {subj} from {add.index.min()}"
            except Exception as ex:                                        # noqa: BLE001
                src[c] = src.get(c, "") + f" [{subj} failed: {repr(ex)[:60]}]"
        if s is not None:
            out[c] = s
    R = pd.DataFrame(out).reindex(pd.PeriodIndex(months, freq="M"))
    eur = R["EUR"] if "EUR" in R else pd.Series(np.nan, index=R.index)
    if "DEM" in R:
        eur = eur.where(eur.notna() & (eur.index >= pd.Period("1999-01", "M")), R["DEM"].where(R.index < pd.Period("1999-01", "M"), eur))
    R["EUR"] = eur
    R = R.reindex(columns=list(CUR8)) / 100.0
    return R.shift(1), src                                                 # known one month later


def month_end_mask(idx):
    m = idx.to_period("M")
    return np.r_[m[1:] != m[:-1], True]


def week_end_mask(idx):
    iso = idx.isocalendar()
    k = (iso.year.astype(int) * 100 + iso.week.astype(int)).to_numpy()
    return np.r_[k[1:] != k[:-1], True]


def build(X, RT):
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    idx = X.index
    me, we = month_end_mask(idx), week_end_mask(idx)
    rt_d = RT.copy()
    rt_d.index = rt_d.index.to_timestamp(how="start")
    rt_d = rt_d.reindex(idx, method="ffill")                               # the month's known rate, every day
    gap = np.r_[1.0, np.diff(idx.values).astype("timedelta64[D]").astype(float)]
    pnl = {k: pd.Series(0.0, index=idx) for k in STYLES + ["USD_LONG"]}
    usd_pos = {}
    lr_all = {}
    for s in FX:
        a, b = s[:3], s[3:]
        c = (X[b] / X[a]).dropna()
        if len(c) < 600:
            continue
        ci = c.index
        ret = c.pct_change()
        lr_all[s] = np.log(c).diff()
        vol = (ret.ewm(span=60, min_periods=40).std() * math.sqrt(252)).shift(1)
        size = (0.10 / vol).clip(upper=5.0)
        rd = (rt_d[a] - rt_d[b]).reindex(ci)
        carry = (rd.fillna(0.0) * pd.Series(gap, index=idx).reindex(ci) / 365.0)
        mm = ci.to_period("M")
        mE = np.r_[mm[1:] != mm[:-1], True]
        iso = ci.isocalendar()
        wk = (iso.year.astype(int) * 100 + iso.week.astype(int)).to_numpy()
        wE = np.r_[wk[1:] != wk[:-1], True]

        def run(raw_pos, mask):
            p = pd.Series(np.where(mask, raw_pos * size, np.nan), index=ci).ffill().shift(1).fillna(0.0)
            g = p * (ret.fillna(0.0) + carry)
            return g - COST * p.diff().abs().fillna(p.abs()), p

        t_sig = np.sign(c / c.shift(252) - 1.0)
        pnl["F2"] = pnl["F2"].add(run(t_sig, mE)[0], fill_value=0.0)
        pnl["F1"] = pnl["F1"].add(run(np.sign(rd), mE)[0], fill_value=0.0)
        pnl["F5"] = pnl["F5"].add(run(-np.sign(c / c.shift(5) - 1.0), wE)[0], fill_value=0.0)
        # F7: the weekly Connors family, long only, on close-only weekly bars
        w = c[wE]
        wv = w.to_numpy(float)
        bars = pd.DataFrame({"open": np.r_[wv[0], wv[:-1]], "high": wv, "low": wv, "close": wv},
                            index=pd.DatetimeIndex(w.index).tz_localize("UTC"))
        share = np.zeros(len(w))
        for rid in RULES:
            e = CATALOG[rid]
            share += (e.fn(bars.copy(), **e.params)["side"].to_numpy() > 0).astype(float)
        share = pd.Series(share / len(RULES), index=w.index).reindex(ci)
        pnl["F7"] = pnl["F7"].add(run(share.fillna(0.0), wE)[0], fill_value=0.0)
        if "USD" in (a, b):
            long_f = 1.0 if b == "USD" else -1.0
            g, p = run(pd.Series(long_f, index=ci), mE)
            pnl["USD_LONG"] = pnl["USD_LONG"].add(g, fill_value=0.0)
            usd_pos[s] = (long_f, size, ret.fillna(0.0) + carry, mE, ci)
    return pnl, usd_pos, pd.DataFrame(lr_all), rt_d


def features(LR, RT, months):
    idx = LR.index
    me = idx[month_end_mask(idx)]
    r2 = LR ** 2
    rv = np.sqrt(252 * r2.rolling(63, min_periods=50).mean())
    F = pd.DataFrame(index=me)
    F["RV63"] = rv.loc[me].median(axis=1)
    F["RV_CHG"] = np.log(F.RV63 / F.RV63.shift(12))
    usd = [s for s in LR.columns if "USD" in s]
    U = pd.DataFrame({(s[:3] if s[3:] == "USD" else s[3:]): (LR[s] if s[3:] == "USD" else -LR[s]) for s in usd})
    F["DISP63"] = LR.rolling(63, min_periods=50).sum().loc[me].std(axis=1) / (F.RV63 * math.sqrt(63 / 252))
    F["ER252"] = (LR.rolling(252, min_periods=200).sum().abs() / LR.abs().rolling(252, min_periods=200).sum()).loc[me].median(axis=1)
    r5 = LR.rolling(5).sum()
    F["VR5"] = (r5.rolling(504, min_periods=400).var() / (5 * LR.rolling(504, min_periods=400).var())).loc[me].median(axis=1)
    cc = []
    for e in me:
        u = U.loc[:e].tail(126).dropna()
        if len(u) >= 100:
            cm = u.corr().to_numpy()
            k = cm.shape[0]
            cc.append((cm.sum() - k) / (k * (k - 1)))
        else:
            cc.append(np.nan)
    F["CORR126"] = cc
    F["USD12"] = -U.rolling(252, min_periods=200).sum().mean(axis=1).loc[me]
    F.index = F.index.to_period("M").astype(str)
    R = RT.copy()
    R.index = R.index.astype(str)
    R = R.reindex(F.index)
    foreign = [c for c in CUR8 if c != "USD"]
    F["AFD"] = R["USD"] - R[foreign].mean(axis=1)
    F["CARRY_SD"] = R[list(CUR8)].std(axis=1)
    F["US3M_CHG12"] = R["USD"] - R["USD"].shift(12)
    return F.reindex(months)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--placebos", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import regime16 as G
    t0 = time.time()
    X = boe_rates()
    months_all = list(pd.period_range("1975-01", "2025-12", freq="M").astype(str))
    RT, rate_src = oecd_rates(months_all)
    print("BoE", X.index[0].date(), X.index[-1].date(), len(X), X.notna().sum().to_dict(), flush=True)
    print("rates", json.dumps(rate_src), {c: str(RT[c].first_valid_index()) for c in RT}, flush=True)
    pnl, usd_pos, LR, rt_d = build(X, RT)
    months = list(pd.period_range("1976-01", "2025-12", freq="M").astype(str))
    M = pd.DataFrame({k: v.groupby(v.index.to_period("M").astype(str)).sum() for k, v in pnl.items()}).reindex(months).fillna(0.0)
    M.to_csv(os.path.join(a.out, "styles_monthly.csv"))
    F = features(LR, RT, months)
    # ---------------------------------------------------------------- regime machinery of round 16, new windows
    G.INNER, G.CHECK, G.OOS0 = ("1999-01", "2025-12"), ("1981-01", "1998-12"), "1981-01"
    base = ["RV63", "RV_CHG", "CORR126", "DISP63", "ER252", "VR5", "AFD", "CARRY_SD", "US3M_CHG12", "USD12"]
    extra = ["TREND12", "REV12", "CARRY6", "TREND36", "DIP36", "CARRY36"]
    G.PRICE, G.OPT, G.CME, G.MACRO = base + extra, [], [], []
    G.FEATS = G.PRICE
    tier = G.Tier("FACTOR", M[STYLES].to_numpy(float), STYLES)
    zf = pd.DataFrame(tier.zw, index=months, columns=STYLES)
    for col, f in (("TREND36", "F2"), ("DIP36", "F7"), ("CARRY36", "F1")):
        F[col] = zf[f].rolling(36, min_periods=36).mean()
    D = G.Data(months, F, {"FACTOR": tier}, {})
    print("features used", D.feats, "dropped", D.dropped, flush=True)
    F.to_csv(os.path.join(a.out, "features_monthly.csv"))
    out = {"rate_sources": rate_src, "features": D.feats, "dropped": D.dropped,
           "boe": {"first": str(X.index[0].date()), "last": str(X.index[-1].date()), "days": int(len(X))}}
    # L1: styles by decade
    dec = {}
    yr = np.array([int(m[:4]) for m in months])
    for lab, lo, hi in (("1976-1985", 1976, 1985), ("1986-1995", 1986, 1995), ("1996-2005", 1996, 2005),
                        ("2006-2015", 2006, 2015), ("2016-2025", 2016, 2025), ("1981-1998", 1981, 1998),
                        ("1999-2025", 1999, 2025), ("1976-2025", 1976, 2025)):
        m = (yr >= lo) & (yr <= hi)
        dec[lab] = {STYLE_NAME[s]: {"sharpe": G.sharpe(M[s].to_numpy()[m]), "t": G.nw_t(M[s].to_numpy()[m])} for s in STYLES}
        dec[lab]["USD_LONG"] = {"sharpe": G.sharpe(M["USD_LONG"].to_numpy()[m]), "t": G.nw_t(M["USD_LONG"].to_numpy()[m])}
    out["L1_by_period"] = dec
    out["L1_corr"] = M[STYLES].rename(columns=STYLE_NAME).corr().round(3).to_dict()
    # L2: families
    cache = G.Cache()
    ks = (1, 2)
    surf, series = [], {}

    def add(fam, p, Sc):
        for k, R in G.evaluate(D, tier, Sc, ks).items():
            nm = G.pname(fam, p, k)
            series[nm] = R
            surf.append({"family": fam, "variant": nm, **p, "k": k, **G.series_stats(D, R)})

    for w in ("exp", "60"):
        add("B1", {"window": w}, G.scores_for(D, tier, "B1", {"window": w}, cache))
    for L in (12, 36, 60):
        add("M1", {"L": L}, G.scores_for(D, tier, "M1", {"L": L}, cache))
    for f in D.feats:
        for K in (2, 3):
            for n0 in (0, 12):
                for w in ("exp", "60"):
                    p = {"f": f, "K": K, "n0": n0, "window": w}
                    add("M3", p, G.scores_for(D, tier, "M3", p, cache))
    S = pd.DataFrame(surf)
    S.to_csv(os.path.join(a.out, "L2_surface.csv"), index=False)
    # C1: the literature rule (TREND only in the favourable state), and always-TREND
    i2 = STYLES.index("F2")
    fv = lambda c: D.FV[:, D.fcol[c]]                                      # noqa: E731
    rows = D.ridx
    fav = np.zeros(D.T, bool)
    for j in range(D.j0, D.T):
        r = rows[j] - 1
        hist_rv, hist_d = fv("RV63")[: r + 1], fv("DISP63")[: r + 1]
        ok = np.isfinite([fv("AFD")[r], fv("RV63")[r], fv("DISP63")[r]]).all()
        fav[j] = bool(ok and fv("AFD")[r] > 0 and fv("RV63")[r] < np.nanmedian(hist_rv) and fv("DISP63")[r] < np.nanmedian(hist_d))
    Sc_c1 = np.full(tier.z.shape, np.nan)
    Sc_c1[fav, i2] = 1.0
    Sc_tr = np.full(tier.z.shape, np.nan)
    Sc_tr[:, i2] = 1.0
    R_c1, R_tr = G.evaluate(D, tier, Sc_c1, (1,))[1], G.evaluate(D, tier, Sc_tr, (1,))[1]
    # choose on 1999-2025 (inner), verify on 1981-1998 (check)
    b1 = S[S.family == "B1"].loc[lambda x: x.inner_t.idxmax()]
    R_b1 = series[b1.variant]
    res = []
    for fam in ("M1", "M3"):
        r = S[S.family == fam].loc[lambda x: x.inner_t.idxmax()]
        R = series[r.variant]
        d = (R - R_b1)[D.mask_check]
        rec = {"family": fam, "variant": r.variant, "chosen_t_1999_2025": r.inner_t, "verify_mean": r.check_mean,
               "verify_t_1981_1998": r.check_t, "verify_sharpe": r.check_sharpe, "b1": b1.variant, "b1_verify_t": b1.check_t,
               "minus_b1_t": G.nw_t(d)}
        if fam == "M3":
            p = {"f": r.f, "K": int(r.K), "n0": int(r.n0), "window": r.window}
            sh = []
            for o in G.offsets_for(D, a.placebos, f"L2|{r.variant}"):
                Rp = G.evaluate(D, tier, G.scores_for(D.rolled(int(o)), tier, "M3", p, G.Cache()), (int(r.k),))[int(r.k)]
                sh.append(G.sharpe(Rp[D.mask_check]))
            sh = np.array(sh)
            sh = sh[np.isfinite(sh)]
            rec["placebo_p95_sharpe"] = float(np.quantile(sh, 0.95))
            rec["P3"] = bool(r.check_sharpe >= rec["placebo_p95_sharpe"])
        res.append(rec)
    st_c1, st_tr = G.series_stats(D, R_c1), G.series_stats(D, R_tr)
    res.append({"family": "C1", "variant": "TREND when AFD > 0, low volatility, low dispersion; else cash",
                "chosen_t_1999_2025": st_c1["inner_t"], "verify_mean": st_c1["check_mean"],
                "verify_t_1981_1998": st_c1["check_t"], "verify_sharpe": st_c1["check_sharpe"],
                "b1": b1.variant, "b1_verify_t": b1.check_t, "minus_b1_t": G.nw_t((R_c1 - R_b1)[D.mask_check]),
                "minus_always_trend_t": G.nw_t((R_c1 - R_tr)[D.mask_check]), "always_trend_verify_t": st_tr["check_t"],
                "favourable_share_verify": float(fav[D.mask_check].mean())})
    out["L2"] = res
    out["L2_surface_summary"] = S.groupby("family").agg(settings=("variant", "size"), med_inner_t=("inner_t", "median"),
                                                        med_check_t=("check_t", "median"),
                                                        share_check_ge_1645=("check_t", lambda x: float((x >= 1.645).mean())),
                                                        best_check_t=("check_t", "max")).round(3).reset_index().to_dict("records")
    # the best single-feature splits on the verification period, for information
    out["L2_M3_by_feature"] = (S[S.family == "M3"].groupby("f").agg(med_inner_t=("inner_t", "median"), med_check_t=("check_t", "median"))
                               .round(2).reset_index().to_dict("records"))
    # L3: global FX volatility and the dollar
    rv = F["RV63"].to_numpy()
    sign = np.full(len(months), np.nan)
    for j in range(1, len(months)):
        h = rv[:j]
        h = h[np.isfinite(h)]
        if len(h) >= 36 and np.isfinite(rv[j - 1]):
            sign[j] = 1.0 if rv[j - 1] > np.median(h) else -1.0           # decided at the end of month j-1
    sgn_m = pd.Series(sign, index=months)
    timed = pd.Series(0.0, index=X.index)
    for s, (long_f, size, tot, mE, ci) in usd_pos.items():
        sg = sgn_m.reindex(ci.to_period("M").astype(str)).to_numpy()      # the sign that applies in each day's month
        raw = pd.Series(np.where(mE, long_f * size, np.nan), index=ci).ffill().shift(1).fillna(0.0)
        p = raw * np.nan_to_num(sg)
        timed = timed.add(p * tot - COST * p.diff().abs().fillna(p.abs()), fill_value=0.0)
    T3 = timed.groupby(timed.index.to_period("M").astype(str)).sum().reindex(months).fillna(0.0).to_numpy()
    L3 = {}
    for lab, lo, hi in (("1981-1998", 1981, 1998), ("1999-2025", 1999, 2025), ("1981-2025", 1981, 2025)):
        m = (yr >= lo) & (yr <= hi)
        L3[lab] = {"timed_mean": float(T3[m].mean()), "timed_t": G.nw_t(T3[m]), "timed_sharpe": G.sharpe(T3[m]),
                   "long_t": G.nw_t(M["USD_LONG"].to_numpy()[m]), "minus_long_t": G.nw_t((T3 - M["USD_LONG"].to_numpy())[m]),
                   "share_long_foreign": float(np.nanmean(sign[m] > 0))}
    out["L3"] = L3
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(a.out, "long_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("L1_by_period", "L2", "L3")}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
