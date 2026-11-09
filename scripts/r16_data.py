"""Round 16 data (docs/round16_preregistration.md, sections 1-2): the strategy library, the FACTOR tier, the regime
features and the options data. Every series is cut at 2025-12-31.

Subcommands
  lib   --inp S15_ARTIFACTS --out DIR   library matrices (FX book, monthly net P&L per configuration) and F7
  fx    --inp D1P_ARTIFACTS --out DIR   PRICE features 1-8, 11, 13-16 and factors F1-F6, F8, F9
  cme   --dbn DIR --compact DIR --out DIR   CME at-the-money IV and 25-delta risk reversals at month ends
  web   --out DIR                       CBOE volatility indices, FRED fallbacks, Treasury yields, NFCI, EPU
  merge --inp DIR --out DIR             monthly feature table, factor table, coverage report
"""
from __future__ import annotations

import argparse
import glob
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

DEV_END = pd.Timestamp("2025-12-31")
LIB_MONTHS = pd.period_range("2008-01", "2025-12", freq="M").astype(str)
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
FAMILY_W1 = ["S.DAILY.DOUBLE_SEVEN", "S.DAILY.CUM_RSI", "S.FAMOUS.CONNORS_RSI2", "S.BOOK.CONNORS_R3",
             "S.BOOK.CONNORS_PCTB", "S.BOOK.CONNORS_CRSI", "S.DAILY.THREE_DOWN"]
USD_PAIRS = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "NZDUSD": 1, "USDJPY": -1, "USDCAD": -1, "USDCHF": -1}


def cfg_name(p: dict):
    return ",".join(f"{k}={p[k]}" for k in sorted(p)) or "default"


# ================================================================================================ library
def cmd_lib(a):
    import long_lab as LL
    fx = set(LL.FX_SYMBOLS)
    acc, act, nfiles = None, None, 0
    for pat in ("SAB_*.parquet", "SC_*.parquet"):
        for f in sorted(glob.glob(os.path.join(a.inp, "**", pat), recursive=True)):
            sym = os.path.basename(f).split("_", 1)[1][:-8]
            if sym not in fx:
                continue
            d = pd.read_parquet(f, columns=["unit", "cfg", "dir", "month", "pnl"])
            d["month"] = d["month"].astype(str)
            d = d[(d.month >= LIB_MONTHS[0]) & (d.month <= LIB_MONTHS[-1])]
            for c in ("unit", "cfg"):
                d[c] = d[c].astype(str)
            pv = d.pivot_table(index=["unit", "cfg", "dir"], columns="month", values="pnl", aggfunc="sum").astype(float)
            acc = pv if acc is None else acc.add(pv, fill_value=0.0)
            nfiles += 1
            print("read", f, len(d), flush=True)
            del d, pv
    acc = acc.reindex(columns=LIB_MONTHS).fillna(0.0)
    acc.columns = [str(c) for c in acc.columns]
    u = acc.index.get_level_values("unit").astype(str)
    plc = u.str.startswith(("PLACEBO_", "PLACEBOF_"))
    os.makedirs(a.out, exist_ok=True)
    acc[~plc].to_parquet(os.path.join(a.out, "lib_cfg.parquet"))
    acc[plc].to_parquet(os.path.join(a.out, "lib_placebo.parquet"))
    # F7: the weekly dip-buying family, direction +1, default setting
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    parts, used = [], {}
    for rid in FAMILY_W1:
        key = f"{rid}|W1"
        sub = acc[(u == key) & (acc.index.get_level_values("dir") == 1)]
        if sub.empty:
            used[rid] = "missing"
            continue
        dflt = cfg_name(dict(CATALOG[rid].params)) if rid in CATALOG else "default"
        cfgs = sub.index.get_level_values("cfg").astype(str)
        row = sub[cfgs == dflt]
        if row.empty:
            row = sub
            used[rid] = f"default {dflt} absent: mean of {len(sub)} settings"
        else:
            used[rid] = dflt
        parts.append(row.mean(axis=0))
    F7 = pd.concat(parts, axis=1).mean(axis=1) if parts else pd.Series(0.0, index=LIB_MONTHS)
    pd.DataFrame({"F7": F7}).to_parquet(os.path.join(a.out, "factor_f7.parquet"))
    json.dump({"files": nfiles, "configs_real": int((~plc).sum()), "configs_placebo": int(plc.sum()),
               "units_real": int(pd.Index(u[~plc]).nunique()), "f7_rows": used},
              open(os.path.join(a.out, "lib_meta.json"), "w"), indent=1)
    print("library", int((~plc).sum()), "real configs,", int(plc.sum()), "placebo; F7", used, flush=True)


# ================================================================================================ FX features
def cmd_fx(a):
    import long_lab as LL
    import trend14 as T
    P = T.load_panels(a.inp)
    print("panels", len(P), sorted(P), flush=True)
    fx = [s for s in LL.FX_SYMBOLS if s in P]
    closes = {}
    for s, d in P.items():
        closes[s] = pd.Series(d.c_mid.to_numpy(float), index=pd.DatetimeIndex(d.tday))
    C = pd.DataFrame(closes).sort_index()
    C = C[C.index <= DEV_END]
    R = np.log(C).diff()
    mkey = C.index.to_period("M")
    is_me = pd.Series(np.r_[mkey[1:] != mkey[:-1], True], index=C.index)
    me_dates = C.index[is_me.to_numpy()]
    months = me_dates.to_period("M").astype(str)
    rv21 = np.sqrt(252 * (R ** 2).rolling(21, min_periods=18).mean())
    rv252 = np.sqrt(252 * (R ** 2).rolling(252, min_periods=200).mean())
    out = pd.DataFrame(index=months)
    out["RV21"] = rv21.loc[me_dates, fx].median(axis=1).to_numpy()
    out["RV_CHG"] = np.log(out.RV21 / out.RV21.shift(3))
    out["RV252"] = rv252.loc[me_dates, fx].median(axis=1).to_numpy()
    U = pd.DataFrame({s[:3] if sg == 1 else s[3:]: sg * R[s] for s, sg in USD_PAIRS.items() if s in R})
    L21 = np.log(C).diff(21)
    out["DISP21"] = (L21.loc[me_dates, fx].std(axis=1) / (out.RV21.to_numpy() * math.sqrt(21 / 252))).to_numpy()
    absr = R[fx].abs()
    er = (np.log(C[fx]).diff(63).abs() / absr.rolling(63, min_periods=60).sum())
    out["ER63"] = er.loc[me_dates].median(axis=1).to_numpy()
    # weekly closes (last close of each ISO week)
    iso = C.index.isocalendar()
    wk = (iso.year * 100 + iso.week).to_numpy()
    wend = np.r_[wk[1:] != wk[:-1], True]
    Wc = C[wend]
    Wr = np.log(Wc[fx]).diff()
    corr, vr, acw = [], [], []
    for e in me_dates:
        u = U.loc[:e].dropna().tail(63)
        if len(u) >= 55:
            cm = u.corr().to_numpy()
            k = cm.shape[0]
            corr.append(float((cm.sum() - k) / (k * (k - 1))))
        else:
            corr.append(np.nan)
        r = R.loc[:e, fx].tail(250)
        vals = []
        if len(r) >= 250:
            for s in fx:
                x = r[s].to_numpy()
                if np.isfinite(x).sum() < 240:
                    continue
                x = np.nan_to_num(x)
                v1 = x.var()
                b5 = x.reshape(50, 5).sum(axis=1)
                if v1 > 0:
                    vals.append(b5.var() / (5 * v1))
        vr.append(float(np.median(vals)) if vals else np.nan)
        w = Wr.loc[:e].tail(52)
        vals = []
        if len(w) >= 52:
            for s in fx:
                x = w[s].dropna()
                if len(x) >= 45 and x.std() > 0:
                    vals.append(x.autocorr(1))
        acw.append(float(np.nanmedian(vals)) if vals else np.nan)
    out["CORR63"] = corr
    out["VR5"] = vr
    out["AC_W"] = acw
    rd = pd.DataFrame({s: pd.Series(P[s].rd.to_numpy(float), index=pd.DatetimeIndex(P[s].tday)) for s in fx}).sort_index()
    rd = rd.reindex(C.index).ffill()
    out["CARRY_SD"] = rd.loc[me_dates].std(axis=1).to_numpy()
    usd = -U.rolling(63, min_periods=60).sum().mean(axis=1)
    out["USD3"] = usd.loc[me_dates].to_numpy()
    for s, col in (("USA500IDXUSD", "SPX3"), ("XAUUSD", "GOLD3")):
        out[col] = np.log(C[s]).diff(63).loc[me_dates].to_numpy() if s in C else np.nan
    out["SPX_RV21"] = rv21.loc[me_dates, "USA500IDXUSD"].to_numpy() if "USA500IDXUSD" in C else np.nan
    for s in ("EURUSD", "USDJPY"):
        out[f"RV21_{s}"] = rv21.loc[me_dates, s].to_numpy()
    out.index.name = "month"
    os.makedirs(a.out, exist_ok=True)
    out.to_parquet(os.path.join(a.out, "fx_features.parquet"))
    print(out.describe().T.round(4).to_string(), flush=True)
    # ---------------------------------------------------------------- factors F1-F6, F8, F9 on the 28 crosses
    fac = {}
    for s in fx:
        d = P[s].reset_index(drop=True)
        c, h, lo = d.close, d.high, d.low
        vol = T.ewvol_ann(c)
        size = 0.10 / vol
        buf = (0.25 * size).shift(1).to_numpy()
        res = {}
        res["F1"] = T.run_cont(d, (np.sign(d.rd) * size).shift(1).to_numpy(), buf)
        res["F2"] = T.run_cont(d, T.monthly_pos(d, np.sign(c / c.shift(252) - 1), vol, tv=0.10))
        res["F3"] = T.run_cont(d, T.monthly_pos(d, np.sign(c / c.shift(63) - 1), vol, tv=0.10))
        ew = np.sign(c.ewm(span=16, min_periods=64).mean() - c.ewm(span=64, min_periods=64).mean())
        res["F4"] = T.run_cont(d, (ew * size).shift(1).to_numpy(), buf)
        iso = d.tday.dt.isocalendar()
        key = (iso.year * 100 + iso.week).to_numpy()
        we = pd.Series(np.r_[key[1:] != key[:-1], True], index=d.index)
        res["F5"] = T.run_cont(d, (-np.sign(c / c.shift(5) - 1) * size).where(we).ffill().shift(1).to_numpy())
        res["F6"] = T.run_cont(d, (-np.sign(c / c.shift(1) - 1) * size).shift(1).to_numpy())
        hi20, lo20 = T.prior_max(h, 20), T.prior_min(lo, 20)
        st = pd.Series(np.where(c > hi20, 1.0, np.where(c < lo20, -1.0, np.nan)), index=d.index).ffill()
        res["F8"] = T.run_cont(d, (st * size).shift(1).to_numpy(), buf)
        if s in USD_PAIRS:
            res["F9"] = T.run_cont(d, (-USD_PAIRS[s] * size).shift(1).to_numpy(), buf)
        for k, (pnl, gp, sw) in res.items():
            m = pd.Series(pnl, index=d.month).groupby(level=0).sum()
            fac[k] = m if k not in fac else fac[k].add(m, fill_value=0.0)
    F = pd.DataFrame(fac).reindex(LIB_MONTHS).fillna(0.0)
    F.index.name = "month"
    F.to_parquet(os.path.join(a.out, "factors_fx.parquet"))
    print("factors", F.columns.tolist(), "months", len(F), flush=True)


# ================================================================================================ CME options
def cmd_cme(a):
    import databento_lab as DL
    import round9_lab as R9
    R9.A = argparse.Namespace(dbn=a.dbn, compact=a.compact, out=a.out)
    os.makedirs(a.out, exist_ok=True)
    extra = {"GBP": ("6B", ("6B_OPT", "GBU_OPT")), "AUD": ("6A", ("6A_OPT", "ADU_OPT")),
             "CAD": ("6C", ("6C_OPT", "CAU_OPT")), "CHF": ("6S", ("6S_OPT", "CHU_OPT"))}
    # the round-16 purchase (if any) has the same layout under the O2fx_ prefix
    for ccy, (fut, opts) in extra.items():
        if any(glob.glob(os.path.join(a.compact, f"O2fx_defs_{o}.parquet")) for o in opts):
            for o in opts:
                for kind in ("defs", "stats"):
                    src = os.path.join(a.compact, f"O2fx_{kind}_{o}.parquet")
                    dst = os.path.join(a.compact, f"O1fx_{kind}_{o}.parquet")
                    if os.path.exists(src) and not os.path.exists(dst):
                        os.symlink(os.path.abspath(src), dst)
            R9.PARENTS[ccy] = (fut, opts)
    rows, meta = [], {}
    for ccy in R9.PARENTS:
        t0 = time.time()
        L = R9.load_opts(ccy)
        if L is None:
            meta[ccy] = "data missing"
            continue
        S, O, C = L
        S = S[S.date <= DEV_END]
        dates = pd.Series(sorted(S.date.unique()))
        me = dates.groupby(dates.dt.to_period("M")).max()
        n_ok = 0
        for per, d in me.items():
            d = pd.Timestamp(d)
            day = S[S.date == d]
            exps = [e for e in day.exp.unique() if 10 <= (pd.Timestamp(e) - d).days <= 60]
            rec = {"month": str(per), "ccy": ccy, "date": d, "iv": np.nan, "rr25": np.nan, "days": np.nan}
            if exps:
                E = min(exps, key=lambda e: abs((pd.Timestamp(e) - d).days - 30))
                dd = day[day.exp == E]
                u = dd.underlying.value_counts().index[0]
                dd = dd[dd.underlying == u]
                F = float(dd.F.iloc[0])
                Tm = (pd.Timestamp(E) - d).days / 365.0
                ivs = []
                for cp in ("C", "P"):
                    x = dd[dd.cp == cp]
                    if len(x):
                        j = (x.K - F).abs().idxmin()
                        iv = DL.implied(F, float(x.K.loc[j]), Tm, float(x.P.loc[j]), cp)
                        if np.isfinite(iv) and abs(x.K.loc[j] / F - 1) < 0.01:
                            ivs.append(iv)
                rec.update(iv=float(np.mean(ivs)) if ivs else np.nan, rr25=R9.rr25(dd, F, Tm), days=(pd.Timestamp(E) - d).days)
                n_ok += int(np.isfinite(rec["iv"]))
            rows.append(rec)
        meta[ccy] = {"months": int(len(me)), "with_iv": n_ok, "first": str(me.index.min()), "last": str(me.index.max()),
                     "s": round(time.time() - t0, 1), "units": R9.LOG.get("opt_units", {}).get(ccy)}
        print(ccy, meta[ccy], flush=True)
    D = pd.DataFrame(rows)
    D.to_parquet(os.path.join(a.out, "cme_iv.parquet"))
    json.dump(meta, open(os.path.join(a.out, "cme_meta.json"), "w"), indent=1, default=str)


# ================================================================================================ web
def get(url, tries=3, deadline=60):
    """GET with a hard total deadline per attempt (a slow, trickling response cannot hang the job)."""
    import requests
    last = None
    for k in range(tries):
        t0 = time.time()
        try:
            with requests.get(url, headers=UA, timeout=(10, 20), stream=True) as r:
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}"
                else:
                    buf = bytearray()
                    for chunk in r.iter_content(65536):
                        buf.extend(chunk)
                        if time.time() - t0 > deadline:
                            raise TimeoutError(f"deadline {deadline}s after {len(buf)} bytes")
                    if len(buf) > 200:
                        print(f"  got {url} {len(buf)} bytes in {time.time() - t0:.1f}s", flush=True)
                        return buf.decode("utf-8", "replace")
                    last = f"short body {len(buf)}"
        except Exception as ex:                                            # noqa: BLE001
            last = repr(ex)[:160]
        print(f"  attempt {k + 1} {url}: {last} ({time.time() - t0:.1f}s)", flush=True)
        time.sleep(3 * (k + 1))
    raise RuntimeError(last)


def parse_cboe(txt):
    d = pd.read_csv(io.StringIO(txt))
    d.columns = [c.strip().upper() for c in d.columns]
    dc = d.columns[0]
    vc = "CLOSE" if "CLOSE" in d.columns else d.columns[-1]
    s = pd.Series(pd.to_numeric(d[vc], errors="coerce").to_numpy(), index=pd.to_datetime(d[dc], errors="coerce"))
    return s[s.index.notna()].dropna().sort_index()


def parse_fred(txt):
    d = pd.read_csv(io.StringIO(txt))
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0], errors="coerce"))
    return s[s.index.notna()].dropna().sort_index()


def cmd_web(a):
    os.makedirs(a.out, exist_ok=True)
    cboe = ["VIX", "VIX3M", "VVIX", "SKEW", "GVZ", "OVX", "EVZ"]
    fred_fb = {"VIX": "VIXCLS", "VIX3M": "VXVCLS", "GVZ": "GVZCLS", "OVX": "OVXCLS", "EVZ": "EVZCLS"}
    fred = {"DGS2": "DGS2", "DGS10": "DGS10", "NFCI": "NFCI", "EPU": "USEPUINDXD"}
    series, src = {}, {}
    for k in cboe:
        try:
            s = parse_cboe(get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{k}_History.csv"))
            series[k], src[k] = s, "cboe"
        except Exception as ex:                                            # noqa: BLE001
            src[k] = f"cboe failed: {ex}"
        if k in fred_fb and (k not in series or series[k].index.max() < pd.Timestamp("2025-06-30")):
            try:
                s = parse_fred(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_fb[k]}"))
                if k not in series or s.index.max() > series[k].index.max():
                    series[k], src[k] = s, f"fred {fred_fb[k]}" + (f" (cboe: {src[k]})" if k in src else "")
            except Exception as ex:                                        # noqa: BLE001
                src[k] = str(src.get(k, "")) + f"; fred failed: {ex}"
    for k, sid in fred.items():
        try:
            series[k], src[k] = parse_fred(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")), f"fred {sid}"
        except Exception as ex:                                            # noqa: BLE001
            src[k] = f"fred failed: {ex}"
    meta = {}
    for k, s in series.items():
        s = s[s.index <= DEV_END]                                          # development data only
        series[k] = s
        meta[k] = {"source": src[k], "first": str(s.index.min().date()) if len(s) else None,
                   "last": str(s.index.max().date()) if len(s) else None, "rows": int(len(s))}
    for k in set(src) - set(series):
        meta[k] = {"source": src[k], "rows": 0}
    D = pd.DataFrame(series).sort_index()
    D.index.name = "date"
    D.to_parquet(os.path.join(a.out, "web_daily.parquet"))
    json.dump(meta, open(os.path.join(a.out, "web_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)


# ================================================================================================ merge
def month_end_value(s: pd.Series, months, lag_days=0, mean_days=0):
    out = []
    for m in months:
        e = pd.Period(m, "M").end_time.normalize() - pd.Timedelta(days=lag_days)
        x = s[s.index <= e]
        if not len(x) or (e - x.index[-1]).days > 10 + (7 if lag_days else 0):
            out.append(np.nan)
            continue
        out.append(float(x[x.index > e - pd.Timedelta(days=mean_days)].mean()) if mean_days else float(x.iloc[-1]))
    return np.array(out)


def cmd_merge(a):
    fxf = pd.read_parquet(glob.glob(os.path.join(a.inp, "**", "fx_features.parquet"), recursive=True)[0])
    months = pd.period_range("2007-01", "2025-12", freq="M").astype(str)
    X = fxf.reindex(months)
    web = glob.glob(os.path.join(a.inp, "**", "web_daily.parquet"), recursive=True)
    W = pd.read_parquet(web[0]) if web else pd.DataFrame()
    def col(k, **kw):
        return month_end_value(W[k].dropna(), months, **kw) if k in W else np.full(len(months), np.nan)
    vix = col("VIX")
    X["VIX"] = np.log(vix)
    X["VIX_TS"] = vix / col("VIX3M")
    X["VVIX"] = np.log(col("VVIX"))
    X["SKEW"] = col("SKEW")
    X["GVZ"] = np.log(col("GVZ"))
    X["OVX"] = np.log(col("OVX"))
    evz = col("EVZ")
    X["EVZ"] = np.log(evz)
    X["VRP_SPX"] = vix / 100 - X["SPX_RV21"].to_numpy()
    X["DVIX"] = np.r_[np.nan, np.diff(np.log(vix))]
    cme = glob.glob(os.path.join(a.inp, "**", "cme_iv.parquet"), recursive=True)
    if cme:
        Cm = pd.read_parquet(cme[0])
        for ccy, g in Cm.groupby("ccy"):
            g = g.set_index("month")
            X[f"{ccy}_IV"] = g.iv.reindex(months).to_numpy()
            X[f"{ccy}_RR25"] = g.rr25.reindex(months).to_numpy()
        parts = []
        if "EUR_IV" in X:
            parts.append(X["EUR_IV"] - X["RV21_EURUSD"])
        if "JPY_IV" in X:
            parts.append(X["JPY_IV"] - X["RV21_USDJPY"])
        if parts:
            X["FX_VRP"] = pd.concat(parts, axis=1).mean(axis=1, skipna=False)
    dgs2, dgs10 = col("DGS2"), col("DGS10")
    X["US2Y_CHG3"] = dgs2 - np.r_[np.full(3, np.nan), dgs2[:-3]]
    X["SLOPE"] = dgs10 - dgs2
    X["NFCI"] = col("NFCI", lag_days=7)
    X["EPU"] = col("EPU", lag_days=1, mean_days=30)
    X.index.name = "month"
    # EVZ is used only if its history runs to 2025 (pre-registration, feature 23)
    evz_ok = bool(np.isfinite(X.loc["2025-01":"2025-12", "EVZ"]).sum() >= 6) if "EVZ" in X else False
    if not evz_ok:
        X["EVZ"] = np.nan
    os.makedirs(a.out, exist_ok=True)
    X.to_parquet(os.path.join(a.out, "features.parquet"))
    F = pd.read_parquet(glob.glob(os.path.join(a.inp, "**", "factors_fx.parquet"), recursive=True)[0])
    F7 = pd.read_parquet(glob.glob(os.path.join(a.inp, "**", "factor_f7.parquet"), recursive=True)[0])
    F["F7"] = F7["F7"].reindex(F.index).fillna(0.0)
    F = F[[f"F{i}" for i in range(1, 10)]]
    F.to_parquet(os.path.join(a.out, "factors.parquet"))
    rng = (X.index >= "2008-01")
    cov = {}
    for c in X.columns:
        v = X.loc[rng, c]
        ok = v.notna()
        cov[c] = {"first": v[ok].index.min() if ok.any() else None, "last": v[ok].index.max() if ok.any() else None,
                  "missing_2008_2025": int((~ok).sum()), "mean": float(v.mean()) if ok.any() else None,
                  "sd": float(v.std()) if ok.sum() > 2 else None}
    rep = {"features": cov, "evz_used": evz_ok,
           "factors_nonzero_months": {k: int((F[k] != 0).sum()) for k in F.columns}}
    for f in ("lib_meta.json", "web_meta.json", "cme_meta.json"):
        g = glob.glob(os.path.join(a.inp, "**", f), recursive=True)
        if g:
            rep[f[:-5]] = json.load(open(g[0]))
    json.dump(rep, open(os.path.join(a.out, "coverage.json"), "w"), indent=1, default=str)
    lines = ["| Feature | First month | Last month | Missing months 2008-2025 |", "|---|---|---|---|"]
    for c, v in cov.items():
        lines.append(f"| {c} | {v['first']} | {v['last']} | {v['missing_2008_2025']} |")
    open(os.path.join(a.out, "coverage.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in rep.items() if k != "features"}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lib", "fx", "cme", "web", "merge"])
    ap.add_argument("--inp", default="")
    ap.add_argument("--dbn", default="dbn")
    ap.add_argument("--compact", default="compact")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"lib": cmd_lib, "fx": cmd_fx, "cme": cmd_cme, "web": cmd_web, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
