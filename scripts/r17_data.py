"""Round 17 data (docs/round17_preregistration.md, sections 1-2). Every series is cut at 2025-12-31; weekend
trading days are folded into the following Monday.

Subcommands
  book  --inp S17_ARTIFACTS --out DIR   FX-book daily matrices (real / placebo configurations of both engines), the
                                        PAIR-UNIT weekly and monthly matrices, F7 daily, intraday state features
  fx    --inp D1P_ARTIFACTS --web WEB --out DIR   daily PRICE features, per-instrument features, factors F1-F6, F8, F9
  cme   --dbn DIR --compact DIR --out DIR   daily CME at-the-money IV and 25-delta risk reversals (euro, yen)
  merge --inp DIR --web WEB --out DIR   the daily market feature table (PRICE / INTRA / OPT / CME / MACRO), factors
"""
from __future__ import annotations

import argparse
import glob
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
USD_PAIRS = {"EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "NZDUSD": 1, "USDJPY": -1, "USDCAD": -1, "USDCHF": -1}
FAMILY_W1 = ["S.DAILY.DOUBLE_SEVEN", "S.DAILY.CUM_RSI", "S.FAMOUS.CONNORS_RSI2", "S.BOOK.CONNORS_R3",
             "S.BOOK.CONNORS_PCTB", "S.BOOK.CONNORS_CRSI", "S.DAILY.THREE_DOWN"]
F7_DEFAULT = {"S.DAILY.DOUBLE_SEVEN": "n=7,short=0", "S.DAILY.CUM_RSI": "k=2,lo=35,short=0,x=65",
              "S.FAMOUS.CONNORS_RSI2": "lo=10", "S.BOOK.CONNORS_R3": "short=0", "S.BOOK.CONNORS_PCTB": "short=0",
              "S.BOOK.CONNORS_CRSI": "lo=10,short=0", "S.DAILY.THREE_DOWN": "f=1,n=3,short=0"}


def fx_symbols():
    import long_lab as LL
    return list(LL.FX_SYMBOLS)


def fold(days) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(pd.to_datetime(days))
    wd = d.dayofweek
    return d + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")


def is_placebo(unit: str) -> bool:
    return unit.startswith(("PLACEBO_", "PLACEBOF_", "PLACEBO17_"))


def is_rule(unit: str) -> bool:
    return not unit.startswith(("FEAT:", "PLACEBO_", "PLACEBOF_", "PLACEBO17_"))


def period_keys(days: pd.DatetimeIndex):
    iso = days.isocalendar()
    wk = (iso.year.astype(int) * 100 + iso.week.astype(int)).astype(str).to_numpy()
    mo = days.strftime("%Y-%m").to_numpy()
    return wk, mo


def agg_periods(mat: np.ndarray, keys: np.ndarray):
    """Sum columns (days, sorted in time) of mat by period key; periods are contiguous runs of equal keys."""
    keys = np.asarray(keys).astype(str)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return keys[starts], np.add.reduceat(np.asarray(mat, np.float64), starts, axis=1)


# ================================================================================================ book
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    days = fold(z["days"])
    ok = days <= DEV_END
    return z["unit"].astype(str), z["cfg"].astype(str), z["dir"].astype(int), days[ok], z["pnl"][:, ok]


def cmd_book(a):
    os.makedirs(a.out, exist_ok=True)
    fx = set(fx_symbols())
    files = {}
    for f in glob.glob(os.path.join(a.inp, "**", "daily1[57]_*.npz"), recursive=True):
        sym = os.path.basename(f).split("_", 1)[1][:-4]
        files.setdefault(sym, []).append(f)
    print("symbols", len(files), sorted(files), flush=True)
    # the calendar: every (folded) trading day of any FX file
    cal = None
    for sym in sorted(files):
        if sym not in fx:
            continue
        for f in files[sym]:
            d = fold(np.load(f, allow_pickle=True)["days"])
            d = d[d <= DEV_END]
            cal = d.unique() if cal is None else cal.union(d.unique())
    cal = pd.DatetimeIndex(sorted(cal))
    pos = pd.Series(np.arange(len(cal)), index=cal)
    keyrow, book_rows = {}, []
    acc = {}
    pair_unit_rows, pair_unit_keys = [], []
    meta_pair = {}
    feats = []
    for sym in sorted(files):
        t0 = time.time()
        sym_units = {}
        for f in files[sym]:
            eng = "D" if os.path.basename(f).startswith("daily15") else "I"
            unit, cfg, dr, days, P = load_npz(f)
            # fold duplicates created by weekend folding
            dser = pd.Series(np.arange(len(days)), index=days)
            if dser.index.has_duplicates:
                uniq = days.unique()
                Q = np.zeros((P.shape[0], len(uniq)), np.float32)
                ii = pd.Series(np.arange(len(uniq)), index=uniq)[days].to_numpy()
                for c in range(P.shape[1]):
                    Q[:, ii[c]] += P[:, c]
                P, days = Q, uniq
            cols = pos.reindex(days).to_numpy()
            okc = np.isfinite(cols)
            cols = cols[okc].astype(int)
            P = P[:, okc]
            if sym in fx:
                for i in range(len(unit)):
                    k = (unit[i], cfg[i], int(dr[i]), eng)
                    if k not in acc:
                        acc[k] = np.zeros(len(cal), np.float64)
                    acc[k][cols] += P[i]
            # PAIR-UNIT: rule and placebo units of every instrument, mean over the unit's settings
            for i in range(len(unit)):
                u = unit[i]
                if u.startswith("FEAT:") or u.startswith("PLACEBOF_"):
                    continue
                k = (sym, u, int(dr[i]), eng)
                if k not in sym_units:
                    sym_units[k] = [np.zeros(len(cal), np.float64), 0]
                sym_units[k][0][cols] += P[i]
                sym_units[k][1] += 1
            del P
        for k, (v, n) in sym_units.items():
            pair_unit_keys.append(k)
            pair_unit_rows.append((v / max(n, 1)).astype(np.float32))
        ff = glob.glob(os.path.join(a.inp, "**", f"feats17_{sym}.parquet"), recursive=True)
        if ff:
            F = pd.read_parquet(ff[0])
            F.index = fold(F.index)
            F.index.name = "day"
            F = F[~F.index.duplicated(keep="last")]
            F = F[F.index <= DEV_END]
            F["symbol"] = sym
            feats.append(F.reset_index())
        print(sym, "units", len(sym_units), round(time.time() - t0, 1), "s", flush=True)
    keys = list(acc)
    meta = pd.DataFrame(keys, columns=["unit", "cfg", "dir", "engine"])
    meta["tf"] = meta.unit.str.split("|").str[-1]
    meta["placebo"] = meta.unit.map(is_placebo)
    import fxlab.strategies  # noqa: F401
    from fxlab.registry import CATALOG
    def fam(u):
        rid = u.split("|")[0]
        if rid.startswith("FEAT:"):
            return "feature:" + (CATALOG[rid[5:].split(":")[0]].family if rid[5:].split(":")[0] in CATALOG else "?")
        if rid.startswith("PLACEBO"):
            return "placebo"
        return CATALOG[rid].family if rid in CATALOG else "round12"
    meta["family"] = meta.unit.map(fam)
    M = np.vstack([acc[k] for k in keys]).astype(np.float32)
    days_s = np.array(cal.strftime("%Y-%m-%d"), dtype="U10")
    for lab, msk in (("real", ~meta.placebo.to_numpy()), ("placebo", meta.placebo.to_numpy())):
        np.save(os.path.join(a.out, f"book_{lab}.npy"), M[msk])
        meta[msk].reset_index(drop=True).to_parquet(os.path.join(a.out, f"book_{lab}_meta.parquet"))
    np.save(os.path.join(a.out, "days.npy"), days_s)
    # F7 daily (FX book): the seven weekly Connors-family units, direction +1, default settings
    parts = []
    for rid, c in F7_DEFAULT.items():
        k = (f"{rid}|W1", c, 1, "D")
        if k in acc:
            parts.append(acc[k])
    pd.DataFrame({"F7": np.mean(parts, axis=0) if parts else np.zeros(len(cal))}, index=cal).to_parquet(
        os.path.join(a.out, "f7_daily.parquet"))
    # PAIR-UNIT weekly and monthly
    PU = np.vstack(pair_unit_rows)
    pk = pd.DataFrame(pair_unit_keys, columns=["symbol", "unit", "dir", "engine"])
    pk["tf"] = pk.unit.str.split("|").str[-1]
    pk["placebo"] = pk.unit.map(is_placebo)
    pk["family"] = pk.unit.map(fam)
    wk, mo = period_keys(cal)
    for lab, keys_ in (("W", wk), ("M", mo)):
        uk, X = agg_periods(PU, keys_)
        for pl, msk in (("real", ~pk.placebo.to_numpy()), ("placebo", pk.placebo.to_numpy())):
            np.save(os.path.join(a.out, f"pair_{lab}_{pl}.npy"), X[msk].astype(np.float32))
            pk[msk].reset_index(drop=True).to_parquet(os.path.join(a.out, f"pair_{pl}_meta.parquet"))
        np.save(os.path.join(a.out, f"pair_{lab}_periods.npy"), uk)
    if feats:
        pd.concat(feats, ignore_index=True).to_parquet(os.path.join(a.out, "intraday_feats_long.parquet"))
    rep = {"calendar": [str(cal[0].date()), str(cal[-1].date()), len(cal)], "book_real": int((~meta.placebo).sum()),
           "book_placebo": int(meta.placebo.sum()), "pair_real": int((~pk.placebo).sum()),
           "pair_placebo": int(pk.placebo.sum()), "by_tf": meta[~meta.placebo].tf.value_counts().to_dict(),
           "by_engine": meta[~meta.placebo].engine.value_counts().to_dict(), "pair_symbols": int(pk.symbol.nunique())}
    json.dump(rep, open(os.path.join(a.out, "book_meta.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


# ================================================================================================ fx
def roll_corr(x: pd.Series, y: pd.Series, n: int):
    return x.rolling(n, min_periods=int(n * 0.8)).corr(y)


def cmd_fx(a):
    import trend14 as T
    os.makedirs(a.out, exist_ok=True)
    P = T.load_panels(a.inp)
    fx = [s for s in fx_symbols() if s in P]
    cal = pd.DatetimeIndex(sorted(set().union(*[set(fold(P[s].tday)) for s in fx])))
    cal = cal[cal <= DEV_END]
    C, RD = {}, {}
    for s, d in P.items():
        idx = fold(d.tday)
        c = pd.Series(d.c_mid.to_numpy(float), index=idx)
        c = c[~c.index.duplicated(keep="last")]
        C[s] = c.reindex(cal).ffill(limit=3)
        RD[s] = pd.Series(d.rd.to_numpy(float), index=idx)[lambda x: ~x.index.duplicated(keep="last")].reindex(cal).ffill(limit=3)
    C = pd.DataFrame(C)
    R = np.log(C).diff()
    per = {}
    rv21 = np.sqrt(252 * (R ** 2).rolling(21, min_periods=18).mean())
    per["RV21"] = rv21
    per["RV_CHG"] = np.log(rv21 / rv21.shift(63))
    per["ER63"] = np.log(C).diff(63).abs() / R.abs().rolling(63, min_periods=55).sum()
    r5 = np.log(C).diff(5)
    per["VR5"] = r5.rolling(250, min_periods=200).var() / (5 * R.rolling(250, min_periods=200).var())
    per["AC_D"] = pd.DataFrame({s: roll_corr(R[s], R[s].shift(1), 63) for s in C})
    per["AC_W"] = pd.DataFrame({s: roll_corr(r5[s], r5[s].shift(5), 260) for s in C})
    per["CARRY"] = pd.DataFrame(RD)
    per["TS12"] = np.log(C).diff(252)
    per["MA200Z"] = (C - C.rolling(200, min_periods=180).mean()) / (C * rv21 / math.sqrt(252))
    for k, v in per.items():
        v.index.name = "day"
        v.to_parquet(os.path.join(a.out, f"pairfeat_{k}.parquet"))
    # market-wide PRICE features (FX book)
    M = pd.DataFrame(index=cal)
    med = lambda X: X[fx].median(axis=1)                                     # noqa: E731
    M["RV5"] = med(np.sqrt(252 * (R ** 2).rolling(5, min_periods=4).mean()))
    M["RV21"] = med(rv21)
    M["RV_CHG"] = np.log(M.RV21 / M.RV21.shift(63))
    M["RV252"] = med(np.sqrt(252 * (R ** 2).rolling(252, min_periods=200).mean()))
    for n in (21, 63):
        M[f"ER{n}"] = med(np.log(C).diff(n).abs() / R.abs().rolling(n, min_periods=int(n * 0.85)).sum())
    M["VR5"] = med(per["VR5"])
    M["AC_D"] = med(per["AC_D"])
    M["AC_W"] = med(per["AC_W"])
    for n in (5, 21):
        M[f"DISP{n}"] = np.log(C[fx]).diff(n).std(axis=1) / (M.RV21 * math.sqrt(n / 252))
    U = pd.DataFrame({s[:3] if sg == 1 else s[3:]: sg * R[s] for s, sg in USD_PAIRS.items() if s in R})
    cols = list(U.columns)
    for n in (21, 63):
        cs = [roll_corr(U[cols[i]], U[cols[j]], n) for i in range(len(cols)) for j in range(i + 1, len(cols))]
        M[f"CORR{n}"] = pd.concat(cs, axis=1).mean(axis=1)
    M["CARRY_SD"] = per["CARRY"][fx].std(axis=1)
    for n in (5, 63):
        M[f"USD{n}"] = -U.rolling(n, min_periods=int(n * 0.8)).sum().mean(axis=1)
    for s in ("EURUSD", "USDJPY"):
        M[f"RV21_{s}"] = rv21[s]
    M.index.name = "day"
    M.to_parquet(os.path.join(a.out, "market_price_fx.parquet"))
    print(M.describe().T.round(4).to_string(), flush=True)
    # factors F1-F6, F8, F9 daily (FX book)
    fac = {}
    for s in fx:
        d = P[s].reset_index(drop=True)
        c, h, lo = d.close, d.high, d.low
        vol = T.ewvol_ann(c)
        size = 0.10 / vol
        buf = (0.25 * size).shift(1).to_numpy()
        res = {"F1": T.run_cont(d, (np.sign(d.rd) * size).shift(1).to_numpy(), buf),
               "F2": T.run_cont(d, T.monthly_pos(d, np.sign(c / c.shift(252) - 1), vol, tv=0.10)),
               "F3": T.run_cont(d, T.monthly_pos(d, np.sign(c / c.shift(63) - 1), vol, tv=0.10))}
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
        idx = fold(d.tday)
        for k, (pnl, gp, sw) in res.items():
            x = pd.Series(pnl, index=idx).groupby(level=0).sum().reindex(cal).fillna(0.0)
            fac[k] = x if k not in fac else fac[k] + x
    F = pd.DataFrame(fac)
    F.index.name = "day"
    F.to_parquet(os.path.join(a.out, "factors_fx_daily.parquet"))
    print("factors", F.columns.tolist(), len(F), flush=True)


# ================================================================================================ cme (daily)
def cmd_cme(a):
    import databento_lab as DL
    import round9_lab as R9
    R9.A = argparse.Namespace(dbn=a.dbn, compact=a.compact, out=a.out)
    os.makedirs(a.out, exist_ok=True)
    rows, meta = [], {}
    for ccy in R9.PARENTS:
        t0 = time.time()
        L = R9.load_opts(ccy)
        if L is None:
            meta[ccy] = "data missing"
            continue
        S = L[0]
        S = S[S.date <= DEV_END]
        n_ok = 0
        for d, day in S.groupby("date", sort=True):
            d = pd.Timestamp(d)
            exps = [e for e in day.exp.unique() if 10 <= (pd.Timestamp(e) - d).days <= 60]
            rec = {"date": d, "ccy": ccy, "iv": np.nan, "rr25": np.nan}
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
                        if abs(x.K.loc[j] / F - 1) < 0.01:
                            iv = DL.implied(F, float(x.K.loc[j]), Tm, float(x.P.loc[j]), cp)
                            if np.isfinite(iv):
                                ivs.append(iv)
                rec.update(iv=float(np.mean(ivs)) if ivs else np.nan, rr25=R9.rr25(dd, F, Tm))
                n_ok += int(np.isfinite(rec["iv"]))
            rows.append(rec)
        meta[ccy] = {"days_with_iv": n_ok, "s": round(time.time() - t0, 1)}
        print(ccy, meta[ccy], flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(a.out, "cme_daily.parquet"))
    json.dump(meta, open(os.path.join(a.out, "cme_daily_meta.json"), "w"), indent=1)


# ================================================================================================ merge
def asof_daily(s: pd.Series, cal: pd.DatetimeIndex, lag_days=0, tol=7):
    s = s.dropna().sort_index()
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=lag_days)
    return s.reindex(cal, method="ffill", tolerance=pd.Timedelta(days=tol))


def cmd_merge(a):
    os.makedirs(a.out, exist_ok=True)
    g = lambda pat: glob.glob(os.path.join(a.inp, "**", pat), recursive=True)                # noqa: E731
    M = pd.read_parquet(g("market_price_fx.parquet")[0])
    cal = pd.DatetimeIndex(M.index)
    W = pd.read_parquet(glob.glob(os.path.join(a.web, "**", "web_daily.parquet"), recursive=True)[0])
    W.index = pd.DatetimeIndex(W.index)
    W = W[W.index <= DEV_END]
    X = M.copy()
    for k, n in (("GSPC", "SPX"), ("GLD", "GOLD")):
        px = W[k].dropna()
        lr = np.log(px).diff()
        if n == "SPX":
            X["SPX5"] = asof_daily(np.log(px).diff(5), cal)
            X["SPX63"] = asof_daily(np.log(px).diff(63), cal)
            X["SPX_RV21"] = asof_daily(np.sqrt(252 * (lr ** 2).rolling(21, min_periods=18).mean()), cal)
        else:
            X["GOLD63"] = asof_daily(np.log(px).diff(63), cal)
    vix = asof_daily(W["VIX"], cal)
    X["VIX"] = np.log(vix)
    X["VIX_TS"] = vix / asof_daily(W["VIX3M"], cal)
    X["VVIX"] = np.log(asof_daily(W["VVIX"], cal))
    X["SKEW"] = asof_daily(W["SKEW"], cal)
    X["GVZ"] = np.log(asof_daily(W["GVZ"], cal))
    X["OVX"] = np.log(asof_daily(W["OVX"], cal))
    X["VRP_SPX"] = vix / 100 - X["SPX_RV21"]
    lv = np.log(W["VIX"].dropna())
    X["DVIX5"] = asof_daily(lv.diff(5), cal)
    X["DVIX21"] = asof_daily(lv.diff(21), cal)
    d2, d10 = asof_daily(W["DGS2"], cal), asof_daily(W["DGS10"], cal)
    X["US2Y_CHG63"] = asof_daily(W["DGS2"].dropna().diff(63), cal)
    X["SLOPE"] = d10 - d2
    X["NFCI"] = asof_daily(W["NFCI"], cal, lag_days=7, tol=14)
    X["EPU"] = asof_daily(W["EPU"].dropna().rolling(30, min_periods=20).mean(), cal, lag_days=1)
    cm = g("cme_daily.parquet")
    if cm:
        Cm = pd.read_parquet(cm[0])
        for ccy, gg in Cm.groupby("ccy"):
            gg = gg.set_index(pd.DatetimeIndex(gg.date)).sort_index()
            X[f"{ccy}_IV"] = asof_daily(gg.iv, cal, tol=5)
            X[f"{ccy}_RR25"] = asof_daily(gg.rr25, cal, tol=5)
        X["FX_VRP"] = pd.concat([X.EUR_IV - X.RV21_EURUSD, X.JPY_IV - X.RV21_USDJPY], axis=1).mean(axis=1, skipna=False)
    lf = g("intraday_feats_long.parquet")
    if lf:
        L = pd.read_parquet(lf[0])
        L["day"] = pd.DatetimeIndex(L["day"])
        fxs = set(fx_symbols())
        L = L[L.symbol.isin(fxs)]
        for k in ("RV_5M", "JUMP", "AC_5M", "ASIA_SHARE"):
            wide = L.pivot_table(index="day", columns="symbol", values=k).reindex(cal)
            m21 = wide.rolling(21, min_periods=15).mean()
            X[k] = m21.median(axis=1)
            if k in ("RV_5M", "AC_5M"):                                 # per-instrument versions (PAIR tier)
                allw = pd.read_parquet(lf[0]).assign(day=lambda x: pd.DatetimeIndex(x["day"]))
                pw = allw.pivot_table(index="day", columns="symbol", values=k).reindex(cal).rolling(21, min_periods=15).mean()
                pw.index.name = "day"
                pw.to_parquet(os.path.join(a.out, f"pairfeat_{k}.parquet"))
    # style trailing returns (TREND12 / REV12 / CARRY6) from the daily factor P&L scaled by its trailing RMS
    F = pd.read_parquet(g("factors_fx_daily.parquet")[0])
    F7 = pd.read_parquet(g("f7_daily.parquet")[0])
    F7.index = pd.DatetimeIndex(F7.index)
    F["F7"] = F7["F7"].reindex(F.index).fillna(0.0)
    F = F[[f"F{i}" for i in range(1, 10)]]
    rms = np.sqrt((F ** 2).rolling(252, min_periods=60).mean().shift(1))
    zf = (F / rms).clip(-5, 5)
    for col, f, n in (("TREND12", "F2", 252), ("REV12", "F5", 252), ("CARRY6", "F1", 126)):
        X[col] = zf[f].rolling(n, min_periods=int(n * 0.8)).mean().reindex(cal)
    X.index.name = "day"
    X.to_parquet(os.path.join(a.out, "features_daily.parquet"))
    F.to_parquet(os.path.join(a.out, "factors_daily.parquet"))
    for f in g("pairfeat_*.parquet"):
        b = os.path.basename(f)
        if not os.path.exists(os.path.join(a.out, b)):
            import shutil
            shutil.copy(f, os.path.join(a.out, b))
    cov = {}
    for c in X.columns:
        v = X.loc[X.index >= "2007-06-01", c]
        cov[c] = {"first": str(v.first_valid_index().date()) if v.notna().any() else None,
                  "last": str(v.last_valid_index().date()) if v.notna().any() else None,
                  "missing_share": round(float(v.isna().mean()), 4)}
    json.dump(cov, open(os.path.join(a.out, "coverage17.json"), "w"), indent=1)
    lines = ["| Feature | First day | Last day | Missing share since 2007-06 |", "|---|---|---|---|"]
    lines += [f"| {c} | {v['first']} | {v['last']} | {v['missing_share']} |" for c, v in cov.items()]
    open(os.path.join(a.out, "coverage17.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["book", "fx", "cme", "merge"])
    ap.add_argument("--inp", default="")
    ap.add_argument("--web", default="")
    ap.add_argument("--dbn", default="dbn")
    ap.add_argument("--compact", default="compact")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"book": cmd_book, "fx": cmd_fx, "cme": cmd_cme, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
