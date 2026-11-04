"""Tests on the Databento CME data (docs/databento_preregistration.md). 2010-06 to 2025-12.
--f1  V1a / V1b rerun with the real daily futures volume (DBN files F1_volume_*.dbn.zst).
--k1  the dollar before FOMC rate changes (Karnaukh) with the fed funds contracts (K1_zq_ZQ_FUT.dbn.zst) and the
      FOMC dates scraped from the Federal Reserve's calendar pages (--fomc CSV).
--o1  selling rich euro / yen volatility (compact definitions and settlements from the batch job).
Usage: python scripts/databento_lab.py --dbn DIR [--compact DIR] [--fomc CSV] --out OUT --f1 --k1 --o1"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = pd.Timestamp("2025-12-31")
LOG = {}
PRODUCT = {"6E": "EUR", "6J": "JPY", "6B": "GBP", "6A": "AUD", "6C": "CAD", "6S": "CHF", "6N": "NZD", "6M": "MXN",
           "GC": "GOLD", "SI": "SILVER"}
MONTH = dict(zip("FGHJKMNQUVXZ", range(1, 13)))


def read_dbn(path):
    import databento as db
    df = db.DBNStore.from_file(path).to_df()
    return df.reset_index()


def outright(sym):
    return isinstance(sym, str) and "-" not in sym and ":" not in sym and " " not in sym


# ------------------------------------------------------------------------------------------- F1
def daily_volume_and_return(df):
    """Per CME trade date: total outright volume, and the return of the most traded contract (close / its own
    previous close, so no roll jump)."""
    df = df[df["symbol"].map(outright)].copy()
    df["date"] = pd.DatetimeIndex(df["ts_event"]).tz_convert("America/Chicago").tz_localize(None).normalize()
    vol = df.groupby("date")["volume"].sum()
    df = df.sort_values(["symbol", "date"])
    df["ret"] = np.log(df["close"]).groupby(df["symbol"]).diff()
    top = df.loc[df.groupby("date")["volume"].idxmax()].set_index("date")
    return pd.DataFrame({"Volume": vol, "r": top["ret"]}).sort_index()


def f1(a):
    """Two passes (as in flow_volume_lab): record every trade, price each market's trades with its book loaded once,
    then replay; V1b alternates markets day by day, which otherwise reloads the minute books thousands of times."""
    import flow_volume_lab as V
    V.PASS["mode"] = "record"
    V.PASS["want"].clear()
    f1_body(a)
    print("F1 trades to price:", len(V.PASS["want"]), flush=True)
    V.price_all()
    V.PASS["mode"] = "replay"
    return f1_body(a)


def f1_body(a):
    import flow_volume_lab as V
    V.ROWS.clear()
    Z = {}
    for p, name in PRODUCT.items():
        f = glob.glob(os.path.join(a.dbn, f"F1_volume_{p}_FUT.dbn.zst"))
        if not f:
            continue
        d = daily_volume_and_return(read_dbn(f[0]))
        d = d[d.index <= DEV_END]
        LOG.setdefault("f1_series", {})[name] = [str(d.index.min().date()), str(d.index.max().date()), int(len(d)),
                                                 float(d.Volume.median())]
        v = np.log(d.Volume.where(d.Volume > 0))
        z = (v - v.rolling(60, min_periods=40).mean().shift(1)) / v.rolling(60, min_periods=40).std().shift(1)
        Z[name] = pd.DataFrame({"z": z, "r": d.r})
        LOG.setdefault("f1_events", {})[name] = {"z_ge_2": int((z >= 2).sum()), "r_notna": int(d.r.notna().sum()),
                                                 "both": int(((z >= 2) & d.r.notna()).sum()),
                                                 "z_max": float(z.max()) if z.notna().any() else None,
                                                 "z_notna": int(z.notna().sum())}
        sym, sg = V.FUT_MAP[name]
        for n in (1, 5):
            rows = []
            for dd, x in Z[name][(Z[name].z >= 2) & Z[name].r.notna()].iterrows():
                side = int(np.sign(x.r) * sg)
                if side == 0:
                    continue
                rows.append(V.ev_row(dd, V.daily_trade(sym, dd, side, n), V.daily_trade(sym, dd, -side, n)))
            V.evaluate(f"F1 V1a {name} real volume z >= 2, {n} day(s)", V.frame([r for r in rows if r]))
        print("F1", name, flush=True)
    cc = [k for k in Z if k in V.CCY_PAIR]
    if len(cc) >= 5:
        P = pd.concat({k: Z[k].z for k in cc}, axis=1).dropna(thresh=5)
        R = pd.concat({k: Z[k].r for k in cc}, axis=1)
        rows = []
        for dd, zr in P.iterrows():
            fo, fa = [], []
            for k in zr.dropna().nsmallest(3).index:
                rr = R.at[dd, k]
                if not np.isfinite(rr) or rr == 0:
                    continue
                sym, sg = V.CCY_PAIR[k]
                side = int(np.sign(rr) * sg)
                x, y = V.daily_trade(sym, dd, side, 1), V.daily_trade(sym, dd, -side, 1)
                if x and y:
                    fo.append(x)
                    fa.append(y)
            if fo:
                rows.append({"date": dd, "follow_net": np.mean([q[0] for q in fo]), "follow_gross": np.mean([q[1] for q in fo]),
                             "fade_net": np.mean([q[0] for q in fa])})
        V.evaluate("F1 V1b low-volume reversal, 3 lowest real-volume currency futures, 1 day", V.frame(rows))
    R = pd.DataFrame(V.ROWS)
    if "check_t" in R:
        from scipy import stats as st
        m = R.check_t.notna()
        p = pd.Series(np.nan, index=R.index)
        p[m] = st.norm.sf(R.check_t[m])
        k = int(m.sum())
        if k:
            idx = p[m].sort_values().index
            q = p[idx].values * k / np.arange(1, k + 1)
            R.loc[idx, "bh_q"] = np.minimum.accumulate(q[::-1])[::-1]
        R["counts"] = m & (R.check_net_bp > 0) & (R.check_t >= 1.645) & (R.get("bh_q", 1.0) <= 0.10)
    R.to_csv(os.path.join(a.out, "f1.csv"), index=False)
    return R.to_dict("records")


# ------------------------------------------------------------------------------------------- K1
def zq_table(path):
    """Daily settlement-like close of every fed funds contract, keyed by contract month."""
    df = read_dbn(path)
    df = df[df["symbol"].map(lambda s: isinstance(s, str) and re.fullmatch(r"ZQ[FGHJKMNQUVXZ]\d{1,2}", s) is not None)]
    df["date"] = pd.DatetimeIndex(df["ts_event"]).tz_convert("America/Chicago").tz_localize(None).normalize()
    rows = []
    for r in df.itertuples():
        m = MONTH[r.symbol[2]]
        dig = int(r.symbol[3:])
        y = r.date.year - (r.date.year % (10 if dig < 10 else 100)) + dig
        if y < r.date.year - 1:
            y += 10 if dig < 10 else 100
        rows.append((r.date, pd.Period(f"{y}-{m:02d}", "M"), r.close))
    T = pd.DataFrame(rows, columns=["date", "contract", "close"]).drop_duplicates(["date", "contract"], keep="last")
    return T.pivot(index="date", columns="contract", values="close").sort_index()


def fomc_dates(a):
    if not a.fomc or not os.path.exists(a.fomc):
        return None
    d = pd.read_csv(a.fomc, parse_dates=["date"])
    d = d[(d.date >= "2010-06-01") & (d.date <= DEV_END)].drop_duplicates("date").sort_values("date")
    LOG["fomc_meetings"] = [int(len(d)), str(d.date.min().date()), str(d.date.max().date())]
    return list(d.date)


def k1(a):
    import flows_lab as FL
    import edges_lab2 as E
    import untested_intraday_lab as U
    f = glob.glob(os.path.join(a.dbn, "K1_zq_ZQ_FUT.dbn.zst"))
    meetings = fomc_dates(a)
    if not f or not meetings:
        return {"status": "data missing", "zq": bool(f), "fomc": bool(meetings)}
    Z = zq_table(f[0])
    LOG["zq"] = [str(Z.index.min().date()), str(Z.index.max().date()), int(Z.shape[1])]
    legs = {"EURUSD": -1, "USDJPY": 1, "GBPUSD": -1, "USDCHF": 1}               # +1 = long the pair when long USD
    sig = []
    Zf = Z.ffill()                                       # an expired contract keeps its last (final) price
    for m in meetings:
        m = pd.Timestamp(m)
        sd = (m - pd.offsets.BDay(3)).normalize()
        row = Zf[Zf.index <= sd]
        if not len(row):
            continue
        row = row.iloc[-1]
        mm = pd.Period(m, "M")
        nxt, cur, prev = row.get(mm + 1, np.nan), row.get(mm, np.nan), row.get(mm - 1, np.nan)
        r1 = 100 - nxt                                       # expected rate after the meeting (next-month contract)
        D = mm.days_in_month
        d1 = m.day                                           # days at the old rate (the change applies the day after)
        if m.day <= 10:
            r0 = 100 - prev                                  # previous month: no meeting in it (meetings ~6 weeks apart)
        else:
            r0 = (D * (100 - cur) - (D - d1) * r1) / d1      # FedWatch: the meeting month's average = mix of r0 and r1
        if not (np.isfinite(r1) and np.isfinite(r0)):
            continue
        spread = r1 - r0
        sig.append({"meeting": pd.Timestamp(m), "expected_bp": spread * 100,
                    "side": 1 if spread * 100 >= 12.5 else (-1 if spread * 100 <= -12.5 else 0)})
    S = pd.DataFrame(sig)
    if not len(S):
        return {"status": "no signals"}
    books = {p: E.book(p) for p in legs}                                     # four FX books (memory permitting)
    rows = []
    for r in S.itertuples():
        m = r.meeting
        if m.year <= 2010:
            hh, mi = 14, 0
        elif m.year <= 2012:
            hh, mi = 12, 15
        else:
            hh, mi = 13, 45
        t0 = FL.local_ts(m - pd.offsets.BDay(2), "America/New_York", 14, 0)
        t1 = FL.local_ts(m, "America/New_York", hh, mi)
        side = r.side if r.side != 0 else 1
        nets = []
        for p, s in legs.items():
            bk = books[p]
            x = bk.trade(t0, t1, int(s * side), tol_min=10) if bk is not None else None
            if x is not None:
                nets.append(U.net_of(x))
        if nets:
            rows.append({"meeting": m, "expected_bp": r.expected_bp, "side": r.side,
                         "net_long_usd_if_no_signal": float(np.mean(nets)) if r.side == 0 else np.nan,
                         "net": float(np.mean(nets)) if r.side != 0 else np.nan})
    R = pd.DataFrame(rows).set_index("meeting")
    R.to_csv(os.path.join(a.out, "k1_meetings.csv"))
    tr = R.net.dropna()

    def summ(x):
        x = pd.Series(x).dropna()
        if len(x) < 3:
            return {"n": int(len(x))}
        return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": float(x.mean() / x.std() * math.sqrt(len(x))),
                "win": float((x > 0).mean())}
    res = {"meetings": int(len(S)), "signals": int((S.side != 0).sum()), "trades": summ(tr),
           "post_publication_2016_2025": summ(tr[tr.index >= "2016-01-01"]),
           "no_signal_meetings_long_usd (information)": summ(R.net_long_usd_if_no_signal)}
    t = res["trades"]
    res["counts_own_rule"] = bool(t.get("n", 0) > 2 and t.get("mean_bp", -1) > 0 and (t.get("t") or 0) >= 1.645)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbn", default="dbn")
    ap.add_argument("--compact", default="compact")
    ap.add_argument("--fomc", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--f1", action="store_true")
    ap.add_argument("--k1", action="store_true")
    ap.add_argument("--o1", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for flag, nm in ((a.f1, "F1"), (a.k1, "K1"), (a.o1, "O1")):
        if not flag:
            continue
        try:
            res[nm] = {"F1": f1, "K1": k1, "O1": o1}[nm](a)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[nm] = {"status": f"error {ex!r}"[:300]}
        print(nm, json.dumps(res[nm], default=str)[:4000], flush=True)
    json.dump(res, open(os.path.join(a.out, "databento_lab.json"), "w"), indent=1, default=str)
    json.dump(LOG, open(os.path.join(a.out, "log_databento.json"), "w"), indent=1, default=str)


def bs76(F, K, T, vol, cp):
    from scipy.stats import norm
    if T <= 0 or vol <= 0:
        return max(F - K, 0.0) if cp == "C" else max(K - F, 0.0), (1.0 if (cp == "C" and F > K) else (-1.0 if cp == "P" and F < K else 0.0))
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if cp == "C":
        return F * norm.cdf(d1) - K * norm.cdf(d2), norm.cdf(d1)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1), norm.cdf(d1) - 1.0


def implied(F, K, T, price, cp):
    from scipy.optimize import brentq
    intrinsic = max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    if not (price > intrinsic + 1e-12) or T <= 0:
        return np.nan
    try:
        return brentq(lambda v: bs76(F, K, T, v, cp)[0] - price, 1e-4, 3.0, maxiter=200)
    except ValueError:
        return np.nan


def futures_closes(path):
    """Daily close of every outright contract (CME trade date) -> DataFrame[date x raw symbol]; and the most traded
    contract's daily return (for the volatility forecast)."""
    df = read_dbn(path)
    df = df[df["symbol"].map(outright)].copy()
    df["date"] = pd.DatetimeIndex(df["ts_event"]).tz_convert("America/Chicago").tz_localize(None).normalize()
    C = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    df = df.sort_values(["symbol", "date"])
    df["ret"] = np.log(df["close"]).groupby(df["symbol"]).diff()
    top = df.loc[df.groupby("date")["volume"].idxmax()].set_index("date")["ret"]
    return C, top.sort_index()


def har_month(r, fit_end="2017-12-31"):
    """HAR in levels on daily squared returns: next 21 days' mean r^2 on the last 1 / 5 / 22 days' means; fitted on
    windows ending by fit_end, then fixed. Returns the annualised volatility forecast by date."""
    q = r.dropna() ** 2
    X = pd.DataFrame({"d": q, "w": q.rolling(5).mean(), "m": q.rolling(22).mean()})
    y = q[::-1].rolling(21).mean()[::-1].shift(-1)
    D = pd.concat([X, y.rename("y")], axis=1).dropna()
    fit = D[D.index <= pd.Timestamp(fit_end) - pd.Timedelta(days=35)]
    A = np.c_[np.ones(len(fit)), fit[["d", "w", "m"]].values]
    beta = np.linalg.lstsq(A, fit.y.values, rcond=None)[0]
    Xa = X.dropna()
    pred = np.c_[np.ones(len(Xa)), Xa.values] @ beta
    LOG.setdefault("o1_har", []).append({"beta": beta.tolist(), "fit_n": int(len(fit))})
    return pd.Series(np.sqrt(np.clip(pred, 1e-12, None) * 252), index=Xa.index)


def o1(a):
    import glob as G
    res = {}
    TICK = {"EUR": 5e-5, "JPY": 5e-7}
    PARENTS = {"EUR": ("6E", ("6E_OPT", "EUU_OPT")), "JPY": ("6J", ("6J_OPT", "JPU_OPT"))}
    cycles_all = {}
    for ccy, (fut, opts) in PARENTS.items():
        fpath = G.glob(os.path.join(a.dbn, f"F1_volume_{fut}_FUT.dbn.zst"))
        if not fpath:
            res[ccy] = {"status": "futures bars missing"}
            continue
        C, ret = futures_closes(fpath[0])
        sig = har_month(ret)
        defs, stats = [], []
        for o in opts:
            fd = os.path.join(a.compact, f"O1fx_defs_{o}.parquet")
            fs = os.path.join(a.compact, f"O1fx_stats_{o}.parquet")
            if os.path.exists(fd) and os.path.exists(fs):
                defs.append(pd.read_parquet(fd))
                stats.append(pd.read_parquet(fs))
        if not defs:
            res[ccy] = {"status": "option files missing"}
            continue
        D = pd.concat(defs).drop_duplicates("instrument_id", keep="last")
        D = D[D["instrument_class"].astype(str).str.upper().isin(["C", "P", "CALL", "PUT"])].copy()
        D["cp"] = D["instrument_class"].astype(str).str.upper().str[0]
        D["exp"] = pd.DatetimeIndex(pd.to_datetime(D["expiration"], utc=True)).tz_convert("America/Chicago").tz_localize(None).normalize()
        S = pd.concat(stats)
        st_ = S["stat_type"].astype(str).str.upper()
        S = S[st_.str.contains("3$|SETTLEMENT", regex=True)].copy()
        ref = S["ts_ref"] if "ts_ref" in S else S["ts_event"]
        S["date"] = pd.DatetimeIndex(pd.to_datetime(ref, utc=True)).tz_convert("America/Chicago").tz_localize(None).normalize()
        S = S.sort_values("ts_event").drop_duplicates(["instrument_id", "date"], keep="last")
        S = S.merge(D[["instrument_id", "strike_price", "cp", "exp", "underlying"]], on="instrument_id", how="inner")
        S = S[(S.date <= S.exp) & (S.date >= S.exp - pd.Timedelta(days=500)) & (S.price > 0)]
        # strike units vs the future's price: rescale by a power of ten if needed (checked on the data)
        under_ok = S["underlying"].isin(C.columns)
        S = S[under_ok]
        fprice = np.array([C.at[d, u] if d in C.index else np.nan for d, u in zip(S.date, S.underlying)])
        ratio = np.nanmedian(S.strike_price.values / fprice)
        scale = 10.0 ** (-round(math.log10(ratio))) if np.isfinite(ratio) and ratio > 0 else 1.0
        S["K"] = S.strike_price * scale
        near = (S.K / fprice - 1).abs() < 0.01
        pr = float(np.nanmedian(S.price.values[near.values] / fprice[near.values])) if near.any() else np.nan
        pscale = 1.0 if (np.isfinite(pr) and 1e-4 <= pr <= 1e-1) else (10.0 ** (-round(math.log10(pr / 0.01))) if np.isfinite(pr) and pr > 0 else 1.0)
        LOG.setdefault("o1_units_premium", {})[ccy] = {"atm_premium_over_future_median": pr, "scale": pscale}
        S["P"] = S.price * pscale
        S["F"] = fprice
        LOG.setdefault("o1_units", {})[ccy] = {"strike_over_future_median": float(ratio), "scale": scale,
                                              "settlement_rows": int(len(S))}
        S = S.dropna(subset=["F"])
        exps = sorted(e for e in S.exp.unique() if pd.Timestamp("2010-07-01") <= e <= DEV_END)
        rows = []
        for k in range(1, len(exps)):
            E, t0 = pd.Timestamp(exps[k]), pd.Timestamp(exps[k - 1])
            days_left = (E - t0).days
            if days_left < 10 or days_left > 45:
                continue
            day0 = S[(S.date == t0) & (S.exp == E)]
            if day0.empty:
                later = S[(S.date > t0) & (S.date <= t0 + pd.Timedelta(days=4)) & (S.exp == E)]
                if later.empty:
                    continue
                t0 = later.date.min()
                day0 = later[later.date == t0]
            u = day0.underlying.value_counts().index[0]
            day0 = day0[day0.underlying == u]
            F0 = float(day0.F.iloc[0])
            Ks = sorted(set(day0[day0.cp == "C"].K) & set(day0[day0.cp == "P"].K))
            if not Ks:
                continue
            K = min(Ks, key=lambda x: abs(x - F0))
            c0 = float(day0[(day0.cp == "C") & (day0.K == K)].P.iloc[0])
            p0 = float(day0[(day0.cp == "P") & (day0.K == K)].P.iloc[0])
            T0 = days_left / 365.0
            ivc, ivp = implied(F0, K, T0, c0, "C"), implied(F0, K, T0, p0, "P")
            iv = np.nanmean([ivc, ivp])
            if not np.isfinite(iv):
                continue
            fc = sig.asof(t0) if len(sig) else np.nan
            # daily delta hedge with the underlying future at its close, entry implied vol, to expiry
            path = C[u][(C.index >= t0) & (C.index <= E)].dropna()
            if len(path) < 5:
                continue
            h_prev, hedge_pnl, turn = 0.0, 0.0, 0.0
            for i in range(len(path) - 1):
                Ft, dt = float(path.iloc[i]), path.index[i]
                T = max((E - dt).days, 0) / 365.0
                dc = bs76(Ft, K, T, iv, "C")[1]
                dp = bs76(Ft, K, T, iv, "P")[1]
                h = dc + dp                                  # the short straddle's delta is -(dc + dp): hold +h futures
                turn += abs(h - h_prev)
                hedge_pnl += h * (float(path.iloc[i + 1]) - Ft)
                h_prev = h
            turn += abs(h_prev)
            FE = float(path.iloc[-1])
            payoff = max(FE - K, 0.0) + max(K - FE, 0.0)
            prem = c0 + p0
            cost = 0.04 * prem + 0.5 * TICK[ccy] * turn
            rows.append({"ccy": ccy, "entry": t0, "expiry": E, "F0": F0, "K": K, "iv": iv, "forecast": fc,
                         "hedged": (prem - payoff + hedge_pnl - cost) / F0,
                         "unhedged": (prem - payoff - 0.04 * prem) / F0, "premium_pct": prem / F0})
        Rc = pd.DataFrame(rows)
        cycles_all[ccy] = Rc
        LOG.setdefault("o1_cycles", {})[ccy] = int(len(Rc))
    if not cycles_all:
        return {"status": "data missing", **res}
    R = pd.concat(cycles_all.values(), ignore_index=True)
    R.to_csv(os.path.join(a.out, "o1_cycles.csv"), index=False)
    R["month"] = pd.DatetimeIndex(R.expiry).to_period("M")
    R["rich"] = (R.iv - R.forecast) > 0.01

    def pool(df, col):
        x = df.groupby("month")[col].mean()
        x.index = x.index.to_timestamp()
        return x

    def summ(x):
        x = pd.Series(x).dropna()
        if len(x) < 3:
            return {"n": int(len(x))}
        return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": float(x.mean() / x.std() * math.sqrt(len(x))),
                "win": float((x > 0).mean()), "worst_bp": float(x.min() * 1e4), "sd_bp": float(x.std() * 1e4)}
    chk = R[R.expiry >= "2018-01-01"]
    out = {"cycles": {k: int(len(v)) for k, v in cycles_all.items()},
           "iv_minus_forecast_vol_points_median": float((R.iv - R.forecast).median() * 100),
           "share_rich": float(R.rich.mean())}
    out["O1a sell when rich, hedged, 2018-2025 (count)"] = summ(pool(chk[chk.rich], "hedged"))
    out["O1a 2010-2017 (reference)"] = summ(pool(R[(R.expiry < "2018-01-01") & R.rich], "hedged"))
    out["O1b always sell, hedged, 2018-2025"] = summ(pool(chk, "hedged"))
    out["O1b always sell, hedged, 2010-2017"] = summ(pool(R[R.expiry < "2018-01-01"], "hedged"))
    out["O1c always sell, unhedged, 2018-2025"] = summ(pool(chk, "unhedged"))
    out["per currency 2018-2025 hedged (always)"] = {c: summ(v[v.expiry >= "2018-01-01"].set_index("expiry").hedged)
                                                     for c, v in cycles_all.items()}
    o = out["O1a sell when rich, hedged, 2018-2025 (count)"]
    out["counts_own_rule"] = bool(o.get("n", 0) > 2 and o.get("mean_bp", -1) > 0 and (o.get("t") or 0) >= 1.645)
    return out


if __name__ == "__main__":
    main()
