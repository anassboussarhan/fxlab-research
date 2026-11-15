"""Round 22 parts D, M, B, C2, C3, S, K, V (docs/round22_preregistration.md): daily, weekly and monthly families with
every parameter optimised walk-forward (scripts/wf22.py). Scrambled-Sobol configurations.

  d7      --out DIR              D: dip-buying family on ECB fixings (28 crosses), daily or weekly, carry filter
  macro   --out DIR              M: currency signals; B: bonds across countries (BoE spot, OECD rates via DBnomics)
  commod  --out DIR              C3: commodity ETFs cross-section / seasonality; C2: CFTC managed money
  stocks  --data DIR --out DIR   S: anomalies, pairs, swing setups, deletions (round-20 S&P 500 panel)
  crypto  --funding DIR --out DIR  K: cross-section (Binance spot, downloaded here) and cash-and-carry
  vol     --out DIR              V: S&P variance at the VIX, SVXY, BTC DVOL with optimised filters
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
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402

DEV_END = "2025-12-31"
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}

try:
    from numba import njit
except Exception:                                                          # noqa: BLE001
    def njit(*a, **k):
        def w(f):
            return f
        return w(a[0]) if a and callable(a[0]) else w


def pick(u, opts):
    return opts[min(int(u * len(opts)), len(opts) - 1)]


def rng(u, lo, hi):
    return lo + u * (hi - lo)


def sobol(d, n, seed):
    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def run_wf(rows, ser, part, family, symbol, M, dates, first, extra=None, min_active=30, desc=None, min_valid=50):
    s5, s1, ch = wf22.walk_forward(M, dates, first, min_active=min_active, min_valid=min_valid)
    st, st1 = wf22.stream_stats(s5), wf22.stream_stats(s1)
    r = {"part": part, "family": family, "symbol": symbol, "placebo": False, "n_cfg": int(M.shape[0]), **st,
         "top1_p2_mean": st1["p2_mean"], "top1_p2_t": st1["p2_t"], "own": wf22.counts_rule(st),
         "choices": json.dumps({str(k): ([desc[i] for i in v["cfg"]] if desc else v["cfg"]) for k, v in ch.items()},
                               default=str)[:4000]}
    if extra:
        r.update(extra)
    rows.append(r)
    ser[f"{family}|{symbol}"] = s5
    print(family, symbol, json.dumps({k: r[k] for k in ("oos_first", "p1_mean", "p2_mean", "p2_t", "own")}), flush=True)


def save(out, tag, rows, ser):
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"{tag}_rows.csv"), index=False)
    if ser:
        pd.DataFrame(ser).to_parquet(os.path.join(out, f"{tag}_series.parquet"))


# ============================================================================================ D
@njit(cache=True)
def state_machine(entry, exitsig, tstop, side):
    n = len(entry)
    pos = np.zeros(n)
    cur, held = 0.0, 0
    for t in range(n):
        if cur != 0.0:
            held += 1
            if exitsig[t] * cur > 0 or held >= tstop:
                cur, held = 0.0, 0
        if cur == 0.0 and entry[t] != 0 and (side == 0 or entry[t] > 0):
            cur, held = float(entry[t]), 0
        pos[t] = cur
    return pos


def rsi(c, n):
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()


def cmd_d7(a):
    import forward_d7 as FD
    import r18_long as RL
    X = FD.load(None)
    X = X[X.index <= DEV_END]
    RT, _ = RL.oecd_rates(list(pd.period_range("1999-01", "2025-12", freq="M").astype(str)))
    RT = RT.ffill()
    crosses = RL.FX
    D = {}
    for fq in ("D", "W"):
        Pm = {}
        for s in crosses:
            b, q = s[:3], s[3:]
            px = (X[q] / X[b]).dropna()
            if fq == "W":
                iso = px.index.isocalendar()
                key = (iso.year * 100 + iso.week).to_numpy()
                px = px[np.r_[key[1:] != key[:-1], True]]
            per = px.index.to_period("M")
            diff = (RT[b].reindex(per).to_numpy() - RT[q].reindex(per).to_numpy())
            Pm[s] = (px, diff)
        D[fq] = Pm
    wk_cal = None
    U = sobol(13, 256, 7)
    M, desc = [], []
    for u in U:
        fq = pick(u[0], ["D", "W"])
        et = pick(u[1], ["NLOW", "RSI2", "CONSEC", "PCTB"])
        N = int(rng(u[2], 3, 20))
        thr = rng(u[3], 2, 30)
        kd = int(rng(u[4], 2, 5))
        bbk = rng(u[5], 1, 2.5)
        sma = pick(u[6], [0, 50, 100, 150, 200, 300])
        xt = pick(u[7], ["NHIGH", "RSI2", "SMA5"])
        Mx = int(rng(u[8], 2, 20))
        xthr = rng(u[9], 50, 90)
        T = int(rng(u[10], 2, 26)) * (5 if fq == "D" else 1)
        carry_f = u[11] < 0.5
        side = 0 if u[12] < 0.5 else 1                                      # 0 long only, 1 both sides
        parts = []
        for s, (px, diff) in D[fq].items():
            c = px.to_numpy(float)
            r = np.r_[np.nan, c[1:] / c[:-1] - 1]
            vol = pd.Series(r).rolling(20, min_periods=10).std().shift().to_numpy()
            ser_c = pd.Series(c)
            if et == "NLOW":
                lo_ = c <= ser_c.rolling(N).min().to_numpy()
                hi_ = c >= ser_c.rolling(N).max().to_numpy()
            elif et == "RSI2":
                rr = rsi(c, 2)
                lo_, hi_ = rr < thr, rr > 100 - thr
            elif et == "CONSEC":
                dn = (np.diff(c, prepend=np.nan) < 0).astype(float)
                up = (np.diff(c, prepend=np.nan) > 0).astype(float)
                lo_ = pd.Series(dn).rolling(kd).sum().to_numpy() == kd
                hi_ = pd.Series(up).rolling(kd).sum().to_numpy() == kd
            else:
                m20, s20 = ser_c.rolling(20).mean().to_numpy(), ser_c.rolling(20).std().to_numpy()
                lo_, hi_ = c < m20 - bbk * s20, c > m20 + bbk * s20
            if sma:
                ma = ser_c.rolling(sma).mean().to_numpy()
                lo_ = lo_ & (c > ma)
                hi_ = hi_ & (c < ma)
            if carry_f:
                lo_ = lo_ & (diff > 0)
                hi_ = hi_ & (diff < 0)
            entry = np.where(lo_, 1, np.where(hi_, -1, 0)).astype(np.float64)
            if xt == "NHIGH":
                xl = c >= ser_c.rolling(Mx).max().to_numpy()
                xs = c <= ser_c.rolling(Mx).min().to_numpy()
            elif xt == "RSI2":
                rr = rsi(c, 2)
                xl, xs = rr > xthr, rr < 100 - xthr
            else:
                m5 = ser_c.rolling(5).mean().to_numpy()
                xl, xs = c > m5, c < m5
            exitsig = np.where(xl, 1, 0) + np.where(xs, -1, 0)             # +1 exits longs, -1 exits shorts
            pos = state_machine(np.nan_to_num(entry), np.nan_to_num(exitsig).astype(np.float64), T, side)
            held = np.r_[0.0, pos[:-1]]
            per_yr = 252 if fq == "D" else 52
            carry = held * np.nan_to_num(diff) / per_yr - np.abs(held) * 0.0075 / per_yr
            net = held * np.nan_to_num(r) - np.abs(np.diff(np.r_[0.0, held])) * 1.5e-4 + carry
            z = net / np.where(np.isfinite(vol) & (vol > 0), vol, np.nan) * 0.01   # vol-scaled to 1% per bar
            parts.append(pd.Series(z, index=px.index))
        book = pd.concat(parts, axis=1).mean(axis=1)
        wk = book.groupby(book.index.to_period("W-FRI")).sum()
        wk.index = wk.index.to_timestamp(how="end").normalize()
        M.append(wk)
        desc.append({"fq": fq, "entry": et, "N": N, "thr": round(thr, 1), "k": kd, "bbk": round(bbk, 2), "sma": sma, "exit": xt,
                     "M": Mx, "xthr": round(xthr), "T": T, "carry": carry_f, "side": side})
    W = pd.concat(M, axis=1)
    W = W[(W.index >= "2000-01-01") & (W.index <= DEV_END)]
    rows, ser = [], {}
    run_wf(rows, ser, "D", "D-dip", "ECB28", W.to_numpy().T, W.index, 2004, desc=desc)
    save(a.out, "D", rows, ser)


# ============================================================================================ M and B
def macro_data():
    import r18_long as RL
    import r21_rest as R21
    X = RL.boe_rates()
    Xm = X.resample("ME").last()
    Xm.index = Xm.index.to_period("M")
    P = pd.DataFrame({c: Xm["USD"] / Xm[c] for c in RL.CUR8})
    months = pd.period_range("1975-01", "2025-12", freq="M")
    S, _ = RL.oecd_rates(list(months.astype(str)))
    S.index = pd.PeriodIndex(S.index, freq="M")
    Lg = R21.oecd_series("IRLTLT01", "ST", 0) / 100.0
    CPI = R21.oecd_series("CPALTT01", "GY", 0) / 100.0
    CPIx = R21.oecd_series("CPALTT01", "IXOB", 0)
    S, Lg, CPI, CPIx, P = (z.reindex(months).reindex(columns=list(RL.CUR8)) for z in (S, Lg, CPI, CPIx, P))
    return months, P, S, Lg, CPI, CPIx


def book_returns(Z, R, n, reb, cost):
    """Long the top n, short the bottom n of signal Z (rows months), rebalanced every reb months."""
    out = []
    w_prev = pd.Series(0.0, index=Z.columns)
    w = w_prev.copy()
    for i, m in enumerate(Z.index[:-1]):
        if i % reb == 0:
            z = Z.loc[m].dropna()
            z = z[R.loc[m].reindex(z.index).notna()]
            if len(z) >= 2 * n:
                o = z.sort_values()
                w = pd.Series(0.0, index=Z.columns)
                w[o.index[-n:]] = 1.0 / n
                w[o.index[:n]] = -1.0 / n
        turn = float((w - w_prev).abs().sum())
        w_prev = w
        out.append(float((w * R.loc[m].fillna(0.0)).sum()) - cost * turn)
    return pd.Series(out, index=Z.index[:-1])


def cmd_macro(a):
    months, P, S, Lg, CPI, CPIx = macro_data()
    rows, ser = [], {}
    # M: currencies
    ret = (P.shift(-1) / P - 1) + S.sub(S["USD"], axis=0) / 12.0
    realP = np.log(P * CPIx / CPIx["USD"].to_numpy()[:, None])
    U = sobol(7, 256, 11)
    M, desc = [], []
    for u in U:
        sig = pick(u[0], ["carry", "real_carry", "slope", "mom", "value", "composite"])
        lb = int(rng(u[1], 1, 12))
        sk = int(rng(u[2], 0, 2))
        vl = int(rng(u[3], 36, 84))
        n = int(rng(u[4], 1, 3.999))
        reb = int(rng(u[5], 1, 3.999))
        lag = int(rng(u[6], 1, 2.999))
        Sx, Lx, Cx = S.shift(lag), Lg.shift(lag), CPI.shift(lag + 1)
        mom = np.log(P).shift(sk) - np.log(P).shift(sk + lb)
        val = -(realP - realP.shift(vl))
        comps = {"carry": Sx, "real_carry": Sx - Cx, "slope": -(Lx - Sx), "mom": mom, "value": val}
        if sig == "composite":
            Z = sum(v.rank(axis=1, pct=True) for v in comps.values()) / len(comps)
        else:
            Z = comps[sig]
        M.append(book_returns(Z, ret, n, reb, 3e-4))
        desc.append({"sig": sig, "lb": lb, "skip": sk, "vl": vl, "n": n, "reb": reb, "lag": lag})
    W = pd.concat(M, axis=1)
    W.index = W.index.to_timestamp(how="end").normalize()
    W = W[(W.index >= "1976-01-01") & (W.index <= DEV_END)]
    run_wf(rows, ser, "M", "M-fx-signals", "G8", W.to_numpy().T, W.index, 1985, desc=desc, min_active=24, min_valid=12)
    # B: bonds, local currency, duration 7
    bret = Lg.shift(1) / 12 - 7 * (Lg - Lg.shift(1))
    bret_next = bret.shift(-1)
    U = sobol(5, 256, 13)
    M, desc = [], []
    for u in U:
        sig = pick(u[0], ["mom", "carry", "value", "composite"])
        lb = int(rng(u[1], 1, 12))
        n = int(rng(u[2], 1, 3.999))
        reb = int(rng(u[3], 1, 3.999))
        lag = int(rng(u[4], 1, 2.999))
        mom = bret.rolling(lb).sum().shift(lag - 1)
        comps = {"mom": mom, "carry": (Lg - S).shift(lag), "value": (Lg - CPI.shift(1)).shift(lag)}
        Z = (sum(v.rank(axis=1, pct=True) for v in comps.values()) / 3) if sig == "composite" else comps[sig]
        M.append(book_returns(Z, bret_next, n, reb, 2e-4))
        desc.append({"sig": sig, "lb": lb, "n": n, "reb": reb, "lag": lag})
    W = pd.concat(M, axis=1)
    W.index = W.index.to_timestamp(how="end").normalize()
    W = W[(W.index >= "1976-01-01") & (W.index <= DEV_END)]
    run_wf(rows, ser, "B", "B-bonds", "G8", W.to_numpy().T, W.index, 1985, desc=desc, min_active=24, min_valid=12)
    save(a.out, "MB", rows, ser)


# ============================================================================================ C2, C3
ETFS = ["USO", "BNO", "UNG", "UGA", "GLD", "SLV", "PPLT", "PALL", "CPER", "DBB", "DBA", "CORN", "WEAT", "SOYB", "CANE",
        "JO", "NIB", "BAL", "DBC", "DBE", "DBP", "DBO", "UNL", "USL"]
CFTC = {"067651": "USO", "023651": "UNG", "088691": "GLD", "084691": "SLV", "085692": "CPER", "002602": "CORN",
        "001602": "WEAT", "005602": "SOYB", "080732": "CANE", "083731": "JO", "073732": "NIB", "076651": "PPLT"}


def yahoo_adj(symbols, start="1999-01-01"):
    import yfinance as yf
    out = {}
    for s in symbols:
        for _ in range(3):
            try:
                h = yf.download(s, start=start, end="2026-01-01", progress=False, auto_adjust=True)
                break
            except Exception:                                              # noqa: BLE001
                h = pd.DataFrame()
                time.sleep(3)
        if h is None or h.empty:
            print("no data", s, flush=True)
            continue
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) else pd.to_datetime(h.index)
        out[s] = h[h.index <= DEV_END]
    return out


def cftc_mm():
    import requests
    rows = []
    urls = ["https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip"] + \
        [f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{y}.zip" for y in range(2017, 2026)]
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=180)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        except Exception as ex:                                            # noqa: BLE001
            print("cftc", url[-30:], repr(ex)[:100], flush=True)
            continue
        code = d["CFTC_Contract_Market_Code"].astype(str).str.strip().str.zfill(6)
        keep = code.isin(CFTC)
        d = d[keep]
        dc = [c for c in d.columns if c.startswith("Report_Date_as_YYYY")][0]
        rows.append(pd.DataFrame({"date": pd.to_datetime(d[dc]), "code": code[keep],
                                  "net": d["M_Money_Positions_Long_All"] - d["M_Money_Positions_Short_All"],
                                  "oi": d["Open_Interest_All"]}))
        print("cftc", url[-30:], len(d), flush=True)
    D = pd.concat(rows).drop_duplicates(["date", "code"])
    D["etf"] = D.code.map(CFTC)
    D["x"] = D.net / D.oi
    return D.pivot_table(index="date", columns="etf", values="x").sort_index()


def cmd_commod(a):
    rows, ser = [], {}
    E = yahoo_adj(ETFS)
    C = pd.DataFrame({k: v["Close"] for k, v in E.items()}).sort_index()
    meta = {k: str(v.index.min().date()) for k, v in E.items()}
    # C3: monthly cross-section / seasonality
    Cm = C.resample("ME").last()
    R = Cm.pct_change(fill_method=None)
    Rn = R.shift(-1)
    U = sobol(6, 256, 17)
    M, desc = [], []
    for u in U:
        sig = pick(u[0], ["mom", "rev", "season"])
        lb = int(rng(u[1], 1, 12))
        sk = int(rng(u[2], 0, 2))
        n = int(rng(u[3], 1, 5.999))
        ls = u[4] < 0.5
        yrs = int(rng(u[5], 3, 10))
        if sig == "mom":
            Z = np.log(Cm).shift(sk) - np.log(Cm).shift(sk + lb)
        elif sig == "rev":
            Z = -(np.log(Cm).shift(sk) - np.log(Cm).shift(sk + lb))
        else:
            mo = R.index.month
            Z = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
            for i in range(len(R)):
                nxt_m = (mo[i] % 12) + 1
                past = R.iloc[:i + 1]
                past = past[past.index.month == nxt_m].tail(yrs)
                if len(past) >= 3:
                    Z.iloc[i] = past.mean()
        out = []
        w_prev = pd.Series(0.0, index=C.columns)
        for i, m in enumerate(Z.index[:-1]):
            z = Z.loc[m].dropna()
            z = z[Rn.loc[m].reindex(z.index).notna()]
            if len(z) < max(4, 2 * n):
                out.append(0.0)
                continue
            o = z.sort_values()
            w = pd.Series(0.0, index=C.columns)
            w[o.index[-n:]] = 1.0 / n
            if ls:
                w[o.index[:n]] = -1.0 / n
            turn = float((w - w_prev).abs().sum())
            w_prev = w
            r = float((w * Rn.loc[m].fillna(0)).sum())
            if not ls:
                r -= float(Rn.loc[m, z.index].mean())                       # long-only measured against the equal weight
            out.append(r - 5e-4 * turn)
        M.append(pd.Series(out, index=Z.index[:-1]))
        desc.append({"sig": sig, "lb": lb, "skip": sk, "n": n, "long_short": ls, "years": yrs})
    W = pd.concat(M, axis=1)
    W = W[W.index <= DEV_END]
    run_wf(rows, ser, "C", "C3-commodity-xs", "ETFs", W.to_numpy().T, W.index, 2010, desc=desc, min_active=24, min_valid=12)
    # C2: CFTC managed money, weekly (report Tuesday, public Friday; traded from the next Monday close)
    try:
        X = cftc_mm()
        Cw = C.resample("W-FRI").last()
        Rw = Cw.pct_change(fill_method=None).shift(-1)                      # the week after the Friday release
        Xw = X.copy()
        Xw.index = (Xw.index + pd.Timedelta(days=3)).to_period("W-FRI").to_timestamp(how="end").normalize()
        Xw = Xw[~Xw.index.duplicated(keep="last")].reindex(Rw.index).ffill(limit=2)
        U = sobol(4, 128, 19)
        M, desc = [], []
        for u in U:
            lb = int(rng(u[0], 52, 260))
            ext = rng(u[1], 0.05, 0.30)
            mode = pick(u[2], [-1, 1])
            hold = int(rng(u[3], 1, 12))
            pct = Xw.rolling(lb, min_periods=int(lb * 0.8)).rank(pct=True)
            sig = pd.DataFrame(0.0, index=pct.index, columns=pct.columns)
            sig[pct >= 1 - ext] = 1.0 * mode
            sig[pct <= ext] = -1.0 * mode
            sig = sig.replace(0, np.nan).ffill(limit=hold - 1).fillna(0.0) if hold > 1 else sig
            cols = [c for c in sig.columns if c in Rw.columns]
            pos = sig[cols]
            turn = pos.diff().abs().sum(axis=1).fillna(0)
            r = (pos * Rw[cols]).sum(axis=1) / max(len(cols), 1) - 5e-4 * turn / max(len(cols), 1)
            M.append(r)
            desc.append({"lb": lb, "ext": round(ext, 2), "mode": "trend" if mode > 0 else "contrarian", "hold": hold})
        W = pd.concat(M, axis=1)
        W = W[(W.index >= "2007-01-01") & (W.index <= DEV_END)]
        run_wf(rows, ser, "C", "C2-cftc-managed-money", "ETFs", W.to_numpy().T, W.index, 2011, desc=desc, min_active=30)
    except Exception as ex:                                                # noqa: BLE001
        print("C2 failed", repr(ex)[:200], flush=True)
        rows.append({"part": "C", "family": "C2-cftc-managed-money", "symbol": "ETFs", "note": f"failed: {repr(ex)[:150]}"})
    save(a.out, "C", rows, ser)
    json.dump(meta, open(os.path.join(a.out, "C_etf_starts.json"), "w"), indent=1)


# ============================================================================================ S
def stocks_panel(d):
    O, H, L, C = (pd.read_parquet(os.path.join(d, f"{k}.parquet")) for k in ("open", "high", "low", "close"))
    Mb = pd.read_parquet(os.path.join(d, "members.parquet"))
    idx = C.index[(C.index >= "2003-01-01") & (C.index <= DEV_END)]
    spy = C["SPY"].reindex(idx)
    names = [c for c in Mb.columns if c in C.columns]
    O, H, L, C = (x.reindex(idx)[names] for x in (O, H, L, C))
    M = Mb.reindex(index=idx, method="ffill").fillna(False).astype(bool)[names]
    return O, H, L, C, M, spy, idx, names


@njit(cache=True)
def pair_trade(si, sj, base_i, base_j, sd, ez, xz):
    """Returns list of (entry, exit, side) as arrays over a trading window; side -1: short i / long j."""
    n = len(si)
    ent = np.full(n, -1)
    ext = np.full(n, -1)
    sides = np.zeros(n)
    k = 0
    open_, e0 = 0, -1
    for q in range(n):
        a, b = si[q], sj[q]
        if not (np.isfinite(a) and np.isfinite(b)):
            if open_ != 0:
                ent[k], ext[k], sides[k] = e0, q - 1, open_
                k += 1
                open_ = 0
            break
        s = a / base_i - b / base_j
        if open_ == 0 and abs(s) > ez * sd:
            open_ = -1 if s > 0 else 1
            e0 = q
        elif open_ != 0 and ((open_ == -1 and s <= xz * sd) or (open_ == 1 and s >= -xz * sd)):
            ent[k], ext[k], sides[k] = e0, q, open_
            k += 1
            open_ = 0
    if open_ != 0:
        ent[k], ext[k], sides[k] = e0, n - 1, open_
        k += 1
    return ent[:k], ext[:k], sides[:k]


def cmd_stocks(a):
    O, H, L, C, M, spy, idx, names = stocks_panel(a.data)
    rows, ser = [], {}
    Cf = C.ffill(limit=25)
    r1 = C.pct_change(fill_method=None)
    elig = M & C.notna() & (C.rolling(252).count() >= 252)
    me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last().to_numpy()
    me = [t for t in me if t >= pd.Timestamp("2004-12-01")]
    on = np.log(O / C.shift())
    EL = elig.loc[me[:-1]].to_numpy()
    NX = np.vstack([(Cf.loc[t1] / C.loc[t0] - 1).to_numpy(float) for t0, t1 in zip(me[:-1], me[1:])])
    # S-ANOM
    U = sobol(6, 256, 23)
    Mx, desc = [], []
    cache = {}
    for u in U:
        sig = pick(u[0], ["mom", "rev", "52wh", "lowvol", "overnight"])
        lb = int(rng(u[1], 5, 252))
        sk = int(rng(u[2], 0, 21))
        qn = rng(u[3], 0.05, 0.30)
        hold = int(rng(u[4], 1, 6.999))
        key = (sig, lb, sk)
        if key not in cache:
            if sig == "mom":
                Z = C.shift(sk) / C.shift(sk + lb) - 1
            elif sig == "rev":
                Z = -(C.shift(sk) / C.shift(sk + lb) - 1)
            elif sig == "52wh":
                Z = C / H.rolling(max(lb, 20), min_periods=int(max(lb, 20) * 0.8)).max()
            elif sig == "lowvol":
                Z = -r1.rolling(max(lb, 20), min_periods=int(max(lb, 20) * 0.8)).std()
            else:
                Z = on.rolling(max(lb, 5), min_periods=int(max(lb, 5) * 0.7)).sum()
            cache = {key: Z.loc[me]}                                         # one signal at a time (memory)
        Zm = cache[key].to_numpy(float)
        out = np.zeros(len(me) - 1)
        w_prev = np.zeros(len(names))
        w = np.zeros(len(names))
        for i in range(len(me) - 1):
            e = EL[i]
            if i % hold == 0:
                z = np.where(e, Zm[i], np.nan)
                ok = np.flatnonzero(np.isfinite(z))
                if len(ok) >= 100:
                    k = max(1, int(len(ok) * qn))
                    top = ok[np.argsort(z[ok])[-k:]]
                    w = np.zeros(len(names))
                    w[top] = 1.0 / len(top)
            nx = NX[i]
            uni = nx[e & np.isfinite(nx)]
            port = float(np.nansum(w * np.nan_to_num(nx)))
            turn = float(np.abs(w - w_prev).sum())
            w_prev = w
            out[i] = (port - float(uni.mean()) - 5e-4 * turn) if len(uni) else 0.0
        Mx.append(pd.Series(out, index=pd.DatetimeIndex(me[1:])))
        desc.append({"sig": sig, "lb": lb, "skip": sk, "q": round(qn, 2), "hold": hold})
    W = pd.concat(Mx, axis=1)
    run_wf(rows, ser, "S", "S-anomalies", "SP500", W.to_numpy().T, W.index, 2008, desc=desc, min_active=24, min_valid=12)
    # S-DEL
    Mv = M.to_numpy()
    flat = np.flatnonzero(((~Mv[1:]) & Mv[:-1]).ravel())
    ev = [(i + 1, j) for i, j in (divmod(f, Mv.shape[1]) for f in flat)]
    Ca = C.to_numpy(float)
    sp = spy.to_numpy(float)
    Mx, desc = [], []
    for hold in range(5, 125, 5):
        recs = []
        for d, j in ev:
            p0 = Ca[d, j]
            if not np.isfinite(p0):
                continue
            k = min(len(idx) - 1, d + hold)
            seg = Ca[d:k + 1, j]
            fin = np.flatnonzero(np.isfinite(seg))
            if len(fin) < 2:
                continue
            kk = d + fin[-1]
            recs.append((idx[kk], seg[fin[-1]] / p0 - 1 - 10e-4 - (sp[kk] / sp[d] - 1)))
        s = pd.Series([r for _, r in recs], index=pd.DatetimeIndex([t for t, _ in recs]))
        s = s.groupby(s.index.to_period("M")).mean()
        s.index = s.index.to_timestamp(how="end").normalize()
        Mx.append(s)
        desc.append({"hold": hold})
    W = pd.concat(Mx, axis=1).reindex(pd.date_range("2005-01-31", DEV_END, freq="ME")).fillna(0.0)
    run_wf(rows, ser, "S", "S-deletions", "SP500", W.to_numpy().T, W.index, 2008, desc=desc, min_active=12, min_valid=12)
    # S-PAIRS
    Mv = M.to_numpy()
    pos_of = {t: idx.get_loc(t) for t in me}
    top_pairs = {}
    for F in (126, 252, 378, 504):
        for t in me:
            p = pos_of[t]
            if p < F:
                continue
            cols = np.flatnonzero(Mv[p] & np.isfinite(Ca[p - F:p + 1]).all(axis=0))
            if len(cols) < 100:
                continue
            Wn = Ca[p - F:p + 1, cols]
            Xn = Wn / Wn[0]
            sq = (Xn ** 2).sum(axis=0)
            Dm = sq[:, None] + sq[None, :] - 2 * Xn.T @ Xn
            iu = np.triu_indices(len(cols), 1)
            best = np.argsort(Dm[iu])[:50]
            pr = []
            for b in best:
                i, j = cols[iu[0][b]], cols[iu[1][b]]
                spf = Xn[:, iu[0][b]] - Xn[:, iu[1][b]]
                pr.append((i, j, Ca[p - F, i], Ca[p - F, j], spf.std()))
            top_pairs[(F, p)] = pr
    U = sobol(5, 64, 29)
    Mx, desc = [], []
    for u in U:
        F = pick(u[0], [126, 252, 378, 504])
        ez = rng(u[1], 1.5, 3.0)
        xz = rng(u[2], 0.0, 1.0)
        npair = int(rng(u[3], 5, 50))
        Tm = int(rng(u[4], 3, 12))
        recs = []
        for t in me:
            p = pos_of[t]
            if (F, p) not in top_pairs:
                continue
            end = min(len(idx) - 1, p + 21 * Tm)
            for (i, j, bi, bj, sd) in top_pairs[(F, p)][:npair]:
                if sd <= 0:
                    continue
                en, ex, sd_ = pair_trade(Ca[p + 1:end + 1, i], Ca[p + 1:end + 1, j], bi, bj, sd, ez, xz)
                for e0, x0, s0 in zip(en, ex, sd_):
                    e1, x1 = p + 1 + e0, p + 1 + x0
                    ri, rj = Ca[x1, i] / Ca[e1, i] - 1, Ca[x1, j] / Ca[e1, j] - 1
                    recs.append((idx[x1], ((rj - ri) if s0 < 0 else (ri - rj)) - 4 * 5e-4))
        s = pd.Series([r for _, r in recs], index=pd.DatetimeIndex([t for t, _ in recs]))
        s = s.groupby(s.index.to_period("M")).sum() / (npair * Tm)
        s.index = s.index.to_timestamp(how="end").normalize()
        Mx.append(s)
        desc.append({"F": F, "entry_z": round(ez, 2), "exit_z": round(xz, 2), "pairs": npair, "months": Tm})
    W = pd.concat(Mx, axis=1).reindex(pd.date_range("2005-01-31", DEV_END, freq="ME")).fillna(0.0)
    run_wf(rows, ser, "S", "S-pairs", "SP500", W.to_numpy().T, W.index, 2008, desc=desc, min_active=24, min_valid=12)
    save(a.out, "S", rows, ser)


# ============================================================================================ K
def cmd_crypto(a):
    import r21_rest as R21
    rows, ser = [], {}
    cd = os.path.join(a.out, "cd")
    if not glob.glob(os.path.join(cd, "crypto_daily.parquet")):
        R21.cmd_cdata(argparse.Namespace(out=cd))
    D = pd.read_parquet(os.path.join(cd, "crypto_daily.parquet"))
    C = D.pivot(index="date", columns="sym", values="close").sort_index()
    Q = D.pivot(index="date", columns="sym", values="qvol").reindex_like(C)
    C = C[C.index <= DEV_END]
    Q = Q.reindex(C.index)
    Cf = C.ffill(limit=10)
    hist = C.notna().cumsum()
    qv30 = Q.rolling(30, min_periods=20).sum()
    U = sobol(6, 256, 31)
    Mx, desc = [], []
    for u in U:
        lb = int(rng(u[0], 3, 90))
        sk = int(rng(u[1], 0, 7))
        qn = rng(u[2], 0.10, 0.30)
        nu = int(rng(u[3], 20, 100))
        reb = int(rng(u[4], 1, 4.999))
        mode = pick(u[5], ["mom_ls", "rev_ls", "mom_long", "rev_long"])
        sat = C.index[C.index.dayofweek == 5]
        out, w_prev, w = [], pd.Series(0.0, index=C.columns), pd.Series(0.0, index=C.columns)
        dts = []
        for i, (t0, t1) in enumerate(zip(sat[:-1], sat[1:])):
            if t0 < pd.Timestamp("2018-12-01"):
                continue
            ok = (hist.loc[t0] >= 60) & C.loc[t0].notna()
            v = qv30.loc[t0][ok].dropna()
            uni = v.sort_values().index[-nu:]
            if i % reb == 0 and len(uni) >= 20:
                past = (Cf.shift(sk).loc[t0, uni] / Cf.shift(sk + lb).loc[t0, uni] - 1).dropna()
                if len(past) >= 20:
                    o = past.sort_values()
                    q = max(1, int(len(o) * qn))
                    top, bot = o.index[-q:], o.index[:q]
                    longs, shorts = (top, bot) if mode.startswith("mom") else (bot, top)
                    w = pd.Series(0.0, index=C.columns)
                    w[longs] = 1.0 / q
                    if mode.endswith("ls"):
                        w[shorts] = -1.0 / q
            nxt = Cf.loc[t1] / C.loc[t0] - 1
            r = float((w * nxt.fillna(0)).sum())
            if mode.endswith("long"):
                r -= float(nxt[uni].dropna().mean()) if len(uni) else 0.0
            turn = float((w - w_prev).abs().sum())
            w_prev = w
            out.append(r - 10e-4 * turn)
            dts.append(t1)
        Mx.append(pd.Series(out, index=pd.DatetimeIndex(dts)))
        desc.append({"lb": lb, "skip": sk, "q": round(qn, 2), "universe": nu, "reb_weeks": reb, "mode": mode})
    W = pd.concat(Mx, axis=1)
    run_wf(rows, ser, "K", "K-crypto-xs", "Binance", W.to_numpy().T, W.index, 2021, desc=desc, min_active=30)
    # K-CC cash-and-carry on BTC / ETH
    for sym in ("BTCUSDT", "ETHUSDT"):
        ff = glob.glob(os.path.join(a.funding, "**", f"funding_{sym}.csv"), recursive=True)
        pf = sorted(glob.glob(os.path.join(a.funding, "**", "m1", sym, "*.parquet"), recursive=True))
        if not ff or not pf or sym not in C:
            continue
        Fd = pd.read_csv(ff[0], parse_dates=["ts"])
        Fd["day"] = Fd.ts.dt.tz_convert(None).dt.normalize()
        fday = Fd.groupby("day").rate.sum()
        perp = pd.concat([pd.read_parquet(x, columns=["ts", "bid_c", "ask_c"]) for x in pf])
        perp = perp.set_index(pd.DatetimeIndex(perp.ts))
        pc = (0.5 * (perp.bid_c + perp.ask_c)).groupby(perp.index.tz_convert(None).normalize()).last()
        sp = C[sym]
        idx2 = pc.index.intersection(sp.dropna().index)
        pc, sp2 = pc.reindex(idx2), sp.reindex(idx2)
        basis_ret = sp2.pct_change(fill_method=None) - pc.pct_change(fill_method=None)
        fd = fday.reindex(idx2).fillna(0.0)
        U = sobol(3, 128, 37 + len(sym))
        Mx, desc = [], []
        for u in U:
            w = int(rng(u[0], 1, 30))
            en = rng(u[1], 0, 3e-4) * 3                                        # per day (3 stamps)
            ex = en * rng(u[2], 0, 1)
            trail = fd.rolling(w).mean().shift(1)
            pos = np.zeros(len(idx2))
            cur = 0.0
            tv = trail.to_numpy()
            for t in range(len(idx2)):
                if cur == 0 and np.isfinite(tv[t]) and tv[t] > en:
                    cur = 1.0
                elif cur == 1 and np.isfinite(tv[t]) and tv[t] < ex:
                    cur = 0.0
                pos[t] = cur
            ps = pd.Series(pos, index=idx2)
            net = ps * (fd + basis_ret.fillna(0.0)) - ps.diff().abs().fillna(0.0) * 2 * 10e-4
            Mx.append(net)
            desc.append({"window": w, "enter_per_day": round(en, 6), "exit_per_day": round(ex, 6)})
        W = pd.concat(Mx, axis=1)
        run_wf(rows, ser, "K", "K-cash-and-carry", sym, W.to_numpy().T, W.index, 2023, desc=desc, min_active=30)
    save(a.out, "K", rows, ser)


# ============================================================================================ V
def cmd_vol(a):
    import missed_edges_lab as ME
    rows, ser = [], {}
    Y = ME.yahoo(["^GSPC", "^VIX", "^VIX3M", "SVXY"], start="1990-01-01")
    spx, vix = Y["^GSPC"]["Close"], Y["^VIX"]["Close"]
    v3 = Y["^VIX3M"]["Close"] if "^VIX3M" in Y else pd.Series(dtype=float)
    lr = np.log(spx).diff()
    rv21 = np.sqrt(252 * (lr ** 2).rolling(21).mean()) * 100
    me = spx.groupby(spx.index.to_period("M")).apply(lambda s: s.index[-1])
    recs = []
    for i in range(len(me) - 1):
        t0, t1 = me.iloc[i], me.iloc[i + 1]
        K = vix.asof(t0)
        seg = lr[(lr.index > t0) & (lr.index <= t1)]
        if len(seg) < 15 or not np.isfinite(K):
            continue
        RV = math.sqrt(252 * (seg ** 2).mean()) * 100
        recs.append({"m": t1, "pnl": (K ** 2 - RV ** 2) / (2 * K) - 1.0, "vix": K, "ratio": K / v3.asof(t0) if len(v3) else np.nan,
                     "gap": K - rv21.asof(t0)})
    Mo = pd.DataFrame(recs).set_index("m")
    U = sobol(5, 256, 41)
    Mx, desc = [], []
    for u in U:
        use_r = u[0] < 0.5
        rcut = rng(u[1], 0.85, 1.05)
        use_g = u[2] < 0.5
        gcut = rng(u[3], -5, 10)
        vhi = rng(u[4], 20, 60)
        on = (Mo.vix <= vhi)
        if use_r:
            on &= Mo.ratio < rcut
        if use_g:
            on &= Mo.gap > gcut
        Mx.append(Mo.pnl.where(on, 0.0))
        desc.append({"ratio_filter": round(rcut, 3) if use_r else None, "gap_filter": round(gcut, 1) if use_g else None,
                     "vix_max": round(vhi, 1)})
    W = pd.concat(Mx, axis=1)
    run_wf(rows, ser, "V", "V-spx-variance", "SPX", W.to_numpy().T, W.index, 1995, desc=desc, min_active=24, min_valid=12)
    if "SVXY" in Y:
        sv = Y["SVXY"]["Adj Close"] if "Adj Close" in Y["SVXY"] else Y["SVXY"]["Close"]
        r = sv.pct_change(fill_method=None)
        ratio = (vix / v3).shift(2).reindex(r.index)
        gap = (vix - rv21).shift(2).reindex(r.index)
        U = sobol(4, 256, 43)
        Mx, desc = [], []
        for u in U:
            use_r, rcut, use_g, gcut = u[0] < 0.7, rng(u[1], 0.80, 1.05), u[2] < 0.5, rng(u[3], -5, 10)
            on = pd.Series(True, index=r.index)
            if use_r:
                on &= ratio < rcut
            if use_g:
                on &= gap > gcut
            pos = on.astype(float)
            Mx.append((pos * r - pos.diff().abs().fillna(0) * 5e-4).fillna(0.0))
            desc.append({"ratio_filter": round(rcut, 3) if use_r else None, "gap_filter": round(gcut, 1) if use_g else None})
        W = pd.concat(Mx, axis=1)
        W = W[W.index >= "2012-01-01"]
        run_wf(rows, ser, "V", "V-svxy", "SVXY", W.to_numpy().T, W.index, 2015, desc=desc, min_active=30)
    try:
        import crypto_vrp_lab as CV
        dv = CV.dvol("BTC")
        px = CV.spot("BTC-USD")
        Mo2 = CV.months(dv, px)
        Bm = pd.DataFrame({"pnl": Mo2.net_vp, "gap": Mo2.K - Mo2.rv_prev})
        print("V-btc months", len(Bm), flush=True)
        Mx, desc = [], []
        for g in np.linspace(-10, 20, 31):
            Mx.append(Bm.pnl.where(Bm.gap > g, 0.0))
            desc.append({"gap_filter": round(float(g), 1)})
        Mx.append(Bm.pnl)
        desc.append({"gap_filter": None})
        W = pd.concat(Mx, axis=1)
        run_wf(rows, ser, "V", "V-btc-dvol", "BTC", W.to_numpy().T, W.index, 2023, desc=desc, min_active=12, min_valid=12)
    except Exception as ex:                                                # noqa: BLE001
        print("V-btc failed", repr(ex)[:200], flush=True)
    save(a.out, "V", rows, ser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", default="")
    ap.add_argument("--funding", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"d7": cmd_d7, "macro": cmd_macro, "commod": cmd_commod, "stocks": cmd_stocks, "crypto": cmd_crypto,
     "vol": cmd_vol}[a.cmd](a)


if __name__ == "__main__":
    main()
