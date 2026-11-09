"""Round 15 part D tests on the free datasets (docs/round15_preregistration.md, addendum 3).

D1a  drift after high-impact surprises (Forex Factory archive, M1 bid / ask from the store)
D1b  weekly economic-surprise index per currency, traded on the 28 crosses
D2   (separate: scripts/long_lab.py on the new Dukascopy instruments, combined by --part d2)
D4   CBOE option-selling indices: alpha against the S&P 500 total return
D5   two-year rate-differential momentum on 15 crosses
Usage: python scripts/data15_tests.py --part d1a|d1b|d4|d5|d2 [--panels DIR] [--ff FILE] [--d2 DIR] --out DIR"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
DEV_END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
FIT = ("2008-01", "2018-12")
CHK = ("2019-01", "2025-12")
CUR8 = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")
FX28 = sorted({a + b for i, a in enumerate(CUR8) for b in CUR8[i + 1:]})
INVERT = ("Unemployment", "Jobless", "Claimant")


def nw_t(x, lags=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else np.nan


def cl_t(x, groups):
    x = pd.Series(np.asarray(x, float))
    g = pd.Series(np.asarray(groups))
    m = x.mean()
    s = (x - m).groupby(g.values).sum()
    se = math.sqrt((s ** 2).sum()) / len(x)
    return float(m / se) if se > 0 else np.nan


def get(url, tries=6, timeout=120):
    import requests
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            last = r.status_code
        except Exception as ex:                                               # noqa: BLE001
            last = repr(ex)[:100]
        time.sleep(min(120, 5 * 2 ** k))
    raise RuntimeError(f"{url}: {last}")


def months_split(m: pd.Series):
    g = pd.period_range(FIT[0], CHK[1], freq="M").astype(str)
    m = m.reindex(g).fillna(0.0)
    return m[m.index <= FIT[1]], m[m.index >= CHK[0]]


def row(name, monthly: pd.Series, gross: pd.Series | None = None, extra=None):
    f, c = months_split(monthly)
    r = {"row": name, "fit_mean": float(f.mean()), "fit_t": nw_t(f), "check_mean": float(c.mean()), "check_t": nw_t(c)}
    if gross is not None:
        gf, gc = months_split(gross)
        r["check_gross_t"] = nw_t(gc)
    if extra:
        r.update(extra)
    return r


# ================================================================================ Forex Factory surprises
def parse_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace(",", "").strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([KMBT%]?)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v * {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2), 1.0)


def surprises(ff_file):
    F = pd.read_csv(ff_file) if ff_file.endswith(".csv") else pd.read_parquet(ff_file)
    F["t"] = pd.to_datetime(F["DateTime"], utc=True, errors="coerce")
    F = F[F.t.notna() & (F.t <= DEV_END) & F.Currency.isin(CUR8)].copy()
    F["A"], F["Fc"] = F.Actual.map(parse_num), F.Forecast.map(parse_num)
    F = F[np.isfinite(F.A) & np.isfinite(F.Fc)].sort_values("t")
    F["s"] = F.A - F.Fc
    F["key"] = F.Currency + "|" + F.Event.astype(str)
    sd = F.groupby("key").s.transform(lambda x: x.shift(1).expanding(min_periods=8).std())
    F["z"] = F.s / sd
    sign = np.where(F.Event.astype(str).str.contains("|".join(INVERT), case=False), -1.0, 1.0)
    F["z"] = F.z * sign
    F = F[np.isfinite(F.z)]
    return F


def part_d1a(a, out):
    """Drift after high-impact surprises: enter 5 minutes after the release at the bid / ask, exit after 60 minutes
    and after one day."""
    import edges_lab2 as E2
    F = surprises(a.ff)
    F = F[(F.Impact.astype(str).str.contains("High", case=False)) & (F.z.abs() >= 1)].copy()
    legs = {"EUR": [("EURUSD", 1)], "GBP": [("GBPUSD", 1)], "AUD": [("AUDUSD", 1)], "NZD": [("NZDUSD", 1)],
            "CAD": [("USDCAD", -1)], "CHF": [("USDCHF", -1)], "JPY": [("USDJPY", -1)],
            "USD": [("EURUSD", -1), ("USDJPY", 1), ("GBPUSD", -1)]}
    pairs = sorted({p for v in legs.values() for p, _ in v})
    books = {}
    for p in pairs:
        bk = E2.book(p)
        if bk is None:
            continue
        m = bk.m
        books[p] = (m.index.as_unit("ns").asi8, m.bid_o.to_numpy(float), m.ask_o.to_numpy(float))
        print("loaded", p, len(m), flush=True)
    rows = []
    for _, e in F.iterrows():
        d = 1.0 if e.z > 0 else -1.0
        t_in = e.t + pd.Timedelta(minutes=5)
        for hz, dt in (("60m", pd.Timedelta(minutes=60)), ("1d", pd.Timedelta(days=1))):
            rets = []
            for p, sgn in legs[e.Currency]:
                if p not in books:
                    continue
                T, B, A = books[p]
                i = np.searchsorted(T, t_in.value)
                if i >= len(T) or T[i] - t_in.value > 10 * 60 * 1e9:
                    continue
                t_out = pd.Timestamp(T[i], tz="UTC") + dt
                if hz == "1d" and t_out.weekday() >= 5:
                    t_out = t_out + pd.Timedelta(days=7 - t_out.weekday())
                j = np.searchsorted(T, t_out.value)
                if j >= len(T) or T[j] - t_out.value > 60 * 60 * 1e9:
                    continue
                side = d * sgn
                if side > 0:
                    r = B[j] / A[i] - 1
                    g = (B[j] + A[j]) / (B[i] + A[i]) - 1
                else:
                    r = 1 - A[j] / B[i]
                    g = 1 - (B[j] + A[j]) / (B[i] + A[i])
                rets.append((r, g))
            if rets:
                rows.append({"t": e.t, "ccy": e.Currency, "event": e.Event, "z": e.z, "hz": hz,
                             "net_bp": 1e4 * np.mean([x[0] for x in rets]), "gross_bp": 1e4 * np.mean([x[1] for x in rets])})
    T_ = pd.DataFrame(rows)
    T_.to_csv(os.path.join(out, "D1a_trades.csv.gz"), index=False)
    res = []
    for hz, g in T_.groupby("hz"):
        day = g.t.dt.strftime("%Y-%m-%d")
        fit, chk = g[g.t < "2019-01-01"], g[g.t >= "2019-01-01"]
        res.append({"row": f"D1a|{hz}", "trades_fit": int(len(fit)), "trades_check": int(len(chk)),
                    "fit_mean": float(fit.net_bp.mean()), "fit_t": cl_t(fit.net_bp, fit.t.dt.strftime("%Y-%m-%d")),
                    "check_mean": float(chk.net_bp.mean()), "check_t": cl_t(chk.net_bp, chk.t.dt.strftime("%Y-%m-%d")),
                    "check_gross_mean": float(chk.gross_bp.mean()), "check_gross_t": cl_t(chk.gross_bp, chk.t.dt.strftime("%Y-%m-%d")),
                    "units": "bp per trade"})
        del day
    return res


# ================================================================================ weekly cross positions (round-14 engine)
def panels(pdir):
    import trend14 as T14
    return T14.load_panels(pdir), T14


def weekly_cross_book(P, T14, signal_fn, syms, name):
    """signal_fn(sym, d) -> Series of target notional decided at each week's last D1 bar (NaN elsewhere)."""
    mon = []
    for s in syms:
        if s not in P:
            continue
        d = P[s]
        sig = signal_fn(s, d)
        if sig is None:
            continue
        iso = d.tday.dt.isocalendar()
        key = (iso.year * 100 + iso.week).to_numpy()
        wend = np.r_[key[1:] != key[:-1], True]
        tg = pd.Series(np.where(wend, sig.to_numpy(float), np.nan), index=d.index).ffill().shift(1)
        pnl, gp, sw = T14.run_cont(d, tg.to_numpy())
        mon.append(pd.DataFrame({"pnl": pnl, "gpnl": gp, "month": d.month.to_numpy()}).groupby("month").sum())
    M = pd.concat(mon).groupby(level=0).sum()
    return row(name, M.pnl, M.gpnl, {"instruments": len(mon)})


def vol_ann(d):
    c = d.close
    return np.sqrt((np.log(c / c.shift(1)) ** 2).ewm(span=36, min_periods=36).mean() * 256)


def part_d1b(a, out):
    P, T14 = panels(a.panels)
    F = surprises(a.ff)
    F = F[F.Impact.astype(str).str.contains("High|Medium", case=False)]
    F["zc"] = F.z.clip(-3, 3)
    # weekly index per currency: sum of the last 13 weeks' capped z, known at the Friday close (21:00 UTC)
    F["wk"] = (F.t - pd.Timedelta(hours=21)).dt.to_period("W-FRI")
    W = F.pivot_table(index="wk", columns="Currency", values="zc", aggfunc="sum").sort_index()
    W = W.reindex(pd.period_range(W.index.min(), W.index.max(), freq="W-FRI")).fillna(0.0)
    I = W.rolling(13, min_periods=13).sum()
    sd = I.std(axis=1)
    I.index = I.index.end_time.normalize()                                # the Friday of each week

    def sig(s, d):
        x, y = s[:3], s[3:]
        if x not in I or y not in I:
            return None
        diff = ((I[x] - I[y]) / sd.reindex(I.index).replace(0, np.nan).to_numpy()).clip(-2, 2)
        td = d.tday.dt.normalize()
        v = diff.reindex(td.values, method="ffill").to_numpy()
        return pd.Series(v, index=d.index) * 0.10 / vol_ann(d)
    return [weekly_cross_book(P, T14, sig, FX28, "D1b|surprise index")]


# ================================================================================ CBOE indices
def part_d4(a, out):
    import statsmodels.api as sm
    import yfinance as yf
    spx = yf.download("^SP500TR", start="2006-12-01", end="2026-01-01", progress=False, auto_adjust=False)
    sp = spx["Close"].squeeze().dropna()
    spm = sp.resample("ME").last().pct_change()
    res = []
    for ix in ("BXM", "BXMD", "PUT", "CNDR", "BFLY", "CLL", "CMBO", "WPUT", "PPUT", "CLLZ"):
        r = get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{ix}_History.csv")
        df = pd.read_csv(io.StringIO(r.text))
        df["DATE"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        s.index = df.DATE
        s = s.dropna().sort_index()
        s = s[s.index <= "2025-12-31"]
        m = s.resample("ME").last().pct_change()
        J = pd.concat([m.rename("y"), spm.rename("x")], axis=1).dropna()
        J.index = J.index.to_period("M").astype(str)
        rec = {"row": f"D4|{ix}", "first": str(s.index.min().date())}
        for nm, lo, hi in (("fit", FIT[0], FIT[1]), ("check", CHK[0], CHK[1])):
            j = J[(J.index >= lo) & (J.index <= hi)]
            if len(j) < 24:
                continue
            o = sm.OLS(j.y.to_numpy(), sm.add_constant(j.x.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            rec[f"{nm}_alpha_pct_month"] = float(o.params[0] * 100)
            rec[f"{nm}_alpha_t"] = float(o.tvalues[0])
            rec[f"{nm}_beta"] = float(o.params[1])
            rec[f"{nm}_mean_pct_month"] = float(j.y.mean() * 100)
            rec[f"{nm}_sharpe"] = float(j.y.mean() / j.y.std() * math.sqrt(12))
            rec[f"{nm}_spx_sharpe"] = float(j.x.mean() / j.x.std() * math.sqrt(12))
        rec["fit_mean"] = rec.get("fit_alpha_pct_month", np.nan)
        rec["check_mean"] = rec.get("check_alpha_pct_month", np.nan)
        rec["check_t"] = rec.get("check_alpha_t", np.nan)
        rec["fit_t"] = rec.get("fit_alpha_t", np.nan)
        res.append(rec)
    return res


# ================================================================================ two-year yields
def yields2y():
    """Daily two-year yields (percent): USD, EUR, CAD, AUD, CHF, JPY, from the official free files."""
    Y = {}
    # US Treasury par curve, one file a year
    us = []
    for y in range(2006, 2026):
        r = get(f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
        d = pd.read_csv(io.StringIO(r.text))
        us.append(pd.Series(pd.to_numeric(d["2 Yr"], errors="coerce").to_numpy(), index=pd.to_datetime(d["Date"])))
    Y["USD"] = pd.concat(us).sort_index()
    # ECB AAA euro-area curve, 2-year spot
    r = get("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y?format=csvdata")
    d = pd.read_csv(io.StringIO(r.text))
    Y["EUR"] = pd.Series(pd.to_numeric(d.OBS_VALUE, errors="coerce").to_numpy(), index=pd.to_datetime(d.TIME_PERIOD)).sort_index()
    # Bank of Canada Valet
    r = get("https://www.bankofcanada.ca/valet/observations/BD.CDN.2YR.DQ.YLD/csv")
    txt = r.text
    i = txt.find('"date"')
    d = pd.read_csv(io.StringIO(txt[i:]))
    Y["CAD"] = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0], errors="coerce")).dropna().sort_index()
    # RBA F2 (current) and F2.1 (historical): 2-year Commonwealth government bond yield
    au = []
    for url in ("https://www.rba.gov.au/statistics/tables/csv/f2.1-data.csv", "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"):
        try:
            raw = get(url).content.decode("utf-8", errors="ignore")
            lines = raw.splitlines()
            hdr = next(k for k, l in enumerate(lines) if l.startswith("Series ID"))
            ids = lines[hdr].split(",")
            titles = next(l for l in lines if l.startswith("Title")).split(",")
            col = next(k for k, t in enumerate(titles) if "2 years" in t or "2 year" in t)
            body = pd.read_csv(io.StringIO("\n".join(lines[hdr + 1:])), header=None)
            au.append(pd.Series(pd.to_numeric(body.iloc[:, col], errors="coerce").to_numpy(),
                                index=pd.to_datetime(body.iloc[:, 0], format="%d-%b-%Y", errors="coerce")))
            del ids
        except Exception as ex:                                                # noqa: BLE001
            print("RBA", url, repr(ex)[:120])
    if au:
        s = pd.concat(au).dropna()
        Y["AUD"] = s[~s.index.duplicated(keep="last")].sort_index()
    # SNB Swiss Confederation bond yields (cube rendoblid), 2-year
    try:
        raw = get("https://data.snb.ch/api/cube/rendoblid/data/csv/en").content.decode("utf-8-sig", errors="ignore")
        lines = raw.splitlines()
        hdr = next(k for k, l in enumerate(lines) if l.startswith('"Date"') or l.startswith("Date"))
        d = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=";")
        dim = [c for c in d.columns if c not in ("Date", "Value")][0]
        two = d[d[dim].astype(str).str.upper().isin(["2J", "2Y", "2"])]
        Y["CHF"] = pd.Series(pd.to_numeric(two.Value, errors="coerce").to_numpy(), index=pd.to_datetime(two.Date, errors="coerce")).dropna().sort_index()
    except Exception as ex:                                                    # noqa: BLE001
        print("SNB", repr(ex)[:160])
    # Japan Ministry of Finance JGB curve
    try:
        raw = get("https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv").content.decode("utf-8", errors="ignore")
        lines = raw.splitlines()
        hdr = next(k for k, l in enumerate(lines) if l.startswith("Date"))
        d = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
        Y["JPY"] = pd.Series(pd.to_numeric(d["2Y"], errors="coerce").to_numpy(), index=pd.to_datetime(d["Date"], errors="coerce")).dropna().sort_index()
    except Exception as ex:                                                    # noqa: BLE001
        print("MoF", repr(ex)[:160])
    for k, v in Y.items():
        v = v[(v.index >= "2005-01-01") & (v.index <= "2025-12-31")]
        Y[k] = v[~v.index.duplicated(keep="last")]
        print("yield", k, len(Y[k]), Y[k].index.min().date() if len(Y[k]) else None, Y[k].index.max().date() if len(Y[k]) else None, flush=True)
    return Y


def part_d5(a, out):
    P, T14 = panels(a.panels)
    Y = yields2y()
    cur = [c for c in ("USD", "EUR", "CAD", "AUD", "CHF", "JPY") if c in Y and len(Y[c]) > 500]
    syms = [s for s in FX28 if s[:3] in cur and s[3:] in cur]
    Yd = pd.DataFrame(Y).sort_index().ffill()
    res = []
    for wks in (4, 13):
        def sig(s, d, wks=wks):
            spread = (Yd[s[:3]] - Yd[s[3:]]).asfreq("D").ffill()
            ch = spread - spread.shift(7 * wks)                                # calendar weeks
            # yields of the day before (some curves are published the next morning)
            v = ch.reindex((d.tday.dt.normalize() - pd.Timedelta(days=1)).values, method="ffill").to_numpy()
            return pd.Series(np.sign(v), index=d.index) * 0.10 / vol_ann(d)
        r = weekly_cross_book(P, T14, sig, syms, f"D5|{wks}w")
        r["currencies"] = ",".join(cur)
        res.append(r)
    return res


# ================================================================================ D2: the new instruments
def part_d2(a, out):
    groups = {"commodities": ["BRENTCMDUSD", "LIGHTCMDUSD", "GASCMDUSD", "COPPERCMDUSD"],
              "indices": ["DEUIDXEUR", "GBRIDXGBP", "FRAIDXEUR", "EUSIDXEUR", "HKGIDXHKD", "CHEIDXCHF"],
              "bonds": ["BUNDTREUR", "USTBONDTRUSD", "UKGILTTRGBP"]}
    fs = sorted(glob.glob(os.path.join(a.d2, "**", "A_daily_*.parquet"), recursive=True))
    D = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    D["month"] = pd.to_datetime(D.date).dt.to_period("M").astype(str)
    Mo = D.groupby(["system", "symbol", "month"])[["pnl", "gpnl"]].sum().reset_index()
    Mo.to_csv(os.path.join(out, "D2_monthly_by_symbol.csv"), index=False)
    res = []
    for g, syms in groups.items():
        have = sorted(set(Mo.symbol) & set(syms))
        if not have:
            continue
        bh = Mo[(Mo.system == "BH") & Mo.symbol.isin(have)].groupby("month").pnl.sum()
        for sysn in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "BH"]:
            x = Mo[(Mo.system == sysn) & Mo.symbol.isin(have)].groupby("month")[["pnl", "gpnl"]].sum()
            if not len(x):
                continue
            grid = pd.period_range("2013-01", CHK[1], freq="M").astype(str)
            m, gm, b = x.pnl.reindex(grid).fillna(0), x.gpnl.reindex(grid).fillna(0), bh.reindex(grid).fillna(0)
            pre, chk = m[m.index < "2019-01"], m[m.index >= "2019-01"]
            # excess over long-only, both scaled to 10% vol on 2013-2018
            sc = 0.10 / (pre.std() * math.sqrt(12)) if pre.std() > 0 else np.nan
            bs = 0.10 / (b[b.index < "2019-01"].std() * math.sqrt(12)) if b[b.index < "2019-01"].std() > 0 else np.nan
            ex = (m * sc - b * bs)[m.index >= "2019-01"]
            res.append({"row": f"D2|{g}|{sysn}", "instruments": ",".join(have), "pre_2013_2018_t": nw_t(pre),
                        "fit_mean": float(pre.mean()), "fit_t": nw_t(pre), "check_mean": float(chk.mean()), "check_t": nw_t(chk),
                        "check_gross_t": nw_t(gm[gm.index >= "2019-01"]), "excess_check_t": nw_t(ex) if sysn != "BH" else np.nan,
                        "check_mean_pct_at_10vol": float(chk.mean() * sc * 100)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--panels", default="")
    ap.add_argument("--ff", default="")
    ap.add_argument("--d2", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = {"d1a": part_d1a, "d1b": part_d1b, "d4": part_d4, "d5": part_d5, "d2": part_d2}[a.part]
    res = fn(a, a.out)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(a.out, f"{a.part}_rows.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
