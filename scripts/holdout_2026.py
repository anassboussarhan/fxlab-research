"""The 2026 holdout, run once: the frozen rules of docs/holdout_2026_preregistration.md on 2026-01-01..2026-10-31.

Every rule is the development code (imported or copied line for line, with its source named); only the dates
change. The M1 loader opens 2026 only with the unlock phrase whose SHA-256 is in results/frozen_candidates.json.
Development bars from 2025-10 serve as warm-up (the 20-day range of I3); no 2025 trade is counted.
t-statistics (fixed with the pre-registration, before the run): per-trade tests (H1-H6) iid t; the daily R1 / E1
series Newey-West with 5 lags and the weekly C5 series with 4 lags, as in their labs, the variance never below
the iid variance. The verdict uses that t.
Usage: python scripts/holdout_2026.py --cot-data DIR --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

H0, H1END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-31")
YEND = "2026-11-01"                                         # Yahoo end (exclusive)
FROZEN = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                                     "frozen_candidates.json")))
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]


def nw_t(x, lags):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s0 = (e @ e) / n
    s = s0
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    s = max(s, s0)
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def verdict(x, dev_mean, lags=0):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 3:
        return {"n": int(len(x)), "verdict": "no data"}
    t = nw_t(x, lags)
    sd = float(x.std())
    t_dev = float((x.mean() - dev_mean) / (sd / math.sqrt(len(x)))) if sd > 0 else np.nan
    if np.isfinite(t_dev) and t_dev <= -1.645:
        v = "contradicted"
    elif x.mean() > 0 and t >= 1.645:
        v = "confirmed"
    elif x.mean() > 0:
        v = "consistent"
    else:
        v = "not supported"
    months = x.groupby(pd.DatetimeIndex(x.index).month).sum()
    return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "sd_bp": sd * 1e4, "t": t,
            "t_iid": float(x.mean() / sd * math.sqrt(len(x))) if sd > 0 else np.nan,
            "p_one_sided": float(st.norm.sf(t)) if np.isfinite(t) else np.nan,
            "hit": float((x > 0).mean()), "sum_pct": float(x.sum() * 100), "worst_bp": float(x.min() * 1e4),
            "months_positive": f"{int((months > 0).sum())}/{len(months)}",
            "dev_mean_bp": dev_mean * 1e4, "t_vs_dev": t_dev, "verdict": v}


# ----------------------------------------------------------------------------------------------- Gotobi (H1-H5)
DRY = {}


def book26(sym):
    from fxlab.data.store import holdout_unlocked, load_m1
    if DRY:                                                  # code check on a development year, no 2026 data
        y = DRY["year"]
        m = load_m1(sym, f"{y - 1}-10-01", f"{y}-12-31 23:59", columns=COLS).dropna()
    else:
        if not holdout_unlocked():
            raise SystemExit("holdout locked: the unlock phrase is not set or does not match the frozen file")
        m = load_m1(sym, "2025-10-01", "2026-10-31 23:59", allow_holdout=True, columns=COLS).dropna()
    m = m[(m.ask_o >= m.bid_o) & (m.bid_o > 0)]
    U.COVERAGE[sym] = {f"{y}-{mo:02d}": int(n) for (y, mo), n in m.groupby([m.index.year, m.index.month]).size().items()}
    return FL.Book(m, 0.01, sym)


def gotobi_2026(bk):
    import flow_edges_lab as F
    G, _, _ = F.calendar(bk)
    return [d for d in G if H0 <= d <= H1END]


def i3(bk, days):
    """Copied from scripts/untested_intraday_lab.py (the I3 rule)."""
    from fxlab.validation.metrics import trading_day
    rng = bk.avg_range()
    out = {}
    for d in days:
        t0, t1 = U.jst(d - pd.Timedelta(days=1), 23, 0), U.jst(d, 9, 53)
        i, j = bk.at(t0, 10), bk.at(t1, 65)
        if i < 0 or j < 0 or j <= i:
            continue
        td = trading_day(pd.DatetimeIndex([bk.idx[i]]))[0]
        rg = float(rng.get(td, np.nan))
        if not np.isfinite(rg) or rg <= 0:
            continue
        r = bk.bracket(i, j, +1, 1e9, 0.5 * rg)
        if r is not None:
            out[d] = U.rec(r)
    return U.frame(out)


def f1(bk, days):
    """F1 as in scripts/flows_lab.py gotobi(): 03:00 -> 09:55 Tokyo, Mondays skipped, tolerance 65 minutes."""
    out = {}
    for d in days:
        if d.weekday() == 0:
            continue
        r = bk.trade(FL.local_ts(d, "Asia/Tokyo", 3, 0), FL.local_ts(d, "Asia/Tokyo", 9, 55), +1, tol_min=65)
        if r is not None:
            out[d] = U.rec(r)
    return U.frame(out)


def no_stop(bk, days):
    import gotobi_opt_lab as GO
    return GO.all_settings_one(bk, days, GO.BASE)


# ----------------------------------------------------------------------------------------------- daily data
def yahoo(t, start="2002-07-01", adjust=True):
    import yfinance as yf
    h = yf.download(t, start=start, end=YEND, auto_adjust=adjust, progress=False)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h.index = pd.DatetimeIndex(h.index).tz_localize(None)
    return h


def nasdaq_overnight():
    """Copied from scripts/prop_mix_sim.py build(): series O."""
    q, irx = yahoo("QQQ", "2018-01-01", adjust=False), yahoo("^IRX", "2018-01-01", adjust=False)
    f = q["Adj Close"] / q["Close"]
    o, c = q["Open"] * f, q["Adj Close"]
    rf = (irx["Close"] / 100).reindex(q.index).ffill().fillna(0)
    night = o / c.shift(1) - 1
    weekday_night = (q.index.to_series().diff().dt.days == 1)
    O = (night - 1e-4 - (rf.shift(1) + 0.025) / 360)[weekday_night].dropna()
    return O[(O.index >= H0) & (O.index <= H1END)]


def rebalancing():
    """Copied from scripts/rebalance_lab.py main(): R1 and E1."""
    import rebalance_lab as RB
    spy, ief, irx = yahoo("SPY")["Close"], yahoo("IEF")["Close"], yahoo("^IRX")["Close"]
    px = pd.concat([spy.rename("spy"), ief.rename("ief")], axis=1).dropna()
    px = px[px.index <= H1END]
    re, rb = px.spy.pct_change().fillna(0), px.ief.pct_change().fillna(0)
    P = RB.positions(re, rb)
    rf = (irx.reindex(px.index).ffill().fillna(0) / 100).shift(1).fillna(0)
    days = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1)
    spread = re - rb
    p = P["R1"]
    turn = p.diff().abs().fillna(p.abs())
    r1 = p * spread - turn * 2 * 1e-4
    held = p.shift(1).fillna(0)
    fin = np.where(held > 0, -(rf + 0.025), np.where(held < 0, rf - 0.025, 0.0)) * np.abs(held) * days / 360
    e1 = p * re - turn * 2e-4 + pd.Series(fin, index=px.index) * (held != 0)
    sel = (px.index >= H0) & (px.index <= H1END)
    return r1[sel], e1[sel], P["R1"][sel]


def cot_eur(data_dir):
    """scripts/cot_lab.py C5 on the euro, with the end date moved to 2026-10-31."""
    import cot_lab as C
    C.END = H1END
    cot = pd.read_csv(os.path.join(data_dir, "cot.csv"), parse_dates=["date"])
    prices = pd.read_csv(os.path.join(data_dir, "prices.csv"), index_col=0, parse_dates=True)
    panel = C.weekly_panel(cot, prices)
    P = C.positions(panel)
    X = C.pnl(panel, P["C5"])
    x = X.get("EUR", pd.Series(dtype=float))
    last_report = str(cot[cot.market == "EUR"].date.max())[:10]
    return x[(x.index >= H0) & (x.index <= H1END)], last_report


def fetch_cot(out):
    """The COT lab's download, with 2026 added (deacot2026.zip) and prices to 2026-10-31."""
    import io
    import zipfile
    import requests
    import cot_lab as C
    os.makedirs(out, exist_ok=True)
    frames = []
    urls = ["https://www.cftc.gov/files/dea/history/deacot1986_2016.zip"] + \
           [f"https://www.cftc.gov/files/dea/history/deacot{y}.zip" for y in range(2017, 2027)]
    for u in urls:
        r = requests.get(u, timeout=120, headers={"User-Agent": "Mozilla/5.0 research"})
        print(u, r.status_code, len(r.content), flush=True)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for n in z.namelist():
            if n.lower().endswith(".txt"):
                frames.append(pd.read_csv(z.open(n), low_memory=False))
    raw = pd.concat(frames, ignore_index=True)
    C.END = H1END
    cot = C.tidy_cot(raw)
    cot.to_csv(os.path.join(out, "cot.csv"), index=False)
    print(cot.groupby("market").date.agg(["min", "max", "size"]), flush=True)
    px = {}
    for k, (tick, _) in C.YAHOO.items():
        h = yahoo(tick, "2003-01-01")
        px[k] = h["Close"]
    P = pd.DataFrame(px)
    P[P.index <= H1END].to_csv(os.path.join(out, "prices.csv"))


def gold_cny_2026():
    """Information only (lit3 D6, one event a year): GLD over the 15 trading days before 17 February 2026."""
    g = yahoo("GLD", "2025-10-01")["Close"].pct_change().dropna()
    before = g[g.index < pd.Timestamp("2026-02-17")].iloc[-15:]
    return float((1 + before).prod() - 1) - 2e-4, str(before.index.min())[:10], str(before.index.max())[:10]


# ----------------------------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cot-data", default="cotdata")
    ap.add_argument("--fetch-cot", action="store_true")
    ap.add_argument("--dry", type=int, default=0, help="code check: run the Gotobi rules on this development year")
    a = ap.parse_args()
    if a.fetch_cot:
        fetch_cot(a.cot_data)
        return
    if a.dry:
        global H0, H1END
        DRY["year"] = a.dry
        H0, H1END = pd.Timestamp(f"{a.dry}-01-01"), pd.Timestamp(f"{a.dry}-12-31")
    os.makedirs(a.out, exist_ok=True)
    C = FROZEN["candidates"]
    rows, series, info = [], {}, {}

    def add(hid, name, x, dev_bp, lags=0):
        series[hid] = x.net if isinstance(x, pd.DataFrame) else x
        s = series[hid]
        rows.append({"id": hid, "test": name, **verdict(s, dev_bp * 1e-4, lags)})
        print(rows[-1], flush=True)

    errors = {}
    try:
        uj, ej = book26("USDJPY"), book26("EURJPY")
        Gu, Ge = gotobi_2026(uj), gotobi_2026(ej)
        info["gotobi_days_2026"] = [str(d.date()) for d in Gu]
        add("H1", C["H1"]["rule"], i3(uj, Gu), C["H1"]["dev_mean_bp"])
        add("H2", C["H2"]["rule"], f1(uj, Gu), C["H2"]["dev_mean_bp"])
        h3, h4 = no_stop(uj, Gu), no_stop(ej, Ge)
        add("H3", C["H3"]["rule"], h3, C["H3"]["dev_mean_bp"])
        add("H4", C["H4"]["rule"], h4, C["H4"]["dev_mean_bp"])
        b = pd.concat([h3.net.rename("uj"), h4.net.rename("ej")], axis=1).mean(axis=1).dropna()
        add("H5", C["H5"]["rule"], b, C["H5"]["dev_mean_bp"])
    except SystemExit as ex:
        errors["gotobi"] = str(ex)
    except Exception:                                                       # noqa: BLE001
        import traceback
        errors["gotobi"] = traceback.format_exc()[-1500:]
    for hid, fn in (() if a.dry else (("H6", nasdaq_overnight), ("H7", rebalancing), ("H8", cot_eur))):
        try:
            if hid == "H6":
                add("H6", C["H6"]["rule"], fn(), C["H6"]["dev_mean_bp"])
            elif hid == "H7":
                r1, e1, pos = fn()
                add("H7", C["H7"]["rule"], r1, C["H7"]["dev_mean_bp_day"], lags=5)
                add("H7b", C["H7b"]["rule"], e1, C["H7b"]["dev_mean_bp_day"], lags=5)
                info["H7 active days"] = int((pos != 0).sum())
            else:
                x, last = fn(a.cot_data)
                info["H8 last EUR report"] = last
                add("H8", C["H8"]["rule"], x, C["H8"]["dev_mean_bp_week"], lags=4)
        except Exception:                                                   # noqa: BLE001
            import traceback
            errors[hid] = traceback.format_exc()[-1500:]
    try:
        if not a.dry:
            info["gold before Chinese New Year 2026 (information)"] = gold_cny_2026()
    except Exception as ex:                                                 # noqa: BLE001
        errors["cny"] = repr(ex)[:200]
    R = pd.DataFrame(rows)
    z_ids = ["H5", "H6", "H7", "H8"]
    zt = [float(R.loc[R.id == i, "t"].iloc[0]) for i in z_ids if (R.id == i).any() and np.isfinite(R.loc[R.id == i, "t"].iloc[0])]
    summary = {"stouffer_ids": z_ids, "stouffer_n": len(zt),
               "stouffer_z": float(np.sum(zt) / math.sqrt(len(zt))) if zt else np.nan,
               "primary": R.loc[R.id == "H1"].to_dict("records")[0] if (R.id == "H1").any() else None,
               "errors": errors, "info": info, "coverage": U.COVERAGE}
    summary["overall_verdict"] = ("confirmed" if summary["stouffer_z"] >= 1.645 else "not confirmed") \
        if np.isfinite(summary["stouffer_z"]) and len(zt) == 4 else "incomplete"
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    for k, v in series.items():
        pd.Series(v).rename("net").to_csv(os.path.join(a.out, f"series_{k}.csv"))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["test"]).round(3).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"}, indent=1, default=str)[:4000])
    for k, e in errors.items():
        print("::error title=holdout " + k + "::" + str(e).replace("\n", " | ")[-800:], flush=True)


if __name__ == "__main__":
    main()
