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
    for m in meetings:
        sd = (pd.Timestamp(m) - pd.offsets.BDay(3)).normalize()
        row = Z[Z.index <= sd]
        if not len(row):
            continue
        row = row.iloc[-1]
        mm = pd.Period(m, "M")
        nxt, prev = row.get(mm + 1, np.nan), row.get(mm - 1, np.nan)
        if not (np.isfinite(nxt) and np.isfinite(prev)):
            continue
        spread = (100 - nxt) - (100 - prev)
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


def o1(a):
    return {"status": "not yet implemented (waiting for the compact option files)"}


if __name__ == "__main__":
    main()
