"""Third literature round on M1 bid/ask bars: end-of-day momentum (Baltussen, Da, Lammers & Martens 2021), the
Japanese investment-trust month-start flow, and the dollar around the ECB fix (Krohn, Mueller & Whelan 2024).
Rules fixed before any result; nothing tuned. Costs as in the flows lab (crossing the quoted spread, 25% of it plus
0.1 pip slippage a side, 0.35 bp commission a side, swap where held over 17:00 New York). Development data only.

PRE-REGISTERED TESTS (one-sided, mean net > 0; Benjamini-Hochberg over all primary tests):
  E1 US500 and NAS100 CFDs: the return from the previous 16:00 New York close to 15:30 predicts 15:30 -> 16:00;
     trade the last half hour in its sign, every weekday, 2019-2025.
  E2 Gold (XAUUSD): the return from the previous COMEX close (13:30 New York) to 13:00 predicts 13:00 -> 13:30;
     trade it in its sign (data with those hours: 2007-2018).
  E3 FX: the return from the previous CME settlement (15:00 New York) to 14:30 predicts 14:30 -> 15:00; trade it in
     its sign on EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD; per pair and pooled per day.
  T1 Investment-trust launches: long USDJPY 08:00 -> 15:00 Tokyo on the first three Tokyo business days of the
     month; control: the same trade on the other Tue-Fri business days (Welch t reported).
  K1 The dollar before the ECB fix: long the USD 13:15 -> 14:15 Frankfurt; K2 after it: short the USD 14:15 ->
     15:15; K3 both legs; weekdays, EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, pooled per day.
Usage: python scripts/lit3_intraday_lab.py --out DIR [--only eod,toshin,ecb]"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

U.PIPS.update({"GBPUSD": 1e-4, "USDCAD": 1e-4, "USDCHF": 1e-4, "NZDUSD": 1e-4})
FX = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")
USD_BASE = {"USDJPY", "USDCAD", "USDCHF"}


def mid_at(bk, i):
    return 0.5 * (bk.bo[i] + bk.ao[i])


def eod(bk, tz, close_hm, signal_hm, days=None):
    """Trade signal_hm -> close_hm in the sign of the return from the previous close_hm to signal_hm."""
    out = {}
    days = days if days is not None else pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date()))
    prev_close = None
    for d in days:
        t_close = FL.local_ts(d, tz, *close_hm)
        t_sig = FL.local_ts(d, tz, *signal_hm)
        ic, isg = bk.at(t_close, 3), bk.at(t_sig, 3)
        if prev_close is not None and isg >= 0 and ic > isg:
            r = mid_at(bk, isg) / mid_at(bk, prev_close) - 1
            if r != 0:
                tr = bk.trade_ij(isg, ic, 1 if r > 0 else -1, t_sig, t_close)
                if tr is not None:
                    out[d] = U.rec(tr)
        prev_close = ic if ic >= 0 else None
    return U.frame(out)


def pooled(frames):
    net = pd.concat([f.net.rename(k) for k, f in frames.items() if len(f)], axis=1)
    gross = pd.concat([f.gross.rename(k) for k, f in frames.items() if len(f)], axis=1)
    return pd.DataFrame({"net": net.mean(axis=1), "gross": gross.mean(axis=1)}).dropna(subset=["net"])


def eod_tests(res):
    for s in ("USA500IDXUSD", "USATECHIDXUSD"):
        bk = U.book(s)
        if bk is not None:
            res[f"E1 {s} end-of-day momentum (15:30-16:00 New York)"] = (eod(bk, "America/New_York", (16, 0), (15, 30)), 252)
    bk = U.book("XAUUSD")
    if bk is not None:
        res["E2 XAUUSD end-of-day momentum (13:00-13:30 New York)"] = (eod(bk, "America/New_York", (13, 30), (13, 0)), 252)
    fr = {}
    for s in FX:
        bk = U.book(s)
        if bk is None:
            continue
        fr[s] = eod(bk, "America/New_York", (15, 0), (14, 30))
        res[f"E3 {s} end-of-day momentum (14:30-15:00 New York)"] = (fr[s], 252)
    if fr:
        res["E3 FX pooled end-of-day momentum"] = (pooled(fr), 252)


def toshin(res):
    bk = U.book("USDJPY")
    if bk is None:
        return
    start, end = str(bk.idx[0].date()), str(bk.idx[-1].date())
    hol = FL.jp_holidays(range(bk.idx[0].year - 1, bk.idx[-1].year + 2))
    bdays = [d for d in pd.bdate_range(start, end) if d not in hol]
    month = pd.Series([d.to_period("M") for d in bdays], index=bdays)
    rank = month.groupby(month).cumcount()
    first3 = [d for d, k in rank.items() if k < 3]
    others = [d for d, k in rank.items() if k >= 3 and d.weekday() != 0]
    f = lambda days: U.timed(bk, days, lambda d: (U.jst(d, 8, 0), U.jst(d, 15, 0)), +1)   # noqa: E731
    ev, ctl = f(first3), f(others)
    res["T1 USDJPY first three Tokyo business days, 08:00-15:00 Tokyo"] = (ev, 36)
    res["T1c control: other Tue-Fri business days"] = (ctl, 200)
    if len(ev) > 10 and len(ctl) > 10:
        res["T1 first-three minus control (Welch t)"] = float(st.ttest_ind(ev.net, ctl.net, equal_var=False).statistic)


def ecb(res):
    legs = {"K1": {}, "K2": {}}
    for s in ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"):
        bk = U.book(s)
        if bk is None:
            continue
        usd_long = +1 if s in USD_BASE else -1
        days = pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date()))
        legs["K1"][s] = U.timed(bk, days, lambda d: (FL.local_ts(d, "Europe/Berlin", 13, 15),
                                                     FL.local_ts(d, "Europe/Berlin", 14, 15)), usd_long)
        legs["K2"][s] = U.timed(bk, days, lambda d: (FL.local_ts(d, "Europe/Berlin", 14, 15),
                                                     FL.local_ts(d, "Europe/Berlin", 15, 15)), -usd_long)
    for k, lab in (("K1", "USD long 13:15-14:15 Frankfurt (into the ECB fix)"),
                   ("K2", "USD short 14:15-15:15 Frankfurt (after the fix)")):
        if legs[k]:
            res[f"{k} {lab}, pooled"] = (pooled(legs[k]), 252)
    if legs["K1"] and legs["K2"]:
        a, b = pooled(legs["K1"]), pooled(legs["K2"])
        j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
        res["K3 both legs around the ECB fix, pooled"] = (pd.DataFrame({"net": j.net_1 + j.net_2,
                                                                       "gross": j.gross_1 + j.gross_2}), 252)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="eod,toshin,ecb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for g, fn in (("eod", eod_tests), ("toshin", toshin), ("ecb", ecb)):
        if g in a.only.split(","):
            try:
                fn(res)
            except Exception:                                               # noqa: BLE001
                import traceback
                res[f"{g} error"] = traceback.format_exc()[-1500:]
                print(res[f"{g} error"], flush=True)
            print(g, "done", flush=True)
    T, out = U.tabulate(res, a.out)
    from cross_market_ta_lab import bh_q
    if len(T):
        prim = ~T.test.str.contains("control", case=False)
        T["q_bh_primary"] = np.nan
        T.loc[prim, "q_bh_primary"] = bh_q(T.loc[prim, "p_one_sided"].to_numpy())
    out["_coverage_m1_bars_per_year"] = U.COVERAGE
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    if len(T):
        print(T.drop(columns=["first", "last"]).round(3).to_string())
    for k, v in out.items():
        if not isinstance(v, dict) or k.startswith("_"):
            print(k, v)


if __name__ == "__main__":
    main()
