"""Literature round 2 (run 22): the effects found in the second literature search (docs/literature_check.md,
section 4) that this data can test, in their published form, with rules fixed before the run. Development data
only (metals 2010-11 / 2011-05 to 2018-06; FX 2007-2018, EURUSD to 2022; no 2026 data).

PRE-REGISTERED PRIMARY TESTS (Benjamini-Hochberg over these 11 in scripts/lit2_report.py):
  S1   silver fix (Caminschi 2014): short XAGUSD 11:45 -> 12:15 London, every business day
  G1   gold PM fix (Abrantes-Metz & Metz 2014): short XAUUSD 14:45 -> 15:15 London, every business day
  G2   gold AM fix: short XAUUSD 10:15 -> 10:45 London
  G3   gold PM fix opening minutes (Caminschi & Heaney 2014): the sign of the 15:00 -> 15:03 London mid move,
       traded 15:03 -> 15:15 in that direction
  A1   gold in Asian hours (gold overnight effect; CBS 2019 "hat-shaped" pattern): long XAUUSD from 18:00 New York
       the evening before (Sunday evening for Mondays) to 08:00 London
  A2   gold in Western hours: short XAUUSD 08:00 London -> 16:55 New York
  SEAS gold seasonality (Baur 2012): long XAUUSD through September and November (17:05 New York on the last
       business day of the previous month -> 16:55 New York on the last business day of the month)
  DC   dollar carry (Lustig, Roussanov & Verdelhan): long the 7 foreign currencies with rate data vs USD (equal
       weight) when their average 3-month rate is above the US rate, else short; monthly
  FM1  dollar-factor momentum (Zhang 2021): long the same basket if its excess return over the past 3 months is
       positive, else short; monthly
  FM2  carry-factor momentum (Zhang 2021): the carry factor (long the 3 highest-yielding, short the 3 lowest of
       the 7) held long if its past-3-month return is positive, else short; monthly
  WM   weekly currency momentum (Raza, Marshall & Visaltanachoti): every 4 weeks, long the 3 currencies of the 9
       that rose most against USD over the past 4 weeks, short the 3 that rose least, held 4 weeks
DIAGNOSTICS (not in the BH family): the same A1 / A2 / SEAS rules on silver; the dollar 2 days before scheduled
FOMC rate changes (Karnaukh), using the REALISED decision as a stand-in for the fed funds futures signal the paper
uses (look-ahead, not a tradable strategy): long USD before hikes, short before cuts, 14:00 ET two business days
before -> 13:55 ET on the day; and the same window long foreign currencies before every scheduled meeting.
Execution and costs as scripts/flows_lab.py (bid/ask M1, slippage, commission, swap through 17:00 New York; metals
swap = -(USD 3-month rate) - 0.75% for a long position). The FX tests (DC, FM1, FM2, WM) are built in the report
from the daily closes exported here.
Usage: python scripts/lit2_lab.py --pair XAUUSD [--out DIR]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import flows_lab as FL  # noqa: E402
from flows_lab import Book, local_ts, prev_bday, rec  # noqa: E402

DEV_END = "2025-12-31"
METALS = ("XAUUSD", "XAGUSD")
FX = {"EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "JPY", "AUDUSD": "AUD", "USDCAD": "CAD", "USDCHF": "CHF",
      "USDSEK": "SEK", "USDNOK": "NOK", "NZDUSD": "NZD"}
# scheduled FOMC meetings 2007-2018 at which the target rate changed (intermeeting moves of 2008-01-22 and
# 2008-10-08 excluded); all are in flows_lab.FOMC_FALLBACK, which matched the Fed's pages in run 17
FOMC_CHANGES = {"2007-09-18": -1, "2007-10-31": -1, "2007-12-11": -1, "2008-01-30": -1, "2008-03-18": -1,
                "2008-04-30": -1, "2008-10-29": -1, "2008-12-16": -1, "2015-12-16": 1, "2016-12-14": 1,
                "2017-03-15": 1, "2017-06-14": 1, "2017-12-13": 1, "2018-03-21": 1, "2018-06-13": 1}


def metal_book(m1, pip, pair):
    """Book whose 'rate differential' is -(USD 3-month rate): a long metal position pays USD interest (lease rates
    are about zero), so the rate swap is d x (-r_USD) - mark-up per night."""
    from fxlab.data.rates import known_at
    bk = Book(m1, pip)
    days = pd.date_range(m1.index[0].floor("D"), m1.index[-1].floor("D"), freq="D")
    bk.rd = -known_at(days)["USD"]
    return bk


def last_bday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def metals(book, pair, start, end):
    rows = []
    gold = pair == "XAUUSD"
    for d in pd.bdate_range(start, end):
        p = prev_bday(d)
        if not gold:
            r = book.trade(local_ts(d, "Europe/London", 11, 45), local_ts(d, "Europe/London", 12, 15), -1)
            if r is not None:
                rows.append(rec("S1", d, r))
        else:
            for name, h0, m0, h1, m1 in (("G1", 14, 45, 15, 15), ("G2", 10, 15, 10, 45)):
                r = book.trade(local_ts(d, "Europe/London", h0, m0), local_ts(d, "Europe/London", h1, m1), -1)
                if r is not None:
                    rows.append(rec(name, d, r))
            i0, i3 = book.at(local_ts(d, "Europe/London", 15, 0)), book.at(local_ts(d, "Europe/London", 15, 3))
            if i0 >= 0 and i3 > i0:
                m0_, m3_ = 0.5 * (book.bo[i0] + book.ao[i0]), 0.5 * (book.bo[i3] + book.ao[i3])
                if np.isfinite(m0_) and np.isfinite(m3_) and m3_ != m0_:
                    side = 1.0 if m3_ > m0_ else -1.0
                    r = book.trade(local_ts(d, "Europe/London", 15, 3), local_ts(d, "Europe/London", 15, 15), side)
                    if r is not None:
                        rows.append(rec("G3", d, r, side=side))
        sfx = "" if gold else "_silver"
        # Asian session from 18:00 New York the evening before (Sunday evening for Mondays: the weekly open)
        r = book.trade(local_ts(d - pd.Timedelta(days=1), "America/New_York", 18, 0), local_ts(d, "Europe/London", 8, 0),
                       +1, tol_min=65)
        if r is not None:
            rows.append(rec("A1" + sfx, d, r))
        r = book.trade(local_ts(d, "Europe/London", 8, 0), local_ts(d, "America/New_York", 16, 55), -1, tol_min=10)
        if r is not None:
            rows.append(rec("A2" + sfx, d, r))
    for m in pd.period_range(start, end, freq="M"):
        if m.month not in (9, 11):
            continue
        prev = m - 1
        t_in = local_ts(last_bday(prev.year, prev.month), "America/New_York", 17, 5)
        t_out = local_ts(last_bday(m.year, m.month), "America/New_York", 16, 55)
        r = book.trade(t_in, t_out, +1, tol_min=4 * 1440)
        if r is not None:
            rows.append(rec("SEAS" + ("" if gold else "_silver"), m.start_time, r, month=int(m.month)))
    return rows


def pre_fomc(book, pair, start, end):
    """Diagnostics: 2-day pre-FOMC window, realised-decision direction (look-ahead) and long-foreign on all meetings."""
    base, quote = pair[:3], pair[3:]
    foreign = base if quote == "USD" else quote
    long_f = 1.0 if base == foreign else -1.0
    days = [pd.Timestamp(f"{y}-{x}") for y, s_ in FL.FOMC_FALLBACK.items() for x in s_.split()]
    rows = []
    for d in days:
        if not (pd.Timestamp(start) <= d <= pd.Timestamp(end)):
            continue
        p2 = prev_bday(prev_bday(d))
        t_in, t_out = local_ts(p2, "America/New_York", 14, 0), local_ts(d, "America/New_York", 13, 55)
        r = book.trade(t_in, t_out, long_f, tol_min=10)
        if r is not None:
            rows.append(rec("PF_all", d, r, ccy=foreign))
        ch = FOMC_CHANGES.get(str(d.date()))
        if ch is not None:
            r = book.trade(t_in, t_out, -ch * long_f, tol_min=10)        # hike: long USD = short foreign
            if r is not None:
                rows.append(rec("PF_change_lookahead", d, r, ccy=foreign, change=ch))
    return rows


def daily_closes(pair):
    from fxlab.orchestration.runners import cached_bars
    b = cached_bars(pair, "D1", "2007-01-01", DEV_END)
    return pd.DataFrame({"bid_c": b["bid_c"], "ask_c": b["ask_c"]}, index=b.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default=DEV_END)
    ap.add_argument("--out", default=os.path.join(FL.ROOT, "results", "reports", "lit2"))
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.pair not in METALS and a.pair not in FX:
        json.dump({"pair": a.pair, "skipped": True}, open(os.path.join(a.out, f"{a.pair}.json"), "w"))
        print(f"{a.pair}: skipped")
        return
    from fxlab.data.instruments import get as instrument
    from fxlab.data.store import load_m1
    meta, rows, errors = {"pair": a.pair}, [], {}
    m1 = load_m1(a.pair, a.start, a.end, columns=["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"])
    start, end = str(m1.index[0].date()), str(min(m1.index[-1], pd.Timestamp(DEV_END, tz="UTC")).date())
    meta.update(start=start, end=end)
    pip = instrument(a.pair).pip
    try:
        if a.pair in METALS:
            rows += metals(metal_book(m1, pip, a.pair), a.pair, start, end)
        else:
            rows += pre_fomc(Book(m1, pip, a.pair), a.pair, start, end)
            dc = daily_closes(a.pair)
            dc.to_csv(os.path.join(a.out, f"daily_{FX[a.pair]}.csv"))
            meta["daily_rows"] = int(len(dc))
    except Exception as ex:                                             # noqa: BLE001
        errors["lab"] = repr(ex)[:500]
        print(f"{a.pair}: ERROR {ex}", flush=True)
    meta["errors"] = errors
    meta["seconds"] = round(time.time() - t0, 1)
    R = pd.DataFrame(rows)
    if len(R):
        R["pair"] = a.pair
        R.to_parquet(os.path.join(a.out, f"{a.pair}.parquet"))
        print(R.groupby("test")[["net", "gross"]].agg(["count", "mean"]).to_string())
    json.dump(meta, open(os.path.join(a.out, f"{a.pair}.json"), "w"), indent=1, default=str)
    print(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
