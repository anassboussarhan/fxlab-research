"""Edges batch (rules in docs/edges_batch_preregistration.md, fixed before any data was read). This file holds the
calendar tests that need only M1 prices: E6 gold options expiry and E9 MSCI review days. The other tests are added
next to them as their data arrives; nothing here is tuned after a result.

E6: expiry day = the 4th-last US business day of each month (COMEX gold options rule; US federal holidays plus
    Good Friday stand in for the exchange calendar). Windows on XAUUSD: (a) 13:30-17:30 UTC on the expiry day;
    (b) 13:30 UTC the business day before -> 13:30 UTC on the expiry day.
E9: MSCI review day = the last business day of February, May, August and November. Windows on USDJPY and EURJPY:
    (a) 14:00-21:00 UTC that day; (b) 00:00-03:00 UTC the next business day.
For every window: the same window on all other business days is the control. Direction = sign of the train
(2007-2018) event-minus-control gross difference. Reported: net mean in that direction on train and test, one-sided
t, and the event-minus-control difference. The BH step over the whole batch is done in the aggregate.
Usage: python scripts/edges_lab.py --out DIR [--tests E6 E9]"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import untested_intraday_lab as U  # noqa: E402

TRAIN_END = pd.Timestamp("2018-12-31")
END = pd.Timestamp("2025-12-31")


def us_bdays(start="2006-12-01", end="2026-01-31"):
    hol = set(USFederalHolidayCalendar().holidays(start, end))
    gf = set(GoodFriday.dates(start, end))
    return pd.DatetimeIndex([d for d in pd.bdate_range(start, end) if d not in hol and d not in gf])


def gold_expiry_days(bd):
    s = pd.Series(bd, index=bd)
    return pd.DatetimeIndex(s.groupby([bd.year, bd.month]).apply(lambda x: x.iloc[-4] if len(x) >= 4 else pd.NaT).dropna())


def msci_days(bd):
    s = pd.Series(bd, index=bd)
    last = s.groupby([bd.year, bd.month]).max()
    return pd.DatetimeIndex([d for d in last if d.month in (2, 5, 8, 11)])


def utc(d, h, m=0):
    return pd.Timestamp(d.date()).tz_localize("UTC") + pd.Timedelta(hours=h, minutes=m)


def window_returns(bk, days, t_in, t_out, tol=5):
    """Long trade per day from t_in(d) to t_out(d): DataFrame[net, gross] (bid/ask fills, costs, swap)."""
    rows = {}
    for d in days:
        if d > END:
            continue
        r = bk.trade(t_in(d), t_out(d), +1, tol_min=tol)
        if r is not None:
            rows[d] = {"net": r[0], "gross": r[1]}
    f = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    if f.empty:
        f = pd.DataFrame(columns=["net", "gross"], index=pd.DatetimeIndex([]))
    f.index = pd.DatetimeIndex(f.index)
    return f


def one_sided(x):
    x = pd.Series(x).dropna()
    if len(x) < 8 or x.std() == 0:
        return np.nan, np.nan
    t = float(x.mean() / x.std() * np.sqrt(len(x)))
    return t, float(1 - st.t.cdf(t, len(x) - 1))


def evaluate(name, sym, ev, ctl):
    """ev / ctl: long-trade frames on event and control days. Direction from train gross difference."""
    out = {"test": name, "instrument": sym, "n_events": int(len(ev))}
    if ev.empty or ctl.empty:
        out["status"] = "no trades (data)"
        return out
    tr_e, tr_c = ev[ev.index <= TRAIN_END], ctl[ctl.index <= TRAIN_END]
    if len(tr_e) < 8:
        out["status"] = "too few train events"
        return out
    diff = tr_e.gross.mean() - tr_c.gross.mean()
    sgn = 1 if diff > 0 else -1
    out["direction"] = "long" if sgn > 0 else "short"
    out["train_diff_bp"] = diff * 1e4
    out["train_diff_t"] = float(st.ttest_ind(tr_e.gross, tr_c.gross, equal_var=False).statistic)
    for per, m_e, m_c in (("train", ev.index <= TRAIN_END, ctl.index <= TRAIN_END),
                          ("test", ev.index > TRAIN_END, ctl.index > TRAIN_END)):
        e, c = ev[m_e], ctl[m_c]
        if len(e) == 0:
            out[f"{per}_n"] = 0
            continue
        # a short pays the spread too: net of a short = -gross - (gross - net) = -2 gross + net
        net = e.net if sgn > 0 else (-2 * e.gross + e.net)
        t, p = one_sided(net)
        out.update({f"{per}_n": int(len(e)), f"{per}_net_bp": net.mean() * 1e4, f"{per}_t": t, f"{per}_p": p,
                    f"{per}_diff_bp": (e.gross.mean() - c.gross.mean()) * sgn * 1e4,
                    f"{per}_win": float((net > 0).mean())})
    return out


def run_e6(rows, frames):
    bk = U.book("XAUUSD")
    if bk is None:
        rows.append({"test": "E6", "status": "no XAUUSD data"})
        return
    bd = us_bdays()
    ex = gold_expiry_days(bd)
    prev = {d: bd[bd.get_loc(d) - 1] for d in ex}
    others = bd.difference(ex)
    wins = {"E6a expiry 13:30-17:30": (lambda d: utc(d, 13, 30), lambda d: utc(d, 17, 30), ex, others),
            "E6b day before 13:30 -> expiry 13:30": (None, None, ex, others)}
    for name, (fi, fo, evd, ctd) in wins.items():
        if fi is None:
            pmap = {d: bd[bd.get_loc(d) - 1] for d in bd[1:]}
            fi = lambda d: utc(pmap[d], 13, 30)                              # noqa: E731
            fo = lambda d: utc(d, 13, 30)                                    # noqa: E731
            evd, ctd = ex[ex > bd[0]], others[others > bd[0]]
        ev, ctl = window_returns(bk, evd, fi, fo), window_returns(bk, ctd, fi, fo)
        frames[f"{name}|XAUUSD"] = ev
        rows.append(evaluate(name, "XAUUSD", ev, ctl))
        print(rows[-1], flush=True)
    del prev


def run_e9(rows, frames):
    bd = us_bdays()
    ms = msci_days(bd)
    nxt = {d: bd[bd.get_loc(d) + 1] for d in bd[:-1]}
    others = bd[:-1].difference(ms)
    for sym in ("USDJPY", "EURJPY"):
        bk = U.book(sym)
        if bk is None:
            rows.append({"test": "E9", "instrument": sym, "status": "no data"})
            continue
        # E9b: the 2007-2018 files have no 00:00-01:00 UTC bars, so the entry is the first quote within 65 minutes
        # of 00:00 (01:00 on train); a data constraint, not a rule change.
        for name, fi, fo, tol in (("E9a review day 14:00-21:00", lambda d: utc(d, 14), lambda d: utc(d, 21), 5),
                                  ("E9b next day 00:00-03:00", lambda d: utc(nxt[d], 0), lambda d: utc(nxt[d], 3), 65)):
            ev, ctl = window_returns(bk, ms, fi, fo, tol), window_returns(bk, others, fi, fo, tol)
            frames[f"{name}|{sym}"] = ev
            rows.append(evaluate(name, sym, ev, ctl))
            print(rows[-1], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", nargs="+", default=["E6", "E9"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, frames = [], {}
    if "E6" in a.tests:
        run_e6(rows, frames)
    if "E9" in a.tests:
        run_e9(rows, frames)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "edges_calendar.csv"), index=False)
    for k, f in frames.items():
        f.to_csv(os.path.join(a.out, "events_" + k.split(" ")[0] + "_" + k.split("|")[1] + ".csv"))
    json.dump(rows, open(os.path.join(a.out, "edges_calendar.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
