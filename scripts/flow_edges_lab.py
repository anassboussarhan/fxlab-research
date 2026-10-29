"""Order-flow edges at known times: the only kind of edge that has survived costs in this project (the Gotobi /
Tokyo-fix flow in USD/JPY and EUR/JPY). This lab extends it to the other yen crosses, asks where within the
calendar the flow is strongest, tests a passive entry on the original rule, and tests a second documented flow:
equity-hedge rebalancing into the London 4 pm month-end fix (Melvin & Prins 2015). Rules fixed before any result;
nothing tuned. M1 bid/ask bars, costs as in the flows lab. Development data only (to 2025-12-31).

PRE-REGISTERED TESTS (one-sided, mean net > 0 unless stated):
  Tokyo fix (the I3 rule of scripts/untested_intraday_lab.py: long 23:00 Tokyo the evening before -> 09:53 Tokyo on
  Gotobi days; exits accept the 01:00 UTC bar, because the 2007-2018 mirror has no bars from 00:00 to 01:00 UTC):
    T1 AUDJPY, T2 GBPJPY: the I3 trade with no stop (primary); T1s / T2s: with a stop at the mean high-low range of
       the same window on the previous 20 sampled days (the 2019-2025 files hold only these windows, so the I3
       stop of 0.5 x the 20-day daily range cannot be computed; the window range is about half the daily range).
       Controls: the same trade on a third of the other Tue-Fri Tokyo business days (T1c, T2c), and the Welch t of
       Gotobi minus control.
    T3 yen basket: on each Gotobi day, the average no-stop I3 return of USDJPY, EURJPY, AUDJPY and GBPJPY (those
       with data that day).
    T4 USDJPY, I3 exactly (stop 0.5 x the 20-day daily range): (a) on Gotobi days that are the 25th or the month's
       last business day; (b) the Welch t of those minus the other Gotobi days > 0.
    T5 USDJPY, I3 exactly: (a) Gotobi days in March, June, September and December; (b) the Welch t against the
       other months > 0.
    T6 USDJPY day before Gotobi: the I3 trade ending at 09:53 Tokyo on the business day before each Gotobi day
       (when that day is not itself a Gotobi day and not a Monday).
    T7 USDJPY I3 with a passive entry: a buy limit at the bid at 23:00 Tokyo, valid 30 minutes, filled when the ask
       reaches it (no slippage); stop 0.5 x the 20-day range from the fill; exit 09:53 at market.
  London month-end fix (equity-hedge rebalancing): on the last weekday of the month, with d = the month-to-date
    return of the S&P 500 minus that of the foreign index up to the previous trading day's close (Euro Stoxx 50,
    FTSE 100, Nikkei 225, S&P/ASX 200, S&P/TSX, SMI, NZX 50 from Yahoo), hold the foreign currency against the USD
    in the direction of sign(d) (US equities up relative to foreign -> foreign holders of US equities sell USD to
    re-hedge -> USD weak):
    M1 10:00 -> 16:00 London on the fix day; M2 16:00 London the weekday before -> 16:00 London on the fix day;
    M3 M1 only when |d| > 2%. Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD; each pair and the
    pooled average per month-end.
Usage: python scripts/flow_edges_lab.py --out DIR [--only tokyo,monthend]"""
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

U.PIPS.update({"AUDJPY": 0.01, "GBPJPY": 0.01, "GBPUSD": 1e-4, "USDCAD": 1e-4, "USDCHF": 1e-4, "NZDUSD": 1e-4})
EQ = {"EURUSD": ("^STOXX50E", 1), "GBPUSD": ("^FTSE", 1), "USDJPY": ("^N225", -1), "AUDUSD": ("^AXJO", 1),
      "USDCAD": ("^GSPTSE", -1), "USDCHF": ("^SSMI", -1), "NZDUSD": ("^NZ50", 1)}
US = "^GSPC"
DAY = pd.Timedelta(days=1)


# ----------------------------------------------------------------------------------------------- Tokyo
def calendar(bk):
    start, end = str(bk.idx[0].date()), str(bk.idx[-1].date())
    hol = FL.jp_holidays(range(bk.idx[0].year - 1, bk.idx[-1].year + 2))
    G = FL.gotobi_days(start, end, hol)
    Gs = set(G)
    C = [d for d in pd.bdate_range(start, end) if d not in Gs and d not in hol and d.weekday() != 0]
    return G, C[::3], hol


def window(bk, d):
    t0, t1 = U.jst(d - DAY, 23, 0), U.jst(d, 9, 53)
    i, j = bk.at(t0, 10), bk.at(t1, 65)
    return (i, j) if (i >= 0 and j > i) else (-1, -1)


def open_loss(bk, i, j, sl_dist, net):
    """Worst open loss of a long from minute i (entry at the ask plus slippage) until its exit: the lowest bid
    before the stop (or the time exit), as a return; a stopped trade's open loss is its realised loss."""
    si = bk.sf * (bk.ao[i] - bk.bo[i]) + bk.sp * bk.pip
    E = bk.ao[i] + si
    if np.isfinite(sl_dist):
        hit = np.flatnonzero(bk.bl[i:j] <= E - sl_dist)
        if len(hit):
            return min(net, 0.0)
    return float(min(np.min(bk.bl[i:j]) / E - 1.0, net, 0.0))


def overnight(bk, days, stop="none", rng=None):
    """I3 trade on each day: stop "none", "daily" (0.5 x the 20-day daily range, the I3 rule) or "window" (the mean
    range of the same window on the previous 20 sampled days). Returns DataFrame[net, gross, mae]."""
    out = {}
    mae = {}
    bk._hl()
    hist = []
    for d in sorted(days):
        i, j = window(bk, d)
        if i < 0:
            continue
        sl_dist = np.inf
        if stop == "none":
            r = bk.trade_ij(i, j, +1, None, None)
        elif stop == "daily":
            from fxlab.validation.metrics import trading_day
            td = trading_day(pd.DatetimeIndex([bk.idx[i]]))[0]
            rg = float(rng.get(td, np.nan))
            if not np.isfinite(rg) or rg <= 0:
                continue
            sl_dist = 0.5 * rg
            r = bk.bracket(i, j, +1, 1e9, sl_dist)
        else:
            w = float(np.max(bk.mh[i:j + 1]) - np.min(bk.ml[i:j + 1]))
            ok = U.complete(bk, i, j, 0.85)
            prev = hist[-20:]
            if ok:
                hist.append(w)
            if len(prev) < 10 or not ok:
                continue
            sl_dist = float(np.mean(prev))
            r = bk.bracket(i, j, +1, 1e9, sl_dist)
        if r is not None:
            out[d] = U.rec(r)
            mae[d] = open_loss(bk, i, j, sl_dist, out[d][0])
    f = U.frame(out)
    f["mae"] = pd.Series(mae).reindex(f.index)
    return f


def welch(a, b):
    if len(a) > 10 and len(b) > 10:
        return float(st.ttest_ind(a.net, b.net, equal_var=False).statistic)
    return np.nan


def tokyo(res):
    import passive_entry_lab as PE
    books = {}
    for s in ("AUDJPY", "GBPJPY", "EURJPY"):
        bk = U.book(s)
        if bk is not None:
            books[s] = bk
    m1, bk_uj, rc, swl, sws, pip = PE.load_book("USDJPY")
    books["USDJPY"] = bk_uj
    # T1 / T2: the other yen crosses
    for tag, s in (("T1", "AUDJPY"), ("T2", "GBPJPY")):
        bk = books.get(s)
        if bk is None:
            res[f"{tag} {s} error"] = "no data"
            continue
        G, C, _ = calendar(bk)
        ev, ctl = overnight(bk, G), overnight(bk, C)
        res[f"{tag} {s} Gotobi overnight, no stop"] = (ev, 62)
        res[f"{tag}c {s} control days, no stop"] = (ctl, 80)
        res[f"{tag} {s} Gotobi minus control (Welch t)"] = welch(ev, ctl)
        evs, ctls = overnight(bk, G, "window"), overnight(bk, C, "window")
        res[f"{tag}s {s} Gotobi overnight, window-range stop"] = (evs, 62)
        res[f"{tag}sc {s} control days, window-range stop"] = (ctls, 80)
        res[f"{tag}s {s} Gotobi minus control (Welch t)"] = welch(evs, ctls)
    # T3: basket of the four yen pairs (no stop)
    per, per_mae = {}, {}
    for s, bk in books.items():
        G, _, _ = calendar(bk)
        f = overnight(bk, G)
        per[s], per_mae[s] = f.net, f.mae
        res[f"T3 reference: {s} Gotobi overnight, no stop"] = (f, 62)
    B = pd.DataFrame(per)
    basket = B.mean(axis=1, skipna=True).dropna()
    bm = pd.DataFrame(per_mae).mean(axis=1, skipna=True).reindex(basket.index)   # simultaneous: mean of the pairs
    res["T3 yen basket Gotobi overnight, no stop"] = (pd.DataFrame({"net": basket, "gross": basket * np.nan,
                                                                     "mae": bm}), 62)
    res["T3 pairs per day (mean)"] = float(B.notna().sum(axis=1).mean())
    # T4 / T5 / T6 on USDJPY with the I3 rule
    bk = bk_uj
    G, C, hol = calendar(bk)
    rng = bk.avg_range()
    allg = overnight(bk, G, "daily", rng)
    big = [d for d in G if d.day == 25 or d == max(x for x in G if x.year == d.year and x.month == d.month)
           or (d.day in (23, 24) and not any(x.day == 25 for x in G if x.year == d.year and x.month == d.month))]
    bigs = set(big)
    a4 = allg[allg.index.isin(bigs)]
    b4 = allg[~allg.index.isin(bigs)]
    res["T4a USDJPY I3 on the 25th / month-end Gotobi days"] = (a4, 24)
    res["T4b USDJPY I3 25th / month-end minus other Gotobi days (Welch t)"] = welch(a4, b4)
    q = allg.index.month.isin([3, 6, 9, 12])
    res["T5a USDJPY I3 in quarter-end months"] = (allg[q], 18)
    res["T5b USDJPY I3 quarter-end months minus others (Welch t)"] = welch(allg[q], allg[~q])
    Gs = set(G)
    before = []
    for d in G:
        p = d - DAY
        while p.weekday() >= 5 or p in hol:
            p -= DAY
        if p not in Gs and p.weekday() != 0:
            before.append(p)
    res["T6 USDJPY I3 on the business day before Gotobi"] = (overnight(bk, before, "daily", rng), 62)
    # T7 passive entry on the I3 rule
    arr = tuple(x.astype(float) for x in (bk.bo, bk.ao, bk.bh, bk.bl, bk.ah, bk.al))
    from fxlab.validation.metrics import trading_day
    out, fills = {}, 0
    for d in G:
        i, j = window(bk, d)
        if i < 0:
            continue
        td = trading_day(pd.DatetimeIndex([bk.idx[i]]))[0]
        rg = float(rng.get(td, np.nan))
        if not np.isfinite(rg) or rg <= 0:
            continue
        w = bk.at(bk.idx[i] + pd.Timedelta(minutes=30), 30)
        w = w if w > i else i + 1
        net, gross, k, xf = PE.sim_passive(*arr, i, w, j, j, 1.0, 1e9, 0.5 * rg, bk.sf, bk.sp * bk.pip, bk.comm,
                                           rc, swl, sws, 0.0, False)
        if xf >= 0:
            fills += 1
            out[d] = (net, gross)
    res["T7 USDJPY I3 with a passive entry (limit at the bid, 30 min)"] = (U.frame(out), 62)
    res["T7 fill rate"] = fills / max(1, len(allg))
    res["T7 reference: USDJPY I3 market entry, same days"] = (allg, 62)


# ----------------------------------------------------------------------------------------------- month-end
def equity_closes():
    import yfinance as yf
    out = {}
    for t in [US] + [v[0] for v in EQ.values()]:
        d = yf.download(t, start="2006-06-01", end="2026-01-01", auto_adjust=False, progress=False)
        c = d["Close"]
        c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        c.index = pd.DatetimeIndex(c.index).tz_localize(None)
        out[t] = c.dropna()
        print(t, len(c), flush=True)
    return out


def mtd_diff(eq, foreign, fix_day):
    """MTD return of the S&P 500 minus the foreign index, from the previous month's last close to the last close
    strictly before the fix day."""
    def mtd(c):
        prev_m = c[c.index < fix_day.replace(day=1)]
        upto = c[c.index < fix_day]
        if not len(prev_m) or not len(upto) or upto.index[-1] < fix_day.replace(day=1):
            return np.nan
        return upto.iloc[-1] / prev_m.iloc[-1] - 1
    return mtd(eq[US]) - mtd(eq[foreign])


def monthend(res, eq=None):
    eq = eq if eq is not None else equity_closes()
    pooled = {k: {} for k in ("M1", "M2", "M3")}
    for pair, (idx_t, sgn) in EQ.items():
        bk = U.book(pair)
        if bk is None:
            res[f"M {pair} error"] = "no data"
            continue
        start, end = str(bk.idx[0].date()), str(bk.idx[-1].date())
        days = FL.month_end_days(start, end)
        rows = {"M1": {}, "M2": {}, "M3": {}}
        for d in days:
            dd = mtd_diff(eq, idx_t, pd.Timestamp(d))
            if not np.isfinite(dd) or dd == 0:
                continue
            side = sgn * (1 if dd > 0 else -1)                         # long the foreign currency when d > 0
            t_in, t_out = FL.local_ts(d, "Europe/London", 10, 0), FL.local_ts(d, "Europe/London", 16, 0)
            i, j = bk.at(t_in, 10), bk.at(t_out, 10)
            if i >= 0 and j > i:
                r = bk.trade_ij(i, j, side, t_in, t_out)
                if r is not None:
                    rows["M1"][d] = U.rec(r)
                    if abs(dd) > 0.02:
                        rows["M3"][d] = U.rec(r)
            p = pd.Timestamp(d) - pd.offsets.BDay(1)
            t_prev = FL.local_ts(p, "Europe/London", 16, 0)
            i2 = bk.at(t_prev, 10)
            if i2 >= 0 and j > i2:
                r = bk.trade_ij(i2, j, side, t_prev, t_out)
                if r is not None:
                    rows["M2"][d] = U.rec(r)
        for k, lab in (("M1", "10:00-16:00 London"), ("M2", "16:00 the day before -> 16:00"),
                       ("M3", "10:00-16:00, |d| > 2%")):
            f = U.frame(rows[k])
            res[f"{k} {pair} month-end fix rebalancing {lab}"] = (f, 12)
            for dt, v in f.net.items():
                pooled[k].setdefault(dt, []).append(v)
    for k in pooled:
        s = pd.Series({d: float(np.mean(v)) for d, v in pooled[k].items()}).sort_index()
        res[f"{k} pooled (mean over pairs per month-end)"] = (pd.DataFrame({"net": s, "gross": s * np.nan}), 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="tokyo,monthend")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for g, fn in (("tokyo", tokyo), ("monthend", monthend)):
        if g in a.only.split(","):
            try:
                fn(res)
            except Exception:                                               # noqa: BLE001
                import traceback
                res[f"{g} error"] = traceback.format_exc()[-2000:]
                print(res[f"{g} error"], flush=True)
            print(g, "done", flush=True)
    T, out = U.tabulate(res, a.out)
    from cross_market_ta_lab import bh_q
    if len(T):
        prim = ~T.test.str.contains("control|reference|pooled", case=False)
        T["q_bh_primary"] = np.nan
        T.loc[prim, "q_bh_primary"] = bh_q(T.loc[prim, "p_one_sided"].to_numpy())
    out["_coverage_m1_bars_per_year"] = U.COVERAGE
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    if len(T):
        print(T.drop(columns=["first", "last"]).round(3).to_string())
    for k, v in out.items():
        if not isinstance(v, dict) or k.startswith("_"):
            print(k, v)


if __name__ == "__main__":
    main()
