"""Sixth literature round: the most-watched YouTube day-trading rules that have a written, mechanical form. Each is
copied from its source and fixed before any result; nothing is tuned. M1 bid/ask fills and the flows-lab costs
(the engines of scripts/lit5_lab.py: market entries, stops and time exits cross the spread plus 25% of it and 0.1
pip, 0.35 bp commission a side, swap over 17:00 New York). Development data only (to 2025-12-31).

FIXED BEFORE ANY RESULT (times New York):
  CRT1 Data Trader, "The best 5 minute scalping strategy ever" (YouTube; 23 trades shown): the range of the day's
     first 4-hour candle (01:00-05:00, the first 4-hour bar of the calendar day in TradingView's FX alignment);
     after 05:00 a 5-minute close outside it, then a 5-minute close back inside the same day -> enter at the next
     open back into the range, stop at the breakout candle's high (short) / low (long), target 2R, out at 17:00;
     the first signal of the day only. EURUSD, GBPUSD, USDJPY, XAUUSD.
  AS1 The "5-rule SMC" / Judas-swing model (YouTube, 230 trades shown, 33% wins): Asia range 20:00-00:00; in the
     London kill zone (02:00-05:00) a 5-minute candle trades beyond the Asia high (low) and a later one closes back
     inside -> short (long) at the next open, stop at the sweep extreme, target the other side of the Asia range,
     out at 12:00. Same symbols.
  TS1 Turtle Soup (Connors-Raschke, as coded in the quant-study notes): daily bars (17:00 roll); the day's low
     takes the lowest low of the previous 20 days and the close is back above it -> long at the next open, stop
     1 x ATR(14), target 2 x the stop, at most 20 days; shorts mirrored. Same symbols.
  ORB5 Zarattini & Aziz (SSRN 2023, "Can day trading really be profitable?"): the 09:30-09:35 candle; up candle ->
     long at 09:35, down -> short, flat if open = close; stop at the candle's low (high), target 10R, out at
     16:00. US500 and NAS100 CFDs (2019-2025), with costs (the paper used none).
  ST1 Supertrend (ATR 10, factor 3, the TradingView default; the HackerNoon test and the viral YouTube videos):
     always in the market on H1 and on H4 bars, reversing at the next bar's open after the trend flips.
     EURUSD, GBPUSD, USDJPY, XAUUSD.
PRE-REGISTERED TESTS (one-sided, mean net per trade > 0); Benjamini-Hochberg over all of them.
Usage: python scripts/lit6_lab.py --out DIR [--only crt,as,ts,orb,st] [--symbols ...]"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flows_lab as FL  # noqa: E402
import lit5_lab as L5  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

NY = "America/New_York"
FXG = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")
IDX = ("USA500IDXUSD", "USATECHIDXUSD")
DAY = pd.Timedelta(days=1)


def bars(bk, rule):
    f = pd.DataFrame({"o": bk.mo, "h": bk.mh, "l": bk.ml, "c": bk.mc, "k": np.arange(len(bk.idx))}, index=bk.idx)
    B = f.resample(rule).agg({"o": "first", "h": "max", "l": "min", "c": "last", "k": "first"}).dropna()
    B["k"] = B["k"].astype(int)
    return B


def days(bk):
    a, b = bk.idx[0].normalize().tz_localize(None), bk.idx[-1].normalize().tz_localize(None)
    return pd.bdate_range(a + DAY, b - DAY)


def seg5(B5, s, e):
    """5-minute bars starting in [s, e - 5 min], as (times, highs, lows, closes) arrays (binary search, not masks)."""
    a = int(B5.index.searchsorted(s))
    b = int(B5.index.searchsorted(e - pd.Timedelta(minutes=5), side="right"))
    return B5.index[a:b], B5.h.to_numpy()[a:b], B5.l.to_numpy()[a:b], B5.c.to_numpy()[a:b]


def window_idx(bk, d0, hm0, d1, hm1):
    t0, t1 = FL.local_ts(d0, NY, *hm0), FL.local_ts(d1, NY, *hm1)
    return int(bk.idx.searchsorted(t0)), int(bk.idx.searchsorted(t1)), t0, t1


# ----------------------------------------------------------------------------------------------- CRT1
def crt(bk, B5):
    out = {}
    for d in days(bk):
        i0, i1, t0, t1 = window_idx(bk, d, (1, 0), d, (5, 0))
        if i1 - i0 < 200:
            continue
        hi, lo = float(np.max(bk.mh[i0:i1])), float(np.min(bk.ml[i0:i1]))
        s, e = FL.local_ts(d, NY, 5, 0), FL.local_ts(d, NY, 17, 0)
        j = bk.at(e, 10)
        if j < 0:
            continue
        T5, H5, L5_, C5 = seg5(B5, s, e)
        brk = None
        for t, rh, rl, rc in zip(T5, H5, L5_, C5):
            if brk is None:
                if rc > hi:
                    brk = (-1, rh)
                elif rc < lo:
                    brk = (+1, rl)
            elif lo <= rc <= hi:
                d_, ext = brk
                k = bk.at(t + pd.Timedelta(minutes=5), 3)
                if k < 0 or k >= j:
                    break
                E0 = bk.ao[k] if d_ > 0 else bk.bo[k]
                sd = (E0 - ext) * d_
                if sd > 0:
                    out[d] = L5.market_trade(bk, k, j, d_, sd, 2 * sd)
                break
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- AS1
def asia_sweep(bk, B5):
    out = {}
    for d in days(bk):
        i0, i1, _, _ = window_idx(bk, d - DAY, (20, 0), d, (0, 0))
        if i1 - i0 < 160:
            continue
        hi, lo = float(np.max(bk.mh[i0:i1])), float(np.min(bk.ml[i0:i1]))
        s, e, x = FL.local_ts(d, NY, 2, 0), FL.local_ts(d, NY, 5, 0), FL.local_ts(d, NY, 12, 0)
        j = bk.at(x, 10)
        if j < 0:
            continue
        T5, H5, L5_, C5 = seg5(B5, s, e)
        side, ext = 0, None
        for t, rh, rl, rc in zip(T5, H5, L5_, C5):
            if side == 0:
                if rh > hi:
                    side, ext = -1, rh
                elif rl < lo:
                    side, ext = +1, rl
                if side == 0:
                    continue
            else:
                ext = max(ext, rh) if side < 0 else min(ext, rl)
            if lo <= rc <= hi:
                k = bk.at(t + pd.Timedelta(minutes=5), 3)
                if k < 0 or k >= j:
                    break
                E0 = bk.ao[k] if side > 0 else bk.bo[k]
                sd, td = (E0 - ext) * side, ((lo if side < 0 else hi) - E0) * side
                if sd > 0 and td > 0:
                    out[d] = L5.market_trade(bk, k, j, side, sd, td)
                break
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- TS1
def turtle_soup(bk, n=20, hold=20):
    from fxlab.validation.metrics import trading_day
    td = trading_day(bk.idx)
    f = pd.DataFrame({"h": bk.mh, "l": bk.ml, "c": bk.mc, "k": np.arange(len(bk.idx))}, index=bk.idx)
    g = f.groupby(td)
    D = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "k0": g.k.first(), "k1": g.k.last()})
    D = D[(D.k1 - D.k0) > 600]
    pc = D.c.shift(1)
    tr = pd.concat([D.h - D.l, (D.h - pc).abs(), (D.l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    lo_n, hi_n = D.l.shift(1).rolling(n).min(), D.h.shift(1).rolling(n).max()
    out = {}
    idx = list(D.index)
    busy_until = -1
    for p in range(n + 15, len(idx) - 1):
        t = idx[p]
        a = float(atr.iloc[p])
        if not np.isfinite(a) or a <= 0 or p <= busy_until:
            continue
        d_ = 0
        if D.l.iloc[p] < lo_n.iloc[p] and D.c.iloc[p] > lo_n.iloc[p]:
            d_ = +1
        elif D.h.iloc[p] > hi_n.iloc[p] and D.c.iloc[p] < hi_n.iloc[p]:
            d_ = -1
        if d_ == 0:
            continue
        k = int(D.k0.iloc[p + 1])
        q = min(p + hold, len(idx) - 1)
        j = int(D.k1.iloc[q])
        out[bk.idx[k]] = L5.market_trade(bk, k, j, d_, a, 2 * a)
        busy_until = q                                     # one trade at a time
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- ORB5
def orb5(bk):
    out = {}
    for d in days(bk):
        i0, i1, t0, t1 = window_idx(bk, d, (9, 30), d, (9, 35))
        j = bk.at(FL.local_ts(d, NY, 16, 0), 5)
        if i1 - i0 < 4 or j < 0 or (bk.idx[i0] - t0) > pd.Timedelta(minutes=1) or i1 >= j:
            continue
        o, c = bk.mo[i0], bk.mc[i1 - 1]
        if o == c:
            continue
        d_ = 1 if c > o else -1
        ext = float(np.min(bk.ml[i0:i1])) if d_ > 0 else float(np.max(bk.mh[i0:i1]))
        k = i1
        E0 = bk.ao[k] if d_ > 0 else bk.bo[k]
        sd = (E0 - ext) * d_
        if sd > 0:
            out[d] = L5.market_trade(bk, k, j, d_, sd, 10 * sd)
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- ST1
def supertrend_dir(B, n=10, f=3.0):
    h, l, c = B.h.to_numpy(), B.l.to_numpy(), B.c.to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()   # RMA, as TradingView's ta.atr
    mid = (h + l) / 2
    ub, lb = mid + f * atr, mid - f * atr
    fu, fl = ub.copy(), lb.copy()
    d = np.zeros(len(c), int)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] <= 0:
            d[i] = 1 if c[i] > fu[i - 1] else -1
        else:
            d[i] = -1 if c[i] < fl[i - 1] else 1
    d[:n] = 0
    return d


def supertrend(bk, rule):
    B = bars(bk, rule)
    d = supertrend_dir(B)
    K = B.k.to_numpy()
    out = {}
    pos, k_in = 0, None
    for i in range(1, len(B) - 1):
        if d[i] != 0 and d[i] != pos:
            k = int(K[i + 1])                              # next bar's open
            if pos != 0:
                E = bk.ao[k_in] + L5.slip(bk, k_in) if pos > 0 else bk.bo[k_in] - L5.slip(bk, k_in)
                X = bk.bo[k] - L5.slip(bk, k) if pos > 0 else bk.ao[k] + L5.slip(bk, k)
                out[bk.idx[k]] = L5.settle(bk, k_in, k, pos, E, X, bk.mo[k_in], bk.mo[k])
            pos, k_in = int(d[i]), k
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- driver
def run(res, only, symbols):
    for sym in symbols:
        bk = L5.book(sym)
        if bk is None:
            print(sym, "no data", flush=True)
            continue
        if sym in FXG:
            B5 = bars(bk, "5min") if ("crt" in only or "as" in only) else None
            if "crt" in only:
                res[f"CRT1 {sym} first 4-hour range re-entry, 2R"] = (crt(bk, B5), 250)
            if "as" in only:
                res[f"AS1 {sym} Asia sweep in the London kill zone"] = (asia_sweep(bk, B5), 250)
            if "ts" in only:
                res[f"TS1 {sym} Turtle Soup, 20 days"] = (turtle_soup(bk), 12)
            if "st" in only:
                res[f"ST1 {sym} Supertrend (10, 3), H1"] = (supertrend(bk, "1h"), 100)
                res[f"ST1 {sym} Supertrend (10, 3), H4"] = (supertrend(bk, "4h"), 40)
        if sym in IDX and "orb" in only:
            res[f"ORB5 {sym} 5-minute opening range, 10R"] = (orb5(bk), 252)
        print(sym, "done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="crt,as,ts,orb,st")
    ap.add_argument("--symbols", default=",".join(FXG + IDX))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    try:
        run(res, a.only.split(","), a.symbols.split(","))
    except Exception:                                                       # noqa: BLE001
        import traceback
        res["error"] = traceback.format_exc()[-2000:]
        print(res["error"], flush=True)
        print("::error title=lit6::" + res["error"].replace("\n", " | ")[-900:], flush=True)
    T, out = U.tabulate(res, a.out)
    from cross_market_ta_lab import bh_q
    if len(T):
        T["q_bh"] = bh_q(T["p_one_sided"].to_numpy())
        for k, v in res.items():
            if isinstance(v, tuple):
                x = v[0].net.dropna()
                T.loc[T.test == k, "profit_factor"] = float(x[x > 0].sum() / -x[x < 0].sum()) if (x < 0).any() else np.nan
    out["_coverage_m1_bars_per_year"] = U.COVERAGE
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    if len(T):
        print(T.drop(columns=["first", "last"], errors="ignore").round(3).to_string())


if __name__ == "__main__":
    main()
