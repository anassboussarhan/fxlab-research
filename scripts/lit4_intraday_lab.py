"""Fourth literature round (blogs, forums, commercial systems) on M1 bid/ask bars. Each rule is copied from its
published source and fixed before any result; nothing is tuned. Costs as in the flows lab (the stop and limit fills
of Book.bracket; crossing the quoted spread plus 25% of it and 0.1 pip slippage a side; 0.35 bp commission a side;
swap on positions held over 17:00 New York). Development data only (to 2025-12-31).

PRE-REGISTERED TESTS (one-sided, mean net per trade > 0; Benjamini-Hochberg over the primary tests):
  L1 Gold London breakout, the rule of MQL5 blog post 777540 (MT5 real ticks 2023-2026, profit factor 1.38): server
     time = New York + 7 h. Asian range = bid high and low of 03:00-08:00 server; the day is skipped if the range is
     under 0.30% or over 1.60% of price. At 08:00 a buy stop at the high + 0.01% of price (filled when the ask reaches
     it) and a sell stop at the low - 0.01% (filled when the bid reaches it), one cancelling the other; stop at the
     opposite side of the range, capped at 0.6% of the entry; target 2.5R; stop moved to break-even once the trade
     is 1.5R in profit; anything open is closed at 21:00 server; one trade a day. XAUUSD. Primary: all trades;
     long and short reported separately (information).
  L2 Larry Williams' Smash Day as tested by Rogue Quant (futures 2014-2024): daily bars of FX trading days (17:00
     New York); buy setup when the close is below the lowest low of the 3 previous days, sell setup when it is above
     their highest high; on the next day only, a buy stop at the setup day's high (sell stop at its low); stop and
     reverse on the opposite signal, no stop loss. EURUSD both sides (the article's best currency); XAUUSD long
     only (the article: gold profitable only long), a long closed at the next sell trigger.
  L3 The commercial Gotobi EA of MQL5 blog post 777443 ("Nakane", vendor backtest 2015-2026 profit factor 1.24): on
     Gotobi days (5th, 10th, ..., month end; the previous business day when that is a weekend or a Japanese
     holiday), long 07:30 -> 09:55 Tokyo, stop 2.5 x ATR(14) of H1 bars, no target; no trade when the hour before
     entry has a high-low range above 2 x ATR(H1). USDJPY and EURJPY. References: the same trade without the filter
     and the stop; the EA's short leg 09:55 -> 10:30 Tokyo (tested before as I1).
Usage: python scripts/lit4_intraday_lab.py --out DIR [--only lb,smash,ea]"""
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
import untested_intraday_lab as U  # noqa: E402

NY = "America/New_York"
DAY = pd.Timedelta(days=1)
U.PIPS.update({"EURUSD": 1e-4, "XAUUSD": 0.01, "USDJPY": 0.01, "EURJPY": 0.01})


def slip(bk, k):
    return bk.sf * (bk.ao[k] - bk.bo[k]) + bk.sp * bk.pip


def swap_ret(bk, i, j, d, mid):
    n, flat, rate = bk.swaps(i, j, d, mid)
    return rate if np.isfinite(rate) else flat


# ----------------------------------------------------------------------------------------------- L1 London breakout
def lb_trade(bk, i1, j, hi, lo, buf_frac=1e-4, cap=0.006, tp_r=2.5, be_r=1.5):
    """One London-breakout day from bar i1 (08:00 server) to the time exit at bar j (21:00 server), range [lo, hi]
    on the bid. Returns (net, gross, side, r_multiple) or None when neither stop order fills."""
    ref = 0.5 * (bk.bo[i1] + bk.ao[i1])
    L, S = hi + buf_frac * ref, lo - buf_frac * ref
    up = np.flatnonzero(bk.ah[i1:j] >= L)
    dn = np.flatnonzero(bk.bl[i1:j] <= S)
    ku = int(up[0]) + i1 if len(up) else None
    kd = int(dn[0]) + i1 if len(dn) else None
    if ku is None and kd is None:
        return None
    if ku is not None and (kd is None or ku < kd or (ku == kd and L - bk.ao[ku] <= bk.bo[kd] - S)):
        d, k = +1, ku
        E0 = max(L, bk.ao[k])
        E = E0 + slip(bk, k)
        stop = max(lo, E * (1 - cap))
        R = E - stop
        tp = E + tp_r * R
    else:
        d, k = -1, kd
        E0 = min(S, bk.bo[k])
        E = E0 - slip(bk, k)
        stop = min(hi, E * (1 + cap))
        R = stop - E
        tp = E - tp_r * R
    if R <= 0:
        return None
    px, m, at_tp = None, None, False
    # entry bar: only the stop can be hit (the order of prices inside the minute is unknown; the conservative case)
    if (d > 0 and bk.bl[k] <= stop) or (d < 0 and bk.ah[k] >= stop):
        px, m = stop - d * slip(bk, k), k
    be = False
    for q in range(k + 1, j):
        if px is not None:
            break
        if d > 0:
            if bk.bl[q] <= stop:
                px, m = min(stop, bk.bo[q]) - slip(bk, q), q
            elif bk.bh[q] >= tp:
                px, m, at_tp = tp, q, True
            elif not be and bk.bh[q] >= E + be_r * R:
                stop, be = E, True
        else:
            if bk.ah[q] >= stop:
                px, m = max(stop, bk.ao[q]) + slip(bk, q), q
            elif bk.al[q] <= tp:
                px, m, at_tp = tp, q, True
            elif not be and bk.al[q] <= E - be_r * R:
                stop, be = E, True
    if px is None:
        m = j
        px = bk.bo[j] - slip(bk, j) if d > 0 else bk.ao[j] + slip(bk, j)
    net = (px / E - 1.0) if d > 0 else (1.0 - px / E)
    net -= 2 * bk.comm
    net += swap_ret(bk, k, m, d, E)
    half_k = 0.5 * (bk.ao[k] - bk.bo[k])
    half_m = 0.5 * (bk.ao[m] - bk.bo[m])
    e_mid = E0 - d * half_k                                     # the trigger price on the mid
    x_mid = tp + d * half_m if at_tp else px + d * slip(bk, m) + d * half_m
    gross = d * (x_mid / e_mid - 1.0)
    return float(net), float(gross), d, float(net / (R / E))


def london_breakout(bk, lo_frac=0.003, hi_frac=0.016):
    bk._hl()
    out, side, rm = {}, {}, {}
    start, end = bk.idx[0].normalize().tz_localize(None), bk.idx[-1].normalize().tz_localize(None)
    for d in pd.bdate_range(start + DAY, end - DAY):
        t_a, t_e, t_x = (FL.local_ts(d - DAY, NY, 20, 0), FL.local_ts(d, NY, 1, 0), FL.local_ts(d, NY, 14, 0))
        i0, i1 = int(bk.idx.searchsorted(t_a)), int(bk.idx.searchsorted(t_e))
        j = bk.at(t_x, 10)
        if j < 0 or i1 >= j or i1 - i0 < 240 or (bk.idx[i1] - t_e) > pd.Timedelta(minutes=10):
            continue
        hi, lo = float(np.max(bk.bh[i0:i1])), float(np.min(bk.bl[i0:i1]))
        ref = 0.5 * (bk.bo[i1] + bk.ao[i1])
        if not (lo_frac <= (hi - lo) / ref <= hi_frac):
            continue
        r = lb_trade(bk, i1, j, hi, lo)
        if r is not None:
            out[d] = r[:2]
            side[d], rm[d] = r[2], r[3]
    f = U.frame(out)
    f["side"] = pd.Series(side).reindex(f.index)
    f["r"] = pd.Series(rm).reindex(f.index)
    return f


# ----------------------------------------------------------------------------------------------- L2 Smash Day
def daily_bars(bk):
    from fxlab.validation.metrics import trading_day
    bk._hl()
    td = trading_day(bk.idx)
    m = bk.m
    mc = 0.5 * (m["bid_c"].to_numpy() + m["ask_c"].to_numpy())
    f = pd.DataFrame({"h": bk.mh, "l": bk.ml, "c": mc, "k": np.arange(len(bk.idx))}, index=bk.idx)
    g = f.groupby(td)
    D = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "k0": g.k.first(), "k1": g.k.last()})
    D = D[(D.k1 - D.k0) > 600]                                 # full trading days only
    return D


def smash_signals(D, n=3):
    """Per day t: +1 buy setup (close below the lowest low of the n previous days), -1 sell setup, 0 none."""
    lo_n = D.l.shift(1).rolling(n).min()
    hi_n = D.h.shift(1).rolling(n).max()
    s = pd.Series(0, index=D.index)
    s[D.c < lo_n] = 1
    s[D.c > hi_n] = -1
    return s


def smash_day(bk, long_only=False, n=3):
    """Stop-and-reverse system (long_only: long, flat on a sell trigger). Returns per-trade DataFrame[net, gross]
    indexed by the exit time."""
    D = daily_bars(bk)
    sig = smash_signals(D, n)
    pos, E, Emid, ke = 0, None, None, None
    out = {}
    days = list(D.index)
    for a, b in zip(days[:-1], days[1:]):
        s = int(sig[a])
        if s == 0 or s == pos or (long_only and s < 0 and pos == 0):
            continue
        k0, k1 = int(D.k0[b]), int(D.k1[b])
        lvl = float(D.h[a]) if s > 0 else float(D.l[a])
        hit = np.flatnonzero(bk.mh[k0:k1 + 1] >= lvl) if s > 0 else np.flatnonzero(bk.ml[k0:k1 + 1] <= lvl)
        if not len(hit):
            continue
        k = k0 + int(hit[0])
        half = 0.5 * (bk.ao[k] - bk.bo[k])
        trig = max(lvl, bk.mo[k]) if s > 0 else min(lvl, bk.mo[k])
        if pos != 0:                                           # close the open trade at the trigger
            X = trig - pos * (half + slip(bk, k))
            net = (X / E - 1.0) if pos > 0 else (1.0 - X / E)
            net += swap_ret(bk, ke, k, pos, Emid) - 2 * bk.comm
            out[bk.idx[k]] = (float(net), float(pos * (trig / Emid - 1.0)))
            pos = 0
        if long_only and s < 0:
            continue
        pos, ke, Emid = s, k, trig
        E = trig + s * (half + slip(bk, k))
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- L3 Gotobi EA
def atr_h1(bk, n=14):
    """ATR(n) of H1 mid bars, as known at the START of each hour (from completed hours only)."""
    bk._hl()
    mc = 0.5 * (bk.m["bid_c"].to_numpy() + bk.m["ask_c"].to_numpy())
    f = pd.DataFrame({"h": bk.mh, "l": bk.ml, "c": mc}, index=bk.idx)
    H = f.resample("1h").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    pc = H.c.shift(1)
    tr = pd.concat([H.h - H.l, (H.h - pc).abs(), (H.l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    atr.index = atr.index + pd.Timedelta(hours=1)              # known when the hour has closed
    return atr


def gotobi_ea(bk, G, filt=True, stop=True):
    atr = atr_h1(bk)
    out = {}
    for d in G:
        t0, t1 = U.jst(d, 7, 30), U.jst(d, 9, 55)
        i, j = bk.at(t0, 10), bk.at(t1, 65)
        if i < 0 or j <= i:
            continue
        a = atr[atr.index <= t0]
        if not len(a) or not np.isfinite(a.iloc[-1]) or (t0 - a.index[-1]) > pd.Timedelta(hours=3):
            continue
        A = float(a.iloc[-1])
        if filt:
            h0 = int(bk.idx.searchsorted(t0 - pd.Timedelta(hours=1)))
            if i - h0 < 30:
                continue
            if float(np.max(bk.mh[h0:i]) - np.min(bk.ml[h0:i])) > 2 * A:
                continue
        r = bk.bracket(i, j, +1, 1e9, 2.5 * A) if stop else bk.trade_ij(i, j, +1, t0, t1)
        if r is not None:
            out[d] = U.rec(r)
    return U.frame(out)


def run(res, only):
    if "lb" in only:
        bk = U.book("XAUUSD")
        if bk is not None:
            f = london_breakout(bk)
            res["L1 XAUUSD London breakout, MQL5 777540 rule"] = (f[["net", "gross"]], 200)
            for s, lab in ((1, "long"), (-1, "short")):
                res[f"L1 side: {lab} trades"] = (f[f.side == s][["net", "gross"]], 200)
            res["L1 mean R multiple (net)"] = float(f.r.mean()) if len(f) else np.nan
            res["L1 profit factor (net)"] = float(f.net[f.net > 0].sum() / -f.net[f.net < 0].sum()) if len(f) else np.nan
            res["L1 trades"] = int(len(f))
    if "smash" in only:
        for sym, lo in (("EURUSD", False), ("XAUUSD", True)):
            bk = U.book(sym)
            if bk is not None:
                lab = "long only" if lo else "stop and reverse"
                res[f"L2 {sym} Smash Day (3 days), {lab}"] = (smash_day(bk, long_only=lo), 30)
    if "ea" in only:
        import flow_edges_lab as F
        for sym in ("USDJPY", "EURJPY"):
            bk = U.book(sym)
            if bk is None:
                continue
            G, _, _ = F.calendar(bk)
            res[f"L3 {sym} Gotobi EA rule (07:30-09:55, 2.5 ATR stop, volatility filter)"] = (gotobi_ea(bk, G), 62)
            res[f"L3 reference: {sym} 07:30-09:55, no filter, no stop"] = (gotobi_ea(bk, G, False, False), 62)
            res[f"L3 reference: {sym} EA short leg 09:55-10:30"] = (
                U.timed(bk, G, lambda d: (U.jst(d, 9, 55), U.jst(d, 10, 30)), -1, tol_in=65), 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="lb,smash,ea")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    try:
        run(res, a.only.split(","))
    except Exception:                                                       # noqa: BLE001
        import traceback
        res["error"] = traceback.format_exc()[-2000:]
        print(res["error"], flush=True)
        print("::error title=lit4 intraday::" + res["error"].replace("\n", " | ")[-900:], flush=True)
    T, out = U.tabulate(res, a.out)
    from cross_market_ta_lab import bh_q
    if len(T):
        prim = ~T.test.str.contains("reference|side:", case=False)
        T["q_bh_primary"] = np.nan
        T.loc[prim, "q_bh_primary"] = bh_q(T.loc[prim, "p_one_sided"].to_numpy())
    out["_coverage_m1_bars_per_year"] = U.COVERAGE
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    if len(T):
        print(T.drop(columns=["first", "last"], errors="ignore").round(3).to_string())
    for k, v in out.items():
        if not isinstance(v, dict) or k.startswith("_"):
            print(k, v)


if __name__ == "__main__":
    main()
