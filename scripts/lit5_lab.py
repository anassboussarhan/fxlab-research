"""Fifth literature round: the "smart money" and volume methods that YouTube, prop-firm blogs and forums credit
for funded accounts (order blocks, liquidity sweeps, volume profile / market profile). Each rule is taken from a
written source, made mechanical, and fixed before any result; nothing is tuned. M1 bid/ask path for every fill;
costs as in the flows lab (limit entries fill at their price with no slippage; market entries, stops and time exits
cross the spread plus 25% of it and 0.1 pip; 0.35 bp commission a side; swap over 17:00 New York). Development data
only (to 2025-12-31). Symbols: EURUSD, GBPUSD, USDJPY, XAUUSD; US500 and NAS100 CFDs for the profile tests.

FIXED BEFORE ANY RESULT:
  Bars: 15-minute bars of mid prices built from M1. Swing highs / lows: a bar whose high (low) is above (below) the
    three bars on each side, known three bars later. Break of structure (BOS) up: the first close above the last
    known swing high; down: mirror.
  OB1 Order block, FXNX's baseline (1,000 setups, 38-48% wins): at a BOS up, the bullish order block is the last
    down candle (close < open) among the five bars up to and including the lowest bar of the leg since the broken
    swing high; a buy limit at the middle of that candle, stop one pip (gold $0.10) below its low, target 2.5R;
    the order lives 96 bars (24 h), the trade at most 96 bars after the fill. Mirror for BOS down.
  OB2 OB1 with two of FXNX's filters: H4 bias (the last completed H4 close above its 50-bar EMA for longs, below
    for shorts) and fills only in the London (02:00-05:00 New York) or New York (07:00-10:00) kill zones (their
    15-minute sweep and 5-minute displacement triggers are not modelled).
  Every limit order is void once price has traded at its stop before it could fill. (Added after the first local
    run: without it, an order kept waiting outside a kill zone could fill below a stop already crossed and exit
    at the stop for a profit; that run showed OB2 at +2.4 bp, t 13, which was this error.)
  LS1 Liquidity sweep, as found by Mahadzva (SSRN 2026: in 65 of 66 folds a swept level predicted continuation, not
    the taught reversal): a bar whose high takes the last swing high but closes below it -> long at the next bar's
    open (low taken and close above -> short); stop and target 1 x ATR(14) of the 15-minute bars, at most 16 bars.
  VP1 The value-area "80% rule" (Dalton; FTMO blog "Market Profile: master the 80% strategy"): the previous
    session's volume profile from M1 tick counts (each bar's ticks spread evenly over its high-low range, 100 bins),
    point of control (POC) and 70% value area; if the session opens outside it and two consecutive 30-minute
    periods close inside it, enter at the end of the second in the direction of the far edge; target the far edge,
    stop at the session's extreme so far beyond the value area, exit at the session close. Sessions (New York time):
    US500 / NAS100 09:30-16:00, gold 08:20-13:30 (COMEX), FX 08:00-15:00. One trade a session.
  VP2 The naked POC as a magnet (NexusFi, FTMO "hidden magnets"): if the session opens outside the previous value
    area, trade at the open towards the previous POC; target the POC, stop the same distance on the other side,
    exit at the session close.
PRE-REGISTERED TESTS (one-sided, mean net per trade > 0): OB1, OB2, LS1 per FX / gold symbol; VP1, VP2 per symbol;
  Benjamini-Hochberg over all of them. Gross (mid, no costs) reported for each.
Usage: python scripts/lit5_lab.py --out DIR [--only ob,ls,vp] [--symbols ...]"""
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
FXG = ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD")
IDX = ("USA500IDXUSD", "USATECHIDXUSD")
U.PIPS.update({"GBPUSD": 1e-4, "EURUSD": 1e-4, "USDJPY": 0.01, "XAUUSD": 0.01})
TICK = {"XAUUSD": 0.10}                                   # "one pip" buffer for gold
SESSION = {"USA500IDXUSD": ((9, 30), (16, 0)), "USATECHIDXUSD": ((9, 30), (16, 0)), "XAUUSD": ((8, 20), (13, 30)),
           "EURUSD": ((8, 0), (15, 0)), "GBPUSD": ((8, 0), (15, 0)), "USDJPY": ((8, 0), (15, 0))}
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
REAL_TICKS = {}                                           # False: no tick counts, the profile is time at price


def book(sym):
    from fxlab.data.store import load_m1, available
    if not available(sym):
        return None
    try:
        m = load_m1(sym, "2007-01-01", U.END, columns=COLS + ["n_ticks"])
    except Exception:                                                       # noqa: BLE001
        m = U.load(sym)
        if m is None:
            return None
        m["n_ticks"] = 1.0
    m = m.dropna(subset=COLS)
    m = m[(m.ask_o >= m.bid_o) & (m.bid_o > 0)]
    if "n_ticks" not in m:
        m["n_ticks"] = 1.0
    REAL_TICKS[sym] = bool(m["n_ticks"].nunique() > 1)
    m["n_ticks"] = m["n_ticks"].fillna(1.0).clip(lower=1.0)
    U.COVERAGE[sym] = {int(k): int(v) for k, v in m.groupby(m.index.year).size().items()}
    pair = sym if (len(sym) == 6 and sym != "XAUUSD") else None
    bk = FL.Book(m, U.PIPS[sym], pair)
    bk._hl()
    bk.mc = 0.5 * (m["bid_c"].to_numpy() + m["ask_c"].to_numpy())
    return bk


def slip(bk, k):
    return bk.sf * (bk.ao[k] - bk.bo[k]) + bk.sp * bk.pip


def swap_ret(bk, i, j, d, px):
    n, flat, rate = bk.swaps(i, j, d, px)
    return rate if np.isfinite(rate) else flat


# ----------------------------------------------------------------------------------------------- trade engines
def run_path(bk, k, j, d, E, stop, tp):
    """From fill bar k (entry price E, all-in) to the time exit at bar j: stop (checked first, also in the fill
    bar), then target (from the next bar). Returns (exit price all-in, exit bar, at_target)."""
    if (d > 0 and bk.bl[k] <= stop) or (d < 0 and bk.ah[k] >= stop):
        px = min(stop, bk.bo[k]) if d > 0 else max(stop, bk.ao[k])   # opened through the stop: out at the open
        return px - d * slip(bk, k), k, False
    for q in range(k + 1, j):
        if d > 0:
            if bk.bl[q] <= stop:
                return min(stop, bk.bo[q]) - slip(bk, q), q, False
            if bk.bh[q] >= tp:
                return tp, q, True
        else:
            if bk.ah[q] >= stop:
                return max(stop, bk.ao[q]) + slip(bk, q), q, False
            if bk.al[q] <= tp:
                return tp, q, True
    return (bk.bo[j] - slip(bk, j)) if d > 0 else (bk.ao[j] + slip(bk, j)), j, False


def settle(bk, k, m, d, E, X, e_mid, x_mid):
    net = (X / E - 1.0) if d > 0 else (1.0 - X / E)
    net += swap_ret(bk, k, m, d, E) - 2 * bk.comm
    return float(net), float(d * (x_mid / e_mid - 1.0))


def exit_mid(bk, m, X, d, at_tp):
    h = 0.5 * (bk.ao[m] - bk.bo[m])
    return X + d * h if at_tp else X + d * slip(bk, m) + d * h


def limit_trade(bk, k0, k_exp, d, L, stop, tp, hold, allowed=None):
    """Buy (d=+1) limit at L filled when the ask low reaches it (sell limit: the bid high), between bars k0 and
    k_exp; `allowed` is an optional boolean mask of M1 bars where a fill may happen. Then run_path for `hold` bars."""
    seg = bk.al[k0:k_exp] <= L if d > 0 else bk.ah[k0:k_exp] >= L
    if allowed is not None:
        seg = seg & allowed[k0:k_exp]
    hit = np.flatnonzero(seg)
    if not len(hit):
        return None
    k = k0 + int(hit[0])
    # the setup is void once price has traded at the stop before the order could fill (e.g. outside a kill zone)
    gone = np.flatnonzero(bk.bl[k0:k] <= stop) if d > 0 else np.flatnonzero(bk.ah[k0:k] >= stop)
    if len(gone):
        return None
    E = min(L, bk.ao[k]) if d > 0 else max(L, bk.bo[k])     # a gap through the limit fills at the open
    j = min(k + hold, len(bk.idx) - 1)
    X, m, at_tp = run_path(bk, k, j, d, E, stop, tp)
    e_mid = E - d * 0.5 * (bk.ao[k] - bk.bo[k])
    return settle(bk, k, m, d, E, X, e_mid, exit_mid(bk, m, X, d, at_tp))


def market_trade(bk, k, j, d, stop_dist, tp_dist):
    E = bk.ao[k] + slip(bk, k) if d > 0 else bk.bo[k] - slip(bk, k)
    stop, tp = E - d * stop_dist, E + d * tp_dist
    X, m, at_tp = run_path(bk, k, j, d, E, stop, tp)
    return settle(bk, k, m, d, E, X, bk.mo[k], exit_mid(bk, m, X, d, at_tp))


# ----------------------------------------------------------------------------------------------- 15-minute bars
def m15(bk):
    f = pd.DataFrame({"o": bk.mo, "h": bk.mh, "l": bk.ml, "c": bk.mc, "k": np.arange(len(bk.idx))}, index=bk.idx)
    B = f.resample("15min").agg({"o": "first", "h": "max", "l": "min", "c": "last", "k": "first"}).dropna()
    B["k"] = B["k"].astype(int)
    return B


def swings(B, w=3):
    """Swing highs / lows (strict fractals of width w); `known` = the bar position when each becomes known."""
    h, lo = B.h.to_numpy(), B.l.to_numpy()
    n = len(B)
    sh, sl = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(w, n - w):
        win_h = np.r_[h[i - w:i], h[i + 1:i + w + 1]]
        win_l = np.r_[lo[i - w:i], lo[i + 1:i + w + 1]]
        sh[i] = h[i] > win_h.max()
        sl[i] = lo[i] < win_l.min()
    return sh, sl


def atr(B, n=14):
    pc = B.c.shift(1)
    tr = pd.concat([B.h - B.l, (B.h - pc).abs(), (B.l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


def h4_bias(B):
    """+1 / -1 / 0 per 15-minute bar: the last COMPLETED H4 close vs its 50-bar EMA."""
    H = B.c.resample("4h").last().dropna()
    ema = H.ewm(span=50, adjust=False).mean()
    s = np.sign(H - ema)
    s[ema.index < ema.index[0] + pd.Timedelta(hours=4 * 50)] = 0
    s.index = s.index + pd.Timedelta(hours=4)                 # known when the H4 bar closes
    return s.reindex(B.index + pd.Timedelta(minutes=15), method="ffill").fillna(0).to_numpy()


def killzone_mask(bk):
    ny = bk.idx.tz_convert(NY)
    hm = ny.hour * 60 + ny.minute
    return ((hm >= 120) & (hm < 300)) | ((hm >= 420) & (hm < 600))


def structure_events(B, w=3):
    """BOS and sweep events from swing points known in real time. Returns lists of dicts."""
    sh, sl = swings(B, w)
    h, lo, c, o = B.h.to_numpy(), B.l.to_numpy(), B.c.to_numpy(), B.o.to_numpy()
    n = len(B)
    last_h = last_l = None                                   # (price, position) of the last known swing
    used_h = used_l = False
    bos, sweeps = [], []
    for t in range(n):
        p = t - w                                            # a swing at p becomes known at the close of bar t
        if p >= 0 and sh[p]:
            last_h, used_h = (h[p], p), False
        if p >= 0 and sl[p]:
            last_l, used_l = (lo[p], p), False
        if last_h is not None and not used_h and t > last_h[1]:
            if c[t] > last_h[0]:
                a = last_h[1] + 1
                m = a + int(np.argmin(lo[a:t + 1]))
                ob = [q for q in range(max(a - 1, m - 4), m + 1) if c[q] < o[q]]
                if ob:
                    bos.append({"t": t, "d": 1, "ob": ob[-1]})
                used_h = True
            elif h[t] > last_h[0] and c[t] < last_h[0]:
                sweeps.append({"t": t, "d": 1})
                used_h = True
        if last_l is not None and not used_l and t > last_l[1]:
            if c[t] < last_l[0]:
                a = last_l[1] + 1
                m = a + int(np.argmax(h[a:t + 1]))
                ob = [q for q in range(max(a - 1, m - 4), m + 1) if c[q] > o[q]]
                if ob:
                    bos.append({"t": t, "d": -1, "ob": ob[-1]})
                used_l = True
            elif lo[t] < last_l[0] and c[t] > last_l[0]:
                sweeps.append({"t": t, "d": -1})
                used_l = True
    return bos, sweeps


def order_blocks(bk, sym, B, bos, filtered=False):
    buf = TICK.get(sym, bk.pip)
    bias = h4_bias(B) if filtered else None
    kz = killzone_mask(bk) if filtered else None
    K = B.k.to_numpy()
    out = {}
    for e in bos:
        t, d, q = e["t"], e["d"], e["ob"]
        if t + 1 >= len(B):
            continue
        if filtered and bias[t] != d:
            continue
        hi, lo = float(B.h.iloc[q]), float(B.l.iloc[q])
        L = 0.5 * (hi + lo)
        stop = lo - buf if d > 0 else hi + buf
        R = abs(L - stop)
        if R <= 0:
            continue
        tp = L + d * 2.5 * R
        k0 = int(K[t + 1])
        k_exp = int(K[min(t + 97, len(B) - 1)])
        r = limit_trade(bk, k0, k_exp, d, L, stop, tp, hold=96 * 15, allowed=kz)
        if r is not None:
            out[bk.idx[k0]] = r
    return U.frame(out)


def sweep_trades(bk, B, sweeps):
    A = atr(B)
    K = B.k.to_numpy()
    out = {}
    for e in sweeps:
        t, d = e["t"], e["d"]
        if t + 17 >= len(B) or not np.isfinite(A[t]) or A[t] <= 0:
            continue
        k, j = int(K[t + 1]), int(K[t + 17])
        if (bk.idx[k] - B.index[t]) > pd.Timedelta(minutes=30):
            continue                                         # the next bar is not the next quarter-hour (gap)
        out[bk.idx[k]] = market_trade(bk, k, j, d, A[t], A[t])
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- volume profile
def profile(lo_arr, hi_arr, vol, nb=100, va=0.70):
    """(POC, VAL, VAH) of bars with ranges [lo, hi] and volumes vol, volume spread evenly over each range."""
    lo0, hi0 = float(np.min(lo_arr)), float(np.max(hi_arr))
    if not hi0 > lo0:
        return None
    edges = np.linspace(lo0, hi0, nb + 1)
    w = edges[1] - edges[0]
    prof = np.zeros(nb)
    a = np.clip(((lo_arr - lo0) / w).astype(int), 0, nb - 1)
    b = np.clip(((hi_arr - lo0) / w).astype(int), 0, nb - 1)
    for i in range(len(vol)):
        prof[a[i]:b[i] + 1] += vol[i] / (b[i] - a[i] + 1)
    p = int(np.argmax(prof))
    tot, acc, L, H = prof.sum(), prof[p], p, p
    while acc < va * tot and (L > 0 or H < nb - 1):        # Dalton's expansion: add the larger neighbour
        up = prof[H + 1] if H < nb - 1 else -1.0
        dn = prof[L - 1] if L > 0 else -1.0
        if up >= dn:
            H += 1
            acc += up
        else:
            L -= 1
            acc += dn
    centre = lambda i: edges[i] + w / 2                      # noqa: E731
    return centre(p), edges[L], edges[H + 1]


def sessions(bk, sym):
    (h0, m0), (h1, m1) = SESSION[sym]
    start, end = bk.idx[0].normalize().tz_localize(None), bk.idx[-1].normalize().tz_localize(None)
    out = []
    for d in pd.bdate_range(start, end):
        t0, t1 = FL.local_ts(d, NY, h0, m0), FL.local_ts(d, NY, h1, m1)
        i0, i1 = int(bk.idx.searchsorted(t0)), int(bk.idx.searchsorted(t1))
        mins = (t1 - t0).total_seconds() / 60
        if i1 - i0 < 0.8 * mins or (bk.idx[i0] - t0) > pd.Timedelta(minutes=5) or i1 >= len(bk.idx):
            continue
        out.append((d, i0, i1, t0))
    return out


def value_area_tests(bk, sym):
    vol = bk.m["n_ticks"].to_numpy(dtype=float)
    S = sessions(bk, sym)
    v1, v2 = {}, {}
    prev = None
    for d, i0, i1, t0 in S:
        if prev is not None and (d - prev[0]).days <= 4:
            poc, val, vah = prev[1]
            op = bk.mo[i0]
            # VP2: open outside value -> towards the previous POC, 1:1, session-close exit
            if op > vah or op < val:
                dd = 1 if op < val else -1
                dist = abs(poc - op)
                if dist > 0:
                    v2[d] = market_trade(bk, i0, i1, dd, dist, dist)
            # VP1: open outside value, two consecutive 30-minute closes inside -> to the far edge
            if op > vah or op < val:
                dd = 1 if op < val else -1
                nper = (i1 - i0) // 30
                inside_prev = False
                for p in range(nper):
                    a, b = i0 + 30 * p, i0 + 30 * (p + 1)
                    cl = bk.mc[b - 1]
                    inside = val <= cl <= vah
                    if inside and inside_prev and b < i1:
                        far = vah if dd > 0 else val
                        ext = float(np.min(bk.ml[i0:b])) if dd > 0 else float(np.max(bk.mh[i0:b]))
                        E0 = bk.ao[b] if dd > 0 else bk.bo[b]
                        sd, td = (E0 - ext) * dd, (far - E0) * dd
                        if sd > 0 and td > 0:
                            v1[d] = market_trade(bk, b, i1, dd, sd, td)
                        break
                    inside_prev = inside
        pr = profile(bk.ml[i0:i1], bk.mh[i0:i1], vol[i0:i1])
        prev = (d, pr) if pr is not None else None
    return U.frame(v1), U.frame(v2)


# ----------------------------------------------------------------------------------------------- driver
def run(res, only, symbols):
    for sym in symbols:
        bk = book(sym)
        if bk is None:
            print(sym, "no data", flush=True)
            continue
        if sym in FXG and ("ob" in only or "ls" in only):
            B = m15(bk)
            bos, sw = structure_events(B)
            res[f"_events {sym}"] = {"bos": len(bos), "sweeps": len(sw), "m15_bars": len(B)}
            if "ob" in only:
                res[f"OB1 {sym} order block, FXNX baseline"] = (order_blocks(bk, sym, B, bos), 250)
                res[f"OB2 {sym} order block, H4 bias + kill zones"] = (order_blocks(bk, sym, B, bos, True), 250)
            if "ls" in only:
                res[f"LS1 {sym} liquidity sweep, continuation"] = (sweep_trades(bk, B, sw), 250)
        if "vp" in only and sym in SESSION:
            a, b = value_area_tests(bk, sym)
            res[f"VP1 {sym} value-area 80% rule"] = (a, 60)
            res[f"VP2 {sym} open outside value, to the previous POC"] = (b, 120)
        print(sym, "done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="ob,ls,vp")
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
        print("::error title=lit5::" + res["error"].replace("\n", " | ")[-900:], flush=True)
    T, out = U.tabulate(res, a.out)
    from cross_market_ta_lab import bh_q
    if len(T):
        T["q_bh"] = bh_q(T["p_one_sided"].to_numpy())
        for k, (x, _) in ((k, v) for k, v in res.items() if isinstance(v, tuple)):
            x = x.net.dropna()
            pf = float(x[x > 0].sum() / -x[x < 0].sum()) if (x < 0).any() else np.nan
            T.loc[T.test == k, "profit_factor"] = pf
    out["_coverage_m1_bars_per_year"] = U.COVERAGE
    out["_tick_volume_used"] = REAL_TICKS
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
