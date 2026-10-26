"""Intraday tests of the remaining MQL5 / free-batch ideas on M1 bid/ask bars (the project's GitHub mirror for
2007-2018, Dukascopy for 2019-2025), rules fixed in advance (nothing tuned), costs as in the flows lab
(fxlab.execution.costs.BASE: 25% of the quoted spread + 0.1 pip slippage per side on top of crossing the quoted
spread, 0.35 bp commission per side, the rate-differential swap on FX positions held over 17:00 New York). Every bar
after 2025-12-31 is refused by the loader (2026 stays locked). Each test reports 2007-2018 and 2019-2025 separately
where both exist.

PRE-REGISTERED TESTS (one-sided, mean net > 0):
  I1  USDJPY post-fix short (the Nakane EA's second leg): short 09:55 -> 10:30 Tokyo on Gotobi days.
  I2  USDJPY Nakane long: 07:30 -> 09:55 Tokyo on Gotobi days (Mondays included).
  I3  USDJPY "Gotobi Overnight JPY": long 23:00 Tokyo the evening before -> 09:53 Tokyo on Gotobi days, stop 0.5 x
      the previous 20 days' mean daily range (the EA's 0.5 ATR).
  I4  EURJPY Gotobi: long 03:00 -> 09:55 Tokyo on Gotobi days, Mondays skipped (the USDJPY F1 rule).
  I5  USDJPY Japanese-holiday rule: long 03:00 -> 09:55 Tokyo on the first Tokyo business day after a weekday
      Japanese public holiday, when that day is not a Gotobi day and not a Monday.
  I6  Home-currency depreciation in home hours (Ranaldo 2009): (a) long USDJPY 00:00 -> 07:00 UTC; (b) short EURUSD
      07:00 -> 12:00 UTC; (c) long EURUSD 13:00 -> 20:00 UTC; (d) short USDJPY 13:00 -> 20:00 UTC; weekdays.
  I7  Night scalping on AUDNZD, EURGBP, GBPCAD: M5 mid Bollinger(20, 2); from 21:00 to 01:00 UTC (Monday-Thursday
      nights), sell a close above the upper band / buy a close below the lower band at the next bar's open, exit at
      the middle band (next bar open after a close beyond it), at a stop of one band-width beyond the entry, or at
      01:00 UTC; one position at a time per pair.
  I8  Fading retail candle setups on EURUSD H1 (the Warrior_EA mirror test): bullish pin bar (lower wick >= 60% of
      the range, body <= 30%, close in the top third) -> the retail trade is long at the next open with the stop
      1 pip under the low and the target at 2x the risk; the fade is the mirror (short, stop at the retail target,
      target at the retail stop), at most 24 bars; bearish pins mirrored. Inside bars: when a later bar first breaks
      the mother bar's high (low), the retail trade buys (sells) the break with the stop at the other side and a 2R
      target; the fade is its mirror.
  I9  Gold London breakout (XAUUSD): range 00:00-07:00 London; after 07:00 the first M1 close beyond the range ->
      enter at the next open in that direction, stop at the opposite side of the range, exit 16:00 London.
  I10 Weekend-gap fade on AUDUSD and EURJPY (the project's pre-2019 near-miss; only 2019-2025 counts): if the first
      quote after the weekend is more than 0.5 x the 20-day mean range away from Friday's last mid, trade toward
      Friday's close; target Friday's close, stop one gap beyond the entry, exit after 24 hours.
  I11 Index-review closing auctions (PropFlow): on the third Friday of March, June, September and December, short
      the FTSE 100 CFD 15:50 -> 16:40 London, the CAC 40 CFD 17:20 -> 17:45 Paris, the SMI CFD 17:10 -> 17:40 Zurich;
      control: the same windows on every other Friday.
  I12 Index intraday momentum: (a) US500 and NAS100: sign of 09:30 -> 10:00 New York -> hold 15:30 -> 16:00 (Gao,
      Han, Li & Zhou); (b) opening-range breakout with breakeven and trailing (Krueger's TRAIL 0.5 / 1.0 / 0.5): the
      first 30 minutes' range (US500, NAS100 from 09:30 New York; DAX 09:00 Frankfurt; FTSE 08:00 London); the
      first M1 close beyond it -> enter at the next open, stop at the other side (risk R); at +0.5 R the stop moves to
      breakeven, beyond +1 R it trails 1 R behind the best price in 0.5 R steps; exit at the cash close.

IMPLEMENTATION CONVENTIONS (fixed with the rules):
  * Times are matched to the first M1 bar at or after the target within 10 minutes. The mirror has no bars from
    00:00 to 01:00 UTC for most pairs; as in the project's earlier Gotobi tests, exits at 09:53/09:55 Tokyo (00:53/
    00:55 UTC) and the I6a entry at 00:00 UTC accept the 01:00 UTC bar there (tolerance 65 minutes) - so 2007-2018
    exits happen at 10:00 Tokyo. I1 enters at 09:55 itself and needs that bar (no tolerance), so I1 has no 2007-2018
    trades.
  * I8: while a fade is open (until its 24-bar window ends) new setups are ignored. I10: the reference range is the
    mean high-low of the previous 20 complete FX trading days (>= 1200 M1 bars), because the 2019-2025 files hold only
    the hours around the weekend.
  * Index CFDs (I11, I12) exist only in the 2019-2025 Dukascopy files; their positions are intraday (no financing).
Usage: python scripts/untested_intraday_lab.py --out DIR [--only tokyo,eurusd,night,gold,gap,review,index]"""
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

END = "2025-12-31 23:59"
PIPS = {"USDJPY": 0.01, "EURJPY": 0.01, "EURUSD": 1e-4, "AUDUSD": 1e-4, "AUDNZD": 1e-4, "EURGBP": 1e-4,
        "GBPCAD": 1e-4, "XAUUSD": 0.01, "USA500IDXUSD": 0.1, "USATECHIDXUSD": 0.1, "DEUIDXEUR": 0.1,
        "GBRIDXGBP": 0.1, "FRAIDXEUR": 0.1, "CHEIDXCHF": 0.1}
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
MIN30 = pd.Timedelta(minutes=30)
COVERAGE = {}


def load(sym):
    from fxlab.data.store import load_m1, available
    if not available(sym):
        return None
    try:
        m = load_m1(sym, "2007-01-01", END, columns=COLS)
    except FileNotFoundError:
        return None
    m = m.dropna()
    m = m[(m.ask_o >= m.bid_o) & (m.bid_o > 0)]
    if not len(m):
        return None
    yrs = m.groupby(m.index.year).size()
    COVERAGE[sym] = {int(k): int(v) for k, v in yrs.items()}
    return m


def book(sym):
    m = load(sym)
    if m is None:
        return None
    return FL.Book(m, PIPS[sym], sym if len(sym) == 6 and "IDX" not in sym else None)


def summ(x, per_year):
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 8:
        return {"n": int(len(x))}
    sd = x.std()
    t = float(x.mean() / sd * math.sqrt(len(x))) if sd > 0 else np.nan
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": t,
            "p_one_sided": float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan,
            "sharpe": float(x.mean() / sd * math.sqrt(per_year)) if sd > 0 else np.nan,
            "hit": float((x > 0).mean()), "worst_bp": float(x.min() * 1e4),
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}", "first": str(x.index.min())[:10],
            "last": str(x.index.max())[:10]}


def by_period(x, per_year):
    x = pd.Series(x, dtype=float).dropna()
    x.index = pd.DatetimeIndex(x.index)
    if x.index.tz is not None:
        x.index = x.index.tz_convert("UTC").tz_localize(None)
    return {"all": summ(x, per_year), "2007_2018": summ(x[x.index < "2019-01-01"], per_year),
            "2019_2025": summ(x[x.index >= "2019-01-01"], per_year)}


def net_of(r):
    """Net return including the swap (the rate-differential swap where rates exist, else the flat platform swap)."""
    net, gross, _, _, n, flat, rate = r
    return net + (rate if np.isfinite(rate) else flat)


def timed(bk, days, t_fn, side, tol_in=10, tol_out=10):
    out = {}
    for d in days:
        t0, t1 = t_fn(d)
        i, j = bk.at(t0, tol_in), bk.at(t1, tol_out)
        if i < 0 or j < 0 or j <= i:
            continue
        r = bk.trade_ij(i, j, side, t0, t1)
        if r is not None:
            out[d] = net_of(r)
    return pd.Series(out, dtype=float)


def jst(d, h, m):
    return FL.local_ts(d, "Asia/Tokyo", h, m)


def utc(d, h, m=0):
    return pd.Timestamp(d.date()).tz_localize("UTC") + pd.Timedelta(hours=h, minutes=m)


def first_after(bk, ts, tol_min):
    """First bar at or after ts within tol, else the first bar after ts (data gap), else -1."""
    j = bk.at(ts, tol_min)
    if j >= 0:
        return j
    j = int(bk.idx.searchsorted(ts))
    return j if j < len(bk.idx) else -1


# ----------------------------------------------------------------------------------------------- Tokyo
def tokyo_tests(res, bk=None, bk2=None):
    bk = bk if bk is not None else book("USDJPY")
    if bk is None:
        return
    start, end = str(bk.idx[0].date()), str(bk.idx[-1].date())
    hol = FL.jp_holidays(range(bk.idx[0].year, bk.idx[-1].year + 1))
    G = FL.gotobi_days(start, end, hol)
    res["I1 USDJPY post-fix short 09:55-10:30 Tokyo"] = (timed(bk, G, lambda d: (jst(d, 9, 55), jst(d, 10, 30)), -1), 62)
    res["I2 USDJPY Nakane long 07:30-09:55 Tokyo"] = (
        timed(bk, G, lambda d: (jst(d, 7, 30), jst(d, 9, 55)), +1, tol_out=65), 62)
    rng = bk.avg_range()
    from fxlab.validation.metrics import trading_day
    out = {}
    for d in G:
        t0, t1 = jst(d - pd.Timedelta(days=1), 23, 0), jst(d, 9, 53)
        i, j = bk.at(t0, 10), bk.at(t1, 65)
        if i < 0 or j < 0 or j <= i:
            continue
        td = trading_day(pd.DatetimeIndex([bk.idx[i]]))[0]
        rg = float(rng.get(td, np.nan))
        if not np.isfinite(rg) or rg <= 0:
            continue
        r = bk.bracket(i, j, +1, 1e9, 0.5 * rg)
        if r is not None:
            out[d] = net_of(r)
    res["I3 USDJPY Gotobi overnight 23:00-09:53 Tokyo, 0.5-range stop"] = (pd.Series(out, dtype=float), 62)
    Gs = set(G)
    days = set()
    for h in sorted(hol):
        if h.weekday() >= 5 or not (pd.Timestamp(start) <= h <= pd.Timestamp(end)):
            continue
        d = h + pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in hol:
            d += pd.Timedelta(days=1)
        if d not in Gs and d.weekday() != 0:
            days.add(d)
    res["I5 USDJPY day after a Japanese holiday 03:00-09:55 Tokyo"] = (
        timed(bk, sorted(days), lambda d: (jst(d, 3, 0), jst(d, 9, 55)), +1, tol_out=65), 12)
    wd = list(pd.bdate_range(start, end))
    res["I6a long USDJPY 00:00-07:00 UTC (JPY weak at home)"] = (
        timed(bk, wd, lambda d: (utc(d, 0), utc(d, 7)), +1, tol_in=65), 252)
    res["I6d short USDJPY 13:00-20:00 UTC (USD weak at home)"] = (
        timed(bk, wd, lambda d: (utc(d, 13), utc(d, 20)), -1), 252)
    bk2 = bk2 if bk2 is not None else book("EURJPY")
    if bk2 is not None:
        s2, e2 = str(bk2.idx[0].date()), str(bk2.idx[-1].date())
        hol2 = FL.jp_holidays(range(bk2.idx[0].year, bk2.idx[-1].year + 1))
        G2 = [d for d in FL.gotobi_days(s2, e2, hol2) if d.weekday() != 0]
        res["I4 EURJPY Gotobi 03:00-09:55 Tokyo"] = (
            timed(bk2, G2, lambda d: (jst(d, 3, 0), jst(d, 9, 55)), +1, tol_out=65), 62)


def eurusd_tests(res, bk=None):
    bk = bk if bk is not None else book("EURUSD")
    if bk is None:
        return
    wd = list(pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date())))
    res["I6b short EURUSD 07:00-12:00 UTC (EUR weak at home)"] = (timed(bk, wd, lambda d: (utc(d, 7), utc(d, 12)), -1), 252)
    res["I6c long EURUSD 13:00-20:00 UTC (USD weak at home)"] = (timed(bk, wd, lambda d: (utc(d, 13), utc(d, 20)), +1), 252)
    res.update(candle_fade(bk))


# ----------------------------------------------------------------------------------------------- I8
def candle_fade(bk):
    bk._hl()
    m = bk.m
    mid = pd.DataFrame({"o": bk.mo, "h": bk.mh, "l": bk.ml, "c": 0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy())},
                       index=bk.idx)
    H = mid.resample("1h", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    pip = bk.pip
    tl = H.index
    o, h, l, c = (H[k].to_numpy() for k in "ohlc")
    first = bk.idx.searchsorted(tl)                           # first M1 bar of each H1 bar
    nH = len(H)

    def bar_i(n):
        i = int(first[n])
        return i if i < len(bk.idx) and bk.idx[i] < tl[n] + pd.Timedelta(hours=1) else -1

    pins, insides = {}, {}
    busy = -1
    for n in range(1, nH - 25):
        if n <= busy:
            continue
        r_ = h[n] - l[n]
        if r_ <= 0:
            continue
        body = abs(c[n] - o[n])
        lw, uw = min(o[n], c[n]) - l[n], h[n] - max(o[n], c[n])
        if lw >= 0.6 * r_ and body <= 0.3 * r_ and c[n] >= l[n] + 2 / 3 * r_:
            side, stop_px = +1, l[n] - pip                    # bullish pin: the retail trade is long
        elif uw >= 0.6 * r_ and body <= 0.3 * r_ and c[n] <= h[n] - 2 / 3 * r_:
            side, stop_px = -1, h[n] + pip
        else:
            continue
        i, j = bar_i(n + 1), bar_i(n + 25)
        if i < 0 or j < 0 or j <= i:
            continue
        risk = abs(bk.mo[i] - stop_px)
        if risk <= 2 * pip:
            continue
        r = bk.bracket(i, j, -side, risk, 2 * risk)            # fade: target = retail stop, stop = retail target
        if r is not None:
            pins[bk.idx[i]] = net_of(r)
            busy = n + 24
    busy = -1
    for n in range(1, nH - 26):
        if n <= busy:
            continue
        mh, ml = h[n - 1], l[n - 1]
        if not (h[n] < mh and l[n] > ml):
            continue
        i0, iend = bar_i(n + 1), bar_i(n + 25)
        if i0 < 0 or iend < 0 or iend <= i0:
            continue
        up = np.flatnonzero(bk.mh[i0:iend] > mh)
        dn = np.flatnonzero(bk.ml[i0:iend] < ml)
        if not len(up) and not len(dn):
            continue
        side, k0 = (+1, up[0]) if len(up) and (not len(dn) or up[0] < dn[0]) else (-1, dn[0])
        if len(up) and len(dn) and up[0] == dn[0]:
            continue                                            # both sides in one minute: retail direction unknown
        i = i0 + k0 + 1                                         # the retail stop order filled; the fade enters next
        if i >= iend:
            continue
        risk = mh - ml
        if risk <= 2 * pip:
            continue
        r = bk.bracket(i, iend, -side, risk, 2 * risk)
        if r is not None:
            insides[bk.idx[i]] = net_of(r)
            busy = n + 25
    return {"I8a EURUSD pin-bar fade (H1)": (pd.Series(pins, dtype=float), 252),
            "I8b EURUSD inside-bar breakout fade (H1)": (pd.Series(insides, dtype=float), 252)}


# ----------------------------------------------------------------------------------------------- I7
def night_scalp(sym, bk=None):
    bk = bk if bk is not None else book(sym)
    if bk is None:
        return None
    bk._hl()
    m = bk.m
    c1 = pd.Series(0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy()), index=bk.idx)
    c5s = c1.resample("5min", label="left", closed="left").last().dropna()
    c5 = c5s.to_numpy()
    ma = c5s.rolling(20).mean().to_numpy()
    sd = c5s.rolling(20).std().to_numpy()
    up, lo = ma + 2 * sd, ma - 2 * sd
    t5 = c5s.index
    hrs, wds = t5.hour.to_numpy(), t5.weekday.to_numpy()
    in_win = ((hrs >= 21) & (wds <= 3)) | ((hrs == 0) & (wds >= 1) & (wds <= 4))
    gap_next = np.r_[np.asarray((t5[1:] - t5[:-1]) > MIN30), True]          # resolution-safe (ns or us index)
    out = {}
    n, N = 20, len(t5) - 1
    while n < N:
        if not in_win[n] or not np.isfinite(up[n]) or gap_next[n]:
            n += 1
            continue
        side = -1 if c5[n] > up[n] else (+1 if c5[n] < lo[n] else 0)
        if side == 0:
            n += 1
            continue
        t_in = t5[n + 1]
        i = bk.at(t_in, 5)
        day0 = t5[n].normalize()
        lim = (day0 + pd.Timedelta(days=1) if hrs[n] >= 21 else day0) + pd.Timedelta(hours=1)
        if i < 0 or t_in >= lim:
            n += 1
            continue
        k = n + 1
        t_out = lim
        while k < N and t5[k] < lim:
            crossed = (c5[k] <= ma[k]) if side < 0 else (c5[k] >= ma[k])
            if crossed or gap_next[k]:
                t_out = min(t5[k + 1], lim)
                break
            k += 1
        j = first_after(bk, t_out, 10)
        if j <= i:
            n += 1
            continue
        stop_d = float(up[n] - lo[n])
        r = bk.bracket(i, j, side, 1e9, stop_d)
        if r is None:
            n += 1
            continue
        out[t_in] = net_of(r)
        # where the position actually ended (the stop may have been hit before t_out)
        si = bk.sf * (bk.ao[i] - bk.bo[i]) + bk.sp * bk.pip
        if side > 0:
            E = bk.ao[i] + si
            hit = np.flatnonzero(bk.bl[i:j] <= E - stop_d)
        else:
            E = bk.bo[i] - si
            hit = np.flatnonzero(bk.ah[i:j] >= E + stop_d)
        t_end = bk.idx[i + hit[0]] if len(hit) else bk.idx[j]
        n = max(n + 1, int(t5.searchsorted(t_end, side="right")) - 1)
    return pd.Series(out, dtype=float)


# ----------------------------------------------------------------------------------------------- I9
def gold_london(res, bk=None):
    bk = bk if bk is not None else book("XAUUSD")
    if bk is None:
        return
    m = bk.m
    mid = 0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy())
    out = {}
    for d in pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date())):
        a, b = FL.local_ts(d, "Europe/London", 0, 0), FL.local_ts(d, "Europe/London", 7, 0)
        e = FL.local_ts(d, "Europe/London", 16, 0)
        ia, ib, ie = bk.idx.searchsorted([a, b, e])
        if ib - ia < 200 or ie <= ib:
            continue
        hi, lo = mid[ia:ib].max(), mid[ia:ib].min()
        seg = mid[ib:ie]
        br = np.flatnonzero((seg > hi) | (seg < lo))
        if not len(br):
            continue
        side = +1 if seg[br[0]] > hi else -1
        i = ib + br[0] + 1
        j = bk.at(e, 10)
        if j < 0 or i >= j:
            continue
        ent = 0.5 * (bk.bo[i] + bk.ao[i])
        stop_d = (ent - lo) if side > 0 else (hi - ent)
        if stop_d <= 0:
            continue
        r = bk.bracket(i, j, side, 1e9, stop_d)
        if r is not None:
            out[d] = net_of(r)
    res["I9 XAUUSD London breakout of the Asian range"] = (pd.Series(out, dtype=float), 252)


# ----------------------------------------------------------------------------------------------- I10
def full_day_range(bk, min_bars=1200):
    """Mean mid high-low of the previous 20 complete FX trading days, as a function of the trading day."""
    from fxlab.validation.metrics import trading_day
    bk._hl()
    td = trading_day(bk.idx)
    g = pd.DataFrame({"h": bk.mh, "l": bk.ml, "td": td}).groupby("td")
    d = pd.DataFrame({"r": g.h.max() - g.l.min(), "n": g.size()})
    rr = d.r[d.n >= min_bars].rolling(20, min_periods=10).mean()

    def get(day):
        k = int(rr.index.searchsorted(day)) - 1          # last complete day strictly before `day`
        return float(rr.iloc[k]) if k >= 0 else np.nan
    return get


def weekend_gap(sym, bk=None):
    bk = bk if bk is not None else book(sym)
    if bk is None:
        return None
    from fxlab.validation.metrics import trading_day
    ref = full_day_range(bk)
    idx = bk.idx
    gaps = np.flatnonzero(np.asarray((idx[1:] - idx[:-1]) > pd.Timedelta(hours=36)))   # a break > 36 h = a weekend
    mid_c = 0.5 * (bk.m.bid_c.to_numpy() + bk.m.ask_c.to_numpy())
    mid_o = 0.5 * (bk.bo + bk.ao)
    out = {}
    for g in gaps:
        i = g + 1
        if idx[i].weekday() not in (6, 0) or idx[g].weekday() != 4:
            continue
        gap = mid_o[i] - mid_c[g]
        rg = ref(trading_day(pd.DatetimeIndex([idx[i]]))[0])
        if not np.isfinite(rg) or abs(gap) <= 0.5 * rg:
            continue
        side = -1 if gap > 0 else +1
        j = first_after(bk, idx[i] + pd.Timedelta(hours=24), 30)
        if j <= i:
            continue
        r = bk.bracket(i, j, side, abs(gap), abs(gap))
        if r is not None:
            out[idx[i]] = net_of(r)
    return pd.Series(out, dtype=float)


# ----------------------------------------------------------------------------------------------- I11 / I12
def third_friday(y, m):
    d = pd.Timestamp(year=y, month=m, day=1)
    d += pd.Timedelta(days=(4 - d.weekday()) % 7)
    return d + pd.Timedelta(days=14)


REVIEW = {"GBRIDXGBP": ("Europe/London", (15, 50), (16, 40), "FTSE 100"),
          "FRAIDXEUR": ("Europe/Paris", (17, 20), (17, 45), "CAC 40"),
          "CHEIDXCHF": ("Europe/Zurich", (17, 10), (17, 40), "SMI")}


def review_auctions(res, books=None):
    rev = {third_friday(y, m) for y in range(2007, 2026) for m in (3, 6, 9, 12)}
    for sym, (tz, a, b, name) in REVIEW.items():
        bk = (books or {}).get(sym) if books is not None else book(sym)
        if bk is None:
            continue
        ev, ctl = {}, {}
        for d in pd.date_range(str(bk.idx[0].date()), str(bk.idx[-1].date()), freq="W-FRI"):
            t0, t1 = FL.local_ts(d, tz, *a), FL.local_ts(d, tz, *b)
            i, j = bk.at(t0, 5), bk.at(t1, 5)
            if i < 0 or j <= i:
                continue
            r = bk.trade_ij(i, j, -1, t0, t1)
            if r is None:
                continue
            (ev if d in rev else ctl)[d] = r[0]
        ev, ctl = pd.Series(ev, dtype=float), pd.Series(ctl, dtype=float)
        res[f"I11 {name} review-day short into the close"] = (ev, 4)
        res[f"I11 {name} control (other Fridays)"] = (ctl, 48)
        if len(ev) > 3 and len(ctl) > 10:
            res[f"I11 {name} review minus control (Welch t)"] = float(st.ttest_ind(ev, ctl, equal_var=False).statistic)


def orb_trail(bk, d, tz, open_hm, close_hm, rng_min=30):
    """One day's opening-range breakout with breakeven and trailing; net return or None."""
    t_open = FL.local_ts(d, tz, *open_hm)
    t_rng = t_open + pd.Timedelta(minutes=rng_min)
    t_close = FL.local_ts(d, tz, *close_hm)
    i0, i1, iz = bk.at(t_open, 3), bk.at(t_rng, 3), bk.at(t_close, 10)
    if min(i0, i1, iz) < 0 or not (i0 < i1 < iz):
        return None
    bk._hl()
    hi, lo = bk.mh[i0:i1].max(), bk.ml[i0:i1].min()
    if not hi > lo:
        return None
    mc = 0.5 * (bk.m.bid_c.to_numpy()[i1:iz] + bk.m.ask_c.to_numpy()[i1:iz])
    br = np.flatnonzero((mc > hi) | (mc < lo))
    if not len(br):
        return None
    k = i1 + br[0] + 1
    if k >= iz:
        return None
    side = +1 if mc[br[0]] > hi else -1
    s_in = bk.sf * (bk.ao[k] - bk.bo[k]) + bk.sp * bk.pip
    E = bk.ao[k] + s_in if side > 0 else bk.bo[k] - s_in
    R = (E - lo) if side > 0 else (hi - E)
    if R <= 0:
        return None
    stop, best = E - side * R, E
    for x in range(k, iz):
        lo_x = bk.bl[x] if side > 0 else bk.al[x]          # the side the position exits on
        hi_x = bk.bh[x] if side > 0 else bk.ah[x]
        if (lo_x <= stop) if side > 0 else (hi_x >= stop):
            sx = bk.sf * (bk.ao[x] - bk.bo[x]) + bk.sp * bk.pip
            px = (min(stop, bk.bo[x]) - sx) if side > 0 else (max(stop, bk.ao[x]) + sx)
            return (px / E - 1 if side > 0 else 1 - px / E) - 2 * bk.comm
        best = max(best, hi_x) if side > 0 else min(best, lo_x)  # the stop moves from the next bar on
        gain = (best - E) * side
        if gain >= 0.5 * R:
            stop = max(stop, E) if side > 0 else min(stop, E)
        if gain > R:
            lock = math.floor((gain - R) / (0.5 * R)) * 0.5 * R          # 1 R behind the best, in 0.5 R steps
            stop = max(stop, E + lock) if side > 0 else min(stop, E - lock)
    sx = bk.sf * (bk.ao[iz] - bk.bo[iz]) + bk.sp * bk.pip
    px = bk.bo[iz] - sx if side > 0 else bk.ao[iz] + sx
    return (px / E - 1 if side > 0 else 1 - px / E) - 2 * bk.comm


INDEX = {"USA500IDXUSD": ("America/New_York", (9, 30), (16, 0), "US500"),
         "USATECHIDXUSD": ("America/New_York", (9, 30), (16, 0), "NAS100"),
         "DEUIDXEUR": ("Europe/Berlin", (9, 0), (17, 30), "DAX"),
         "GBRIDXGBP": ("Europe/London", (8, 0), (16, 30), "FTSE 100")}


def index_momentum(res, books=None):
    for sym, (tz, o_hm, c_hm, name) in INDEX.items():
        bk = (books or {}).get(sym) if books is not None else book(sym)
        if bk is None:
            continue
        days = pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date()))
        orb = {}
        for d in days:
            r = orb_trail(bk, d, tz, o_hm, c_hm)
            if r is not None:
                orb[d] = r
        res[f"I12b {name} opening-range breakout with breakeven and trailing"] = (pd.Series(orb, dtype=float), 252)
        if tz == "America/New_York":
            fl = {}
            for d in days:
                ia, ib = bk.at(FL.local_ts(d, tz, 9, 30), 3), bk.at(FL.local_ts(d, tz, 10, 0), 3)
                t0, t1 = FL.local_ts(d, tz, 15, 30), FL.local_ts(d, tz, 16, 0)
                ic, ie = bk.at(t0, 3), bk.at(t1, 3)
                if min(ia, ib, ic, ie) < 0 or not (ia < ib <= ic < ie):
                    continue
                first = (bk.bo[ib] + bk.ao[ib]) - (bk.bo[ia] + bk.ao[ia])
                if first == 0:
                    continue
                r = bk.trade_ij(ic, ie, 1 if first > 0 else -1, t0, t1)
                if r is not None:
                    fl[d] = r[0]
            res[f"I12a {name} first half-hour -> last half-hour"] = (pd.Series(fl, dtype=float), 252)


# ----------------------------------------------------------------------------------------------- main
def run_groups(groups, res):
    jobs = (("tokyo", lambda: tokyo_tests(res)), ("eurusd", lambda: eurusd_tests(res)),
            ("gold", lambda: gold_london(res)), ("review", lambda: review_auctions(res)),
            ("index", lambda: index_momentum(res)))
    for g, fn in jobs:
        if g in groups:
            try:
                fn()
            except Exception as ex:                                      # noqa: BLE001
                import traceback
                res[f"{g} error"] = traceback.format_exc()[-1500:]
            print(g, "done", flush=True)
    if "night" in groups:
        for s in ("AUDNZD", "EURGBP", "GBPCAD"):
            try:
                x = night_scalp(s)
                if x is not None:
                    res[f"I7 night scalping {s}"] = (x, 252)
            except Exception:                                            # noqa: BLE001
                import traceback
                res[f"I7 {s} error"] = traceback.format_exc()[-1500:]
        print("night done", flush=True)
    if "gap" in groups:
        for s in ("AUDUSD", "EURJPY"):
            try:
                x = weekend_gap(s)
                if x is not None:
                    res[f"I10 weekend-gap fade {s}"] = (x, 52)
            except Exception:                                            # noqa: BLE001
                import traceback
                res[f"I10 {s} error"] = traceback.format_exc()[-1500:]
        print("gap done", flush=True)


def tabulate(res, outdir):
    rows, out = [], {}
    for k, v in res.items():
        if isinstance(v, tuple):
            x, py = v
            bp = by_period(x, py)
            out[k] = bp
            primary = "2019_2025" if k.startswith("I10") else "all"
            s = bp[primary]
            rows.append({"test": k, "family": k.split()[0], "period": primary,
                         **{c: s.get(c) for c in ("n", "mean_bp", "t", "p_one_sided", "sharpe", "hit", "worst_bp",
                                                  "years_positive", "first", "last")},
                         "n_2007_2018": bp["2007_2018"].get("n"), "t_2007_2018": bp["2007_2018"].get("t"),
                         "mean_bp_2007_2018": bp["2007_2018"].get("mean_bp"),
                         "n_2019_2025": bp["2019_2025"].get("n"), "t_2019_2025": bp["2019_2025"].get("t"),
                         "mean_bp_2019_2025": bp["2019_2025"].get("mean_bp")})
            if outdir:
                name = "".join(ch if ch.isalnum() else "_" for ch in k)[:80]
                pd.Series(x, dtype=float).rename("net").to_csv(os.path.join(outdir, f"trades_{name}.csv"))
        else:
            out[k] = v
    return pd.DataFrame(rows), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of groups: tokyo,eurusd,night,gold,gap,review,index")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = set(a.only.split(",")) if a.only else {"tokyo", "eurusd", "night", "gold", "gap", "review", "index"}
    res = {}
    run_groups(groups, res)
    T, out = tabulate(res, a.out)
    out["_coverage_m1_bars_per_year"] = COVERAGE
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
