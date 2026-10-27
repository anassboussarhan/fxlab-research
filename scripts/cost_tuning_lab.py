"""Cost-aware tuning of the intraday rules that earned before costs but lost after them (scripts/untested_intraday_lab.py:
I7 night scalping, I8 candle-pattern fades, I9 gold London breakout, I4 EURJPY Gotobi, I12b index opening-range
breakout). The question is whether settings that trade less often, or skip the widest-spread minutes, survive.

FIXED BEFORE ANY RESULT:
  Grids (the pre-registered setting of the intraday lab is always one of the points):
    night  (AUDNZD, EURGBP, GBPCAD): first entry hour 21, 22 or 23 UTC (window to 01:00 UTC); Bollinger(20, k),
           k = 2.0, 2.5, 3.0; exit at the middle band or at the opposite band; minimum band width 0, 5 or 10 times
           the entry spread -> 54 settings.
    fades  (EURUSD H1 pin bars and inside-bar breakouts, separately): fade target / stop in units of the retail
           risk (1, 2), (1.5, 1.5), (2, 2), (1, 1); minimum risk 2 pips, 5 or 10 spreads; entries at any hour or only
           07:00-16:59 UTC -> 24 settings each.
    gold   (XAUUSD): range from 00:00 London to 06:00, 07:00 or 08:00; exit at 12:00, 16:00 or 20:00 London; stop at
           the opposite side or the middle of the range; minimum range 0 or 10 spreads -> 36 settings.
    eurjpy (EURJPY Gotobi days, Mondays skipped): entry 23:00 Tokyo the evening before, 03:00, 06:00 or 08:00;
           exit 09:55; no stop, or a stop at 0.5 x the 20-day range -> 8 settings.
    orb    (US500, NAS100 from 09:30 New York): range 15, 30 or 60 minutes; trailing as pre-registered (breakeven at
           +0.5 R, 1 R behind beyond +1 R, 0.5 R steps), wide (breakeven at +1 R, 2 R behind beyond +2 R, 1 R steps)
           or none; exit at the cash close or 2 hours after entry -> 18 settings.
  Selection: walk-forward by calendar year. Each test year trades the setting with the highest mean net return
    per trade over all earlier years (at least 30 trades; expanding window). FX tests start in 2010 (three years
    of history; gold in 2014, its mirror history starts in May 2011), the index tests in 2022. Costs, swap and
    fills exactly as in the intraday lab.
PRE-REGISTERED TESTS (one-sided): the walk-forward trades' mean net return > 0, for each rule and instrument;
  Benjamini-Hochberg over them. Reported beside it: the pre-registered setting over the same test years, and the
  best setting chosen with hindsight on all years (shown only to measure how much hindsight adds).
Usage: python scripts/cost_tuning_lab.py --only night,fades,gold,eurjpy,orb --out DIR"""
from __future__ import annotations

import argparse
import itertools
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

MIN30 = pd.Timedelta(minutes=30)


# ----------------------------------------------------------------------------------------------- night scalping
def night_p(bk, start_h=21, k=2.0, exit_mode="mid", min_bw=0.0, _cache={}):
    bk._hl()
    key = id(bk)
    if key not in _cache:
        m = bk.m
        c1 = pd.Series(0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy()), index=bk.idx)
        c5s = c1.resample("5min", label="left", closed="left").last().dropna()
        _cache.clear()
        _cache[key] = (c5s, c5s.rolling(20).mean().to_numpy(), c5s.rolling(20).std().to_numpy())
    c5s, ma, sd = _cache[key]
    c5 = c5s.to_numpy()
    up, lo = ma + k * sd, ma - k * sd
    t5 = c5s.index
    hrs, wds = t5.hour.to_numpy(), t5.weekday.to_numpy()
    in_win = ((hrs >= start_h) & (wds <= 3)) | ((hrs == 0) & (wds >= 1) & (wds <= 4))
    gap_next = np.r_[np.asarray((t5[1:] - t5[:-1]) > MIN30), True]
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
        if i < 0 or t_in >= lim or (up[n] - lo[n]) < min_bw * (bk.ao[i] - bk.bo[i]):
            n += 1
            continue
        kk, t_out = n + 1, lim
        while kk < N and t5[kk] < lim:
            if exit_mode == "mid":
                crossed = (c5[kk] <= ma[kk]) if side < 0 else (c5[kk] >= ma[kk])
            else:
                crossed = (c5[kk] <= lo[kk]) if side < 0 else (c5[kk] >= up[kk])
            if crossed or gap_next[kk]:
                t_out = min(t5[kk + 1], lim)
                break
            kk += 1
        j = U.first_after(bk, t_out, 10)
        if j <= i:
            n += 1
            continue
        stop_d = float(up[n] - lo[n])
        r = bk.bracket(i, j, side, 1e9, stop_d)
        if r is None:
            n += 1
            continue
        out[t_in] = U.rec(r)
        si = bk.sf * (bk.ao[i] - bk.bo[i]) + bk.sp * bk.pip
        E = bk.ao[i] + si if side > 0 else bk.bo[i] - si
        hit = np.flatnonzero(bk.bl[i:j] <= E - stop_d) if side > 0 else np.flatnonzero(bk.ah[i:j] >= E + stop_d)
        t_end = bk.idx[i + hit[0]] if len(hit) else bk.idx[j]
        n = max(n + 1, int(t5.searchsorted(t_end, side="right")) - 1)
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- candle fades
def fade_p(bk, setup="pin", tgt=1.0, stp=2.0, min_risk_sp=0.0, hours="all", _cache={}):
    bk._hl()
    key = id(bk)
    if key not in _cache:
        m = bk.m
        mid = pd.DataFrame({"o": bk.mo, "h": bk.mh, "l": bk.ml, "c": 0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy())},
                           index=bk.idx)
        Hh = mid.resample("1h", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min",
                                                                 "c": "last"}).dropna()
        _cache.clear()
        _cache[key] = (Hh, bk.idx.searchsorted(Hh.index))
    Hh, first = _cache[key]
    tl = Hh.index
    o, h, l, c = (Hh[x].to_numpy() for x in "ohlc")
    nH, pip = len(Hh), bk.pip

    def bar_i(n):
        i = int(first[n])
        return i if i < len(bk.idx) and bk.idx[i] < tl[n] + pd.Timedelta(hours=1) else -1

    def ok_hour(i):
        return hours == "all" or 7 <= bk.idx[i].hour <= 16

    res, busy = {}, -1
    for n in range(1, nH - 26):
        if n <= busy:
            continue
        if setup == "pin":
            r_ = h[n] - l[n]
            if r_ <= 0:
                continue
            body = abs(c[n] - o[n])
            lw, uw = min(o[n], c[n]) - l[n], h[n] - max(o[n], c[n])
            if lw >= 0.6 * r_ and body <= 0.3 * r_ and c[n] >= l[n] + 2 / 3 * r_:
                side, stop_px = +1, l[n] - pip
            elif uw >= 0.6 * r_ and body <= 0.3 * r_ and c[n] <= h[n] - 2 / 3 * r_:
                side, stop_px = -1, h[n] + pip
            else:
                continue
            i, j = bar_i(n + 1), bar_i(n + 25)
            if i < 0 or j < 0 or j <= i:
                continue
            risk = abs(bk.mo[i] - stop_px)
            hold = n + 24
        else:
            mh_, ml_ = h[n - 1], l[n - 1]
            if not (h[n] < mh_ and l[n] > ml_):
                continue
            i0, j = bar_i(n + 1), bar_i(n + 25)
            if i0 < 0 or j < 0 or j <= i0:
                continue
            upx = np.flatnonzero(bk.mh[i0:j] > mh_)
            dnx = np.flatnonzero(bk.ml[i0:j] < ml_)
            if not len(upx) and not len(dnx):
                continue
            if len(upx) and len(dnx) and upx[0] == dnx[0]:
                continue
            side, k0 = (+1, upx[0]) if len(upx) and (not len(dnx) or upx[0] < dnx[0]) else (-1, dnx[0])
            i = i0 + k0 + 1
            if i >= j:
                continue
            risk = mh_ - ml_
            hold = n + 25
        sp = bk.ao[i] - bk.bo[i]
        if risk <= 2 * pip or risk < min_risk_sp * sp or not ok_hour(i):
            continue
        r = bk.bracket(i, j, -side, tgt * risk, stp * risk)
        if r is not None:
            res[bk.idx[i]] = U.rec(r)
            busy = hold
    return U.frame(res)


# ----------------------------------------------------------------------------------------------- gold
def gold_p(bk, rng_end=7, exit_h=16, stop_mode="opp", min_rng_sp=0.0):
    m = bk.m
    mid = 0.5 * (m.bid_c.to_numpy() + m.ask_c.to_numpy())
    out = {}
    for d in pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date())):
        a, b = FL.local_ts(d, "Europe/London", 0, 0), FL.local_ts(d, "Europe/London", rng_end, 0)
        e = FL.local_ts(d, "Europe/London", exit_h, 0)
        ia, ib, ie = bk.idx.searchsorted([a, b, e])
        if ib - ia < 150 or ie <= ib or not U.complete(bk, ib, min(ie, len(bk.idx) - 1), 0.9):
            continue
        hi, lo = mid[ia:ib].max(), mid[ia:ib].min()
        if hi - lo < min_rng_sp * (bk.ao[ib] - bk.bo[ib]):
            continue
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
        ref = lo if stop_mode == "opp" else 0.5 * (hi + lo)
        ref_s = hi if stop_mode == "opp" else 0.5 * (hi + lo)
        stop_d = (ent - ref) if side > 0 else (ref_s - ent)
        if stop_d <= 0:
            continue
        r = bk.bracket(i, j, side, 1e9, stop_d)
        if r is not None:
            out[d] = U.rec(r)
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- EURJPY Gotobi
def eurjpy_p(bk, entry=(3, 0), stop=False, _cache={}):
    key = id(bk)
    if key not in _cache:
        s, e = str(bk.idx[0].date()), str(bk.idx[-1].date())
        hol = FL.jp_holidays(range(bk.idx[0].year, bk.idx[-1].year + 1))
        _cache.clear()
        _cache[key] = ([d for d in FL.gotobi_days(s, e, hol) if d.weekday() != 0], bk.avg_range())
    G, rng = _cache[key]
    from fxlab.validation.metrics import trading_day
    out = {}
    for d in G:
        t0 = U.jst(d - pd.Timedelta(days=1), 23, 0) if entry == "prev" else U.jst(d, *entry)
        t1 = U.jst(d, 9, 55)
        i, j = bk.at(t0, 10), bk.at(t1, 65)
        if i < 0 or j < 0 or j <= i:
            continue
        if stop:
            rg = float(rng.get(trading_day(pd.DatetimeIndex([bk.idx[i]]))[0], np.nan))
            if not np.isfinite(rg) or rg <= 0:
                continue
            r = bk.bracket(i, j, +1, 1e9, 0.5 * rg)
        else:
            r = bk.trade_ij(i, j, +1, t0, t1)
        if r is not None:
            out[d] = U.rec(r)
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- index ORB
TRAILS = {"pre": (0.5, 1.0, 1.0, 0.5), "wide": (1.0, 2.0, 2.0, 1.0), "none": None}   # be, start, behind, step


def orb_p(bk, rng_min=30, trail="pre", exit_mode="close", tz="America/New_York", open_hm=(9, 30), close_hm=(16, 0)):
    bk._hl()
    out = {}
    tr = TRAILS[trail]
    for d in pd.bdate_range(str(bk.idx[0].date()), str(bk.idx[-1].date())):
        t_open = FL.local_ts(d, tz, *open_hm)
        t_close = FL.local_ts(d, tz, *close_hm)
        i0, i1, iz = bk.at(t_open, 3), bk.at(t_open + pd.Timedelta(minutes=rng_min), 3), bk.at(t_close, 10)
        if min(i0, i1, iz) < 0 or not (i0 < i1 < iz) or not U.complete(bk, i0, iz, 0.95):
            continue
        hi, lo = bk.mh[i0:i1].max(), bk.ml[i0:i1].min()
        if not hi > lo:
            continue
        mc = 0.5 * (bk.m.bid_c.to_numpy()[i1:iz] + bk.m.ask_c.to_numpy()[i1:iz])
        br = np.flatnonzero((mc > hi) | (mc < lo))
        if not len(br):
            continue
        k = i1 + br[0] + 1
        if k >= iz:
            continue
        end = iz if exit_mode == "close" else min(iz, k + 120)
        side = +1 if mc[br[0]] > hi else -1
        m_in = 0.5 * (bk.ao[k] + bk.bo[k])
        s_in = bk.sf * (bk.ao[k] - bk.bo[k]) + bk.sp * bk.pip
        E = bk.ao[k] + s_in if side > 0 else bk.bo[k] - s_in
        R = (E - lo) if side > 0 else (hi - E)
        if R <= 0:
            continue
        stop, best, res = E - side * R, E, None
        for x in range(k, end):
            lo_x = bk.bl[x] if side > 0 else bk.al[x]
            hi_x = bk.bh[x] if side > 0 else bk.ah[x]
            if (lo_x <= stop) if side > 0 else (hi_x >= stop):
                sx = bk.sf * (bk.ao[x] - bk.bo[x]) + bk.sp * bk.pip
                px = (min(stop, bk.bo[x]) - sx) if side > 0 else (max(stop, bk.ao[x]) + sx)
                res = ((px / E - 1 if side > 0 else 1 - px / E) - 2 * bk.comm, side * (stop / m_in - 1))
                break
            if tr is None:
                continue
            best = max(best, hi_x) if side > 0 else min(best, lo_x)
            gain = (best - E) * side
            be, start, behind, step = tr
            if gain >= be * R:
                stop = max(stop, E) if side > 0 else min(stop, E)
            if gain > start * R:
                lock = math.floor((gain - behind * R) / (step * R)) * step * R
                stop = max(stop, E + lock) if side > 0 else min(stop, E - lock)
        if res is None:
            sx = bk.sf * (bk.ao[end] - bk.bo[end]) + bk.sp * bk.pip
            px = bk.bo[end] - sx if side > 0 else bk.ao[end] + sx
            res = ((px / E - 1 if side > 0 else 1 - px / E) - 2 * bk.comm,
                   side * (0.5 * (bk.ao[end] + bk.bo[end]) / m_in - 1))
        out[d] = res
    return U.frame(out)


# ----------------------------------------------------------------------------------------------- walk-forward
def walk_forward(frames: dict, first_test: int, pre_key, min_trades=30):
    """frames: setting -> DataFrame[net, gross] indexed by trade time. Returns (oos frame, chosen per year,
    pre-registered setting over the same years, hindsight-best setting and its mean)."""
    def yr(f):
        return pd.DatetimeIndex(f.index).year
    years = sorted({y for f in frames.values() for y in yr(f)})
    oos, chosen, pre = [], {}, []
    for Y in years:
        if Y < first_test:
            continue
        best, best_m = None, -np.inf
        for k, f in frames.items():
            tr = f[yr(f) < Y]
            if len(tr) >= min_trades and tr.net.mean() > best_m:
                best, best_m = k, tr.net.mean()
        if best is None:
            continue
        f = frames[best]
        oos.append(f[yr(f) == Y])
        chosen[Y] = str(best)
        fp = frames[pre_key]
        pre.append(fp[yr(fp) == Y])
    O = pd.concat(oos) if oos else pd.DataFrame(columns=["net", "gross"])
    Pre = pd.concat(pre) if pre else pd.DataFrame(columns=["net", "gross"])
    hind = max(frames.items(), key=lambda kv: kv[1].net.mean() if len(kv[1]) >= min_trades else -np.inf)
    return O, chosen, Pre, hind


def tstat(x):
    x = pd.Series(x, dtype=float).dropna()
    return float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def run_rule(name, sym, bk, fn, grid, pre_key, first_test, outdir):
    frames = {}
    for g in grid:
        try:
            frames[g] = fn(bk, *g)
        except Exception as ex:                                          # noqa: BLE001
            print(name, sym, g, "error", repr(ex)[:200], flush=True)
    frames = {k: v for k, v in frames.items() if len(v)}
    O, chosen, Pre, (hk, hf) = walk_forward(frames, first_test, pre_key)
    t = tstat(O.net)
    row = {"rule": name, "instrument": sym, "settings": len(frames), "first_test": first_test,
           "oos_trades": int(len(O)), "oos_mean_bp": float(O.net.mean() * 1e4) if len(O) else np.nan, "oos_t": t,
           "p_one_sided": float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan,
           "oos_gross_bp": float(O.gross.mean() * 1e4) if len(O) else np.nan,
           "oos_2019_2025_bp": float(O[pd.DatetimeIndex(O.index).year >= 2019].net.mean() * 1e4) if len(O) else np.nan,
           "oos_2019_2025_t": tstat(O[pd.DatetimeIndex(O.index).year >= 2019].net) if len(O) else np.nan,
           "prereg_mean_bp": float(Pre.net.mean() * 1e4) if len(Pre) else np.nan, "prereg_t": tstat(Pre.net),
           "hindsight_setting": str(hk), "hindsight_mean_bp": float(hf.net.mean() * 1e4),
           "hindsight_t": tstat(hf.net), "chosen": json.dumps(chosen)}
    O.to_csv(os.path.join(outdir, f"oos_{name}_{sym}.csv"))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "chosen"}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="night,fades,gold,eurjpy,orb")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = set(a.only.split(","))
    rows = []
    if "night" in groups:
        grid = list(itertools.product((21, 22, 23), (2.0, 2.5, 3.0), ("mid", "opp"), (0.0, 5.0, 10.0)))
        for s in ("AUDNZD", "EURGBP", "GBPCAD"):
            bk = U.book(s)
            if bk is not None:
                rows.append(run_rule("night", s, bk, night_p, grid, (21, 2.0, "mid", 0.0), 2010, a.out))
    if "fades" in groups:
        bk = U.book("EURUSD")
        if bk is not None:
            for setup in ("pin", "inside"):
                grid = [(setup, tg, sp, mr, hr) for (tg, sp) in ((1.0, 2.0), (1.5, 1.5), (2.0, 2.0), (1.0, 1.0))
                        for mr in (0.0, 5.0, 10.0) for hr in ("all", "day")]
                rows.append(run_rule(f"fade_{setup}", "EURUSD", bk, fade_p, grid, (setup, 1.0, 2.0, 0.0, "all"), 2010,
                                     a.out))
    if "gold" in groups:
        bk = U.book("XAUUSD")
        if bk is not None:
            grid = list(itertools.product((6, 7, 8), (12, 16, 20), ("opp", "mid"), (0.0, 10.0)))
            rows.append(run_rule("gold", "XAUUSD", bk, gold_p, grid, (7, 16, "opp", 0.0), 2014, a.out))
    if "eurjpy" in groups:
        bk = U.book("EURJPY")
        if bk is not None:
            grid = [(e, s_) for e in ("prev", (3, 0), (6, 0), (8, 0)) for s_ in (False, True)]
            rows.append(run_rule("eurjpy_gotobi", "EURJPY", bk, eurjpy_p, grid, ((3, 0), False), 2010, a.out))
    if "orb" in groups:
        for s in ("USA500IDXUSD", "USATECHIDXUSD"):
            bk = U.book(s)
            if bk is not None:
                grid = list(itertools.product((15, 30, 60), ("pre", "wide", "none"), ("close", "2h")))
                rows.append(run_rule("orb", s, bk, orb_p, grid, (30, "pre", "close"), 2022, a.out))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump({"coverage": U.COVERAGE}, open(os.path.join(a.out, "coverage.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    if len(T):
        print(T.drop(columns=["chosen"]).round(3).to_string())


if __name__ == "__main__":
    main()
