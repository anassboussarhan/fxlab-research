"""Options mechanics not yet tested (tastytrade's), on the same real end-of-day quotes (DoltHub, 2019-2025) and the
same costs as scripts/options_strategy_lab.py. Rules fixed before running; every position ends by 2025-12-31.

PRE-REGISTERED TESTS
Continuous books (one position open at a time, per underlying; SPY, DIA, XLF, XLE):
  entry: the listed expiry 35-57 days out nearest 45 days (build_position, dte 45), on the first snapshot of the data
  and then again on the close where the previous position was closed or expired (same-day redeployment, both
  spreads paid), or on the first close within 5 days that lists an expiry 35-57 days out. Structures: STRG16 (16/16-delta strangle), BSTRG (bullish 30-delta put / 16-delta call
  strangle), SP30, IC16_05, JADE. Exits: H (hold to expiry), T50 (50% of the credit), D21 (21 days left), T50D21.
  T1  each managed book against the hold-to-expiry book of the same structure and underlying: paired monthly
      difference of P&L (Newey-West t), one-sided "management improves".
  T1b each book on its own: monthly P&L, Sharpe, t, worst month, alpha against the underlying's monthly return.
Rolling the untested side (STRG16, BSTRG; H and D21 exits; at most 2 rolls; only with 10+ days left):
  "tested" = the close has covered half the distance from the entry price to a short strike; then the short on the
  other side is bought back and a new one sold in the same expiry at 30 delta on that close (only if it is closer
  to the money than the old one).
  T2  roll-untested book against the same book without rolls (paired monthly difference).
VIX-based allocation (on the T1 books):
  each new position is sized by the VIX at entry, buying power 25% (VIX < 15), 30% (15-20), 35% (20-30),
  40% (30-40), 50% (>= 40), relative to a constant 30%.
  T3  sized book against the constant book: Sharpe and worst month (reported), paired monthly difference of the
      P&L scaled to the same average size (one-sided).
Earnings premium selling (single stocks with an earnings date; windows exactly as H8 in options_edges_lab):
  entry on the last snapshot 2-10 days before the earnings date, exit on the first snapshot 1-7 days after,
  expiry the first listed 3-45 days after the date.
  E1  short ATM straddle (the mirror of H8);  E2  short 16-delta strangle;  E3  short iron condor 16/5.
Re-check of the earlier monthly-schedule result (all underlyings, with the quote guard):
  T0  the five structures on the original monthly 45-day schedule, each exit rule against holding on the SAME
      entries (paired); ETFs and single stocks separately; plus the share of cycles with an impossible loss
      (larger than the underlying's move + 5%) per exit rule, to show how much bad quotes remain.
  Monthly average P&L per trade across names (Newey-West t), one-sided mean > 0.
COSTS: options at mid +- 0.75 x half-spread, USD 0.65 per contract per leg each way; expiring options at intrinsic.
QUOTE GUARD (added after the first run showed impossible exits on XLF / XLE, e.g. a 16-delta strangle losing 26% of
notional on a 2% move): every quote used is checked against the same date's smile without that quote; a mid off by
more than max(25%, 4 half-spreads, 0.1% of the underlying) is replaced by the smile value. Hold-to-expiry books
settle at intrinsic and are barely affected; early exits are.
UNIT: P&L per unit of underlying notional (per share / price at entry).
Usage: python scripts/untested_options_lab.py --stage symbols --chain DIR --group i/n --out DIR
       python scripts/untested_options_lab.py --stage combine --inputs DIRS --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_strategy_lab as L  # noqa: E402

END = L.END
FILL = L.PRIMARY_FILL
COMM = L.COMM
ETFS = ("SPY", "DIA", "XLF", "XLE")
BOOK_STRUCTS = {"STRG16": L.STRUCTS["STRG16"], "BSTRG": [("P", 0.30, -1), ("C", 0.16, -1)],
                "SP30": L.STRUCTS["SP30"], "IC16_05": L.STRUCTS["IC16_05"], "JADE": L.STRUCTS["JADE"]}
EXITS = ("H", "T50", "D21", "T50D21")
ROLL_STRUCTS = ("STRG16", "BSTRG")
ROLL_EXITS = ("H", "D21")
CLEAN = os.environ.get("UO_CLEAN", "1") == "1"      # quote guard on (1) / off (0)
VIX_BP = [(15, 0.25), (20, 0.30), (30, 0.35), (40, 0.40), (np.inf, 0.50)]
BASE_BP = 0.30


def vix_weight(v):
    if not np.isfinite(v):
        return 1.0
    for hi, bp in VIX_BP:
        if v < hi:
            return bp / BASE_BP
    return VIX_BP[-1][1] / BASE_BP


def first_entry(bk, struct):
    for t in bk.dates:
        t = pd.Timestamp(t)
        S = bk.und.px(t)
        if np.isfinite(S) and L.build_position(bk, t, struct, 45, S) is not None:
            return t
    return None


def open_position(bk, t, struct):
    """Legs on date t (or the next snapshot within 5 days if none can be built at t)."""
    for u in bk.dates[(bk.dates >= np.datetime64(t)) & (bk.dates <= np.datetime64(t + pd.Timedelta(days=5)))]:
        u = pd.Timestamp(u)
        S = bk.und.px(u)
        if not np.isfinite(S):
            continue
        bp = L.build_position(bk, u, struct, 45, S)
        if bp is None:
            continue
        legs, front = bp
        if L.strike_parity_breaks(bk, u, legs, front):
            continue
        return u, legs, front
    return None


GUARD = {"checked": 0, "replaced": 0}


def loo_model(bk, t, e, cp, k):
    """Value of (e, cp, k) on t from t's smile WITHOUT k's own quote (leave-one-out), or None."""
    srf = bk.surf.get((t, e, cp))
    if srf is None:
        return None
    F, x, iv, half = srf
    xk = math.log(k / F)
    keep = np.abs(x - xk) > 1e-9
    if keep.sum() < 2:
        return None
    bk.surf[(t, e, cp)] = (F, x[keep], iv[keep], half[keep])
    try:
        m = bk.model(t, e, cp, k)
    finally:
        bk.surf[(t, e, cp)] = srf
    return m


def clean_paths(bk, T, legs):
    """Quote guard: a listed mid far from the leave-one-out smile value of the same date (a bad print in the
    database) is replaced by that value and its half-spread. Bad = |mid - model| > max(25% of the larger of the
    two, 4 half-spreads, 0.1% of the underlying)."""
    for i, (kind, cp, e, k, q) in enumerate(legs):
        if kind != "O" or f"m{i}" not in T:
            continue
        for t in T.index:
            mid, half = T.at[t, f"m{i}"], T.at[t, f"h{i}"]
            if not np.isfinite(mid):
                continue
            m = loo_model(bk, t, e, cp, k)
            GUARD["checked"] += 1
            if m is None:
                continue
            mv, mh = m
            S = bk.und.px(t)
            tol = max(0.25 * max(mv, mid), 4 * max(half, mh), 0.001 * S if np.isfinite(S) else 0.0)
            if abs(mid - mv) > tol:
                T.at[t, f"m{i}"], T.at[t, f"h{i}"] = mv, mh
                GUARD["replaced"] += 1
    return T


def continuous_book(bk, struct, mgmt, vix):
    """One position at a time, redeployed on the exit close. Returns cycles and daily P&L increments (per unit of
    the underlying notional at each cycle's entry, scaled to the first cycle's notional is NOT done: each cycle is a
    return on its own notional, so increments are returns)."""
    t = first_entry(bk, struct)
    cycles, incs = [], []
    guard = 0
    while t is not None and t < END and guard < 400:
        guard += 1
        op = open_position(bk, t, struct)
        if op is None:
            nxt = bk.dates[bk.dates > np.datetime64(t + pd.Timedelta(days=5))]
            t = pd.Timestamp(nxt[0]) if len(nxt) else None
            continue
        d, legs, front = op
        T = L.cycle_paths(bk, d, legs, front)
        if CLEAN:
            T = clean_paths(bk, T, legs)
        r = L.evaluate(bk, d, legs, front, T, mgmt, FILL)
        if r is None:
            nxt = bk.dates[bk.dates > np.datetime64(d)]
            t = pd.Timestamp(nxt[0]) if len(nxt) else None
            continue
        S0 = bk.und.px(d)
        inc = L.mtm_series(bk, d, legs, T, r["exit"], S0, r["pnl"], FILL)
        vv = vix.loc[:d]
        v = float(vv.iloc[-1]) if len(vv) else np.nan
        cycles.append({"entry": d, "exit": r["exit"], "expiry": front, "pnl": r["pnl"], "credit": r["credit"],
                       "early": r["early"], "days": r["days"], "vix": v, "w_vix": vix_weight(v),
                       "und_ret": bk.und.tr(d, r["exit"])})
        incs.append(inc.rename(len(cycles) - 1))
        t = pd.Timestamp(r["exit"])
        if t <= d:
            nxt = bk.dates[bk.dates > np.datetime64(d)]
            t = pd.Timestamp(nxt[0]) if len(nxt) else None
    return pd.DataFrame(cycles), incs


def _iv_delta(bk, t, e, cp, k, S, mid):
    """|delta| from the option's own mid (Black-76 on the parity forward approximated by S)."""
    T = max((e - t).days, 1) / 365.0
    if not (np.isfinite(mid) and mid > 0 and S > 0):
        return np.nan
    lo, hi = 0.01, 5.0
    for _ in range(60):
        v = 0.5 * (lo + hi)
        p = float(L._b76(cp == "C", S, k, T, v))
        if p > mid:
            hi = v
        else:
            lo = v
    return abs(L.bs_delta(cp, S, k, T, 0.5 * (lo + hi)))


def clean_path(bk, e, cp, k, p):
    """The quote guard on one contract's path (DataFrame with mid, half)."""
    if not CLEAN or not len(p):
        return p
    T = pd.DataFrame({"m0": p["mid"], "h0": p["half"]})
    T = clean_paths(bk, T, [("O", cp, e, k, 1)])
    return pd.DataFrame({"mid": T.m0, "half": T.h0})


def roll_untested_book(bk, struct, mgmt, max_rolls=2):
    """Continuous strangle book with the untested side rolled toward the money (see docstring)."""
    t = first_entry(bk, struct)
    cycles, incs = [], []
    guard = 0
    while t is not None and t < END and guard < 400:
        guard += 1
        op = open_position(bk, t, struct)
        if op is None:
            nxt = bk.dates[bk.dates > np.datetime64(t + pd.Timedelta(days=5))]
            t = pd.Timestamp(nxt[0]) if len(nxt) else None
            continue
        d, legs, front = op
        S0 = bk.und.px(d)
        open_legs = []                       # [cp, k, q, path]
        cash = 0.0
        for kind, cp, e, k, q in legs:
            p = clean_path(bk, e, cp, k, bk.path(e, cp, k, d, front))
            if not len(p) or p.index[0] != d:
                open_legs = None
                break
            open_legs.append([cp, k, q, p])
            cash -= q * (float(p.mid.iloc[0]) + np.sign(q) * FILL * float(p.half.iloc[0])) + COMM * abs(q)
        if open_legs is None:
            nxt = bk.dates[bk.dates > np.datetime64(d)]
            t = pd.Timestamp(nxt[0]) if len(nxt) else None
            continue
        credit = cash
        days = [pd.Timestamp(x) for x in bk.dates if np.datetime64(d) <= x < np.datetime64(front)]
        marks = {}
        rolls, exit_d, early = 0, None, False
        for dd in days:
            S = bk.und.px(dd)
            val = 0.0
            ok = True
            for cp, k, q, p in open_legs:
                m = p.mid.reindex([dd]).iloc[0] if dd in p.index else np.nan
                if not np.isfinite(m):
                    ok = False
                    break
                val += q * m
            if ok:
                marks[dd] = cash + val
            if dd == d:
                continue
            left = (front - dd).days
            if mgmt == "D21" and left <= 21 and ok:
                for cp, k, q, p in open_legs:
                    cash += q * (float(p.mid.loc[dd]) - np.sign(q) * FILL * float(p.half.loc[dd])) - COMM * abs(q)
                open_legs, exit_d, early = [], dd, True
                break
            if rolls < max_rolls and left >= 10 and ok and np.isfinite(S):
                put = [x for x in open_legs if x[0] == "P" and x[2] < 0]
                call = [x for x in open_legs if x[0] == "C" and x[2] < 0]
                if put and call:
                    kp, kc = put[0][1], call[0][1]
                    side = None
                    if S <= S0 - 0.5 * (S0 - kp):
                        side = "C"                    # puts tested -> roll the call down
                    elif S >= S0 + 0.5 * (kc - S0):
                        side = "P"                    # calls tested -> roll the put up
                    if side is not None:
                        k_new = bk.pick(dd, front, side, 0.30, S)
                        old = call[0] if side == "C" else put[0]
                        closer = k_new is not None and ((side == "C" and k_new < old[1]) or (side == "P" and k_new > old[1]))
                        if closer:
                            pn = clean_path(bk, front, side, k_new, bk.path(front, side, k_new, dd, front))
                            if len(pn) and pn.index[0] == dd and dd in old[3].index:
                                cash += old[2] * (float(old[3].mid.loc[dd]) - np.sign(old[2]) * FILL *
                                                  float(old[3].half.loc[dd])) - COMM * abs(old[2])
                                cash -= -1 * (float(pn.mid.iloc[0]) - FILL * float(pn.half.iloc[0])) + COMM
                                open_legs = [x for x in open_legs if x is not old] + [[side, k_new, -1, pn]]
                                rolls += 1
                                S0 = S                # a new reference price after the roll
                                val = sum(q * float(p.mid.loc[dd]) for cp, k, q, p in open_legs
                                          if dd in p.index)
                                marks[dd] = cash + val
        if exit_d is None:
            exit_d = front
            Sx = bk.und.px(front)
            for cp, k, q, p in open_legs:
                cash += q * (max(Sx - k, 0.0) if cp == "C" else max(k - Sx, 0.0))
        S_entry = bk.und.px(d)
        pnl = cash / S_entry
        v = pd.Series(marks).sort_index() / S_entry
        v.loc[pd.Timestamp(exit_d)] = pnl
        v = v.sort_index()
        inc = v.diff()
        inc.iloc[0] = v.iloc[0]
        cycles.append({"entry": d, "exit": exit_d, "expiry": front, "pnl": pnl, "credit": credit / S_entry,
                       "early": early, "rolls": rolls, "days": int((exit_d - d).days),
                       "und_ret": bk.und.tr(d, exit_d)})
        incs.append(inc.rename(len(cycles) - 1))
        t = pd.Timestamp(exit_d)
        if t <= d:
            nxt = bk.dates[bk.dates > np.datetime64(d)]
            t = pd.Timestamp(nxt[0]) if len(nxt) else None
    return pd.DataFrame(cycles), incs


def monthly_schedule(bk, struct):
    """T0: the original monthly schedule (entry about 45 days before each third Friday) with the quote guard;
    every exit rule on the same entries. Rows: entry, mgmt, pnl."""
    rows = []
    for d in L.entry_dates(bk, 45):
        S = bk.und.px(d)
        if not np.isfinite(S):
            continue
        bp = L.build_position(bk, d, struct, 45, S)
        if bp is None:
            continue
        legs, front = bp
        if L.strike_parity_breaks(bk, d, legs, front):
            continue
        T = L.cycle_paths(bk, d, legs, front)
        if CLEAN:
            T = clean_paths(bk, T, legs)
        for mg in EXITS:
            r = L.evaluate(bk, d, legs, front, T, mg, FILL)
            if r is not None:
                rows.append({"entry": d, "exit": r["exit"], "mgmt": mg, "pnl": r["pnl"], "credit": r["credit"],
                             "und_ret": bk.und.tr(d, r["exit"])})
    return pd.DataFrame(rows)


def earnings_trades(sym, bk, und, earn):
    out = []
    import options_edges_lab as OE
    for E in earn:
        if E > END - pd.Timedelta(days=10) or E < pd.Timestamp(bk.dates[0]) + pd.Timedelta(days=10):
            continue
        pre = [pd.Timestamp(t) for t in bk.dates if E - pd.Timedelta(days=10) <= t <= E - pd.Timedelta(days=2)]
        post = [pd.Timestamp(t) for t in bk.dates if E + pd.Timedelta(days=1) <= t <= E + pd.Timedelta(days=7)]
        if not pre or not post:
            continue
        d0, d1 = pre[-1], post[0]
        S0 = und.px(d0)
        if not np.isfinite(S0) or und.has_event(d0, d1):
            continue
        ex = [e for e in bk.expiries(d0) if E + pd.Timedelta(days=3) <= e <= E + pd.Timedelta(days=45)]
        if not ex:
            continue
        e = ex[0]
        k = OE.atm_strike(bk, d0, e, S0)
        kp16, kc16 = bk.pick(d0, e, "P", 0.16, S0), bk.pick(d0, e, "C", 0.16, S0)
        kp05, kc05 = bk.pick(d0, e, "P", 0.05, S0), bk.pick(d0, e, "C", 0.05, S0)
        books = {}
        if k is not None:
            books["E1_short_straddle"] = [("P", k, -1), ("C", k, -1)]
        if kp16 is not None and kc16 is not None and kp16 < S0 < kc16:
            books["E2_short_strangle16"] = [("P", kp16, -1), ("C", kc16, -1)]
            if kp05 is not None and kc05 is not None and kp05 < kp16 and kc05 > kc16:
                books["E3_short_iron_condor16_05"] = [("P", kp16, -1), ("P", kp05, 1), ("C", kc16, -1), ("C", kc05, 1)]
        for name, legs in books.items():
            v0, v1, ok = 0.0, 0.0, True
            for cp, kk, q in legs:
                p = clean_path(bk, e, cp, kk, bk.path(e, cp, kk, d0, d1))
                if not len(p) or p.index[0] != d0 or d1 not in p.index:
                    ok = False
                    break
                v0 += -q * (p.mid.loc[d0] + np.sign(q) * FILL * p.half.loc[d0])       # cash at entry
                v1 += q * (p.mid.loc[d1] - np.sign(q) * FILL * p.half.loc[d1])        # cash at exit
            if not ok:
                continue
            pnl = v0 + v1 - 2 * COMM * sum(abs(q) for _, _, q in legs)
            out.append({"symbol": sym, "test": name, "earnings": E, "entry": d0, "exit": d1, "expiry": e,
                        "pnl": pnl / S0, "credit": v0 / S0})
    return out


def stage_symbols(a):
    X = pd.read_csv(os.path.join(a.chain, "closes.csv"), parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    vix = X["VIX"].dropna()
    syms = sorted(f[len("chain_"):-len(".parquet")] for f in os.listdir(a.chain)
                  if f.startswith("chain_") and f.endswith(".parquet"))
    i, n = map(int, a.group.split("/"))
    syms = syms[i::n]
    E = {}
    ef = os.path.join(a.chain, "earnings.csv")
    if os.path.exists(ef):
        Ed = pd.read_csv(ef)
        Ed.columns = [c.lower() for c in Ed.columns]
        Ed["date"] = pd.to_datetime(Ed["date"], errors="coerce")
        E = {k: pd.DatetimeIndex(sorted(v.dropna().unique())) for k, v in Ed.groupby("act_symbol").date}
    meta, cyc, ern, mtm, sched = {}, [], [], [], []
    for sym in syms:
        if f"{sym}_raw" not in X.columns:
            continue
        try:
            und = L.Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
            C = L.load_chain(a.chain, sym)
            if C.empty:
                continue
            C = L.trim_chain(C, und)
            bk = L.Book(C, und)
        except Exception as ex:                                       # noqa: BLE001
            meta[sym] = {"error": repr(ex)[:300]}
            continue
        m = {"chain_dates": int(len(bk.dates))}
        for sname, struct in BOOK_STRUCTS.items():
            t0 = monthly_schedule(bk, struct)
            if len(t0):
                t0["symbol"], t0["struct"] = sym, sname
                sched.append(t0)
        if sym in ETFS:
            for sname, struct in BOOK_STRUCTS.items():
                for mg in EXITS:
                    c, incs = continuous_book(bk, struct, mg, vix)
                    if len(c):
                        c["symbol"], c["struct"], c["mgmt"], c["book"] = sym, sname, mg, "T1"
                        cyc.append(c)
                        for j, s in enumerate(incs):
                            mtm.append(pd.DataFrame({"symbol": sym, "struct": sname, "mgmt": mg, "book": "T1",
                                                     "cycle": j, "w_vix": c.w_vix.iloc[j], "day": s.index,
                                                     "inc": s.to_numpy()}))
                    print(sym, sname, mg, len(c), flush=True)
            for sname in ROLL_STRUCTS:
                for mg in ROLL_EXITS:
                    c, incs = roll_untested_book(bk, BOOK_STRUCTS[sname], mg)
                    if len(c):
                        c["symbol"], c["struct"], c["mgmt"], c["book"] = sym, sname, mg, "T2"
                        cyc.append(c)
                        for j, s in enumerate(incs):
                            mtm.append(pd.DataFrame({"symbol": sym, "struct": sname, "mgmt": mg, "book": "T2",
                                                     "cycle": j, "w_vix": 1.0, "day": s.index, "inc": s.to_numpy()}))
                    print(sym, sname, mg, "roll-untested", len(c), int(c.rolls.sum()) if len(c) else 0, flush=True)
            m["paths_modeled"] = int(bk.n_modeled)
        elif sym in E:
            r = earnings_trades(sym, bk, und, E[sym])
            ern.extend(r)
            m["earnings_trades"] = len(r)
        m["quote_guard"] = dict(GUARD)
        GUARD["checked"] = GUARD["replaced"] = 0
        meta[sym] = m
        print(sym, m, flush=True)
    tag = a.group.replace("/", "of")
    if cyc:
        pd.concat(cyc, ignore_index=True).to_csv(os.path.join(a.out, f"cycles_{tag}.csv.gz"), index=False)
    if mtm:
        pd.concat(mtm, ignore_index=True).to_csv(os.path.join(a.out, f"mtm_{tag}.csv.gz"), index=False)
    if ern:
        pd.DataFrame(ern).to_csv(os.path.join(a.out, f"earnings_{tag}.csv.gz"), index=False)
    if sched:
        pd.concat(sched, ignore_index=True).to_csv(os.path.join(a.out, f"schedule_{tag}.csv.gz"), index=False)
    json.dump(meta, open(os.path.join(a.out, f"meta_{tag}.json"), "w"), indent=1, default=str)


def nw_t(x, lag=2):
    x = np.asarray(pd.Series(x).dropna(), float)
    return L.nw_t(x, lag) if len(x) >= 8 else np.nan


def monthly(series):
    s = pd.Series(series)
    return s.groupby(pd.DatetimeIndex(s.index).to_period("M")).sum()


def describe(m, und_m=None):
    m = m.dropna()
    out = {"months": int(len(m)), "mean_month_pct": float(m.mean() * 100), "t": nw_t(m),
           "sharpe": float(m.mean() / m.std() * math.sqrt(12)) if m.std() > 0 else np.nan,
           "worst_month_pct": float(m.min() * 100),
           "max_drawdown_pct": float((m.cumsum() - m.cumsum().cummax()).min() * 100)}
    if und_m is not None:
        j = pd.concat([m.rename("y"), und_m.rename("x")], axis=1).dropna()
        if len(j) > 12 and j.x.var() > 0:
            b = np.cov(j.y, j.x)[0, 1] / j.x.var()
            res = j.y - b * j.x
            out.update({"beta": float(b), "alpha_month_pct": float(res.mean() * 100), "alpha_t": nw_t(res)})
    return out


def stage_combine(a):
    dirs = [d for d in a.inputs.split(",") if d]
    C, M, R, Sd = [], [], [], []
    for d in dirs:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if f.startswith("cycles_"):
                C.append(pd.read_csv(p, parse_dates=["entry", "exit", "expiry"]))
            elif f.startswith("mtm_"):
                M.append(pd.read_csv(p, parse_dates=["day"]))
            elif f.startswith("earnings_"):
                R.append(pd.read_csv(p, parse_dates=["earnings", "entry", "exit", "expiry"]))
            elif f.startswith("schedule_"):
                Sd.append(pd.read_csv(p, parse_dates=["entry", "exit"]))
    os.makedirs(a.out, exist_ok=True)
    tests, books = [], []
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index() if a.closes else None
    if M:
        M = pd.concat(M, ignore_index=True)
        M = M[M.day <= END]
        keyc = ["book", "symbol", "struct", "mgmt"]
        series = {}
        for key, g in M.groupby(keyc):
            series[key] = monthly(g.groupby("day").inc.sum())
            series[key + ("vix",)] = monthly((g.inc * g.w_vix).groupby(g.day).sum())
        for key, m in series.items():
            sym = key[1]
            und_m = None
            if X is not None and f"{sym}_adj" in X.columns:
                px = X[f"{sym}_adj"].dropna()
                und_m = px.groupby(px.index.to_period("M")).last().pct_change()
            books.append({"book": key[0], "symbol": sym, "struct": key[2], "mgmt": key[3],
                          "sizing": "vix" if len(key) == 5 else "constant", **describe(m, und_m)})
        # T1: management vs hold (paired monthly differences), per underlying and pooled over the 4 ETFs
        for sname in BOOK_STRUCTS:
            for mg in EXITS[1:]:
                diffs = []
                for sym in ETFS:
                    a_, b_ = series.get(("T1", sym, sname, mg)), series.get(("T1", sym, sname, "H"))
                    if a_ is None or b_ is None:
                        continue
                    d = (a_ - b_).dropna()
                    diffs.append(d.rename(sym))
                    tests.append({"test": f"T1 {sname} {mg} vs hold, {sym}", "family": "T1", **describe(d)})
                if diffs:
                    pooled = pd.concat(diffs, axis=1).mean(axis=1)
                    tests.append({"test": f"T1 {sname} {mg} vs hold, 4 ETFs", "family": "T1", "primary": True,
                                  **describe(pooled)})
        # T2: roll untested vs the same exit without rolls
        for sname in ROLL_STRUCTS:
            for mg in ROLL_EXITS:
                diffs = []
                for sym in ETFS:
                    a_, b_ = series.get(("T2", sym, sname, mg)), series.get(("T1", sym, sname, mg))
                    if a_ is None or b_ is None:
                        continue
                    d = (a_ - b_).dropna()
                    diffs.append(d)
                    tests.append({"test": f"T2 roll untested {sname} {mg}, {sym}", "family": "T2", **describe(d)})
                if diffs:
                    tests.append({"test": f"T2 roll untested {sname} {mg}, 4 ETFs", "family": "T2", "primary": True,
                                  **describe(pd.concat(diffs, axis=1).mean(axis=1))})
        # T3: VIX sizing vs constant, scaled to the same average size
        for sname in BOOK_STRUCTS:
            for mg in EXITS:
                diffs = []
                for sym in ETFS:
                    v, c = series.get(("T1", sym, sname, mg, "vix")), series.get(("T1", sym, sname, mg))
                    if v is None or c is None:
                        continue
                    w = M[(M.book == "T1") & (M.symbol == sym) & (M.struct == sname) & (M.mgmt == mg)]
                    scale = 1.0 / w.groupby("cycle").w_vix.first().mean()
                    diffs.append((v * scale - c).dropna())
                if diffs:
                    tests.append({"test": f"T3 VIX sizing {sname} {mg}, 4 ETFs", "family": "T3", "primary": True,
                                  **describe(pd.concat(diffs, axis=1).mean(axis=1))})
        pd.DataFrame(books).to_csv(os.path.join(a.out, "books.csv"), index=False)
        M.groupby(keyc + ["day"]).inc.sum().reset_index().to_csv(os.path.join(a.out, "book_daily.csv.gz"),
                                                                  index=False)
    if C:
        pd.concat(C, ignore_index=True).to_csv(os.path.join(a.out, "cycles.csv.gz"), index=False)
    if R:
        R = pd.concat(R, ignore_index=True)
        R = R[R.exit <= END]
        R.to_csv(os.path.join(a.out, "earnings_trades.csv.gz"), index=False)
        for name, g in R.groupby("test"):
            m = g.groupby(g.entry.dt.to_period("M")).pnl.mean()
            yrs = g.groupby(g.entry.dt.year).pnl.mean()
            tests.append({"test": name, "family": "E", "primary": True, **describe(m), "trades": int(len(g)),
                          "mean_trade_pct": float(g.pnl.mean() * 100), "median_trade_pct": float(g.pnl.median() * 100),
                          "worst_trade_pct": float(g.pnl.min() * 100), "win_rate": float((g.pnl > 0).mean()),
                          "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}"})
    if Sd:
        Sd = pd.concat(Sd, ignore_index=True)
        Sd = Sd[Sd.exit <= END]
        Sd.to_csv(os.path.join(a.out, "schedule_cycles.csv.gz"), index=False)
        Sd["group"] = np.where(Sd.symbol.isin(ETFS), "4 ETFs", "single stocks")
        Sd["implausible"] = Sd.pnl < -(Sd.und_ret.abs() + 0.05)
        imp = Sd.groupby(["group", "mgmt"]).implausible.mean().unstack()
        imp.to_csv(os.path.join(a.out, "schedule_implausible_share.csv"))
        W = Sd.pivot_table(index=["symbol", "struct", "entry"], columns="mgmt", values="pnl")
        for sname in BOOK_STRUCTS:
            for grp in ("4 ETFs", "single stocks"):
                for mg in EXITS[1:]:
                    w = W.loc[[i for i in W.index if i[1] == sname and ((i[0] in ETFS) == (grp == "4 ETFs"))]]
                    if mg not in w or "H" not in w or not len(w):
                        continue
                    dif = (w[mg] - w["H"]).dropna()
                    ent = pd.DatetimeIndex([i[2] for i in dif.index])
                    mo = dif.groupby(ent.to_period("M")).mean()
                    tests.append({"test": f"T0 {sname} {mg} vs hold, monthly schedule, {grp}", "family": "T0",
                                  "primary": grp == "4 ETFs", **describe(mo), "cycles": int(len(dif)),
                                  "mean_cycle_diff_bp": float(dif.mean() * 1e4),
                                  "share_better": float((dif > 0).mean())})
    T = pd.DataFrame(tests)
    if len(T):
        from scipy import stats as st
        T["p_one_sided"] = [float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan for t in T.t]
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["symbols", "combine"])
    ap.add_argument("--chain")
    ap.add_argument("--group", default="0/1")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--closes", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stage_symbols(a) if a.stage == "symbols" else stage_combine(a)


if __name__ == "__main__":
    main()
