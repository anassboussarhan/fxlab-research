"""Options edges, second batch: the remaining hedged, event and calendar trades from the research, on the same real
end-of-day quotes (DoltHub, 2019-2025) as scripts/options_edges_lab.py. Parameters fixed in advance (nothing tuned).
Development data only: quotes after 2025-12-31 are dropped and every position ends by then.

PRE-REGISTERED TESTS (one-sided, mean > 0; Benjamini-Hochberg over E1-E14; 2019-2022 vs 2023-2025 reported):
  E1  SPY delta-hedged risk reversal: monthly, the listed expiry 20-42 days out (nearest 30), SELL the 25-delta put,
      BUY the 25-delta call, delta-hedged at every snapshot close, held to expiry (index skew premium; Bakshi,
      Kapadia & Madan 2003).
  E2  DIA, the same.
  E3  SPY delta-hedged SHORT 10-delta put, same schedule (crash premium; Bondarenko 2014).
  E4  DIA, the same.
  E5  Dispersion (Driessen, Maenhout & Vilkov 2009): each month, SHORT the SPY delta-hedged ATM straddle (1 unit of
      notional) and LONG delta-hedged ATM straddles on every stock entered that month (equal weight, 1 unit in
      total); at least 20 stocks.
  E6  Weekend effect, SPY (Jones & Shemesh 2018): SELL the ATM straddle of the 20-42-day expiry at the last close of
      the week, delta-hedged, buy it back at the next trading day's close (only when that next trading day is a
      snapshot); all costs.
  E7  E6 on stocks: equal weight across stocks each weekend (3 bp hedge).
  E8  The weekend effect itself on SPY, gross (mid to mid, no costs): each week, the Friday-to-Monday P&L of E6
      minus the average one-trading-day P&L of the same hedged straddle on that week's weekdays (daily snapshots,
      2024-2025).
  E9  FOMC: SELL the SPY ATM straddle (listed expiry 14-60 days out, nearest 30) delta-hedged at the last snapshot
      1-5 days before each scheduled FOMC statement; buy back at the statement day's close (or the first snapshot
      up to 4 days after).
  E10 CPI: the same around each CPI release (8:30 am; exit at that day's close or the first snapshot after).
  E11 SPY delta-hedged calendar straddle (term structure of the variance premium): SELL the front (20-42 days) ATM
      straddle, BUY the ATM straddle of the expiry 14-45 days later (nearest 28) at the same strike; to the front
      expiry, the back straddle sold at the quote.
  E12 DIA, the same.
  E13 SPY short variance by a strip (Carr & Madan 1998): SELL every listed out-of-the-money put and call of the
      20-42-day expiry with weights 2 dK / K^2, delta-hedged, held to expiry.
  E14 DIA, the same.
DIAGNOSTIC (not a trading test): box-spread arbitrage on SPY / DIA: on every snapshot, every pair of strikes with
  calls and puts quoted within 85-100% of the forward (out-of-the-money puts, so the options' American early
  exercise matters little) and at least 2% apart, on the 20-42-day expiry: does buying (at the asks) or selling
  (at the bids) the box beat the 13-week T-bill after commissions?
COSTS: options at mid +- 0.75 x half-spread and USD 0.65 per contract; hedges 1 bp of traded notional on SPY / DIA,
3 bp on stocks; expiring options at intrinsic. Unit: P&L per unit of underlying notional at entry.
Usage: python scripts/options_edges2_lab.py --stage symbols --chain DIR --closes FILE --group i/n --out DIR
       python scripts/options_edges2_lab.py --stage combine --inputs DIRS --out DIR"""
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
import options_strategy_lab as L  # noqa: E402
from options_edges_lab import atm_strike, smile_delta, stats  # noqa: E402

END = L.END
FILL = 0.75
COMM = L.COMM
INDEX = ("SPY", "DIA")
ETFS = ("SPY", "DIA", "XLF", "XLE")
FOMC = [  # scheduled statement days (federalreserve.gov); March 2020's scheduled meeting was replaced by
    # unscheduled ones (3 and 15 March), which are excluded
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"]
COUNT = {"positions": 0, "delta_fallbacks": 0, "deltas": 0}


# ----------------------------------------------------------------------------------------------- engine
def position(bk, und, d, exit_d, legs, hedge, hedge_cost, fill=FILL, comm=COMM, cash_settle=True):
    """legs [(cp, expiry, strike, qty)] opened at d's close at mid +- fill x half-spread plus commission; closed at
    exit_d: legs expiring on exit_d settle at intrinsic against the close, the others are closed at mid -+ fill x
    half-spread plus commission (valued from the day's smile when unlisted). If hedge, the total option delta
    (Black-76 on each day's smile) is offset with the underlying at every snapshot close from d to the last one
    before exit_d. Returns P&L per unit of notional (the close at d), the premium and the hedge cost."""
    S0 = und.px(d)
    if not np.isfinite(S0) or exit_d <= d:
        return None
    days = [pd.Timestamp(t) for t in bk.dates if np.datetime64(d) <= t < np.datetime64(exit_d)]
    if not days or days[0] != d:
        return None
    paths = {}
    for cp, e, k, q in legs:
        if e < exit_d:
            return None
        p = bk.path(e, cp, k, d, exit_d)
        if not len(p) or p.index[0] != d:
            return None
        if e > exit_d and exit_d not in p.index:
            return None
        paths[(cp, e, k)] = p
    cash = 0.0
    for cp, e, k, q in legs:
        p = paths[(cp, e, k)]
        cash -= q * (float(p.mid.iloc[0]) + np.sign(q) * fill * float(p.half.iloc[0]))
    cash -= comm * sum(abs(q) for *_, q in legs)
    premium = cash
    stock, cost, held = 0.0, 0.0, 0.0
    if hedge:
        for i, t in enumerate(days):
            St = und.px(t)
            dl = 0.0
            for cp, e, k, q in legs:
                x = smile_delta(bk, t, e, cp, k)
                COUNT["deltas"] += 1
                if x is None:
                    COUNT["delta_fallbacks"] += 1
                    x = (1.0 if St > k else 0.0) if cp == "C" else (-1.0 if St < k else 0.0)
                dl += q * x
            cost += abs(-dl - held) * St * hedge_cost
            held = -dl
            t1 = days[i + 1] if i + 1 < len(days) else exit_d
            a0, a1 = und.adj.loc[:t].iloc[-1], und.adj.loc[:t1].iloc[-1]
            stock += held * St * (a1 / a0 - 1.0)
        cost += abs(held) * und.px(exit_d) * hedge_cost
    Sx = und.px(exit_d)
    close = 0.0
    for cp, e, k, q in legs:
        if e == exit_d and cash_settle:
            close += q * (max(Sx - k, 0.0) if cp == "C" else max(k - Sx, 0.0))
        else:
            p = paths[(cp, e, k)]
            m, h = float(p.mid.loc[exit_d]), float(p.half.loc[exit_d])
            close += q * (m - np.sign(q) * fill * h) - comm * abs(q)
    COUNT["positions"] += 1
    pnl = cash + close + stock - cost
    return {"pnl": pnl / S0, "premium": premium / S0, "hedge_cost": cost / S0, "days": int((exit_d - d).days)}


def front_expiry(bk, d, lo=20, hi=42, tgt=30):
    c = [e for e in bk.expiries(d) if lo <= (e - d).days <= hi and e <= END]
    return min(c, key=lambda e: abs((e - d).days - tgt)) if c else None


def entry_ok(bk, und, d):
    S = und.px(d)
    if not np.isfinite(S):
        return False
    Sp = bk.parity_px(d)
    return not (np.isfinite(Sp) and abs(math.log(Sp / S)) > 0.03)


def straddle_legs(bk, d, e, S, q):
    k = atm_strike(bk, d, e, S)
    return None if k is None else [("P", e, k, q), ("C", e, k, q)]


def same_strike(bk, d, e, k, cp):
    g = bk.by_day[d]
    ks = g[(g.expiration == e) & (g.cp == cp) & (g.bid > 0)].strike.to_numpy()
    if not len(ks):
        return None
    j = ks[np.argmin(np.abs(ks - k))]
    return float(j) if abs(j - k) <= 0.01 * k else None


def strip_legs(bk, d, e, S):
    """Out-of-the-money puts below and calls at / above the forward, weights 2 dK / K^2 (per unit of notional S)."""
    s = bk.surf.get((d, e, "C")) or bk.surf.get((d, e, "P"))
    if s is None:
        return None, None
    F = s[0]
    g = bk.by_day[d]
    g = g[(g.expiration == e) & (g.bid > 0)]
    P = sorted(g[(g.cp == "P") & (g.strike < F)].strike.unique())
    C = sorted(g[(g.cp == "C") & (g.strike >= F)].strike.unique())
    ks = P + C
    if len(P) < 3 or len(C) < 3:
        return None, None
    legs = []
    for i, k in enumerate(ks):
        dk = (ks[min(i + 1, len(ks) - 1)] - ks[max(i - 1, 0)]) / (2 if 0 < i < len(ks) - 1 else 1)
        w = 2 * dk / k ** 2 * S
        legs.append(("P" if k < F else "C", e, float(k), -w))
    return legs, F


def next_trading_day(und, d):
    idx = und.raw.index
    j = idx.searchsorted(d, side="right")
    return idx[j] if j < len(idx) else None


# ----------------------------------------------------------------------------------------------- per symbol
def run_symbol(sym, chain_dir, und, cpi):
    C = L.load_chain(chain_dir, sym)
    if C.empty:
        return None
    C = L.trim_chain(C, und)
    bk = L.Book(C, und)
    hc = 1e-4 if sym in INDEX else 3e-4
    dates = pd.DatetimeIndex(bk.dates)
    dset = set(dates)
    rows = []

    def add(kind, d, r, **kw):
        if r is not None:
            rows.append({"symbol": sym, "kind": kind, "entry": d, **kw, **r})

    # monthly positions
    for d in L.entry_dates(bk, 30):
        if not entry_ok(bk, und, d):
            continue
        S = und.px(d)
        e = front_expiry(bk, d)
        if e is None or und.has_event(d, e):
            continue
        ex = dates[dates.searchsorted(e, side="right") - 1]                    # last snapshot on or before e
        exit_d = e                                                             # settle on e (intrinsic)
        if sym in INDEX:
            kp, kc = bk.pick(d, e, "P", 0.25, S), bk.pick(d, e, "C", 0.25, S)
            if kp is not None and kc is not None and kp < kc:
                add("risk_reversal_hedged", d, position(bk, und, d, exit_d, [("P", e, kp, -1), ("C", e, kc, 1)], True, hc))
            k10 = bk.pick(d, e, "P", 0.10, S)
            if k10 is not None:
                add("short_put10_hedged", d, position(bk, und, d, exit_d, [("P", e, k10, -1)], True, hc))
            sl = straddle_legs(bk, d, e, S, -1)
            if sl is not None:
                add("short_straddle_hedged", d, position(bk, und, d, exit_d, sl, True, hc))
                later = [b for b in bk.expiries(d) if 14 <= (b - e).days <= 45 and b <= END + pd.Timedelta(days=60)]
                if later:
                    b = min(later, key=lambda b: abs((b - e).days - 28))
                    k = sl[0][2]
                    kb = same_strike(bk, d, b, k, "P")
                    if kb is not None and same_strike(bk, d, b, k, "C") == kb:
                        cal = sl + [("P", b, kb, 1), ("C", b, kb, 1)]
                        if ex == e:
                            add("calendar_straddle_hedged", d, position(bk, und, d, e, cal, True, hc))
                        else:                                      # front expiry not a snapshot: close the day before
                            add("calendar_straddle_hedged", d, position(bk, und, d, ex, cal, True, hc,
                                                                        cash_settle=False))
            legs, F = strip_legs(bk, d, e, S)
            if legs is not None:
                r = position(bk, und, d, exit_d, legs, True, hc)
                if r is not None:
                    T = (e - d).days / 365.0
                    mid = sum(-q * float(bk.path(ee, cp, k, d, d).mid.iloc[0]) for cp, ee, k, q in legs) / S
                    kv = math.sqrt(max(mid, 1e-12) / T)
                    r.update({"strip_var_strike": mid / T, "strip_vol": kv,
                              "pnl_vol_points": (r["pnl"] / T) / (2 * kv) * 100, "n_strikes": len(legs)})
                    add("variance_strip_hedged", d, r)
        else:
            sl = straddle_legs(bk, d, e, S, 1)
            if sl is not None:
                add("long_straddle_hedged", d, position(bk, und, d, exit_d, sl, True, hc))
    # weekends (and, on SPY, one-day weekday holds for the gross comparison)
    for i, d in enumerate(dates[:-1]):
        nx = dates[i + 1]
        td = next_trading_day(und, d)
        if td is None or nx != td or not entry_ok(bk, und, d):
            continue
        weekend = nx.isocalendar()[1] != d.isocalendar()[1] or (nx - d).days >= 3     # the hold spans a weekend
        weekday1 = (nx - d).days == 1                                                  # two consecutive weekdays
        if not weekend and not (sym == "SPY" and weekday1):
            continue
        S = und.px(d)
        e = front_expiry(bk, d)
        if e is None or und.has_event(d, nx):
            continue
        sl = straddle_legs(bk, d, e, S, -1)
        if sl is None:
            continue
        if weekend:
            add("weekend_short_straddle_hedged", d, position(bk, und, d, nx, sl, True, hc), exit=nx)
        if sym == "SPY":
            add("weekend_gross" if weekend else "weekday_gross", d,
                position(bk, und, d, nx, sl, True, 0.0, fill=0.0, comm=0.0), exit=nx)
    # FOMC / CPI (SPY)
    if sym == "SPY":
        for kind, evs in (("fomc_short_straddle_hedged", [pd.Timestamp(x) for x in FOMC]),
                          ("cpi_short_straddle_hedged", list(cpi))):
            for A in evs:
                if A > END - pd.Timedelta(days=5):
                    continue
                pre = dates[(dates >= A - pd.Timedelta(days=5)) & (dates < A)]
                post = dates[(dates >= A) & (dates <= A + pd.Timedelta(days=4))]
                if not len(pre) or not len(post):
                    continue
                d, x = pre[-1], post[0]
                if not entry_ok(bk, und, d) or und.has_event(d, x):
                    continue
                e = front_expiry(bk, d, 14, 60, 30)
                if e is None or e <= x:
                    continue
                sl = straddle_legs(bk, d, e, und.px(d), -1)
                if sl is not None:
                    add(kind, d, position(bk, und, d, x, sl, True, hc), event=A, exit=x)
    # box-spread diagnostic (SPY, DIA)
    box = []
    if sym in INDEX:
        for d in dates:
            e = front_expiry(bk, d)
            if e is None:
                continue
            g = bk.by_day[d]
            g = g[(g.expiration == e) & (g.bid > 0)]
            s = bk.surf.get((d, e, "C"))
            if s is None:
                continue
            F = s[0]
            Cq = g[g.cp == "C"].drop_duplicates("strike").set_index("strike")
            Pq = g[g.cp == "P"].drop_duplicates("strike").set_index("strike")
            # strikes at or below the forward only: the puts are then out of the money, so the American early-
            # exercise premium (which makes boxes on ITM puts worth more than the discounted width) stays small
            ks = np.array(sorted(k for k in Cq.index.intersection(Pq.index) if 0.85 * F <= k <= F))
            if len(ks) < 2:
                continue
            c, p = Cq.loc[ks], Pq.loc[ks]
            i, j = np.triu_indices(len(ks), 1)
            box.append(pd.DataFrame({
                "symbol": sym, "date": d, "expiry": e, "T": (e - d).days / 365.0, "F": F, "width": ks[j] - ks[i],
                "buy": c.ask.to_numpy()[i] - c.bid.to_numpy()[j] + p.ask.to_numpy()[j] - p.bid.to_numpy()[i],
                "sell": c.bid.to_numpy()[i] - c.ask.to_numpy()[j] + p.bid.to_numpy()[j] - p.ask.to_numpy()[i],
                "mid": c.mid.to_numpy()[i] - c.mid.to_numpy()[j] + p.mid.to_numpy()[j] - p.mid.to_numpy()[i]}))
    box = pd.concat(box, ignore_index=True) if box else pd.DataFrame()
    meta = {"chain_dates": int(len(dates)), "positions": len(rows), "box_rows": int(len(box)),
            "delta_fallbacks": COUNT["delta_fallbacks"], "deltas": COUNT["deltas"]}
    COUNT.update({"positions": 0, "delta_fallbacks": 0, "deltas": 0})
    return pd.DataFrame(rows), box, meta


def stage_symbols(a):
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    syms = [x for x in a.symbols.split(",") if x] or sorted(
        f[len("chain_"):-len(".parquet")] for f in os.listdir(a.chain) if f.startswith("chain_") and f.endswith(".parquet"))
    if a.group:
        i, n = map(int, a.group.split("/"))
        syms = syms[i::n]
    cpi = []
    if a.cpi and os.path.exists(a.cpi):
        cpi = sorted(pd.to_datetime(pd.read_csv(a.cpi).date))
    meta, P, B = {"cpi_releases": len(cpi)}, [], []
    for sym in syms:
        if f"{sym}_raw" not in X.columns or not X[f"{sym}_raw"].notna().any():
            meta[sym] = {"error": "no closes"}
            continue
        try:
            und = L.Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
            out = run_symbol(sym, a.chain, und, cpi)
        except Exception as ex:                                       # noqa: BLE001
            meta[sym] = {"error": repr(ex)[:300]}
            continue
        if out is None:
            continue
        p, b, m = out
        meta[sym] = m
        print(sym, m, flush=True)
        P.append(p)
        B.append(b)
    tag = (a.group or "all").replace("/", "of")
    P = [x for x in P if len(x)]
    B = [x for x in B if len(x)]
    if P:
        pd.concat(P, ignore_index=True).to_csv(os.path.join(a.out, f"positions2_{tag}.csv.gz"), index=False)
    if B:
        pd.concat(B, ignore_index=True).to_csv(os.path.join(a.out, f"box_{tag}.csv.gz"), index=False)
    json.dump(meta, open(os.path.join(a.out, f"meta2_{tag}.json"), "w"), indent=1, default=str)


# ----------------------------------------------------------------------------------------------- tests
def stage_combine(a):
    dirs = [d for d in a.inputs.split(",") if d]
    P, B, meta = [], [], {}
    for d in dirs:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if f.startswith("positions2_"):
                P.append(pd.read_csv(p, parse_dates=["entry"]))
            elif f.startswith("box_"):
                B.append(pd.read_csv(p, parse_dates=["date", "expiry"]))
            elif f.startswith("meta2_"):
                meta.update(json.load(open(p)))
    P = pd.concat(P, ignore_index=True)
    for c in ("exit", "event"):
        if c in P.columns:
            P[c] = pd.to_datetime(P[c])
    tests, series = [], {}

    def reg(name, x, per_year, **extra):
        x = x.sort_index()
        series[name] = x
        tests.append({"test": name, **stats(x, per_year), **extra})

    def kind(sym, k):
        return P[(P.symbol == sym) & (P.kind == k)].set_index("entry").pnl

    reg("E1 SPY delta-hedged risk reversal (sell 25d put, buy 25d call)", kind("SPY", "risk_reversal_hedged"), 12)
    reg("E2 DIA delta-hedged risk reversal", kind("DIA", "risk_reversal_hedged"), 12)
    reg("E3 SPY delta-hedged short 10-delta put", kind("SPY", "short_put10_hedged"), 12)
    reg("E4 DIA delta-hedged short 10-delta put", kind("DIA", "short_put10_hedged"), 12)
    stk = P[(~P.symbol.isin(ETFS)) & (P.kind == "long_straddle_hedged")]
    idx = kind("SPY", "short_straddle_hedged")
    g = stk.groupby("entry").pnl.agg(["mean", "size"])
    g = g[g["size"] >= 20]
    disp = (idx.reindex(g.index) + g["mean"]).dropna()
    reg("E5 Dispersion: short SPY hedged straddle, long hedged stock straddles", disp, 12,
        mean_stocks_per_month=float(g["size"].mean()) if len(g) else np.nan)
    sp = P[(P.symbol == "SPY") & (P.kind == "weekend_short_straddle_hedged")].set_index("entry").pnl
    reg("E6 Weekend effect SPY: sell hedged straddle Friday, buy back next day (net)", sp, 52)
    ws = P[(~P.symbol.isin(ETFS)) & (P.kind == "weekend_short_straddle_hedged")].groupby("entry").pnl.agg(["mean", "size"])
    ws = ws[ws["size"] >= 20]
    reg("E7 Weekend effect stocks (equal weight, net)", ws["mean"], 52,
        mean_stocks_per_weekend=float(ws["size"].mean()) if len(ws) else np.nan)
    wg = P[(P.symbol == "SPY") & (P.kind == "weekend_gross")].set_index("entry").pnl
    dg = P[(P.symbol == "SPY") & (P.kind == "weekday_gross")].set_index("entry").pnl
    diff = {}
    for d, v in wg.items():
        wk = dg[(dg.index > d - pd.Timedelta(days=7)) & (dg.index < d)]
        if len(wk) >= 3:
            diff[d] = v - wk.mean()
    diff = pd.Series(diff, dtype=float)
    reg("E8 Weekend minus weekday one-day P&L, SPY hedged straddle (gross)", diff, 52,
        weekend_gross_mean=float(wg.mean()) if len(wg) else np.nan,
        weekday_gross_mean=float(dg.mean()) if len(dg) else np.nan)
    reg("E9 FOMC: sell SPY hedged straddle before the statement, buy back after", kind("SPY", "fomc_short_straddle_hedged"), 8)
    cp = kind("SPY", "cpi_short_straddle_hedged")
    reg("E10 CPI: sell SPY hedged straddle before the release, buy back after", cp, 12)
    reg("E11 SPY delta-hedged calendar straddle (short front, long back)", kind("SPY", "calendar_straddle_hedged"), 12)
    reg("E12 DIA delta-hedged calendar straddle", kind("DIA", "calendar_straddle_hedged"), 12)
    vs = P[(P.symbol == "SPY") & (P.kind == "variance_strip_hedged")].set_index("entry")
    reg("E13 SPY short variance strip, delta-hedged", vs.pnl, 12,
        mean_vol_points=float(vs.pnl_vol_points.mean()) if len(vs) else np.nan,
        t_vol_points=float(L.nw_t(vs.pnl_vol_points.to_numpy())) if len(vs) >= 8 else np.nan,
        mean_strikes=float(vs.n_strikes.mean()) if len(vs) else np.nan)
    vd = P[(P.symbol == "DIA") & (P.kind == "variance_strip_hedged")].set_index("entry")
    reg("E14 DIA short variance strip, delta-hedged", vd.pnl, 12,
        mean_vol_points=float(vd.pnl_vol_points.mean()) if len(vd) else np.nan,
        t_vol_points=float(L.nw_t(vd.pnl_vol_points.to_numpy())) if len(vd) >= 8 else np.nan)
    T = pd.DataFrame(tests)
    T["bh_q"] = L.bh(T.p_one_sided.to_numpy()) if "p_one_sided" in T else np.nan
    T.to_csv(os.path.join(a.out, "tests2.csv"), index=False)
    pd.concat(series, axis=1).to_csv(os.path.join(a.out, "series2.csv"))
    P.groupby(["symbol", "kind"]).pnl.agg(["count", "mean", "std"]).reset_index().to_csv(
        os.path.join(a.out, "positions2_summary.csv"), index=False)
    P.to_csv(os.path.join(a.out, "positions2_all.csv.gz"), index=False)
    # dispersion, premium-matched (diagnostic): stock straddles scaled so their premium equals the index premium
    diag = {}
    if len(stk) and len(idx):
        ip = P[(P.symbol == "SPY") & (P.kind == "short_straddle_hedged")].set_index("entry")
        sg = stk.groupby("entry").agg(pnl=("pnl", "mean"), prem=("premium", "mean"), n=("pnl", "size"))
        sg = sg[sg.n >= 20].join(ip[["pnl", "premium"]].rename(columns={"pnl": "pnl_i", "premium": "premium_i"}),
                                 how="inner")
        if len(sg):
            scale = sg.premium_i.abs() / sg.prem.abs()
            x = sg.pnl_i + scale * sg.pnl
            diag["dispersion_premium_matched"] = stats(x, 12)
    # box spreads against the 13-week T-bill
    if B and a.rates and os.path.exists(a.rates):
        Bx = pd.concat(B, ignore_index=True)
        r = pd.read_csv(a.rates, parse_dates=["date"]).set_index("date").sort_index()["IRX"] / 100.0
        Bx["r"] = r.reindex(Bx.date, method="ffill").to_numpy()
        Bx = Bx.dropna(subset=["r"])
        Bx = Bx[Bx.width >= 0.02 * Bx.F]                      # quote rounding dominates narrower boxes
        Bx["pv"] = Bx.width * np.exp(-Bx.r * Bx["T"])
        c4 = 4 * COMM
        Bx["buy_arb"] = Bx.buy + c4 < Bx.pv
        Bx["sell_arb"] = Bx.sell - c4 > Bx.pv
        with np.errstate(all="ignore"):
            Bx["implied_rate_mid"] = -np.log(Bx.mid / Bx.width) / Bx["T"]
        out = {}
        for sym, g in Bx.groupby("symbol"):
            by_day = g.groupby("date")[["buy_arb", "sell_arb"]].any()
            ok = g[g.mid > 0]
            out[sym] = {"boxes": int(len(g)), "snapshots": int(g.date.nunique()),
                        "share_boxes_buy_below_tbill_pv": float(g.buy_arb.mean()),
                        "share_boxes_sell_above_tbill_pv": float(g.sell_arb.mean()),
                        "share_days_any_buy_arb": float(by_day.buy_arb.mean()),
                        "share_days_any_sell_arb": float(by_day.sell_arb.mean()),
                        "median_buy_edge_bp_of_width": float(((g.pv - g.buy - c4) / g.width).median() * 1e4),
                        "median_implied_rate_mid_minus_tbill_bp": float((ok.implied_rate_mid - ok.r).median() * 1e4)}
        diag["box_spreads"] = out
    json.dump({"meta": meta, "tests": T.to_dict("records"), "diagnostics": diag},
              open(os.path.join(a.out, "summary2.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = ["test", "n", "mean", "t_nw", "sharpe", "win_rate", "worst", "years_positive", "mean_2019_2022",
            "t_2019_2022", "mean_2023_2025", "t_2023_2025", "bh_q"]
    print(T[[c for c in cols if c in T.columns]].round(5).to_string())
    print(json.dumps(diag, indent=1, default=str)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("symbols", "combine"), required=True)
    ap.add_argument("--chain", default="")
    ap.add_argument("--closes", default="")
    ap.add_argument("--cpi", default="", help="CSV with a 'date' column of CPI release days")
    ap.add_argument("--rates", default="", help="CSV date, IRX (13-week T-bill, percent)")
    ap.add_argument("--group", default="")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "symbols":
        stage_symbols(a)
    else:
        stage_combine(a)


if __name__ == "__main__":
    main()
