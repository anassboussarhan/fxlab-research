"""Options edges from the research literature, on real end-of-day quotes (DoltHub, 2019-2025), with parameters
fixed in advance (nothing tuned). Development data only: quotes after 2025-12-31 are dropped and every position
must end by then.

PRE-REGISTERED TESTS (one-sided; Benjamini-Hochberg over H1-H8; 2019-2022 vs 2023-2025 reported separately):
  H1  SPY delta-hedged SHORT at-the-money straddle, entered monthly about 30 days to expiry (listed expiry 20-42
      days out), held to expiry; the option delta (Black-76 on the day's smile) is hedged with SPY at every
      snapshot close. The textbook harvest of the index variance risk premium (Bakshi & Kapadia 2003; Carr & Wu
      2009).
  H2  H1 only when VIX < VIX3M at entry (the timing that worked for the variance-swap proxy, docs/options_retail.md).
  H3  SPY delta-hedged SHORT 25-delta strangle, same schedule (skew premium).
  H4  Vasquez (2017): each month, among stocks, ATM IV term slope = IV(longest listed expiry) - IV(nearest expiry
      7+ days): LONG unhedged ATM straddles on the top fifth, SHORT on the bottom fifth, held to the ~30-day
      expiry; equal weight.
  H5  Cao & Han (2013): idiosyncratic volatility (63-day residual vol vs SPY): delta-hedged SHORT ATM straddles on
      the top fifth, delta-hedged LONG on the bottom fifth, monthly, held to expiry.
  H6  Cremers & Weinbaum (2010): weekly, call-minus-put IV at the same strikes (|ln K/F| <= 0.1, nearest expiry
      7+ days): LONG the stocks in the top fifth, SHORT the bottom fifth, from the next trading day's close for a
      week.
  H7  Xing, Zhang & Zhao (2010): weekly, smirk = IV(OTM put, K/F nearest 0.92) - IV(ATM call): LONG the bottom
      fifth, SHORT the top fifth, same timing.
  H8  Gao, Xing & Zhang (2018): LONG ATM straddle on the last snapshot 2-10 days before an earnings date, expiry
      at least 3 days after it, sold on the first snapshot 1-7 days after.
COSTS: options at mid +- 0.75 x half-spread and USD 0.65 per contract; hedge trades 1 bp of the traded notional on
SPY and 3 bp on stocks; stock legs of H6 / H7 3 bp per side with full weekly turnover assumed. Expiring options at
intrinsic. Unit: P&L per unit of underlying notional (per share / price at entry); long-short tests average each
side over its names and add the two sides.
Usage: python scripts/options_edges_lab.py --stage symbols --chain DIR --closes FILE --group i/n --out DIR
       python scripts/options_edges_lab.py --stage combine --inputs DIRS --closes FILE --out DIR"""
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

END = L.END
FILL = 0.75
COMM = L.COMM
SPLIT = pd.Timestamp("2023-01-01")
FALLBACK = [0, 0]          # hedge deltas that fell back to intrinsic (0 / +-1), and all hedge deltas


def ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ----------------------------------------------------------------------------------------------- smile helpers
def smile_delta(bk, d, e, cp, k):
    """Black-76 delta from the day's smile (across expiries when e itself is not listed that day)."""
    sm = bk.smile(d, e, cp, k)
    T = (e - d).days / 365.0
    if sm is None or T <= 0 or sm[0] <= 0:
        return None
    v, F, _ = sm
    S = bk.und.px(d)
    if np.isfinite(S) and abs(math.log(F / S)) > 0.02:     # bad quotes on that snapshot: use the close
        F = S
    sd = v * math.sqrt(T)
    d1 = (math.log(F / k) + 0.5 * sd * sd) / sd
    return ncdf(d1) if cp == "C" else ncdf(d1) - 1.0


def atm_iv(bk, d, e):
    out = []
    for cp in ("C", "P"):
        s = bk.surf.get((d, e, cp))
        if s is not None:
            out.append(float(np.interp(0.0, s[1], s[2])))
    return float(np.mean(out)) if out else np.nan


def expiries_on(bk, d):
    return sorted({e for (dd, e, cp) in bk.surf if dd == d})


# ----------------------------------------------------------------------------------------------- positions
def option_position(bk, d, front, legs, hedge, hedge_cost, und):
    """legs: [(cp, strike, qty)] on expiry `front`, opened at d's close. If hedge, the option delta is offset with
    the underlying at every snapshot close. Returns P&L per unit of notional and the premium paid / received."""
    S0 = und.px(d)
    days = [pd.Timestamp(t) for t in bk.dates if np.datetime64(d) <= t < np.datetime64(front)]
    if not days or days[0] != d or not np.isfinite(S0):
        return None
    paths = {}
    for cp, k, q in legs:
        p = bk.path(front, cp, k, d, front)
        if not len(p) or p.index[0] != d:
            return None
        paths[(cp, k)] = p.reindex(days).ffill()
    cash = 0.0
    for cp, k, q in legs:
        m, h = float(paths[(cp, k)].mid.iloc[0]), float(paths[(cp, k)].half.iloc[0])
        cash -= q * (m + np.sign(q) * FILL * h)
    cash -= COMM * sum(abs(q) for _, _, q in legs)
    premium = cash
    stock_pnl, cost, held = 0.0, 0.0, 0.0
    if hedge:
        for i, t in enumerate(days):
            St = und.px(t)
            dl = 0.0
            for cp, k, q in legs:
                x = smile_delta(bk, t, front, cp, k)
                if x is None:
                    x = (1.0 if St > k else 0.0) if cp == "C" else (-1.0 if St < k else 0.0)
                    FALLBACK[0] += 1
                FALLBACK[1] += 1
                dl += q * x
            target = -dl
            cost += abs(target - held) * St * hedge_cost
            held = target
            t1 = days[i + 1] if i + 1 < len(days) else front
            S1 = und.px(t1)
            a0, a1 = und.adj.loc[:t].iloc[-1], und.adj.loc[:t1].iloc[-1]
            stock_pnl += held * St * (a1 / a0 - 1.0)            # total-return move of the hedge (dividends incl.)
        cost += abs(held) * und.px(front) * hedge_cost
    Sx = und.px(front)
    settle = sum(q * (max(Sx - k, 0.0) if cp == "C" else max(k - Sx, 0.0)) for cp, k, q in legs)
    pnl = cash + settle + stock_pnl - cost
    return {"pnl": pnl / S0, "premium": premium / S0, "hedge_cost": cost / S0, "days": int((front - d).days)}


def hedged_daily(bk, d, front, legs, hedge_cost, und):
    """Daily P&L increments (per unit of notional at entry) of a delta-hedged option position: options marked at
    mid on each snapshot (listed or smile), the hedge at the close, entry spread and costs on the day they occur;
    the increments add up to option_position's realised P&L."""
    r = option_position(bk, d, front, legs, True, hedge_cost, und)
    if r is None:
        return None
    S0 = und.px(d)
    days = [pd.Timestamp(t) for t in bk.dates if np.datetime64(d) <= t < np.datetime64(front)]
    paths = {(cp, k): bk.path(front, cp, k, d, front).reindex(days).ffill() for cp, k, q in legs}
    marks, held, stock, cost = [], 0.0, 0.0, 0.0
    cash = -sum(q * (paths[(cp, k)].mid.iloc[0] + np.sign(q) * FILL * paths[(cp, k)].half.iloc[0]) for cp, k, q in legs)
    cash -= COMM * sum(abs(q) for _, _, q in legs)
    for i, t in enumerate(days):
        St = und.px(t)
        if i > 0:
            a0, a1 = und.adj.loc[:days[i - 1]].iloc[-1], und.adj.loc[:t].iloc[-1]
            stock += held * und.px(days[i - 1]) * (a1 / a0 - 1.0)
        dl = 0.0
        for cp, k, q in legs:
            x = smile_delta(bk, t, front, cp, k)
            if x is None:
                x = (1.0 if St > k else 0.0) if cp == "C" else (-1.0 if St < k else 0.0)
            dl += q * x
        cost += abs(-dl - held) * St * hedge_cost
        held = -dl
        opt = sum(q * paths[(cp, k)].mid.iloc[i] for cp, k, q in legs)
        marks.append((t, (cash + opt + stock - cost) / S0))
    s = pd.Series(dict(marks))
    s.loc[pd.Timestamp(front)] = r["pnl"]
    s = s.sort_index()
    inc = s.diff()
    inc.iloc[0] = s.iloc[0]
    return inc


def atm_strike(bk, d, e, S):
    g = bk.by_day[d]
    g = g[g.expiration == e]
    ks = set(g[g.cp == "P"].strike) & set(g[g.cp == "C"].strike)
    return min(ks, key=lambda k: abs(k - S)) if ks else None


def delta_strike(bk, d, e, cp, target, S):
    k = bk.pick(d, e, cp, target, S)
    return k


def monthly_front(bk, d):
    exps = bk.expiries(d)
    c = [e for e in exps if 20 <= (e - d).days <= 42 and e <= END]
    return min(c, key=lambda e: abs((e - d).days - 30)) if c else None


# ----------------------------------------------------------------------------------------------- per symbol
def run_symbol(sym, chain_dir, und, spy_ret, vix, earn):
    C = L.load_chain(chain_dir, sym)
    if C.empty:
        return None
    C = L.trim_chain(C, und)
    bk = L.Book(C, und)
    hc = 1e-4 if sym in ("SPY", "DIA") else 3e-4
    pos, sig, ern, daily = [], [], [], []
    # monthly positions (about 30 days)
    for d in L.entry_dates(bk, 30):
        S = und.px(d)
        e = monthly_front(bk, d)
        if e is None or not np.isfinite(S) or und.has_event(d, e):
            continue
        Sp = bk.parity_px(d)
        if np.isfinite(Sp) and abs(math.log(Sp / S)) > 0.03:
            continue
        k = atm_strike(bk, d, e, S)
        vv = vix.loc[:d]
        cont = bool(len(vv) and vv.iloc[-1, 0] < vv.iloc[-1, 1])
        base = {"symbol": sym, "entry": d, "expiry": e, "contango": cont}
        if k is not None and sym in ("SPY", "DIA"):
            inc = hedged_daily(bk, d, e, [("P", k, -1), ("C", k, -1)], hc, und)
            if inc is not None:
                daily.append(pd.DataFrame({"symbol": sym, "kind": "short_straddle_hedged", "entry": d,
                                           "contango": cont, "day": inc.index, "inc": inc.to_numpy()}))
        if k is not None:
            straddle = [("P", k, -1), ("C", k, -1)]
            long_ = [(cp, kk, -q) for cp, kk, q in straddle]
            for name, legs, hedge in (("short_straddle_hedged", straddle, True),
                                      ("short_straddle_unhedged", straddle, False),
                                      ("long_straddle_unhedged", long_, False),
                                      ("long_straddle_hedged", long_, True)):
                r = option_position(bk, d, e, legs, hedge, hc, und)
                if r:
                    pos.append({**base, "kind": name, "strike": k, **r})
        kp, kc = delta_strike(bk, d, e, "P", 0.25, S), delta_strike(bk, d, e, "C", 0.25, S)
        if kp is not None and kc is not None and kp < kc:
            r = option_position(bk, d, e, [("P", kp, -1), ("C", kc, -1)], True, hc, und)
            if r:
                pos.append({**base, "kind": "short_strangle25_hedged", "strike": kp, **r})
    # weekly signals
    ent = L.entry_dates(bk, L.WEEKLY_DTE)
    for d in ent:
        S = und.px(d)
        exps = [e for e in expiries_on(bk, d) if (e - d).days >= 7]
        if not exps or not np.isfinite(S):
            continue
        near, far = exps[0], exps[-1]
        slope = atm_iv(bk, d, far) - atm_iv(bk, d, near) if far != near else np.nan
        g = bk.by_day[d]
        g = g[(g.expiration == near) & g.iv.notna() & (g.iv > 0)]
        P = g[g.cp == "P"].set_index("strike").iv
        Cc = g[g.cp == "C"].set_index("strike").iv
        F = bk.surf.get((d, near, "C"), bk.surf.get((d, near, "P"), (S,)))[0]
        ks = [k for k in P.index.intersection(Cc.index) if abs(math.log(k / F)) <= 0.1]
        ivs = float(np.mean([Cc.loc[k] - P.loc[k] for k in ks])) if ks else np.nan
        mp = [(abs(k / F - 0.92), k) for k in P.index if 0.85 <= k / F <= 0.97]
        mc = [(abs(k / F - 1.0), k) for k in Cc.index if 0.97 <= k / F <= 1.03]
        smirk = float(P.loc[min(mp)[1]] - Cc.loc[min(mc)[1]]) if mp and mc else np.nan
        r = np.log(und.adj.loc[:d]).diff().dropna().iloc[-63:]
        m = spy_ret.reindex(r.index).dropna()
        r = r.reindex(m.index)
        ivol = np.nan
        rr, mm = r.to_numpy(float), m.to_numpy(float)
        ok = np.isfinite(rr) & np.isfinite(mm)
        rr, mm = rr[ok], mm[ok]
        if len(rr) >= 40 and mm.var() > 0:
            beta = np.cov(rr, mm)[0, 1] / mm.var(ddof=1)
            res = rr - rr.mean() - beta * (mm - mm.mean())
            ivol = float(res.std(ddof=2) * math.sqrt(252))
        sig.append({"symbol": sym, "date": d, "slope": slope, "iv_spread": ivs, "smirk": smirk, "ivol": ivol,
                    "atm_iv_near": atm_iv(bk, d, near)})
    # earnings straddles
    if earn is not None and len(earn):
        for E in earn:
            if E > END - pd.Timedelta(days=10) or E < pd.Timestamp(bk.dates[0]) + pd.Timedelta(days=10):
                continue
            pre = [pd.Timestamp(t) for t in bk.dates if E - pd.Timedelta(days=10) <= t <= E - pd.Timedelta(days=2)]
            post = [pd.Timestamp(t) for t in bk.dates if E + pd.Timedelta(days=1) <= t <= E + pd.Timedelta(days=7)]
            if not pre or not post:
                continue
            d0, d1 = pre[-1], post[0]
            S0 = und.px(d0)
            if und.has_event(d0, d1):
                continue
            ex = [e for e in bk.expiries(d0) if E + pd.Timedelta(days=3) <= e <= E + pd.Timedelta(days=45)]
            if not ex:
                continue
            e = ex[0]
            k = atm_strike(bk, d0, e, S0)
            if k is None:
                continue
            val0, val1, ok = 0.0, 0.0, True
            for cp in ("P", "C"):
                p = bk.path(e, cp, k, d0, d1)
                if not len(p) or p.index[0] != d0 or d1 not in p.index:
                    ok = False
                    break
                val0 += p.mid.loc[d0] + FILL * p.half.loc[d0]
                val1 += p.mid.loc[d1] - FILL * p.half.loc[d1]
            if not ok:
                continue
            pnl = val1 - val0 - 4 * COMM
            ern.append({"symbol": sym, "earnings": E, "entry": d0, "exit": d1, "expiry": e,
                        "pnl": pnl / S0, "premium": val0 / S0, "ret_on_premium": pnl / val0 if val0 > 0 else np.nan})
    meta = {"chain_dates": int(len(bk.dates)), "positions": len(pos), "signals": len(sig), "earnings_trades": len(ern),
            "hedge_delta_fallbacks": int(FALLBACK[0]), "hedge_deltas": int(FALLBACK[1]),
            "path_points_unvalued": int(getattr(bk, "n_gap", 0)), "across_expiry_check": bk.term_check(800)}
    FALLBACK[0] = FALLBACK[1] = 0
    if daily:
        meta["daily"] = pd.concat(daily, ignore_index=True)
    return pd.DataFrame(pos), pd.DataFrame(sig), pd.DataFrame(ern), meta


def stage_symbols(a):
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    vix = X[["VIX", "VIX3M"]].dropna()
    spy_ret = np.log(X["SPY_adj"].dropna()).diff().dropna()
    syms = [x for x in a.symbols.split(",") if x] or sorted(f[len("chain_"):-len(".parquet")] for f in os.listdir(a.chain)
                                                       if f.startswith("chain_") and f.endswith(".parquet"))
    if a.group:
        i, n = map(int, a.group.split("/"))
        syms = syms[i::n]
    E = None
    ef = os.path.join(a.chain, "earnings.csv")
    if os.path.exists(ef):
        Ed = pd.read_csv(ef)
        Ed.columns = [c.lower() for c in Ed.columns]
        Ed["date"] = pd.to_datetime(Ed["date"], errors="coerce")
        E = {k: pd.DatetimeIndex(sorted(v.dropna().unique())) for k, v in Ed.groupby("act_symbol").date}
    meta, P, S, R = {}, [], [], []
    for sym in syms:
        if f"{sym}_raw" not in X.columns or not X[f"{sym}_raw"].notna().any():
            meta[sym] = {"error": "no closes"}
            continue
        try:
            und = L.Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
            out = run_symbol(sym, a.chain, und, spy_ret, vix, (E or {}).get(sym) if sym not in ("SPY", "DIA", "XLF", "XLE") else None)
        except Exception as ex:                                       # noqa: BLE001
            meta[sym] = {"error": repr(ex)[:300]}
            continue
        if out is None:
            continue
        p, s, r, m = out
        if "daily" in m:
            m.pop("daily").to_csv(os.path.join(a.out, f"daily_{sym}.csv.gz"), index=False)
        meta[sym] = m
        print(sym, m, flush=True)
        P.append(p), S.append(s), R.append(r)
    tag = (a.group or "all").replace("/", "of")
    for name, parts in (("positions", P), ("signals", S), ("earnings", R)):
        parts = [x for x in parts if len(x)]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(a.out, f"{name}_{tag}.csv.gz"), index=False)
    json.dump(meta, open(os.path.join(a.out, f"meta_{tag}.json"), "w"), indent=1, default=str)


# ----------------------------------------------------------------------------------------------- tests
def stats(x, per_year):
    x = pd.Series(x).dropna()
    if len(x) < 8:
        return {"n": int(len(x))}
    t = L.nw_t(x.to_numpy(), lag=1)
    yrs = x.groupby(pd.DatetimeIndex(x.index).year).sum()
    is_ = x[x.index < SPLIT]
    oos = x[x.index >= SPLIT]
    sd = x.std()
    return {"n": int(len(x)), "mean": float(x.mean()), "t_nw": float(t), "p_one_sided": float(st.norm.sf(t)),
            "sharpe": float(x.mean() / sd * math.sqrt(per_year)) if sd > 0 else np.nan,
            "win_rate": float((x > 0).mean()), "worst": float(x.min()), "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}",
            "mean_2019_2022": float(is_.mean()) if len(is_) else np.nan,
            "t_2019_2022": float(L.nw_t(is_.to_numpy())) if len(is_) >= 8 else np.nan,
            "mean_2023_2025": float(oos.mean()) if len(oos) else np.nan,
            "t_2023_2025": float(L.nw_t(oos.to_numpy())) if len(oos) >= 8 else np.nan,
            **{f"y{y}": float(v) for y, v in yrs.items()}}


MIN_NAMES = int(os.environ.get("EDGES_MIN_NAMES", "10"))


def quintile_ls(df, key, value, top_long=True, min_names=None):
    """Per date: mean `value` of the top fifth minus mean of the bottom fifth by `key` (or the reverse)."""
    min_names = min_names or MIN_NAMES
    out = {}
    for d, g in df.dropna(subset=[key, value]).groupby("date"):
        if len(g) < min_names:
            continue
        q = g[key].rank(pct=True)
        top, bot = g[value][q > 0.8].mean(), g[value][q <= 0.2].mean()
        out[d] = (top - bot) if top_long else (bot - top)
    return pd.Series(out).sort_index()


def quintile_two_books(M, key, top_kind, bottom_kind, min_names=None):
    """Per date: mean P&L of `top_kind` positions on the top fifth by `key` plus mean P&L of `bottom_kind`
    positions on the bottom fifth (each side pays its own costs)."""
    min_names = min_names or MIN_NAMES
    out = {}
    for d, g in M.dropna(subset=[key]).groupby("date"):
        names = g.drop_duplicates("symbol")[["symbol", key]]
        if len(names) < min_names:
            continue
        q = names.set_index("symbol")[key].rank(pct=True)
        top, bot = set(q[q > 0.8].index), set(q[q <= 0.2].index)
        a = g[(g.kind == top_kind) & g.symbol.isin(top)].pnl.mean()
        b = g[(g.kind == bottom_kind) & g.symbol.isin(bot)].pnl.mean()
        if np.isfinite(a) and np.isfinite(b):
            out[d] = a + b
    return pd.Series(out).sort_index()


def stage_combine(a):
    dirs = [d for d in a.inputs.split(",") if d]
    P, S, R, meta = [], [], [], {}
    for d in dirs:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if f.startswith("positions_"):
                P.append(pd.read_csv(p, parse_dates=["entry", "expiry"]))
            elif f.startswith("signals_"):
                S.append(pd.read_csv(p, parse_dates=["date"]))
            elif f.startswith("earnings_"):
                R.append(pd.read_csv(p, parse_dates=["earnings", "entry", "exit", "expiry"]))
            elif f.startswith("meta_"):
                meta.update(json.load(open(p)))
    P = pd.concat(P, ignore_index=True)
    S = pd.concat(S, ignore_index=True) if S else pd.DataFrame()
    R = pd.concat(R, ignore_index=True) if R else pd.DataFrame()
    P = P[P.expiry <= END]
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    tests, series = [], {}
    spy = P[P.symbol == "SPY"].set_index("entry").sort_index()
    for name, kind, cond in (("H1 SPY delta-hedged short ATM straddle", "short_straddle_hedged", None),
                             ("H2 H1 in contango only", "short_straddle_hedged", "contango"),
                             ("H3 SPY delta-hedged short 25-delta strangle", "short_strangle25_hedged", None)):
        g = spy[spy.kind == kind]
        if cond:
            g = g[g[cond].astype(bool)]
        series[name] = g.pnl
        tests.append({"test": name, **stats(g.pnl, 12), "mean_premium": float(g.premium.mean()) if len(g) else np.nan,
                      "mean_hedge_cost": float(g.hedge_cost.mean()) if len(g) else np.nan})
    stocks = P[~P.symbol.isin(["SPY", "DIA", "XLF", "XLE"])].copy()
    # H4 Vasquez: monthly, slope at the entry date (signal from the weekly table on or before the entry)
    if len(S):
        S = S.sort_values("date")
        sig_m = []
        for sym, g in stocks.groupby("symbol"):
            s = S[S.symbol == sym].set_index("date").sort_index()
            if not len(s):
                continue
            idx = s.index.searchsorted(g.entry, side="right") - 1
            ok = idx >= 0
            gg = g[ok].copy()
            for c in ("slope", "ivol"):
                gg[c] = s[c].to_numpy()[idx[ok]]
            sig_m.append(gg)
        M = pd.concat(sig_m, ignore_index=True) if sig_m else pd.DataFrame()
        if len(M):
            M["date"] = M.entry.dt.to_period("M").dt.to_timestamp()
            h4 = quintile_two_books(M, "slope", "long_straddle_unhedged", "short_straddle_unhedged")
            series["H4 Vasquez IV term slope, straddles"] = h4
            tests.append({"test": "H4 Vasquez IV term slope, straddles", **stats(h4, 12)})
            h5 = quintile_two_books(M, "ivol", "short_straddle_hedged", "long_straddle_hedged")
            series["H5 Cao-Han idiosyncratic vol, delta-hedged"] = h5
            tests.append({"test": "H5 Cao-Han idiosyncratic vol, delta-hedged", **stats(h5, 12)})
        # H6 / H7: weekly stock returns from the next trading day's close to the next signal's next close
        adj = {c[:-4]: X[c].dropna() for c in X.columns if c.endswith("_adj")}
        rows = []
        for sym, g in S[~S.symbol.isin(["SPY", "DIA", "XLF", "XLE"])].groupby("symbol"):
            if sym not in adj:
                continue
            a_ = adj[sym]
            dts = g.date.to_numpy()
            for i in range(len(dts) - 1):
                i0 = a_.index.searchsorted(pd.Timestamp(dts[i]), side="right")        # next trading day
                i1 = a_.index.searchsorted(pd.Timestamp(dts[i + 1]), side="right")
                if i1 >= len(a_) or i0 >= i1:
                    continue
                rows.append({"symbol": sym, "date": pd.Timestamp(dts[i]),
                             "ret": float(a_.iloc[i1] / a_.iloc[i0] - 1),
                             "iv_spread": g.iv_spread.iloc[i], "smirk": g.smirk.iloc[i]})
        W = pd.DataFrame(rows)
        if len(W):
            h6 = quintile_ls(W, "iv_spread", "ret", top_long=True)
            h7 = quintile_ls(W, "smirk", "ret", top_long=False)
            # long and short side each pay 3 bp in and 3 bp out every week (full turnover assumed): 12 bp a week
            h6, h7 = h6 - 4 * 3e-4, h7 - 4 * 3e-4
            series["H6 Cremers-Weinbaum IV spread, stocks"] = h6
            series["H7 Xing-Zhang-Zhao smirk, stocks"] = h7
            tests.append({"test": "H6 Cremers-Weinbaum IV spread, stocks", **stats(h6, 52)})
            tests.append({"test": "H7 Xing-Zhang-Zhao smirk, stocks", **stats(h7, 52)})
    if len(R):
        R = R[R.exit <= END]
        er = R.groupby(R.entry.dt.to_period("M").dt.to_timestamp()).pnl.mean()
        series["H8 Gao-Xing-Zhang earnings straddles"] = er
        tests.append({"test": "H8 Gao-Xing-Zhang earnings straddles", **stats(er, 12),
                      "trades": int(len(R)), "mean_per_trade": float(R.pnl.mean()),
                      "mean_ret_on_premium": float(R.ret_on_premium.mean()),
                      "median_ret_on_premium": float(R.ret_on_premium.median())})
    # provider simulation of the delta-hedged SPY books (daily marks), Options Funding Growth and Imperial Elite
    dly = []
    for d in dirs:
        for f in os.listdir(d):
            if f.startswith("daily_") and f.endswith(".csv.gz"):
                dly.append(pd.read_csv(os.path.join(d, f), parse_dates=["entry", "day"]))
    if dly:
        from options_prop_sim import run as prop_run
        Dd = pd.concat(dly, ignore_index=True)
        Dd = Dd[Dd.day <= END]
        books = {}
        for sym in ("SPY", "DIA"):
            g = Dd[(Dd.symbol == sym) & (Dd.kind == "short_straddle_hedged")]
            if len(g):
                books[f"{sym} hedged short straddle"] = g.groupby("day").inc.sum()
                books[f"{sym} hedged short straddle, contango"] = g[g.contango.astype(bool)].groupby("day").inc.sum()
        rows = []
        bdays = pd.bdate_range("2019-02-01", END)
        for name, ser in books.items():
            x = ser.reindex(bdays).fillna(0.0)
            first = x.ne(0).idxmax()
            x = x.loc[first:]
            sd = x.std()
            starts = pd.date_range(x.index[0] + pd.offsets.MonthBegin(1), x.index[-1] - pd.DateOffset(months=30), freq="MS")
            idx = [i for i in (x.index.get_indexer([t], method="bfill")[0] for t in starts) if i >= 0]
            eq = x.cumsum()
            for tgt in (0.004, 0.006, 0.008, 0.010):
                lam = min(tgt / sd, 4.0) if sd > 0 else 0.0
                for prov in ("OF", "IMP"):
                    Rr = pd.DataFrame([prop_run(x, lam, prov, i) for i in idx])
                    net = Rr.paid - Rr.fees
                    rows.append({"book": name, "provider": prov, "daily_sd_target": tgt, "notional_per_account": lam,
                                 "book_sum": float(x.sum()), "book_max_drawdown": float((eq - eq.cummax()).min()),
                                 "worst_day_account": float(x.min() * lam), "starts": int(len(Rr)),
                                 "pass_rate": float(Rr.passed.mean()), "paid_rate": float((Rr.paid > 0).mean()),
                                 "mean_net": float(net.mean()), "median_net": float(net.median()),
                                 "p10_net": float(net.quantile(0.1)), "p90_net": float(net.quantile(0.9)),
                                 "breach_rate": float(Rr.breach.mean())})
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "prop_hedged_spy.csv"), index=False)
        pd.DataFrame(books).to_csv(os.path.join(a.out, "daily_books.csv"))
    T = pd.DataFrame(tests)
    T["bh_q"] = L.bh(T.p_one_sided.to_numpy())
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    pd.concat({k: v for k, v in series.items()}, axis=1).to_csv(os.path.join(a.out, "series.csv"))
    # diagnostics: every position kind on SPY / DIA and stocks overall, by contango
    D = P.groupby(["symbol", "kind"]).pnl.agg(["count", "mean", "std"]).reset_index()
    D.to_csv(os.path.join(a.out, "positions_summary.csv"), index=False)
    if len(R):
        R.to_csv(os.path.join(a.out, "earnings_trades.csv.gz"), index=False)
    json.dump({"meta": meta, "tests": T.to_dict("records")}, open(os.path.join(a.out, "summary.json"), "w"),
              indent=1, default=str)
    pd.set_option("display.width", 250)
    cols = ["test", "n", "mean", "t_nw", "sharpe", "win_rate", "worst", "years_positive", "mean_2019_2022",
            "t_2019_2022", "mean_2023_2025", "t_2023_2025", "bh_q"]
    print(T[[c for c in cols if c in T.columns]].round(5).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("symbols", "combine"), required=True)
    ap.add_argument("--chain", default="")
    ap.add_argument("--closes", required=True)
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
