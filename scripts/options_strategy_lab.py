"""Options strategy lab: many option structures x many underlyings x management rules x entry filters, on REAL
end-of-day option quotes (bid / ask), with multiple-testing control and a walk-forward choice of the best.

Data: a chain panel (one row per date, underlying, expiration, strike, call/put with bid, ask and, where available,
implied volatility and delta) prepared by scripts/options_chain_fetch.py, plus daily closes of the underlyings and
of the VIX / VIX3M from Yahoo. Every date after 2025-12-31 is dropped on load (2026 is the locked holdout).

UNIT: P&L per share of underlying (one option = one share), divided by the underlying price at entry, i.e. the
return on the underlying notional of one contract set. Scale-free statistics (Sharpe, t) compare structures
with different leverage; tail statistics (worst cycle, CVaR 5%) are in the same unit.

STRUCTURES (legs: call/put, target |delta| or "atm", quantity; S = the underlying itself):
  SP30, SP16, SP50          short put 30 / 16 delta / at the money (cash-secured put)
  PCS30_16, PCS20_05        put credit spreads (short / long delta)
  CCS30_16                  call credit spread
  STRG16                    short strangle 16 / 16 delta
  STRD                      short straddle at the money
  IC16_05, IC30_16          iron condors (short / long delta, both sides)
  IFLY16                    iron butterfly: short ATM straddle, long 16-delta wings
  PRAT                      put ratio spread: long 1 x 40-delta put, short 2 x 20-delta puts
  CC30, BW50                covered call 30 delta / at the money (buy-write)
  COLL30                    collar: long underlying, long 30-delta put, short 30-delta call
  PCAL                      put calendar: short front ATM put, long the next expiry's ATM put (same strike)
  LP30, LSTRD               controls: long 30-delta put, long ATM straddle (buying premium)
  STK                       benchmark: long the underlying over the same cycle
ENTRY: once per standard monthly expiration (third Friday), on the first chain date at or after expiration
  minus DTE calendar days (DTE 30 and 45); weekly entries (DTE 7) where weekly expirations exist.
MANAGEMENT: H hold to expiration; T50 take profit at 50% of the credit (or of the debit); D21 close at 21 days to
  expiration; T50D21 both; T50SL2 take profit at 50%, stop when the loss reaches 2x the credit. D21 and T50D21
  are not run on weekly (7-day) entries.
  Decisions use the closing quotes of each later chain date (end-of-day; no intraday stops).
FILTERS at entry: ALL; CONT (VIX < VIX3M); IVHI (the underlying's ATM implied volatility above its trailing
  252-day median); VRP (ATM implied volatility above the underlying's realised volatility of the past 21 days).
FILLS: options at mid +- f x half-spread (buying pays more, selling receives less); f = 0.75 primary (effective
  spread about 75% of quoted, Muravyev & Pearson 2020), 1.0 (at the quote) and 0.4 (patient limit orders) as
  sensitivities; commission USD 0.65 per contract per leg (0.0065 per share); underlying 1 bp per side.
  Expiration: options settle at intrinsic value against the underlying close (no cost); assignment risk before
  expiration and dividends on short calls are ignored; the underlying leg earns dividends (Yahoo adjusted close).

STATISTICS: per combination, cycles, mean / median return per cycle, Newey-West (1 lag) t of the mean, Sharpe
  (annualised by cycles per year), share of positive cycles, worst cycle, CVaR 5%, alpha and beta against the
  underlying's return over the same cycle (the options premium net of plain equity exposure), and the same by year.
  Benjamini-Hochberg over every combination at the primary fill (one-sided, mean > 0), separately on alpha.
  Deflated Sharpe ratio of the best combination given the number of combinations tried.
WALK-FORWARD: each year Y from the third year of data, pick the combination with the best Sharpe on all cycles that
  ended before Y (minimum 24 cycles; families: all, defined-risk only, per underlying), trade it in Y; the
  concatenated out-of-sample cycles are the honest estimate of "picking the best".
Usage: python scripts/options_strategy_lab.py --chain DIR --out DIR"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as st

END = pd.Timestamp("2025-12-31")
COMM = 0.0065                      # USD 0.65 per contract per leg, per share
STK_COST = 1e-4                    # 1 bp per side on the underlying
FILLS = (0.75, 1.0, 0.4)
PRIMARY_FILL = 0.75
DTES = (30, 45)
WEEKLY_DTE = 7
MGMT = ("H", "T50", "D21", "T50D21", "T50SL2")
FILTERS = ("ALL", "CONT", "IVHI", "VRP")
SPLIT = pd.Timestamp("2023-01-01")       # in-sample / out-of-sample split of the real-quote years (both <= 2025)

# (kind, target, qty, expiry) ; kind in P, C, S ; target |delta| or "atm" ; expiry "front" or "back"
STRUCTS = {
    "SP30": [("P", 0.30, -1)],
    "SP16": [("P", 0.16, -1)],
    "SP50": [("P", "atm", -1)],
    "PCS30_16": [("P", 0.30, -1), ("P", 0.16, 1)],
    "PCS20_05": [("P", 0.20, -1), ("P", 0.05, 1)],
    "CCS30_16": [("C", 0.30, -1), ("C", 0.16, 1)],
    "STRG16": [("P", 0.16, -1), ("C", 0.16, -1)],
    "STRD": [("P", "atm", -1), ("C", "atm", -1)],
    "IC16_05": [("P", 0.16, -1), ("P", 0.05, 1), ("C", 0.16, -1), ("C", 0.05, 1)],
    "IC30_16": [("P", 0.30, -1), ("P", 0.16, 1), ("C", 0.30, -1), ("C", 0.16, 1)],
    "IFLY16": [("P", "atm", -1), ("C", "atm", -1), ("P", 0.16, 1), ("C", 0.16, 1)],
    "PRAT": [("P", 0.40, 1), ("P", 0.20, -2)],
    "CC30": [("S", None, 1), ("C", 0.30, -1)],
    "BW50": [("S", None, 1), ("C", "atm", -1)],
    "COLL30": [("S", None, 1), ("P", 0.30, 1), ("C", 0.30, -1)],
    "PCAL": [("P", "atm", -1, "front"), ("P", "atm", 1, "back")],
    "LP30": [("P", 0.30, 1)],
    "LSTRD": [("P", "atm", 1), ("C", "atm", 1)],
    "STK": [("S", None, 1)],
}
DEFINED_RISK = {"PCS30_16", "PCS20_05", "CCS30_16", "IC16_05", "IC30_16", "IFLY16", "PCAL", "LP30", "LSTRD",
                "COLL30"}
SHORT_PREMIUM = {"SP30", "SP16", "SP50", "PCS30_16", "PCS20_05", "CCS30_16", "STRG16", "STRD", "IC16_05",
                 "IC30_16", "IFLY16", "PRAT", "CC30", "BW50"}


# ----------------------------------------------------------------------------------------------- data
def bs_delta(cp, S, K, T, iv):
    if not (T > 0 and iv > 0 and S > 0 and K > 0):
        return np.nan
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T))
    nd = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    return nd if cp == "C" else nd - 1


def load_chain(path, sym):
    f = os.path.join(path, f"chain_{sym}.parquet")
    if not os.path.exists(f):
        f = os.path.join(path, f"chain_{sym}.csv.gz")
    C = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    C["date"] = pd.to_datetime(C["date"])
    C["expiration"] = pd.to_datetime(C["expiration"])
    C = C[(C.date <= END) & (C.expiration > C.date)]
    C["cp"] = C["cp"].astype(str).str[0].str.upper()
    C = C[(C.bid >= 0) & (C.ask > 0) & (C.ask >= C.bid)].copy()
    C["mid"] = (C.bid + C.ask) / 2
    C["half"] = (C.ask - C.bid) / 2
    return C


def third_fridays(start, end):
    out = []
    for m in pd.date_range(start, end, freq="MS"):
        d = m + pd.Timedelta(days=(4 - m.weekday()) % 7) + pd.Timedelta(days=14)
        out.append(d)
    return out


class Underlying:
    """Raw (split-adjusted, not dividend-adjusted) and total-return closes of the underlying; option strikes in
    the chain are as quoted on each date, so the raw close is reconstructed from the chain where splits matter."""

    def __init__(self, raw, adj):
        self.raw = raw.dropna()
        self.adj = adj.reindex(self.raw.index).ffill()

    def px(self, d):
        s = self.raw.loc[:d]
        return float(s.iloc[-1]) if len(s) else np.nan

    def tr(self, d0, d1):
        a0, a1 = self.adj.loc[:d0], self.adj.loc[:d1]
        return float(a1.iloc[-1] / a0.iloc[-1] - 1) if len(a0) and len(a1) else np.nan

    def rv21(self, d):
        r = np.log(self.adj.loc[:d]).diff().dropna().iloc[-21:]
        return float(r.std() * np.sqrt(252)) if len(r) >= 15 else np.nan


# ----------------------------------------------------------------------------------------------- engine
class Book:
    """Fast access to one underlying's chain: per date, per expiration, the puts / calls with quotes."""

    def __init__(self, C, und):
        self.C = C
        self.und = und
        self.dates = np.array(sorted(C.date.unique()))
        self.by_day = {d: g for d, g in C.groupby("date")}
        self.key = C.set_index(["expiration", "cp", "strike", "date"])[["bid", "ask", "mid", "half"]].sort_index()
        self.atm_iv = self._atm_iv_series()

    def _atm_iv_series(self):
        if "iv" not in self.C.columns:
            return pd.Series(dtype=float)
        out = {}
        for d, g in self.by_day.items():
            S = self.und.px(d)
            g = g[(g.expiration - d).dt.days.between(20, 60) & g.iv.notna() & (g.iv > 0)]
            if not len(g) or not np.isfinite(S):
                continue
            g = g.assign(dist=(g.strike - S).abs())
            out[d] = float(g.nsmallest(4, "dist").iv.mean())
        return pd.Series(out).sort_index()

    def parity_px(self, d):
        """Underlying implied by put-call parity at the strike nearest the close (nearest expiry >= 7 days)."""
        g = self.by_day.get(d)
        S = self.und.px(d)
        if g is None or not np.isfinite(S):
            return np.nan
        g = g[(g.expiration - d).dt.days >= 7]
        if not len(g):
            return np.nan
        e = g.expiration.min()
        g = g[g.expiration == e]
        P = g[g.cp == "P"].set_index("strike").mid
        Cc = g[g.cp == "C"].set_index("strike").mid
        ks = P.index.intersection(Cc.index)
        if not len(ks):
            return np.nan
        k = ks[np.argmin(np.abs(ks - S))]
        return float(k + Cc.loc[k] - P.loc[k]) if np.isscalar(Cc.loc[k]) and np.isscalar(P.loc[k]) else np.nan

    def expiries(self, d):
        g = self.by_day.get(d)
        return [pd.Timestamp(e) for e in sorted(g.expiration.unique())] if g is not None else []

    def pick(self, d, exp, cp, target, S, atm_strike=None):
        g = self.by_day[d]
        g = g[(g.expiration == exp) & (g.cp == cp) & (g.bid > 0)]
        if not len(g):
            return None
        if target == "atm":
            if atm_strike is not None and (g.strike == atm_strike).any():
                return float(atm_strike)
            return float(g.loc[(g.strike - S).abs().idxmin(), "strike"])
        T = max((exp - d).days, 1) / 365.0
        if "delta" in g.columns and g["delta"].notna().mean() > 0.5:
            dl = g["delta"].abs()
        else:
            dl = pd.Series([abs(bs_delta(cp, S, k, T, v)) for k, v in zip(g.strike, g.iv)], index=g.index)
        ok = dl.notna()
        if not ok.any():
            return None
        dist = (dl[ok] - target).abs()
        if dist.min() > 0.08:                 # nothing near the target delta
            return None
        return float(g.loc[dist.idxmin(), "strike"])

    def quote(self, exp, cp, k, d):
        try:
            r = self.key.loc[(exp, cp, k, d)]
            return float(r.mid), float(r.half)
        except KeyError:
            return None

    def path(self, exp, cp, k, d0, d1):
        try:
            s = self.key.loc[(exp, cp, k)]
        except KeyError:
            return pd.DataFrame(columns=["mid", "half"])
        return s.loc[(s.index >= d0) & (s.index <= d1), ["mid", "half"]]


def build_position(bk, d, struct, dte, S):
    """Choose expiries and strikes on entry date d; return legs [(kind, cp, exp, strike, qty)] or None."""
    exps = bk.expiries(d)
    if not len(exps):
        return None
    days = np.array([(e - d).days for e in exps])
    tgt = d + pd.Timedelta(days=dte)
    if dte == WEEKLY_DTE:
        cand = [e for e, dd in zip(exps, days) if 5 <= dd <= 9]
    else:
        exps_t = [pd.Timestamp(e) for e in exps]
        fri = [e for e in exps_t if 15 <= e.day <= 21 and e.weekday() == 4]
        fri_months = {(e.year, e.month) for e in fri}
        thu = [e for e in exps_t if 15 <= e.day <= 21 and e.weekday() == 3 and (e.year, e.month) not in fri_months
               and (e + pd.Timedelta(days=1)).day <= 21]          # third Friday a holiday (Good Friday)
        monthly = fri + thu
        cand = [e for e in monthly if abs((e - tgt).days) <= 10]
    if not cand:
        return None
    front = min(cand, key=lambda e: abs((e - tgt).days))
    back = None
    if any(len(l) > 3 and l[3] == "back" for l in struct):
        later = [e for e in exps if 21 <= (e - front).days <= 45]
        if not later:
            return None
        back = min(later, key=lambda e: abs((e - front).days - 28))
    legs, atm_k = [], None
    for leg in struct:
        kind, target, q = leg[0], leg[1], leg[2]
        if kind == "S":
            legs.append(("S", None, None, None, q))
            continue
        e = back if (len(leg) > 3 and leg[3] == "back") else front
        k = bk.pick(d, e, kind, target, S, atm_strike=atm_k if target == "atm" else None)
        if k is None:
            return None
        if target == "atm":
            atm_k = k
        legs.append(("O", kind, e, k, q))
    # wings must be further out of the money than the short legs of the same type
    for cp in ("P", "C"):
        sh = [l[3] for l in legs if l[0] == "O" and l[1] == cp and l[4] < 0 and l[2] == front]
        lo = [l[3] for l in legs if l[0] == "O" and l[1] == cp and l[4] > 0 and l[2] == front]
        if sh and lo and struct not in (STRUCTS["PRAT"],):
            if cp == "P" and min(lo) >= max(sh):
                return None
            if cp == "C" and max(lo) <= min(sh):
                return None
    return legs, front


def cycle_paths(bk, d, legs, front):
    """Daily table from entry to the front expiration: per option leg mid / half on each chain date (NaN when a
    quote is missing), and the underlying raw close."""
    days = bk.dates[(bk.dates >= d) & (bk.dates <= front)]
    T = pd.DataFrame(index=pd.DatetimeIndex(days))
    for i, (kind, cp, e, k, q) in enumerate(legs):
        if kind != "O":
            continue
        p = bk.path(e, cp, k, d, front)
        T[f"m{i}"] = p["mid"].reindex(T.index)
        T[f"h{i}"] = p["half"].reindex(T.index)
    return T


def evaluate(bk, d, legs, front, T, mgmt, f):
    """Return (pnl_per_share, exit_date, credit, days_held, mtm series at mid)."""
    S0 = bk.und.px(d)
    opt = [(i, l) for i, l in enumerate(legs) if l[0] == "O"]
    stk = [l for l in legs if l[0] == "S"]
    if T.empty or any(not np.isfinite(T.iloc[0].get(f"m{i}", np.nan)) for i, _ in opt):
        return None
    row0 = T.iloc[0]
    entry_cf = 0.0                                           # cash received at entry (credit > 0)
    for i, (_, cp, e, k, q) in opt:
        entry_cf += -q * (row0[f"m{i}"] + np.sign(q) * f * row0[f"h{i}"])
    n_contracts = sum(abs(l[4]) for _, l in opt)
    comm = COMM * n_contracts
    credit = entry_cf
    ref = abs(credit) if abs(credit) > 1e-9 else np.nan

    def close_value(row):
        v = 0.0
        for i, (_, cp, e, k, q) in opt:
            m, h = row[f"m{i}"], row[f"h{i}"]
            if not np.isfinite(m):
                return np.nan
            v += q * (m - np.sign(q) * f * h)                  # long sells at bid side, short buys at ask side
        return v

    exit_d, exit_val, closed_early = None, None, False
    for dd, row in T.iloc[1:].iterrows():
        if dd >= front:
            break
        dte_left = (front - dd).days
        if mgmt in ("D21", "T50D21") and dte_left <= 21:
            v = close_value(row)
            if np.isfinite(v):
                exit_d, exit_val, closed_early = dd, v, True
                break
        if mgmt in ("T50", "T50D21", "T50SL2") and np.isfinite(ref):
            v = close_value(row)
            if np.isfinite(v):
                pnl_now = entry_cf + v - 2 * comm
                if pnl_now >= 0.5 * ref:
                    exit_d, exit_val, closed_early = dd, v, True
                    break
                if mgmt == "T50SL2" and credit > 0 and pnl_now <= -2 * credit:
                    exit_d, exit_val, closed_early = dd, v, True
                    break
    if exit_d is None:                                       # expiration: front legs at intrinsic
        exit_d = front
        Sx = bk.und.px(front)
        v = 0.0
        for i, (_, cp, e, k, q) in opt:
            if e == front:
                v += q * (max(Sx - k, 0) if cp == "C" else max(k - Sx, 0))
            else:                                            # back leg of a calendar closed at the quote
                p = bk.path(e, cp, k, front - pd.Timedelta(days=4), front)
                if not len(p):
                    return None
                m, h = float(p["mid"].iloc[-1]), float(p["half"].iloc[-1])
                v += q * (m - np.sign(q) * f * h)
                comm_back = COMM * abs(q)
                v -= comm_back
        exit_val = v
        comm_close = 0.0                                     # expiring options: no closing commission
    else:
        comm_close = comm
    pnl = entry_cf + exit_val - comm - comm_close
    for (_, _, _, _, q) in stk:
        r = bk.und.tr(d, exit_d)
        pnl += q * S0 * (r - 2 * STK_COST)
    return {"pnl": pnl / S0, "credit": credit / S0, "exit": exit_d, "early": closed_early,
            "days": int((exit_d - d).days)}


def mtm_series(bk, d, legs, T, exit_d, S0):
    """Daily mark-to-market at mid (P&L increments per share / S0) from entry to exit, for portfolio statistics."""
    opt = [(i, l) for i, l in enumerate(legs) if l[0] == "O"]
    U = T.loc[:exit_d]
    v = pd.Series(0.0, index=U.index)
    for i, (_, cp, e, k, q) in opt:
        v = v + q * U[f"m{i}"].ffill()
    for (_, _, _, _, q) in [l for l in legs if l[0] == "S"]:
        adj = bk.und.adj.reindex(U.index).ffill()
        v = v + q * S0 * adj / adj.iloc[0]
    return v.diff().fillna(0.0) / S0


# ----------------------------------------------------------------------------------------------- statistics
def nw_t(x, lag=1):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.nan
    e = x - x.mean()
    g0 = e @ e / n
    s = g0
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pp = p[ok]
    o = np.argsort(pp)
    r = pp[o] * len(pp) / (np.arange(len(pp)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(len(pp))
    qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def lr_sd(x, lag=1):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return math.sqrt(s) if s > 0 else np.nan


def _sr(x, per_year):
    x = np.asarray(x, float)
    if len(x) < 8:
        return np.nan
    sd = lr_sd(x)
    return float(x.mean() / sd * math.sqrt(per_year)) if sd and np.isfinite(sd) else np.nan


def summarise(R, per_year):
    x = R.pnl.to_numpy()
    n = len(x)
    if n < 8:
        return {"cycles": n}
    t = nw_t(x)
    u = R.und_ret.to_numpy()
    ok = np.isfinite(u)
    if ok.sum() >= 8 and np.std(u[ok]) > 0:
        b, a = np.polyfit(u[ok], x[ok], 1)
        res = x[ok] - (a + b * u[ok])
        ta = a / (res.std(ddof=2) / math.sqrt(ok.sum())) if res.std() > 0 else np.nan
    else:
        a = b = ta = np.nan
    yrs = R.groupby(R.entry.dt.year).pnl.sum()
    sd = x.std(ddof=1)
    k = max(1, int(round(0.05 * n)))
    is_ = R.entry < SPLIT
    return {"cycles": n, "mean": float(x.mean()), "median": float(np.median(x)), "sd": float(sd), "t_nw": t,
            "p_one_sided": float(1 - st.norm.cdf(t)) if np.isfinite(t) else np.nan,
            "sharpe": _sr(x, per_year), "sharpe_naive": float(x.mean() / sd * math.sqrt(per_year)) if sd > 0 else np.nan,
            "win_rate": float((x > 0).mean()), "worst": float(x.min()), "cvar5": float(np.sort(x)[:k].mean()),
            "alpha": float(a), "beta": float(b), "t_alpha": float(ta),
            "p_alpha": float(1 - st.t.cdf(ta, ok.sum() - 2)) if np.isfinite(ta) else np.nan,
            "years_positive": float((yrs > 0).mean()), "worst_year": float(yrs.min()),
            "skew": float(st.skew(x)), "kurt": float(st.kurtosis(x)),
            "early_exit_share": float(R.early.mean()), "mean_days": float(R.days.mean()),
            "mean_credit": float(R.credit.mean()),
            "cycles_is": int(is_.sum()), "mean_is": float(x[is_.to_numpy()].mean()) if is_.any() else np.nan,
            "sharpe_is": _sr(x[is_.to_numpy()], per_year),
            "cycles_oos": int((~is_).sum()), "mean_oos": float(x[~is_.to_numpy()].mean()) if (~is_).any() else np.nan,
            "sharpe_oos": _sr(x[~is_.to_numpy()], per_year),
            **{f"y{y}": float(v) for y, v in yrs.items()},
            "first": str(R.entry.min().date()), "last": str(R.entry.max().date())}


def deflated_sharpe(sr_best, n_trials, n_obs, skew, kurt, sr_var):
    """Bailey & Lopez de Prado (2014), per-period Sharpe ratios."""
    if n_trials < 2 or not np.isfinite(sr_best):
        return np.nan
    emc = 0.5772156649
    z = (1 - emc) * st.norm.ppf(1 - 1 / n_trials) + emc * st.norm.ppf(1 - 1 / (n_trials * math.e))
    sr0 = math.sqrt(max(sr_var, 0)) * z
    den = math.sqrt(max(1 - skew * sr_best + ((kurt + 2) / 4) * sr_best ** 2, 1e-9))   # kurt is excess kurtosis
    return float(st.norm.cdf((sr_best - sr0) * math.sqrt(n_obs - 1) / den))


# ----------------------------------------------------------------------------------------------- main loop
def entry_dates(bk, dte):
    if dte == WEEKLY_DTE:
        fr = [d for d in bk.dates if pd.Timestamp(d).weekday() == 4]
        return [pd.Timestamp(d) for d in fr]
    out = []
    for e in third_fridays(bk.dates[0], bk.dates[-1] + np.timedelta64(70, "D")):
        tgt = e - pd.Timedelta(days=dte)
        nxt = bk.dates[bk.dates >= np.datetime64(tgt)]
        if len(nxt) and (pd.Timestamp(nxt[0]) - tgt).days <= 4 and pd.Timestamp(nxt[0]) < e:
            out.append(pd.Timestamp(nxt[0]))
    return out


def run_book(sym, bk, und, vix, keep_mtm=True, dtes=None):
    """Every structure x entry date x management x fill on one underlying's book (real or synthetic)."""
    dtes = dtes or (DTES + (WEEKLY_DTE,))
    has_weekly = False
    rows, mtm = [], {}
    ivmed = bk.atm_iv.rolling(252, min_periods=120).median()
    n_mismatch = 0
    for dte in dtes:
        for d in entry_dates(bk, dte):
            S = und.px(d)
            if not np.isfinite(S):
                continue
            if isinstance(bk, Book) and type(bk) is Book:
                Sp = bk.parity_px(d)
                if np.isfinite(Sp) and abs(math.log(Sp / S)) > 0.03:     # chain and close disagree (split, bad row)
                    n_mismatch += 1
                    continue
            iv = bk.atm_iv.get(d, np.nan)
            vv = vix.loc[:d]
            flt = {"ALL": True,
                   "CONT": bool(len(vv) and vv.iloc[-1, 0] < vv.iloc[-1, 1]),
                   "IVHI": bool(np.isfinite(iv) and np.isfinite(ivmed.get(d, np.nan)) and iv > ivmed.get(d)),
                   "VRP": bool(np.isfinite(iv) and iv > und.rv21(d))}
            for name, struct in STRUCTS.items():
                bp = build_position(bk, d, struct, dte, S)
                if bp is None:
                    continue
                legs, front = bp
                if dte == WEEKLY_DTE:
                    has_weekly = True
                T = cycle_paths(bk, d, legs, front)
                for mg in MGMT:
                    if name == "STK" and mg != "H":
                        continue
                    if dte == WEEKLY_DTE and mg in ("D21", "T50D21"):      # 21-day exit undefined for weeklies
                        continue
                    for f in FILLS:
                        if f != PRIMARY_FILL and mg not in ("H", "T50D21" if dte != WEEKLY_DTE else "T50"):
                            continue
                        r = evaluate(bk, d, legs, front, T, mg, f)
                        if r is None:
                            continue
                        r.update({"symbol": sym, "struct": name, "dte": dte, "mgmt": mg, "fill": f, "entry": d,
                                  "expiry": front, "und_ret": und.tr(d, r["exit"]),
                                  **{f"f_{k}": v for k, v in flt.items()},
                                  "legs": ";".join(f"{l[1]}{l[3]:g}x{l[4]}" if l[0] == "O" else f"Sx{l[4]}"
                                                   for l in legs)})
                        rows.append(r)
                        if keep_mtm and f == PRIMARY_FILL:
                            mtm[(sym, name, dte, mg, d)] = mtm_series(bk, d, legs, T, r["exit"], S)
    R = pd.DataFrame(rows)
    meta = {"weekly_expiries": has_weekly, "chain_dates": int(len(bk.dates)), "entries_skipped_price_mismatch": n_mismatch,
            "first": str(pd.Timestamp(bk.dates[0]).date()), "last": str(pd.Timestamp(bk.dates[-1]).date())}
    return R, meta, mtm


def trim_chain(C, und, max_dte=100, lo=0.6, hi=1.4):
    """Keep what any structure can use: expiries within 100 days, strikes within 60-140% of the underlying."""
    S = und.raw.reindex(C.date.unique()).rename("S")
    C = C.merge(S, left_on="date", right_index=True, how="inner")
    keep = ((C.expiration - C.date).dt.days <= max_dte) & (C.strike >= lo * C.S) & (C.strike <= hi * C.S)
    return C[keep].drop(columns="S")


def run_symbol(sym, chain_dir, und, vix, keep_mtm=True):
    C = load_chain(chain_dir, sym)
    if C.empty:
        return pd.DataFrame(), {}, {}
    n0 = len(C)
    C = trim_chain(C, und)
    R, meta, mtm = run_book(sym, Book(C, und), und, vix, keep_mtm)
    meta.update({"rows_loaded": int(n0), "rows_used": int(len(C))})
    return R, meta, mtm


def table(R):
    """One row per combination (symbol, struct, dte, mgmt, filter, fill)."""
    out = []
    for (sym, s, dte, mg, f), g in R.groupby(["symbol", "struct", "dte", "mgmt", "fill"]):
        per_year = 52.0 if dte == WEEKLY_DTE else 12.0
        for flt in FILTERS:
            h = g[g[f"f_{flt}"]]
            if len(h) < 8:
                continue
            out.append({"symbol": sym, "struct": s, "dte": dte, "mgmt": mg, "filter": flt, "fill": f,
                        "defined_risk": s in DEFINED_RISK, "short_premium": s in SHORT_PREMIUM,
                        **summarise(h.sort_values("entry"), per_year)})
    return pd.DataFrame(out)


def walk_forward(R, T, family, min_cycles=24):
    """Each year: pick the best-Sharpe combination on cycles that ended before the year, trade it in that year."""
    P = R[R.fill == PRIMARY_FILL].copy()
    P["key"] = P.symbol + "|" + P.struct + "|" + P.dte.astype(str) + "|" + P.mgmt
    years = sorted(P.entry.dt.year.unique())
    picks, oos = [], []
    for y in years[2:]:
        hist = P[P.exit < pd.Timestamp(f"{y}-01-01")]
        best, best_sr, best_flt = None, -np.inf, None
        for flt in FILTERS:
            H = hist[hist[f"f_{flt}"]]
            for key, g in H.groupby("key"):
                if not family(g.iloc[0]) or len(g) < min_cycles:
                    continue
                per_year = 52.0 if g.dte.iloc[0] == WEEKLY_DTE else 12.0
                sr = _sr(g.sort_values("entry").pnl, per_year)
                if np.isfinite(sr) and sr > best_sr:
                    best, best_sr, best_flt = key, sr, flt
        if best is None:
            continue
        nxt = P[(P.key == best) & (P.entry.dt.year == y) & P[f"f_{best_flt}"]]
        picks.append({"year": int(y), "pick": best, "filter": best_flt, "in_sample_sharpe": float(best_sr),
                      "oos_cycles": int(len(nxt)), "oos_sum": float(nxt.pnl.sum()), "oos_mean": float(nxt.pnl.mean())
                      if len(nxt) else np.nan})
        oos.append(nxt)
    O = pd.concat(oos) if oos else pd.DataFrame()
    return pd.DataFrame(picks), O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--closes", required=True, help="CSV of daily closes: date, <SYM>_raw, <SYM>_adj, VIX, VIX3M")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= END]
    vix = X[["VIX", "VIX3M"]].dropna()
    syms = [s for s in a.symbols.split(",") if s] or sorted(
        f.split("_", 1)[1].split(".")[0] for f in os.listdir(a.chain) if f.startswith("chain_"))
    allR, meta, MTM = [], {}, {}
    for sym in syms:
        if f"{sym}_raw" not in X.columns:
            meta[sym] = {"error": "no closes"}
            continue
        und = Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
        R, m, mtm = run_symbol(sym, a.chain, und, vix, keep_mtm=True)
        meta[sym] = m
        print(sym, m, len(R), flush=True)
        if len(R):
            allR.append(R)
            MTM.update(mtm)
    R = pd.concat(allR, ignore_index=True)
    R.to_csv(os.path.join(a.out, "cycles.csv.gz"), index=False)
    T = table(R)
    prim = T.fill == PRIMARY_FILL
    T.loc[prim, "bh_q_mean"] = bh(T.loc[prim, "p_one_sided"])
    T.loc[prim, "bh_q_alpha"] = bh(T.loc[prim, "p_alpha"])
    n_trials = int(prim.sum())
    best = T[prim].sort_values("sharpe", ascending=False)
    if len(best):
        b = best.iloc[0]
        per = 52.0 if b.dte == WEEKLY_DTE else 12.0
        sr_pp = b.sharpe / math.sqrt(per)
        sr_var = float((T.loc[prim, "sharpe"] / np.sqrt(np.where(T.loc[prim, "dte"] == WEEKLY_DTE, 52, 12))).var())
        T.loc[best.index[0], "deflated_sharpe_prob"] = deflated_sharpe(sr_pp, n_trials, int(b["cycles"]), b["skew"],
                                                                       b["kurt"], sr_var)
    T.to_csv(os.path.join(a.out, "combos.csv"), index=False)
    fams = {"all": lambda r: True,
            "defined_risk": lambda r: r.struct in DEFINED_RISK,
            "short_premium_defined": lambda r: r.struct in DEFINED_RISK and r.struct in SHORT_PREMIUM,
            "SPY_QQQ": lambda r: r.symbol in ("SPY", "QQQ")}
    wf = {}
    for name, fam in fams.items():
        picks, O = walk_forward(R, T, fam)
        picks.to_csv(os.path.join(a.out, f"walkforward_{name}.csv"), index=False)
        if len(O):
            m = O.groupby(O.entry.dt.to_period("M")).pnl.sum()
            wf[name] = {"oos_cycles": int(len(O)), "oos_months": int(len(m)), "oos_mean_cycle": float(O.pnl.mean()),
                        "oos_mean_month": float(m.mean()), "oos_t_nw_month": nw_t(m.to_numpy()),
                        "oos_sharpe_month": _sr(m.to_numpy(), 12.0), "oos_worst_cycle": float(O.pnl.min()),
                        "oos_win": float((O.pnl > 0).mean()), "years": picks.year.tolist(),
                        "picks": picks.pick.tolist()}
    # does in-sample rank predict out-of-sample rank? (primary fill, no filter)
    Z = T[prim & (T["filter"] == "ALL")].dropna(subset=["sharpe_is", "sharpe_oos"])
    rank_stab = {"combos": int(len(Z)),
                 "spearman_is_oos": float(st.spearmanr(Z.sharpe_is, Z.sharpe_oos)[0]) if len(Z) > 10 else np.nan}
    if len(Z) > 10:
        top_is = Z.sort_values("sharpe_is", ascending=False).head(max(1, len(Z) // 10))
        rank_stab.update({"top_decile_is_mean_oos_sharpe": float(top_is.sharpe_oos.mean()),
                          "all_mean_oos_sharpe": float(Z.sharpe_oos.mean()),
                          "top_decile_is_share_oos_positive": float((top_is.mean_oos > 0).mean())})
    # daily mark-to-market portfolio (one new position per cycle, overlapping positions summed) of the 30 best
    # primary combinations by Sharpe, any filter: max drawdown, worst day, daily Sharpe; saved for the prop sim
    top = T[prim & (T.cycles >= 24)].sort_values("sharpe", ascending=False).head(30)
    daily, dd_rows = {}, []
    for _, r in top.iterrows():
        g = R[(R.symbol == r.symbol) & (R.struct == r.struct) & (R.dte == r.dte) & (R.mgmt == r.mgmt) &
              (R.fill == PRIMARY_FILL) & R[f"f_{r['filter']}"]]
        parts = [MTM[(r.symbol, r.struct, r.dte, r.mgmt, d)] for d in g.entry if (r.symbol, r.struct, r.dte, r.mgmt, d) in MTM]
        if not parts:
            continue
        sser = pd.concat(parts, axis=1).sum(axis=1).sort_index()
        key = f"{r.symbol}|{r.struct}|{r.dte}|{r.mgmt}|{r['filter']}"
        daily[key] = sser
        eq = sser.cumsum()
        dd_rows.append({"combo": key, "days": int(len(sser)), "sum": float(sser.sum()),
                        "daily_sharpe_ann": float(sser.mean() / sser.std() * math.sqrt(252)) if sser.std() > 0 else np.nan,
                        "max_drawdown": float((eq - eq.cummax()).min()), "worst_day": float(sser.min()),
                        "worst_day_date": str(sser.idxmin().date())})
    if daily:
        pd.DataFrame(daily).sort_index().fillna(0.0).to_csv(os.path.join(a.out, "daily_mtm_top30.csv"))
        pd.DataFrame(dd_rows).to_csv(os.path.join(a.out, "top30_drawdowns.csv"), index=False)
    summ = {"meta": meta, "n_combinations_primary": n_trials, "walk_forward": wf, "rank_stability": rank_stab,
            "bh_survivors_mean": int((T.loc[prim, "bh_q_mean"] < 0.05).sum()),
            "bh_survivors_alpha": int((T.loc[prim, "bh_q_alpha"] < 0.05).sum()),
            "best_primary": best.head(25).to_dict("records") if len(best) else []}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "best_primary"}, indent=1, default=str))
    pd.set_option("display.width", 250)
    cols = ["symbol", "struct", "dte", "mgmt", "filter", "cycles", "mean", "t_nw", "sharpe", "win_rate", "worst",
            "alpha", "t_alpha", "bh_q_mean", "bh_q_alpha"]
    print(best[cols].head(40).round(4).to_string())


if __name__ == "__main__":
    main()
