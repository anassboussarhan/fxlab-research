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
ENTRY: once a month (on the first chain date at or after the third Friday minus DTE calendar days; DTE 30 and 45,
  into the listed expiry 20-42 / 35-57 days out) and every week (DTE 14: the expiry 9-18 days out). The database
  lists about three expiries per date (about 2, 4 and 6-8 weeks), not always standard monthlies, and 10-27 strikes
  per expiry, so legs are matched to the nearest listed delta within 0.08.
MANAGEMENT: H hold to expiration; T50 take profit at 50% of the credit (or of the debit); D21 close at 21 days to
  expiration; T50D21 both; T50SL2 take profit at 50%, stop when the loss reaches 2x the credit. D21 and T50D21
  are not run on the 14-day weekly entries.
  Decisions use the closing quotes of each later chain date (end-of-day; no intraday stops).
FILTERS at entry: ALL; CONT (VIX < VIX3M); IVHI (the underlying's ATM implied volatility above its trailing
  252-day median); VRP (ATM implied volatility above the underlying's realised volatility of the past 21 days);
  NOERN (single stocks: no earnings announcement between entry and expiration; DoltHub post-no-preference/earnings).
  Snapshots: weekly in 2019, about three a week (Mon / Wed / Fri) 2020-2023, daily from 2024, so management is
  checked on those closes.
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
WEEKLY_DTE = 14                    # weekly entries into the nearest (about two-week) expiry
MGMT = ("H", "T50", "D21", "T50D21", "T50SL2")
FILTERS = ("ALL", "CONT", "IVHI", "VRP", "NOERN")
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
        # corporate-action days: the traded price and the total-return price move apart by more than 3% in a day
        # (splits, reverse splits, special dividends, spin-offs that Yahoo adjusts); option contracts are
        # re-written on those days, so a position spanning one cannot be valued from the quoted strikes
        gap = (np.log(self.raw).diff() - np.log(self.adj).diff()).abs()
        self.events = pd.DatetimeIndex(gap[gap > 0.03].index)

    def has_event(self, d0, d1):
        return bool(((self.events > d0) & (self.events <= d1)).any())

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
        self.surf = self._surfaces() if "iv" in C.columns else {}
        self.exp_by = {}
        for (d, e, cp) in self.surf:
            self.exp_by.setdefault((d, cp), []).append(e)
        for k in self.exp_by:
            self.exp_by[k].sort()
        self.n_listed = 0
        self.n_modeled = 0
        self.n_term = 0

    def _surfaces(self):
        """Per snapshot date and expiry: the forward (put-call parity at the strike nearest the close) and, per
        call / put, the listed strikes' log-moneyness, implied vol and half-spread. The database lists a
        different set of strikes on each snapshot, so a contract held is often unquoted on a later date; it is
        then valued from that date's own smile (Black-76 on the forward, vol and half-spread interpolated in
        log-moneyness, flat beyond the listed range)."""
        out = {}
        for (d, e), g in self.C.groupby(["date", "expiration"]):
            S = self.und.px(d)
            P = g[g.cp == "P"].set_index("strike").mid
            Cc = g[g.cp == "C"].set_index("strike").mid
            ks = P.index.intersection(Cc.index)
            F = S
            if len(ks) and np.isfinite(S):
                k0 = ks[np.argmin(np.abs(ks - S))]
                try:
                    Fx = float(k0 + Cc.loc[k0] - P.loc[k0])
                    if 0.8 * S < Fx < 1.2 * S:
                        F = Fx
                except (TypeError, ValueError):
                    pass
            if not np.isfinite(F) or F <= 0:
                continue
            for cp in ("P", "C"):
                h = g[(g.cp == cp) & (g.iv > 0)].sort_values("strike")
                if len(h) >= 2:
                    out[(d, e, cp)] = (F, np.log(h.strike.to_numpy() / F), h.iv.to_numpy(float), h.half.to_numpy(float))
        return out

    def smile(self, d, e, cp, k):
        """(implied vol, forward, half-spread) of strike k, expiry e on date d. From e's own smile when e is listed
        on d; otherwise (the database lists different expiries on each snapshot, so a held expiry is often absent)
        total implied variance is interpolated linearly in time between the listed expiries bracketing e, at the
        same log-moneyness (flat in vol beyond the first / last listed expiry)."""
        T = (e - d).days
        if T <= 0:
            return None
        s = self.surf.get((d, e, cp))
        if s is not None:
            F, x, iv, half = s
            xk = math.log(k / F)
            return float(np.interp(xk, x, iv)), F, float(np.interp(xk, x, half))
        lst = self.exp_by.get((d, cp)) or self.exp_by.get((d, "C" if cp == "P" else "P"))
        if not lst:
            return None
        cpx = cp if (d, cp) in self.exp_by else ("C" if cp == "P" else "P")
        lo = [x for x in lst if x < e]
        hi = [x for x in lst if x > e]
        pts = ([lo[-1]] if lo else []) + ([hi[0]] if hi else [])
        vals = []
        for ex in pts:
            F, x, iv, half = self.surf[(d, ex, cpx)]
            xk = math.log(k / F)
            vals.append(((ex - d).days, float(np.interp(xk, x, iv)), F, float(np.interp(xk, x, half))))
        self.n_term += 1
        if len(vals) == 1:
            _, v, F, h = vals[0]
            return v, F, h
        (t1, v1, F1, h1), (t2, v2, F2, h2) = vals
        w = (T - t1) / (t2 - t1)
        tv = (1 - w) * v1 * v1 * t1 + w * v2 * v2 * t2
        v = math.sqrt(max(tv, 1e-12) / T)
        return v, (1 - w) * F1 + w * F2, (1 - w) * h1 + w * h2

    def model(self, d, e, cp, k):
        sm = self.smile(d, e, cp, k)
        if sm is None:
            return None
        v, F, h = sm
        T = (e - d).days / 365.0
        if T <= 0 or v <= 0:
            return None
        sd = v * math.sqrt(T)
        d1 = (math.log(F / k) + 0.5 * sd * sd) / sd
        n1, n2 = 0.5 * (1 + math.erf(d1 / math.sqrt(2))), 0.5 * (1 + math.erf((d1 - sd) / math.sqrt(2)))
        price = F * n1 - k * n2 if cp == "C" else k * (1 - n2) - F * (1 - n1)
        return max(price, 0.0), max(h, 0.005)

    def term_check(self, n=2000, seed=11):
        """Accuracy of the across-expiry valuation: on a listed (date, expiry), hide that whole expiry (both
        calls and puts) and value its listed quotes from the other expiries of the same date."""
        rng = np.random.default_rng(seed)
        keys = [(d, e) for (d, e, cp) in self.surf if cp == "P" and len(self.exp_by.get((d, "P"), [])) >= 3]
        if not keys:
            return {}
        errs = []
        for _ in range(n):
            d, e = keys[rng.integers(len(keys))]
            g = self.by_day[d]
            g = g[(g.expiration == e) & (g.bid > 0)]
            if not len(g):
                continue
            r = g.iloc[rng.integers(len(g))]
            saved = {cp: self.surf.pop((d, e, cp), None) for cp in ("P", "C")}
            saved_lists = {cp: list(self.exp_by.get((d, cp), [])) for cp in ("P", "C")}
            for cp in ("P", "C"):
                if e in self.exp_by.get((d, cp), []):
                    self.exp_by[(d, cp)].remove(e)
            m = self.model(d, e, r.cp, float(r.strike))
            for cp in ("P", "C"):
                if saved[cp] is not None:
                    self.surf[(d, e, cp)] = saved[cp]
                self.exp_by[(d, cp)] = saved_lists[cp]
            if m is None:
                continue
            S = self.und.px(d)
            errs.append((m[0] - r.mid) / S)
        if not errs:
            return {}
        e_ = np.abs(np.array(errs))
        return {"n": int(len(errs)), "median_abs_error_pct_of_underlying": float(np.median(e_) * 100),
                "p90_abs_error_pct_of_underlying": float(np.quantile(e_, 0.9) * 100),
                "mean_error_pct_of_underlying": float(np.mean(errs) * 100)}

    def model_check(self, n=3000, seed=7):
        """Leave-one-out accuracy of the smile valuation on listed quotes (each removed from its own smile)."""
        rng = np.random.default_rng(seed)
        keys = [k for k, v in self.surf.items() if len(v[1]) >= 4]
        if not keys:
            return {}
        err, err_s = [], []
        for _ in range(n):
            d, e, cp = keys[rng.integers(len(keys))]
            F, x, iv, half = self.surf[(d, e, cp)]
            j = int(rng.integers(1, len(x) - 1))                     # interior strikes only
            k = F * math.exp(x[j])
            self.surf[(d, e, cp)] = (F, np.delete(x, j), np.delete(iv, j), np.delete(half, j))
            m = self.model(d, e, cp, k)
            self.surf[(d, e, cp)] = (F, x, iv, half)
            q = self.quote(e, cp, round(k, 2), d) or self.quote(e, cp, k, d)
            if m is None or q is None:
                continue
            err.append(m[0] - q[0])
            err_s.append((m[0] - q[0]) / F)
        if not err:
            return {}
        err, err_s = np.array(err), np.array(err_s)
        return {"n": int(len(err)), "median_abs_error_per_share": float(np.median(np.abs(err))),
                "median_abs_error_pct_of_underlying": float(np.median(np.abs(err_s)) * 100),
                "p90_abs_error_pct_of_underlying": float(np.quantile(np.abs(err_s), 0.9) * 100),
                "mean_error_pct_of_underlying": float(err_s.mean() * 100)}

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

    def path(self, exp, cp, k, d0, d1, listed_only=False):
        """Quotes of one contract from d0 to d1 (before expiry): listed where quoted, otherwise valued from that
        date's smile."""
        try:
            s = self.key.loc[(exp, cp, k)]
            s = s.loc[(s.index >= d0) & (s.index <= d1), ["mid", "half"]]
        except KeyError:
            s = pd.DataFrame(columns=["mid", "half"])
        if not self.surf or listed_only:
            return s
        days = self.dates[(self.dates >= np.datetime64(d0)) & (self.dates <= np.datetime64(d1)) &
                          (self.dates < np.datetime64(exp))]
        have = set(s.index)
        add = {}
        for t in days:
            t = pd.Timestamp(t)
            if t in have:
                continue
            m = self.model(t, exp, cp, k)
            if m is not None:
                add[t] = m
        self.n_listed += len(have)
        self.n_modeled += len(add)
        self.n_gap = getattr(self, "n_gap", 0) + int(sum(1 for t in days if pd.Timestamp(t) not in have and
                                                          pd.Timestamp(t) not in add))
        if add:
            s = pd.concat([s, pd.DataFrame(list(add.values()), index=pd.DatetimeIndex(list(add.keys())),
                                           columns=["mid", "half"])]).sort_index()
        return s


def build_position(bk, d, struct, dte, S):
    """Choose expiries and strikes on entry date d; return legs [(kind, cp, exp, strike, qty)] or None."""
    exps = bk.expiries(d)
    if not len(exps):
        return None
    days = np.array([(e - d).days for e in exps])
    tgt = d + pd.Timedelta(days=dte)
    # the database lists about three expiries per date (about 2, 4 and 6-8 weeks out, not always the standard
    # monthly): take the listed expiry nearest the target
    lo, hi = (9, 18) if dte == WEEKLY_DTE else (dte - 10, dte + 12)
    cand = [e for e, dd in zip(exps, days) if lo <= dd <= hi]
    if not cand:
        return None
    front = min(cand, key=lambda e: abs((e - tgt).days))
    if front > END:                          # every position must expire inside the development data
        return None
    if bk.und.has_event(d, front):           # split / spin-off / special dividend during the position
        return None
    back = None
    if any(len(l) > 3 and l[3] == "back" for l in struct):
        later = [e for e in exps if 14 <= (e - front).days <= 45]
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


def strike_parity_breaks(bk, d, legs, front, tol=0.05):
    """Contract re-written during the position (corporate action, bad data): on two or more snapshot dates the
    underlying implied by put-call parity at a leg's own strike differs from the close by more than 5%."""
    for kind, cp, e, k, q in legs:
        if kind != "O" or e != front:
            continue
        C, P = bk.path(e, "C", k, d, front, listed_only=True), bk.path(e, "P", k, d, front, listed_only=True)
        j = C[["mid"]].join(P[["mid"]], lsuffix="_c", rsuffix="_p", how="inner")
        if len(j) < 2:
            continue
        Sx = np.array([bk.und.px(t) for t in j.index])
        imp = k + j.mid_c.to_numpy() - j.mid_p.to_numpy()
        with np.errstate(all="ignore"):
            bad = np.abs(np.log(np.where(imp > 0, imp, np.nan) / Sx)) > tol
        if np.nansum(bad) >= 2:
            return True
    return False


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


def mtm_series(bk, d, legs, T, exit_d, S0, pnl, f=PRIMARY_FILL):
    """Daily P&L increments (per unit of notional) from entry to exit that add up to the realised P&L: marked at
    mid on every snapshot, the entry day carries the entry spread and commission, the exit day ends exactly at
    the realised P&L (execution or settlement)."""
    opt = [(i, l) for i, l in enumerate(legs) if l[0] == "O"]
    U = T.loc[:exit_d]
    row0 = T.iloc[0]
    entry_cf = sum(-q * (row0[f"m{i}"] + np.sign(q) * f * row0[f"h{i}"]) for i, (_, cp, e, k, q) in opt)
    comm = COMM * sum(abs(l[4]) for _, l in opt)
    v = pd.Series(entry_cf - comm, index=U.index)
    for i, (_, cp, e, k, q) in opt:
        v = v + q * U[f"m{i}"].ffill()
    for (_, _, _, _, q) in [l for l in legs if l[0] == "S"]:
        adj = bk.und.adj.reindex(U.index).ffill()
        v = v + q * S0 * (adj / adj.iloc[0] - 1 - STK_COST)
    v = v / S0
    v.loc[pd.Timestamp(exit_d)] = pnl                  # realised value on the exit / expiry date
    v = v.sort_index()
    inc = v.diff()
    inc.iloc[0] = v.iloc[0]
    return inc


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


def _sr(x, per_year, min_n=12):
    x = np.asarray(x, float)
    if len(x) < min_n:
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
    ds = pd.DatetimeIndex(bk.dates)
    if dte == WEEKLY_DTE:                    # the last Friday (or Saturday-stamped Friday close) of each week
        fs = ds[ds.weekday >= 4]
        return list(pd.Series(fs, index=fs).groupby(fs.to_period("W-SUN")).max())
    out = []
    for e in third_fridays(ds[0], ds[-1] + pd.Timedelta(days=70)):
        tgt = e - pd.Timedelta(days=dte)
        nxt = ds[ds >= tgt]
        if len(nxt) and (nxt[0] - tgt).days <= 7 and nxt[0] < e:     # 7 days: the chain is weekly before 2021
            out.append(nxt[0])
    return out


def run_book(sym, bk, und, vix, keep_mtm=True, dtes=None, earnings=None):
    """Every structure x entry date x management x fill on one underlying's book (real or synthetic)."""
    dtes = dtes or (DTES + (WEEKLY_DTE,))
    has_weekly = False
    rows, mtm = [], {}
    ivmed = bk.atm_iv.rolling("365D", min_periods=20).median() if len(bk.atm_iv) else bk.atm_iv
    n_mismatch = 0
    n_parity_break = 0
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
                if type(bk) is Book and strike_parity_breaks(bk, d, legs, front):
                    n_parity_break += 1
                    continue
                if dte == WEEKLY_DTE:
                    has_weekly = True
                if earnings is not None and len(earnings):
                    flt["NOERN"] = not bool(((earnings > d) & (earnings <= front)).any())
                else:
                    flt["NOERN"] = None
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
                                  "expiry": front, "und_ret": und.tr(d, r["exit"]), "iv_entry": iv,
                                  "rv21_entry": und.rv21(d),
                                  **{f"f_{k}": v for k, v in flt.items()},
                                  "legs": ";".join(f"{l[1]}{l[3]:g}x{l[4]}" if l[0] == "O" else f"Sx{l[4]}"
                                                   for l in legs)})
                        rows.append(r)
                        if keep_mtm and f == PRIMARY_FILL:
                            mtm[(sym, name, dte, mg, str(pd.Timestamp(d).date()))] = mtm_series(
                                bk, d, legs, T, r["exit"], S, r["pnl"])
    R = pd.DataFrame(rows)
    meta = {"weekly_expiries": has_weekly, "chain_dates": int(len(bk.dates)), "entries_skipped_price_mismatch": n_mismatch,
            "positions_skipped_parity_break": n_parity_break,
            "path_points_listed": int(getattr(bk, "n_listed", 0)), "path_points_valued_from_smile": int(getattr(bk, "n_modeled", 0)),
            "path_points_unvalued": int(getattr(bk, "n_gap", 0)), "valuations_across_expiries": int(getattr(bk, "n_term", 0)), "corporate_action_days": [str(x.date()) for x in und.events],
            "first": str(pd.Timestamp(bk.dates[0]).date()), "last": str(pd.Timestamp(bk.dates[-1]).date())}
    return R, meta, mtm


def trim_chain(C, und, max_dte=100, lo=0.6, hi=1.4):
    """Keep what any structure can use: expiries within 100 days, strikes within 60-140% of the underlying."""
    dts = pd.DatetimeIndex(sorted(C.date.unique()))
    S = pd.Series([und.px(d) for d in dts], index=dts, name="S")
    C = C.merge(S, left_on="date", right_index=True, how="inner")
    keep = ((C.expiration - C.date).dt.days <= max_dte) & (C.strike >= lo * C.S) & (C.strike <= hi * C.S)
    return C[keep].drop(columns="S")


def run_symbol(sym, chain_dir, und, vix, keep_mtm=True, earnings=None):
    C = load_chain(chain_dir, sym)
    if C.empty:
        return pd.DataFrame(), {}, {}
    n0 = len(C)
    C = trim_chain(C, und)
    bk = Book(C, und)
    R, meta, mtm = run_book(sym, bk, und, vix, keep_mtm, earnings=earnings)
    meta.update({"rows_loaded": int(n0), "rows_used": int(len(C)), "smile_valuation_check": bk.model_check(),
                 "across_expiry_valuation_check": bk.term_check()})
    return R, meta, mtm


def flag(df, flt):
    c = f"f_{flt}"
    if c not in df.columns:
        return pd.Series(False, index=df.index)
    return df[c].map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) or (isinstance(v, (int, float)) and
                                                                            np.isfinite(v)) else
                     (str(v).lower() == "true")).astype(bool)


def table(R):
    """One row per combination (symbol, struct, dte, mgmt, filter, fill)."""
    out = []
    for (sym, s, dte, mg, f), g in R.groupby(["symbol", "struct", "dte", "mgmt", "fill"]):
        per_year = 52.0 if dte == WEEKLY_DTE else 12.0
        for flt in FILTERS:
            if flt == "NOERN" and (f"f_{flt}" not in g.columns or g[f"f_{flt}"].isna().all()):
                continue
            h = g[flag(g, flt)]
            if len(h) < 8:
                continue
            out.append({"symbol": sym, "struct": s, "dte": dte, "mgmt": mg, "filter": flt, "fill": f,
                        "defined_risk": s in DEFINED_RISK, "short_premium": s in SHORT_PREMIUM,
                        **summarise(h.sort_values("entry"), per_year)})
    return pd.DataFrame(out)


def walk_forward(R, T, family, min_cycles=24, criterion="sharpe"):
    """Each year: pick the best combination on cycles that ended before the year, trade it in that year.
    criterion "sharpe" = long-run Sharpe; "t" = Newey-West t of the mean (favours longer, steadier records)."""
    P = R[R.fill == PRIMARY_FILL].copy()
    P["key"] = P.symbol + "|" + P.struct + "|" + P.dte.astype(str) + "|" + P.mgmt
    years = sorted(P.entry.dt.year.unique())
    picks, oos = [], []
    for y in years[2:]:
        hist = P[P.exit < pd.Timestamp(f"{y}-01-01")]
        best, best_sr, best_flt = None, -np.inf, None
        for flt in FILTERS:
            H = hist[flag(hist, flt)]
            for key, g in H.groupby("key"):
                if not family(g.iloc[0]) or len(g) < min_cycles:
                    continue
                per_year = 52.0 if g.dte.iloc[0] == WEEKLY_DTE else 12.0
                x = g.sort_values("entry").pnl
                sr = _sr(x, per_year) if criterion == "sharpe" else nw_t(x.to_numpy())
                if np.isfinite(sr) and sr > best_sr:
                    best, best_sr, best_flt = key, sr, flt
        if best is None:
            continue
        nxt = P[(P.key == best) & (P.entry.dt.year == y) & flag(P, best_flt)]
        picks.append({"year": int(y), "pick": best, "filter": best_flt, "in_sample_sharpe": float(best_sr),
                      "oos_cycles": int(len(nxt)), "oos_sum": float(nxt.pnl.sum()), "oos_mean": float(nxt.pnl.mean())
                      if len(nxt) else np.nan})
        oos.append(nxt)
    O = pd.concat(oos) if oos else pd.DataFrame()
    return pd.DataFrame(picks), O


INDEX_ETFS = {"SPY", "DIA", "QQQ", "IWM"}
AGG_KEYS = ["struct", "dte", "mgmt", "fill", "entry"]


def _agg(h, sym):
    g = h.groupby(AGG_KEYS).agg(pnl=("pnl", "mean"), und_ret=("und_ret", "mean"), credit=("credit", "mean"),
                                days=("days", "mean"), early=("early", "mean"), n_names=("pnl", "size"),
                                exit=("exit", "max"), expiry=("expiry", "max")).reset_index()
    g["symbol"] = sym
    return g[g.n_names >= 5]


def portfolios(R):
    """Pseudo-underlyings built from the single-stock cycles (one position per name, equal weight per entry):
      EW     every stock that passes the filter;
      XSQ5   the fifth of stocks with the largest implied-minus-realised volatility at entry (Goyal & Saretto 2009);
      XSQ1   the fifth with the smallest;
      GS_LS  struct GSLS: short ATM straddles on XSQ5 and long ATM straddles on XSQ1 (their long-short)."""
    S = R[~R.symbol.isin(INDEX_ETFS)]
    if S.symbol.nunique() < 10:
        return pd.DataFrame()
    out = []
    for flt in FILTERS:
        g = _agg(S[flag(S, flt)], "EW")
        for F in FILTERS:
            g[f"f_{F}"] = F == flt
        out.append(g)
    S2 = S.assign(spread=S.iv_entry - S.rv21_entry).dropna(subset=["spread"])
    S2["q"] = S2.groupby(AGG_KEYS).spread.rank(pct=True)
    for sym, sel in (("XSQ5", S2.q > 0.8), ("XSQ1", S2.q <= 0.2)):
        g = _agg(S2[sel], sym)
        for F in FILTERS:
            g[f"f_{F}"] = F == "ALL"
        out.append(g)
    top = _agg(S2[(S2.q > 0.8) & (S2.struct == "STRD")], "GS_LS").set_index(["dte", "mgmt", "fill", "entry"])
    bot = _agg(S2[(S2.q <= 0.2) & (S2.struct == "LSTRD")], "GS_LS").set_index(["dte", "mgmt", "fill", "entry"])
    j = top.join(bot[["pnl", "und_ret", "credit"]], rsuffix="_b", how="inner")
    if len(j):
        j["pnl"] = j.pnl + j.pnl_b
        j["und_ret"] = j.und_ret - j.und_ret_b
        j["credit"] = j.credit + j.credit_b
        j = j.drop(columns=["pnl_b", "und_ret_b", "credit_b"]).reset_index()
        j["struct"] = "GSLS"
        for F in FILTERS:
            j[f"f_{F}"] = F == "ALL"
        out.append(j)
    P = pd.concat(out, ignore_index=True)
    P["early"] = P.early > 0.5
    return P


def stage_cycles(a, X, vix):
    syms = [s for s in a.symbols.split(",") if s] or sorted(
        f[len("chain_"):-len(".parquet")] for f in os.listdir(a.chain) if f.startswith("chain_") and f.endswith(".parquet"))
    if a.group:
        i, n = map(int, a.group.split("/"))
        syms = syms[i::n]
    meta = {}
    ern = None
    ef = os.path.join(a.chain, "earnings.csv")
    if os.path.exists(ef):
        E = pd.read_csv(ef)
        E.columns = [c.lower() for c in E.columns]
        E["date"] = pd.to_datetime(E["date"], errors="coerce")
        ern = {k: pd.DatetimeIndex(sorted(v.dropna())) for k, v in E.groupby("act_symbol").date}
    for sym in syms:
        if f"{sym}_raw" not in X.columns or not X[f"{sym}_raw"].notna().any():
            meta[sym] = {"error": "no closes"}
            continue
        try:
            und = Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
            R, m, mtm = run_symbol(sym, a.chain, und, vix, keep_mtm=True,
                                   earnings=(ern or {}).get(sym) if sym not in INDEX_ETFS else None)
        except Exception as ex:                                         # noqa: BLE001
            meta[sym] = {"error": repr(ex)[:300]}
            continue
        meta[sym] = m
        print(sym, json.dumps(m, default=str), len(R), flush=True)
        if len(R):
            R.to_csv(os.path.join(a.out, f"cycles_{sym}.csv.gz"), index=False)
            comp = {k: (v.index.values.astype("datetime64[D]").astype(np.int32), v.to_numpy(np.float32))
                    for k, v in mtm.items()}
            pd.to_pickle(comp, os.path.join(a.out, f"mtm_{sym}.pkl"))
            del comp, mtm
    with open(os.path.join(a.out, f"meta_{(a.group or 'all').replace('/', 'of')}.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)


def _member_mtm(R, MTM, r, year=None):
    """Daily mark-to-market of a combination (optionally only positions entered in one year)."""
    if r.symbol in ("EW", "XSQ5", "XSQ1", "GS_LS"):
        return None
    g = R[(R.symbol == r.symbol) & (R.struct == r.struct) & (R.dte == r.dte) & (R.mgmt == r.mgmt) &
          (R.fill == PRIMARY_FILL) & flag(R, r["filter"])]
    if year is not None:
        g = g[g.entry.dt.year == year]
    if isinstance(MTM.get(r.symbol), str):
        MTM[r.symbol] = pd.read_pickle(MTM[r.symbol])
    if r.symbol not in MTM:
        return None
    M = MTM[r.symbol]
    parts = [pd.Series(M[k][1], index=pd.to_datetime(M[k][0].astype("datetime64[D]")))
             for k in ((r.symbol, r.struct, int(r.dte), r.mgmt, str(pd.Timestamp(d).date())) for d in g.entry) if k in M]
    return pd.concat(parts, axis=1).sum(axis=1).sort_index() if parts else None


def stage_combine(a):
    dirs = [d for d in a.inputs.split(",") if d]
    Rs, MTM, meta = [], {}, {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if f.startswith("cycles_") and f.endswith(".csv.gz"):
                Rs.append(pd.read_csv(p, parse_dates=["entry", "exit", "expiry"]))
            elif f.startswith("mtm_") and f.endswith(".pkl"):
                MTM[f[4:-4]] = p                                 # loaded lazily, only for the best combinations
            elif f.startswith("meta_") and f.endswith(".json"):
                meta.update(json.load(open(p)))
    R = pd.concat(Rs, ignore_index=True)
    R = R[R.entry <= END]
    for c in [c for c in R.columns if c.startswith("f_") and c != "f_NOERN"] + ["early"]:
        R[c] = R[c].astype(bool)
    P = portfolios(R)
    RA = pd.concat([R, P], ignore_index=True) if len(P) else R
    RA.to_csv(os.path.join(a.out, "cycles_all.csv.gz"), index=False)
    T = table(RA)
    prim = T.fill == PRIMARY_FILL
    T.loc[prim, "bh_q_mean"] = bh(T.loc[prim, "p_one_sided"])
    T.loc[prim, "bh_q_alpha"] = bh(T.loc[prim, "p_alpha"])
    n_trials = int(prim.sum())
    best = T[prim & (T.cycles >= 24)].sort_values("sharpe", ascending=False)
    if len(best):
        b = best.iloc[0]
        per = 52.0 if b.dte == WEEKLY_DTE else 12.0
        sr_var = float((T.loc[prim, "sharpe"] / np.sqrt(np.where(T.loc[prim, "dte"] == WEEKLY_DTE, 52, 12))).var())
        T.loc[best.index[0], "deflated_sharpe_prob"] = deflated_sharpe(b.sharpe / math.sqrt(per), n_trials,
                                                                       int(b["cycles"]), b["skew"], b["kurt"], sr_var)
    T.to_csv(os.path.join(a.out, "combos.csv"), index=False)
    fams = {"all": lambda r: True,
            "defined_risk": lambda r: r.struct in DEFINED_RISK,
            "short_premium_defined": lambda r: r.struct in DEFINED_RISK and r.struct in SHORT_PREMIUM,
            "index_etfs": lambda r: r.symbol in INDEX_ETFS,
            "single_stocks": lambda r: r.symbol not in INDEX_ETFS and r.symbol not in ("EW", "XSQ5", "XSQ1", "GS_LS"),
            "stock_portfolios": lambda r: r.symbol in ("EW", "XSQ5", "XSQ1", "GS_LS")}
    wf = {}
    for (name, fam), crit in itertools.product(fams.items(), ("sharpe", "t")):
        picks, O = walk_forward(RA, T, fam, criterion=crit)
        picks.to_csv(os.path.join(a.out, f"walkforward_{name}_{crit}.csv"), index=False)
        if len(O):
            m = O.groupby(O.entry.dt.to_period("M")).pnl.sum()
            wf[f"{name}|{crit}"] = {"oos_cycles": int(len(O)), "oos_months": int(len(m)), "oos_mean_cycle": float(O.pnl.mean()),
                        "oos_mean_month": float(m.mean()), "oos_t_nw_month": nw_t(m.to_numpy()),
                        "oos_sharpe_month": _sr(m.to_numpy(), 12.0), "oos_worst_cycle": float(O.pnl.min()),
                        "oos_win": float((O.pnl > 0).mean()), "years": picks.year.tolist(),
                        "picks": picks.pick.tolist()}
    Z = T[prim & (T["filter"] == "ALL")].dropna(subset=["sharpe_is", "sharpe_oos"])
    rank_stab = {"combos": int(len(Z))}
    if len(Z) > 10:
        top_is = Z.sort_values("sharpe_is", ascending=False).head(max(1, len(Z) // 10))
        rank_stab.update({"spearman_is_oos": float(st.spearmanr(Z.sharpe_is, Z.sharpe_oos)[0]),
                          "top_decile_is_mean_oos_sharpe": float(top_is.sharpe_oos.mean()),
                          "all_mean_oos_sharpe": float(Z.sharpe_oos.mean()),
                          "top_decile_is_share_oos_positive": float((top_is.mean_oos > 0).mean())})
    # daily mark-to-market (overlapping positions summed), for the provider simulation:
    #   WF|family|criterion  the walk-forward strategy itself (each year the combination picked on earlier years);
    #   the 20 combinations with the highest t over the full sample among those with 48+ cycles (in sample).
    daily, dd_rows = {}, []
    for key_wf in wf:
        name, crit = key_wf.split("|")
        P = pd.read_csv(os.path.join(a.out, f"walkforward_{name}_{crit}.csv"))
        parts = []
        for _, pk in P.iterrows():
            sym, st_, dte, mg = pk.pick.split("|")
            ser = _member_mtm(R, MTM, pd.Series({"symbol": sym, "struct": st_, "dte": int(dte), "mgmt": mg,
                                                 "filter": pk["filter"]}), year=int(pk.year))
            if ser is not None:
                parts.append(ser)
        if parts:
            daily[f"WF|{name}|{crit}"] = pd.concat(parts).groupby(level=0).sum().sort_index()
    robust = T[prim & (T.cycles >= 48)].sort_values("t_nw", ascending=False)
    for _, r in robust.iterrows():
        if sum(not k.startswith("WF|") for k in daily) >= 20:
            break
        sser = _member_mtm(R, MTM, r)
        if sser is not None:
            daily[f"{r.symbol}|{r.struct}|{r.dte}|{r.mgmt}|{r['filter']}"] = sser
    for key, sser in daily.items():
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
    print(json.dumps({k: v for k, v in summ.items() if k not in ("best_primary", "meta")}, indent=1, default=str))
    pd.set_option("display.width", 250)
    cols = ["symbol", "struct", "dte", "mgmt", "filter", "cycles", "mean", "t_nw", "sharpe", "win_rate", "worst",
            "alpha", "t_alpha", "bh_q_mean", "bh_q_alpha", "sharpe_is", "sharpe_oos"]
    print(best[cols].head(40).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("cycles", "combine", "both"), default="both")
    ap.add_argument("--chain", default="")
    ap.add_argument("--closes", default="", help="CSV of daily closes: date, <SYM>_raw, <SYM>_adj, VIX, VIX3M")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--group", default="", help="i/n: take every n-th symbol starting at i (parallel jobs)")
    ap.add_argument("--inputs", default="", help="combine: comma-separated directories of cycles_*.csv.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage in ("cycles", "both"):
        X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
        X = X[X.index <= END]
        vix = X[["VIX", "VIX3M"]].dropna()
        stage_cycles(a, X, vix)
    if a.stage in ("combine", "both"):
        if not a.inputs:
            a.inputs = a.out
        stage_combine(a)


if __name__ == "__main__":
    main()
