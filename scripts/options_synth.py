"""Synthetic option chains for the long history (a MODEL, not market quotes), so the structures chosen on real
quotes can be checked over 1990-2018 (1998 / 2008 / 2011 / 2015 / 2018 crashes) where no free chain data exists.

Model, per underlying with a Cboe volatility index:
  implied vol of strike K, expiry e, on date d:  IV = VolIndex(d) / 100 x g(x),  x = ln(K / S) / (s30 sqrt(T)),
  s30 = VolIndex / 100, T = (e - d) / 365; g is a piecewise-linear "smile ratio" calibrated on the REAL chains of
  the matching ETF (median of IV / VolIndex by x bucket, 15-70 days to expiry, 2019-2025);
  half-spread = max(tick / 2, h(|delta|) x price), h the median half-spread / mid by delta bucket in the real chains;
  prices Black-Scholes with zero rates and dividends; strikes on a grid of 0.5% of the price when each expiry is
  listed (70 days before it), so a strike chosen at entry exists until expiry.
Validation: the same structures priced by the model on 2019-2025 are compared with the real-quote results
(correlation of cycle P&L, mean difference), so the reader can judge how far to trust the model years."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from options_strategy_lab import Book, third_fridays  # noqa: E402

XB = np.arange(-4.0, 3.01, 0.25)
DB = np.array([0.0, 0.03, 0.07, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.01])


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bs(cp, S, K, T, v):
    if T <= 0 or v <= 0:
        itm = S > K if cp == "C" else K > S
        return (max(S - K, 0) if cp == "C" else max(K - S, 0)), (1.0 if cp == "C" else -1.0) * float(itm)
    sd = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if cp == "C":
        return S * _ncdf(d1) - K * _ncdf(d2), _ncdf(d1)
    return K * _ncdf(-d2) - S * _ncdf(-d1), _ncdf(d1) - 1


def calibrate(C, und, volidx):
    """Smile ratio g(x) and half-spread ratio h(|delta|) from a real chain panel."""
    rows = []
    for d, g in C.groupby("date"):
        if d not in volidx.index:
            continue
        S, vi = und.px(d), float(volidx.loc[d]) / 100
        g = g[(g.expiration - d).dt.days.between(15, 70) & g.iv.notna() & (g.iv > 0) & (g.bid > 0)]
        if not len(g) or not np.isfinite(S) or vi <= 0:
            continue
        T = (g.expiration - d).dt.days / 365
        x = np.log(g.strike / S) / (vi * np.sqrt(T))
        # out-of-the-money side only (the liquid quotes)
        otm = ((g.cp == "P") & (g.strike <= S)) | ((g.cp == "C") & (g.strike >= S))
        dl = g["delta"].abs() if "delta" in g.columns else pd.Series(np.nan, index=g.index)
        rows.append(pd.DataFrame({"x": x[otm], "ratio": (g.iv / vi)[otm], "dabs": dl[otm],
                                  "hm": ((g.ask - g.bid) / 2 / g[["bid", "ask"]].mean(axis=1))[otm]}))
    if not rows:
        return None
    Z = pd.concat(rows)
    Z = Z[np.isfinite(Z.x) & np.isfinite(Z.ratio) & Z.ratio.between(0.2, 5)]
    gx = Z.groupby(pd.cut(Z.x, XB)).ratio.median()
    hx = Z[Z.dabs.notna()].groupby(pd.cut(Z.dabs, DB)).hm.median()
    g_mid = np.array([iv.mid for iv in gx.index])
    h_mid = np.array([iv.mid for iv in hx.index])
    return {"g_x": g_mid[gx.notna().to_numpy()], "g": gx.dropna().to_numpy(),
            "h_d": h_mid[hx.notna().to_numpy()], "h": hx.dropna().to_numpy(), "n": int(len(Z))}


class SynthBook(Book):
    """Same interface as Book, quotes from the model."""

    def __init__(self, und, volidx, cal, start, end, tick=0.01):
        self.und = und
        self.vol = (volidx / 100).dropna()
        idx = und.raw.index.intersection(self.vol.index)
        idx = idx[(idx >= start) & (idx <= end)]
        self.dates = np.array(idx.values)
        self.cal = cal
        self.tick = tick
        self.exp_grid = {}
        self.atm_iv = self.vol.reindex(idx) * float(np.interp(0.0, cal["g_x"], cal["g"]))
        self._cache = {}

    def iv(self, d, K, e):
        S, s30 = self.und.px(d), float(self.vol.loc[:d].iloc[-1])
        T = max((e - d).days, 1) / 365
        x = math.log(K / S) / (s30 * math.sqrt(T))
        return s30 * float(np.interp(x, self.cal["g_x"], self.cal["g"]))

    def expiries(self, d):
        d = pd.Timestamp(d)
        ex = [e for e in third_fridays(d - pd.Timedelta(days=31), d + pd.Timedelta(days=100)) if 0 < (e - d).days <= 100]
        fr = [d + pd.Timedelta(days=k) for k in range(1, 10) if (d + pd.Timedelta(days=k)).weekday() == 4]
        return np.array(sorted(set(ex) | set(fr[:2])))

    def _grid(self, e):
        if e not in self.exp_grid:
            lst = e - pd.Timedelta(days=100)
            S = self.und.px(lst) if len(self.und.raw.loc[:lst]) else float(self.und.raw.iloc[0])
            step = 0.005 * S
            self.exp_grid[e] = (S, step)
        return self.exp_grid[e]

    def _snap(self, e, K):
        S0, step = self._grid(e)
        return round(S0 + round((K - S0) / step) * step, 6)

    def quote(self, exp, cp, k, d):
        key = (exp, cp, k, d)
        if key in self._cache:
            return self._cache[key]
        S = self.und.px(d)
        T = (exp - d).days / 365
        v = self.iv(d, k, exp)
        p, dl = bs(cp, S, k, T, v)
        h = max(self.tick / 2, float(np.interp(abs(dl), self.cal["h_d"], self.cal["h"])) * p)
        r = (p, h)
        self._cache[key] = r
        return r

    def pick(self, d, exp, cp, target, S, atm_strike=None):
        d = pd.Timestamp(d)
        if target == "atm":
            return float(atm_strike) if atm_strike is not None else self._snap(exp, S)
        T = max((exp - d).days, 1) / 365
        S0, step = self._grid(exp)
        base = self._snap(exp, S)
        K = base + np.arange(-200, 201) * step
        K = K[K > 0]
        s30 = float(self.vol.loc[:d].iloc[-1])
        v = s30 * np.interp(np.log(K / S) / (s30 * math.sqrt(T)), self.cal["g_x"], self.cal["g"])
        sd = v * math.sqrt(T)
        d1 = (np.log(S / K) + 0.5 * sd * sd) / sd
        dl = ndtr(d1) if cp == "C" else ndtr(d1) - 1
        prem = (S * ndtr(d1) - K * ndtr(d1 - sd)) if cp == "C" else (K * ndtr(sd - d1) - S * ndtr(-d1))
        dist = np.where(prem >= self.tick, np.abs(np.abs(dl) - target), 9.0)
        j = int(np.argmin(dist))
        return float(round(K[j], 6)) if dist[j] <= 0.08 else None

    def path(self, exp, cp, k, d0, d1):
        days = self.dates[(self.dates >= np.datetime64(d0)) & (self.dates <= np.datetime64(d1))]
        rows = [self.quote(exp, cp, k, pd.Timestamp(dd)) for dd in days if pd.Timestamp(dd) < exp]
        idx = [pd.Timestamp(dd) for dd in days if pd.Timestamp(dd) < exp]
        return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["mid", "half"])


PAIRS = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX", "DIA": "VXD", "GLD": "GVZ", "USO": "OVX", "EEM": "VXEEM"}


def main():
    import argparse
    import json
    import os

    import options_strategy_lab as L
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="real chains, for the calibration")
    ap.add_argument("--closes", required=True)
    ap.add_argument("--real", default="", help="real-quote cycles.csv.gz, for the validation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = pd.read_csv(a.closes, parse_dates=["date"]).set_index("date").sort_index()
    X = X[X.index <= L.END]
    vix = X[["VIX", "VIX3M"]].dropna()
    allR, meta = [], {}
    for sym, vi in PAIRS.items():
        if f"{sym}_raw" not in X.columns or vi not in X.columns:
            meta[sym] = {"error": "no closes or no volatility index"}
            continue
        und = L.Underlying(X[f"{sym}_raw"], X[f"{sym}_adj"])
        try:
            C = L.load_chain(a.chain, sym)
        except Exception as ex:                                    # noqa: BLE001
            meta[sym] = {"error": f"no real chain to calibrate: {ex}"[:200]}
            continue
        if "iv" not in C.columns or C["iv"].notna().mean() < 0.5:
            meta[sym] = {"error": "real chain has no implied volatility"}
            continue
        cal = calibrate(C[C.date >= C.date.max() - pd.Timedelta(days=7 * 366)], und, X[vi].dropna())
        if cal is None:
            meta[sym] = {"error": "calibration failed"}
            continue
        vol = X[vi].dropna()
        start = max(vol.index[0], und.raw.index[0]) + pd.Timedelta(days=400)
        bk = SynthBook(und, vol, cal, start, L.END)
        R, m, _ = L.run_book(sym, bk, und, vix, keep_mtm=False, dtes=L.DTES, structs=L.BASE_STRUCTS,
                             mgmts=L.BASE_MGMT, wheel=False)
        meta[sym] = {**m, "vol_index": vi, "calibration_points": cal["n"],
                     "smile_ratio_at": {f"{x:+.2f}": float(g) for x, g in zip(cal["g_x"], cal["g"])},
                     "half_spread_over_mid_at_abs_delta": {f"{x:.3f}": float(h) for x, h in zip(cal["h_d"], cal["h"])}}
        print(sym, json.dumps(meta[sym])[:600], len(R), flush=True)
        if len(R):
            allR.append(R)
    R = pd.concat(allR, ignore_index=True)
    R.to_csv(os.path.join(a.out, "cycles_synth.csv.gz"), index=False)
    T = L.table(R)
    prim = T.fill == L.PRIMARY_FILL
    T.to_csv(os.path.join(a.out, "combos_synth.csv"), index=False)
    # model years only (before the real quotes start) and the crisis years
    pre = R[R.entry < pd.Timestamp("2019-01-01")]
    Tpre = L.table(pre) if len(pre) else pd.DataFrame()
    if len(Tpre):
        Tpre.to_csv(os.path.join(a.out, "combos_synth_pre2019.csv"), index=False)
    val = {}
    if a.real and os.path.exists(a.real):
        Rr = pd.read_csv(a.real, parse_dates=["entry", "exit", "expiry"])
        k = ["symbol", "struct", "dte", "mgmt", "fill", "entry"]
        M = Rr[Rr.fill == L.PRIMARY_FILL].merge(R[R.fill == L.PRIMARY_FILL], on=k, suffixes=("_real", "_model"))
        if len(M):
            g = M.groupby(["symbol", "struct", "dte", "mgmt"])
            V = pd.DataFrame({"n": g.size(), "corr": g.apply(lambda h: h.pnl_real.corr(h.pnl_model)),
                              "mean_real": g.pnl_real.mean(), "mean_model": g.pnl_model.mean()}).reset_index()
            V.to_csv(os.path.join(a.out, "validation_model_vs_real.csv"), index=False)
            val = {"pairs": int(len(M)), "median_corr": float(V["corr"].median()),
                   "mean_real_minus_model": float((V.mean_real - V.mean_model).mean()),
                   "share_same_sign_mean": float((np.sign(V.mean_real) == np.sign(V.mean_model)).mean())}
    with open(os.path.join(a.out, "summary_synth.json"), "w") as fh:
        json.dump({"meta": meta, "validation": val, "combos": int(prim.sum())}, fh, indent=1, default=str)
    print(json.dumps(val, indent=1))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    main()
