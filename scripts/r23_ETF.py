"""Round 23 part ETF (docs/round23_preregistration.md): sector, country and cross-asset ETF rotation, Sobol 256 per
universe, default and nested walk-forward. Counted: the book's monthly return minus the equal-weight return of the
same universe (members available that month), so market beta is not counted as an edge. yfinance adjusted closes.
Usage: python scripts/r23_ETF.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402
import wf23  # noqa: E402

UNIV = {"sectors": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
        "countries": ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWM", "EWN", "EWO", "EWP", "EWQ",
                      "EWS", "EWU", "EWW"],
        "cross_asset": ["SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "AGG", "GLD", "DBC", "VNQ"]}
COST = 5e-4


def monthly(symbols):
    import yfinance as yf
    out = {}
    for s in symbols:
        h = None
        for _ in range(4):
            try:
                h = yf.download(s, start="1995-01-01", end="2026-01-01", progress=False, auto_adjust=True)
                if h is not None and len(h):
                    break
            except Exception:                                              # noqa: BLE001
                time.sleep(5)
        if h is None or h.empty:
            print("no data", s, flush=True)
            continue
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        c = h["Close"]
        c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.to_datetime(c.index)
        out[s] = c
    D = pd.DataFrame(out).sort_index()
    D = D[D.index <= wf22.DEV_END]
    return D.resample("ME").last(), D


def run(Pm, Pd, tbill):
    R = Pm.shift(-1) / Pm - 1                                              # return over the next month
    vol = np.log(Pd).diff().rolling(63).std().resample("ME").last().reindex(Pm.index)
    ew = R.mean(axis=1, skipna=True)
    U = qmc.Sobol(8, scramble=True, seed=2302).random(256)
    M, desc = [], []
    k_max = max(1, Pm.shape[1] // 2)
    for u in U:
        lb, sk = int(1 + u[0] * 11.999), int(u[1] * 1.999)
        risk_adj = u[2] < 0.5
        n = int(1 + u[3] * 4.999)
        n = min(n, k_max)
        absf = ["none", "zero", "tbill"][min(2, int(u[4] * 3))]
        invvol = u[5] < 0.5
        reb = int(1 + u[6] * 2.999)
        mom = Pm.shift(sk) / Pm.shift(sk + lb) - 1
        Z = mom / vol if risk_adj else mom
        out, w_prev, w = [], pd.Series(0.0, index=Pm.columns), pd.Series(0.0, index=Pm.columns)
        for i, m in enumerate(Pm.index[:-1]):
            if i % reb == 0:
                z = Z.loc[m].dropna()
                z = z[R.loc[m].reindex(z.index).notna()]
                w = pd.Series(0.0, index=Pm.columns)
                if len(z) >= 2 * n:
                    top = z.sort_values().index[-n:]
                    if absf == "zero":
                        top = [s for s in top if mom.loc[m, s] > 0]
                    elif absf == "tbill":
                        top = [s for s in top if mom.loc[m, s] > lb / 12.0 * float(tbill.get(m, 0.0))]
                    if len(top):
                        iv = (1.0 / vol.loc[m, top]).replace([np.inf], np.nan).fillna(0.0) if invvol else pd.Series(1.0, index=top)
                        if iv.sum() > 0:
                            w[top] = iv / iv.sum()
            turn = float((w - w_prev).abs().sum())
            w_prev = w
            rf = float(tbill.get(m, 0.0)) / 12.0 * (1.0 - float(w.sum()))   # cash earns the T-bill
            raw = float((w * R.loc[m].fillna(0.0)).sum()) + rf - COST * turn
            out.append((raw, raw - float(ew.loc[m]) if np.isfinite(ew.loc[m]) else np.nan))
        M.append(pd.DataFrame(out, index=Pm.index[:-1], columns=["raw", "excess"]))
        desc.append({"lb": lb, "skip": sk, "risk_adj": risk_adj, "n": n, "abs": absf, "invvol": invvol, "reb": reb})
    return M, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import yfinance as yf
    irx = yf.download("^IRX", start="1995-01-01", end="2026-01-01", progress=False, auto_adjust=False)
    if isinstance(irx.columns, pd.MultiIndex):
        irx.columns = irx.columns.get_level_values(0)
    tb = (irx["Close"] / 100.0).resample("ME").last()
    tb.index = pd.to_datetime(tb.index).tz_localize(None) if getattr(tb.index, "tz", None) else tb.index
    rows, ser, meta = [], {}, {}
    for uni, syms in UNIV.items():
        Pm, Pd = monthly(syms)
        tbill = tb.reindex(Pm.index).ffill()
        M, desc = run(Pm, Pd, tbill)
        meta[uni] = {s: str(Pm[s].first_valid_index().date()) for s in Pm.columns}
        for col in ("excess", "raw"):
            W = pd.concat([m[col] for m in M], axis=1)
            W.index = W.index.normalize()
            ag = wf23.Agg(W.index, P=W.to_numpy().T)
            res = wf23.both(ag, 2003, min_active=12, min_valid=12)
            for kind, (s, st, own, ch) in res.items():
                rows.append({"part": "ETF", "family": f"ETF-{uni}-{col}-{kind}", "symbol": uni, "placebo": False,
                             "counted": col == "excess", "selection": kind, "n_cfg": len(M), **st, "own": own})
                ser[f"ETF_{uni}_{col}_{kind}"] = s
                print(uni, col, kind, json.dumps({k: rows[-1][k] for k in ("oos_first", "p1_mean", "p2_mean", "p2_t", "own")},
                                                 default=str), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "ETF_rows.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, "ETF_series.parquet"))
    json.dump(meta, open(os.path.join(a.out, "ETF_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
