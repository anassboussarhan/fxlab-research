"""Retail 'no-prediction' systems that claim to profit from trade management alone:

  * HEDGED (bi-directional) GRID: buy-stops every g above an anchor and sell-stops every g below it; every
    position takes profit one step later and has no individual stop. Longs and shorts are held at the same time
    (locked hedges). The whole basket closes at a basket take-profit (+B steps of floating profit) or a
    stop-out (floating loss <= -S steps, or more than N open positions per side), then re-anchors at price.
  * ZONE RECOVERY (hedging martingale): random-time entry in a random direction; if price moves one zone width
    against the position, open the opposite side with a larger size so that a move of one take-profit width
    in either direction closes the whole basket in profit; repeat up to K times, then stop out.

Both are simulated on bar closes with execution at the next open (the platform's timing contract). Gross P&L
is net exposure x price moves (exact for equal-size positions); COSTS ARE CHARGED ON GROSS TURNOVER - every
leg of a locked hedge pays spread, slippage and commission, and overnight financing is charged on BOTH legs
(gross open units), as retail brokers do. No parameter selection: every configuration is reported (null test).
Instruments: every local instrument, H1 / H4 / D1, 2007-2018 development data (EURUSD to 2022).
Output: results/reports/no_prediction/{runs.csv, summary.csv}"""
from __future__ import annotations

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY",
           "XAUUSD", "XAGUSD", "EURCHF", "AUDJPY"]
TFS = ["H1", "H4", "D1"]
GRID_CFGS = [dict(step=s, basket=b, stop=S, N=10) for s in (0.5, 1.0, 2.0) for b, S in ((1.0, 10.0), (3.0, 20.0))]
ZONE_CFGS = [dict(zone=z, tp=t, K=K) for z, t in ((1.0, 2.0), (2.0, 2.0)) for K in (4, 7)]
SEEDS = int(os.environ.get("NP_SEEDS", 8))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reports", "no_prediction")


@njit(cache=False)
def _hedged_grid(c, a, step_k, basket_k, stop_k, N):
    """Returns (net exposure, gross open units, gross units traded) per decision bar, unit = 1 position."""
    n = len(c)
    net = np.zeros(n); gross_open = np.zeros(n); traded = np.zeros(n)
    buys = np.zeros(4 * N + 4); sells = np.zeros(4 * N + 4)      # entry levels of open positions
    nb = 0; ns = 0
    anchor = np.nan; g = 0.0; up = 0; dn = 0                        # next level index above / below
    for t in range(n):
        tr = 0.0
        if anchor != anchor:
            if a[t] == a[t] and a[t] > 0:
                anchor = c[t]; g = step_k * a[t]; up = 1; dn = 1
            net[t] = 0.0; gross_open[t] = 0.0; traded[t] = 0.0
            continue
        # take-profits of individual positions (one step in favour)
        i = 0
        while i < nb:
            if c[t] >= buys[i] + g:
                buys[i] = buys[nb - 1]; nb -= 1; tr += 1
            else:
                i += 1
        i = 0
        while i < ns:
            if c[t] <= sells[i] - g:
                sells[i] = sells[ns - 1]; ns -= 1; tr += 1
            else:
                i += 1
        # new stop orders triggered (levels crossed at this close)
        while c[t] >= anchor + up * g and nb < 4 * N:
            buys[nb] = anchor + up * g; nb += 1; up += 1; tr += 1
        while c[t] <= anchor - dn * g and ns < 4 * N:
            sells[ns] = anchor - dn * g; ns += 1; dn += 1; tr += 1
        # re-arm levels that were taken profit (price can cross them again)
        if c[t] < anchor + (up - 1) * g:
            up = max(1, int(np.floor((c[t] - anchor) / g)) + 1)
        if c[t] > anchor - (dn - 1) * g:
            dn = max(1, int(np.floor((anchor - c[t]) / g)) + 1)
        fl = 0.0
        for i in range(nb):
            fl += c[t] - buys[i]
        for i in range(ns):
            fl += sells[i] - c[t]
        fl /= g                                                    # floating P&L in steps (1 unit each)
        if (nb + ns > 0) and (fl >= basket_k or fl <= -stop_k or nb > N or ns > N):
            tr += nb + ns
            nb = 0; ns = 0
            anchor = c[t]; g = step_k * a[t] if a[t] == a[t] else g; up = 1; dn = 1
        net[t] = nb - ns
        gross_open[t] = nb + ns
        traded[t] = tr
    return net, gross_open, traded


@njit(cache=False)
def _zone_recovery(x, a, pulse, zone_k, tp_k, K):
    """Zone recovery on price x for a LONG first leg (run it on -price for a short first leg). Longs are all
    opened at P, shorts at P - z. Each new leg is sized so that a move of tp beyond the zone in its direction
    closes the whole basket with +1 unit x tp profit; after K recovery legs the next trigger stops everything
    out. Returns (net exposure, gross open units, gross units traded) per decision bar."""
    n = len(x)
    net = np.zeros(n); gross_open = np.zeros(n); traded = np.zeros(n)
    active = False; P = 0.0; z = 0.0; tp = 0.0; k = 0; last_long = True
    L = 0.0; S = 0.0
    for t in range(n):
        tr = 0.0
        if active:
            if x[t] >= P + tp or x[t] <= P - z - tp:                 # basket take-profit on either side
                tr += L + S; L = 0.0; S = 0.0; active = False
            elif last_long and x[t] <= P - z:
                if k >= K:
                    tr += L + S; L = 0.0; S = 0.0; active = False    # stop-out
                else:
                    need = (L * (z + tp) + tp) / tp
                    add = max(need - S, 0.0); S += add; tr += add; k += 1; last_long = False
            elif (not last_long) and x[t] >= P:
                if k >= K:
                    tr += L + S; L = 0.0; S = 0.0; active = False
                else:
                    need = (S * (z + tp) + tp) / tp
                    add = max(need - L, 0.0); L += add; tr += add; k += 1; last_long = True
        if not active and pulse[t] and a[t] == a[t] and a[t] > 0:
            active = True; k = 0; z = zone_k * a[t]; tp = tp_k * a[t]; P = x[t]; L = 1.0; S = 0.0
            last_long = True; tr += 1.0
        net[t] = L - S
        gross_open[t] = L + S
        traded[t] = tr
    return net, gross_open, traded


def _evaluate(bars, pip, net_units, gross_open, traded, unit, cost):
    """Vector P&L of the net exposure (zero-cost engine) minus costs on GROSS turnover and gross open units."""
    from fxlab.execution.simulator import vector_backtest, rollover_counts, _arrays
    from fxlab.execution.costs import ZERO
    roll = rollover_counts(bars.index)
    pos = pd.Series(np.clip(net_units * unit, -1e6, 1e6), index=bars.index)
    # vector_backtest clips |position| <= 1; scale the unit so the cap is never binding
    scale = max(1.0, float(np.abs(pos).max()))
    g = vector_backtest(bars, pos / scale, pip, ZERO, roll)["gross"] * scale
    bo, _, _, bc, ao, _, _, ac = _arrays(bars)
    mid_o = (bo + ao) / 2
    per_unit = (cost.spread_mult * (ao - bo) / 2 + cost.slip_spread_frac * (ao - bo) + cost.slip_fixed_pips * pip) \
        / mid_o + cost.commission_bps_per_side * 1e-4
    tr_exec = np.r_[0.0, traded[:-1]] * unit                       # decided at close t-1, executed at open t
    open_held = np.r_[0.0, gross_open[:-1]] * unit
    fin = open_held * roll * cost.swap_pips_per_night * pip / ((bc + ac) / 2)
    net = g.to_numpy() - tr_exec * per_unit - fin
    return pd.Series(net, index=bars.index)


def _stats(r):
    from fxlab.validation.metrics import daily
    d = daily(r)
    eq = (1 + r.fillna(0)).cumprod()
    m = (1 + r.fillna(0)).groupby(r.index.to_period("M")).prod() - 1
    return {"sharpe": float(d.mean() / d.std() * np.sqrt(260)) if d.std() > 0 else 0.0,
            "ann": float(d.mean() * 260), "max_dd": float((eq / eq.cummax() - 1).min()),
            "share_months_positive": float((m > 0).mean()), "worst_month": float(m.min()),
            "ruined": bool(eq.min() <= 0.5)}


def job(args):
    sym, tf = args
    import warnings
    warnings.filterwarnings("ignore")
    from fxlab.orchestration.runners import cached_bars
    from fxlab.execution.costs import BASE, ZERO
    from fxlab.data.instruments import get as instrument
    from fxlab.features._util import atr
    bars = cached_bars(sym, tf, "2007-01-01", "2025-12-31")
    pip = instrument(sym).pip
    c = bars["close"].to_numpy(float)
    a = atr(bars, 14).to_numpy(float)
    rows = []
    for gi, gc in enumerate(GRID_CFGS):
        nu, go, tr = _hedged_grid(c, a, gc["step"], gc["basket"], gc["stop"], gc["N"])
        unit = 1.0 / gc["N"]                                       # N open units on one side = 1x notional
        rec = {"symbol": sym, "tf": tf, "system": "hedged_grid", "cfg": str(gc), "seed": 0,
               "turnover_units_per_year": float(tr.sum() * unit / max(1e-9, (bars.index[-1] - bars.index[0]).days / 365.25))}
        for name, cm in (("net", BASE), ("gross", ZERO)):
            st = _stats(_evaluate(bars, pip, nu, go, tr, unit, cm))
            rec.update({f"{name}_{k}": v for k, v in st.items()})
        rows.append(rec)
    per_day = {"H1": 24, "H4": 6, "D1": 1}[tf]
    for zi, zc in enumerate(ZONE_CFGS):
        for seed in range(SEEDS):
            rng = np.random.default_rng([seed, zi, zlib.crc32(f"{sym}_{tf}".encode())])
            pulse = rng.random(len(c)) < 1.0 / per_day
            d0 = 1.0 if seed % 2 == 0 else -1.0                   # long-first / short-first (mirrored price)
            nu, go, tr = _zone_recovery(d0 * c, a, pulse, zc["zone"], zc["tp"], zc["K"])
            nu = d0 * nu
            unit = 0.25                                            # first leg = 0.25x notional
            rec = {"symbol": sym, "tf": tf, "system": "zone_recovery", "cfg": str(zc), "seed": seed}
            for name, cm in (("net", BASE), ("gross", ZERO)):
                st = _stats(_evaluate(bars, pip, nu, go, tr, unit, cm))
                rec.update({f"{name}_{k}": v for k, v in st.items()})
            rec["max_units"] = float(np.abs(go).max() * unit)
            rows.append(rec)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = [(s, tf) for s in SYMBOLS for tf in TFS]
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NP_WORKERS", 2))) as ex:
        for i, r in enumerate(ex.map(job, tasks)):
            rows += r
            print(f"{i + 1}/{len(tasks)} {tasks[i]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "runs.csv"), index=False)
    s = df.groupby(["system", "cfg", "tf"]).agg(
        runs=("net_sharpe", "size"), gross_sharpe=("gross_sharpe", "median"), net_sharpe=("net_sharpe", "median"),
        share_net_positive=("net_ann", lambda x: float((x > 0).mean())),
        months_positive=("net_share_months_positive", "median"), worst_month=("net_worst_month", "min"),
        median_max_dd=("net_max_dd", "median"), share_ruined=("net_ruined", "mean")).reset_index()
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
