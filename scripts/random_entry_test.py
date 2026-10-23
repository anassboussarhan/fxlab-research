"""No-prediction test: can trade management alone (exits, pyramiding, martingale sizing) make money?

Entries are RANDOM: a coin-flip timing process (on average one entry opportunity per trading day) with a fixed
direction per run (long-only runs and short-only runs, reported together, so the experiment is direction-neutral
and a pair's drift cancels). Each run is managed with:
  * every exit configuration of the joint search (campaign.RISK_GRID: fixed / ATR / volatility / structural /
    quantile stops, R-multiple / power / log / piecewise / polynomial targets, trailing, breakeven, partial
    take-profit, time exits, volatility sizing), through the trade engine (stop-first, gaps fill at the open);
  * pyramiding: 1 unit at the random entry, +1 unit after every further +1 ATR in favour (max 4), all units out
    when price gives back 2 ATR from the best close;
  * martingale: take-profit and stop at +-1 ATR (closes), double the size after every loss (max 5 doublings),
    reset after a win.
No parameter is selected, so no walk-forward is needed: this is a null experiment over 2007-2018 development
data (EURUSD to 2022) on every local instrument at H1, H4 and D1, many random seeds each. If management alone
created an edge, the GROSS (zero-cost) results would be positive across instruments and seeds.
Intrabar order (RE_INTRABAR): m1 (default) replays the M1 path when a bar holds both stop and target;
stopfirst = conservative stop-first rule (biases tight exits downward; kept for comparison).
Output: results/reports/random_entry/{runs.csv, summary.csv, summary.json}"""
from __future__ import annotations

import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY",
           "XAUUSD", "XAGUSD", "EURCHF", "AUDJPY"]
TFS = {"H1": 24, "H4": 6, "D1": 1}          # entry opportunities ~ once per trading day
SEEDS = int(os.environ.get("RE_SEEDS", 12))
M1_REPLAY = os.environ.get("RE_INTRABAR", "m1") == "m1"     # m1 = replay minute paths; stopfirst = conservative
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reports",
                   "random_entry" + ("" if M1_REPLAY else "_stopfirst"))


@njit(cache=False)
def _pyramid_random(c, a, pulse, d, N, give_back):
    n = len(c)
    pos = np.zeros(n)
    units = 0; best = 0.0; last_add = 0.0
    for t in range(n):
        if units == 0:
            if pulse[t] and a[t] == a[t]:
                units = 1; best = c[t]; last_add = c[t]
        else:
            if d * (c[t] - best) > 0:
                best = c[t]
            if d * (best - c[t]) >= give_back * a[t]:
                units = 0
            elif units < N and d * (c[t] - last_add) >= a[t]:
                units += 1; last_add = c[t]
        pos[t] = d * units / N
    return pos


@njit(cache=False)
def _martingale_random(c, a, pulse, d, K):
    n = len(c)
    pos = np.zeros(n)
    inpos = False; entry = 0.0; rng_ = 0.0; losses = 0
    for t in range(n):
        if inpos:
            m = d * (c[t] - entry)
            if m >= rng_:
                inpos = False; losses = 0
            elif m <= -rng_:
                inpos = False; losses = min(losses + 1, K)
        if not inpos and pulse[t] and a[t] == a[t]:
            inpos = True; entry = c[t]; rng_ = a[t]
        pos[t] = d * (2.0 ** losses) / (2.0 ** K) if inpos else 0.0
    return pos


def _sharpe(r):
    from fxlab.validation.metrics import daily
    x = daily(r)
    return float(x.mean() / x.std() * np.sqrt(260)) if len(x) > 20 and x.std() > 0 else 0.0


def _ann(r):
    from fxlab.validation.metrics import daily
    x = daily(r)
    return float(x.mean() * 260)


def _mdd(r):
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def job(args):
    sym, tf = args
    import importlib.util
    import fxlab.strategies  # noqa: F401
    from fxlab.orchestration.runners import cached_bars
    from fxlab.execution.costs import BASE, ZERO
    from fxlab.execution.risk import RiskConfig, build_orders
    from fxlab.execution.simulator import simulate_trades, vector_backtest, rollover_counts
    from fxlab.data.instruments import get as instrument
    from fxlab.features._util import atr
    spec = importlib.util.spec_from_file_location("campaign_mod", os.path.join(os.path.dirname(__file__), "campaign.py"))
    cm = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(cm)
    sys.argv = argv
    bars = cached_bars(sym, tf, "2007-01-01", "2025-12-31")
    from fxlab.data.store import load_m1, M1_PATH_COLUMNS
    m1 = load_m1(sym, "2007-01-01", "2025-12-31", columns=M1_PATH_COLUMNS) if M1_REPLAY else None
    pip = instrument(sym).pip
    roll = rollover_counts(bars.index)
    c = bars["close"].to_numpy(float)
    a14 = atr(bars, 14).to_numpy(float)
    n = len(bars)
    rows = []
    base_orders = {}
    for ri, ro in enumerate(cm.RISK_GRID):
        for d in (1.0, -1.0):
            base_orders[(ri, d)] = build_orders(bars, pd.Series(d, index=bars.index), RiskConfig(**ro), pip)
    for seed in range(SEEDS):
        for d in (1.0, -1.0):
            rng = np.random.default_rng([seed, 1 if d > 0 else 2, zlib.crc32(f"{sym}_{tf}".encode())])
            pulse = rng.random(n) < 1.0 / TFS[tf]
            side = pd.Series(np.where(pulse, d, 0.0), index=bars.index)
            for ri, ro in enumerate(cm.RISK_GRID):
                o = base_orders[(ri, d)].copy()
                o["side"] = side
                rec = {"symbol": sym, "tf": tf, "seed": seed, "dir": d, "method": f"exit_{ri:02d}"}
                for name, cost in (("net", BASE), ("gross", ZERO)):
                    res = simulate_trades(bars, o, pip, cost, exit_on_flat=False, rollovers=roll, m1=m1)
                    r = res.bars["net"]
                    rec[f"{name}_sharpe"] = _sharpe(r)
                    rec[f"{name}_ann"] = _ann(r)
                    if name == "net":
                        t = res.trades
                        rec["trades"] = len(t)
                        rec["win_rate"] = float((t["net_ret"] > 0).mean()) if len(t) else np.nan
                        rec["max_dd"] = _mdd(r)
                rows.append(rec)
            for meth, pos in (("pyramid", _pyramid_random(c, a14, pulse, d, 4, 2.0)),
                              ("martingale", _martingale_random(c, a14, pulse, d, 5))):
                p = pd.Series(pos, index=bars.index)
                rec = {"symbol": sym, "tf": tf, "seed": seed, "dir": d, "method": meth}
                for name, cost in (("net", BASE), ("gross", ZERO)):
                    bt = vector_backtest(bars, p, pip, cost, roll)
                    rec[f"{name}_sharpe"] = _sharpe(bt["net"])
                    rec[f"{name}_ann"] = _ann(bt["net"])
                    if name == "net":
                        rec["max_dd"] = _mdd(bt["net"])
                        s = np.sign(pos)
                        starts = np.flatnonzero((s != 0) & (np.r_[0, s[:-1]] == 0))
                        ends = np.flatnonzero((s != 0) & (np.r_[s[1:], 0] == 0))
                        cum = np.r_[0, np.cumsum(bt["net"].to_numpy())]
                        k = min(len(starts), len(ends))
                        pnl = cum[np.minimum(ends[:k] + 2, n)] - cum[starts[:k] + 1]   # held bars start+1..end+1
                        rec["trades"] = int(k)
                        rec["win_rate"] = float((pnl > 0).mean()) if k else np.nan
                rows.append(rec)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = [(s, tf) for s in SYMBOLS for tf in TFS]
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("RE_WORKERS", 2))) as ex:
        for i, r in enumerate(ex.map(job, tasks)):
            rows += r
            print(f"{i + 1}/{len(tasks)} {tasks[i]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "runs.csv"), index=False)
    # direction-neutral: average the long-only and the short-only run of the same seed
    dn = df.groupby(["symbol", "tf", "seed", "method"]).agg(
        gross_ann=("gross_ann", "mean"), net_ann=("net_ann", "mean"), gross_sharpe=("gross_sharpe", "mean"),
        net_sharpe=("net_sharpe", "mean"), win_rate=("win_rate", "mean"), trades=("trades", "sum"),
        max_dd=("max_dd", "min")).reset_index()
    import importlib.util
    spec = importlib.util.spec_from_file_location("campaign_mod", os.path.join(os.path.dirname(__file__), "campaign.py"))
    cm = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["x"]
    spec.loader.exec_module(cm)
    sys.argv = argv
    label = {f"exit_{i:02d}": json.dumps(ro) for i, ro in enumerate(cm.RISK_GRID)}
    label.update({"pyramid": "random entry + pyramid (+1 unit per +1 ATR, max 4, out at 2 ATR give-back)",
                  "martingale": "random entry + martingale (TP/SL 1 ATR, double after loss, max 5x)"})
    summ = dn.groupby(["method", "tf"]).agg(
        runs=("net_sharpe", "size"), gross_sharpe_mean=("gross_sharpe", "mean"),
        gross_sharpe_median=("gross_sharpe", "median"), share_gross_positive=("gross_ann", lambda x: float((x > 0).mean())),
        net_sharpe_mean=("net_sharpe", "mean"), share_net_positive=("net_ann", lambda x: float((x > 0).mean())),
        win_rate=("win_rate", "median"), worst_dd=("max_dd", "min")).reset_index()
    summ["config"] = summ["method"].map(label)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    # t-test of the cross-run mean gross Sharpe per method (runs are independent seeds x instruments)
    tests = {}
    for m, g in dn.groupby("method"):
        x = g["gross_sharpe"].to_numpy()
        tests[m] = {"mean_gross_sharpe": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))),
                    "t": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12)),
                    "mean_net_sharpe": float(g["net_sharpe"].mean()), "share_net_positive": float((g["net_ann"] > 0).mean()),
                    "median_win_rate": float(g["win_rate"].median()), "config": label[m]}
    json.dump({"seeds": SEEDS, "instruments": SYMBOLS, "tfs": list(TFS), "runs": int(len(dn)), "by_method": tests},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(pd.DataFrame(tests).T[["mean_gross_sharpe", "t", "mean_net_sharpe", "share_net_positive",
                                 "median_win_rate"]].round(3).to_string())


if __name__ == "__main__":
    main()
