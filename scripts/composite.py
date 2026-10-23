"""Cost-gated composite of out-of-sample signals, with a clean discovery / validation split.

1. DISCOVERY (2010-01-01 .. 2013-12-31 only): every target-position family's saved OOS positions are valued at
   zero cost; families whose median per-instrument gross Sharpe > 0.5 and that are positive on >= 70% of
   instruments are selected. Nothing after 2013 is used for this choice.
2. COMPOSITE per instrument = equal-weight mean of the selected families' OOS positions (no fitted weights).
3. RULE GRID (36 configs): enter when |composite| >= c_in; keep while |composite| >= c_out and same sign
   (hysteresis cuts turnover); trade only when the trailing EWMA-volatility percentile >= q (bigger moves per
   unit of spread); entry by market order or 0.5-ATR pullback limit (6-bar time-to-live).
4. VALIDATION (2014 ..): calendar walk-forward, expanding training window from 2012; at each 6-month test window
   the config with the best TRAINING portfolio net Sharpe (pooled over instruments) is applied. Concatenated
   test windows = out-of-sample result. Retail ECN costs; 2x-cost and +1-bar latency stress."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

import fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.registry import CATALOG  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.orchestration.runners import cached_bars  # noqa: E402
from fxlab.execution.costs import BASE, STRESS, LATENCY  # noqa: E402
from fxlab.execution.simulator import vector_backtest, order_backtest, rollover_counts  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.validation.metrics import daily, trading_day  # noqa: E402
from fxlab.validation.stats import sharpe_ci  # noqa: E402

DISC_END = pd.Timestamp("2013-12-31 23:59", tz="UTC")
VAL_START = pd.Timestamp("2014-01-01", tz="UTC")
GRID = list(itertools.product([0.25, 0.5, 0.75], ["zero", "half"], [0.0, 0.5, 0.75], ["market", "pullback"]))


def sr(d):
    d = d.dropna()
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0


def gross_daily(b, side):
    held = side.shift(1).fillna(0.0).clip(-1, 1).to_numpy()
    mo = ((b["bid_o"] + b["ask_o"]) / 2).to_numpy()
    g = pd.Series(held * (np.r_[mo[1:], mo[-1]] / mo - 1.0), index=b.index)
    return daily(g[side.notna()])


def rule_positions(comp, volpct, c_in, c_out_kind, q):
    c_out = 0.0 if c_out_kind == "zero" else c_in / 2
    x = comp.to_numpy(); v = volpct.to_numpy()
    out = np.zeros(len(x)); p = 0.0
    for i in range(len(x)):
        ci = x[i]
        if np.isnan(ci):
            p = 0.0
        elif abs(ci) >= c_in and (q == 0.0 or (not np.isnan(v[i]) and v[i] >= q)):
            p = np.sign(ci)
        elif p != 0.0 and np.sign(ci) == p and abs(ci) >= c_out and abs(ci) > 0:
            pass
        else:
            p = 0.0
        out[i] = p
    return pd.Series(out, index=comp.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["screen_h1_a", "screen_h1_b"])
    ap.add_argument("--tf", default="H1")
    a = ap.parse_args()
    con = connect(read_only=True)
    rows = con.execute("SELECT target, symbol, tf, config, artifacts FROM experiments WHERE status='ok' AND kind='strategy' "
                       "AND campaign IN (%s)" % ",".join("?" * len(a.names)), a.names).fetchall()
    con.close()
    sides, cfgs = {}, {}
    for tgt, sym, tf, cfg, art in rows:
        e = CATALOG.get(tgt)
        cfg, art = json.loads(cfg), json.loads(art)
        if tf != a.tf or e is None or e.extra.get("mode", "state") != "state" or cfg.get("exec") or cfg.get("risk_grid"):
            continue
        p = art.get("oos_returns")
        if p and os.path.exists(p):
            x = pd.read_parquet(p)
            if "side" in x:
                sides[(tgt, sym)] = x["side"]
                cfgs[sym] = cfg
    symbols = sorted({s for _, s in sides})
    bars = {s: cached_bars(s, a.tf, cfgs[s].get("start"), cfgs[s].get("end")) for s in symbols}
    # ---------------------------------------------------------------- 1) discovery: family selection <= 2013
    disc = {}
    for (tgt, sym), sd in sides.items():
        b = bars[sym]
        s = sd.reindex(b.index)
        s = s.where(b.index <= DISC_END)
        disc.setdefault(tgt, {})[sym] = sr(gross_daily(b, s))
    D = pd.DataFrame(disc)
    score = pd.DataFrame({"median_gross_sharpe_2010_13": D.median(), "share_positive": (D > 0).mean()})
    selected = score[(score.median_gross_sharpe_2010_13 > 0.5) & (score.share_positive >= 0.7)].index.tolist()
    print("families selected on 2010-2013 only:", selected)
    if not selected:
        print("no family passed the discovery filter"); return
    # ---------------------------------------------------------------- 2) composite + gates per instrument
    comp, volp = {}, {}
    for sym in symbols:
        b = bars[sym]
        M = pd.DataFrame({t: sides[(t, sym)].reindex(b.index) for t in selected if (t, sym) in sides})
        comp[sym] = M.mean(axis=1, skipna=False)
        lr = np.log(b["close"]).diff()
        ev = np.sqrt((lr ** 2).ewm(alpha=0.06, adjust=False, min_periods=50).mean())
        volp[sym] = ev.rolling(2000, min_periods=500).rank(pct=True)
    # ---------------------------------------------------------------- 3) simulate every config once (causal)
    sims = {}
    for gi, (c_in, c_out, q, entry) in enumerate(GRID):
        for sym in symbols:
            b = bars[sym]
            pos = rule_positions(comp[sym], volp[sym], c_in, c_out, q)
            pip = instrument(sym).pip
            roll = rollover_counts(b.index)
            if entry == "market":
                bt = vector_backtest(b, pos, pip, BASE, roll)
            else:
                bt = order_backtest(b, pos, pip, BASE, entry="limit_pullback", k=0.5, ttl=6, rollovers=roll)
            sims[(gi, sym)] = (pos, bt["net"])
    # ---------------------------------------------------------------- 4) walk-forward selection, pooled portfolio
    def portfolio(gi, lo, hi):
        S = {}
        for sym in symbols:
            r = sims[(gi, sym)][1]
            r = r[(r.index >= lo) & (r.index < hi) & comp[sym].notna()]
            if len(r):
                S[sym] = daily(r)
        P = pd.DataFrame(S)
        return P.mean(axis=1) if len(P.columns) else pd.Series(dtype=float)  # equal notional across instruments

    first = max(min(comp[s].dropna().index.min() for s in symbols), pd.Timestamp("2012-01-01", tz="UTC"))
    end = max(bars[s].index.max() for s in symbols)
    oos, chosen = [], []
    t0 = VAL_START
    while t0 < end:
        t1 = t0 + pd.DateOffset(months=6)
        scores = [sr(portfolio(gi, first, t0)) for gi in range(len(GRID))]
        best = int(np.argmax(scores))
        oos.append(portfolio(best, t0, t1))
        chosen.append({"test_start": str(t0.date()), "config": GRID[best], "train_sharpe": scores[best],
                       "abstain": bool(scores[best] <= 0)})
        t0 = t1
    R = pd.concat(oos).sort_index()
    # variant with abstention: stay flat in a test window when no config had a positive TRAINING net Sharpe
    R_abst = pd.concat([o * (0.0 if c["abstain"] else 1.0) for o, c in zip(oos, chosen)]).sort_index()
    _, lo, hi = sharpe_ci(R.to_numpy(), n_boot=1000)
    t = st.ttest_1samp(R, 0.0, alternative="greater")
    # stress: re-simulate the chosen configs under 2x costs and +1 bar latency
    def stressed(cm):
        parts = []
        for ch, (t0s) in zip(chosen, [c["test_start"] for c in chosen]):
            gi = GRID.index(tuple(ch["config"]))
            tt0 = pd.Timestamp(t0s, tz="UTC"); tt1 = tt0 + pd.DateOffset(months=6)
            S = {}
            for sym in symbols:
                b = bars[sym]; pos = sims[(gi, sym)][0]
                pip = instrument(sym).pip
                if GRID[gi][3] == "market":
                    bt = vector_backtest(b, pos, pip, cm)
                else:
                    bt = order_backtest(b, pos, pip, cm, entry="limit_pullback", k=0.5, ttl=6)
                r = bt["net"]; r = r[(r.index >= tt0) & (r.index < tt1) & comp[sym].notna()]
                if len(r):
                    S[sym] = daily(r)
            parts.append(pd.DataFrame(S).mean(axis=1))
        return sr(pd.concat(parts))
    res = {"families_selected_2010_2013": selected, "grid_size": len(GRID), "validation_start": str(VAL_START.date()),
           "oos_sharpe": sr(R), "oos_ci95": [lo, hi], "oos_ann_return": float(R.mean() * 260),
           "oos_ann_vol": float(R.std() * np.sqrt(260)), "p_one_sided": float(t.pvalue),
           "by_year": {int(k): float(v) for k, v in ((1 + R).groupby(R.index.year).prod() - 1).items()},
           "stress_cost_x2_sharpe": stressed(STRESS[2.0]), "stress_latency1_sharpe": stressed(LATENCY[1]),
           "with_abstention": {"windows_traded": int(sum(not c["abstain"] for c in chosen)), "windows": len(chosen),
                               "oos_sharpe": sr(R_abst), "oos_ann_return": float(R_abst.mean() * 260)},
           "chosen_configs": chosen, "discovery_scores": score.sort_values("median_gross_sharpe_2010_13", ascending=False).head(15).round(3).to_dict()}
    dest = RESULTS_DIR / "reports" / f"composite_{a.tf}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    R.to_frame("ret").to_parquet(dest / "oos_daily_returns.parquet")
    print(json.dumps({k: v for k, v in res.items() if k not in ("discovery_scores",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
