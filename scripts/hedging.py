"""Hedging comparison on out-of-sample strategy positions of a campaign.

For each OOS stream on instrument S, a hedge leg in instrument H (the panel instrument with the highest
trailing return correlation to S, re-chosen monthly from PAST data) is held at -beta_t x position_t,
beta_t = trailing min-variance hedge ratio (shifted). The hedge leg is simulated with the SAME bid/ask
execution model (spread, slippage, commission, financing), so costs, basis risk and correlated losses
are all inside the comparison. Margin: the hedge adds |beta| x notional of gross exposure, reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.orchestration.runners import cached_bars  # noqa: E402
from fxlab.execution.simulator import vector_backtest  # noqa: E402
from fxlab.execution.costs import BASE  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.validation.metrics import performance  # noqa: E402
from fxlab.data.store import available  # noqa: E402

CANDIDATES = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "XAUUSD", "XAGUSD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-streams", type=int, default=40)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    out = RESULTS_DIR / "reports" / a.name
    con = connect()
    df = con.execute("SELECT id, target, symbol, tf, metrics, artifacts, config FROM experiments "
                     "WHERE campaign=? AND status='ok'", [a.name]).fetchdf()
    con.close()
    df["sharpe"] = df["metrics"].apply(lambda s: json.loads(s).get("sharpe", np.nan))
    df = df.sort_values("sharpe", ascending=False).head(a.max_streams)
    rows = []
    bars_cache = {}
    for _, r in df.iterrows():
        cfg = json.loads(r["config"])
        art = json.loads(r["artifacts"])
        if not art.get("oos_returns") or not os.path.exists(art["oos_returns"]):
            continue
        x = pd.read_parquet(art["oos_returns"])
        sym, tf = r["symbol"], r["tf"]
        key = (sym, tf)
        if key not in bars_cache:
            bars_cache[key] = cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))
        b = bars_cache[key]
        rs = np.log(b["close"]).diff()
        best, best_c = None, 0.0
        cors = {}
        for h in CANDIDATES:
            if h == sym or not available(h):
                continue
            hb = bars_cache.setdefault((h, tf), cached_bars(h, tf, cfg.get("start"), cfg.get("end")))
            rh = np.log(hb["close"]).diff().reindex(b.index)
            cors[h] = rs.rolling(a.n, min_periods=a.n // 2).corr(rh).shift(1)
        if not cors:
            continue
        C = pd.DataFrame(cors).reindex(x.index)
        # choose hedge instrument monthly by the trailing |corr| known at month start
        month = x.index.to_period("M")
        choice = pd.Series(index=x.index, dtype=object)
        for m in month.unique():
            first = x.index[month == m][0]
            row = C.loc[first].abs().dropna()
            choice[month == m] = row.idxmax() if len(row) else None
        hedge_ret = pd.Series(0.0, index=x.index)
        gross_add = []
        for h in choice.dropna().unique():
            hb = bars_cache[(h, tf)]
            rh = np.log(hb["close"]).diff().reindex(b.index)
            beta = (rs.rolling(a.n, min_periods=a.n // 2).cov(rh) / (rh.rolling(a.n, min_periods=a.n // 2).var() + 1e-18)).shift(1)
            pos = (-beta.reindex(x.index) * x["side"]).where(choice == h, 0.0).fillna(0.0).clip(-2, 2)
            bt = vector_backtest(hb.reindex(x.index).ffill(), pos / 2.0, instrument(h).pip, BASE)  # sim caps at |1|; scale back
            hedge_ret = hedge_ret.add(bt["net"] * 2.0, fill_value=0.0)
            gross_add.append(pos.abs().mean())
        un = x["ret"]
        hed = un + hedge_ret
        pu, ph = performance(un), performance(hed)
        rows.append({"stream": f"{r['target']}|{sym}|{tf}", "unhedged_sharpe": pu.get("sharpe"),
                     "hedged_sharpe": ph.get("sharpe"), "unhedged_mdd": pu.get("max_drawdown"),
                     "hedged_mdd": ph.get("max_drawdown"), "unhedged_ann_vol": float(un.std() * np.sqrt(len(un) / pu.get("years", 1))),
                     "hedged_ann_vol": float(hed.std() * np.sqrt(len(hed) / ph.get("years", 1))),
                     "hedge_cost_drag_ann": float(-hedge_ret[hedge_ret < 0].sum() / pu.get("years", 1)),
                     "extra_gross_exposure_avg": float(np.mean(gross_add)) if gross_add else 0.0})
    T = pd.DataFrame(rows)
    T.to_csv(out / "hedging_comparison.csv", index=False)
    if len(T):
        print(T[["stream", "unhedged_sharpe", "hedged_sharpe", "unhedged_mdd", "hedged_mdd"]].to_string())
        print("median sharpe change", float((T.hedged_sharpe - T.unhedged_sharpe).median()))


if __name__ == "__main__":
    main()
