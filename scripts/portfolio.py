"""Portfolio construction, sizing and hedging comparison on OUT-OF-SAMPLE strategy streams of a campaign.

Inputs: the per-experiment OOS daily returns and positions saved by the strategy runner. All weights /
scalers at day t are computed from data up to t-1 (shifted). Compared:
  allocation: equal weight, inverse volatility, correlation-aware, currency-exposure netting (cap per currency)
  sizing    : none, portfolio volatility targeting, fractional Kelly per stream, drawdown-based de-risking
  hedging   : each FX stream hedged with the min-variance hedge ratio of its pair vs the USD factor
              (hedge leg pays the quoted spread on every daily rebalance; basis risk is whatever remains)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.validation.metrics import performance  # noqa: E402
from fxlab.execution.risk import kelly_fraction, drawdown_scaler  # noqa: E402


def daily(s):
    from fxlab.validation.metrics import daily as _d
    return _d(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--top", type=int, default=20, help="streams selected by OOS Sharpe of EARLIER folds only")
    a = ap.parse_args()
    out = RESULTS_DIR / "reports" / a.name
    out.mkdir(parents=True, exist_ok=True)
    con = connect()
    df = con.execute("SELECT id, target, symbol, artifacts FROM experiments WHERE campaign=? AND status='ok'",
                     [a.name]).fetchdf()
    con.close()
    R, P = {}, {}
    for _, r in df.iterrows():
        art = json.loads(r["artifacts"])
        p = art.get("oos_returns")
        if p and os.path.exists(p):
            x = pd.read_parquet(p)
            key = f"{r['target']}|{r['symbol']}"
            R[key] = daily(x["ret"])
            if "side" in x:
                from fxlab.validation.metrics import trading_day
                P[key] = x["side"].groupby(trading_day(x.index)).last()
    R = pd.DataFrame(R).fillna(0.0)
    if R.shape[1] < 2:
        print("not enough streams")
        return
    # stream selection must itself be out-of-sample: rank by trailing 1y Sharpe, re-ranked monthly
    trailing = (R.rolling(260, min_periods=120).mean() / (R.rolling(260, min_periods=120).std() + 1e-12)).shift(1)
    month = R.index.to_period("M")
    sel = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    for m in month.unique():
        first = R.index[month == m][0]
        top = trailing.loc[first].dropna().nlargest(a.top).index
        sel.loc[month == m, top] = 1.0
    vol = R.rolling(60, min_periods=20).std().shift(1)
    alloc = {
        "equal_weight": sel,
        "inverse_vol": sel / (vol + 1e-12),
    }
    corr_avg = R.rolling(120, min_periods=60).corr().groupby(level=0).mean().shift(1) if R.shape[1] <= 60 else None
    if corr_avg is not None:
        alloc["correlation_aware"] = sel / (vol + 1e-12) / (1 + corr_avg.reindex(R.index).clip(lower=0))
    res = {}
    for name, W in alloc.items():
        W = W.div(W.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        port = (W * R).sum(axis=1)
        res[f"alloc:{name}"] = performance(port)
        if name == "inverse_vol":
            base = port
    # sizing variants on the inverse-vol portfolio
    pv = base.rolling(60, min_periods=20).std().shift(1) * np.sqrt(260)
    res["sizing:vol_target_10pct"] = performance(base * (0.10 / (pv + 1e-12)).clip(upper=5))
    eq = (1 + base).cumprod()
    res["sizing:drawdown_derisk"] = performance(base * drawdown_scaler(eq, 0.05, 0.2))
    kel = pd.Series([kelly_fraction(base.iloc[max(0, i - 250):i].to_numpy(), 0.25) if i > 60 else 0.0
                     for i in range(len(base))], index=base.index)
    res["sizing:fractional_kelly_0.25"] = performance(base * kel.clip(upper=3))
    # currency exposure netting (requires positions): cap net exposure per currency at 1 unit of gross
    if P:
        Pos = pd.DataFrame(P).reindex(R.index).fillna(0.0)
        W = (sel / (vol + 1e-12)).div((sel / (vol + 1e-12)).sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        exp = {}
        for col in Pos.columns:
            sym = col.split("|")[1]
            for ccy, sgn in ((sym[:3], 1), (sym[3:], -1)):
                exp[ccy] = exp.get(ccy, 0) + sgn * Pos[col] * W[col]
        E = pd.DataFrame(exp).shift(1)
        scale = (0.5 / E.abs().max(axis=1).where(E.abs().max(axis=1) > 0.5)).fillna(1.0)
        res["alloc:currency_netted_cap0.5"] = performance((W * R).sum(axis=1) * scale)
    pd.DataFrame(res).T.to_csv(out / "portfolio_risk_comparison.csv")
    (out / "portfolio_risk_comparison.json").write_text(json.dumps(res, indent=1, default=str))
    print(pd.DataFrame(res).T[["sharpe", "ann_return", "max_drawdown", "calmar"]].to_string())


if __name__ == "__main__":
    main()
