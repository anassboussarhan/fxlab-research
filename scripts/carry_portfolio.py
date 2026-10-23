"""Classic cross-sectional currency carry factor (Lustig-Roussanov-Verdelhan style), built from the 7 USD pairs.

Each month-end (decision at the last D1 close of the month, using rates KNOWN then = previous month's average):
rank the 8 currencies (USD included) by 3-month rate; go long the top-k and short the bottom-k currencies,
equal weight (1/k per currency per leg). Currency weights are mapped onto the USD pairs (EURUSD, GBPUSD, AUDUSD,
NZDUSD long = long the currency; USDJPY, USDCHF, USDCAD long = short the currency); USD's own weight is the
residual. No parameter is fitted: the rule is fixed in advance (k = 2 and 3 reported, plus a version scaled to
10% annual volatility with trailing 60-day volatility). Each pair is priced with the vector engine: real
bid/ask spreads + slippage + commission on every rebalance, overnight financing from the rate differential
minus a 0.75% p.a. broker mark-up. Stress: 2x costs (incl. mark-up)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.runners import cached_bars  # noqa: E402
from fxlab.execution.costs import BASE, STRESS  # noqa: E402
from fxlab.execution.simulator import vector_backtest, rollover_counts  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.data.rates import known_at, rate_diff  # noqa: E402
from fxlab.validation.metrics import daily, performance  # noqa: E402
from fxlab.validation.stats import sharpe_ci, psr  # noqa: E402

PAIRS = {"EURUSD": ("EUR", 1), "GBPUSD": ("GBP", 1), "AUDUSD": ("AUD", 1), "NZDUSD": ("NZD", 1),
         "USDJPY": ("JPY", -1), "USDCHF": ("CHF", -1), "USDCAD": ("CAD", -1)}
OUT = RESULTS_DIR / "reports" / "carry_factor"


def sr(d):
    d = pd.Series(d).dropna()
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0


def run(k, cost, bars, common):
    R = {}
    for sym, (ccy, sgn) in PAIRS.items():
        b = bars[sym].loc[common[0]:common[-1]]
        K = known_at(b.index)
        mon = (b.index.year * 12 + b.index.month).to_numpy()
        is_month_end = np.r_[mon[1:] != mon[:-1], True]
        # weights decided at month-end closes, held through the next month
        w = pd.Series(np.nan, index=b.index)
        for t in b.index[is_month_end]:
            r = K.loc[t].dropna().sort_values()
            if len(r) < 2 * k:
                continue
            lo, hi = set(r.index[:k]), set(r.index[-k:])
            w.loc[t] = (1.0 / k if ccy in hi else -1.0 / k if ccy in lo else 0.0) * sgn
        pos = w.ffill().fillna(0.0)
        bt = vector_backtest(b, pos, instrument(sym).pip, cost, rollover_counts(b.index),
                             carry=rate_diff(sym, b.index).to_numpy())
        R[sym] = bt["net"]
    P = pd.DataFrame(R).fillna(0.0).sum(axis=1)
    return daily(P), pd.DataFrame(R)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bars = {s: cached_bars(s, "D1", "2007-01-01", "2025-12-31") for s in PAIRS}
    start = max(b.index[0] for b in bars.values())
    end = min(b.index[-1] for b in bars.values())
    common = (start, end)
    res = {"period": [str(start.date()), str(end.date())], "note": "pre-specified rule, no fitted parameters"}
    for k in (2, 3):
        d, per_pair = run(k, BASE, bars, common)
        d2, _ = run(k, STRESS[2.0], bars, common)
        vol = d.rolling(60, min_periods=20).std().shift(1) * np.sqrt(260)
        dv = d * (0.10 / vol).clip(upper=3.0)
        _, lo, hi = sharpe_ci(d.to_numpy(), n_boot=2000)
        eq = (1 + d).cumprod()
        perf = {"max_drawdown": float((eq / eq.cummax() - 1).min()), "skew_daily": float(d.skew())}
        monthly = (1 + d).groupby(d.index.to_period("M")).prod() - 1
        res[f"top{k}_bottom{k}"] = {
            "net_sharpe": sr(d), "ci95": [lo, hi], "psr_0": float(psr(d.to_numpy())),
            "ann_return": float(d.mean() * 260), "ann_vol": float(d.std() * np.sqrt(260)),
            "max_drawdown": perf["max_drawdown"], "skew_daily": perf["skew_daily"],
            "worst_month": float(monthly.min()), "worst_month_date": str(monthly.idxmin()),
            "cost_x2_sharpe": sr(d2), "vol_target_10pct_sharpe": sr(dv),
            "vol_target_10pct_ann_return": float(dv.mean() * 260),
            "by_year": {int(y): round(float(v), 4) for y, v in ((1 + d).groupby(d.index.year).prod() - 1).items()},
            "per_pair_ann_return": (per_pair.sum() / max((end - start).days / 365.25, 1e-9)).round(4).to_dict()}
        d.to_frame("ret").to_parquet(OUT / f"daily_top{k}.parquet")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
