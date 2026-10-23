"""Recompute daily-based performance statistics of finished strategy experiments from their SAVED out-of-sample
returns (after the trading-day aggregation fix). Simulations are not re-run, so fold-level parameter choices
are unchanged. The previous values are kept under metrics['_pre_fix'] for comparison. Stress-test Sharpes are
not recomputed (their return streams are not stored) and are flagged."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fxlab.orchestration.campaign import connect, _to_json  # noqa: E402
from fxlab.validation.metrics import performance, daily  # noqa: E402
from fxlab.validation import stats  # noqa: E402

KEYS = ["net_return", "ann_return", "sharpe", "sortino", "calmar", "max_drawdown", "dd_duration_bars", "cvar_5_daily",
        "skew_daily", "kurt_daily", "n_days", "years", "trades", "trades_per_year", "win_rate", "profit_factor",
        "expectancy", "payoff_ratio", "avg_bars_held"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    a = ap.parse_args()
    con = connect(read_only=True)
    rows = con.execute("SELECT id, metrics, artifacts FROM experiments WHERE status='ok' AND kind='strategy' AND campaign IN (%s)"
                       % ",".join("?" * len(a.names)), a.names).fetchall()
    con.close()
    changed = []
    for eid, m, art in rows:
        m, art = json.loads(m), json.loads(art)
        if "_pre_fix" in m:
            continue
        p = art.get("oos_returns")
        if not p or not os.path.exists(p):
            continue
        r = pd.read_parquet(p)["ret"]
        tp = art.get("oos_trades")
        trades = pd.read_parquet(tp) if tp and os.path.exists(tp) else None
        new = performance(r, trades)
        d = daily(r).to_numpy()
        new["psr_0"] = stats.psr(d)
        new["ttest_p"] = stats.ttest_mean(d)
        _, lo, hi = stats.sharpe_ci(d, n_boot=500)
        new["sharpe_ci95"] = [lo, hi]
        m["_pre_fix"] = {k: m.get(k) for k in ("sharpe", "sortino", "psr_0", "ttest_p", "sharpe_ci95", "n_days")}
        for k in KEYS + ["psr_0", "ttest_p", "sharpe_ci95"]:
            if k in new:
                m[k] = new[k]
        m["stress_note"] = "stress Sharpes computed before the trading-day aggregation fix (Sunday-evening bars excluded)"
        c = connect(retries=9000)   # hold the write lock only for this single update
        c.execute("UPDATE experiments SET metrics = ? WHERE id = ?", [_to_json(m), eid])
        c.close()
        changed.append((m["_pre_fix"]["sharpe"], m["sharpe"]))
    if changed:
        df = pd.DataFrame(changed, columns=["before", "after"])
        print(f"updated {len(df)} experiments; median Sharpe before {df.before.median():.3f} after {df.after.median():.3f}; "
              f"max |change| {(df.after - df.before).abs().max():.3f}; share>0 before {(df.before > 0).mean():.3f} "
              f"after {(df.after > 0).mean():.3f}")
    else:
        print("nothing to update")


if __name__ == "__main__":
    main()
