"""Re-execute every saved out-of-sample position stream (target-position families) under each ENTRY ORDER TYPE.
Positions are exactly those chosen by the walk-forward under market execution; only the entry mechanics change,
so this isolates the effect of the order type. Exits are market orders. Results per stream and per family
portfolio (equal-risk, weights from past data)."""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.registry import CATALOG  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.orchestration.runners import cached_bars  # noqa: E402
from fxlab.execution.costs import BASE  # noqa: E402
from fxlab.execution.simulator import vector_backtest, order_backtest, rollover_counts  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.validation.metrics import daily  # noqa: E402

ENTRY = {
    "market": None,
    "limit_touch_requote": ("limit_touch", 0.0, 0),
    "limit_touch_ttl3": ("limit_touch", 0.0, 3),
    "limit_pullback_0.25atr_ttl3": ("limit_pullback", 0.25, 3),
    "limit_pullback_0.5atr_ttl6": ("limit_pullback", 0.5, 6),
    "stop_breakout_0.25atr_ttl3": ("stop_breakout", 0.25, 3),
    "stop_breakout_0.5atr_ttl6": ("stop_breakout", 0.5, 6),
}


def sr(d):
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--tf", default=None)
    ap.add_argument("--out", default="entry_types")
    a = ap.parse_args()
    con = connect(read_only=True)
    rows = con.execute("SELECT id, target, symbol, tf, config, artifacts FROM experiments WHERE status='ok' AND kind='strategy' "
                       "AND campaign IN (%s)" % ",".join("?" * len(a.names)), a.names).fetchall()
    con.close()
    recs, streams, bars_cache = [], {k: {} for k in ENTRY}, {}
    for eid, tgt, sym, tf, cfg, art in rows:
        e = CATALOG.get(tgt)
        if e is None or e.extra.get("mode", "state") != "state" or (a.tf and tf != a.tf):
            continue
        cfg, art = json.loads(cfg), json.loads(art)
        if cfg.get("exec") or cfg.get("risk_grid"):
            continue
        p = art.get("oos_returns")
        if not p or not os.path.exists(p):
            continue
        x = pd.read_parquet(p)
        if "side" not in x:
            continue
        key = (sym, tf, cfg.get("start"), cfg.get("end"))
        if key not in bars_cache:
            bars_cache = {key: cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))}  # keep memory bounded
        b = bars_cache[key]
        side = x["side"].reindex(b.index)
        oos = side.notna()
        pos = side.fillna(0.0)
        pip = instrument(sym).pip
        roll = rollover_counts(b.index)
        row = {"id": eid, "target": tgt, "symbol": sym, "tf": tf}
        for name, spec in ENTRY.items():
            if spec is None:
                bt = vector_backtest(b, pos, pip, BASE, roll)
                fr = 1.0
            else:
                bt = order_backtest(b, pos, pip, BASE, entry=spec[0], k=spec[1], ttl=spec[2], rollovers=roll)
                fr = bt.attrs["fill_rate"]
            d = daily(bt["net"][oos])
            row[name] = sr(d)
            row[name + ":fill"] = fr
            streams[name][f"{tgt}|{sym}|{tf}"] = d
        recs.append(row)
    T = pd.DataFrame(recs)
    dest = RESULTS_DIR / "reports" / a.out
    dest.mkdir(parents=True, exist_ok=True)
    T.to_csv(dest / "per_stream.csv", index=False)
    names = list(ENTRY)
    summ = {"streams": int(len(T)), "by_entry": {}}
    for n in names:
        summ["by_entry"][n] = {"median_sharpe": float(T[n].median()), "share_positive": float((T[n] > 0).mean()),
                               "best": float(T[n].max()), "median_fill_rate": float(T[n + ":fill"].median())}
    T["family"] = T.target
    fam = T.groupby("family")[names].median()
    fam.to_csv(dest / "family_median_by_entry.csv")
    best = T[names].idxmax(axis=1).value_counts().to_dict()
    summ["best_entry_type_count"] = best
    (dest / "summary.json").write_text(json.dumps(summ, indent=1))
    pd.set_option("display.width", 250)
    print(json.dumps(summ, indent=1))
    print(fam.round(2).sort_values("market", ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
