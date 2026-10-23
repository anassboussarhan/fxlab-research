"""Break-even cost analysis of out-of-sample strategy streams (target-position families).

For each experiment, the saved OOS desired position (decided at close t, held from open t+1) is re-valued
at MID prices with zero costs. That gives the gross out-of-sample P&L and the traded notional, so
    break-even cost (bp per side) = gross P&L / traded notional x 1e4.
A strategy is only viable for a trader whose all-in cost per side (half spread + slippage + commission)
is below this number. We compare it with this study's retail ECN cost model and with an indicative
institutional cost level."""
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
from fxlab.data.instruments import get as instrument  # noqa: E402


def retail_cost_bps_per_side(bars, pip):
    sp = (bars["ask_o"] - bars["bid_o"]).median()
    mid = bars["close"].median()
    return float(((sp / 2 + BASE.slip_spread_frac * sp + BASE.slip_fixed_pips * pip) / mid) * 1e4
                 + BASE.commission_bps_per_side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    a = ap.parse_args()
    con = connect(read_only=True)
    q = "SELECT id, campaign, target, symbol, tf, config, artifacts FROM experiments WHERE status='ok' AND campaign IN (%s)" % \
        ",".join("?" * len(a.names))
    rows = con.execute(q, a.names).fetchall()
    con.close()
    out, bars_cache, retail = [], {}, {}
    for eid, camp, tgt, sym, tf, cfg, art in rows:
        e = CATALOG.get(tgt)
        if e is None or e.extra.get("mode", "state") != "state":
            continue
        cfg, art = json.loads(cfg), json.loads(art)
        p = art.get("oos_returns")
        if not p or not os.path.exists(p):
            continue
        x = pd.read_parquet(p)
        if "side" not in x:
            continue
        key = (sym, tf, cfg.get("start"), cfg.get("end"))
        if key not in bars_cache:
            bars_cache[key] = cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))
            retail[key] = retail_cost_bps_per_side(bars_cache[key], instrument(sym).pip)
        b = bars_cache[key]
        side = x["side"].reindex(b.index)                     # NaN outside OOS windows
        held = side.shift(1).fillna(0.0).clip(-1, 1).to_numpy()
        mo = ((b["bid_o"] + b["ask_o"]) / 2).to_numpy()
        nxt = np.r_[mo[1:], mo[-1]]
        gross = held * (nxt / mo - 1.0)
        turnover = np.abs(np.diff(np.r_[0.0, held]))
        oos = side.notna().to_numpy()
        years = max(oos.sum() / max(1, len(b)) * (b.index[-1] - b.index[0]).days / 365.25, 1e-9)
        g, tv = gross[oos].sum(), turnover[oos].sum()
        d = pd.Series(gross[oos], index=b.index[oos])
        from fxlab.validation.metrics import daily
        dd = daily(d)
        gs = float(dd.mean() / dd.std() * np.sqrt(260)) if dd.std() > 0 else 0.0
        out.append({"id": eid, "campaign": camp, "target": tgt, "symbol": sym, "tf": tf,
                    "exec": (cfg.get("exec") or {}).get("type", "market"),
                    "gross_oos_sharpe": gs, "gross_ann_bps": g / years * 1e4, "turnover_per_year": tv / years,
                    "breakeven_bps_per_side": g / tv * 1e4 if tv > 0 else np.nan,
                    "retail_cost_bps_per_side": retail[key]})
    T = pd.DataFrame(out)
    dest = RESULTS_DIR / "reports" / "breakeven"
    dest.mkdir(parents=True, exist_ok=True)
    T.to_csv(dest / "breakeven.csv", index=False)
    T["viable_retail"] = T.breakeven_bps_per_side > T.retail_cost_bps_per_side
    for thr in (0.1, 0.25, 0.5):
        T[f"viable_at_{thr}bp"] = T.breakeven_bps_per_side > thr
    s = {"streams": int(len(T)), "gross_oos_sharpe_median": float(T.gross_oos_sharpe.median()),
         "share_gross_positive": float((T.gross_oos_sharpe > 0).mean()),
         "breakeven_bps_median_of_positive": float(T[T.breakeven_bps_per_side > 0].breakeven_bps_per_side.median()),
         "viable_at_retail_cost": int(T.viable_retail.sum()),
         **{f"viable_at_{thr}bp_per_side": int(T[f"viable_at_{thr}bp"].sum()) for thr in (0.1, 0.25, 0.5)},
         "retail_cost_bps_per_side_by_symbol": T.groupby("symbol").retail_cost_bps_per_side.first().round(2).to_dict()}
    (dest / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    cols = ["target", "symbol", "tf", "exec", "gross_oos_sharpe", "breakeven_bps_per_side", "retail_cost_bps_per_side",
            "turnover_per_year"]
    print(T.sort_values("gross_oos_sharpe", ascending=False)[cols].head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
