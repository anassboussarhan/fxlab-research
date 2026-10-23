"""Deep dive on one strategy family across instruments, using the SAVED out-of-sample positions only
(parameters were chosen by the walk-forward on training windows; nothing is re-optimised here).

* per-instrument and equal-risk portfolio Sharpe under a ladder of execution-cost assumptions
  (real quoted spread of each execution bar is always paid; extras vary);
* per-year stability of the portfolio;
* significance of the GROSS portfolio (block-bootstrap CI, t-test) with a Bonferroni factor for the number of
  families that were screened (the effect was found by looking at all of them)."""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.orchestration.runners import cached_bars  # noqa: E402
from fxlab.execution.costs import CostModel  # noqa: E402
from fxlab.execution.simulator import vector_backtest, passive_backtest  # noqa: E402
from fxlab.data.instruments import get as instrument  # noqa: E402
from fxlab.validation.stats import sharpe_ci  # noqa: E402

LADDER = {
    "retail_ecn (spread+slip+0.35bp comm)": CostModel(),
    "low (spread + 0.1bp comm, no slip)": CostModel(slip_spread_frac=0, slip_fixed_pips=0, commission_bps_per_side=0.1),
    "spread_only": CostModel(slip_spread_frac=0, slip_fixed_pips=0, commission_bps_per_side=0.0),
    "zero_cost (diagnostic)": CostModel(0, 0, 0, 0, 0, 0, "zero"),
}


def daily(r):
    from fxlab.validation.metrics import daily as _d
    return _d(r)


def sr(d):
    return float(d.mean() / d.std() * np.sqrt(260)) if d.std() > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--families-screened", type=int, default=66)
    ap.add_argument("--tf", default=None)
    a = ap.parse_args()
    con = connect(read_only=True)
    rows = con.execute("SELECT id, symbol, tf, config, artifacts FROM experiments WHERE status='ok' AND target=? AND "
                       "campaign IN (%s)" % ",".join("?" * len(a.names)), [a.target] + a.names).fetchall()
    con.close()
    if a.tf:
        rows = [r for r in rows if r[2] == a.tf]
    streams = {name: {} for name in list(LADDER) + ["passive_limit (0.35bp comm)"]}
    per = []
    for eid, sym, tf, cfg, art in rows:
        cfg, art = json.loads(cfg), json.loads(art)
        if (cfg.get("exec") or {}).get("type") == "passive":
            continue
        x = pd.read_parquet(art["oos_returns"])
        b = cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))
        side = x["side"].reindex(b.index)
        oos = side.notna()
        pos = side.fillna(0.0)
        pip = instrument(sym).pip
        row = {"symbol": sym, "tf": tf}
        for name, cm in LADDER.items():
            r = vector_backtest(b, pos, pip, cm)["net"][oos]
            streams[name][f"{sym}_{tf}"] = daily(r)
            row[name] = sr(daily(r))
        rp = passive_backtest(b, pos, pip, CostModel())
        streams["passive_limit (0.35bp comm)"][f"{sym}_{tf}"] = daily(rp["net"][oos])
        row["passive_limit (0.35bp comm)"] = sr(daily(rp["net"][oos]))
        row["fill_rate"] = rp.attrs["fill_rate"]
        per.append(row)
    P = pd.DataFrame(per).sort_values("symbol")
    out = {"target": a.target, "instruments": int(len(P))}
    port = {}
    for name, S in streams.items():
        D = pd.DataFrame(S)                                          # NaN outside each stream's own OOS span
        vol = D.rolling(60, min_periods=20, ).std().shift(1)        # equal-risk weights from past data only
        W = (1 / vol.where(vol > 1e-7)).where(D.notna())             # only streams that exist and have risk
        W = W.div(W.sum(axis=1), axis=0).fillna(0.0)
        pr = (W * D.fillna(0.0)).sum(axis=1)
        pr = pr[pr.index >= D.index[60]]
        port[name] = pr
        est, lo, hi = sharpe_ci(pr.to_numpy(), n_boot=1000)
        t = st.ttest_1samp(pr, 0.0, alternative="greater")
        out[name] = {"portfolio_sharpe": sr(pr), "ci95": [lo, hi], "ann_return": float(pr.mean() * 260),
                     "p_one_sided": float(t.pvalue), "p_bonferroni_families": float(min(1, t.pvalue * a.families_screened)),
                     "instruments_positive": int((P[name] > 0).sum())}
    yr = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in port.items()})
    out["by_year_zero_cost"] = yr["zero_cost (diagnostic)"].round(4).to_dict()
    out["by_year_retail"] = yr["retail_ecn (spread+slip+0.35bp comm)"].round(4).to_dict()
    dest = RESULTS_DIR / "reports" / (f"deep_dive_{a.target}" + (f"_{a.tf}" if a.tf else ""))
    dest.mkdir(parents=True, exist_ok=True)
    P.to_csv(dest / "per_instrument.csv", index=False)
    yr.to_csv(dest / "portfolio_by_year.csv")
    (dest / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(P.round(2).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("by_year")}, indent=1, default=str))
    print(yr.round(4).to_string())


if __name__ == "__main__":
    main()
