"""Cross-instrument FAMILY portfolios: for each (strategy family, timeframe, execution, financing) the OOS daily
net returns of all its instruments are averaged with equal weight (each instrument-stream is already a
walk-forward OOS record). A family that genuinely works should work on many instruments; averaging gives far
more statistical power than any single stream. One test per (family, timeframe, execution, financing);
Benjamini-Hochberg and Holm across all of them. Output: results/reports/family_portfolios/{table.csv,summary.json}"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.orchestration.campaign import connect  # noqa: E402
from fxlab.validation.metrics import daily  # noqa: E402
from fxlab.validation.stats import benjamini_hochberg, holm, sharpe_ci  # noqa: E402

OUT = RESULTS_DIR / "reports" / "family_portfolios"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = connect(read_only=True)
    rows = con.execute("SELECT target, symbol, tf, config, artifacts, metrics FROM experiments WHERE status='ok' "
                       "AND kind='strategy'").fetchall()
    con.close()
    groups = {}
    for tgt, sym, tf, cfg, art, m in rows:
        cfg, art = json.loads(cfg), json.loads(art)
        if cfg.get("risk_grid"):
            continue
        ex = "grid" if cfg.get("exec_grid") else (cfg.get("exec") or {}).get("type", "market")
        key = (tgt, tf, ex, cfg.get("financing", "flat"))
        p = art.get("oos_returns")
        if p and os.path.exists(p):
            groups.setdefault(key, {})[sym] = p          # one stream per instrument (latest wins)
    recs, series = [], {}
    for key, streams in groups.items():
        if len(streams) < 3:
            continue
        D = pd.DataFrame({s: daily(pd.read_parquet(p)["ret"]) for s, p in streams.items()})
        port = D.mean(axis=1, skipna=True).dropna()
        if len(port) < 250 or port.std() == 0:
            continue
        sr = port.mean() / port.std() * np.sqrt(260)
        t = port.mean() / port.std() * np.sqrt(len(port))
        per = (D.mean() / D.std() * np.sqrt(260))
        recs.append({"family": key[0], "tf": key[1], "exec": key[2], "financing": key[3], "instruments": len(streams),
                     "portfolio_sharpe": float(sr), "t": float(t), "p": float(st.t.sf(t, len(port) - 1)),
                     "median_instrument_sharpe": float(per.median()), "share_instruments_positive": float((per > 0).mean()),
                     "first_day": str(port.index[0].date()), "last_day": str(port.index[-1].date())})
        series["|".join(key)] = port
    T = pd.DataFrame(recs)
    if T.empty:
        (OUT / "summary.json").write_text(json.dumps({"tests": 0, "note": "no family with >= 3 instruments"}))
        print("no family with >= 3 instruments")
        return
    T["p_bh"] = benjamini_hochberg(T.p.to_numpy())
    T["p_holm"] = holm(T.p.to_numpy())
    T = T.sort_values("portfolio_sharpe", ascending=False)
    T.to_csv(OUT / "table.csv", index=False)
    pd.DataFrame(series).to_parquet(OUT / "portfolio_daily.parquet")
    res = {"tests": int(len(T)), "median_portfolio_sharpe": float(T.portfolio_sharpe.median()),
           "share_positive": float((T.portfolio_sharpe > 0).mean()), "bh10": int((T.p_bh < 0.10).sum()),
           "holm05": int((T.p_holm < 0.05).sum()), "top15": T.head(15).round(4).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print({k: v for k, v in res.items() if k != "top15"})
    print(T.head(20)[["family", "tf", "exec", "financing", "instruments", "portfolio_sharpe", "p", "p_bh",
                      "median_instrument_sharpe", "share_instruments_positive"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
