"""E21 of docs/edges_batch_preregistration.md: replication of the strategy catalog on 2019-2025.
Families: the 10 with a positive 2007-2018 zero-cost Sharpe among the top rows of docs/research_report.md section 4
(S.VOL.EXPANSION, S.RG.CHANGE_POINT, S.BO.RANGE_EXPANSION, S.XA.STAT_ARB, S.TREND.REGRESSION, S.XA.RELATIVE_VALUE,
S.XA.GOLD_FX, S.GEO.HARMONIC, S.SEAS.WEEKEND_GAP_FADE, S.BO.VOL_BREAKOUT), plus the gold London breakout (I9) and
the night fade (I7) of scripts/untested_intraday_lab.py.
Catalog families: the project's walk-forward exactly as the screen campaigns (scripts/campaign.py defaults: 6
candidate settings, 3 training years, 6-month test blocks, market entries, flat financing, panel of the 14
campaign instruments), H1 and H4, on 11 FX pairs and gold, 2007-2025; only the 2019-2025 test blocks are scored.
I7 / I9: the fixed rules, scored on 2019-2025.
Counts: daily net mean > 0 with Newey-West t >= 1.645 on 2019-2025; Benjamini-Hochberg over E21 (q <= 0.10).
Usage: python scripts/catalog_replication_lab.py --family S.VOL.EXPANSION --out DIR
       python scripts/catalog_replication_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILIES = ["S.VOL.EXPANSION", "S.RG.CHANGE_POINT", "S.BO.RANGE_EXPANSION", "S.XA.STAT_ARB", "S.TREND.REGRESSION",
            "S.XA.RELATIVE_VALUE", "S.XA.GOLD_FX", "S.GEO.HARMONIC", "S.SEAS.WEEKEND_GAP_FADE", "S.BO.VOL_BREAKOUT"]
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "EURGBP", "AUDNZD",
           "GBPCAD", "XAUUSD"]
PANEL = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY",
         "XAUUSD", "XAGUSD", "EURCHF", "AUDJPY"]
START = pd.Timestamp("2019-01-01", tz="UTC")


def nw_t(y, lags=5):
    y = np.asarray(pd.Series(y).dropna(), float)
    n = len(y)
    if n < 30:
        return np.nan
    e = y - y.mean()
    s = (e @ e) / n
    s0 = s
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    s = max(s, s0)
    return float(y.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def score(fam, sym, tf, r):
    r = pd.Series(r, dtype=float)
    r.index = pd.DatetimeIndex(r.index)
    if r.index.tz is None:
        r.index = r.index.tz_localize("UTC")
    d = r.groupby(r.index.normalize()).sum()
    x = d[d.index >= START]
    pre = d[d.index < START]
    return {"family": fam, "symbol": sym, "tf": tf, "days": int(len(x)), "net_bp_day": float(x.mean() * 1e4) if len(x) else np.nan,
            "t": nw_t(x), "sharpe": float(x.mean() / x.std() * math.sqrt(252)) if len(x) > 2 and x.std() > 0 else np.nan,
            "active_days": int((x != 0).sum()), "pre2019_sharpe": float(pre.mean() / pre.std() * math.sqrt(252))
            if len(pre) > 2 and pre.std() > 0 else np.nan}


def run_family(fam, out):
    from fxlab.orchestration.runners import strategy_wf
    rows = []
    for tf in ("H1", "H4"):
        for sym in SYMBOLS:
            t0 = time.time()
            cfg = {"kind": "strategy", "target": fam, "symbol": sym, "tf": tf, "start": "2007-01-01", "end": "2025-12-31",
                   "n_trials": 6, "train_years": 3, "test_months": 6, "seed": 0, "panel": [p for p in PANEL if p != sym]}
            wd = Path(out) / "wf" / fam / f"{sym}_{tf}"
            try:
                res = strategy_wf(cfg, wd)
                r = pd.read_parquet(res["artifacts"]["oos_returns"])["ret"]
                rows.append(score(fam, sym, tf, r) | {"seconds": round(time.time() - t0, 1)})
            except Exception as ex:                                         # noqa: BLE001
                rows.append({"family": fam, "symbol": sym, "tf": tf, "error": repr(ex)[:300],
                             "trace": traceback.format_exc()[-600:]})
            print(rows[-1], flush=True)
            shutil.rmtree(wd, ignore_errors=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"rows_{fam}.csv"), index=False)


def run_lit(out):
    import untested_intraday_lab as U
    rows = []
    res = {}
    try:
        U.gold_london(res)
        for k, v in res.items():
            if isinstance(v, tuple):
                f = v[0]
                rows.append(score("I9 gold London breakout", "XAUUSD", "M1", f.net) | {"note": k})
    except Exception as ex:                                                 # noqa: BLE001
        rows.append({"family": "I9 gold London breakout", "error": repr(ex)[:300]})
    for s in ("AUDNZD", "EURGBP", "GBPCAD"):
        try:
            f = U.night_scalp(s)
            if f is not None:
                rows.append(score("I7 night fade", s, "M5", f.net))
        except Exception as ex:                                             # noqa: BLE001
            rows.append({"family": "I7 night fade", "symbol": s, "error": repr(ex)[:300]})
    pd.DataFrame(rows).to_csv(os.path.join(out, "rows_lit.csv"), index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)


def combine(d, out):
    from scipy import stats as st
    from statsmodels.stats.multitest import multipletests
    R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "**", "rows_*.csv"), recursive=True)
                   if os.path.getsize(f) > 5], ignore_index=True)
    R["p"] = st.norm.sf(R.t.astype(float))
    ok = R.p.notna()
    R.loc[ok, "bh_q"] = multipletests(R.p[ok], method="fdr_bh")[1]
    R["counts"] = (R.net_bp_day > 0) & (R.t >= 1.645) & (R.bh_q <= 0.10)
    os.makedirs(out, exist_ok=True)
    R.to_csv(os.path.join(out, "e21.csv"), index=False)
    pd.set_option("display.width", 220)
    print(R.groupby(["family", "tf"]).agg(n=("t", "count"), median_sharpe=("sharpe", "median"),
                                         positive=("net_bp_day", lambda x: int((x > 0).sum())),
                                         t_above_1645=("t", lambda x: int((x >= 1.645).sum()))).to_string())
    print(R.sort_values("t", ascending=False).head(15).round(3).to_string())
    print("counts:", R[R.counts].to_dict("records"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.combine:
        combine(a.combine, a.out)
    elif a.family == "lit":
        run_lit(a.out)
    else:
        run_family(a.family, a.out)


if __name__ == "__main__":
    main()
