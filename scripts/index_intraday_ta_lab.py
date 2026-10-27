"""The strategy catalog on intraday bars of the US500 and NAS100 CFDs (Dukascopy M1 bid/ask, 2019-2025, cash-session
hours 13-21 UTC), on M5, M15, M30, H1 and H4: the timeframes the daily cross-market run (scripts/cross_market_ta_lab.py)
could not cover. Development data only, to 2025-12-31.

FIXED BEFORE ANY RESULT:
  * Families: every single-instrument catalog family whose timeframe scope includes the bar size, except model-driven
    (S.ML.*), FX-panel (S.XA.*, S.TREND.XS_MOMENTUM) and FX-session or FX-calendar families (S.SES.*, S.SEAS.GOTOBI,
    S.SEAS.FIX_REVERSAL, S.SEAS.MONTH_END_FIX, S.EVT.GOLD_FIX, S.EVT.FOMC_USD, S.SMC.SILVER_BULLET,
    S.BO.OPENING_RANGE, which use London or Tokyo session times), and the daily-only S.DAILY.* families.
  * Walk-forward as in the FX sweep and the daily run: 3 years of training, 6-month test blocks, default parameters plus
    11 random draws of the search space chosen on training Sharpe. First test block: 2022.
  * Costs: the quoted Dukascopy bid/ask, 25% of the spread plus 0.1 point slippage a side, 0.35 bp commission a side;
    overnight financing 3-month USD rate + 2.5% a year per night on either side (conservative for shorts, which on a
    CFD would receive the rate).
PRE-REGISTERED TESTS (one-sided): out-of-sample alpha against holding the CFD over the same bars (Newey-West), per
  family x instrument x timeframe; Benjamini-Hochberg over all of them.
Usage: python scripts/index_intraday_ta_lab.py --symbol USA500IDXUSD --tf H1 --out DIR
       python scripts/index_intraday_ta_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_market_ta_lab as CM  # noqa: E402

END = "2025-12-31 23:59"
MARKUP = 0.025
SKIP_PREFIX = ("S.ML.", "S.XA.", "S.SES.", "S.DAILY.")
SKIP = {"S.TREND.XS_MOMENTUM", "S.SEAS.GOTOBI", "S.SEAS.FIX_REVERSAL", "S.SEAS.MONTH_END_FIX", "S.EVT.GOLD_FIX",
        "S.EVT.FOMC_USD", "S.SMC.SILVER_BULLET", "S.BO.OPENING_RANGE"}
ORDER = ["M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1"]


def families(tf):
    import fxlab.features, fxlab.models, fxlab.strategies  # noqa: F401
    from fxlab.registry import by_kind, requirements_met
    out = []
    for e in sorted(by_kind("strategy"), key=lambda e: e.id):
        lo, hi = getattr(e, "frequencies", "M1..D1").split("..")
        if not (ORDER.index(lo) <= ORDER.index(tf) <= ORDER.index(hi)):
            continue
        if e.id.startswith(SKIP_PREFIX) or e.id in SKIP or e.multi_asset:
            continue
        if requirements_met(e)[0]:
            out.append(e.id)
    return out


def install(symbol, tf):
    import fxlab.evaluate as EV
    import fxlab.orchestration.runners as RN
    from fxlab.data import instruments as INS
    from fxlab.data.store import load_m1, resample
    from fxlab.execution.costs import BASE
    from fxlab.execution.simulator import rollover_counts
    from fxlab.data import rates as R
    m1 = load_m1(symbol, "2019-01-01", END)
    bars = resample(m1, tf).dropna(subset=["bid_o", "ask_o"])
    pip = 0.1
    INS.INSTRUMENTS[symbol] = INS.Instrument(symbol, symbol, "USD", pip, 1.0, "index")
    RN.cached_bars = lambda s, t, start=None, end=None: bars
    rc = rollover_counts(bars.index)
    usd = R.known_at(pd.DatetimeIndex(bars.index.floor("D")))["USD"].to_numpy()
    mid_o = 0.5 * (bars["bid_o"] + bars["ask_o"]).to_numpy()
    # financing per bar = roll x swap_pips x pip / price  ->  nights x (USD rate + mark-up) / 360 of the notional
    roll = rc * (np.nan_to_num(usd) + MARKUP) / 360.0 * mid_o / (BASE.swap_pips_per_night * pip)
    idx = bars.index

    def rollovers(index):
        if len(index) == len(idx) and index[0] == idx[0] and index[-1] == idx[-1]:
            return roll.astype(float)
        pos = idx.get_indexer(index)
        return np.where(pos >= 0, roll[np.clip(pos, 0, None)], 0.0).astype(float)
    EV.rollover_counts = rollovers
    return bars


def run(a):
    from fxlab.orchestration.runners import strategy_wf
    from scipy import stats as st
    os.makedirs(a.out, exist_ok=True)
    bars = install(a.symbol, a.tf)
    mid_o = 0.5 * (bars["bid_o"] + bars["ask_o"])
    bh = (mid_o.shift(-1) / mid_o - 1).rename("bh")
    slow = ("S.RG.HMM", "S.RG.CHANGE_POINT", "S.SP.", "S.RG.", "S.FAMOUS.TD_SEQUENTIAL")
    fams = sorted(families(a.tf), key=lambda f: (f.startswith(slow), f))     # slow families last
    print(a.symbol, a.tf, len(bars), "bars", len(fams), "families", flush=True)
    rows = []
    for fid in fams:
        t0 = time.time()
        cfg = {"kind": "strategy", "target": fid, "symbol": a.symbol, "tf": a.tf, "start": None, "end": None,
               "n_trials": CM.N_TRIALS, "train_years": 3, "test_months": 6, "seed": 0, "panel": []}
        try:
            res = strategy_wf(cfg, __import__("pathlib").Path(a.out) / "wf" / fid)
            m = res["metrics"]
            r = pd.read_parquet(res["artifacts"]["oos_returns"])["ret"]
            x = bh.reindex(r.index).fillna(0.0)
            rd = r.groupby(r.index.normalize()).sum()
            xd = x.groupby(x.index.normalize()).sum()
            alpha, ta, beta = CM.nw_t(rd.to_numpy(), xd.reindex(rd.index).fillna(0).to_numpy())
            stress = m.get("stress", {}).get("cost_x2.0", {})
            rows.append({"symbol": a.symbol, "tf": a.tf, "family": fid, "days": int(len(rd)),
                         "sharpe": float(rd.mean() / rd.std() * math.sqrt(252)) if rd.std() > 0 else np.nan,
                         "alpha_ann": alpha * 252, "alpha_t": ta, "beta": beta,
                         "p_alpha": float(1 - st.norm.cdf(ta)) if np.isfinite(ta) else np.nan,
                         "trades": m.get("trades"),
                         "sharpe_cost_x2": stress.get("sharpe") if isinstance(stress, dict) else None,
                         "zero_cost_sharpe_default": m.get("diagnostic_zero_cost_default_sharpe"),
                         "seconds": round(time.time() - t0, 1)})
        except Exception as ex:                                           # noqa: BLE001
            rows.append({"symbol": a.symbol, "tf": a.tf, "family": fid, "error": repr(ex)[:300],
                         "trace": traceback.format_exc()[-600:]})
        print(fid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                    if k in ("sharpe", "alpha_t", "trades", "error", "seconds")}, flush=True)
        import shutil
        shutil.rmtree(os.path.join(a.out, "wf", fid), ignore_errors=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"rows_{a.symbol}_{a.tf}.csv"), index=False)  # after each family


def combine(a):
    fs = glob.glob(os.path.join(a.combine, "**", "rows_*.csv"), recursive=True)
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    ok = R[R["error"].isna()].copy() if "error" in R else R.copy()
    ok["q_bh"] = CM.bh_q(ok.p_alpha.to_numpy())
    os.makedirs(a.out, exist_ok=True)
    ok.sort_values("p_alpha").to_csv(os.path.join(a.out, "pairs.csv"), index=False)
    if "error" in R:
        R[R["error"].notna()].to_csv(os.path.join(a.out, "errors.csv"), index=False)
    by_tf = ok.groupby(["symbol", "tf"]).agg(families=("family", "size"), median_sharpe=("sharpe", "median"),
                                             share_alpha_t_gt2=("alpha_t", lambda x: (x > 2).mean()),
                                             share_alpha_t_lt_m2=("alpha_t", lambda x: (x < -2).mean()))
    by_tf.to_csv(os.path.join(a.out, "by_timeframe.csv"))
    pd.set_option("display.width", 250)
    print(by_tf.round(3).to_string())
    print(ok.sort_values("p_alpha").head(25)[["symbol", "tf", "family", "sharpe", "alpha_ann", "alpha_t", "q_bh",
                                             "sharpe_cost_x2"]].round(3).to_string())
    json.dump({"pairs": int(len(ok)), "errors": int(len(R) - len(ok)), "q10": int((ok.q_bh <= 0.10).sum())},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", default="")
    a = ap.parse_args()
    combine(a) if a.combine else run(a)


if __name__ == "__main__":
    main()
