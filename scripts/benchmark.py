"""Benchmark representative workloads on the current machine and extrapolate the
full research campaign's compute requirements. Writes results/benchmark.json."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fxlab.features, fxlab.models, fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR  # noqa: E402
from fxlab.registry import by_kind, CATALOG  # noqa: E402
from fxlab.data.store import load_m1, resample  # noqa: E402


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def strategies_only():
    """Re-time the strategy walk-forward runner with the current code; keeps the other sections."""
    from fxlab.orchestration.runners import strategy_wf
    from pathlib import Path
    out = json.loads((RESULTS_DIR / "benchmark.json").read_text())
    st = {}
    for sid in ("S.TREND.MA_CROSS", "S.MR.BOLLINGER", "S.BO.OPENING_RANGE", "S.RG.FRACTAL", "S.XA.CURRENCY_STRENGTH",
                "S.ML.SUPERVISED_DIRECTION"):
        cfg = {"kind": "strategy", "target": sid, "symbol": "EURUSD", "tf": "H1", "start": "2007-01-01",
               "end": "2018-12-31", "n_trials": 6, "train_years": 3, "test_months": 6, "seed": 0,
               "panel": ["GBPUSD", "XAUUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]}
        t = time.time()
        try:
            r = strategy_wf(cfg, Path(RESULTS_DIR / "artifacts" / "benchmark" / sid))
            st[sid] = {"seconds": round(time.time() - t, 1), "oos_sharpe": r["metrics"].get("sharpe")}
        except Exception as ex:
            st[sid] = {"seconds": round(time.time() - t, 1), "error": repr(ex)[:300]}
    out["strategy_wf_first_pass"] = out.get("strategy_wf")
    out["strategy_wf"] = st
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(st, indent=1))


def main():
    if "--strategies-only" in sys.argv:
        return strategies_only()
    out = {"machine": {"cpus": os.cpu_count(), "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
                       "gpu": "none"}}
    # 1. data loading / resampling
    t = time.time(); m1 = load_m1("EURUSD", "2007-01-01", "2018-12-31"); tl = time.time() - t
    res = {}
    for tf in ("M5", "M15", "H1", "H4"):
        t = time.time(); b = resample(m1, tf); res[tf] = {"seconds": round(time.time() - t, 2), "bars": len(b)}
    out["data"] = {"m1_rows": len(m1), "load_seconds": round(tl, 2), "resample": res, "peak_rss_mb": round(rss())}
    h1 = resample(m1, "H1")
    del m1
    # 2. features: every feature on 2 years of H1
    sub = h1.loc["2011-01-01":"2012-12-31"]
    panel = None
    from fxlab.orchestration.runners import load_panel_closes
    panel = load_panel_closes(["GBPUSD", "XAUUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "XAGUSD"], "H1",
                              sub.index, "2011-01-01", "2012-12-31")
    ft = {}
    for e in sorted(by_kind("feature"), key=lambda e: e.id):
        t = time.time()
        try:
            e.fn(sub, panel=panel, symbol="EURUSD", **e.params) if e.multi_asset else e.fn(sub, **e.params)
            ft[e.id] = round(time.time() - t, 3)
        except Exception as ex:
            ft[e.id] = f"error {ex!r}"[:200]
    secs = [v for v in ft.values() if isinstance(v, float)]
    out["features"] = {"bars": len(sub), "total_seconds": round(sum(secs), 1), "per_feature": ft,
                       "slowest": sorted(((v, k) for k, v in ft.items() if isinstance(v, float)), reverse=True)[:15]}
    # 3. models: fit+predict on 20k x 20
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12000, 20)); y = X[:, 1] * 0.3 + rng.normal(size=12000)
    mt = {}
    for e in sorted(by_kind("model"), key=lambda e: e.id):
        task = e.extra.get("tasks", ["clf"])[0]
        yy = (y > 0).astype(int) if task == "clf" else (np.abs(y) if task == "vol" else y)
        t = time.time()
        try:
            m = e.fn(task=task, **e.params)
            m.fit(X[:10000], yy[:10000])
            m.predict(X[10000:])
            mt[e.id] = round(time.time() - t, 2)
        except Exception as ex:
            mt[e.id] = f"error {ex!r}"[:200]
    ms = [v for v in mt.values() if isinstance(v, float)]
    out["models"] = {"rows": 10000, "cols": 20, "total_seconds": round(sum(ms), 1), "per_model": mt}
    # 4. strategy walk-forward runner on full EURUSD H1 history
    from fxlab.orchestration.runners import strategy_wf
    from pathlib import Path
    st = {}
    for sid in ("S.TREND.MA_CROSS", "S.MR.BOLLINGER", "S.BO.OPENING_RANGE", "S.RG.FRACTAL", "S.XA.CURRENCY_STRENGTH",
                "S.ML.SUPERVISED_DIRECTION"):
        cfg = {"kind": "strategy", "target": sid, "symbol": "EURUSD", "tf": "H1", "start": "2007-01-01",
               "end": "2018-12-31", "n_trials": 6, "train_years": 3, "test_months": 6, "seed": 0,
               "panel": ["GBPUSD", "XAUUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]}
        t = time.time()
        try:
            r = strategy_wf(cfg, Path(RESULTS_DIR / "artifacts" / "benchmark" / sid))
            st[sid] = {"seconds": round(time.time() - t, 1), "oos_sharpe": r["metrics"].get("sharpe")}
        except Exception as ex:
            st[sid] = {"seconds": round(time.time() - t, 1), "error": repr(ex)[:300], "tb": traceback.format_exc()[-1500:]}
    out["strategy_wf"] = st
    out["peak_rss_mb"] = round(rss())
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v if k != "features" else {kk: vv for kk, vv in v.items() if kk != "per_feature"})
                      for k, v in out.items() if k != "models"}, indent=1, default=str)[:4000])
    print("models total s", out["models"]["total_seconds"])


if __name__ == "__main__":
    main()
