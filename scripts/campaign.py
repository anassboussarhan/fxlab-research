"""Plan and execute research campaigns (resumable).

  python scripts/campaign.py plan  --name screen_h1 --kind strategy --symbols EURUSD GBPUSD --tfs H1
  python scripts/campaign.py run   --name screen_h1 [--workers 2] [--time-budget-min 110] [--retry-failed]
  python scripts/campaign.py status --name screen_h1

A plan is written to results/manifests/<name>.json (the unified, machine-readable experiment manifest).
`run` skips experiments already recorded as ok (checkpoint/resume); failures keep their traceback and config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fxlab.features, fxlab.models, fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR, LOG_DIR  # noqa: E402
from fxlab.registry import by_kind, requirements_met  # noqa: E402
from fxlab.data.store import available, is_event_tf  # noqa: E402
from fxlab.orchestration.campaign import run_campaign, summary, exp_id, connect  # noqa: E402

RISK_GRID = [
    {"stop": "fixed", "stop_params": {"pips": 20}, "target": "affine", "target_params": {"a": 0, "b": 1.5}},
    {"stop": "atr", "stop_params": {"mult": 1.5}, "target": "affine", "target_params": {"a": 0, "b": 1.0}},
    {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine", "target_params": {"a": 0, "b": 2.0}},
    {"stop": "atr", "stop_params": {"mult": 3.0}, "target": "none", "target_params": {}, "trail_atr": 3.0},
    {"stop": "realized_vol", "stop_params": {"mult": 2.0}, "target": "power", "target_params": {"b": 1.5, "gamma": 0.8}},
    {"stop": "structural", "stop_params": {"k": 3}, "target": "log", "target_params": {"a": 0.5, "b": 2.0}},
    {"stop": "quantile", "stop_params": {"q": 0.9}, "target": "piecewise", "target_params": {"knots": [[1.5, 2.0], [3.0, 1.5]]}},
    {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "polynomial", "target_params": {"a": 0.5, "b": 0.3}},
    {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine", "target_params": {"a": 0, "b": 1.5}, "max_hold": 24},
    {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine", "target_params": {"a": 0, "b": 1.5}, "sizing": "vol_target"},
    {"stop": "atr", "stop_params": {"mult": 6.0}, "target": "none", "target_params": {}, "max_hold": 12},  # no tight stop: catastrophic stop + time exit
    {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine", "target_params": {"a": 0, "b": 3.0},
     "breakeven_r": 1.0},                                                       # stop to breakeven after +1R
    {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "none", "target_params": {}, "trail_atr": 3.0,
     "partial_r": 1.0, "partial_frac": 0.5, "breakeven_r": 1.0},               # half off at +1R, trail the rest
]

MAN_DIR = RESULTS_DIR / "manifests"
MAN_DIR.mkdir(parents=True, exist_ok=True)
PANEL = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY",
         "XAUUSD", "XAGUSD", "EURCHF", "AUDJPY"]


# entry order types searched per fold (exits are market orders); "market" keeps the baseline in the choice set
EXEC_GRID = [{"type": "order", "entry": "market"},
             {"type": "order", "entry": "limit_pullback", "k": 0.25, "ttl": 3},
             {"type": "order", "entry": "limit_pullback", "k": 0.5, "ttl": 6},
             {"type": "order", "entry": "stop_breakout", "k": 0.25, "ttl": 3},
             {"type": "order", "entry": "limit_touch", "k": 0.0, "ttl": 3},
             {"type": "confirm", "k": 1},                                          # wait for a confirming close
             {"type": "session", "tz": "Europe/London", "start": 7.0, "end": 17.0},  # London/NY-hours entries only
             {"type": "scale_in", "k": 0.5, "ttl": 6}]                             # half market, half pullback


def plan(a):
    man, skipped = [], []
    end = a.end
    if a.kind == "strategy":
        for sym in a.symbols:
            if not available(sym):
                skipped.append({"symbol": sym, "reason": "no data"})
                continue
            for tf in a.tfs:
                for e in sorted(by_kind("strategy"), key=lambda e: e.id):
                    ok, missing = requirements_met(e)
                    if not ok:
                        skipped.append({"target": e.id, "symbol": sym, "tf": tf, "reason": f"blocked: {missing}"})
                        continue
                    if a.only and e.id not in a.only:
                        continue
                    lo, hi = e.frequencies.split("..") if hasattr(e, "frequencies") and ".." in e.frequencies else ("M1", "D1")
                    order = ["M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1"]
                    tfo = "H1" if is_event_tf(tf) else tf   # adaptive event bars average ~1 hour (TICK60 etc.)
                    if not (order.index(lo) <= order.index(tfo) <= order.index(hi)):
                        skipped.append({"target": e.id, "symbol": sym, "tf": tf, "reason": "timeframe outside family scope"})
                        continue
                    if a.risk_grid and e.extra.get("ml"):
                        continue  # model-driven families: risk grid evaluated separately (cost of retraining)
                    cfg = {"kind": "strategy", "target": e.id, "symbol": sym, "tf": tf, "start": a.start,
                           "end": end, "n_trials": a.n_trials, "train_years": a.train_years,
                           "test_months": a.test_months, "seed": a.seed, "panel": [p for p in PANEL if p != sym]}
                    if e.extra.get("ml") and a.ml_light:
                        cfg["n_trials"] = 3
                        # ~2-3 years of training data and a few months of test per refit, whatever the timeframe
                        cfg["fixed_params"] = {"D1": {"train": 750, "test": 125}, "H8": {"train": 2000, "test": 400},
                                               "H4": {"train": 4000, "test": 750}}.get(tf, {"train": 4000, "test": 2000})
                    if a.exec == "passive":
                        if e.extra.get("mode", "state") != "state":
                            skipped.append({"target": e.id, "symbol": sym, "tf": tf,
                                            "reason": "passive execution only for target-position families"})
                            continue
                        cfg["exec"] = {"type": "passive", "offset": 0.0, "max_wait": 0}
                    if a.exec == "grid":
                        if e.extra.get("mode", "state") != "state":
                            skipped.append({"target": e.id, "symbol": sym, "tf": tf,
                                            "reason": "entry-type grid only for target-position families"})
                            continue
                        cfg["exec_grid"] = EXEC_GRID
                    if a.risk_grid:
                        cfg["risk_grid"] = RISK_GRID
                        if e.extra.get("mode", "state") != "trade":
                            cfg["mode_override"] = "trade"
                    if a.optimizer:
                        cfg["optimizer"] = a.optimizer
                    if a.financing == "rates":
                        if not sym[:3] + sym[3:] or sym.startswith(("XAU", "XAG")):
                            skipped.append({"target": e.id, "symbol": sym, "tf": tf, "reason": "no rate data for metals"})
                            continue
                        cfg["financing"] = "rates"
                    man.append(cfg)
        runner = "fxlab.orchestration.runners.strategy_wf"
    else:
        for sym in a.symbols:
            for tf in a.tfs:
                for e in sorted(by_kind("model"), key=lambda e: e.id):
                    if a.only and e.id not in a.only:
                        continue
                    tasks = e.extra.get("tasks", ["clf", "reg"])
                    for task in [t for t in a.tasks if t in tasks]:
                        for h in a.horizons:
                            man.append({"kind": "model", "target": e.id, "symbol": sym, "tf": tf, "task": task, "h": h,
                                        "start": a.start, "end": end, "train": a.train, "test": a.test, "seed": a.seed})
        runner = "fxlab.orchestration.runners.model_wf"
    doc = {"name": a.name, "runner": runner, "n": len(man), "skipped": skipped, "experiments": man,
           "ids": [exp_id(c) for c in man]}
    (MAN_DIR / f"{a.name}.json").write_text(json.dumps(doc, indent=1))
    print(f"planned {len(man)} experiments ({len(skipped)} skipped) -> {MAN_DIR / (a.name + '.json')}")


def run(a):
    doc = json.loads((MAN_DIR / f"{a.name}.json").read_text())
    s = run_campaign(doc["experiments"], doc["runner"], a.name, workers=a.workers, retry_failed=a.retry_failed,
                     log_path=LOG_DIR / f"campaign_{a.name}.jsonl",
                     time_budget_s=a.time_budget_min * 60 if a.time_budget_min else None)
    print(json.dumps(status_dict(a.name), indent=1))


def status_dict(name):
    doc = json.loads((MAN_DIR / f"{name}.json").read_text())
    con = connect()
    rows = dict(con.execute("SELECT id, status FROM experiments WHERE campaign = ?", [name]).fetchall())
    con.close()
    st = {"planned": doc["n"], "skipped_at_planning": len(doc["skipped"]),
          "ok": sum(1 for i in doc["ids"] if rows.get(i) == "ok"),
          "failed": sum(1 for i in doc["ids"] if rows.get(i) == "failed")}
    st["not_yet_run"] = st["planned"] - st["ok"] - st["failed"]
    return st


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=["strategy", "model"], default="strategy")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--tfs", nargs="+", default=["H1"])
    p.add_argument("--only", nargs="*")
    p.add_argument("--start", default="2007-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--train-years", type=float, default=3)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--tasks", nargs="+", default=["clf"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    p.add_argument("--train", type=int, default=10000)
    p.add_argument("--test", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--risk-grid", action="store_true", help="compare risk configs (trade-mode families only)")
    p.add_argument("--optimizer", choices=["optuna"], default=None)
    p.add_argument("--financing", choices=["flat", "rates"], default="flat",
                   help="rates = overnight swaps from point-in-time interest-rate differentials (FX pairs only)")
    p.add_argument("--exec", choices=["market", "passive", "grid"], default="market",
                   help="grid = entry order type selected per fold on training data (EXEC_GRID)")
    p.add_argument("--ml-light", action="store_true", help="model-driven families: 3 candidates, 2000-bar test blocks")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--time-budget-min", type=float)
    s = sub.add_parser("status")
    s.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a)
    elif a.cmd == "run":
        run(a)
    else:
        print(json.dumps(status_dict(a.name), indent=1))


if __name__ == "__main__":
    main()
