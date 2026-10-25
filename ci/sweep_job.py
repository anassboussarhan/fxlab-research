"""Cloud sweep, step 3 (one job per instrument x timeframe, or per instrument for models).

--part strategies : every strategy family (model-driven ones in light mode) walk-forward on <pair, tf>;
                    with --joint (H1/H4/D1) also the joint per-fold optimisation of parameters x entry x exit;
                    on H1 additionally the walk-forward entry-type grid; on H4/D1 FX pairs with interest-rate data
                    use rate-differential financing. Runs in parallel with the all-feature scan (M30..D1) and, on
                    H1, the all-feature cost-aware ML model.
--part models     : every model family on <pair, H1>, tasks clf/vol/pol, horizons 1, 5, 20.
Everything stops gracefully at --budget-min. Output: out/res-<pair>-<tf|models>/ with
  results.duckdb (experiments + folds; artifact paths rewritten to daily/<id>.parquet),
  daily/<id>.parquet  (OOS net returns aggregated to FX trading days),
  feature_scan/<pair>_<tf>/, ml_allfeatures/<...>/ when produced."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
START, END = "2005-01-01", "2025-12-31"
FEATURE_TFS = {"M30", "H1", "H2", "H4", "H8", "D1"}
JOINT_TFS = {"H1", "H4", "D1"}
RATE_CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}


def sh(args, log, timeout=None):
    print("+", " ".join(args), flush=True)
    with open(log, "a") as f:
        return subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def popen(args, log):
    print("+ (bg)", " ".join(args), flush=True)
    return subprocess.Popen(args, cwd=ROOT, stdout=open(log, "a"), stderr=subprocess.STDOUT)


def export(out, pair, tf):
    """Copy the job's results DB and convert every OOS return stream to daily returns (small, portable)."""
    import duckdb
    import pandas as pd
    from fxlab.validation.metrics import daily, trading_day
    os.makedirs(os.path.join(out, "daily"), exist_ok=True)
    src = os.path.join(ROOT, "results", "results.duckdb")
    if not os.path.exists(src):
        return 0
    dst = os.path.join(out, "results.duckdb")
    shutil.copy(src, dst)
    con = duckdb.connect(dst)
    rows = con.execute("SELECT id, artifacts FROM experiments").fetchall()
    n = 0
    for eid, art in rows:
        art = json.loads(art) if art else {}
        p = art.get("oos_returns")
        if p and os.path.exists(p):
            x = pd.read_parquet(p)
            d = daily(x["ret"])
            d.index = pd.DatetimeIndex(d.index).tz_localize("UTC") + pd.Timedelta(hours=12)
            frame = d.to_frame("ret")
            if "side" in x:
                s = x["side"].groupby(trading_day(x.index)).mean()
                s.index = pd.DatetimeIndex(s.index).tz_localize("UTC") + pd.Timedelta(hours=12)
                frame["mean_side"] = s.reindex(frame.index)
            frame.astype("float32").to_parquet(os.path.join(out, "daily", f"{eid}.parquet"))
            art["oos_returns"] = f"daily/{eid}.parquet"
            art.pop("oos_trades", None)
            con.execute("UPDATE experiments SET artifacts = ? WHERE id = ?", [json.dumps(art), eid])
            n += 1
    con.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--part", choices=["strategies", "models"], default="strategies")
    ap.add_argument("--budget-min", type=float, default=320)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated strategy family ids (empty = all)")
    ap.add_argument("--no-features", action="store_true", help="skip the all-feature scan")
    ap.add_argument("--no-extras", action="store_true", help="skip the H1 entry-type grid and the all-feature ML")
    ap.add_argument("--joint", action="store_true",
                    help=f"on {sorted(JOINT_TFS)}: also run the JOINT per-fold optimisation of strategy parameters x "
                         "entry type x exit configuration for EVERY family (ignores --only)")
    ap.add_argument("--joint-trials", type=int, default=16)
    ap.add_argument("--intrabar", choices=["stopfirst", "m1"], default="stopfirst")
    ap.add_argument("--projections-only", action="store_true",
                    help="run only scripts/projection_lab.py (non-random projection search) for <pair, tf>")
    ap.add_argument("--projections-set", default="", help="wave2 = only the second-wave projections")
    ap.add_argument("--random-entry-opt-only", action="store_true",
                    help="run only scripts/random_entry_opt.py (optimised management of random entries)")
    ap.add_argument("--flows-only", action="store_true",
                    help="run only scripts/flows_lab.py (flow, calendar and option-implied effects)")
    ap.add_argument("--random-entry-pool-only", action="store_true",
                    help="run only scripts/random_entry_pool.py (shared candidate set for pooled selection)")
    ap.add_argument("--reo-families", default="", help="families for --random-entry-opt-only (default: the script's)")
    ap.add_argument("--reo-tag", default="", help="output-file suffix for --random-entry-opt-only")
    ap.add_argument("--edge-harvest-opt-only", action="store_true",
                    help="run only scripts/edge_harvest_opt.py (TPE-optimised trade management per fold)")
    ap.add_argument("--edge-harvest-only", action="store_true",
                    help="run only scripts/edge_harvest.py (composite forecast x entry/exit/sizing/pyramid/passive grid)")
    ap.add_argument("--feature-ic-only", action="store_true",
                    help="run only scripts/feature_ic_lab.py (predictive information of every feature column)")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x.strip()]
    t0 = time.time()
    left = lambda: max(5.0, a.budget_min - (time.time() - t0) / 60)   # noqa: E731
    tag = a.tf if a.part == "strategies" else "models"
    out = os.path.join(ROOT, "out", f"res-{a.pair}-{tag}")
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "job.log")
    others = []
    if a.random_entry_opt_only:
        extra = (["--families", a.reo_families] if a.reo_families else []) + (["--tag", a.reo_tag] if a.reo_tag else [])
        sh([PY, "scripts/random_entry_opt.py", "--pair", a.pair, "--tf", a.tf, "--start", START, "--end", END,
            "--out", os.path.join(out, "random_entry_opt")] + extra, log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "random_entry_opt", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.flows_only:
        sh([PY, "scripts/flows_lab.py", "--pair", a.pair, "--start", START, "--end", END,
            "--out", os.path.join(out, "flows")], log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "flows", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.random_entry_pool_only:
        sh([PY, "scripts/random_entry_pool.py", "--pair", a.pair, "--tf", a.tf, "--start", START, "--end", END,
            "--out", os.path.join(out, "random_entry_pool")], log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "random_entry_pool", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.edge_harvest_opt_only:
        sh([PY, "scripts/edge_harvest_opt.py", "--pair", a.pair, "--tf", a.tf, "--start", START, "--end", END,
            "--out", os.path.join(out, "edge_harvest_opt")], log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "edge_harvest_opt", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.edge_harvest_only:
        sh([PY, "scripts/edge_harvest.py", "--pair", a.pair, "--tf", a.tf, "--start", START, "--end", END,
            "--out", os.path.join(out, "edge_harvest")], log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "edge_harvest", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.feature_ic_only:
        sh([PY, "scripts/feature_ic_lab.py", "--pair", a.pair, "--tf", a.tf, "--start", START, "--end", END,
            "--out", os.path.join(out, "feature_ic")], log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "feature_ic", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.projections_only:
        if a.projections_set:
            os.environ["PL_SET"] = a.projections_set
        sh([PY, "scripts/projection_lab.py", "--pair", a.pair, "--tf", a.tf, "--out", os.path.join(out, "projections")],
           log, timeout=int((a.budget_min - 5) * 60))
        json.dump({"pair": a.pair, "tf": a.tf, "part": "projections", "streams_exported": 0,
                   "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
        return
    if a.part == "models":
        name = f"cloud_models_{a.pair}"
        sh([PY, "scripts/campaign.py", "plan", "--name", name, "--kind", "model", "--symbols", a.pair, "--tfs", "H1",
            "--tasks", "clf", "vol", "pol", "--horizons", "1", "5", "20", "--start", START, "--end", END], log)
        sh([PY, "scripts/campaign.py", "run", "--name", name, "--workers", str(a.workers),
            "--time-budget-min", f"{left() - 15:.0f}"], log)
    else:
        if a.tf in FEATURE_TFS and not a.no_features:
            others.append(popen([PY, "scripts/feature_scan.py", "--symbols", a.pair, "--tf", a.tf,
                                 "--start", START, "--end", END], log + ".features"))
        names = []
        fx_rates = a.pair[:3] in RATE_CCY and a.pair[3:] in RATE_CCY
        base0 = [PY, "scripts/campaign.py", "plan", "--kind", "strategy", "--symbols", a.pair, "--tfs", a.tf,
                 "--ml-light", "--start", START, "--end", END]
        base = base0 + (["--only"] + only if only else [])
        name = f"cloud_{a.tf}_{a.pair}"
        extra = (["--financing", "rates"] if (a.tf in ("H4", "D1") and fx_rates) else []) + ["--intrabar", a.intrabar]
        sh(base[:3] + ["--name", name] + base[3:] + extra, log)
        names.append(name)
        if a.joint and a.tf in JOINT_TFS:
            jn = f"cloud_joint_{a.tf}_{a.pair}"
            sh(base0[:3] + ["--name", jn] + base0[3:] + extra + ["--joint", "--joint-trials", str(a.joint_trials)], log)
            names.append(jn)
        if a.tf == "H1" and not a.no_extras:
            g = f"cloud_H1grid_{a.pair}"
            sh(base[:3] + ["--name", g] + base[3:] + ["--exec", "grid"], log)
            names.append(g)
        # slow model-driven families last, so a budget stop costs the least information
        sh([PY, "-c", "import json,sys\nSLOW={'S.ML.RL_POLICY':3,'S.RG.HMM':2,'S.ML.TRIPLE_BARRIER':2}\n"
            "for n in sys.argv[1:]:\n p=f'results/manifests/{n}.json'; d=json.load(open(p))\n"
            " z=sorted(zip(d['experiments'],d['ids']),key=lambda x:(SLOW.get(x[0]['target'],0),x[0]['target'].startswith('S.ML')))\n"
            " d['experiments'],d['ids']=[x for x,_ in z],[y for _,y in z]; json.dump(d,open(p,'w'))"] + names, log)
        for n in names:
            sh([PY, "scripts/campaign.py", "run", "--name", n, "--workers", str(a.workers),
                "--time-budget-min", f"{left() - 15:.0f}"], log)
        if a.tf == "H1" and left() > 40 and not a.no_extras:
            sh([PY, "scripts/ml_allfeatures.py", "--symbols", a.pair, "--tf", "H1", "--horizons", "4", "24",
                "--start", START, "--end", END], log, timeout=int((left() - 10) * 60))
    for p in others:                        # the feature scan gets whatever time is left, then is stopped
        try:
            p.wait(timeout=max(60, (left() - 8) * 60))
        except subprocess.TimeoutExpired:
            p.kill()
            print("feature scan stopped at the time budget", flush=True)
    n = export(out, a.pair, tag)
    fs = os.path.join(ROOT, "results", "reports", "feature_scan", f"{a.pair}_{a.tf}")
    if os.path.exists(os.path.join(fs, "summary.csv")):
        shutil.copytree(fs, os.path.join(out, "feature_scan", f"{a.pair}_{a.tf}"), dirs_exist_ok=True)
    ml = os.path.join(ROOT, "results", "reports", "ml_allfeatures")
    if os.path.isdir(ml):
        shutil.copytree(ml, os.path.join(out, "ml_allfeatures"), dirs_exist_ok=True)
    json.dump({"pair": a.pair, "tf": a.tf, "part": a.part, "streams_exported": n,
               "minutes": round((time.time() - t0) / 60, 1)}, open(os.path.join(out, "job.json"), "w"))
    print(f"exported {n} streams in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
