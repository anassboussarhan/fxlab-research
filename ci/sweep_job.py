"""Cloud sweep, step 3 (one job per instrument x timeframe, or per instrument for models).

--part strategies : every strategy family (model-driven ones in light mode) walk-forward on <pair, tf>;
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
    a = ap.parse_args()
    t0 = time.time()
    left = lambda: max(5.0, a.budget_min - (time.time() - t0) / 60)   # noqa: E731
    tag = a.tf if a.part == "strategies" else "models"
    out = os.path.join(ROOT, "out", f"res-{a.pair}-{tag}")
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "job.log")
    others = []
    if a.part == "models":
        name = f"cloud_models_{a.pair}"
        sh([PY, "scripts/campaign.py", "plan", "--name", name, "--kind", "model", "--symbols", a.pair, "--tfs", "H1",
            "--tasks", "clf", "vol", "pol", "--horizons", "1", "5", "20", "--start", START, "--end", END], log)
        sh([PY, "scripts/campaign.py", "run", "--name", name, "--workers", str(a.workers),
            "--time-budget-min", f"{left() - 15:.0f}"], log)
    else:
        if a.tf in FEATURE_TFS:
            others.append(popen([PY, "scripts/feature_scan.py", "--symbols", a.pair, "--tf", a.tf,
                                 "--start", START, "--end", END], log + ".features"))
        names = []
        fx_rates = a.pair[:3] in RATE_CCY and a.pair[3:] in RATE_CCY
        base = [PY, "scripts/campaign.py", "plan", "--kind", "strategy", "--symbols", a.pair, "--tfs", a.tf,
                "--ml-light", "--start", START, "--end", END]
        name = f"cloud_{a.tf}_{a.pair}"
        extra = ["--financing", "rates"] if (a.tf in ("H4", "D1") and fx_rates) else []
        sh(base[:3] + ["--name", name] + base[3:] + extra, log)
        names.append(name)
        if a.tf == "H1":
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
        if a.tf == "H1" and left() > 40:
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
