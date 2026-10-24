"""Cloud sweep, step 4: merge every job's results and run the program-level statistics.

Input: in/res-*/ (downloaded artifacts). Builds results/results.duckdb, results/daily/, results/reports/
feature_scan|ml_allfeatures, then runs scripts/program_report.py, scripts/family_portfolios.py and
scripts/feature_scan_report.py, and writes compact summaries to sweep_results/ (committed to the `results`
branch; raw price data is never written there)."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable


def main():
    import duckdb
    import pandas as pd
    from fxlab.orchestration.campaign import connect
    res = os.path.join(ROOT, "results")
    daily_dir = os.path.join(res, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    db = os.path.join(res, "results.duckdb")
    if os.path.exists(db):
        os.remove(db)
    connect().close()                                       # creates the schema
    jobs = []
    for d in sorted(glob.glob(os.path.join(ROOT, "in", "res-*"))):
        info = json.load(open(os.path.join(d, "job.json"))) if os.path.exists(os.path.join(d, "job.json")) else {}
        info["dir"] = os.path.basename(d)
        jobs.append(info)
        src = os.path.join(d, "results.duckdb")
        if os.path.exists(src):
            c = duckdb.connect(src)
            for eid, art in c.execute("SELECT id, artifacts FROM experiments").fetchall():
                a = json.loads(art) if art else {}
                p = a.get("oos_returns")
                if p and not os.path.isabs(p) and os.path.exists(os.path.join(d, p)):
                    dst = os.path.join(daily_dir, os.path.basename(p))
                    shutil.copy(os.path.join(d, p), dst)
                    a["oos_returns"] = dst
                    c.execute("UPDATE experiments SET artifacts = ? WHERE id = ?", [json.dumps(a), eid])
            c.close()
            m = duckdb.connect(db)
            m.execute(f"ATTACH '{src}' AS s (READ_ONLY)")
            m.execute("INSERT INTO experiments SELECT * FROM s.experiments WHERE id NOT IN (SELECT id FROM experiments)")
            m.execute("INSERT INTO folds SELECT * FROM s.folds WHERE id NOT IN (SELECT DISTINCT id FROM folds)")
            m.execute("DETACH s")
            m.close()
        for x in glob.glob(os.path.join(d, "feature_ic", "*")):
            os.makedirs(os.path.join(res, "reports", "feature_ic"), exist_ok=True)
            shutil.copy(x, os.path.join(res, "reports", "feature_ic", os.path.basename(x)))
        for x in glob.glob(os.path.join(d, "projections", "*.json")):
            os.makedirs(os.path.join(res, "reports", "projection_lab"), exist_ok=True)
            shutil.copy(x, os.path.join(res, "reports", "projection_lab", os.path.basename(x)))
        for sub in ("feature_scan", "ml_allfeatures"):
            for x in glob.glob(os.path.join(d, sub, "*")):
                shutil.copytree(x, os.path.join(res, "reports", sub, os.path.basename(x)), dirs_exist_ok=True)
    out = os.path.join(ROOT, "sweep_results")
    os.makedirs(out, exist_ok=True)
    for script in ("program_report.py", "family_portfolios.py", "feature_scan_report.py", "joint_report.py",
                   "projection_report.py", "feature_ic_report.py"):
        with open(os.path.join(out, script.replace(".py", ".log")), "w") as f:
            subprocess.run([PY, os.path.join("scripts", script)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    for src, dst in (("results/reports/program", "program"), ("results/reports/family_portfolios", "family_portfolios"),
                     ("results/reports/joint", "joint"), ("results/reports/projection_lab_summary", "projections"),
                     ("results/reports/feature_ic_summary", "feature_ic")):
        if os.path.isdir(os.path.join(ROOT, src)):
            shutil.copytree(os.path.join(ROOT, src), os.path.join(out, dst), dirs_exist_ok=True)
    fs = os.path.join(ROOT, "results", "reports", "feature_scan")
    os.makedirs(os.path.join(out, "feature_scan"), exist_ok=True)
    for f in glob.glob(os.path.join(fs, "*.json")) + glob.glob(os.path.join(fs, "*.csv")):
        shutil.copy(f, os.path.join(out, "feature_scan", os.path.basename(f)))
    for f in glob.glob(os.path.join(fs, "*", "summary.csv")):
        shutil.copy(f, os.path.join(out, "feature_scan", os.path.basename(os.path.dirname(f)) + "_summary.csv"))
    # data coverage from the per-instrument audits and short job-log tails (for diagnosis; no price data)
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "audits", "audit_*.json"))):
        a = json.load(open(f))
        for x in a.get("audit", []):
            rows.append({"pair": a["pair"], **{k: v for k, v in x.items() if not isinstance(v, (dict, list))}})
        for y, v in a.get("years", {}).items():
            if "error" in v:
                rows.append({"pair": a["pair"], "year": y, "download_error": v["error"]})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "data_coverage.csv"), index=False)
    os.makedirs(os.path.join(out, "job_logs"), exist_ok=True)
    for d in sorted(glob.glob(os.path.join(ROOT, "in", "res-*"))):
        for name in ("job.log", "job.log.features"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                tail = open(p, errors="replace").read()[-4000:]
                open(os.path.join(out, "job_logs", f"{os.path.basename(d)}.{name}.tail.txt"), "w").write(tail)
    con = duckdb.connect(db, read_only=True)
    cov = con.execute("SELECT kind, tf, symbol, status, count(*) AS n FROM experiments GROUP BY ALL ORDER BY ALL").df()
    con.close()
    cov.to_csv(os.path.join(out, "coverage.csv"), index=False)
    pd.DataFrame(jobs).to_csv(os.path.join(out, "jobs.csv"), index=False)
    print(cov.groupby(["kind", "status"]).n.sum())


if __name__ == "__main__":
    main()
