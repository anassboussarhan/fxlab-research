"""Experiment orchestration: deterministic ids, a DuckDB results database,
checkpoint/resume, failure isolation with full tracebacks, resource monitoring
and progress reporting.

Every experiment is a JSON-serialisable config. Its id is the SHA-1 of the
canonical JSON, so re-running a manifest skips experiments that already have an
``ok`` record (resume) and re-runs ``failed`` ones only when asked."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fxlab.config import RESULTS_DIR, ROOT

DB_PATH = RESULTS_DIR / "results.duckdb"
ART_DIR = RESULTS_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)


def exp_id(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def connect(path: Path | None = None, read_only: bool = False, retries: int = 600):
    """Open the results DB (path resolved at call time). Writers hold the file lock only briefly, so
    connection attempts retry while another process is writing."""
    p = str(path or DB_PATH)
    last = None
    for _ in range(retries):
        try:
            con = duckdb.connect(p, read_only=read_only and Path(p).exists())
            break
        except (duckdb.IOException, duckdb.Error) as e:  # lock held by another process
            last = e
            time.sleep(0.2)
    else:
        raise last
    if read_only and Path(p).exists():
        return con
    con.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id VARCHAR PRIMARY KEY, campaign VARCHAR, kind VARCHAR, target VARCHAR, symbol VARCHAR, tf VARCHAR,
        config JSON, status VARCHAR, started TIMESTAMP, finished TIMESTAMP, seconds DOUBLE, error VARCHAR,
        traceback VARCHAR, metrics JSON, artifacts JSON, code_version VARCHAR, host VARCHAR, peak_rss_mb DOUBLE)""")
    con.execute("""CREATE TABLE IF NOT EXISTS folds (
        id VARCHAR, fold INTEGER, train_start TIMESTAMP, train_end TIMESTAMP, test_start TIMESTAMP,
        test_end TIMESTAMP, selected JSON, train_metrics JSON, test_metrics JSON)""")
    return con


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def system_snapshot() -> dict:
    snap = {"cpus": os.cpu_count(), "host": platform.node()}
    try:
        mem = Path("/proc/meminfo").read_text().split("\n")
        d = {l.split(":")[0]: int(l.split()[1]) for l in mem if l}
        snap["mem_total_mb"] = d["MemTotal"] / 1024
        snap["mem_avail_mb"] = d["MemAvailable"] / 1024
        snap["load1"] = os.getloadavg()[0]
        st = os.statvfs(str(RESULTS_DIR))
        snap["disk_free_gb"] = st.f_bavail * st.f_frsize / 1e9
    except Exception:
        pass
    return snap


def _worker(task: tuple) -> dict:
    """Runs in a separate process: failure of one experiment cannot kill the campaign."""
    fn_path, cfg, eid = task
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    mod, name = fn_path.rsplit(".", 1)
    t0 = time.time()
    rec = {"id": eid, "started": datetime.now(timezone.utc), "config": cfg}
    try:
        import importlib
        fn = getattr(importlib.import_module(mod), name)
        out = fn(cfg, ART_DIR / eid)
        rec.update(status="ok", metrics=out.get("metrics", {}), artifacts=out.get("artifacts", {}),
                   folds=out.get("folds", []))
    except Exception as e:
        rec.update(status="failed", error=repr(e)[:2000], traceback=traceback.format_exc()[-20000:])
    rec["finished"] = datetime.now(timezone.utc)
    rec["seconds"] = time.time() - t0
    rec["peak_rss_mb"] = peak_rss_mb()
    return rec


def _to_json(x):
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        return str(o)
    return json.dumps(x, default=default, allow_nan=True).replace("NaN", "null").replace("Infinity", "1e308")


def write_record(con, rec: dict, campaign: str):
    cfg = rec["config"]
    con.execute("DELETE FROM experiments WHERE id = ?", [rec["id"]])
    con.execute("DELETE FROM folds WHERE id = ?", [rec["id"]])
    con.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        rec["id"], campaign, cfg.get("kind"), cfg.get("target"), cfg.get("symbol"), cfg.get("tf"), _to_json(cfg),
        rec["status"], rec["started"], rec["finished"], rec["seconds"], rec.get("error"), rec.get("traceback"),
        _to_json(rec.get("metrics", {})), _to_json(rec.get("artifacts", {})), git_commit(), platform.node(),
        rec.get("peak_rss_mb")])
    for f in rec.get("folds", []) or []:
        con.execute("INSERT INTO folds VALUES (?,?,?,?,?,?,?,?,?)", [
            rec["id"], f["fold"], f.get("train_start"), f.get("train_end"), f.get("test_start"), f.get("test_end"),
            _to_json(f.get("selected")), _to_json(f.get("train_metrics")), _to_json(f.get("test_metrics"))])


def run_campaign(manifest: list[dict], runner: str, campaign: str, workers: int = 1, retry_failed: bool = False,
                 progress_every: int = 1, log_path: Path | None = None, time_budget_s: float | None = None) -> dict:
    """Execute every config in ``manifest`` with ``runner`` ('module.function', signature (cfg, outdir) -> dict).
    Completed experiments are skipped (resume). Returns a summary computed FROM THE DATABASE."""
    con = connect(retries=9000)
    done = {r[0]: r[1] for r in con.execute("SELECT id, status FROM experiments").fetchall()}
    con.close()
    todo, skipped = [], 0
    for cfg in manifest:
        eid = exp_id(cfg)
        st = done.get(eid)
        if st == "ok" or (st == "failed" and not retry_failed):
            skipped += 1
            continue
        todo.append((runner, cfg, eid))
    log = open(log_path, "a") if log_path else None
    t0 = time.time()
    n_done = n_fail = 0

    def report(rec):
        nonlocal n_done, n_fail
        c = connect(retries=9000)  # up to ~30 min: a long reader/writer elsewhere must not kill the campaign
        try:
            write_record(c, rec, campaign)
        finally:
            c.close()
        n_done += rec["status"] == "ok"
        n_fail += rec["status"] == "failed"
        k = n_done + n_fail
        if log and (k % progress_every == 0 or k == len(todo)):
            el = time.time() - t0
            rate = k / el if el > 0 else 0
            line = {"ts": datetime.now(timezone.utc).isoformat(), "campaign": campaign, "planned": len(manifest),
                    "skipped_already_done": skipped, "completed": n_done, "failed": n_fail,
                    "remaining": len(todo) - k, "exp_per_min": round(60 * rate, 2),
                    "eta_min": round((len(todo) - k) / rate / 60, 1) if rate > 0 else None,
                    "last": rec["id"], "last_status": rec["status"], "last_seconds": round(rec["seconds"], 1),
                    **system_snapshot()}
            log.write(json.dumps(line) + "\n")
            log.flush()

    if workers <= 1:
        for task in todo:
            if time_budget_s and time.time() - t0 > time_budget_s:
                break
            report(_worker(task))
    else:
        from concurrent.futures import wait, FIRST_COMPLETED
        with ProcessPoolExecutor(workers, max_tasks_per_child=20) as ex:
            it = iter(todo)
            inflight = set()
            while True:
                while len(inflight) < workers and not (time_budget_s and time.time() - t0 > time_budget_s):
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    inflight.add(ex.submit(_worker, nxt))
                if not inflight:
                    break
                done_, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for f in done_:
                    report(f.result())
    return summary(campaign)


def summary(campaign: str | None = None) -> dict:
    con = connect(read_only=True)
    q = "SELECT status, count(*) FROM experiments" + (" WHERE campaign = ?" if campaign else "") + " GROUP BY status"
    rows = con.execute(q, [campaign] if campaign else []).fetchall()
    con.close()
    return {s: n for s, n in rows}
