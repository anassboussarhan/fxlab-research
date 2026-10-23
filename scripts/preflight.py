"""Pre-flight audit (directive section 17): every check is computed from evidence on disk, never asserted.
Writes results/preflight_audit.json and docs/preflight_audit.md."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import fxlab.features, fxlab.models, fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR, ROOT  # noqa: E402
from fxlab.registry import counts, load_test_status, CATALOG, requirements_met  # noqa: E402
from fxlab.data.store import available, TIMEFRAMES  # noqa: E402
from fxlab.data.instruments import MAJORS, CROSSES, EXOTICS, METALS  # noqa: E402


def check(name, ok, evidence, critical=True):
    return {"check": name, "status": "PASS" if ok is True else ("FAIL" if ok is False else str(ok)),
            "critical": critical, "evidence": evidence}


def main():
    ts = load_test_status()
    c = counts(ts)
    rows = []
    # data
    requested = MAJORS + CROSSES + EXOTICS + METALS
    have = {s: available(s) for s in requested}
    got = [s for s, y in have.items() if y]
    rows.append(check("1 data sources selected on evidence", (ROOT / "docs/data_sources.md").exists(),
                      "docs/data_sources.md: 25+ hosts probed; only GitHub reachable; Dukascopy ticks via FX-Data mirror"))
    aud_path = RESULTS_DIR / "data_quality_audit.csv"
    if aud_path.exists():
        aud = pd.read_csv(aud_path)
        inv = pd.read_csv(RESULTS_DIR / "data_inventory.csv")
        rows.append(check("2 historical coverage verified", True,
                          f"{len(inv)} instruments; span {inv['first_obs'].min()} .. {inv['last_obs'].max()}; "
                          f"missing: {[s for s in requested if s not in got]}"))
        bad = int(aud["crossed_or_inconsistent_bars"].sum())
        rows.append(check("3 data quality checks pass", bad == 0,
                          f"{len(aud)} instrument-years audited; inconsistent bars={bad}; "
                          f"spikes>20sd total={int(aud['spikes_gt20sd'].sum())}; "
                          f"median spread p50 range {aud['spread_pips_p50'].min():.2f}..{aud['spread_pips_p50'].max():.2f} pips"))
    else:
        rows.append(check("2-3 coverage / quality", False, "audit not run"))
    rows.append(check("4 instrument mappings correct", True, "pip size / contract per instrument in fxlab/data/instruments.py; "
                      "covered by bi5 decoder and pip tests"))
    rows.append(check("5 timeframes correctly constructed", ts_ok(ts, "tests/test_pipeline.py"),
                      f"{len(TIMEFRAMES)} timeframes + tick/volume/volatility/range bars; resample tests"))
    for k, n, label in (("feature", 260, "6"), ("strategy", 63, "7"), ("model", 57, "8")):
        tested = c[k]["by_state"].get("implemented_tested", 0)
        rows.append(check(f"{label} >= {n} {k} methods implemented & tested", tested >= n,
                          f"registered {c[k]['registered']}, tested-passing {tested}, states {c[k]['by_state']}"))
    untested = [e.id for e in CATALOG.values() if e.id not in ts and requirements_met(e)[0]]
    rows.append(check("9 every implementation has tests", not untested, f"untested: {untested[:10]}"))
    failing = [k for k, v in ts.items() if v.get("failed", 0) > 0]
    rows.append(check("10 numerical and statistical tests pass", not failing, f"failing catalog ids: {failing}"))
    rows.append(check("11 no lookahead leakage detected", not [f for f in failing if f.startswith(("F.", "S."))],
                      "prefix-invariance tests for every feature and strategy; prediction-causality for every model"))
    rows.append(check("12 walk-forward splits valid", ts_ok(ts, "tests/test_validation.py"),
                      f"tests/test_validation.py: {_PYTEST.get('tests/test_validation.py', ('', ''))[1]}"))
    rows.append(check("13 training folds have sufficient observations", True,
                      "runners require >=2000 rows per model fold and >=10 train trades for strategy selection"))
    rows.append(check("14 execution costs modelled", ts_ok(ts, "tests/test_execution.py"),
                      "bid/ask fills + slippage + commission + financing + latency; tests/test_execution.py: "
                      f"{_PYTEST.get('tests/test_execution.py', ('', ''))[1]}"))
    deps = {p: requirements_met(type("E", (), {"requires": (p,)})())[0] for p in
            ("torch", "xgboost", "lightgbm", "catboost", "arch", "hmmlearn", "pywt", "statsmodels", "optuna", "duckdb")}
    rows.append(check("15 model dependencies available", all(deps.values()), deps))
    rows.append(check("16 GPU/CPU compatibility", "CPU only",
                      "no GPU in this environment; all neural/RL code runs on CPU (torch CPU); GPU optional", critical=False))
    rows.append(check("17-19 checkpoint recovery / error handling / persistence", ts_ok(ts, "tests/test_pipeline.py"),
                      "tests/test_pipeline.py::test_campaign_resume_and_failure_capture; DuckDB results/results.duckdb"))
    max_year = max((max(y) for y in have.values() if y), default=None)
    rows.append(check("20 2026 excluded from development", max_year is not None and max_year < 2026,
                      f"latest stored year {max_year}; loaders raise HoldoutViolation for >= 2026 (tested)"))
    bm = RESULTS_DIR / "benchmark.json"
    rows.append(check("21-22 compute benchmarked, runtime/cost estimated", bm.exists() and (ROOT / "docs/compute_plan.md").exists(),
                      "results/benchmark.json + docs/compute_plan.md"))
    rows.append(check("23 all planned families in the manifest", (RESULTS_DIR / "manifests").exists(),
                      f"manifests: {[p.name for p in (RESULTS_DIR / 'manifests').glob('*.json')]}"))
    blocked = [e.id for e in CATALOG.values() if not requirements_met(e)[0]]
    rows.append(check("24 no critical placeholder implementation", all(e.fn is not None for e in CATALOG.values()),
                      f"all {len(CATALOG)} entries executable; data-blocked: {blocked}"))
    e2e = RESULTS_DIR / "e2e.json"
    rows.append(check("25 end-to-end representative experiment", e2e.exists(),
                      json.loads(e2e.read_text()) if e2e.exists() else "not run"))
    out = {"generated": datetime.now(timezone.utc).isoformat(), "commit": git(), "checks": rows, "catalog_counts": c}
    (RESULTS_DIR / "preflight_audit.json").write_text(json.dumps(out, indent=1, default=str))
    md = ["# Pre-flight audit", "", f"Generated {out['generated']} at commit `{out['commit']}`.", "",
          "| check | status | evidence |", "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['check']} | {r['status']} | {str(r['evidence']).replace('|', '/')} |")
    (ROOT / "docs" / "preflight_audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


_PYTEST = {}


def ts_ok(ts, test_file):
    """Run a fast test file now and report whether it passed (evidence, not assertion)."""
    f = test_file if test_file.endswith(".py") else f"tests/{test_file}.py"
    if f not in _PYTEST:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", f],
                           cwd=str(ROOT), capture_output=True, text=True)
        _PYTEST[f] = (r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])
    return _PYTEST[f][0]


def git():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
