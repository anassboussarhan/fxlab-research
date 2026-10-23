"""Leakage validation report built from the automated test record (results/catalog_test_status.json)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fxlab.features, fxlab.models, fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR, ROOT  # noqa: E402
from fxlab.registry import CATALOG, load_test_status, requirements_met  # noqa: E402

TEST_FOR = {"feature": "test_feature_contract", "strategy": "test_strategy_contract", "model": "test_model_contract"}
WHAT = {"feature": "prefix-invariance of feature values (real EURUSD H1, 2 cut points)",
        "strategy": "prefix-invariance of trading signals (real EURUSD H1 + panel)",
        "model": "prediction causality (prefix of test block reproduces outputs) on real EURUSD H1 features"}


def main():
    ts = load_test_status()
    lines = ["# Leakage validation report", "",
             "Every catalog entry is run through an automated causality test on real data. A value or signal at "
             "bar t that changes when bars after t are deleted is reported as lookahead.", ""]
    summary = {}
    for kind in ("feature", "strategy", "model"):
        ids = sorted(e.id for e in CATALOG.values() if e.kind == kind)
        res = {"passed": [], "failed": [], "not_run": [], "blocked": []}
        for i in ids:
            if not requirements_met(CATALOG[i])[0]:
                res["blocked"].append(i)
                continue
            tests = [t for t in ts.get(i, {}).get("tests", []) if TEST_FOR[kind] in t]
            if not tests:
                res["not_run"].append(i)
            elif all(t.endswith(":passed") for t in tests):
                res["passed"].append(i)
            else:
                res["failed"].append(i)
        summary[kind] = {k: len(v) for k, v in res.items()}
        lines += [f"## {kind.capitalize()}s - {WHAT[kind]}", "",
                  f"passed {len(res['passed'])} / failed {len(res['failed'])} / not run {len(res['not_run'])} / "
                  f"blocked (missing data) {len(res['blocked'])}", ""]
        for k in ("failed", "not_run", "blocked"):
            if res[k]:
                lines.append(f"* {k}: {', '.join(res[k])}")
        lines.append("")
    lines += ["## Additional structural controls", "",
              "* Labels expose `label_end`; walk-forward and purged k-fold splits purge overlapping labels and embargo "
              "the test boundary (`tests/test_validation.py`).",
              "* Every loader refuses holdout timestamps (>= 2026-01-01); the store holds no 2026 data.",
              "* Model-driven strategies train inside a purged walk-forward; signals before the first test block are flat.",
              "* Issues found during development and fixed: test-block normalisation in time-series "
              "model probabilities and in the generic signal mapper (caught by the model causality test), "
              "contemporaneous exogenous regressors in SARIMAX (found in code review; now lagged one bar), and an incomplete final "
              "walk-forward block in model-driven strategies (caught by the strategy causality test)."]
    (ROOT / "docs" / "leakage_report.md").write_text("\n".join(lines) + "\n")
    (RESULTS_DIR / "leakage_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
