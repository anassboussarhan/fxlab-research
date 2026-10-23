"""Export the research catalog (features, strategies, models) with machine-readable
implementation status to results/catalog.json and Markdown tables in docs/."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fxlab.features, fxlab.models, fxlab.strategies  # noqa: E402,F401
from fxlab.config import RESULTS_DIR, ROOT  # noqa: E402
from fxlab.registry import CATALOG, counts, load_test_status  # noqa: E402


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def main():
    ts = load_test_status()
    recs = [e.to_record() for e in CATALOG.values()]
    (RESULTS_DIR / "catalog.json").write_text(json.dumps({"counts": counts(ts), "entries": recs}, indent=1, default=str))
    for kind, fname in (("feature", "catalog_features.md"), ("strategy", "catalog_strategies.md"),
                        ("model", "catalog_models.md")):
        rows = sorted([r for r in recs if r["kind"] == kind], key=lambda r: r["id"])
        c = counts(ts)[kind]
        lines = [f"# {kind.capitalize()} catalog", "",
                 f"Registered: **{c['registered']}** genuinely distinct {kind} methods. Status counts: {c['by_state']}.",
                 "", "Counting rule: one entry = one distinct algorithm; lookback/threshold variants live in the "
                 "parameter space and are never counted separately. Status is derived from the automated test run "
                 "(results/catalog_test_status.json).", "",
                 "| id | family | specification | default params | search space | inputs | complexity | leakage control | status |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            st = r["status"]
            stxt = f"{st['state']} ({st['tests_passed']} pass / {st['tests_failed']} fail)"
            if st["missing"]:
                stxt += f" missing: {','.join(st['missing'])}"
            lines.append("| " + " | ".join(md_escape(x) for x in (
                r["id"], r["family"], r["spec"], json.dumps(r["params"], default=str),
                json.dumps(r["space"], default=str), ",".join(r["inputs"]) if r["inputs"] else "timestamps",
                r["complexity"], r["leakage"], stxt)) + " |")
        (ROOT / "docs" / fname).write_text("\n".join(lines) + "\n")
    print(json.dumps(counts(ts), indent=1))


if __name__ == "__main__":
    main()
