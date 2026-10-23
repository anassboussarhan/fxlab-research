"""Single machine-readable research catalog for features, strategies and models.

Counting rule (applies to all three kinds): one catalog entry = one genuinely
distinct algorithm. Lookback/threshold variations live in ``space`` and never
create a new entry. Aliases that are linear transforms of another entry (e.g.
Williams %R vs stochastic %K) are deliberately NOT registered.

Status is derived, never hand-written: ``status()`` combines (a) whether the
entry imports and has an executable callable, (b) declared hard requirements
(optional packages) and (c) the last automated test run recorded in
``results/catalog_test_status.json`` written by the test suite.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

from fxlab.config import RESULTS_DIR

TEST_STATUS_FILE = RESULTS_DIR / "catalog_test_status.json"


@dataclass
class Entry:
    id: str
    kind: str                      # feature | strategy | model
    family: str
    name: str
    spec: str                      # precise mathematical / algorithmic spec
    fn: Callable | None = field(repr=False, default=None)
    params: dict = field(default_factory=dict)       # defaults
    space: dict = field(default_factory=dict)        # search space
    inputs: tuple = ("close",)
    instruments: str = "any"       # any | fx | metals | panel(multi-asset)
    frequencies: str = "M1..D1"
    complexity: str = "O(n)"
    leakage: str = "trailing data only; value at bar t uses bars <= t"
    requires: tuple = ()           # optional python packages
    multi_asset: bool = False
    tags: tuple = ()
    extra: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        d = asdict(self)
        d.pop("fn")
        d["callable"] = f"{self.fn.__module__}.{self.fn.__qualname__}" if self.fn else None
        d["status"] = status(self)
        return d


CATALOG: dict[str, Entry] = {}


def register(entry: Entry) -> Entry:
    if entry.id in CATALOG:
        raise ValueError(f"duplicate catalog id {entry.id}")
    CATALOG[entry.id] = entry
    return entry


LEAKAGE_DEFAULT = {
    "feature": "trailing data only: value at bar t uses bars <= t (verified by the prefix-invariance test)",
    "strategy": "signal at bar t uses bars <= t (prefix-invariance test); executed at open t+1+latency by the simulator",
    "model": "fit only on the rows passed to fit (purged + embargoed walk-forward training block); prediction for "
             "row t uses rows <= t only (prediction-causality test); no test-block normalisation",
}


def _deco(kind: str):
    def outer(id: str, family: str, spec: str, **kw):
        kw.setdefault("leakage", LEAKAGE_DEFAULT[kind])
        def inner(fn):
            register(Entry(id=id, kind=kind, family=family, name=kw.pop("name", fn.__name__),
                           spec=spec, fn=fn, **kw))
            fn.catalog_id = id
            return fn
        return inner
    return outer


feature = _deco("feature")
strategy = _deco("strategy")
model = _deco("model")


def requirements_met(e: Entry) -> tuple[bool, list[str]]:
    missing = []
    for pkg in e.requires:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    return (not missing), missing


def load_test_status() -> dict:
    if TEST_STATUS_FILE.exists():
        return json.loads(TEST_STATUS_FILE.read_text())
    return {}


def status(e: Entry, test_status: dict | None = None) -> dict:
    ts = load_test_status() if test_status is None else test_status
    ok, missing = requirements_met(e)
    t = ts.get(e.id, {})
    if e.fn is None:
        state = "not_implemented"
    elif not ok:
        state = "blocked_missing_dependency"
    elif t.get("failed", 0) > 0:
        state = "implemented_tests_failing"
    elif t.get("passed", 0) > 0:
        state = "implemented_tested"
    else:
        state = "implemented_untested"
    return {"state": state, "missing": missing, "tests_passed": t.get("passed", 0),
            "tests_failed": t.get("failed", 0), "last_test_run": t.get("ts")}


def load_all() -> dict[str, Entry]:
    """Import every catalog module so all entries register."""
    import fxlab.features  # noqa: F401
    import fxlab.strategies  # noqa: F401
    import fxlab.models  # noqa: F401
    return CATALOG


def by_kind(kind: str) -> list[Entry]:
    return [e for e in CATALOG.values() if e.kind == kind]


def counts(test_status: dict | None = None) -> dict:
    ts = load_test_status() if test_status is None else test_status
    out: dict[str, Any] = {}
    for kind in ("feature", "strategy", "model"):
        es = by_kind(kind)
        states: dict[str, int] = {}
        for e in es:
            s = status(e, ts)["state"]
            states[s] = states.get(s, 0) + 1
        out[kind] = {"registered": len(es), "by_state": states,
                     "families": sorted({e.family for e in es})}
    return out
