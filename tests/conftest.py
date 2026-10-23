"""Shared fixtures + a recorder that maps every test to catalog ids and writes
results/catalog_test_status.json (the machine-readable implementation status)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("POLARS_MAX_THREADS", "1")

from fxlab.config import RESULTS_DIR  # noqa: E402

_RESULTS: dict[str, dict] = {}

REAL_SYMBOL = "EURUSD"
REAL_START, REAL_END = "2012-01-01", "2012-07-31"
PANEL_SYMBOLS = ["XAUUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAGUSD", "EURJPY", "EURGBP"]


def _synthetic_bars(n=3000, seed=7) -> pd.DataFrame:
    """GBM bid/ask bars used ONLY for unit tests of numerical properties."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-05", periods=n, freq="h", tz="UTC")
    r = rng.normal(0, 0.001, n)
    mid = 1.2 * np.exp(np.cumsum(r))
    o = np.r_[mid[0], mid[:-1]]
    h = np.maximum(o, mid) * np.exp(np.abs(rng.normal(0, 0.0005, n)))
    l = np.minimum(o, mid) * np.exp(-np.abs(rng.normal(0, 0.0005, n)))
    sp = 0.00008 + 0.00002 * rng.random(n)
    df = pd.DataFrame(index=idx)
    for k, v in zip("ohlc", (o, h, l, mid)):
        df[f"bid_{k}"] = v - sp / 2
        df[f"ask_{k}"] = v + sp / 2
    df["n_ticks"] = rng.integers(50, 500, n)
    df["spread_mean"] = sp; df["spread_max"] = sp * 1.5; df["spread_min"] = sp * 0.8
    df["bid_liq"] = rng.random(n) * 100; df["ask_liq"] = rng.random(n) * 100
    df["up_ticks"] = rng.integers(10, 200, n); df["down_ticks"] = rng.integers(10, 200, n)
    df["rv_intrabar"] = (r ** 2) * 0.9
    df["n_m1"] = 60
    from fxlab.data.store import add_mid
    return add_mid(df)


@pytest.fixture(scope="session")
def synth():
    return _synthetic_bars()


@pytest.fixture(scope="session")
def real_h1():
    from fxlab.data.store import load_bars, available
    if not available(REAL_SYMBOL):
        pytest.skip("real data not yet acquired")
    return load_bars(REAL_SYMBOL, "H1", REAL_START, REAL_END)


@pytest.fixture(scope="session")
def real_panel(real_h1):
    from fxlab.data.store import load_bars, available
    cols = {}
    for s in PANEL_SYMBOLS:
        if available(s):
            try:
                cols[s] = load_bars(s, "H1", REAL_START, REAL_END)["close"]
            except FileNotFoundError:
                pass
    return pd.DataFrame(cols).reindex(real_h1.index)


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.outcome != "passed"):
        return
    ids = {v for k, v in getattr(report, "user_properties", []) if k == "catalog_id"}
    for cid in ids:
        r = _RESULTS.setdefault(cid, {"passed": 0, "failed": 0, "skipped": 0, "tests": []})
        if report.outcome == "passed":
            r["passed"] += 1
        elif report.outcome == "failed":
            r["failed"] += 1
        else:
            r["skipped"] += 1
        r["tests"].append(f"{report.nodeid}:{report.outcome}")


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    path = RESULTS_DIR / "catalog_test_status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    ts = datetime.now(timezone.utc).isoformat()
    for cid, r in _RESULTS.items():
        r["ts"] = ts
        old[cid] = r  # tests of this run replace older records for the same id
    path.write_text(json.dumps(old, indent=1, sort_keys=True))


@pytest.fixture
def catalog_id(request):
    """Call catalog_id(cid) inside a test to attribute its outcome to a catalog entry."""
    def _tag(cid):
        request.node.user_properties.append(("catalog_id", cid))
    return _tag
