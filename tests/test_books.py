"""Round 15 book systems: every rule is causal (prefix invariance) and runs on daily-like bars."""
import numpy as np
import pandas as pd
import pytest

import fxlab.strategies  # noqa: F401
from fxlab.registry import CATALOG

BOOKS = sorted(k for k, e in CATALOG.items() if e.kind == "strategy" and e.family == "books")


def _bars(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    c = 1.2 * np.exp(np.cumsum(rng.standard_normal(n) * 0.006))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.standard_normal(n) * 0.001)
    h = np.maximum(o, c) * (1 + np.abs(rng.standard_normal(n)) * 0.003)
    l = np.minimum(o, c) * (1 - np.abs(rng.standard_normal(n)) * 0.003)
    idx = pd.date_range("2010-01-04", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_books_registered():
    assert len(BOOKS) >= 25


@pytest.mark.parametrize("sid", BOOKS)
def test_book_prefix_invariant(sid):
    e = CATALOG[sid]
    b = _bars()
    full = e.fn(b.copy(), **e.params)
    for cut in (700, 1100):
        part = e.fn(b.iloc[:cut].copy(), **e.params)
        for col in part.columns:
            a, p = full[col].iloc[:cut].to_numpy(float), part[col].to_numpy(float)
            ok = (np.isnan(a) & np.isnan(p)) | np.isclose(a, p, atol=1e-10)
            assert ok.all(), f"{sid} {col} differs at {np.flatnonzero(~ok)[:5]}"
    assert set(np.unique(full["side"])) <= {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("sid", BOOKS)
def test_book_trades_sometimes(sid):
    e = CATALOG[sid]
    s = e.fn(_bars(3000, 5).copy(), **e.params)["side"]
    assert (s != 0).sum() > 0, f"{sid} never signals"
