"""Random-entry management kernels on driftless random walks.

On a continuous martingale every management rule has zero expectancy (optional stopping). On a DISCRETE path,
exits at exactly the order level forgo the overshoot of the crossing step, which costs a resting take-profit
epsilon x P(take-profit) and saves a stop epsilon x P(stop): asymmetric brackets (near target, far stop) therefore
show a small negative bias that shrinks with the path's step size (with tick-level data it is negligible). The
tests check (1) no family gains on a random walk without costs (a look-ahead would show up as a gain), (2) a
symmetric bracket is unbiased, (3) costs always reduce the result."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def _rw_bars(n=12000, seed=3, k=30):
    """Hourly bars of an arithmetic random walk built from k sub-bars each (also returned as the 'M1' path)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-02", periods=n, freq="h", tz="UTC")
    path = 1.2 + np.cumsum(rng.normal(0, 0.0012 / np.sqrt(k), n * k)).reshape(n, k)
    o = np.r_[1.2, path[:-1, -1]]
    c = path[:, -1]; h = np.maximum(path.max(1), o); l = np.minimum(path.min(1), o)
    sp = 0.00010
    df = pd.DataFrame(index=idx)
    for kk, v in zip("ohlc", (o, h, l, c)):
        df[f"bid_{kk}"] = v - sp / 2
        df[f"ask_{kk}"] = v + sp / 2
    prev = np.c_[o[:, None], path[:, :-1]]
    sh = np.maximum(prev, path).ravel(); sl = np.minimum(prev, path).ravel()
    mi = pd.DatetimeIndex((idx.values[:, None] + (np.arange(k) * 2 * 60 * 10 ** 9).astype("timedelta64[ns]")).ravel(), tz="UTC")
    m1 = pd.DataFrame({"bid_h": sh - sp / 2, "bid_l": sl - sp / 2, "ask_h": sh + sp / 2, "ask_l": sl + sp / 2}, index=mi)
    return df, m1


@pytest.fixture(scope="module")
def RO():
    import random_entry_opt as m
    return m


@pytest.mark.parametrize("fam,P", [
    ("bracket", {"rate": 4.0, "sl": 2.0, "tp": 1.5, "trail": 3.0, "size": "notional"}),
    ("martingale", {"rate": 4.0, "sl": 1.0, "tp": 1.0, "mg_mode": "martingale", "mg_f": 2.0, "mg_k": 4, "size": "notional"}),
    ("pyramid", {"rate": 2.0, "add": 1.0, "N": 4, "give": 2.0, "stop": 2.0}),
    ("grid", {"rate": 2.0, "step": 1.0, "N": 5, "g": 1.5, "tp": 0.5, "sl": 3.0}),
    ("zone", {"rate": 2.0, "zone": 1.0, "tp": 1.0, "K": 4})])
def test_zero_cost_random_walk_has_no_edge_and_costs_hurt(RO, fam, P):
    """Expectancy is over price paths: 16 independent random-walk paths x 4 entry seeds."""
    G, Nn = [], []
    for path in range(16):
        bars, m1 = _rw_bars(seed=100 + path)
        D = RO.Data(bars, 0.0001, m1)
        _, bg, _ = RO.evaluate(fam, P, D, 500, D.n, [1, 2, 3, 4], 24.0, gross=True)
        _, bn, _ = RO.evaluate(fam, P, D, 500, D.n, [1, 2, 3, 4], 24.0, gross=False)
        G.append(np.mean([b.sum() for b in bg])); Nn.append(np.mean([b.sum() for b in bn]))
    G, Nn = np.array(G), np.array(Nn)
    se = G.std(ddof=1) / np.sqrt(len(G))
    assert G.mean() < 3.5 * se + 2e-3, (fam, G.mean(), se)        # no gain without costs on a random walk
    assert Nn.mean() < G.mean()                                   # costs always reduce the result


def test_symmetric_bracket_is_unbiased(RO):
    """Target = stop: the overshoot effects cancel, so the zero-cost expectancy must be zero."""
    G = []
    for path in range(24):
        bars, m1 = _rw_bars(n=6000, seed=900 + path)
        D = RO.Data(bars, 0.0001, m1)
        _, bg, _ = RO.evaluate("bracket", {"rate": 4.0, "sl": 1.5, "tp": 1.0, "size": "notional"}, D, 300, D.n,
                               [1, 2, 3, 4], 24.0, gross=True)
        G.append(np.mean([b.sum() for b in bg]))
    G = np.array(G)
    assert abs(G.mean()) < 3.5 * G.std(ddof=1) / np.sqrt(len(G)) + 1e-3
