"""Random-entry management kernels on driftless random walks.

On a continuous martingale every management rule has zero expectancy (optional stopping). On a DISCRETE path,
exits at exactly the order level forgo the overshoot of the crossing step, which costs a resting take-profit
epsilon x P(take-profit) and saves a stop epsilon x P(stop): asymmetric brackets (near target, far stop) therefore
show a small negative bias that shrinks with the path's step size (with tick-level data it is negligible). The
tests check (1) no family gains on a random walk without costs (a look-ahead would show up as a gain), (2) a
symmetric bracket is unbiased, (3) costs always reduce the result."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def _rw_bars(n=12000, seed=3, k=30, sp=0.00010):
    """Hourly bars of an arithmetic random walk built from k sub-bars each (also returned as the 'M1' path)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-02", periods=n, freq="h", tz="UTC")
    path = 1.2 + np.cumsum(rng.normal(0, 0.0012 / np.sqrt(k), n * k)).reshape(n, k)
    o = np.r_[1.2, path[:-1, -1]]
    c = path[:, -1]; h = np.maximum(path.max(1), o); l = np.minimum(path.min(1), o)
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


def _zero_cost_net(RO, P, seeds0, paths=16):
    """Total return per path (mean over 4 entry seeds) of the NET kernel run with zero spread and zero costs, so
    the M1 replay is active (the gross run switches it off)."""
    G = []
    for path in range(paths):
        bars, m1 = _rw_bars(seed=seeds0 + path, sp=0.0)
        D = RO.Data(bars, 0.0001, m1)
        _, b, _ = RO.evaluate("smart", P, D, 500, D.n, [1, 2, 3, 4], 24.0, costs=(0.0, 0.0, 0.0, 0.0))
        G.append(np.mean([x.sum() for x in b]))
    G = np.array(G)
    return G.mean(), G.std(ddof=1) / np.sqrt(len(G))


@pytest.mark.parametrize("P", [
    {"rate": 4.0, "etype": "limit", "k": 0.3, "ttl": 5, "sl": 1.0, "tp": 0.3, "size": "notional"},
    {"rate": 6.0, "etype": "limit", "k": 0.1, "ttl": 3, "sl": 2.0, "tp": 0.2, "part": 0.1, "be": 0.1,
     "trail": 0.5, "size": "notional"}])
def test_smart_limit_entries_take_nothing_from_the_fill_bar(RO, P):
    """A limit entry fills on a move AGAINST the position, after the bar's favourable extreme in most cases: a
    target, partial, trailing or breakeven taken from the fill bar's extremes would show up as a large gain here
    (limit fills at the level also forgo the crossing overshoot, a small negative bias, so the test is one-sided)."""
    m, se = _zero_cost_net(RO, P, 300)
    assert m < 3.5 * se + 2e-3, (m, se)


def test_smart_market_entries_with_filters_are_unbiased(RO):
    """Symmetric bracket (target = stop) with the session window, volatility filter, stop-and-reverse and flat
    before rollover: the overshoot effects cancel, so the zero-cost expectancy must be zero."""
    P = {"rate": 4.0, "etype": "market", "sl": 1.5, "tp": 1.0, "size": "notional", "h0": 6, "width": 12,
         "vol": "high", "vq": 0.9, "sar": True, "flat_roll": True}
    m, se = _zero_cost_net(RO, P, 700, paths=24)
    assert abs(m) < 3.5 * se + 1e-3, (m, se)


def _one_bar_case(RO, sub, bar1, etype=1, use_m1=True, tp=0.5, **kw):
    """Bar 0 flat at 1.0000 (bid; ask = bid + 0.0001), buy signal at its close, ATR 0.0010; bar 1 given as
    (open, high, low, close) of the bid with its M1 sub-bars sub = [(bid high, bid low), ...]; bars 2-3 flat at
    bar 1's close. Returns the bar returns and the trade count."""
    o, h, l, c = bar1
    bo = np.array([1.0, o, c, c]); bh = np.array([1.0, h, c, c]); bl = np.array([1.0, l, c, c]); bc = np.array([1.0, c, c, c])
    s = 0.0001
    atr = np.full(4, 0.0010); roll = kw.pop("roll", np.zeros(4)); ent = np.array([1, 0, 0, 0], np.int8)
    allow = np.ones(4, np.bool_)
    mbh = np.array([x[0] for x in sub]); mbl = np.array([x[1] for x in sub])
    m1o = np.array([0, 0, len(sub), len(sub), len(sub)], np.int64)
    return RO._smart(bo, bh, bl, bc, bo + s, bh + s, bl + s, bc + s, atr, roll, ent, allow, etype,
                     kw.get("koff", 0.5), 5, 2.0, tp, 0.0, 0.0, 0.0, 0, 0, 0.005, 10.0, kw.get("sar", False),
                     kw.get("flat_roll", False), 0.0, 0.0, 0.0001, 0.0, kw.get("swap", 0.0), m1o, mbh, mbl,
                     mbh + s, mbl + s, use_m1)


def test_smart_fill_bar_target_only_after_the_fill(RO):
    """Buy limit at 0.99955 (mid close 1.00005 - 0.5 ATR), stop 0.99755, target 1.00055. Bar 1 trades
    1.0012 and 0.9990. The target counts only if the M1 path reaches it AFTER the fill."""
    L, tgt = 1.00005 - 0.0005, 1.00005 - 0.0005 + 0.0010
    bar1 = (1.0000, 1.0012, 0.9990, 0.9995)
    # (a) the high came first, then the drop that filled the order: no target; open at the close
    b, n = _one_bar_case(RO, [(1.0012, 1.0000), (1.0012, 0.9990), (0.9995, 0.9990)], bar1)
    assert n == 1 and abs(b[1] - (0.9995 - L) / L) < 1e-12
    # (b) the fill came first, then the rally to the target: take-profit on the fill bar
    b, n = _one_bar_case(RO, [(1.0000, 0.9990), (1.0012, 0.9990), (1.0012, 0.9995)], bar1)
    assert n == 1 and abs(b[1] - (tgt - L) / L) < 1e-12 and b[2] == 0.0
    # (c) without the M1 path only the stop can trigger on the fill bar
    b, n = _one_bar_case(RO, [(1.0000, 0.9990), (1.0012, 0.9990), (1.0012, 0.9995)], bar1, use_m1=False)
    assert n == 1 and abs(b[1] - (0.9995 - L) / L) < 1e-12
    # (d) a gap through the limit at the open fills at the open: the whole bar is after the fill
    b, n = _one_bar_case(RO, [(1.0012, 0.9990)], (0.9990, 1.0012, 0.9990, 0.9995))
    e = 0.9991
    assert n == 1 and abs(b[1] - (e + 0.0010 - e) / e) < 1e-12


def test_smart_flat_before_rollover_and_stop_and_reverse(RO):
    # flat_roll: market buy at bar 1's open (ask 1.0001); bar 2 contains the rollover -> out at its open, no swap
    roll = np.array([0.0, 0.0, 1.0, 0.0])
    b, n = _one_bar_case(RO, [(1.0005, 1.0000)], (1.0000, 1.0005, 1.0000, 1.0004), etype=0, tp=0.0,
                         roll=roll, flat_roll=True, swap=5.0)
    e = 1.0001
    assert n == 1 and abs(b[1] - (1.0004 - e) / e) < 1e-12 and abs(b[2]) < 1e-12 and b[3] == 0.0
    # same without flat_roll: the swap is charged on bar 2
    b2, _ = _one_bar_case(RO, [(1.0005, 1.0000)], (1.0000, 1.0005, 1.0000, 1.0004), etype=0, tp=0.0,
                          roll=roll, swap=5.0)
    assert b2[2] < -1e-5
    # stop-and-reverse: market buy at 1.0001, stop 0.9981; bar 1 falls to 0.9975 -> stopped; short at bar 2's open
    b, n = _one_bar_case(RO, [(1.0000, 0.9975)], (1.0000, 1.0000, 0.9975, 0.9978), etype=0, tp=0.0, sar=True)
    assert n == 2


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
