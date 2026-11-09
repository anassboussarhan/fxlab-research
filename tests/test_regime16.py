"""Checks of the round-16 regime engine (docs/round16_preregistration.md, section 9): no method may use data from
the decision month or later, and a planted look-ahead must show up as an obviously fake result."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import regime16 as G  # noqa: E402

CASES = [("B1", {"window": "exp"}), ("B1", {"window": "60"}), ("M1", {"L": 6}),
         ("M2", {"algo": "kmeans", "K": 3, "set": "PRICE", "n0": 12, "window": "exp"}),
         ("M2", {"algo": "gmm", "K": 2, "set": "ALL", "n0": 0, "window": "60"}),
         ("M3", {"f": "RV21", "K": 3, "n0": 0, "window": "exp"}),
         ("M4", {"K": 2, "set": "PRICE", "n0": 0}), ("M4", {"K": 2, "set": "RET", "n0": 12}),
         ("M5", {"model": "ridge", "set": "PRICE", "alpha": 10}), ("M5", {"model": "gb", "set": "PRICE"})]


def test_scale_uses_only_earlier_months():
    rng = np.random.default_rng(0)
    R = rng.normal(size=(120, 5))
    z1 = G.scale(R)
    R2 = R.copy()
    R2[60:] = rng.normal(size=(60, 5)) * 10
    z2 = G.scale(R2)
    assert np.allclose(np.nan_to_num(z1[:60]), np.nan_to_num(z2[:60]))
    # z[60] uses R[60] itself but its scale only from months 24..59
    assert np.allclose(z2[60] * np.sqrt((R[24:60] ** 2).mean(0)), R2[60])


@pytest.mark.parametrize("fam,p", CASES)
def test_scores_ignore_the_future(fam, p):
    """Perturbing feature rows and returns from month j on must not change any decision up to month j."""
    D = G.synthetic("planted")
    tier = D.tiers["FACTOR"]
    j = D.months.index("2016-06")
    s1 = G.scores_for(D, tier, fam, p, G.Cache())
    D2 = G.synthetic("planted")
    rng = np.random.default_rng(1)
    D2.FV[D2.ridx[j]:] = rng.normal(size=D2.FV[D2.ridx[j]:].shape) * 5
    for t in list(D2.tiers.values()):
        t.z[j:] = rng.normal(size=t.z[j:].shape) * 3
        t.zw[j:] = np.clip(t.z[j:], -5, 5)
    s2 = G.scores_for(D2, D2.tiers["FACTOR"], fam, p, G.Cache())
    a, b = s1[D.j0: j + 1], s2[D.j0: j + 1]
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9)


def test_planted_lookahead_is_obvious():
    def oos_sharpe(kind):
        D = G.synthetic(kind)
        t = D.tiers["FACTOR"]
        Sc = G.scores_for(D, t, "M3", {"f": "RV21", "K": 2, "n0": 0, "window": "exp"}, G.Cache())
        R = G.evaluate(D, t, Sc, (1,))[1]
        return G.sharpe(R[D.j0:])
    leak, shifted = oos_sharpe("leak"), oos_sharpe("leak_shift")
    assert leak > 0.6 and leak > shifted + 0.5


def test_planted_regime_is_recovered():
    D = G.synthetic("planted")
    t = D.tiers["FACTOR"]
    Sc = G.scores_for(D, t, "M3", {"f": "RV21", "K": 2, "n0": 0, "window": "exp"}, G.Cache())
    R = G.evaluate(D, t, Sc, (1,))[1]
    Sb = G.scores_for(D, t, "B1", {"window": "exp"}, G.Cache())
    Rb = G.evaluate(D, t, Sb, (1,))[1]
    assert G.sharpe(R[D.j0:]) > G.sharpe(Rb[D.j0:]) + 0.2


def test_topk_weights_and_costs():
    D = G.synthetic("null")
    t = D.tiers["FACTOR"]
    Sc = np.full(t.z.shape, np.nan)
    Sc[:, 0] = 1.0                                                         # always hold F1 only
    R = G.evaluate(D, t, Sc, (1,))[1]
    exp = t.z[D.j0:, 0].copy()
    exp[0] -= G.COST                                                       # one purchase, no later switches
    assert np.allclose(R[D.j0:], exp)
