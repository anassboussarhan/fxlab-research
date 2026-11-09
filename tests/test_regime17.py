"""Checks of the round-17 engine (docs/round17_preregistration.md, section 6): no method may use data from the
decision period or later; a planted look-ahead must be obvious; a planted regime must be recovered; the incremental
conditional means must equal a direct computation."""
import os
import sys

import math

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import regime17 as G  # noqa: E402

CASES = [("B1", {"window": "exp"}), ("B1", {"window": "long"}), ("M1", {"L": 6}), ("M6", {"eta": 0.3, "hl": 3}),
         ("M8", {"h": 4}),
         ("M2", {"algo": "kmeans", "K": 3, "set": "PRICE", "n0": 12}),
         ("M2", {"algo": "gmm", "K": 2, "set": "ALL", "n0": 0}),
         ("M3", {"f": "RV21", "K": 3, "n0": 0, "window": "exp"}), ("M3", {"f": "VIX", "K": 2, "n0": 12, "window": "long"}),
         ("M4", {"K": 2, "set": "PRICE", "n0": 0}), ("M4", {"K": 2, "set": "RET", "n0": 12}),
         ("M5", {"set": "PRICE", "alpha": 10}), ("M9", {"set": "PRICE"})]


@pytest.mark.parametrize("fam,p", CASES)
def test_scores_ignore_the_future(fam, p):
    D = G.synthetic("planted", "M")
    j = int(np.flatnonzero(D.ends.year == 2016)[5])
    s1 = G.scores_for(D, D.tier, fam, p, G.Cache())
    D2 = G.synthetic("planted", "M")
    rng = np.random.default_rng(1)
    D2.FV[j:] = rng.normal(size=D2.FV[j:].shape) * 5
    D2.tier.z[j:] = rng.normal(size=D2.tier.z[j:].shape).astype(np.float32) * 3
    D2.tier.zw[j:] = np.clip(D2.tier.z[j:], -5, 5)
    s2 = G.scores_for(D2, D2.tier, fam, p, G.Cache())
    a, b = s1[D.j0: j + 1], s2[D.j0: j + 1]
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-5)


def test_incremental_equals_direct():
    D = G.synthetic("planted", "M")
    tier = D.tier
    per = G.split_periods(D, D.FV[:, D.fcol["RV21"]], 3)
    for window in ("exp", "long"):
        static = G.static_scores(D, tier, window)
        sc = G.regime_scores(D, tier, per, window, 12, static)
        for s, e, Gm, P in per[2:4]:
            for j in (s, s + 3, e - 1):
                a = G.window_start(j, window, D)
                r = int(np.argmax(P[j, :, 0]))
                w = Gm[a:j, r, 0]
                zw = tier.zw[a:j]
                fin = np.isfinite(zw)
                nr = w @ fin
                mr = (w @ np.nan_to_num(zw)) / nr
                sh = (nr * mr + 12 * static[j]) / (nr + 12)
                exp = np.where(nr >= D.cfg["minreg"], sh, static[j])
                ok = np.isfinite(exp)
                assert np.allclose(sc[j][ok], exp[ok], atol=1e-5)


def test_planted_lookahead_is_obvious():
    def oos(kind):
        D = G.synthetic(kind, "M")
        Sc = G.scores_for(D, D.tier, "M3", {"f": "RV21", "K": 2, "n0": 0, "window": "exp"}, G.Cache())
        R = G.evaluate(D, D.tier, Sc, (1,))[1]
        return G.sharpe(R[D.j0:], 12)
    assert oos("leak") > 0.5 and oos("leak") > oos("leak_shift") + 0.5


def test_planted_regime_is_recovered():
    D = G.synthetic("planted", "M")
    Sc = G.scores_for(D, D.tier, "M3", {"f": "RV21", "K": 2, "n0": 0, "window": "exp"}, G.Cache())
    Sb = G.scores_for(D, D.tier, "B1", {"window": "exp"}, G.Cache())
    R, Rb = G.evaluate(D, D.tier, Sc, (1,))[1], G.evaluate(D, D.tier, Sb, (1,))[1]
    assert G.sharpe(R[D.j0:], 12) > G.sharpe(Rb[D.j0:], 12) + 0.3


def test_hedge_weights_sum_below_one_and_cash():
    D = G.synthetic("null", "M")
    W = G.hedge_weights(D, D.tier, 1.0, None)
    tot = W[D.j0:].sum(1)
    assert np.all(tot <= 1 + 1e-5) and np.all(tot >= 0)


def test_costs_by_frequency():
    """The same absolute switching cost at every frequency: cost x period volatility (proportional to
    1 / sqrt(periods per year)) is constant."""
    v = [G.FREQ[f]["cost"] / math.sqrt(G.FREQ[f]["ppy"]) for f in ("M", "W", "D")]
    assert max(v) / min(v) < 1.01
