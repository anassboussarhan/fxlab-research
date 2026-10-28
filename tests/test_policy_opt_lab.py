import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import ml_adaptive_barrier_lab as A  # noqa: E402
import policy_opt_lab as P  # noqa: E402
from test_ml_adaptive_barrier_lab import _book  # noqa: E402


def _arrays(seed=0, n=6000):
    bk = _book(seed, n)
    arr = tuple(x.astype(float) for x in (bk.bo, bk.ao, bk.bh, bk.bl, bk.ah, bk.al))
    first = np.arange(0, n, 5, dtype=np.int64)                      # M5 bars over the M1 path
    rc = (np.arange(n) // 700).astype(np.int64)                     # some swap nights inside the horizons
    swl = np.full(n, -2e-5)
    sws = np.full(n, 1e-5)
    return bk, arr, first, rc, swl, sws


def test_payoff_tensor_matches_the_exact_simulation_at_every_axis_point():
    bk, arr, first, rc, swl, sws = _arrays()
    rng = np.random.default_rng(3)
    T = np.sort(rng.choice(np.arange(20, len(first) - P.HMAX - 3), 6, replace=False)).astype(np.int64)
    atr = np.full(len(first), 0.0012)
    sp = bk.sp * bk.pip
    out, rate = P.payoff_tensor(*arr, first, T, atr, P.TPL, P.SLL, P.HL, bk.sf, sp, bk.comm, rc, swl, sws, 5.0)
    worst = worst_rate = 0.0
    for q, t in enumerate(T):
        i = int(first[t + 1])
        for s, side in ((0, 1.0), (1, -1.0)):
            for aa in range(P.NA):
                for b in range(P.NA):
                    for c in range(P.NA):
                        j = int(first[t + 1 + P.HL[c]])
                        net, _, k = A.sim(*arr, i, j, side, P.TPL[aa] * atr[t], P.SLL[b] * atr[t], bk.sf, sp,
                                          bk.comm, rc, swl, sws)
                        worst = max(worst, abs(out[q, s, aa, b, c] - net * 1e4))
                        ref = net * 1e4 / max((k - i) / 5.0, 1.0)
                        worst_rate = max(worst_rate, abs(rate[q, s, aa, b, c] - ref))
    assert worst < 1e-3 and worst_rate < 1e-3                                              # float32 storage of bp values


def test_relaxed_objective_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    n = 40
    Pt = rng.normal(0, 20, (n, 2, P.NA, P.NA, P.NA)).astype(np.float32)
    O = rng.normal(0, 1.5, (n, 8))
    Qt = (Pt / 3.0).astype(np.float32)
    for kind in ("mean", "sharpe", "rate"):
        J, dO, _ = P.relaxed(O, (Pt, Qt), kind)
        for _ in range(30):
            r, c = int(rng.integers(n)), int(rng.integers(8))
            h = 1e-6
            Op, Om = O.copy(), O.copy()
            Op[r, c] += h
            Om[r, c] -= h
            num = (P.relaxed(Op, (Pt, Qt), kind, grad=False)[0] - P.relaxed(Om, (Pt, Qt), kind, grad=False)[0]) / (2 * h)
            assert abs(num - dO[r, c]) < 1e-5 * max(1.0, abs(num)), (kind, r, c, num, dO[r, c])


def test_linear_and_mlp_backward_match_finite_differences():
    rng = np.random.default_rng(7)
    n, d = 30, 6
    X = rng.normal(size=(n, d))
    Pt = rng.normal(0, 20, (n, 2, P.NA, P.NA, P.NA)).astype(np.float32)
    for pol in (P.Linear(d), P.MLP(d, rng, hidden=5, layers=2)):
        for p in pol.params:
            p += rng.normal(0, 0.3, p.shape)
        _, dO, _ = P.relaxed(pol.forward(X), Pt, "mean")
        grads = pol.backward(dO)
        for k, p in enumerate(pol.params):
            flat = p.reshape(-1)
            for _ in range(5):
                e = int(rng.integers(flat.size))
                old = flat[e]
                flat[e] = old + 1e-6
                Jp = P.relaxed(pol.forward(X), Pt, "mean", grad=False)[0]
                flat[e] = old - 1e-6
                Jm = P.relaxed(pol.forward(X), Pt, "mean", grad=False)[0]
                flat[e] = old
                num = (Jp - Jm) / 2e-6
                assert abs(num - grads[k].reshape(-1)[e]) < 1e-5 * max(1.0, abs(num))


def test_decide_maps_integer_coordinates_to_the_axis_points():
    k = np.array([0, 4, 12])
    u = k + 0.0
    z = np.log(u / (P.NA - 1) / (1 - u / (P.NA - 1) + 1e-300) + 1e-300)    # inverse of 12 * sigmoid
    z = np.clip(z, -40, 40)
    O = np.zeros((3, 8))
    O[:, 0] = 1.0
    O[:, 2], O[:, 3], O[:, 4] = z, z, z
    go, G = P.decide(O, 0.0)
    assert go.all() and (G[:, 0] == 1).all()
    assert np.allclose(G[:, 1], P.TPL[k], rtol=1e-6) and np.allclose(G[:, 2], P.SLL[k], rtol=1e-6)
    assert np.array_equal(G[:, 3], P.HL[k].astype(float))


def test_optimiser_learns_a_planted_rule():
    """Long pays when feature 0 > 0 with a 2-ATR target, short pays when feature 0 < 0; flat otherwise loses."""
    rng = np.random.default_rng(11)
    n = 3000
    X = rng.normal(size=(n, 3))
    Pt = np.full((n, 2, P.NA, P.NA, P.NA), -3.0, dtype=np.float32)
    pos = X[:, 0] > 0
    Pt[pos, 0, 6, :, :] = 10.0                                       # long, take-profit index 6 = 2 ATR
    Pt[~pos, 1, 6, :, :] = 10.0
    pol = P.Linear(3)
    P.adam_fit(pol, X, Pt, X, Pt, "mean", 0.05, 1e-6, steps=1500)
    go, G = P.decide(pol.forward(X), 0.0)
    acc = ((G[:, 0] > 0) == pos)[go].mean()
    assert go.mean() > 0.5 and acc > 0.9
    assert np.median(np.abs(np.log2(G[go, 1] / 2.0))) < 0.5
