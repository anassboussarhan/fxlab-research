"""Selection rules of the pooled random-entry harvest (scripts/random_entry_pool_report.py) use training-window
information only, and the shared candidate set is deterministic."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def _cells(test_bonus=0.0):
    """4 pairs x 3 candidates x 2 folds. Candidate 1 is best on average in training, candidate 0 is pair A's own
    best; candidate 2 is bad in training but (optionally) spectacular in the TEST window."""
    days = pd.bdate_range("2010-01-04", periods=40)
    tr = {"A": [[1.0, 1.0], [0.6, 0.6], [-1.0, -1.0]],
          "B": [[-0.5, -0.5], [0.7, 0.7], [-1.0, -1.0]],
          "C": [[0.1, 0.1], [0.5, 0.5], [-1.0, -1.0]],
          "D": [[0.0, 0.0], [-0.2, -0.2], [-1.0, -1.0]]}
    vr = {"A": 0.80, "B": 0.85, "C": 1.10, "D": 1.20}
    cells = {}
    rng = np.random.default_rng(0)
    for p, T in tr.items():
        net = pd.DataFrame(rng.normal(0, 1e-4, (40, 3)), index=days, columns=["0", "1", "2"])
        net["2"] += test_bonus
        net["fold"] = np.r_[np.zeros(20, int), np.ones(20, int)]
        cells[p] = {"keys": ["2010H1", "2010H2"], "folds": [{"vr5": vr[p]}, {"vr5": vr[p]}],
                    "train": np.array(T, np.float32), "trades": np.full((3, 2), 50.0, np.float32),
                    "gross_sh": np.zeros((3, 2)), "ssum": np.zeros((3, 2, 4)), "ssq": np.ones((3, 2, 4)),
                    "nd": np.array([20, 20]), "net": net, "half": net.copy(), "gross": net.copy()}
    return cells


def test_pooled_and_per_pair_rules_follow_training_only():
    import random_entry_pool_report as RP
    R, choices, nC = RP.select(_cells())
    R2, choices2, _ = RP.select(_cells(test_bonus=1.0))      # a huge test-window gain for candidate 2
    assert R == R2 and choices == choices2                    # selection cannot see the test window
    assert all(R["pooled"][p][0] == [1] for p in "ABCD")      # best mean training Sharpe
    assert R["per_pair"]["A"][0] == [0] and R["per_pair"]["B"][0] == [1]
    assert R["apriori"]["C"][1] == [0]
    assert 2 not in R["pooled_top10"]["A"][0][:2]
    # filter: pooled candidate 1 traded only where it was profitable in training (not on D)
    assert set(p for p in "ABCD" if R["pooled_filter"][p]) == {"A", "B", "C"}


def test_mean_reverting_rule_needs_three_pairs_below_the_median():
    import random_entry_pool_report as RP
    R, choices, _ = RP.select(_cells())
    # VR median = 0.975: only A and B are below -> fewer than 3 -> no pick
    assert all(not R["pooled_mr"][p] for p in "ABCD")
    assert all(c["pooled_mr"] is None for c in choices)


def test_stream_and_single_trader():
    import random_entry_pool_report as RP
    c = _cells()["A"]
    s = RP.stream(c, "net", {0: [1], 1: [0, 1]})
    assert len(s) == 40
    assert np.allclose(s.iloc[:20].to_numpy(), c["net"]["1"].iloc[:20].to_numpy())
    assert np.allclose(s.iloc[20:].to_numpy(), c["net"][["0", "1"]].iloc[20:].mean(axis=1).to_numpy())
    assert np.isnan(RP.single_trader(c, {0: [1], 1: [0, 1]}))   # ensembles have no single-trader path


def test_candidates_are_deterministic_and_start_with_the_apriori_grid():
    import random_entry_pool as P
    a, b = P.candidates(), P.candidates()
    assert a == b and len(a) == 321
    assert a[0]["apriori"] and a[0]["fam"] == "grid" and a[0]["P"]["step"] == 1.0
    assert {c["fam"] for c in a} == {"grid", "bracket"}
