import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import prop_best as PB  # noqa: E402


def test_steady_gains_pass_and_get_paid():
    A = np.full(300, 0.002)
    M = np.zeros(300)
    paid, npay, passed, first, fail = PB.simulate(A, M, PB.P["FTMO 2-Step"], 50, 0, 5, 5)
    assert passed.all() and (npay > 0).all() and (fail == 0).all()
    # 1% a day at 5x: phase 1 (10%) and phase 2 (5%) need at least 4 trading days each
    assert (first >= 7).all()


def test_open_loss_beyond_daily_limit_fails_daily():
    A = np.zeros(100)
    M = np.zeros(100)
    A[::5], M[::5] = -0.001, -0.011                      # an 110 bp open loss: 5.5% at 5x
    paid, npay, passed, first, fail = PB.simulate(A, M, PB.P["FTMO 2-Step"], 50, 0, 5, 5)
    assert not passed.any() and (fail == 1).all()


def test_best_day_rule_blocks_one_big_day():
    A = np.zeros(1000)                                   # longer than the evaluation, so the day is not repeated
    M = np.zeros(1000)
    A[0] = 0.04                                          # 20% in one day at 5x: all of the profit
    paid, npay, passed, first, fail = PB.simulate(A, M, PB.P["FTMO 1-Step"], 200, 0, 5, 5)
    assert not passed.any()
    r = dict(PB.P["FTMO 1-Step"], best_day=None)
    paid, npay, passed, first, fail = PB.simulate(A, M, r, 200, 0, 5, 5)
    assert passed.any()                                  # the same day passes without the rule
