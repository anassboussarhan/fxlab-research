import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import options_edges_lab as E  # noqa: E402
import options_strategy_lab as L  # noqa: E402
from test_options_strategy_lab import _chain  # noqa: E402


def test_hedged_straddle_on_a_flat_underlying_keeps_the_premium():
    C, und, exp = _chain()                      # underlying constant at 100
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    k = E.atm_strike(bk, d, exp, 100.0)
    r = E.option_position(bk, d, exp, [("P", k, -1), ("C", k, -1)], True, 1e-4, und)
    # flat price: no hedge P&L; the straddle expires at |S - K| = 0; P&L = premium net of costs
    assert abs(r["pnl"] - (r["premium"] - r["hedge_cost"])) < 1e-9
    assert r["premium"] > 0


def test_hedging_reduces_the_spread_of_outcomes_on_a_trend():
    path = list(np.linspace(100, 80, 55))
    C, und, exp = _chain(path=path)
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    S = float(und.px(d))
    k = E.atm_strike(bk, d, exp, S)
    legs = [("P", k, -1), ("C", k, -1)]
    un = E.option_position(bk, d, exp, legs, False, 1e-4, und)
    he = E.option_position(bk, d, exp, legs, True, 1e-4, und)
    assert he["pnl"] > un["pnl"]               # the short straddle loses on the trend; the hedge recovers most
