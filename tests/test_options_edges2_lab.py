import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import options_edges2_lab as E  # noqa: E402
import options_strategy_lab as L  # noqa: E402
from test_options_strategy_lab import _chain  # noqa: E402


def test_unhedged_short_put_settles_at_intrinsic():
    path = list(np.linspace(100, 90, 55))
    C, und, exp = _chain(path=path)
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    k = 95.0
    r = E.position(bk, und, d, exp, [("P", exp, k, -1)], False, 0.0)
    row = C[(C.date == d) & (C.strike == k) & (C.cp == "P")].iloc[0]
    Sx = und.px(exp)
    expect = (row.mid - 0.75 * row.half - L.COMM - max(k - Sx, 0)) / und.px(d)
    assert abs(r["pnl"] - expect) < 1e-9


def test_early_exit_closes_at_the_quote_and_hedge_offsets_delta():
    C, und, exp = _chain()
    bk = L.Book(C, und)
    d, x = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-01")
    legs = E.straddle_legs(bk, d, exp, 100.0, -1)
    r = E.position(bk, und, d, x, legs, True, 0.0, fill=0.0, comm=0.0)
    # flat underlying, flat vol: a one-day short straddle earns about one day of time decay
    assert r is not None and r["pnl"] > 0 and r["hedge_cost"] == 0.0


def test_strip_weights_price_the_variance():
    C, und, exp = _chain(iv=0.2)
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    legs, F = E.strip_legs(bk, d, exp, 100.0)
    T = (exp - d).days / 365.0
    val = sum(-q * float(bk.path(e, cp, k, d, d).mid.iloc[0]) for cp, e, k, q in legs) / 100.0
    assert abs(val / T - 0.2 ** 2) < 0.2 * 0.2 ** 2          # the strip prices about the implied variance
