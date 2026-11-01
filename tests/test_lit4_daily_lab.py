import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lit4_daily_lab as L  # noqa: E402


def _panel(years=(1971, 2025), seed=0, plant=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{years[0]}-01-31", f"{years[1]}-12-31", freq="ME")
    X = pd.DataFrame({c: rng.normal(0, 0.02, len(idx)) for c in L.CCYS}, index=idx)
    if plant:
        c, m, y0, v = plant
        sel = (X.index.month == m) & (X.index.year >= y0)
        X.loc[sel, c] = v
    return X


def test_rule_needs_fifteen_of_the_previous_twenty_years():
    # JPY positive every August from 1980: the rule may trade it only from 1980 + 15 positives within 20 years
    X = _panel(plant=("JPY", 8, 1980, 0.01))
    pos = L.seasonal_positions(X, first=1991)
    aug = sorted(t.year for (t, c), p in pos.items() if c == "JPY" and t.month == 8 and p == 1)
    assert aug and aug[0] >= 1991
    for y in aug:
        h = X.JPY[(X.index.month == 8) & (X.index.year >= y - 20) & (X.index.year < y)]
        assert (h > 0).sum() >= 15                              # decided from earlier years only


def test_no_position_from_the_test_year_itself():
    X = _panel(seed=1)
    X.loc["2000-08-31", "JPY"] = 5.0                            # a huge August 2000 must not change August 2000's call
    a = L.seasonal_positions(X, first=2000, rule="sign")
    X2 = X.copy()
    X2.loc["2000-08-31", "JPY"] = -5.0
    b = L.seasonal_positions(X2, first=2000, rule="sign")
    t = pd.Timestamp("2000-08-31")
    assert a.get((t, "JPY")) == b.get((t, "JPY"))


def test_portfolio_charges_costs_and_averages():
    X = _panel(seed=2)
    t = X.index[300]
    pos = {(t, "EUR"): 1, (t, "NZD"): -1}
    p = L.portfolio(X, pos)
    want = 0.5 * ((X.at[t, "EUR"] - 2e-4) + (-X.at[t, "NZD"] - 3e-4))
    assert abs(p.loc[t] - want) < 1e-12


def test_published_patterns_only_after_publication():
    X = _panel(seed=3)
    g = pd.Series(0.01, index=X.index)
    s = L.s2_published(X, g)
    assert s.index.min() >= L.PUBLISHED and s.index.max() <= L.END
    # 2022-2025: all 15 patterns; 2021 from June: EUR Jun, AUD Jun, JPY / GBP / AUD / gold Aug, EUR Dec = 7
    assert len(s) == 4 * 15 + 7


def test_turn_of_month_window_is_four_days():
    idx = pd.bdate_range("2010-01-01", "2010-12-31")
    ex = pd.Series(0.0, index=idx)
    ex[idx.to_period("M") != pd.Series(idx.to_period("M"), index=idx).shift(-1).values] = 0.01   # last days
    w, welch, m_in, m_out = L.g1_tom(ex, cost=0.0)
    assert len(w) == 11 and np.allclose(w.to_numpy(), 0.01)


def test_newey_west_never_below_iid():
    x = np.tile([1.0, -1.0], 30) + 0.1                          # strongly negative autocorrelation
    t_iid = x.mean() / x.std(ddof=0) * np.sqrt(len(x))
    assert L.nw_t(x) <= t_iid + 1e-9
