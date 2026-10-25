"""scripts/lit2_lab.py and lit2_report.py: FOMC change dates, metals swap sign, no look-ahead in the FX tests."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


@pytest.fixture(scope="module")
def L():
    import lit2_lab as m
    return m


def test_fomc_change_dates_are_scheduled_meetings(L):
    sched = {f"{y}-{x}" for y, s in L.FL.FOMC_FALLBACK.items() for x in s.split()}
    assert set(L.FOMC_CHANGES) <= sched
    assert sum(v < 0 for v in L.FOMC_CHANGES.values()) == 8 and sum(v > 0 for v in L.FOMC_CHANGES.values()) == 7


def test_long_metal_pays_usd_interest_through_the_rollover(L):
    idx = pd.date_range("2008-03-04 20:00", periods=4 * 60, freq="min", tz="UTC")     # crosses 17:00 New York
    b = np.full(len(idx), 900.0)
    df = pd.DataFrame({"bid_o": b, "ask_o": b + 0.5, "bid_h": b, "bid_l": b, "ask_h": b + 0.5, "ask_l": b + 0.5},
                      index=idx)
    bk = L.metal_book(df, 0.01, "XAUUSD")
    r = bk.trade(idx[0], idx[-1], +1)
    assert r[4] == 1 and r[6] < 0                     # one night, long pays (USD rate + mark-up) / 365
    from fxlab.data.rates import SWAP_MARKUP_PA, known_at
    usd = float(known_at(pd.DatetimeIndex([idx[0].floor("D")]))["USD"].iloc[0])
    assert abs(r[6] - (-usd - SWAP_MARKUP_PA) / 365) < 1e-12
    r = bk.trade(idx[0], idx[-1], -1)
    assert abs(r[6] - (usd - SWAP_MARKUP_PA) / 365) < 1e-12


def test_fx_tests_ignore_returns_after_the_decision(monkeypatch, tmp_path):
    """Changing every return after a date must not change any FX-test return before that date."""
    import lit2_report as R
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2007-01-01", "2012-12-31")
    xr = pd.DataFrame(rng.normal(0, 0.006, (len(days), 9)), index=days,
                      columns=["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "SEK", "NOK"])
    spr = pd.DataFrame(2e-4, index=days, columns=xr.columns)
    rdf = pd.DataFrame(rng.normal(0, 0.01, (len(days), 9)), index=days, columns=xr.columns)
    cut = pd.Timestamp("2010-06-30")
    xr2 = xr.copy()
    xr2[xr2.index > cut] *= -3.0
    a, b = R.monthly_tests(xr, spr, rdf), R.monthly_tests(xr2, spr, rdf)
    for k in a:
        na, nb = a[k][0], b[k][0]
        pre = na.index[na.index <= cut]
        assert len(pre) > 20 and np.allclose(na.loc[pre], nb.loc[pre])
    wa, wb = R.weekly_momentum(xr, xr, spr)[0], R.weekly_momentum(xr2, xr2, spr)[0]
    pre = wa.index[wa.index <= cut]
    assert len(pre) > 20 and np.allclose(wa.loc[pre], wb.loc[pre])
