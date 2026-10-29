import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cot_lab as C  # noqa: E402

RAW_COLS = ["Market and Exchange Names", "As of Date in Form YYMMDD", "As of Date in Form YYYY-MM-DD",
            "CFTC Contract Market Code", "Open Interest (All)", "Noncommercial Positions-Long (All)",
            "Noncommercial Positions-Short (All)", "Noncommercial Positions-Spreading (All)",
            "Commercial Positions-Long (All)", "Commercial Positions-Short (All)", "Change in Open Interest (All)",
            "% of OI-Noncommercial-Long (All)", "Traders-Noncommercial-Long (All)"]


def _fake(weeks=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2003-01-07", periods=weeks, freq="7D")                    # Tuesdays
    rows = []
    for m, code in C.CODES.items():
        ns = np.cumsum(rng.normal(0, 1000, weeks))
        for d, x in zip(dates, ns):
            rows.append([m, d.strftime("%y%m%d"), d.strftime("%Y-%m-%d"), int(code), 100000, 50000 + x, 50000 - x,
                         0, 40000 - x, 40000 + x, 0, 0, 0])
    raw = pd.DataFrame(rows, columns=RAW_COLS)
    days = pd.bdate_range("2003-01-01", "2010-12-31")
    prices = pd.DataFrame({m: np.exp(np.cumsum(rng.normal(0, 0.005, len(days)))) for m in C.CODES}, index=days)
    return raw, prices


def test_tidy_cot_reads_the_legacy_columns():
    raw, _ = _fake(50)
    cot = C.tidy_cot(raw)
    assert set(cot.market) == set(C.CODES) and len(cot) == 50 * len(C.CODES)
    assert (cot.nc_long + cot.nc_short == 100000).all()


def test_z_uses_only_previous_weeks():
    x = pd.Series(np.r_[np.zeros(150), 10.0])
    x.iloc[:150] = np.random.default_rng(1).normal(0, 1, 150)
    z = C.zscore(x)
    prev = x.iloc[-157:-1]
    assert abs(z.iloc[-1] - (10 - prev.mean()) / prev.std()) < 1e-9


def test_entry_is_the_monday_after_publication_and_holds_to_the_next():
    raw, prices = _fake(300)
    cot = C.tidy_cot(raw)
    panel = C.weekly_panel(cot, prices)
    g = panel["EUR"]
    assert (pd.DatetimeIndex(g.index).weekday == 0).all()             # Monday entries (business-day calendar)
    assert (pd.DatetimeIndex(g["exit"]) > pd.DatetimeIndex(g.index)).all()
    # the report behind an entry is at least 6 days old
    rep = cot[cot.market == "EUR"].set_index("date").index
    for e in g.index[:20]:
        assert rep[rep <= e - pd.Timedelta(days=6)].size > 0


def test_cross_sectional_signal_is_dollar_neutral_in_units():
    raw, prices = _fake(400)
    panel = C.weekly_panel(C.tidy_cot(raw), prices)
    P = C.positions(panel)
    D = pd.DataFrame(P["C3"]).dropna()
    assert len(D) > 50 and np.allclose(D.sum(axis=1), 0.0) and np.allclose(D.abs().sum(axis=1), 2.0)
