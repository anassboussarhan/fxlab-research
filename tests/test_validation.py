"""Validation-layer tests: splits, purging/embargo, holdout guard, statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.validation import splits, stats
from fxlab.validation.splits import LeakageError


def test_walk_forward_no_overlap_and_purge():
    n, h = 5000, 10
    label_end = np.minimum(np.arange(n) + h, n - 1)
    sp = splits.walk_forward(n, 2000, 500, label_end=label_end, embargo=5)
    assert len(sp) == 6
    for s in sp:
        assert s.train.max() < s.test.min()
        assert (label_end[s.train] < s.test.min()).all(), "training labels reach into the test window"


def test_purged_kfold_embargo():
    n = 1000
    le = np.minimum(np.arange(n) + 20, n - 1)
    for s in splits.purged_kfold(n, 5, le, embargo=10):
        t0, t1 = s.test.min(), s.test.max()
        before = s.train[s.train < t0]
        after = s.train[s.train > t1]
        assert (le[before] < t0).all()
        assert (after > t1 + 10).all()


def test_split_check_detects_leak():
    sp = splits.Split(np.arange(0, 100), np.arange(100, 150), 0)
    le = np.arange(200) + 5
    with pytest.raises(LeakageError):
        sp.check(le)


def test_holdout_guard_on_index():
    idx = pd.date_range("2025-12-31 22:00", periods=5, freq="h", tz="UTC")
    with pytest.raises(LeakageError):
        splits.assert_dev_only(idx)


def test_loader_refuses_holdout():
    from fxlab.data.store import load_m1, HoldoutViolation, available
    sym = "EURUSD"
    if not available(sym):
        pytest.skip("no data")
    with pytest.raises(HoldoutViolation):
        load_m1(sym, "2025-12-01", "2026-02-01")


def test_nested_walk_forward_inner_inside_outer_train():
    n = 6000
    le = np.minimum(np.arange(n) + 5, n - 1)
    for outer, inner in splits.nested_walk_forward(n, 3000, 1000, 3, label_end=le, embargo=5):
        for s in inner:
            assert np.isin(s.train, outer.train).all() and np.isin(s.test, outer.train).all()
            assert s.test.max() < outer.test.min()


def test_psr_dsr_behaviour():
    rng = np.random.default_rng(0)
    good = rng.normal(0.1, 1, 2000)
    noise = rng.normal(0.0, 1, 2000)
    assert stats.psr(good) > 0.95
    assert 0.2 < stats.psr(noise) < 0.8 or stats.psr(noise) < 0.95
    assert stats.dsr(good, n_trials=1000, var_sr_trials=0.03 ** 2) < stats.psr(good)


def test_white_rc_and_spa():
    rng = np.random.default_rng(1)
    null = rng.normal(0, 1, (1500, 30))
    assert stats.white_reality_check(null, n_boot=500)["p_value"] > 0.05
    alt = null.copy(); alt[:, 7] += 0.15
    assert stats.white_reality_check(alt, n_boot=500)["p_value"] < 0.05
    assert stats.hansen_spa(alt, n_boot=500)["p_consistent"] < 0.05
    assert stats.hansen_spa(null, n_boot=500)["p_consistent"] > 0.05


def test_multiple_testing_corrections():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.5])
    bh, ho = stats.benjamini_hochberg(p), stats.holm(p)
    assert (bh >= p).all() and (ho >= bh - 1e-12).all()
    assert ho[0] == pytest.approx(0.005)


def test_block_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1, 1000)
    est, lo, hi = stats.block_bootstrap_ci(x, n_boot=500)
    assert lo < est < hi and lo < 0.1 < hi


def test_spa_pvalue_ordering():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (1000, 20)); X[:, :10] -= 0.3; X[:, 15] += 0.05
    r = stats.hansen_spa(X, n_boot=500)
    assert r["p_lower"] <= r["p_consistent"] + 1e-9 <= r["p_upper"] + 2e-9


def test_daily_aggregation_keeps_every_bar():
    """Regression test: D1 bars open Sunday 21:00/22:00 UTC (17:00 NY) for the Monday session; daily stats
    must keep them (an earlier version dropped them as 'weekend' rows)."""
    from fxlab.validation.metrics import daily, trading_day
    idx = pd.date_range("2015-01-04 22:00", periods=10, freq="D", tz="UTC")       # Sunday 22:00 UTC first
    ny_dow = idx.tz_convert("America/New_York").dayofweek
    idx = idx[(ny_dow <= 3) | (ny_dow == 6)]            # D1 opens exist Sun-Thu 17:00 NY only (as in real data)
    r = pd.Series(0.01, index=idx)
    d = daily(r)
    assert len(d) == len(r) and (d.index.dayofweek < 5).all()
    assert trading_day(pd.DatetimeIndex([pd.Timestamp("2015-01-04 22:30", tz="UTC")]))[0].dayofweek == 0
