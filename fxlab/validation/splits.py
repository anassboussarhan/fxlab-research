"""Leakage-safe chronological splitting: walk-forward (rolling / expanding),
purged K-fold with embargo (López de Prado), nested walk-forward, and the
global 2026 holdout guard.

Every split is expressed in integer positions of a time-sorted index. When
labels span several bars, pass ``label_end`` (position of the last bar a label
depends on) so overlapping training samples are purged."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxlab.config import HOLDOUT_START


class LeakageError(AssertionError):
    pass


def assert_dev_only(index: pd.DatetimeIndex) -> None:
    if len(index) and index.max() >= pd.Timestamp(HOLDOUT_START):
        raise LeakageError(f"development data contains holdout timestamps (max {index.max()})")


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    fold: int

    def check(self, label_end: np.ndarray | None = None, embargo: int = 0) -> None:
        if len(self.train) == 0 or len(self.test) == 0:
            raise LeakageError("empty split")
        if np.intersect1d(self.train, self.test).size:
            raise LeakageError("train/test overlap")
        if label_end is not None:
            t0, t1 = self.test.min(), self.test.max()
            le = label_end[self.train]
            # a training label that ends inside [t0, t1 + embargo] or starts after t0 (and ends in) leaks
            bad = (le >= t0) & (self.train <= t1 + embargo)
            if bad.any():
                raise LeakageError(f"{bad.sum()} training labels overlap the test window")


def purge(train: np.ndarray, test: np.ndarray, label_end: np.ndarray | None, embargo: int) -> np.ndarray:
    """Remove training samples whose label interval overlaps [test_start, test_end + embargo]."""
    t0, t1 = test.min(), test.max()
    keep = np.ones(len(train), bool)
    le = label_end[train] if label_end is not None else train
    keep &= ~((train >= t0) & (train <= t1 + embargo))           # inside test window or embargo after it
    keep &= ~((train < t0) & (le >= t0))                          # labels running into the test window
    keep &= ~((train > t1) & (train <= t1 + embargo))
    return train[keep]


def walk_forward(n: int, train_size: int, test_size: int, step: int | None = None, expanding: bool = False,
                 label_end: np.ndarray | None = None, embargo: int = 0, min_train: int | None = None,
                 allow_partial: bool = False) -> list[Split]:
    """allow_partial=True also emits the final, incomplete test block (as live deployment would: the
    model trained at the block start keeps predicting new bars as they arrive)."""
    step = step or test_size
    out, k = [], 0
    start = train_size
    while start + (1 if allow_partial else test_size) <= n:
        test = np.arange(start, min(start + test_size, n))
        lo = 0 if expanding else start - train_size
        train = np.arange(lo, start)
        train = purge(train, test, label_end, embargo)
        if min_train is None or len(train) >= min_train:
            sp = Split(train, test, k)
            sp.check(label_end, embargo)
            out.append(sp)
            k += 1
        start += step
    return out


def purged_kfold(n: int, k: int = 5, label_end: np.ndarray | None = None, embargo: int = 0) -> list[Split]:
    """Contiguous folds; training data on both sides of the test fold, purged and embargoed.
    Use only INSIDE a development training window (e.g. for hyper-parameter selection), never across
    the final test period."""
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        test = np.arange(bounds[i], bounds[i + 1])
        train = np.setdiff1d(np.arange(n), test)
        train = purge(train, test, label_end, embargo)
        sp = Split(train, test, i)
        sp.check(label_end, embargo)
        out.append(sp)
    return out


def nested_walk_forward(n: int, outer_train: int, outer_test: int, inner_k: int = 3, expanding: bool = True,
                        label_end: np.ndarray | None = None, embargo: int = 0):
    """Outer chronological walk-forward; for each outer split, inner purged K-fold indices
    (relative to the outer training positions) for model/parameter selection."""
    res = []
    for sp in walk_forward(n, outer_train, outer_test, expanding=expanding, label_end=label_end, embargo=embargo):
        tr = sp.train
        le_inner = None if label_end is None else np.searchsorted(tr, np.minimum(label_end[tr], tr.max()), side="right") - 1
        inner = purged_kfold(len(tr), inner_k, le_inner, embargo)
        res.append((sp, [Split(tr[s.train], tr[s.test], s.fold) for s in inner]))
    return res


def time_walk_forward(index: pd.DatetimeIndex, train_years: float, test_months: int, expanding: bool = False,
                      label_end: np.ndarray | None = None, embargo: int = 0, start=None, end=None) -> list[Split]:
    """Calendar-anchored walk-forward (e.g. 3y train, 6m test, rolling), development data only."""
    assert_dev_only(index)
    t = pd.Series(np.arange(len(index)), index=index)
    first = index.min() if start is None else pd.Timestamp(start, tz="UTC")
    last = index.max() if end is None else pd.Timestamp(end, tz="UTC")
    test_start = first + pd.DateOffset(months=int(train_years * 12))
    out, k = [], 0
    while test_start < last:
        test_end = test_start + pd.DateOffset(months=test_months)
        tr_lo = first if expanding else test_start - pd.DateOffset(months=int(train_years * 12))
        train = t[(t.index >= tr_lo) & (t.index < test_start)].to_numpy()
        test = t[(t.index >= test_start) & (t.index < test_end)].to_numpy()
        if len(train) and len(test):
            train = purge(train, test, label_end, embargo)
            sp = Split(train, test, k)
            sp.check(label_end, embargo)
            out.append(sp)
            k += 1
        test_start = test_end
    return out
