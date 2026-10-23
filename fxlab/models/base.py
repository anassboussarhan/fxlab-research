"""Common model interface.

Every catalog model is a factory ``make(task, **params) -> estimator`` where the
estimator implements ``fit(X, y)``, ``predict(X)`` and, for classification,
``predict_proba(X)`` returning P(class=1). X rows are time-ordered; sequence
and series models may use earlier rows of the SAME call as context (never
later rows). Tasks:
  clf  - binary classification (direction)
  reg  - regression (return magnitude)
  q    - quantile regression (param alpha)
  vol  - volatility forecasting (y = realised vol)
  pol  - trading policy (predict returns a position in [-1, 1]; fit needs y = forward return)
"""
from __future__ import annotations

import numpy as np


class Model:
    task = "clf"
    is_sequence = False   # uses previous rows as context inside predict
    supports = ("clf", "reg")

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        p = np.asarray(self.predict(X), float)
        return 1 / (1 + np.exp(-p))

    def signal(self, X):
        """Map any model output to a raw signal in [-1, 1] (sign of edge)."""
        if self.task == "clf":
            return 2 * self.predict_proba(X) - 1
        if self.task == "pol":
            return np.clip(self.predict(X), -1, 1)
        return np.sign(np.asarray(self.predict(X), float))  # no test-block normalisation (would leak)


class SklearnWrap(Model):
    def __init__(self, est, task):
        self.est, self.task = est, task

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            try:
                self.est.fit(X, y, sample_weight=sample_weight)
                return self
            except TypeError:
                pass
        self.est.fit(X, y)
        return self

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        if hasattr(self.est, "predict_proba"):
            return self.est.predict_proba(X)[:, 1]
        if hasattr(self.est, "decision_function"):
            d = self.est.decision_function(X)
            return 1 / (1 + np.exp(-d))
        return super().predict_proba(X)


def to_binary(y):
    return (np.asarray(y) > 0).astype(int)
