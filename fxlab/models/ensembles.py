"""Ensembles. Base learners are other catalog models. Every meta-level fit uses
predictions that were produced OUT-OF-SAMPLE with respect to the base learner
(purged/embargoed contiguous folds for stacking, a chronological tail for
blending and gating)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from fxlab.registry import model, CATALOG
from fxlab.models.base import Model, to_binary

DEFAULT_BASE = ("M.LIN.LOGISTIC", "M.TREE.LGBM", "M.TREE.EXTRA")
DEFAULT_BASE_REG = ("M.LIN.RIDGE", "M.TREE.LGBM", "M.TREE.EXTRA")
ENS = dict(complexity="O(sum of base costs x folds)")


def _make(mid, task):
    if task != "clf" and mid == "M.LIN.LOGISTIC":
        mid = "M.LIN.RIDGE"
    e = CATALOG[mid]
    return e.fn(task=task, **e.params)


def _out(m, X, task):
    return m.predict_proba(X) if task == "clf" else np.asarray(m.predict(X), float)


class _Stack(Model):
    def __init__(self, task="clf", base=DEFAULT_BASE, k=4, embargo=50):
        self.task, self.base, self.k, self.embargo = task, list(base), k, embargo

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n = len(X)
        oof = np.full((n, len(self.base)), np.nan)
        b = np.linspace(0, n, self.k + 1).astype(int)
        for i in range(1, self.k):  # forward-chaining folds: train strictly before test (minus embargo)
            tr = np.arange(0, max(0, b[i] - self.embargo))
            te = np.arange(b[i], b[i + 1])
            if len(tr) < 200:
                continue
            for j, mid in enumerate(self.base):
                m = _make(mid, self.task).fit(X[tr], y[tr])
                oof[te, j] = _out(m, X[te], self.task)
        ok = ~np.isnan(oof).any(1) & np.isfinite(y.astype(float))
        self.meta = (LogisticRegression(C=1.0) if self.task == "clf" else Ridge(1.0)).fit(
            oof[ok], to_binary(y[ok]) if self.task == "clf" else y[ok].astype(float))
        self.models = [_make(mid, self.task).fit(X, y) for mid in self.base]
        return self

    def _P(self, X):
        return np.column_stack([_out(m, X, self.task) for m in self.models])

    def predict(self, X):
        return self.meta.predict(self._P(X))

    def predict_proba(self, X):
        return self.meta.predict_proba(self._P(X))[:, 1]


@model("M.ENS.STACKING", "ensemble", "stacked generalisation: forward-chaining out-of-fold base predictions (embargo) -> "
       "logistic/ridge meta-learner; base models refit on all training rows", params={"k": 4},
       space={"k": [3, 5]}, **ENS)
def stacking(task="clf", k=4, base=DEFAULT_BASE):
    return _Stack(task, base, k)


class _Blend(_Stack):
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        cut = int(0.7 * len(X))
        self.models = [_make(mid, self.task).fit(X[:cut], y[:cut]) for mid in self.base]
        P = self._P(X[cut + self.embargo:])
        yy = y[cut + self.embargo:]
        ok = np.isfinite(yy.astype(float))
        self.meta = (LogisticRegression(C=1.0) if self.task == "clf" else Ridge(1.0)).fit(
            P[ok], to_binary(yy[ok]) if self.task == "clf" else yy[ok].astype(float))
        return self


@model("M.ENS.BLENDING", "ensemble", "blending: base models fit on the first 70% of training, meta-learner on their "
       "predictions over the final 30% (after embargo)", **ENS)
def blending(task="clf", base=DEFAULT_BASE):
    return _Blend(task, base)


class _MoE(_Stack):
    """Mixture of experts: softmax gate (multinomial logistic on gate features = first g columns) trained to
    predict which expert had the lowest loss on each row of a held-out tail; prediction = gate-weighted mix."""

    def __init__(self, task="clf", base=DEFAULT_BASE, g=5):
        super().__init__(task, base)
        self.g = g

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        cut = int(0.7 * len(X))
        experts = [_make(mid, self.task).fit(X[:cut], y[:cut]) for mid in self.base]
        P = np.column_stack([_out(m, X[cut:], self.task) for m in experts])
        yy = y[cut:].astype(float)
        tgt = to_binary(yy) if self.task == "clf" else yy
        loss = (P - tgt[:, None]) ** 2
        best = loss.argmin(1)
        G = np.nan_to_num(X[cut:, : self.g])
        if len(np.unique(best)) > 1:
            self.gate = LogisticRegression(C=0.1, max_iter=2000).fit(G, best)
        else:
            self.gate = None
            self.only = int(best[0])
        self.models = [_make(mid, self.task).fit(X, y) for mid in self.base]
        return self

    def _w(self, X):
        if self.gate is None:
            w = np.zeros((len(X), len(self.models))); w[:, self.only] = 1
            return w
        pr = self.gate.predict_proba(np.nan_to_num(np.asarray(X, float)[:, : self.g]))
        w = np.zeros((len(X), len(self.models)))
        w[:, self.gate.classes_] = pr
        return w

    def predict(self, X):
        return (self._w(X) * np.column_stack([np.asarray(m.predict(X), float) for m in self.models])).sum(1)

    def predict_proba(self, X):
        return (self._w(X) * self._P(X)).sum(1)


@model("M.ENS.MIXTURE_OF_EXPERTS", "ensemble", "mixture of experts with a softmax gating network on regime features",
       params={"g": 5}, space={"g": [3, 5, 10]}, **ENS)
def moe(task="clf", g=5, base=DEFAULT_BASE):
    return _MoE(task, base, g)


class _DynSel(_Stack):
    """Dynamic model selection: at each row choose the expert with the best accuracy over the last w rows whose
    labels are already REALISED. Requires X[:, 0] = 1-bar return r_t; the label of row t-h is
    sum(r_{t-h+1..t}) which is known at t."""
    is_sequence = True

    def __init__(self, task="clf", base=DEFAULT_BASE, w=200, h=1):
        super().__init__(task, base)
        self.w, self.h = w, h

    def fit(self, X, y):
        self.models = [_make(mid, self.task).fit(X, y) for mid in self.base]
        self.default = 0
        return self

    def _choose(self, X, P):
        r = np.nan_to_num(np.asarray(X, float)[:, 0])
        c = np.r_[0, np.cumsum(r)]
        n = len(X)
        realised = np.full(n, np.nan)  # realised[t-h] known at t
        for t in range(self.h, n):
            realised[t - self.h] = c[t + 1] - c[t - self.h + 1]
        sig = 2 * P - 1 if self.task == "clf" else np.sign(P)
        score = np.sign(sig) * np.sign(realised)[:, None]  # +1 correct
        choice = np.zeros(n, int)
        for t in range(n):
            lo, hi = max(0, t - self.h - self.w), t - self.h  # rows with realised labels at time t
            if hi - lo >= 20:
                choice[t] = int(np.nanmean(score[lo:hi], axis=0).argmax())
            else:
                choice[t] = self.default
        return choice

    def predict_proba(self, X):
        P = self._P(X)
        ch = self._choose(X, P)
        return P[np.arange(len(P)), ch]

    def predict(self, X):
        P = np.column_stack([np.asarray(m.predict(X), float) for m in self.models])
        ch = self._choose(X, self._P(X))
        return P[np.arange(len(P)), ch]


@model("M.ENS.DYNAMIC_SELECTION", "ensemble", "dynamic model selection by trailing realised directional accuracy "
       "(only labels already realised at decision time)", params={"w": 200}, space={"w": [100, 200, 500]}, **ENS)
def dyn_sel(task="clf", w=200, h=1, base=DEFAULT_BASE):
    return _DynSel(task, base, w, h)


class _RegimeGated(Model):
    """Separate expert per regime: regime = tercile of column `col` using TRAINING-set quantile cut points."""

    def __init__(self, task="clf", base="M.TREE.LGBM", col=1, n_reg=3):
        self.task, self.base, self.col, self.n_reg = task, base, col, n_reg

    def _reg(self, X):
        v = np.nan_to_num(np.asarray(X, float)[:, self.col])
        return np.searchsorted(self.cuts, v)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        v = np.nan_to_num(X[:, self.col])
        self.cuts = np.quantile(v, np.linspace(0, 1, self.n_reg + 1)[1:-1])
        r = self._reg(X)
        self.models = {}
        for k in range(self.n_reg):
            m = r == k
            if m.sum() > 200 and len(np.unique(to_binary(y[m]))) > 1:
                self.models[k] = _make(self.base, self.task).fit(X[m], y[m])
        self.fallback = _make(self.base, self.task).fit(X, y)
        return self

    def _apply(self, X, fn):
        X = np.asarray(X, float)
        r = self._reg(X)
        out = np.asarray(fn(self.fallback, X), float).copy()
        for k, m in self.models.items():
            sel = r == k
            if sel.any():
                out[sel] = fn(m, X[sel])
        return out

    def predict(self, X):
        return self._apply(X, lambda m, Z: m.predict(Z))

    def predict_proba(self, X):
        return self._apply(X, lambda m, Z: m.predict_proba(Z))


@model("M.ENS.REGIME_GATED", "ensemble", "regime-gated ensemble: one expert per regime (terciles of a regime feature, "
       "cut points from training data), routing at prediction time", params={"col": 1, "n_reg": 3},
       space={"n_reg": [2, 3]}, **ENS)
def regime_gated(task="clf", col=1, n_reg=3, base="M.TREE.LGBM"):
    return _RegimeGated(task, base, col, n_reg)
