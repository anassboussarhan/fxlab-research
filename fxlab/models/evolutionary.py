"""Symbolic and evolutionary methods (implemented from scratch, deterministic seeds).

* genetic programming   - tree GP (tournament selection, subtree crossover/mutation, parsimony pressure)
* symbolic regression   - SINDy-style sequentially-thresholded least squares over a function library
* grammatical evolution - integer genomes mapped through a BNF grammar to expressions
* evolutionary optimiser - differential evolution of linear signal weights maximising training Sharpe
Fitness is always computed on TRAINING rows only (correlation with target / net Sharpe)."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from fxlab.registry import model
from fxlab.models.base import Model, to_binary

E = dict(complexity="O(generations · population · n)")

UN = {"neg": np.negative, "tanh": np.tanh, "sq": lambda a: np.clip(a, -10, 10) ** 2, "abs": np.abs,
      "sign": np.sign}
BI = {"add": np.add, "sub": np.subtract, "mul": lambda a, b: np.clip(a * b, -1e3, 1e3),
      "div": lambda a, b: a / (np.abs(b) + 1.0), "max": np.maximum, "min": np.minimum}


def _eval(t, Z):
    if t[0] == "x":
        return Z[:, t[1]]
    if t[0] == "c":
        return np.full(len(Z), t[1])
    if t[0] in UN:
        return UN[t[0]](_eval(t[1], Z))
    return BI[t[0]](_eval(t[1], Z), _eval(t[2], Z))


def _size(t):
    return 1 if t[0] in ("x", "c") else 1 + sum(_size(c) for c in t[1:])


def _fitness(pred, y, task):
    if not np.isfinite(pred).all() or pred.std() == 0:
        return -1.0
    yy = to_binary(y) - 0.5 if task == "clf" else y
    return abs(np.corrcoef(pred, yy)[0, 1])


class _GP(Model):
    def __init__(self, task="clf", pop=200, gens=20, max_depth=4, seed=0, parsimony=0.002, max_train=20000):
        self.task, self.pop, self.gens, self.md, self.seed, self.pars = task, pop, gens, max_depth, seed, parsimony
        self.max_train = max_train

    def _rand(self, rng, p, d):
        if d == 0 or (d < self.md and rng.random() < 0.3):
            return ("x", int(rng.integers(p))) if rng.random() < 0.85 else ("c", float(rng.normal()))
        if rng.random() < 0.3:
            return (rng.choice(list(UN)), self._rand(rng, p, d - 1))
        return (rng.choice(list(BI)), self._rand(rng, p, d - 1), self._rand(rng, p, d - 1))

    def _nodes(self, t, path=()):
        yield path, t
        if t[0] not in ("x", "c"):
            for i, c in enumerate(t[1:], 1):
                yield from self._nodes(c, path + (i,))

    def _replace(self, t, path, new):
        if not path:
            return new
        lst = list(t)
        lst[path[0]] = self._replace(t[path[0]], path[1:], new)
        return tuple(lst)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X, y = np.asarray(X, float)[-self.max_train:], np.asarray(y, float)[-self.max_train:]
        ok = np.isfinite(y)
        self.sc = StandardScaler().fit(np.nan_to_num(X))
        Z = np.clip(self.sc.transform(np.nan_to_num(X)), -10, 10)[ok]
        y = y[ok]
        p = Z.shape[1]
        popn = [self._rand(rng, p, self.md) for _ in range(self.pop)]
        score = lambda t: _fitness(_eval(t, Z), y, self.task) - self.pars * _size(t)
        fit = np.array([score(t) for t in popn])
        for _ in range(self.gens):
            new = [popn[int(fit.argmax())]]  # elitism
            while len(new) < self.pop:
                a, b = (popn[max(rng.choice(self.pop, 4), key=lambda i: fit[i])] for _ in range(2))
                r = rng.random()
                if r < 0.7:  # subtree crossover
                    na = list(self._nodes(a)); nb = list(self._nodes(b))
                    pa, _ = na[rng.integers(len(na))]; _, sb = nb[rng.integers(len(nb))]
                    child = self._replace(a, pa, sb)
                elif r < 0.95:  # subtree mutation
                    na = list(self._nodes(a))
                    pa, _ = na[rng.integers(len(na))]
                    child = self._replace(a, pa, self._rand(rng, p, 2))
                else:
                    child = a
                if _size(child) <= 31:
                    new.append(child)
            popn = new
            fit = np.array([score(t) for t in popn])
        self.expr = popn[int(fit.argmax())]
        pred = _eval(self.expr, Z)
        yy = to_binary(y) - 0.5 if self.task == "clf" else y
        self.sign = np.sign(np.corrcoef(pred, yy)[0, 1]) or 1.0
        self.mu, self.sd = pred.mean(), pred.std() + 1e-12
        self.ys = np.std(y) if self.task != "clf" else 1.0
        return self

    def predict(self, X):
        Z = np.clip(self.sc.transform(np.nan_to_num(np.asarray(X, float))), -10, 10)
        return self.sign * (_eval(self.expr, Z) - self.mu) / self.sd * (0.1 * self.ys)

    def predict_proba(self, X):
        Z = np.clip(self.sc.transform(np.nan_to_num(np.asarray(X, float))), -10, 10)
        return 1 / (1 + np.exp(-self.sign * (_eval(self.expr, Z) - self.mu) / self.sd))


@model("M.EVO.GP", "symbolic_evolutionary", "tree genetic programming over features (tournament 4, elitism, subtree "
       "crossover 0.7 / mutation 0.25, parsimony), fitness |corr(expr, target)| on training rows",
       params={"pop": 200, "gens": 20}, space={"pop": [100, 300], "gens": [10, 30]}, **E)
def gp(task="clf", pop=200, gens=20, **kw):
    return _GP(task, pop, gens, **kw)


class _SINDy(Model):
    """Symbolic regression by sparse regression over a library {x_i, x_i^2, x_i x_j (top-m), tanh x_i, |x_i|}
    with sequentially thresholded least squares (Brunton et al. 2016)."""

    def __init__(self, task="reg", thr=0.05, m=6, ridge=1e-3):
        self.task, self.thr, self.m, self.ridge = task, thr, m, ridge

    def _lib(self, Z):
        cols = [np.ones(len(Z))]
        names = ["1"]
        for i in range(Z.shape[1]):
            cols += [Z[:, i], np.clip(Z[:, i], -5, 5) ** 2, np.tanh(Z[:, i]), np.abs(Z[:, i])]
            names += [f"x{i}", f"x{i}^2", f"tanh(x{i})", f"|x{i}|"]
        m = min(self.m, Z.shape[1])
        for i in range(m):
            for j in range(i + 1, m):
                cols.append(Z[:, i] * Z[:, j]); names.append(f"x{i}*x{j}")
        return np.column_stack(cols), names

    def fit(self, X, y):
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        self.sc = StandardScaler().fit(np.nan_to_num(X))
        Z = np.clip(self.sc.transform(np.nan_to_num(X)), -10, 10)
        Th, self.names = self._lib(Z)
        self.lsc = StandardScaler().fit(Th)
        Th = self.lsc.transform(Th)[ok]
        yy = (to_binary(y[ok]) - 0.5) if self.task == "clf" else y[ok] / (np.std(y[ok]) + 1e-12)
        self.ys = 1.0 if self.task == "clf" else np.std(y[ok])
        A = Th.T @ Th + self.ridge * len(Th) * np.eye(Th.shape[1])
        xi = np.linalg.solve(A, Th.T @ yy)
        for _ in range(10):
            small = np.abs(xi) < self.thr * np.abs(xi).max()
            xi[small] = 0
            big = ~small
            if big.sum() == 0:
                break
            xi[big] = np.linalg.solve(A[np.ix_(big, big)], Th[:, big].T @ yy)
        self.xi = xi
        return self

    def equation(self):
        return " + ".join(f"{c:.3g}*{n}" for c, n in zip(self.xi, self.names) if c != 0)

    def _raw(self, X):
        Z = np.clip(self.sc.transform(np.nan_to_num(np.asarray(X, float))), -10, 10)
        return self.lsc.transform(self._lib(Z)[0]) @ self.xi

    def predict(self, X):
        return self._raw(X) * self.ys

    def predict_proba(self, X):
        return np.clip(self._raw(X) + 0.5, 0, 1)


@model("M.EVO.SYMBOLIC_REGRESSION", "symbolic_evolutionary", "SINDy-style symbolic regression: sparse (STLSQ) "
       "coefficients over a library of polynomial / tanh / abs / pairwise-product terms; human-readable equation",
       params={"thr": 0.05}, space={"thr": [0.02, 0.05, 0.1]}, complexity="O(n·lib^2)")
def sindy(task="reg", thr=0.05, **kw):
    return _SINDy(task, thr, **kw)


GRAMMAR = {
    "<e>": [["<b>", "(", "<e>", ",", "<e>", ")"], ["<u>", "(", "<e>", ")"], ["<v>"], ["<v>"]],
    "<b>": [["add"], ["sub"], ["mul"], ["div"], ["max"], ["min"]],
    "<u>": [["tanh"], ["neg"], ["sign"], ["abs"]],
}


class _GE(_GP):
    """Grammatical evolution: genome = list of codons; derivation by leftmost expansion with
    rule = codon mod #choices; wraps at most twice. Evolution by one-point crossover + codon mutation."""

    def __init__(self, task="clf", pop=200, gens=20, glen=40, seed=0, **kw):
        super().__init__(task, pop, gens, seed=seed)
        self.glen = glen

    def _derive(self, g, p):
        out, i, wraps = [], 0, 0

        def expand(sym, depth):
            nonlocal i, wraps
            if sym not in GRAMMAR:
                return sym
            if i >= len(g):
                i, wraps = 0, wraps + 1
                if wraps > 2:
                    raise ValueError
            opts = GRAMMAR[sym] if depth < 5 or sym != "<e>" else [["<v>"]]
            ch = opts[g[i] % len(opts)]
            i += 1
            if sym == "<e>" and ch == ["<v>"]:
                v = ("x", int(g[i % len(g)] % p)); i += 1
                return v
            if ch[0] == "<b>":
                op = expand("<b>", depth + 1)
                return (op[0] if isinstance(op, list) else op, expand("<e>", depth + 1), expand("<e>", depth + 1))
            if ch[0] == "<u>":
                op = expand("<u>", depth + 1)
                return (op[0] if isinstance(op, list) else op, expand("<e>", depth + 1))
            return ch[0]
        return expand("<e>", 0)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, float)[-self.max_train:]
        X = np.asarray(X, float)[-self.max_train:]
        ok = np.isfinite(y)
        self.sc = StandardScaler().fit(np.nan_to_num(X))
        Z = np.clip(self.sc.transform(np.nan_to_num(X)), -10, 10)[ok]
        y = y[ok]
        p = Z.shape[1]

        def score(g):
            try:
                t = self._derive(g, p)
            except (ValueError, RecursionError):
                return -1.0, None
            return _fitness(_eval(t, Z), y, self.task) - self.pars * _size(t), t

        pop = [rng.integers(0, 256, self.glen) for _ in range(self.pop)]
        sc = [score(g) for g in pop]
        for _ in range(self.gens):
            fit = np.array([s[0] for s in sc])
            new = [pop[int(fit.argmax())]]
            while len(new) < self.pop:
                a = pop[max(rng.choice(self.pop, 4), key=lambda i: fit[i])]
                b = pop[max(rng.choice(self.pop, 4), key=lambda i: fit[i])]
                cut = rng.integers(1, self.glen)
                child = np.r_[a[:cut], b[cut:]]
                mut = rng.random(self.glen) < 0.05
                child[mut] = rng.integers(0, 256, mut.sum())
                new.append(child)
            pop = new
            sc = [score(g) for g in pop]
        best = max(sc, key=lambda s: s[0])
        self.expr = best[1] if best[1] is not None else ("x", 0)
        pred = _eval(self.expr, Z)
        yy = to_binary(y) - 0.5 if self.task == "clf" else y
        c = np.corrcoef(pred, yy)[0, 1] if pred.std() > 0 else 0.0
        self.sign = np.sign(c) or 1.0
        self.mu, self.sd = pred.mean(), pred.std() + 1e-12
        self.ys = np.std(y) if self.task != "clf" else 1.0
        return self


@model("M.EVO.GRAMMATICAL", "symbolic_evolutionary", "grammatical evolution: codon genomes mapped through a BNF grammar "
       "(binary/unary operators over features) with wrapping; one-point crossover + 5% codon mutation",
       params={"pop": 200, "gens": 20}, space={"pop": [100, 300], "gens": [10, 30]}, **E)
def ge(task="clf", pop=200, gens=20, **kw):
    return _GE(task, pop, gens, **kw)


class _DESharpe(Model):
    """Evolutionary optimisation (scipy differential evolution) of a linear signal w'x passed through tanh,
    maximising the training Sharpe of position * forward return minus a turnover cost."""
    task = "pol"

    def __init__(self, task="pol", cost=0.0001, maxiter=30, seed=0, max_train=20000):
        self.cost, self.maxiter, self.seed, self.max_train = cost, maxiter, seed, max_train

    def fit(self, X, y):
        from scipy.optimize import differential_evolution
        X, y = np.asarray(X, float)[-self.max_train:], np.asarray(y, float)[-self.max_train:]
        ok = np.isfinite(y)
        self.sc = StandardScaler().fit(np.nan_to_num(X))
        Z = np.clip(self.sc.transform(np.nan_to_num(X)), -10, 10)[ok]
        y = y[ok]

        def neg_sharpe(w):
            pos = np.tanh(Z @ w[:-1] + w[-1])
            pnl = pos * y - self.cost * np.abs(np.diff(np.r_[0, pos]))
            return -(pnl.mean() / (pnl.std() + 1e-12))
        p = Z.shape[1]
        res = differential_evolution(neg_sharpe, [(-2, 2)] * p + [(-0.5, 0.5)], maxiter=self.maxiter, popsize=10,
                                     seed=self.seed, polish=False, tol=1e-6)
        self.w = res.x
        return self

    def predict(self, X):
        Z = np.clip(self.sc.transform(np.nan_to_num(np.asarray(X, float))), -10, 10)
        return np.tanh(Z @ self.w[:-1] + self.w[-1])

    def predict_proba(self, X):
        return (self.predict(X) + 1) / 2


@model("M.EVO.DE_SHARPE", "symbolic_evolutionary", "differential-evolution optimisation of linear signal weights "
       "(tanh position) maximising training net Sharpe with turnover penalty", params={"maxiter": 30},
       space={"maxiter": [20, 50]}, extra={"tasks": ["pol"]}, **E)
def de_sharpe(task="pol", maxiter=30, **kw):
    return _DESharpe(task, maxiter=maxiter, **kw)
