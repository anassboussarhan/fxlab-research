"""Linear/GLM, instance-based, tree, kernel and probabilistic tabular models."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer

from fxlab.registry import model
from fxlab.models.base import Model, SklearnWrap, to_binary

C = dict(complexity="O(n p)")


def _scaled(est):
    return make_pipeline(StandardScaler(), est)


# ------------------------------------------------------------------------------------------ linear / GLM
@model("M.LIN.OLS", "linear", "ordinary least squares y = Xb (reg); clf via sign of fitted regression", **C)
def ols(task="reg", **kw):
    from sklearn.linear_model import LinearRegression
    return SklearnWrap(_scaled(LinearRegression()), "reg")


@model("M.LIN.RIDGE", "linear", "L2-penalised least squares min |y-Xb|^2 + a|b|^2", params={"alpha": 10.0},
       space={"alpha": [0.1, 1, 10, 100, 1000]}, **C)
def ridge(task="reg", alpha=10.0):
    from sklearn.linear_model import Ridge, RidgeClassifier
    return SklearnWrap(_scaled(RidgeClassifier(alpha=alpha) if task == "clf" else Ridge(alpha=alpha)), task)


@model("M.LIN.LASSO", "linear", "L1-penalised least squares (sparse b); L1-logistic for clf", params={"alpha": 1e-4},
       space={"alpha": [1e-5, 1e-4, 1e-3]}, **C)
def lasso(task="reg", alpha=1e-4):
    from sklearn.linear_model import Lasso, LogisticRegression
    if task == "clf":
        return SklearnWrap(_scaled(LogisticRegression(penalty="l1", C=1 / (alpha * 1e4), solver="liblinear")), "clf")
    return SklearnWrap(_scaled(Lasso(alpha=alpha, max_iter=5000)), "reg")


@model("M.LIN.ELASTICNET", "linear", "elastic net a(r|b|_1 + (1-r)/2|b|^2)", params={"alpha": 1e-4, "l1_ratio": 0.5},
       space={"alpha": [1e-5, 1e-4, 1e-3], "l1_ratio": [0.2, 0.5, 0.8]}, **C)
def enet(task="reg", alpha=1e-4, l1_ratio=0.5):
    from sklearn.linear_model import ElasticNet, LogisticRegression
    if task == "clf":
        return SklearnWrap(_scaled(LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, C=1 / (alpha * 1e4),
                                                      solver="saga", max_iter=2000)), "clf")
    return SklearnWrap(_scaled(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)), "reg")


@model("M.LIN.LOGISTIC", "linear", "L2 logistic regression P(y=1)=sigma(Xb)", params={"C": 0.1}, space={"C": [0.01, 0.1, 1]},
       extra={"tasks": ["clf"]}, **C)
def logistic(task="clf", C=0.1):
    from sklearn.linear_model import LogisticRegression
    return SklearnWrap(_scaled(LogisticRegression(C=C, max_iter=2000)), "clf")


@model("M.LIN.BAYES_RIDGE", "linear", "Bayesian ridge regression (Gamma hyper-priors on noise and weight precision, "
       "evidence maximisation)", **C)
def bayes_ridge(task="reg"):
    from sklearn.linear_model import BayesianRidge
    return SklearnWrap(_scaled(BayesianRidge()), "reg")


@model("M.LIN.HUBER", "linear", "robust regression with Huber loss (epsilon=1.35)", params={"epsilon": 1.35},
       space={"epsilon": [1.1, 1.35, 2.0]}, **C)
def huber(task="reg", epsilon=1.35):
    from sklearn.linear_model import HuberRegressor
    return SklearnWrap(_scaled(HuberRegressor(epsilon=epsilon, max_iter=500)), "reg")


class _QuantReg(Model):
    task = "q"

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y):
        import statsmodels.api as sm
        self.sc = StandardScaler().fit(X)
        Xs = sm.add_constant(self.sc.transform(X), has_constant="add")
        self.res = sm.QuantReg(np.asarray(y, float), Xs).fit(q=self.alpha, max_iter=2000)
        return self

    def predict(self, X):
        import statsmodels.api as sm
        return self.res.predict(sm.add_constant(self.sc.transform(X), has_constant="add"))


@model("M.LIN.QUANTILE", "linear", "linear quantile regression minimising pinball loss at quantile alpha (statsmodels QuantReg)",
       params={"alpha": 0.5}, space={"alpha": [0.05, 0.25, 0.5, 0.75, 0.95]}, extra={"tasks": ["q", "reg"]}, **C)
def quantile(task="q", alpha=0.5):
    return _QuantReg(alpha)


@model("M.LIN.GAM", "linear", "generalised additive model: per-feature cubic B-spline basis (5 knots) + ridge penalty "
       "(penalised regression splines); logistic link for clf", params={"n_knots": 5, "alpha": 10.0},
       space={"n_knots": [4, 6], "alpha": [1, 10, 100]}, **C)
def gam(task="reg", n_knots=5, alpha=10.0):
    from sklearn.linear_model import Ridge, LogisticRegression
    head = LogisticRegression(C=1 / alpha, max_iter=2000) if task == "clf" else Ridge(alpha=alpha)
    return SklearnWrap(make_pipeline(StandardScaler(), SplineTransformer(n_knots=n_knots, degree=3), head), task)


@model("M.IB.KNN", "instance_based", "k-nearest-neighbour (distance-weighted) on standardised features", params={"k": 50},
       space={"k": [20, 50, 200]}, complexity="O(n^2 p) predict")
def knn(task="clf", k=50):
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    est = KNeighborsClassifier(k, weights="distance") if task == "clf" else KNeighborsRegressor(k, weights="distance")
    return SklearnWrap(_scaled(est), task)


# ------------------------------------------------------------------------------------------ trees
TS = dict(complexity="O(n p log n · trees)")


@model("M.TREE.CART", "tree", "single CART decision tree (depth-limited)", params={"max_depth": 4},
       space={"max_depth": [3, 4, 6], "min_samples_leaf": [100, 500]}, **TS)
def cart(task="clf", max_depth=4, min_samples_leaf=200):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    k = DecisionTreeClassifier if task == "clf" else DecisionTreeRegressor
    return SklearnWrap(k(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0), task)


@model("M.TREE.RF", "tree", "random forest (bootstrap + feature subsampling)", params={"n": 200, "max_depth": 6},
       space={"n": [100, 300], "max_depth": [4, 6, 10], "min_samples_leaf": [50, 200]}, **TS)
def rf(task="clf", n=200, max_depth=6, min_samples_leaf=100):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    k = RandomForestClassifier if task == "clf" else RandomForestRegressor
    return SklearnWrap(k(n, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                         n_jobs=1, random_state=0), task)


@model("M.TREE.EXTRA", "tree", "extremely randomised trees (random split thresholds, no bootstrap)",
       params={"n": 200, "max_depth": 6}, space={"n": [100, 300], "max_depth": [4, 6, 10]}, **TS)
def extra(task="clf", n=200, max_depth=6, min_samples_leaf=100):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    k = ExtraTreesClassifier if task == "clf" else ExtraTreesRegressor
    return SklearnWrap(k(n, max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_jobs=1, random_state=0), task)


@model("M.TREE.GBM", "tree", "gradient boosting (sklearn, exact splits, shrinkage, subsample 0.7)",
       params={"n": 200, "lr": 0.05, "max_depth": 3}, space={"n": [100, 300], "lr": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4]}, **TS)
def gbm(task="clf", n=200, lr=0.05, max_depth=3):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    k = GradientBoostingClassifier if task == "clf" else GradientBoostingRegressor
    return SklearnWrap(k(n_estimators=n, learning_rate=lr, max_depth=max_depth, subsample=0.7, random_state=0), task)


@model("M.TREE.HGB", "tree", "histogram gradient boosting (binned features, leaf-wise growth, early stopping)",
       params={"lr": 0.05, "max_leaf_nodes": 15}, space={"lr": [0.02, 0.05], "max_leaf_nodes": [7, 15, 31]}, **TS)
def hgb(task="clf", lr=0.05, max_leaf_nodes=15):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    k = HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor
    return SklearnWrap(k(learning_rate=lr, max_leaf_nodes=max_leaf_nodes, l2_regularization=1.0, random_state=0,
                         early_stopping=True, validation_fraction=0.15), task)


@model("M.TREE.XGB", "tree", "XGBoost (second-order boosting, hist tree method)", params={"n": 300, "lr": 0.03, "max_depth": 3},
       space={"n": [200, 500], "lr": [0.01, 0.03], "max_depth": [2, 3, 5]}, requires=("xgboost",), **TS)
def xgb(task="clf", n=300, lr=0.03, max_depth=3):
    import xgboost as xg
    k = xg.XGBClassifier if task == "clf" else xg.XGBRegressor
    return SklearnWrap(k(n_estimators=n, learning_rate=lr, max_depth=max_depth, subsample=0.7, colsample_bytree=0.7,
                         tree_method="hist", n_jobs=1, random_state=0, verbosity=0), task)


@model("M.TREE.LGBM", "tree", "LightGBM (gradient-based one-side sampling, leaf-wise)", params={"n": 300, "lr": 0.03, "leaves": 15},
       space={"n": [200, 500], "lr": [0.01, 0.03], "leaves": [7, 15, 31]}, requires=("lightgbm",), **TS)
def lgbm(task="clf", n=300, lr=0.03, leaves=15):
    import lightgbm as lg
    k = lg.LGBMClassifier if task == "clf" else lg.LGBMRegressor
    return SklearnWrap(k(n_estimators=n, learning_rate=lr, num_leaves=leaves, subsample=0.7, subsample_freq=1,
                         colsample_bytree=0.7, min_child_samples=100, n_jobs=1, random_state=0, verbose=-1), task)


@model("M.TREE.CATBOOST", "tree", "CatBoost (ordered boosting, symmetric trees)", params={"n": 300, "lr": 0.05, "depth": 4},
       space={"n": [200, 500], "lr": [0.03, 0.05], "depth": [3, 4, 6]}, requires=("catboost",), **TS)
def catb(task="clf", n=300, lr=0.05, depth=4):
    import catboost as cb
    k = cb.CatBoostClassifier if task == "clf" else cb.CatBoostRegressor
    return SklearnWrap(k(iterations=n, learning_rate=lr, depth=depth, verbose=0, random_seed=0, thread_count=1), task)


@model("M.TREE.ADABOOST", "tree", "AdaBoost (SAMME) on depth-2 trees", params={"n": 200, "lr": 0.1},
       space={"n": [100, 300], "lr": [0.05, 0.1, 0.5]}, **TS)
def ada(task="clf", n=200, lr=0.1):
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    if task == "clf":
        return SklearnWrap(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=n, learning_rate=lr,
                                              random_state=0), task)
    return SklearnWrap(AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=n, learning_rate=lr,
                                         random_state=0), task)


# ------------------------------------------------------------------------------------------ kernel
def _sub(X, y, m, seed=0):
    if len(X) <= m:
        return X, y
    i = np.random.default_rng(seed).choice(len(X), m, replace=False)
    i.sort()
    return X[i], np.asarray(y)[i]


class _Subsampled(SklearnWrap):
    """Kernel methods are O(n^2..n^3): fit on a random subsample of at most m rows."""

    def __init__(self, est, task, m):
        super().__init__(est, task)
        self.m = m

    def fit(self, X, y, sample_weight=None):
        Xs, ys = _sub(np.asarray(X), y, self.m)
        self.est.fit(Xs, ys)
        return self


@model("M.KER.LINEAR_SVM", "kernel", "linear support-vector machine (hinge loss, L2)", params={"C": 0.01},
       space={"C": [0.001, 0.01, 0.1]}, complexity="O(n p)")
def lsvm(task="clf", C=0.01):
    from sklearn.svm import LinearSVC, LinearSVR
    est = LinearSVC(C=C, max_iter=5000) if task == "clf" else LinearSVR(C=C, max_iter=5000)
    return SklearnWrap(_scaled(est), task)


@model("M.KER.RBF_SVM", "kernel", "RBF-kernel SVM classifier (Platt-free: decision function -> logistic), fit on <= m rows",
       params={"C": 1.0, "m": 4000}, space={"C": [0.1, 1, 10]}, complexity="O(m^2..m^3)", extra={"tasks": ["clf"]})
def rbf_svm(task="clf", C=1.0, m=4000):
    from sklearn.svm import SVC
    return _Subsampled(_scaled(SVC(C=C, kernel="rbf", gamma="scale")), "clf", m)


@model("M.KER.SVR", "kernel", "epsilon-support vector regression (RBF) fit on <= m rows", params={"C": 1.0, "m": 4000},
       space={"C": [0.1, 1, 10]}, complexity="O(m^2..m^3)", extra={"tasks": ["reg"]})
def svr(task="reg", C=1.0, m=4000):
    from sklearn.svm import SVR
    return _Subsampled(_scaled(SVR(C=C, epsilon=0.1)), "reg", m)


@model("M.KER.KRR", "kernel", "kernel ridge regression (RBF) fit on <= m rows", params={"alpha": 1.0, "m": 3000},
       space={"alpha": [0.1, 1, 10]}, complexity="O(m^3)", extra={"tasks": ["reg"]})
def krr(task="reg", alpha=1.0, m=3000):
    from sklearn.kernel_ridge import KernelRidge
    return _Subsampled(_scaled(KernelRidge(alpha=alpha, kernel="rbf")), "reg", m)


@model("M.KER.GP", "kernel", "Gaussian process (RBF + white noise kernel, ML-II hyper-parameters) on <= m rows; "
       "GP classifier (Laplace) for clf", params={"m": 1000}, space={"m": [500, 1000]}, complexity="O(m^3)")
def gp(task="reg", m=1000):
    from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    if task == "clf":
        return _Subsampled(_scaled(GaussianProcessClassifier(RBF(), random_state=0)), "clf", m)
    return _Subsampled(_scaled(GaussianProcessRegressor(RBF() + WhiteKernel(), normalize_y=True, random_state=0)), "reg", m)


# ------------------------------------------------------------------------------------------ probabilistic
@model("M.PB.GNB", "probabilistic", "Gaussian naive Bayes (class-conditional independent Gaussians)", extra={"tasks": ["clf"]})
def gnb(task="clf"):
    from sklearn.naive_bayes import GaussianNB
    return SklearnWrap(_scaled(GaussianNB()), "clf")


@model("M.PB.QDA", "probabilistic", "Bayesian Gaussian classifier with class-specific full covariances (QDA, shrinkage reg)",
       params={"reg": 0.1}, space={"reg": [0.01, 0.1, 0.5]}, extra={"tasks": ["clf"]})
def qda(task="clf", reg=0.1):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return SklearnWrap(_scaled(QuadraticDiscriminantAnalysis(reg_param=reg)), "clf")


class _GMMClassifier(Model):
    task = "clf"

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        yb = to_binary(y)
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.prior = yb.mean()
        self.g1 = GaussianMixture(self.k, covariance_type="diag", random_state=0).fit(Z[yb == 1])
        self.g0 = GaussianMixture(self.k, covariance_type="diag", random_state=0).fit(Z[yb == 0])
        return self

    def predict_proba(self, X):
        Z = self.sc.transform(X)
        l1 = self.g1.score_samples(Z) + np.log(self.prior)
        l0 = self.g0.score_samples(Z) + np.log(1 - self.prior)
        return 1 / (1 + np.exp(np.clip(l0 - l1, -50, 50)))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


@model("M.PB.GMM_CLASSIFIER", "probabilistic", "generative classifier with class-conditional Gaussian mixtures (k comps, diag)",
       params={"k": 3}, space={"k": [2, 3, 5]}, extra={"tasks": ["clf"]})
def gmm_clf(task="clf", k=3):
    return _GMMClassifier(k)


class _HMMPredictor(Model):
    """Regime model: Gaussian HMM on (target-proxy return, features[:2]); prediction = filtered state
    probabilities @ per-state mean of the training target (state-conditional expectation)."""
    is_sequence = True

    def __init__(self, k=2, task="reg"):
        self.k, self.task = k, task

    def fit(self, X, y):
        from hmmlearn.hmm import GaussianHMM
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)[:, : min(3, X.shape[1])]
        self.m = GaussianHMM(self.k, covariance_type="diag", n_iter=50, random_state=0).fit(Z)
        post = self._filter(Z)
        yy = np.asarray(y, float) if self.task != "clf" else to_binary(y).astype(float)
        self.state_mean = (post * yy[:, None]).sum(0) / (post.sum(0) + 1e-12)
        return self

    def _filter(self, Z):
        from fxlab.features.regime import _forward_filter
        covs = np.array([np.diag(c) if c.ndim == 1 else c for c in self.m.covars_])
        if covs.ndim == 2:
            covs = np.array([np.diag(c) for c in covs])
        return _forward_filter(Z.astype(float), self.m.startprob_, self.m.transmat_, self.m.means_, covs)

    def predict(self, X):
        Z = self.sc.transform(X)[:, : min(3, X.shape[1])]
        return self._filter(Z) @ self.state_mean

    def predict_proba(self, X):
        return np.clip(self.predict(X), 0, 1)


@model("M.PB.HMM", "probabilistic", "hidden Markov model (Gaussian, diag) on leading features; forward-filtered state "
       "posteriors times state-conditional mean target", params={"k": 2}, space={"k": [2, 3, 4]}, requires=("hmmlearn",))
def hmm(task="reg", k=2):
    return _HMMPredictor(k, task)


class _DLM(Model):
    """Bayesian dynamic linear model: y_t = x_t' b_t + e_t, b_t = b_{t-1} + w_t (random-walk coefficients),
    estimated with a Kalman filter; prediction uses coefficients filtered through the previous row only."""
    is_sequence = True

    def __init__(self, delta=1e-4, task="reg", k=5):
        self.delta, self.task, self.k = delta, task, k

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = np.c_[np.ones(len(X)), self.sc.transform(X)[:, : self.k]]
        yy = np.asarray(y, float) if self.task != "clf" else to_binary(y) - 0.5
        p = Z.shape[1]
        b = np.zeros(p); P = np.eye(p)
        self.r = np.nanvar(yy) + 1e-12
        for t in range(len(Z)):
            P = P + self.delta * np.eye(p)
            S = Z[t] @ P @ Z[t] + self.r
            K = P @ Z[t] / S
            b = b + K * (yy[t] - Z[t] @ b)
            P = P - np.outer(K, Z[t] @ P)
        self.b, self.P = b, P
        return self

    def predict(self, X):
        # coefficients stay at their end-of-training posterior (no peeking at test targets)
        Z = np.c_[np.ones(len(X)), self.sc.transform(X)[:, : self.k]]
        return Z @ self.b

    def predict_proba(self, X):
        return np.clip(self.predict(X) + 0.5, 0, 1)


@model("M.PB.BAYES_SSM", "probabilistic", "Bayesian state-space dynamic linear regression (random-walk coefficients, "
       "Kalman posterior) on the first k standardised features", params={"delta": 1e-4, "k": 5},
       space={"delta": [1e-5, 1e-4, 1e-3], "k": [3, 5, 10]})
def bayes_ssm(task="reg", delta=1e-4, k=5):
    return _DLM(delta, task, k)
