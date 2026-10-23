"""Time-series forecasting models.

Input convention (task reg/clf): X[:, 0] must be the 1-bar log return r_t known
at t (catalog feature F.PR.LOGRET_N with n=1); additional columns are optional
exogenous/related series. Models are fitted on the training rows, then FILTERED
forward through the test rows with frozen parameters: the forecast emitted at
row t uses r_0..r_t only. Volatility models (task vol) forecast realised
volatility over the label horizon h."""
from __future__ import annotations

import warnings

import numpy as np

from fxlab.registry import model
from fxlab.models.base import Model, to_binary

S = dict(extra={"input": "series", "tasks": ["reg", "clf"]})


class _Series(Model):
    is_sequence = True
    task = "reg"

    def __init__(self, task="reg", h=1, max_fit=8000):
        self.task, self.h, self.max_fit = task, h, max_fit

    def _r(self, X):
        r = np.asarray(X, float)[:, 0]
        return np.nan_to_num(r)

    def fit(self, X, y=None):
        self._fit(X, y)
        # scale for probabilities: std of the model's own in-sample-period forecasts is not available for all
        # models, so use the training-return std times sqrt(h) (a fixed, train-only quantity)
        self.train_scale = float(np.nanstd(self._r(X)) * np.sqrt(self.h) * 0.1) + 1e-12
        return self

    def predict_proba(self, X):
        f = np.asarray(self.predict(X), float)
        return 1 / (1 + np.exp(-f / self.train_scale))


class _AR(_Series):
    def __init__(self, p=5, **kw):
        super().__init__(**kw)
        self.p = p

    def _fit(self, X, y=None):
        r = self._r(X)[-self.max_fit:]
        p = self.p
        Z = np.column_stack([np.ones(len(r) - p)] + [r[p - k - 1:len(r) - k - 1] for k in range(p)])
        self.b = np.linalg.lstsq(Z, r[p:], rcond=None)[0]
        self.hist = self._r(X)[-p:]
        return self

    def predict(self, X):
        r = np.r_[self.hist, self._r(X)]
        p, h = self.p, self.h
        out = np.empty(len(X))
        for t in range(len(X)):
            lag = list(r[t + 1:t + p + 1][::-1])  # newest first, ends at current row
            acc = 0.0
            for _ in range(h):
                f = self.b[0] + np.dot(self.b[1:], lag[:p])
                acc += f
                lag = [f] + lag[:-1]
            out[t] = acc
        return out


@model("M.TS.AR", "time_series", "AR(p) by OLS on training returns; h-step iterated forecast of cumulative return",
       params={"p": 5}, space={"p": [1, 3, 5, 10]}, **S)
def ar(task="reg", p=5, h=1):
    return _AR(p=p, task=task, h=h)


class _SMState(_Series):
    """statsmodels state-space model: fit on the train tail, then .apply() fixed params to train+test and read
    one-step-ahead predictions. Exogenous regressors are LAGGED one bar (r_t is explained by x_{t-1}) so the
    forecast of r_{t+1} made at t only needs x_t."""

    def _exog(self, X):
        return None

    def _next_exog(self, X):
        return None

    def predict(self, X):
        X = np.asarray(X, float)
        r = np.r_[self.train_r, self._r(X)]
        full = np.r_[self.train_X, X]
        exo = self._exog(full)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = self.res.apply(r, exog=exo) if exo is not None else self.res.apply(r)
            pm = np.asarray(app.fittedvalues)  # pm[t] = forecast of r_t made at t-1
            last = app.forecast(1, exog=self._next_exog(full)) if exo is not None else app.forecast(1)
        nxt = np.r_[pm[1:], np.asarray(last).ravel()[:1]]  # forecast made AT t of r_{t+1}
        return nxt[len(self.train_r):] * self.h


class _ARIMA(_SMState):
    def __init__(self, order=(1, 0, 1), **kw):
        super().__init__(**kw)
        self.order = order

    def make(self, e, ex=None):
        from statsmodels.tsa.arima.model import ARIMA
        return ARIMA(e, order=self.order)

    def _fit(self, X, y=None):
        from statsmodels.tsa.arima.model import ARIMA
        r = self._r(X)
        self.train_r, self.train_X = r, np.asarray(X, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = ARIMA(r[-self.max_fit:], order=self.order).fit()
        return self


@model("M.TS.ARIMA", "time_series", "ARIMA(p,d,q) (statsmodels state-space MLE) on returns; one-step-ahead forecast "
       "from the Kalman filter with frozen params (scaled by h)", params={"p": 1, "d": 0, "q": 1},
       space={"p": [1, 2], "d": [0], "q": [0, 1, 2]}, **S)
def arima(task="reg", p=1, d=0, q=1, h=1):
    return _ARIMA(order=(p, d, q), task=task, h=h)


class _SARIMAX(_SMState):
    def __init__(self, s=24, k_exog=2, **kw):
        super().__init__(**kw)
        self.s, self.k = s, k_exog

    def _exog(self, X):
        if X.shape[1] <= 1 or self.k <= 0:
            return None
        E = np.nan_to_num(X[:, 1:1 + self.k])
        return np.r_[np.zeros((1, E.shape[1])), E[:-1]]  # lagged one bar

    def _next_exog(self, X):
        return np.nan_to_num(X[-1:, 1:1 + self.k])

    def _fit(self, X, y=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        r = self._r(X)
        self.train_r, self.train_X = r, np.asarray(X, float)
        ex = self._exog(self.train_X)
        m = min(self.max_fit, 4000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(r[-m:], exog=None if ex is None else ex[-m:], order=(1, 0, 0),
                               seasonal_order=(1, 0, 0, self.s)).fit(disp=False, maxiter=50)
        return self


@model("M.TS.SARIMAX", "time_series", "SARIMAX: AR(1) x seasonal AR(1) with intraday period s plus k exogenous features",
       params={"s": 24, "k_exog": 2}, space={"s": [24, 96], "k_exog": [0, 2, 5]}, **S)
def sarimax(task="reg", s=24, k_exog=2, h=1):
    return _SARIMAX(s=s, k_exog=k_exog, task=task, h=h)


class _VAR(_Series):
    def __init__(self, p=2, k=3, **kw):
        super().__init__(**kw)
        self.p, self.k = p, k

    def _fit(self, X, y=None):
        Y = np.nan_to_num(np.asarray(X, float)[-self.max_fit:, :self.k])
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-12
        Z = (Y - self.mu) / self.sd
        p = self.p
        R = np.column_stack([np.ones(len(Z) - p)] + [Z[p - j - 1:len(Z) - j - 1] for j in range(p)])
        self.B = np.linalg.lstsq(R, Z[p:], rcond=None)[0]  # (1 + k p) x k
        self.hist = Z[-p:]
        return self

    def predict(self, X):
        Z = (np.nan_to_num(np.asarray(X, float)[:, :self.k]) - self.mu) / self.sd
        Z = np.r_[self.hist, Z]
        p = self.p
        out = np.empty(len(X))
        for t in range(len(X)):
            lags = [Z[t + p - j] for j in range(p)]  # newest first, includes current row
            acc = 0.0
            for _ in range(self.h):
                f = self.B[0] + sum(lags[j] @ self.B[1 + j * Z.shape[1]:1 + (j + 1) * Z.shape[1]] for j in range(p))
                acc += f[0] * self.sd[0] + self.mu[0]
                lags = [f] + lags[:-1]
            out[t] = acc
        return out


@model("M.TS.VAR", "time_series", "vector autoregression VAR(p) on the first k columns (target return + related series), "
       "OLS per equation; iterated h-step forecast of the target equation", params={"p": 2, "k": 3},
       space={"p": [1, 2, 4], "k": [2, 3, 5]}, **S)
def var(task="reg", p=2, k=3, h=1):
    return _VAR(p=p, k=k, task=task, h=h)


class _VECM(_Series):
    def __init__(self, k=2, lags=1, **kw):
        super().__init__(**kw)
        self.k, self.lags = k, lags

    def _fit(self, X, y=None):
        from statsmodels.tsa.vector_ar.vecm import VECM
        R = np.nan_to_num(np.asarray(X, float)[:, :self.k])
        L = np.cumsum(R, axis=0)  # log-price levels up to a constant
        self.last_level = L[-1]
        m = L[-self.max_fit:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = VECM(m, k_ar_diff=self.lags, coint_rank=1, deterministic="ci").fit()
        self.alpha, self.beta, self.gamma = self.res.alpha, self.res.beta, self.res.gamma
        self.det = getattr(self.res, "det_coef_coint", np.zeros((1, 1)))
        self.hist_d = np.diff(m[-(self.lags + 1):], axis=0)
        return self

    def predict(self, X):
        R = np.nan_to_num(np.asarray(X, float)[:, :self.k])
        L = self.last_level + np.cumsum(R, axis=0)
        D = np.r_[self.hist_d, R]
        out = np.empty(len(X))
        const = float(self.det.ravel()[0]) if np.size(self.det) else 0.0
        for t in range(len(X)):
            ect = (L[t] @ self.beta).ravel() + const
            dy = self.alpha @ ect
            for j in range(self.lags):
                dy = dy + self.gamma[:, j * self.k:(j + 1) * self.k] @ D[t + self.lags - j]
            out[t] = dy[0] * self.h
        return out


@model("M.TS.VECM", "time_series", "vector error-correction model (Johansen, rank 1) on log-price levels of the first k "
       "series (cumulated returns); forecast of the target's next change from error-correction + lagged differences",
       params={"k": 2, "lags": 1}, space={"k": [2, 3], "lags": [1, 2]}, **S)
def vecm(task="reg", k=2, lags=1, h=1):
    return _VECM(k=k, lags=lags, task=task, h=h)


class _KalmanTrend(_Series):
    def __init__(self, q_level=0.01, q_slope=1e-4, **kw):
        super().__init__(**kw)
        self.ql, self.qs = q_level, q_slope

    def _fit(self, X, y=None):
        r = self._r(X)
        self.rvar = np.var(r) + 1e-16
        self.lp_end = np.cumsum(r)[-1]
        from fxlab.features.econometric import _kalman_trend
        self.state_hist = np.cumsum(r)
        return self

    def predict(self, X):
        from fxlab.features.econometric import _kalman_trend
        y = np.r_[self.state_hist, self.lp_end + np.cumsum(self._r(X))]
        sl = _kalman_trend(y, self.ql * self.rvar, self.qs * self.rvar, self.rvar)
        return sl[len(self.state_hist):] * self.h


@model("M.TS.KALMAN", "time_series", "local-linear-trend Kalman filter on log price; forecast = filtered slope x h",
       params={"q_level": 0.01, "q_slope": 1e-4}, space={"q_slope": [1e-5, 1e-4, 1e-3]}, **S)
def kalman(task="reg", q_level=0.01, q_slope=1e-4, h=1):
    return _KalmanTrend(q_level=q_level, q_slope=q_slope, task=task, h=h)


class _MS(_Series):
    def __init__(self, k=2, **kw):
        super().__init__(**kw)
        self.k = k

    def _fit(self, X, y=None):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        r = self._r(X)
        m = r[-min(self.max_fit, 5000):]
        self.scale = m.std() + 1e-12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MarkovRegression(m / self.scale, k_regimes=self.k, switching_variance=True).fit(disp=False)
        P = np.asarray(res.regime_transition)  # (k, k, 1): P[i, j] = P(s_t=i | s_{t-1}=j)
        self.T = P[:, :, 0].T  # row = from
        pr = dict(zip(res.model.param_names, np.asarray(res.params)))
        self.mu = np.array([pr[f"const[{i}]"] for i in range(self.k)]) * self.scale
        self.sig = np.sqrt(np.array([pr[f"sigma2[{i}]"] for i in range(self.k)])) * self.scale
        self.p0 = np.asarray(res.filtered_marginal_probabilities)[-1]
        return self

    def predict(self, X):
        r = self._r(X)
        p = self.p0.copy()
        out = np.empty(len(r))
        for t in range(len(r)):
            pred = self.T.T @ p
            lik = np.exp(-0.5 * ((r[t] - self.mu) / self.sig) ** 2) / self.sig
            p = pred * lik
            p = p / p.sum() if p.sum() > 0 else pred
            out[t] = (self.T.T @ p) @ self.mu * self.h  # E[r_{t+1} | info_t]
        return out


@model("M.TS.MARKOV_SWITCHING", "time_series", "Hamilton Markov-switching regression (k regimes, switching mean and "
       "variance) on returns; Hamilton-filtered regime probabilities propagated one step for E[r_{t+1}]",
       params={"k": 2}, space={"k": [2, 3]}, **S)
def markov_switching(task="reg", k=2, h=1):
    return _MS(k=k, task=task, h=h)


class _HoltWinters(_Series):
    def __init__(self, s=24, alpha=0.01, gamma=0.05, **kw):
        super().__init__(**kw)
        self.s, self.a, self.g = s, alpha, gamma

    def _run(self, r, level, season):
        out = np.empty(len(r))
        s = self.s
        for t in range(len(r)):
            i = (self.t0 + t) % s
            out[t] = level + season[(i + 1) % s]  # forecast for next bar
            e = r[t] - (level + season[i])
            level = level + self.a * e
            season[i] = season[i] + self.g * e
        return out, level, season

    def _fit(self, X, y=None):
        r = self._r(X)
        self.t0 = 0
        season = np.array([r[i::self.s].mean() for i in range(self.s)]) - r.mean()
        _, self.level, self.season = self._run(r, r.mean(), season.copy())
        self.t0 = len(r)
        return self

    def predict(self, X):
        out, _, _ = self._run(self._r(X), self.level, self.season.copy())
        return out * self.h


@model("M.TS.HOLT_WINTERS", "time_series", "additive Holt-Winters exponential smoothing of returns with level + intraday "
       "seasonal component of period s; forecast next-bar return", params={"s": 24, "alpha": 0.01, "gamma": 0.05},
       space={"s": [24, 96], "gamma": [0.01, 0.05, 0.2]}, **S)
def holt_winters(task="reg", s=24, alpha=0.01, gamma=0.05, h=1):
    return _HoltWinters(s=s, alpha=alpha, gamma=gamma, task=task, h=h)


# ------------------------------------------------------------------------------------- volatility models
V = dict(extra={"input": "series", "tasks": ["vol"]}, requires=("arch",))


class _GarchVol(_Series):
    task = "vol"

    def __init__(self, kind="garch", h=1, **kw):
        super().__init__(task="vol", h=h)
        self.kind = kind

    def _fit(self, X, y=None):
        from arch import arch_model
        r = self._r(X) * 100
        self.train_r = r
        o = 1 if self.kind in ("gjr", "egarch") else 0
        vol = "EGARCH" if self.kind == "egarch" else "GARCH"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r[-self.max_fit:], mean="Zero", vol=vol, p=1, o=o, q=1, rescale=False).fit(disp="off")
        pr = res.params
        self.params = (pr["omega"], pr["alpha[1]"], pr["beta[1]"], pr.get("gamma[1]", 0.0))
        self.s0 = float(res.conditional_volatility[-1] ** 2)
        return self

    def predict(self, X):
        from fxlab.features.econometric import _filter_garch
        kmap = {"garch": 0, "gjr": 1, "egarch": 2}
        r = self._r(X) * 100
        s2 = _filter_garch(np.r_[self.train_r[-2000:], r], *self.params, kmap[self.kind], float(np.var(self.train_r[-2000:])))
        s2 = s2[-len(r):]
        om, al, be, ga = self.params
        if self.kind == "egarch":
            tot = self.h * s2
        else:
            pers = al + be + 0.5 * ga
            lr_var = om / max(1 - pers, 1e-6)
            tot = sum(lr_var + pers ** k * (s2 - lr_var) for k in range(self.h))
        return np.sqrt(np.maximum(tot, 0)) / 100


@model("M.TS.GARCH", "time_series", "GARCH(1,1) h-step realised-volatility forecast sqrt(sum_k E[s2_{t+k}])", **V)
def garch(task="vol", h=1):
    return _GarchVol("garch", h=h)


@model("M.TS.EGARCH", "time_series", "EGARCH(1,1,1) volatility forecast (log-variance with asymmetry)", **V)
def egarch(task="vol", h=1):
    return _GarchVol("egarch", h=h)


@model("M.TS.GJR_GARCH", "time_series", "GJR-GARCH(1,1,1) volatility forecast (threshold leverage term)", **V)
def gjr(task="vol", h=1):
    return _GarchVol("gjr", h=h)


class _HAR(_Series):
    task = "vol"

    def __init__(self, d=24, h=1, **kw):
        super().__init__(task="vol", h=h)
        self.d = d

    def _design(self, r):
        r2 = r ** 2
        c = np.cumsum(np.r_[0, r2])
        def avg(w):
            out = np.full(len(r2), np.nan)
            out[w - 1:] = (c[w:] - c[:-w]) / w
            return out
        return np.column_stack([np.ones(len(r)), avg(self.d), avg(5 * self.d), avg(22 * self.d)])

    def _fit(self, X, y):
        r = self._r(X)
        Z = self._design(r)
        yy = np.asarray(y, float) ** 2 / self.h  # per-bar variance over the horizon
        ok = ~np.isnan(Z).any(1) & np.isfinite(yy)
        self.b = np.linalg.lstsq(Z[ok], yy[ok], rcond=None)[0]
        self.hist = r[-22 * self.d:]
        return self

    def predict(self, X):
        r = np.r_[self.hist, self._r(X)]
        Z = self._design(r)[len(self.hist):]
        return np.sqrt(np.maximum(Z @ self.b, 0) * self.h)


@model("M.TS.HAR", "time_series", "HAR-RV regression of horizon variance on daily/weekly/monthly average squared returns "
       "(Corsi 2009), d = bars per day", params={"d": 24}, space={"d": [24, 96, 288]},
       extra={"input": "series", "tasks": ["vol"]})
def har(task="vol", d=24, h=1):
    return _HAR(d=d, h=h)
