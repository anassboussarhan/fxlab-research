"""Contract tests for EVERY registered model on REAL EURUSD H1 data plus a
planted-signal learnability check on synthetic data:

1. fits on a training block and predicts the following block (shape, finiteness, proba in [0,1]);
2. causality of prediction: predicting a prefix of the test block gives identical outputs for those
   rows (a model that used later rows as context would change them);
3. learnability: on synthetic data with a planted signal, supervised models beat chance clearly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fxlab.features  # noqa: F401
import fxlab.models  # noqa: F401
from fxlab.registry import by_kind, CATALOG
from fxlab.labels import forward_mid_return, volatility_label

MODELS = sorted(by_kind("model"), key=lambda e: e.id)
FAST = {"epochs": 4}
OVR = {
    "M.NN.": FAST, "M.REP.AUTOENCODER": {"epochs": 4}, "M.REP.VAE": {"epochs": 4}, "M.REP.CONTRASTIVE": {"epochs": 3},
    "M.RL.DQN": {"epochs": 1}, "M.RL.PPO": {"epochs": 1}, "M.RL.SAC": {"epochs": 1},
    "M.EVO.GP": {"pop": 60, "gens": 6}, "M.EVO.GRAMMATICAL": {"pop": 60, "gens": 6}, "M.EVO.DE_SHARPE": {"maxiter": 8},
    "M.TREE.RF": {"n": 50}, "M.TREE.EXTRA": {"n": 50}, "M.TREE.GBM": {"n": 50}, "M.TREE.XGB": {"n": 80},
    "M.TREE.LGBM": {"n": 80}, "M.TREE.CATBOOST": {"n": 80}, "M.TREE.ADABOOST": {"n": 50},
    "M.KER.GP": {"m": 400}, "M.KER.RBF_SVM": {"m": 1500}, "M.KER.SVR": {"m": 1500}, "M.KER.KRR": {"m": 1500},
}


def overrides(mid):
    o = {}
    for k, v in OVR.items():
        if mid.startswith(k):
            o.update(v)
    return o


def tasks_of(e):
    return e.extra.get("tasks", ["clf", "reg"])


@pytest.fixture(scope="module")
def dataset(real_h1):
    df = real_h1
    from fxlab.features.interactions import compute
    cols = {"r1": compute(df, "F.PR.LOGRET_N", n=1)}
    for fid in ["F.MO.RSI", "F.TR.REG_SLOPE", "F.VO.HV", "F.MS.TICK_IMBALANCE", "F.PR.CLV", "F.DI.SKEW",
                "F.TR.EMA", "F.VO.ATR_RATIO", "F.MS.SPREAD_Z"]:
        cols[fid] = compute(df, fid)
    X = pd.DataFrame(cols).iloc[300:]
    y_ret, _ = forward_mid_return(df, 1)
    vol, _ = volatility_label(df, 5)
    data = pd.concat([X, y_ret.rename("y"), vol["rv"].rename("rv")], axis=1).dropna()
    return data


def _run(e, task, data, n_tr=2000, n_te=600):
    X = data.drop(columns=["y", "rv"]).to_numpy(float)
    y = data["rv"].to_numpy() if task == "vol" else data["y"].to_numpy()
    if task == "clf":
        y = (y > 0).astype(int)
    kw = dict(e.params) | overrides(e.id)
    if task == "vol":
        kw["h"] = 5
    m = e.fn(task=task, **kw)
    m.fit(X[:n_tr], y[:n_tr])
    Xte = X[n_tr:n_tr + n_te]
    return m, Xte, y[n_tr:n_tr + n_te]


def _out(m, task, X):
    if task == "clf":
        return np.asarray(m.predict_proba(X), float)
    return np.asarray(m.predict(X), float)


CASES = [(e, t) for e in MODELS for t in tasks_of(e)]


@pytest.mark.parametrize("e,task", CASES, ids=[f"{e.id}-{t}" for e, t in CASES])
def test_model_contract(e, task, dataset, catalog_id):
    catalog_id(e.id)
    m, Xte, yte = _run(e, task, dataset)
    p = _out(m, task, Xte)
    assert p.shape == (len(Xte),)
    assert np.isfinite(p).mean() > 0.95, "non-finite predictions"
    if task == "clf":
        assert np.nanmin(p) >= -1e-9 and np.nanmax(p) <= 1 + 1e-9
    if task == "pol":
        assert set(np.unique(np.round(p, 6))) <= {-1.0, 0.0, 1.0} or (np.abs(p) <= 1 + 1e-9).all()
    k = len(Xte) // 2
    p2 = _out(m, task, Xte[:k])
    assert np.allclose(p[:k], p2, rtol=1e-5, atol=1e-8, equal_nan=True), "prediction depends on later rows"


def _planted(n=3000, p=6, seed=1, factor=False):
    rng = np.random.default_rng(seed)
    if factor:  # signal lives in a persistent latent factor observed through noisy features
        z = np.zeros(n)
        for t in range(1, n):
            z[t] = 0.95 * z[t - 1] + rng.normal(0, 0.3)
        L = rng.normal(1, 0.2, p)
        X = z[:, None] * L[None, :] + 0.5 * rng.normal(size=(n, p))
        y = z + 0.3 * rng.normal(size=n)
        return X, y
    X = rng.normal(size=(n, p))
    signal = 0.8 * X[:, 1] - 0.6 * np.tanh(2 * X[:, 2])
    y = signal + 0.6 * rng.normal(size=n)
    X[:, 0] = rng.normal(0, 1, n)  # placeholder "return" column
    return X, y


SUPERVISED = [e for e in MODELS if e.family in ("linear", "tree", "kernel", "instance_based", "neural",
                                                  "representation", "probabilistic", "ensemble")
              or e.id in ("M.EVO.GP", "M.EVO.SYMBOLIC_REGRESSION", "M.EVO.GRAMMATICAL")]


@pytest.mark.parametrize("e", SUPERVISED, ids=[e.id for e in SUPERVISED])
def test_model_learns_planted_signal(e, catalog_id):
    catalog_id(e.id)
    from sklearn.metrics import roc_auc_score
    X, y = _planted(factor=e.id in ("M.REP.LATENT_STATE", "M.PB.HMM"))
    task = "clf" if "clf" in tasks_of(e) else tasks_of(e)[0]
    kw = dict(e.params) | overrides(e.id)
    if e.family == "neural":
        kw |= {"L": 1, "epochs": 15}
    m = e.fn(task=task, **kw)
    yt = (y > 0).astype(int) if task == "clf" else y
    m.fit(X[:2000], yt[:2000])
    out = _out(m, task, X[2000:]) if task in ("clf", "reg", "q") else np.asarray(m.predict(X[2000:]), float)
    auc = roc_auc_score((y[2000:] > 0).astype(int), np.nan_to_num(out))
    floor = 0.6 if e.id in ("M.PB.HMM", "M.REP.LATENT_STATE", "M.ENS.DYNAMIC_SELECTION", "M.EVO.GRAMMATICAL") else 0.7
    assert auc > floor, f"AUC {auc:.3f} on planted signal"


TS_MODELS = [e for e in MODELS if e.extra.get("input") == "series" and "vol" not in tasks_of(e)]


@pytest.mark.parametrize("e", TS_MODELS, ids=[e.id for e in TS_MODELS])
def test_ts_model_learns_ar_structure(e, catalog_id):
    """Series models must exploit planted AR(1) autocorrelation (phi=0.4) in X[:,0]."""
    catalog_id(e.id)
    rng = np.random.default_rng(3)
    n = 3000
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.4 * r[t - 1] + rng.normal(0, 1e-3)
    seasonal = np.tile(np.r_[np.full(12, 2e-4), np.full(12, -2e-4)], n // 24 + 1)[:n]
    if e.id == "M.TS.HOLT_WINTERS":
        r = seasonal + rng.normal(0, 2e-4, n)
    if e.id == "M.TS.KALMAN":  # persistent, slowly varying drift (a trend), not AR(1) noise
        drift = np.cumsum(rng.normal(0, 2e-5, n))
        r = drift + rng.normal(0, 3e-4, n)
    X = np.column_stack([r, rng.normal(size=n), rng.normal(size=n)])
    if e.id == "M.TS.VECM":  # cointegrated pair: target = other + OU spread
        other = np.cumsum(rng.normal(0, 2e-4, n))
        sp = np.zeros(n)
        for t in range(1, n):
            sp[t] = 0.9 * sp[t - 1] + rng.normal(0, 1e-3)
        tgt = other + sp
        X = np.column_stack([np.diff(np.r_[0, tgt]), np.diff(np.r_[0, other])])
    y = np.r_[X[1:, 0], np.nan]
    kw = dict(e.params)
    m = e.fn(task="reg", **kw)
    m.fit(X[:2000], y[:2000])
    f = np.asarray(m.predict(X[2000:-1]), float)
    c = np.corrcoef(np.nan_to_num(f), y[2000:-1])[0, 1]
    assert c > 0.15, f"forecast/realised correlation {c:.3f}"


VOL_MODELS = [e for e in MODELS if "vol" in tasks_of(e)]


@pytest.mark.parametrize("e", VOL_MODELS, ids=[e.id for e in VOL_MODELS])
def test_vol_model_tracks_volatility(e, catalog_id):
    """Volatility forecasts must correlate with realised vol on GARCH-simulated returns."""
    catalog_id(e.id)
    rng = np.random.default_rng(5)
    n = 4000
    s2 = np.empty(n); r = np.empty(n); s2[0] = 1e-6
    for t in range(n):
        if t:
            s2[t] = 1e-8 + 0.08 * r[t - 1] ** 2 + 0.9 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.normal()
    h = 5
    rv = np.array([np.sqrt((r[t + 1:t + 1 + h] ** 2).sum()) if t + h < n else np.nan for t in range(n)])
    X = r[:, None]
    kw = dict(e.params) | {"h": h}
    if e.id == "M.TS.HAR":
        kw["d"] = 10
    m = e.fn(task="vol", **kw)
    m.fit(X[:3000], rv[:3000])
    f = np.asarray(m.predict(X[3000:-h]), float)
    c = np.corrcoef(f, rv[3000:-h])[0, 1]
    assert c > 0.3, f"vol forecast corr {c:.3f}"
