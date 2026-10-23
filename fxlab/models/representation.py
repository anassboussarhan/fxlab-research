"""Representation-learning predictors: learn a latent representation of the
feature vector on the training rows (unsupervised / self-supervised), then fit a
regularised linear/logistic head on the latent codes."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from fxlab.registry import model
from fxlab.models.base import Model, to_binary

R = dict(requires=("torch",), complexity="O(epochs · n · params)")


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


class _LatentHead(Model):
    def __init__(self, task="clf", k=8, epochs=15, seed=0, max_train=30000):
        self.task, self.k, self.epochs, self.seed, self.max_train = task, k, epochs, seed, max_train

    def encode_fit(self, Z):
        raise NotImplementedError

    def encode(self, Z):
        raise NotImplementedError

    def fit(self, X, y):
        X, y = np.asarray(X, float)[-self.max_train:], np.asarray(y)[-self.max_train:]
        self.sc = StandardScaler().fit(np.nan_to_num(X))
        Z = np.clip(self.sc.transform(np.nan_to_num(X)), -10, 10)
        self.encode_fit(Z)
        H = self.encode(Z)
        self.head = LogisticRegression(C=0.1, max_iter=2000) if self.task == "clf" else Ridge(alpha=10.0)
        ok = np.isfinite(np.asarray(y, float))
        self.head.fit(H[ok], to_binary(y[ok]) if self.task == "clf" else np.asarray(y, float)[ok])
        return self

    def _H(self, X):
        return self.encode(np.clip(self.sc.transform(np.nan_to_num(np.asarray(X, float))), -10, 10))

    def predict(self, X):
        return self.head.predict(self._H(X))

    def predict_proba(self, X):
        return self.head.predict_proba(self._H(X))[:, 1]


class _AE(_LatentHead):
    variational = False

    def encode_fit(self, Z):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        p, k = Z.shape[1], self.k
        self.enc = nn.Sequential(nn.Linear(p, 64), nn.ELU(), nn.Linear(64, 2 * k if self.variational else k))
        self.dec = nn.Sequential(nn.Linear(k, 64), nn.ELU(), nn.Linear(64, p))
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=1e-3)
        T = torch.tensor(Z, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(T), generator=g)
            for b in range(0, len(T), 256):
                x = T[perm[b:b + 256]]
                h = self.enc(x)
                if self.variational:
                    mu, lv = h[:, :k], h[:, k:].clamp(-8, 8)
                    z = mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, generator=g)
                    kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1).mean()
                    loss = ((self.dec(z) - x) ** 2).sum(1).mean() + kl
                else:
                    loss = ((self.dec(h) - x) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    def encode(self, Z):
        torch = _torch()
        with torch.no_grad():
            h = self.enc(torch.tensor(Z, dtype=torch.float32)).numpy()
        return h[:, :self.k]  # VAE: posterior mean


@model("M.REP.AUTOENCODER", "representation", "MLP autoencoder (p->64->k->64->p, MSE) + logistic/ridge head on the code",
       params={"k": 8}, space={"k": [4, 8, 16]}, **R)
def ae(task="clf", k=8, **kw):
    return _AE(task, k, **kw)


class _VAE(_AE):
    variational = True


@model("M.REP.VAE", "representation", "variational autoencoder (Gaussian posterior, reparameterisation, ELBO) + head on "
       "posterior means", params={"k": 8}, space={"k": [4, 8]}, **R)
def vae(task="clf", k=8, **kw):
    return _VAE(task, k, **kw)


class _Contrastive(_LatentHead):
    def encode_fit(self, Z):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        p, k = Z.shape[1], self.k
        self.enc = nn.Sequential(nn.Linear(p, 64), nn.ReLU(), nn.Linear(64, k))
        opt = torch.optim.Adam(self.enc.parameters(), lr=1e-3)
        T = torch.tensor(Z, dtype=torch.float32)

        def aug(x):
            m = (torch.rand(x.shape, generator=g) > 0.2).float()   # feature dropout
            return x * m + 0.1 * torch.randn(x.shape, generator=g)

        for _ in range(self.epochs):
            perm = torch.randperm(len(T), generator=g)
            for b in range(0, len(T), 256):
                x = T[perm[b:b + 256]]
                if len(x) < 4:
                    continue
                z1 = nn.functional.normalize(self.enc(aug(x)), dim=1)
                z2 = nn.functional.normalize(self.enc(aug(x)), dim=1)
                logits = z1 @ z2.T / 0.2
                loss = nn.functional.cross_entropy(logits, torch.arange(len(x)))
                opt.zero_grad(); loss.backward(); opt.step()

    def encode(self, Z):
        torch = _torch()
        with torch.no_grad():
            return self.enc(torch.tensor(Z, dtype=torch.float32)).numpy()


@model("M.REP.CONTRASTIVE", "representation", "contrastive (InfoNCE) encoder with feature-dropout + noise augmentations, "
       "linear head on embeddings", params={"k": 8}, space={"k": [8, 16]}, **R)
def contrastive(task="clf", k=8, **kw):
    return _Contrastive(task, k, **kw)


class _DynFactor(_LatentHead):
    """Latent-state model: dynamic factor model (statsmodels) with k latent AR(1) factors driving the first m
    standardised features; Kalman-FILTERED factors (one pass, causal) feed the head."""
    is_sequence = True

    def __init__(self, task="clf", k=2, m=6, **kw):
        super().__init__(task, k, **kw)
        self.m = m

    def encode_fit(self, Z):
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        Y = Z[-5000:, : self.m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = DynamicFactor(Y, k_factors=self.k, factor_order=1).fit(disp=False, maxiter=100)
        self.hist = Z[:, : self.m]

    def encode(self, Z):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = self.res.apply(Z[:, : self.m])
        return np.asarray(app.filtered_state[: self.k]).T


@model("M.REP.LATENT_STATE", "representation", "dynamic factor latent-state model (k AR(1) factors, Kalman-filtered) "
       "+ head on filtered factors", params={"k": 2, "m": 6}, space={"k": [1, 2, 3]}, complexity="O(n·m·k^2)")
def latent_state(task="clf", k=2, m=6, **kw):
    return _DynFactor(task, k, m, **kw)
