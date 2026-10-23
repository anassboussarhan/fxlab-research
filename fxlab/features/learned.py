"""7.17 Learned representations. Every learner is (re)fitted on a fixed
absolute schedule (every R bars) using ONLY the trailing W embedding vectors
that end before the refit bar, then applied to the next R bars. Inputs are
lag embeddings of returns standardised by a trailing (shifted) volatility."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fxlab.registry import feature
from fxlab.features._util import EPS

F = "learned"
SCHED = {"W": [2000, 5000], "R": [500, 2500]}


def embedding(df: pd.DataFrame, L: int) -> np.ndarray:
    lr = np.log(df["close"]).diff()
    sd = lr.ewm(span=500, min_periods=50).std().shift(1)
    z = (lr / (sd + EPS)).clip(-8, 8).to_numpy(float)
    E = np.full((len(z), L), np.nan)
    if len(z) >= L:
        E[L - 1:] = sliding_window_view(z, L)[:, ::-1]  # col 0 = newest
    return E


def schedule(E: np.ndarray, W: int, R: int, fit, transform, k: int, seed: int = 0) -> np.ndarray:
    n = len(E)
    out = np.full((n, k), np.nan)
    for i in range(W, n, R):
        tr = E[i - W:i]
        tr = tr[~np.isnan(tr).any(1)]
        if len(tr) < W // 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit(tr, seed + i)
        end = min(n, i + R)
        blk = E[i:end]
        ok = ~np.isnan(blk).any(1)
        res = np.full((len(blk), k), np.nan)
        if ok.any():
            res[ok] = np.asarray(transform(model, blk[ok])).reshape(ok.sum(), k)
        out[i:end] = res
    return out


def _frame(o, df, prefix):
    return pd.DataFrame(o, index=df.index, columns=[f"{prefix}{j}" for j in range(o.shape[1])])


@feature("F.LF.PCA_EMBED", F, "PCA of L-lag standardised return embeddings; scores on the first k components",
         params={"L": 20, "k": 3, "W": 2000, "R": 500}, space=SCHED | {"L": [10, 20, 50], "k": [3, 5]})
def f_pca_embed(df, L=20, k=3, W=2000, R=500):
    from sklearn.decomposition import PCA
    o = schedule(embedding(df, L), W, R, lambda X, s: PCA(k, random_state=0).fit(X), lambda m, X: m.transform(X), k)
    return _frame(o, df, "pc")


@feature("F.LF.KERNEL_PCA", F, "RBF kernel PCA (gamma=1/L) fitted on a 1000-sample subset of the trailing embeddings; k scores",
         params={"L": 20, "k": 3, "W": 2000, "R": 500}, space=SCHED | {"L": [10, 20]})
def f_kpca(df, L=20, k=3, W=2000, R=500):
    from sklearn.decomposition import KernelPCA
    def fit(X, s):
        rng = np.random.default_rng(s)
        sub = X[rng.choice(len(X), min(1000, len(X)), replace=False)]
        return KernelPCA(k, kernel="rbf", gamma=1.0 / L, random_state=0).fit(sub)
    return _frame(schedule(embedding(df, L), W, R, fit, lambda m, X: m.transform(X), k), df, "kpc")


@feature("F.LF.ICA", F, "FastICA independent sources of the return embedding; k source activations",
         params={"L": 20, "k": 3, "W": 2000, "R": 500}, space=SCHED | {"L": [10, 20]})
def f_ica(df, L=20, k=3, W=2000, R=500):
    from sklearn.decomposition import FastICA
    fit = lambda X, s: FastICA(k, random_state=0, max_iter=400, whiten="unit-variance").fit(X)
    return _frame(schedule(embedding(df, L), W, R, fit, lambda m, X: m.transform(X), k), df, "ic")


@feature("F.LF.DICT_LEARNING", F, "sparse coding: mini-batch dictionary learning (k atoms, alpha=1) of the embedding; "
         "outputs L1 norm of the code and index of the strongest atom", params={"L": 20, "k": 8, "W": 2000, "R": 500},
         space=SCHED | {"k": [8, 16]})
def f_dict(df, L=20, k=8, W=2000, R=500):
    from sklearn.decomposition import MiniBatchDictionaryLearning
    fit = lambda X, s: MiniBatchDictionaryLearning(k, alpha=1.0, random_state=0, max_iter=20).fit(X)
    def tr(m, X):
        c = m.transform(X)
        return np.column_stack([np.abs(c).sum(1), np.abs(c).argmax(1)])
    return _frame(schedule(embedding(df, L), W, R, fit, tr, 2), df, "dl")


@feature("F.LF.KMEANS_CODEBOOK", F, "vector quantisation: k-means codebook (k centroids) of the embedding; distance to the "
         "nearest centroid (novelty) and the nearest centroid's mean next-return computed in-sample at fit time",
         params={"L": 10, "k": 16, "W": 2000, "R": 500}, space=SCHED | {"k": [8, 16, 32]})
def f_kmeans(df, L=10, k=16, W=2000, R=500):
    from sklearn.cluster import KMeans
    E = embedding(df, L)
    nxt = np.roll(E[:, 0], -1)  # next standardised return; only used for rows strictly before the refit bar
    nxt[-1] = np.nan
    n = len(E)
    out = np.full((n, 2), np.nan)
    for i in range(W, n, R):
        idx = np.arange(i - W, i - 1)  # i-1 excluded: its next return is bar i (not yet known at fit time... known at i but keep margin)
        X = E[idx]
        ok = ~np.isnan(X).any(1) & ~np.isnan(nxt[idx])
        if ok.sum() < W // 2:
            continue
        km = KMeans(k, n_init=3, random_state=0).fit(X[ok])
        lab = km.labels_
        means = np.array([nxt[idx][ok][lab == j].mean() if (lab == j).any() else 0.0 for j in range(k)])
        end = min(n, i + R)
        blk = E[i:end]
        g = ~np.isnan(blk).any(1)
        if g.any():
            d = km.transform(blk[g])
            j = d.argmin(1)
            res = np.full((len(blk), 2), np.nan)
            res[g, 0] = d.min(1)
            res[g, 1] = means[j]
            out[i:end] = res
    return pd.DataFrame(out, index=df.index, columns=["novelty", "centroid_next_mean"])


# ------------------------------------------------------------------ torch learners
def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


@feature("F.LF.AUTOENCODER", F, "MLP autoencoder L->32->k->32->L (tanh), MSE, Adam, `epochs` epochs per refit; bottleneck "
         "activations and reconstruction error (novelty)", params={"L": 20, "k": 3, "W": 2000, "R": 500, "epochs": 20},
         space=SCHED | {"k": [2, 3, 5]}, requires=("torch",), complexity="O(fits·W·epochs)")
def f_autoencoder(df, L=20, k=3, W=2000, R=500, epochs=20):
    torch = _torch()
    nn = torch.nn

    def fit(X, s):
        torch.manual_seed(s % (2 ** 31))
        enc = nn.Sequential(nn.Linear(L, 32), nn.Tanh(), nn.Linear(32, k))
        dec = nn.Sequential(nn.Linear(k, 32), nn.Tanh(), nn.Linear(32, L))
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
        T = torch.tensor(X, dtype=torch.float32)
        g = torch.Generator().manual_seed(s % (2 ** 31))
        for _ in range(epochs):
            perm = torch.randperm(len(T), generator=g)
            for b in range(0, len(T), 256):
                xb = T[perm[b:b + 256]]
                loss = ((dec(enc(xb)) - xb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return enc, dec

    def tr(m, X):
        enc, dec = m
        with torch.no_grad():
            T = torch.tensor(X, dtype=torch.float32)
            z = enc(T)
            err = ((dec(z) - T) ** 2).mean(1, keepdim=True)
            return torch.cat([z, err], 1).numpy()

    o = schedule(embedding(df, L), W, R, fit, tr, k + 1)
    return pd.DataFrame(o, index=df.index, columns=[f"z{j}" for j in range(k)] + ["recon_err"])


@feature("F.LF.CONTRASTIVE", F, "self-supervised contrastive encoder (SimCLR-style NT-Xent, temperature 0.2): positives are "
         "two augmentations (amplitude scaling N(1,.1), Gaussian jitter .1, random masking 10%) of the same window; "
         "encoder MLP L->32->k; outputs k embedding coordinates", params={"L": 20, "k": 4, "W": 2000, "R": 500, "epochs": 10},
         space=SCHED | {"k": [4, 8]}, requires=("torch",))
def f_contrastive(df, L=20, k=4, W=2000, R=500, epochs=10):
    torch = _torch()
    nn = torch.nn

    def aug(x, g):
        sc = 1 + 0.1 * torch.randn(len(x), 1, generator=g)
        jit = 0.1 * torch.randn(x.shape, generator=g)
        mask = (torch.rand(x.shape, generator=g) > 0.1).float()
        return (x * sc + jit) * mask

    def fit(X, s):
        torch.manual_seed(s % (2 ** 31))
        g = torch.Generator().manual_seed(s % (2 ** 31))
        enc = nn.Sequential(nn.Linear(L, 32), nn.ReLU(), nn.Linear(32, k))
        opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
        T = torch.tensor(X, dtype=torch.float32)
        for _ in range(epochs):
            perm = torch.randperm(len(T), generator=g)
            for b in range(0, len(T), 256):
                xb = T[perm[b:b + 256]]
                if len(xb) < 4:
                    continue
                z1 = nn.functional.normalize(enc(aug(xb, g)), dim=1)
                z2 = nn.functional.normalize(enc(aug(xb, g)), dim=1)
                z = torch.cat([z1, z2])
                sim = z @ z.T / 0.2
                sim.fill_diagonal_(-1e9)
                m = len(xb)
                tgt = torch.cat([torch.arange(m, 2 * m), torch.arange(0, m)])
                loss = nn.functional.cross_entropy(sim, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
        return enc

    def tr(enc, X):
        with torch.no_grad():
            return enc(torch.tensor(X, dtype=torch.float32)).numpy()

    return _frame(schedule(embedding(df, L), W, R, fit, tr, k), df, "c")


@feature("F.LF.GRU_EMBED", F, "neural sequence embedding: GRU(hidden=h) over the L-lag embedding (oldest->newest) trained "
         "to predict the sign of the next standardised return (BCE) on the trailing window; final hidden state as features",
         params={"L": 30, "h": 4, "W": 2000, "R": 500, "epochs": 5}, space=SCHED | {"h": [4, 8]}, requires=("torch",))
def f_gru(df, L=30, h=4, W=2000, R=500, epochs=5):
    torch = _torch()
    nn = torch.nn
    E = embedding(df, L)
    nxt = np.roll(E[:, 0], -1)
    nxt[-1] = np.nan
    n = len(E)
    out = np.full((n, h), np.nan)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.GRU(1, h, batch_first=True)
            self.o = nn.Linear(h, 1)

        def forward(self, x):
            _, hn = self.g(x)
            return self.o(hn[-1]), hn[-1]

    for i in range(W, n, R):
        idx = np.arange(i - W, i - 1)
        X, y = E[idx], nxt[idx]
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        if ok.sum() < W // 2:
            continue
        torch.manual_seed(i)
        g = torch.Generator().manual_seed(i)
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        T = torch.tensor(X[ok][:, ::-1].copy(), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor((y[ok] > 0).astype(np.float32)).unsqueeze(-1)
        for _ in range(epochs):
            perm = torch.randperm(len(T), generator=g)
            for b in range(0, len(T), 256):
                p, _ = net(T[perm[b:b + 256]])
                loss = nn.functional.binary_cross_entropy_with_logits(p, Y[perm[b:b + 256]])
                opt.zero_grad(); loss.backward(); opt.step()
        end = min(n, i + R)
        blk = E[i:end]
        gmask = ~np.isnan(blk).any(1)
        if gmask.any():
            with torch.no_grad():
                _, hid = net(torch.tensor(blk[gmask][:, ::-1].copy(), dtype=torch.float32).unsqueeze(-1))
            res = np.full((len(blk), h), np.nan)
            res[gmask] = hid.numpy()
            out[i:end] = res
    return pd.DataFrame(out, index=df.index, columns=[f"h{j}" for j in range(h)])
