"""Neural predictors (PyTorch, CPU). Sequence models see a window of the last L
feature rows ending at the current row (never later rows). Training uses the
most recent `max_train` rows, a chronological 15% validation tail for early
stopping, Adam, and loss BCE (clf) / MSE (reg) / pinball (q)."""
from __future__ import annotations

import numpy as np

from fxlab.registry import model
from fxlab.models.base import Model, to_binary

NN = dict(requires=("torch",), complexity="O(epochs · n · params)")


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


class TorchModel(Model):
    is_sequence = True

    def __init__(self, task="clf", L=32, hidden=32, epochs=10, lr=1e-3, max_train=20000, alpha=0.5, seed=0,
                 patience=3, **arch):
        self.task, self.L, self.hidden, self.epochs, self.lr = task, L, hidden, epochs, lr
        self.max_train, self.alpha, self.seed, self.patience, self.arch = max_train, alpha, seed, patience, arch

    # subclasses implement build(p) -> nn.Module taking (B, L, p) and returning (B, 1)
    def build(self, p):
        raise NotImplementedError

    def _windows(self, Z):
        torch = _torch()
        L = self.L
        pad = np.r_[np.zeros((max(0, L - 1), Z.shape[1])), Z]
        W = np.lib.stride_tricks.sliding_window_view(pad, (L, Z.shape[1]))[:, 0]
        return torch.tensor(W.copy(), dtype=torch.float32)

    def fit(self, X, y):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        X = np.asarray(X, float)
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        Z = np.nan_to_num((X - self.mu) / self.sd).clip(-10, 10)
        yy = to_binary(y).astype(np.float32) if self.task == "clf" else np.asarray(y, np.float32)
        if self.task != "clf":
            self.ys = float(np.nanstd(yy) + 1e-12)
            yy = yy / self.ys
        ok = np.isfinite(yy)
        Z, yy = Z[-self.max_train:], yy[-self.max_train:]
        ok = ok[-self.max_train:]
        W = self._windows(Z)[ok]
        Y = torch.tensor(yy[ok]).unsqueeze(1)
        self.ctx = Z[-(self.L - 1):] if self.L > 1 else Z[:0]
        nv = max(1, int(0.15 * len(W)))
        Wt, Yt, Wv, Yv = W[:-nv], Y[:-nv], W[-nv:], Y[-nv:]
        self.net = self.build(Z.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Wt), generator=g)
            for b in range(0, len(Wt), 256):
                i = perm[b:b + 256]
                loss = self._loss(self.net(Wt[i]), Yt[i])
                opt.zero_grad(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(self._loss(self.net(Wv), Yv))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.val_loss = best
        return self

    def _loss(self, out, y):
        torch = _torch()
        if self.task == "clf":
            return torch.nn.functional.binary_cross_entropy_with_logits(out, y)
        if self.task == "q":
            e = y - out
            return torch.maximum(self.alpha * e, (self.alpha - 1) * e).mean()
        return torch.nn.functional.mse_loss(out, y)

    def _raw(self, X):
        torch = _torch()
        Z = np.nan_to_num((np.asarray(X, float) - self.mu) / self.sd).clip(-10, 10)
        Zc = np.r_[self.ctx, Z]
        W = self._windows(Zc)[len(self.ctx):]
        self.net.eval()
        outs = []
        with torch.no_grad():
            for b in range(0, len(W), 2048):
                outs.append(self.net(W[b:b + 2048]).numpy()[:, 0])
        return np.concatenate(outs) if outs else np.array([])

    def predict(self, X):
        o = self._raw(X)
        return o if self.task == "clf" else o * self.ys

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-self._raw(X)))


def _mods():
    torch = _torch()
    return torch, torch.nn


class _MLP(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        L, h = self.L, self.hidden
        return nn.Sequential(nn.Flatten(), nn.Linear(L * p, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.ReLU(),
                             nn.Linear(h, 1))


@model("M.NN.MLP", "neural", "multilayer perceptron on the flattened L x p window (2 hidden ReLU layers, dropout)",
       params={"L": 1, "hidden": 64}, space={"L": [1, 8], "hidden": [32, 64]}, **NN)
def mlp(task="clf", L=1, hidden=64, **kw):
    return _MLP(task, L=L, hidden=hidden, **kw)


class _CNN(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        h = self.hidden

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.c = nn.Sequential(nn.Conv1d(p, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
                s.o = nn.Linear(h, 1)

            def forward(s, x):
                return s.o(s.c(x.transpose(1, 2)).mean(-1))
        return Net()


@model("M.NN.CNN", "neural", "1-D convolutional network over time (2 conv layers k=3, global average pooling)",
       params={"L": 32, "hidden": 32}, space={"L": [16, 32, 64]}, **NN)
def cnn(task="clf", L=32, hidden=32, **kw):
    return _CNN(task, L=L, hidden=hidden, **kw)


class _TCN(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        h = self.hidden

        class Block(nn.Module):
            def __init__(s, cin, d):
                super().__init__()
                s.pad = 2 * d
                s.c1 = nn.Conv1d(cin, h, 3, dilation=d)
                s.c2 = nn.Conv1d(h, h, 3, dilation=d)
                s.res = nn.Conv1d(cin, h, 1)

            def forward(s, x):
                F = torch.nn.functional
                y = F.relu(s.c1(F.pad(x, (s.pad, 0))))  # left padding only => causal convolution
                y = F.relu(s.c2(F.pad(y, (s.pad, 0))))
                return F.relu(y + s.res(x))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.b = nn.Sequential(Block(p, 1), Block(h, 2), Block(h, 4), Block(h, 8))
                s.o = nn.Linear(h, 1)

            def forward(s, x):
                return s.o(s.b(x.transpose(1, 2))[:, :, -1])
        return Net()


@model("M.NN.TCN", "neural", "temporal convolutional network: 4 residual blocks of causal dilated convolutions (d=1,2,4,8)",
       params={"L": 64, "hidden": 32}, space={"L": [32, 64]}, **NN)
def tcn(task="clf", L=64, hidden=32, **kw):
    return _TCN(task, L=L, hidden=hidden, **kw)


class _RNN(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        cell = nn.LSTM if self.arch.get("cell") == "lstm" else nn.GRU
        h = self.hidden

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.r = cell(p, h, batch_first=True)
                s.o = nn.Linear(h, 1)

            def forward(s, x):
                out, _ = s.r(x)
                return s.o(out[:, -1])
        return Net()


@model("M.NN.LSTM", "neural", "LSTM (1 layer) over the window; last hidden state -> linear head", params={"L": 32, "hidden": 32},
       space={"L": [16, 32, 64], "hidden": [16, 32]}, **NN)
def lstm(task="clf", L=32, hidden=32, **kw):
    return _RNN(task, L=L, hidden=hidden, cell="lstm", **kw)


@model("M.NN.GRU", "neural", "GRU (1 layer) over the window; last hidden state -> linear head", params={"L": 32, "hidden": 32},
       space={"L": [16, 32, 64], "hidden": [16, 32]}, **NN)
def gru(task="clf", L=32, hidden=32, **kw):
    return _RNN(task, L=L, hidden=hidden, cell="gru", **kw)


class _Attn(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        h = self.hidden

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Linear(p, h)
                s.q = nn.Parameter(torch.randn(h) / h ** 0.5)
                s.o = nn.Linear(h, 1)

            def forward(s, x):
                e = torch.tanh(s.emb(x))                    # B, L, h
                a = torch.softmax(e @ s.q, dim=1)           # B, L
                return s.o((a.unsqueeze(-1) * e).sum(1))
        return Net()


@model("M.NN.ATTENTION", "neural", "additive attention pooling: learned query attends over embedded time steps",
       params={"L": 32, "hidden": 32}, space={"L": [16, 32, 64]}, **NN)
def attention(task="clf", L=32, hidden=32, **kw):
    return _Attn(task, L=L, hidden=hidden, **kw)


class _Transformer(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        h, L = self.hidden, self.L

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Linear(p, h)
                s.pos = nn.Parameter(torch.zeros(1, L, h))
                layer = nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.1, batch_first=True)
                s.enc = nn.TransformerEncoder(layer, 2)
                s.o = nn.Linear(h, 1)

            def forward(s, x):
                mask = torch.triu(torch.full((L, L), float("-inf")), 1)  # causal self-attention
                return s.o(s.enc(s.emb(x) + s.pos, mask=mask)[:, -1])
        return Net()


@model("M.NN.TRANSFORMER", "neural", "Transformer encoder (2 layers, 4 heads, learned positions, causal mask); last token -> head",
       params={"L": 32, "hidden": 32}, space={"L": [32, 64]}, **NN)
def transformer(task="clf", L=32, hidden=32, **kw):
    return _Transformer(task, L=L, hidden=hidden, **kw)


class _TFT(TorchModel):
    """Simplified Temporal Fusion Transformer: gated residual networks, softmax variable-selection network,
    LSTM encoder, interpretable multi-head attention over the encoded window, gated output. (No static
    covariates or multi-horizon quantile decoder - a reduced but faithful core.)"""

    def build(self, p):
        torch, nn = _mods()
        h, L = self.hidden, self.L

        class GRN(nn.Module):
            def __init__(s, i, o):
                super().__init__()
                s.l1, s.l2, s.g = nn.Linear(i, o), nn.Linear(o, o), nn.Linear(o, 2 * o)
                s.skip = nn.Linear(i, o) if i != o else nn.Identity()
                s.n = nn.LayerNorm(o)

            def forward(s, x):
                y = s.l2(torch.nn.functional.elu(s.l1(x)))
                return s.n(s.skip(x) + torch.nn.functional.glu(s.g(y)))

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.var_grns = nn.ModuleList([GRN(1, h) for _ in range(p)])
                s.sel = GRN(p, p)
                s.lstm = nn.LSTM(h, h, batch_first=True)
                s.att = nn.MultiheadAttention(h, 4, batch_first=True)
                s.post = GRN(h, h)
                s.o = nn.Linear(h, 1)

            def forward(s, x):
                w = torch.softmax(s.sel(x), -1)                                   # B, L, p
                v = torch.stack([g(x[..., i:i + 1]) for i, g in enumerate(s.var_grns)], -2)  # B, L, p, h
                z = (w.unsqueeze(-1) * v).sum(-2)
                e, _ = s.lstm(z)
                mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
                a, _ = s.att(e, e, e, attn_mask=mask)
                return s.o(s.post(a[:, -1] + e[:, -1]))
        return Net()


@model("M.NN.TFT", "neural", "simplified Temporal Fusion Transformer (variable selection GRNs + LSTM + causal interpretable "
       "attention + gated output)", params={"L": 24, "hidden": 16}, space={"L": [16, 24]}, **NN)
def tft(task="clf", L=24, hidden=16, **kw):
    return _TFT(task, L=L, hidden=hidden, **kw)


class _NBeats(TorchModel):
    """N-BEATS (generic): stacks of FC blocks with backcast/forecast heads and doubly-residual stacking,
    applied to the flattened window."""

    def build(self, p):
        torch, nn = _mods()
        L, h = self.L, max(self.hidden, 64)
        nblocks = self.arch.get("blocks", 3)
        pool = [max(1, min(k, L)) for k in self.arch.get("pool", [1] * nblocks)]  # N-HiTS multi-rate pooling

        class Block(nn.Module):
            def __init__(s, k):
                super().__init__()
                s.k = k
                s.inp = (L // k) * p
                s.f = nn.Sequential(nn.Linear(s.inp, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
                s.back = nn.Linear(h, L * p)
                s.fore = nn.Linear(h, 1)

            def forward(s, x):  # x: B, L, p
                xi = x
                if s.k > 1:
                    xi = torch.nn.functional.max_pool1d(x.transpose(1, 2), s.k, s.k).transpose(1, 2)
                hdn = s.f(xi.reshape(len(x), -1))
                return s.back(hdn).reshape(x.shape), s.fore(hdn)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.blocks = nn.ModuleList([Block(k) for k in pool])

            def forward(s, x):
                res, out = x, 0
                for b in s.blocks:
                    back, f = b(res)
                    res = res - back
                    out = out + f
                return out
        return Net()


@model("M.NN.NBEATS", "neural", "N-BEATS generic architecture: 3 doubly-residual FC blocks (backcast/forecast)",
       params={"L": 32, "hidden": 64}, space={"L": [16, 32]}, **NN)
def nbeats(task="clf", L=32, hidden=64, **kw):
    return _NBeats(task, L=L, hidden=hidden, blocks=3, **kw)


@model("M.NN.NHITS", "neural", "N-HiTS: N-BEATS blocks with multi-rate input max-pooling (rates 4, 2, 1) for hierarchical "
       "interpolation of scales", params={"L": 32, "hidden": 64}, space={"L": [32, 64]}, **NN)
def nhits(task="clf", L=32, hidden=64, **kw):
    return _NBeats(task, L=L, hidden=hidden, blocks=3, pool=[4, 2, 1], **kw)


class _PatchTST(TorchModel):
    def build(self, p):
        torch, nn = _mods()
        L, h = self.L, self.hidden
        P = max(1, min(self.arch.get("patch", 8), L))
        n_patch = L // P

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Linear(P, h)
                s.pos = nn.Parameter(torch.zeros(1, n_patch, h))
                s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, 0.1, batch_first=True), 2)
                s.mix = nn.Linear(p * h, 1)

            def forward(s, x):                       # B, L, p ; channel-independent patches
                B = len(x)
                xc = x[:, -n_patch * P:].transpose(1, 2).reshape(B * p, n_patch, P)
                z = s.enc(s.emb(xc) + s.pos)[:, -1]  # B*p, h
                return s.mix(z.reshape(B, p * h))
        return Net()


@model("M.NN.PATCHTST", "neural", "PatchTST: channel-independent patching (patch 8) + Transformer encoder per channel, "
       "channel mixing head", params={"L": 32, "hidden": 16}, space={"L": [32, 64]}, **NN)
def patchtst(task="clf", L=32, hidden=16, **kw):
    return _PatchTST(task, L=L, hidden=hidden, patch=8, **kw)
