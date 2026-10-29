"""Trading as an optimisation problem: the entry, the side, the take-profit, the stop and the time limit of every
trade are explicit functions of the market state, f(x; theta), and theta is found by maximising the strategy's own
net profit (after spread, slippage, commission and swap) on the training years: no grid of combinations is
back-tested and no barrier is fixed by hand. Development data only (to 2025-12-31); the test year is never used for
any choice.

THE OPTIMISATION PROBLEM
  For a bar t with features x_t the policy outputs 8 numbers o = f(x_t; theta):
    o1, o2 -> probabilities of (flat, long, short) = softmax(0, o1, o2)
    o3..o5 -> long take-profit, stop and time limit; o6..o8 -> the same for a short. Each goes through
              u = 12 * sigmoid(o), a continuous coordinate on the axes take-profit = 0.25 * 2^(u/2) ATR (0.25 to
              16 ATR), stop = 0.25 * 2^(u/2) ATR, time limit = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96 bars
              (linear between those points).
  The realised net return of any (side, take-profit, stop, time limit) for a bar is precomputed once on the M1
  bid/ask path at the 13 x 13 x 13 axis points (one pass over the path gives the first-passage minute of every
  level; same fill rules as scripts/ml_adaptive_barrier_lab.py: market entry at the next bar's open, stop checked
  first within a minute, gaps through the stop fill at the open, target is a limit, swap per night), and between
  axis points the return is interpolated (trilinear), so the profit is a piecewise-smooth function of theta.
  Objective: J(theta) = mean over training bars of r_t = p_long R_long + p_short R_short (the expected net return of
    the stochastic policy, in bp per bar, as if a unit position could be opened at every bar), its Sharpe ratio
    mean(r) / std(r), or the return per bar held (each trade's net return divided by its holding time in bars, at
    least one: the objective of a single account that is either in a trade or waiting); which one is tuned.
  Function classes (each a separate pre-registered test):
    linear: o = W x + b (standardised features);
    mlp: 1 or 2 tanh hidden layers of 8-64 units;
    gbm: gradient-boosted trees fitted directly to the objective (LightGBM with a custom objective: every round
         fits the trees to the gradient of J with respect to the 8 outputs, so the trees are the function).
  linear and mlp are fitted by Adam on the exact gradient of J (computed analytically, checked against finite
  differences in tests/test_policy_opt_lab.py).

FIXED BEFORE ANY RESULT (the search spaces, not the values):
  Features: the 55 of scripts/ml_adaptive_barrier_lab.py (price, volatility, trend, session, calendar, Gotobi, fix
    timing, carry), known at the close of the bar.
  Sample: up to 3,000 random bars per calendar year (all bars on D1 and H4 where fewer), purged so that no
    training trade (up to 96 bars) reaches into the validation year and no validation trade into the test year.
  Tuning (Optuna TPE on the year before each test year, 8 trials linear / 8 mlp / 6 gbm): learning rate, L2
    weight decay or tree settings (leaves 4-63, learning rate 0.01-0.3, samples per leaf 50-2,000, row and column
    subsampling, L2), hidden units and layers (mlp), objective (mean, Sharpe or per bar held); early stopping on the validation
    year's objective; trading margin kappa in (0, 0.5, 1, 2) logit units (trade when max(o1, o2) > kappa). The
    trial value is the validation year's realised total net return with the hard policy (exact M1 simulation,
    one position at a time). The chosen settings are refitted on every earlier year with the same number of steps.
  Validation gate: a test year is traded only if the chosen policy's validation-year net total was positive (the
    ungated result is reported as a diagnostic).
  Walk-forward: test years from the fourth year of data. Control: 20 sets of random entries with the policy's
    yearly trade count, long share and its chosen barriers, compared before and after costs.
PRE-REGISTERED TEST (one-sided): mean net return per trade > 0 (t clustered by month), per instrument x
  timeframe x function class; Benjamini-Hochberg over all of them in the combine stage.
Usage: python scripts/policy_opt_lab.py --symbol EURUSD --tf H1 --out DIR
       python scripts/policy_opt_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_triple_barrier_lab as TB  # noqa: E402

NA = 13
TPL = 0.25 * 2.0 ** (np.arange(NA) / 2.0)          # 0.25 .. 16 ATR
SLL = TPL.copy()
HL = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96], dtype=np.int64)
HMAX = int(HL[-1])
KAPPAS = (0.0, 0.5, 1.0, 2.0)
FORMS = ("linear", "mlp", "gbm")
BIGI = 1 << 62
EPS = 1e-6


# ----------------------------------------------------------------------------------------------- payoff tensor
@njit(cache=True)
def payoff_tensor(bo, ao, bh, bl, ah, al, first_m1, T, atr, TPL, SLL, HL, sf, sp, comm, rc, sw_long, sw_short, mpb):
    """Net return in bp of every (side, take-profit, stop, time limit) on the axes, for decision bars T (entry at
    the open of bar t+1), and the same divided by the holding time in bars (at least 1; mpb = M1 rows per bar).
    Shapes (n, 2, nA, nB, nC); side 0 = long, 1 = short. Caller guarantees t+1+HL[-1] exists."""
    n = len(T)
    nA, nB, nC = len(TPL), len(SLL), len(HL)
    out = np.empty((n, 2, nA, nB, nC), dtype=np.float32)
    rate = np.empty((n, 2, nA, nB, nC), dtype=np.float32)
    nbar = len(first_m1)
    nm = len(bo)
    tau = np.empty(nA, np.int64)
    sig = np.empty(nB, np.int64)
    pxs = np.empty(nB)
    Jc = np.empty(nC, np.int64)
    for q in range(n):
        t = T[q]
        i = first_m1[t + 1]
        a = atr[t]
        for c in range(nC):
            Jc[c] = min(first_m1[min(t + 1 + HL[c], nbar - 1)], nm - 1)
        Jmax = Jc[nC - 1]
        si = sf * (ao[i] - bo[i]) + sp
        for s in range(2):
            lg = s == 0
            E = ao[i] + si if lg else bo[i] - si
            for k in range(nA):
                tau[k] = BIGI
            for k in range(nB):
                sig[k] = BIGI
            na = 0
            nb = 0
            for x in range(i, Jmax):
                if lg:
                    while nb < nB and bl[x] <= E - SLL[nb] * a:
                        sig[nb] = x
                        nb += 1
                    while na < nA and bh[x] >= E + TPL[na] * a:
                        tau[na] = x
                        na += 1
                else:
                    while nb < nB and ah[x] >= E + SLL[nb] * a:
                        sig[nb] = x
                        nb += 1
                    while na < nA and al[x] <= E - TPL[na] * a:
                        tau[na] = x
                        na += 1
                if na == nA and nb == nB:
                    break
            for b in range(nB):
                xs = sig[b]
                if xs < BIGI:
                    sx = sf * (ao[xs] - bo[xs]) + sp
                    if lg:
                        SLp = E - SLL[b] * a
                        pxs[b] = (min(SLp, bo[xs]) - sx) if xs > i else SLp - sx
                    else:
                        SLp = E + SLL[b] * a
                        pxs[b] = (max(SLp, ao[xs]) + sx) if xs > i else SLp + sx
            sw = sw_long[i] if lg else sw_short[i]
            for c in range(nC):
                J = Jc[c]
                sj = sf * (ao[J] - bo[J]) + sp
                pxt = bo[J] - sj if lg else ao[J] + sj
                for aa in range(nA):
                    xt = tau[aa]
                    TPp = E + TPL[aa] * a if lg else E - TPL[aa] * a
                    for b in range(nB):
                        xs = sig[b]
                        if xs < J and xs <= xt:
                            px = pxs[b]
                            k = xs
                        elif xt < J:
                            px = TPp
                            k = xt
                        else:
                            px = pxt
                            k = J
                        ret = (px / E - 1.0) if lg else (1.0 - px / E)
                        v = (ret - 2 * comm + (rc[k] - rc[i]) * sw) * 1e4
                        out[q, s, aa, b, c] = v
                        rate[q, s, aa, b, c] = v / max((k - i) / mpb, 1.0)
    return out, rate


# ----------------------------------------------------------------------------------------------- objective
def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def interp(P, s, U):
    """Trilinear interpolation of P[:, s] at coordinates U (n, 3) in [0, NA-1]; returns value and gradient."""
    n = len(U)
    i0 = np.clip(np.floor(U).astype(np.int64), 0, NA - 2)
    f = U - i0
    Pf = P.reshape(n, -1)
    base = s * NA ** 3 + i0[:, 0] * NA * NA + i0[:, 1] * NA + i0[:, 2]
    rows = np.arange(n)
    R = np.zeros(n)
    dR = np.zeros((n, 3))
    w = [(1 - f[:, k], f[:, k]) for k in range(3)]
    for da in (0, 1):
        for db in (0, 1):
            for dc in (0, 1):
                v = Pf[rows, base + da * NA * NA + db * NA + dc].astype(np.float64)
                wa, wb, wc = w[0][da], w[1][db], w[2][dc]
                R += wa * wb * wc * v
                dR[:, 0] += (1 if da else -1) * wb * wc * v
                dR[:, 1] += (1 if db else -1) * wa * wc * v
                dR[:, 2] += (1 if dc else -1) * wa * wb * v
    return R, dR


def relaxed(O, P, kind="mean", grad=True):
    """J, dJ/dO (n, 8) and the per-bar expected return r of the stochastic policy. P is the return tensor, or the
    pair (return tensor, return per bar held); kind "mean" = bp per decision bar, "sharpe" = mean(r) / std(r),
    "rate" = mean of the return per bar held (bp per bar of holding, at least one bar)."""
    if isinstance(P, tuple):
        P = P[1] if kind == "rate" else P[0]
    zL, zS = O[:, 0], O[:, 1]
    m = np.maximum(0.0, np.maximum(zL, zS))
    eF, eL, eS = np.exp(-m), np.exp(zL - m), np.exp(zS - m)
    Z = eF + eL + eS
    pL, pS = eL / Z, eS / Z
    sgL, sgS = sigmoid(O[:, 2:5]), sigmoid(O[:, 5:8])
    RL, dRL = interp(P, 0, (NA - 1) * sgL)
    RS, dRS = interp(P, 1, (NA - 1) * sgS)
    r = pL * RL + pS * RS
    n = len(r)
    if kind in ("mean", "rate"):
        J = float(r.mean())
        g = np.full(n, 1.0 / n)
    else:
        mu = r.mean()
        sd = math.sqrt(max((r * r).mean() - mu * mu, 0.0) + EPS)
        J = float(mu / sd)
        g = (1.0 / sd - mu * (r - mu) / sd ** 3) / n
    if not grad:
        return J, None, r
    dO = np.empty((n, 8))
    dO[:, 0] = g * pL * (RL - r)
    dO[:, 1] = g * pS * (RS - r)
    dO[:, 2:5] = (g * pL)[:, None] * dRL * (NA - 1) * sgL * (1 - sgL)
    dO[:, 5:8] = (g * pS)[:, None] * dRS * (NA - 1) * sgS * (1 - sgS)
    return J, dO, r


def decide(O, kappa):
    """Hard policy: trade when max(o1, o2) > kappa, on the side with the larger logit; barriers from its heads."""
    zL, zS = O[:, 0], O[:, 1]
    side = np.where(zL >= zS, 1.0, -1.0)
    go = np.maximum(zL, zS) > kappa
    U = np.where((side > 0)[:, None], (NA - 1) * sigmoid(O[:, 2:5]), (NA - 1) * sigmoid(O[:, 5:8]))
    tp = 0.25 * 2.0 ** (U[:, 0] / 2.0)
    sl = 0.25 * 2.0 ** (U[:, 1] / 2.0)
    h = np.maximum(1, np.rint(np.interp(U[:, 2], np.arange(NA), HL))).astype(float)
    return go, np.column_stack([side, tp, sl, h])


# ----------------------------------------------------------------------------------------------- function classes
class Linear:
    def __init__(self, d, rng=None, **_):
        self.params = [np.zeros((d, 8)), np.zeros(8)]
        self.decay = [True, False]

    def forward(self, X):
        self.X = X
        return X @ self.params[0] + self.params[1]

    def backward(self, dO):
        return [self.X.T @ dO, dO.sum(0)]


class MLP:
    def __init__(self, d, rng, hidden=32, layers=1, **_):
        sizes = [d] + [hidden] * layers + [8]
        self.params, self.decay = [], []
        for k, (fi, fo) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = (0.1 if k == len(sizes) - 2 else 1.0) / math.sqrt(fi)
            self.params += [rng.normal(0, scale, (fi, fo)), np.zeros(fo)]
            self.decay += [True, False]
        self.L = len(sizes) - 1

    def forward(self, X):
        self.H = [X]
        h = X
        for k in range(self.L):
            z = h @ self.params[2 * k] + self.params[2 * k + 1]
            h = np.tanh(z) if k < self.L - 1 else z
            self.H.append(h)
        return h

    def backward(self, dO):
        grads = [None] * len(self.params)
        d = dO
        for k in range(self.L - 1, -1, -1):
            grads[2 * k] = self.H[k].T @ d
            grads[2 * k + 1] = d.sum(0)
            if k:
                d = (d @ self.params[2 * k].T) * (1 - self.H[k] ** 2)
        return grads


def adam_fit(pol, X, P, Xv, Pv, kind, lr, wd, steps=2000, every=25, patience=10, fixed=None):
    """Maximise J on (X, P) by Adam; early stopping on J of (Xv, Pv) unless `fixed` steps are given."""
    m = [np.zeros_like(p) for p in pol.params]
    v = [np.zeros_like(p) for p in pol.params]
    b1, b2 = 0.9, 0.999
    best = (-np.inf, 0, [p.copy() for p in pol.params])
    bad = 0
    n_steps = fixed if fixed is not None else steps
    for it in range(1, n_steps + 1):
        _, dO, _ = relaxed(pol.forward(X), P, kind)
        grads = pol.backward(-dO)
        for k, (p, g) in enumerate(zip(pol.params, grads)):
            if pol.decay[k]:
                g = g + wd * p
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            p -= lr * (m[k] / (1 - b1 ** it)) / (np.sqrt(v[k] / (1 - b2 ** it)) + 1e-8)
        if fixed is None and it % every == 0:
            Jv = relaxed(pol.forward(Xv), Pv, kind, grad=False)[0]
            if Jv > best[0]:
                best, bad = (Jv, it, [p.copy() for p in pol.params]), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if fixed is None:
        pol.params = best[2]
        return best[1], best[0]
    return n_steps, np.nan


class Standardiser:
    def __init__(self, X):
        self.mu = np.nanmedian(X, axis=0)
        q = np.nanpercentile(X, [25, 75], axis=0)
        self.sd = np.where((q[1] - q[0]) > 1e-12, (q[1] - q[0]) / 1.349, 1.0)
        self.mu = np.nan_to_num(self.mu)

    def __call__(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -5, 5)


class GBMPolicy:
    def __init__(self, hp):
        self.hp = hp

    def fit(self, X, P, Xv, Pv, kind, rounds=None, max_rounds=400):
        import lightgbm as lgb
        state = {}

        def obj(preds, data):
            _, dO, _ = relaxed(preds, P, kind)
            g = -dO * len(preds)
            if "h" not in state:
                state["h"] = np.maximum(np.abs(g).mean(0), 1e-9)
            return g, np.broadcast_to(state["h"], g.shape).copy()

        def feval(preds, data):
            return "J", relaxed(preds, Pv, kind, grad=False)[0], True
        p = {"objective": obj, "num_class": 8, "metric": "None", "verbose": -1, "num_threads": 4, "seed": 0,
             "bagging_freq": 1, "feature_pre_filter": False, **self.hp}
        # small samples (D1, short histories): a leaf may hold at most a tenth of the rows, else LightGBM drops
        # every feature and stops ("num_features > 0")
        p["min_data_in_leaf"] = int(min(p.get("min_data_in_leaf", 20), max(20, len(X) // 10)))
        dtr = lgb.Dataset(X, label=np.zeros(len(X)), free_raw_data=False)
        if rounds is None:
            dva = lgb.Dataset(Xv, label=np.zeros(len(Xv)), reference=dtr)
            self.b = lgb.train(p, dtr, max_rounds, valid_sets=[dva], feval=feval,
                               callbacks=[lgb.early_stopping(30, verbose=False)])
            self.rounds = max(1, int(self.b.best_iteration or max_rounds))
            return self.rounds, relaxed(self.predict(Xv), Pv, kind, grad=False)[0]
        self.b = lgb.train(p, dtr, rounds)
        self.rounds = rounds
        return rounds, np.nan

    def predict(self, X):
        return self.b.predict(X, raw_score=True, num_iteration=self.rounds)


# ----------------------------------------------------------------------------------------------- execution
def execute(mk, T, go, G):
    """One position at a time, exact M1 simulation of each chosen (side, tp, sl, time limit)."""
    trades = []
    busy = -1
    for q in np.flatnonzero(go):
        t = T[q]
        i = int(mk.first_m1[min(t + 1, len(mk.first_m1) - 1)])
        if i <= busy:
            continue
        net, gr, ex = mk.simulate(np.array([t]), G[q][None, :])
        if not np.isfinite(net[0]):
            continue
        trades.append((t, i, G[q].copy(), float(net[0]), float(gr[0])))
        busy = int(ex[0])
    return trades


# ----------------------------------------------------------------------------------------------- walk-forward
def run(a):
    import optuna
    from scipy import stats as st
    import ml_adaptive_barrier_lab as AB
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    mk = AB.Market(a.symbol, a.tf)
    nbar = len(mk.bars)
    Fv = mk.F.values.astype(np.float64)
    yrs = sorted(set(mk.years))
    tidx = np.arange(nbar)
    valid = (np.isfinite(mk.F["r200"].to_numpy()) & np.isfinite(mk.atr) & (mk.atr > 0)
             & (tidx + 1 + HMAX < nbar - 1))
    rng = np.random.default_rng(0)
    samp = []
    for y in yrs:
        c = np.flatnonzero(valid & (mk.years == y))
        samp.append(np.sort(rng.choice(c, size=min(len(c), a.per_year), replace=False)) if len(c) else c)
    S = np.concatenate(samp).astype(np.int64)
    mpb = len(mk.bk.idx) / nbar                                             # M1 rows per bar
    P_all, Q_all = payoff_tensor(*mk.arr, mk.first_m1.astype(np.int64), S, mk.atr.astype(float), TPL, SLL, HL,
                                 mk.sf, mk.sp, mk.comm, mk.rc, mk.sw_long, mk.sw_short, mpb)
    print(a.symbol, a.tf, "bars", nbar, "features", Fv.shape[1], "sample", len(S), "tensor", P_all.shape,
          f"{time.time() - t0:.0f}s", flush=True)
    start = {y: int(np.flatnonzero(mk.years == y)[0]) for y in yrs}
    Sy = mk.years[S]
    results = {f: {"trades": [], "chosen": []} for f in a.forms}
    for Y in yrs[3:]:
        if time.time() - t0 > a.budget_min * 60:
            print("time budget reached before", Y, "- later years not tested", flush=True)
            break
        if Y - 1 not in start:
            continue
        sA = (S + 1 + HMAX < start[Y - 1])
        sV = (Sy == Y - 1) & (S + 1 + HMAX < start[Y])
        sB = (S + 1 + HMAX < start[Y])
        sT = (Sy == Y)
        va = np.flatnonzero((mk.years == Y - 1) & valid & (tidx + 1 + HMAX < start[Y]))
        te = np.flatnonzero((mk.years == Y) & valid)
        if sA.sum() < a.min_train or len(va) < a.min_val or not len(te) or sV.sum() < 50:
            continue
        va_s = va[:: max(1, len(va) // 25000)]

        def cap(mask, seed):
            ix = np.flatnonzero(mask)
            if len(ix) > a.max_fit:
                ix = np.sort(np.random.default_rng(seed).choice(ix, a.max_fit, replace=False))
            return ix
        iA, iB, iV, iT = cap(sA, Y), cap(sB, Y + 1), np.flatnonzero(sV), np.flatnonzero(sT)
        PA, PB, PV, PT = ((P_all[ix], Q_all[ix]) for ix in (iA, iB, iV, iT))
        XA_raw, XB_raw, XV_raw, XT_raw = Fv[S[iA]], Fv[S[iB]], Fv[S[iV]], Fv[S[iT]]
        for form in a.forms:
            tf0 = time.time()
            stdA = Standardiser(XA_raw)

            def build(hp, kind, fixed=None, X=None, P=None, Xv=None, Pv=None, std=None):
                """Fit one policy; returns (policy, steps, validation J, predictor of raw features)."""
                if form == "gbm":
                    pol = GBMPolicy({k: v for k, v in hp.items() if k not in ("kind",)})
                    steps, Jv = pol.fit(X, P, Xv, Pv, kind, rounds=fixed)
                    return pol, steps, Jv, pol.predict
                cls = Linear if form == "linear" else MLP
                pol = cls(X.shape[1], np.random.default_rng(0), hidden=hp.get("hidden", 32),
                          layers=hp.get("layers", 1))
                steps, Jv = adam_fit(pol, std(X), P, std(Xv), Pv, kind, hp["lr"], hp["wd"], fixed=fixed)
                return pol, steps, Jv, (lambda Z, pol=pol, std=std: pol.forward(std(Z)))

            def objective(trial):
                kind = trial.suggest_categorical("kind", ["mean", "sharpe", "rate"])
                if form == "gbm":
                    hp = {"num_leaves": trial.suggest_int("num_leaves", 4, 63, log=True),
                          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                          "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000, log=True),
                          "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                          "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                          "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0)}
                else:
                    hp = {"lr": trial.suggest_float("lr", 1e-3, 0.1 if form == "linear" else 0.03, log=True),
                          "wd": trial.suggest_float("wd", 1e-6, 1.0, log=True)}
                    if form == "mlp":
                        hp["hidden"] = trial.suggest_categorical("hidden", [8, 16, 32, 64])
                        hp["layers"] = trial.suggest_int("layers", 1, 2)
                pol, steps, Jv, pred = build(hp, kind, X=XA_raw, P=PA, Xv=XV_raw, Pv=PV, std=stdA)
                O = pred(Fv[va_s])
                best = (-np.inf, 0.0, 0)
                for kp in KAPPAS:
                    go, G = decide(O, kp)
                    tr = execute(mk, va_s, go, G)
                    tot = float(np.sum([x[3] for x in tr]) * 1e4) if tr else 0.0
                    if tot > best[0]:
                        best = (tot, kp, len(tr))
                trial.set_user_attr("kappa", best[1])
                trial.set_user_attr("val_trades", best[2])
                trial.set_user_attr("steps", int(steps))
                trial.set_user_attr("val_J", float(Jv))
                trial.set_user_attr("hp", hp)
                return best[0]
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=Y))
            study.optimize(objective, n_trials=a.trials_gbm if form == "gbm" else a.trials)
            bt = study.best_trial
            hp, kind, kp, steps = bt.user_attrs["hp"], bt.params["kind"], bt.user_attrs["kappa"], bt.user_attrs["steps"]
            stdB = Standardiser(XB_raw)
            polB, _, _, predB = build(hp, kind, fixed=steps, X=XB_raw, P=PB, Xv=XV_raw, Pv=PV, std=stdB)
            J_train = relaxed(predB(XB_raw), PB, kind, grad=False)[0]
            J_test = relaxed(predB(XT_raw), PT, kind, grad=False)[0]
            J_test_mean = relaxed(predB(XT_raw), PT, "mean", grad=False)[0]
            O_te = predB(Fv[te])
            go, G = decide(O_te, kp)
            tr = execute(mk, te, go, G)
            val_ok = study.best_value > 0
            for (t, i, g, net, gr) in tr:
                results[form]["trades"].append({"time": mk.bk.idx[i], "side": int(g[0]), "tp": g[1], "sl": g[2],
                                                "h": int(g[3]), "net": net, "gross": gr, "val_ok": bool(val_ok)})
            ch = {"year": Y, "kind": kind, "hp": hp, "kappa": kp, "steps": steps,
                  "validation_net_bp_total": study.best_value, "validation_trades": bt.user_attrs["val_trades"],
                  "val_ok": bool(val_ok), "J_train_refit": J_train, "J_val_earlystop": bt.user_attrs["val_J"],
                  "J_test": J_test, "J_test_mean_bp_per_bar": J_test_mean, "test_trades": len(tr),
                  "share_bars_signal": float(go.mean()), "seconds": round(time.time() - tf0, 1)}
            if form == "linear":
                W = polB.params[0]
                names = list(mk.F.columns)
                heads = ["long", "short", "tp_long", "sl_long", "h_long", "tp_short", "sl_short", "h_short"]
                ch["linear_top"] = {hd: [(names[j], round(float(W[j, k]), 4))
                                         for j in np.argsort(-np.abs(W[:, k]))[:8]] for k, hd in enumerate(heads)}
                ch["linear_bias"] = [round(float(x), 4) for x in polB.params[1]]
            results[form]["chosen"].append(ch)
            print(Y, form, kind, "val", round(study.best_value, 1), "kappa", kp, "steps", steps,
                  "J tr/te", round(J_train, 3), round(J_test, 3), "test trades", len(tr),
                  f"{time.time() - t0:.0f}s", flush=True)
    rows = []
    for form in a.forms:
        T = pd.DataFrame(results[form]["trades"])
        json.dump(results[form]["chosen"], open(os.path.join(a.out, f"chosen_{a.symbol}_{a.tf}_{form}.json"), "w"),
                  indent=1, default=str)
        row = {"symbol": a.symbol, "tf": a.tf, "form": form, "years_tested": len(results[form]["chosen"]),
               "years_traded": int(sum(c["val_ok"] for c in results[form]["chosen"])),
               "trades_ungated": int(len(T))}
        if len(T):
            T.to_csv(os.path.join(a.out, f"trades_{a.symbol}_{a.tf}_{form}.csv.gz"), index=False)
            xu = T.set_index("time").net.sort_index()
            row.update({"ungated_mean_bp": float(xu.mean() * 1e4), "ungated_t": TB.month_t(xu),
                        "ungated_gross_bp": float(T.gross.mean() * 1e4)})
        Tg = T[T.val_ok] if len(T) else T
        row["trades"] = int(len(Tg))
        if len(Tg):
            x = Tg.set_index("time").net.sort_index()
            tt = TB.month_t(x)
            draws, draws_g = control(mk, Tg, valid, a.draws)
            row.update({"mean_bp": float(x.mean() * 1e4), "t": tt,
                        "p_one_sided": float(1 - st.norm.cdf(tt)) if np.isfinite(tt) else np.nan,
                        "gross_bp": float(Tg.gross.mean() * 1e4), "hit": float((x > 0).mean()),
                        "long_share": float((Tg.side > 0).mean()), "median_tp": float(Tg.tp.median()),
                        "median_sl": float(Tg.sl.median()), "median_h": float(Tg.h.median()),
                        "random_mean_bp": float(np.nanmean(draws) * 1e4),
                        "random_p": float(np.nanmean(draws >= x.mean())),
                        "random_gross_bp": float(np.nanmean(draws_g) * 1e4),
                        "random_p_gross": float(np.nanmean(draws_g >= Tg.gross.mean())),
                        "first": str(x.index[0])[:10],
                        "years_positive": f"{int((x.groupby(x.index.year).sum() > 0).sum())}/{x.index.year.nunique()}"})
        ch = results[form]["chosen"]
        if ch:
            row.update({"J_train_median": float(np.median([c["J_train_refit"] for c in ch])),
                        "J_test_median": float(np.median([c["J_test"] for c in ch])),
                        "J_test_mean_bp_per_bar": float(np.mean([c["J_test_mean_bp_per_bar"] for c in ch]))})
        print(row, flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"tests_{a.symbol}_{a.tf}.csv"), index=False)
    json.dump({"symbol": a.symbol, "tf": a.tf, "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(a.out, f"meta_{a.symbol}_{a.tf}.json"), "w"))


def control(mk, T, valid, n_draws):
    """Random entries with the policy's yearly trade count, long share and chosen barriers."""
    draws, draws_g = [], []
    rr = np.random.default_rng(19)
    T = T.copy()
    T["year"] = pd.DatetimeIndex(T.time).year
    for _ in range(n_draws):
        nets, grs = [], []
        for Y, g in T.groupby("year"):
            cand = np.flatnonzero((mk.years == Y) & valid)
            picks = np.sort(rr.choice(cand, size=min(len(cand), 3 * len(g)), replace=False))
            combos = g[["side", "tp", "sl", "h"]].to_numpy(float)
            busy, k = -1, 0
            for t in picks:
                if k >= len(g):
                    break
                i = int(mk.first_m1[min(t + 1, len(mk.first_m1) - 1)])
                if i <= busy:
                    continue
                gg = combos[rr.integers(0, len(combos))].copy()
                gg[0] = 1.0 if rr.random() < (g.side > 0).mean() else -1.0
                n_, g_, e_ = mk.simulate(np.array([t]), gg[None, :])
                if not np.isfinite(n_[0]):
                    continue
                nets.append(n_[0])
                grs.append(g_[0])
                busy = int(e_[0])
                k += 1
        draws.append(np.mean(nets) if nets else np.nan)
        draws_g.append(np.mean(grs) if grs else np.nan)
    return np.array(draws), np.array(draws_g)


def combine(a):
    fs = glob.glob(os.path.join(a.combine, "**", "tests_*.csv"), recursive=True)
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "tests_raw.csv"), index=False)
    T = R[R.trades > 0].copy()
    from cross_market_ta_lab import bh_q
    T["q_bh"] = bh_q(T.p_one_sided.to_numpy())
    T.sort_values("p_one_sided").to_csv(os.path.join(a.out, "tests_all.csv"), index=False)
    g = R.groupby(["form", "tf"]).agg(jobs=("symbol", "size"), years_tested=("years_tested", "sum"),
                                     years_traded=("years_traded", "sum"),
                                     with_trades=("trades", lambda x: (x > 0).sum()),
                                     median_net=("mean_bp", "median"), median_gross=("gross_bp", "median"),
                                     median_random_gross=("random_gross_bp", "median"),
                                     share_net_positive=("mean_bp", lambda x: (x > 0).mean()),
                                     median_trades=("trades", "median"),
                                     J_train=("J_train_median", "median"), J_test=("J_test_median", "median"),
                                     J_test_bp_bar=("J_test_mean_bp_per_bar", "median"))
    g.to_csv(os.path.join(a.out, "by_form_timeframe.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(g.round(3).to_string())
    print(T.sort_values("p_one_sided")[["symbol", "tf", "form", "years_traded", "trades", "mean_bp", "t", "gross_bp",
                                        "hit", "median_tp", "median_sl", "median_h", "random_gross_bp",
                                        "random_p_gross", "ungated_mean_bp", "ungated_t", "q_bh",
                                        "years_positive"]].round(2).to_string())
    json.dump({"jobs": int(len(R)), "tests": int(len(T)), "q10": int((T.q_bh <= 0.10).sum()),
               "t_gt_2": int((T.t > 2).sum()), "t_lt_m2": int((T.t < -2).sum())},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forms", default=",".join(FORMS))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--trials-gbm", type=int, default=6)
    ap.add_argument("--per-year", type=int, default=3000)
    ap.add_argument("--max-fit", type=int, default=40000)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--combine", default="")
    ap.add_argument("--budget-min", type=float, default=290, help="stop starting new test years after this")
    ap.add_argument("--min-train", type=int, default=1000)
    ap.add_argument("--min-val", type=int, default=200)
    a = ap.parse_args()
    a.forms = [f for f in a.forms.split(",") if f]
    if a.combine:
        return combine(a)
    try:
        run(a)
    except Exception:                                                     # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, f"error_{a.symbol}_{a.tf}.txt"), "w").write(tb)
        msg = tb[-1800:].replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        print(f"::error title=policy_opt {a.symbol} {a.tf}::{msg}", flush=True)
        raise


if __name__ == "__main__":
    main()
