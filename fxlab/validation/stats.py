"""Statistical validation: Probabilistic / Deflated Sharpe, White's Reality
Check, Hansen's SPA, stationary & block bootstrap, multiple-testing
corrections, Monte Carlo stress tests and randomised baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EULER = 0.5772156649


# ------------------------------------------------------------------------------- Sharpe inference
def psr(r: np.ndarray, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio P(SR_true > sr_star) using per-period SR, skew and kurtosis
    (Bailey & López de Prado 2012). sr_star is per-period."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = st.skew(r)
    g4 = st.kurtosis(r, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(st.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / den))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """E[max SR] among n_trials independent zero-skill strategies with SR variance var_sr."""
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER) * st.norm.ppf(1 - 1 / n_trials) +
                                     EULER * st.norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(r: np.ndarray, n_trials: int, var_sr_trials: float) -> float:
    """Deflated Sharpe Ratio: PSR against the expected maximum SR of n_trials trials
    (Bailey & López de Prado 2014). var_sr_trials: variance of per-period SR across all trials run."""
    return psr(r, expected_max_sr(n_trials, var_sr_trials))


# ------------------------------------------------------------------------------- bootstrap
def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), np.int64)
    for b in range(n_boot):
        i = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(n)
            elif t > 0:
                i = (i + 1) % n
            idx[b, t] = i
    return idx


def _stationary_idx_fast(n, mean_block, n_boot, rng):
    starts = rng.random((n_boot, n)) < 1.0 / mean_block
    starts[:, 0] = True
    new = rng.integers(0, n, (n_boot, n))
    # position = start index of current block + offset since block start
    blk_id = np.cumsum(starts, axis=1) - 1
    out = np.empty((n_boot, n), np.int64)
    for b in range(n_boot):
        s_pos = np.flatnonzero(starts[b])
        off = np.arange(n) - s_pos[blk_id[b]]
        out[b] = (new[b, s_pos][blk_id[b]] + off) % n
    return out


def block_bootstrap_ci(r: np.ndarray, stat=np.mean, mean_block: float = 20, n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float, float]:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(seed)
    idx = _stationary_idx_fast(len(r), mean_block, n_boot, rng)
    bs = np.array([stat(r[i]) for i in idx])
    return float(stat(r)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def sharpe_ci(r_daily: np.ndarray, ppy: float = 260, **kw):
    f = lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(ppy)
    return block_bootstrap_ci(r_daily, f, **kw)


# ------------------------------------------------------------------------------- data-snooping tests
def white_reality_check(perf: np.ndarray, mean_block: float = 10, n_boot: int = 2000, seed: int = 0) -> dict:
    """perf: (T, K) matrix of per-period performance differentials vs the benchmark (benchmark = 0
    exposure => simply the strategies' net returns). H0: no strategy beats the benchmark.
    Returns p-value for the best strategy (White 2000)."""
    X = np.asarray(perf, float)
    X = X[~np.isnan(X).any(1)]
    T, K = X.shape
    m = X.mean(0)
    V = np.sqrt(T) * m.max()
    rng = np.random.default_rng(seed)
    idx = _stationary_idx_fast(T, mean_block, n_boot, rng)
    Vb = np.empty(n_boot)
    for b in range(n_boot):
        Vb[b] = np.sqrt(T) * (X[idx[b]].mean(0) - m).max()
    return {"stat": float(V), "p_value": float((Vb >= V).mean()), "best": int(m.argmax()), "K": K, "T": T}


def hansen_spa(perf: np.ndarray, mean_block: float = 10, n_boot: int = 2000, seed: int = 0) -> dict:
    """Hansen (2005) Superior Predictive Ability test (consistent version, studentised)."""
    X = np.asarray(perf, float)
    X = X[~np.isnan(X).any(1)]
    T, K = X.shape
    m = X.mean(0)
    rng = np.random.default_rng(seed)
    idx = _stationary_idx_fast(T, mean_block, n_boot, rng)
    boot_means = np.stack([X[i].mean(0) for i in idx])          # (B, K)
    omega = np.sqrt(T) * boot_means.std(0) + 1e-12               # bootstrap std of sqrt(T)*mean
    t_obs = max((np.sqrt(T) * m / omega).max(), 0.0)
    thresh = -np.sqrt(2 * np.log(np.log(T))) if T > 15 else -np.inf
    mu_c = np.where(np.sqrt(T) * m / omega >= thresh, m, 0.0)
    tb = np.maximum((np.sqrt(T) * (boot_means - mu_c) / omega).max(1), 0.0)
    # Hansen (2005): Z*_k = dbar*_k - dbar_k + mu_k with mu^l = min(dbar, 0), mu^c = dbar 1{t_k <= -A_n}, mu^u = 0
    tb_l = np.maximum((np.sqrt(T) * (boot_means - np.maximum(m, 0.0)) / omega).max(1), 0.0)
    tb_u = np.maximum((np.sqrt(T) * (boot_means - m) / omega).max(1), 0.0)
    return {"stat": float(t_obs), "p_consistent": float((tb >= t_obs).mean()),
            "p_lower": float((tb_l >= t_obs).mean()), "p_upper": float((tb_u >= t_obs).mean()), "K": K, "T": T}


# ------------------------------------------------------------------------------- multiple testing
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[o])
    out = np.empty(n)
    out[o] = np.minimum(adj, 1)
    return out


def ttest_mean(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return np.nan
    return float(st.ttest_1samp(r, 0.0, alternative="greater").pvalue)


# ------------------------------------------------------------------------------- Monte Carlo
def mc_trade_drawdowns(trade_pnl: np.ndarray, n_sims: int = 2000, seed: int = 0) -> dict:
    """Distribution of max drawdown when the realised trade sequence is reshuffled."""
    x = np.asarray(trade_pnl, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {}
    rng = np.random.default_rng(seed)
    dds = np.empty(n_sims)
    for i in range(n_sims):
        eq = np.cumprod(1 + rng.permutation(x))
        dds[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"mdd_median": float(np.median(dds)), "mdd_p05": float(np.quantile(dds, 0.05)),
            "mdd_p01": float(np.quantile(dds, 0.01))}


def random_signal_baseline(simulate, n: int = 200, seed: int = 0) -> np.ndarray:
    """Run ``simulate(rng) -> sharpe`` for n random-signal strategies with the same turnover/holding
    profile as the candidate; returns the null distribution of Sharpe ratios."""
    rng = np.random.default_rng(seed)
    return np.array([simulate(np.random.default_rng(rng.integers(1 << 31))) for _ in range(n)])
