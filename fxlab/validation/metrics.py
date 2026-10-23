"""Performance metrics computed from per-bar net returns and trade logs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EPS = 1e-12


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    span_years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return len(index) / span_years if span_years > 0 else 252.0


def equity(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def max_drawdown(r: pd.Series) -> tuple[float, int]:
    eq = equity(r)
    peak = eq.cummax()
    dd = eq / peak - 1
    # duration: longest stretch below a previous peak (in bars)
    under = (dd < 0).astype(int).to_numpy()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return float(dd.min()), int(longest)


def sharpe(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    s = r.std()
    return float(r.mean() / s * np.sqrt(ppy)) if s > 0 else 0.0


def sortino(r: pd.Series, ppy: float) -> float:
    r = r.dropna()
    d = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / d * np.sqrt(ppy)) if d > 0 else 0.0


def cvar(r: pd.Series, q: float = 0.05) -> float:
    r = r.dropna().to_numpy()
    if len(r) == 0:
        return 0.0
    k = max(1, int(q * len(r)))
    return float(np.sort(r)[:k].mean())


def trading_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """FX trading day of each timestamp: days roll at 17:00 New York (so Sunday-evening UTC bars belong to
    Monday). Returned as naive dates."""
    ny = index.tz_convert("America/New_York").tz_localize(None)
    return (ny + pd.Timedelta(hours=7)).floor("D")


def daily(r: pd.Series) -> pd.Series:
    """Aggregate bar returns to FX trading days (compounded) for annualised statistics. Every bar belongs to
    exactly one Monday-Friday trading day; nothing is dropped."""
    return (1 + r.fillna(0)).groupby(trading_day(r.index)).prod() - 1


def performance(r: pd.Series, trades: pd.DataFrame | None = None, turnover: pd.Series | None = None,
                costs: pd.Series | None = None) -> dict:
    r = r.fillna(0)
    if len(r) == 0:
        return {}
    d = daily(r)
    ppy = 260.0
    eq = equity(r)
    years = max((r.index[-1] - r.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    total = float(eq.iloc[-1] - 1)
    ann = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    mdd, ddur = max_drawdown(r)
    out = {"net_return": total, "ann_return": ann, "sharpe": sharpe(d, ppy), "sortino": sortino(d, ppy),
           "calmar": ann / abs(mdd) if mdd < 0 else np.nan, "max_drawdown": mdd, "dd_duration_bars": ddur,
           "cvar_5_daily": cvar(d), "skew_daily": float(st.skew(d)) if len(d) > 3 else np.nan,
           "kurt_daily": float(st.kurtosis(d)) if len(d) > 3 else np.nan, "n_days": int(len(d)), "years": years}
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.sum() / years)
    if costs is not None:
        out["cost_per_year"] = float(costs.sum() / years)
    if trades is not None and len(trades):
        x = trades["pnl_frac_equity"] if "pnl_frac_equity" in trades else trades["net_ret"]
        wins, losses = x[x > 0], x[x < 0]
        out.update({"trades": int(len(x)), "trades_per_year": len(x) / years, "win_rate": float((x > 0).mean()),
                    "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.inf,
                    "expectancy": float(x.mean()), "payoff_ratio": float(wins.mean() / -losses.mean())
                    if len(wins) and len(losses) else np.nan,
                    "avg_bars_held": float(trades["bars_held"].mean()) if "bars_held" in trades else np.nan})
    return out


def by_period(r: pd.Series, freq: str = "YE") -> pd.Series:
    return (1 + r.fillna(0)).groupby(pd.Grouper(freq=freq)).prod() - 1


def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y))}
    # calibration: expected calibration error over 10 bins
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"n": int(len(y)), "accuracy": float(((p > 0.5) == y).mean()), "auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)), "ece": float(ece), "base_rate": float(y.mean())}


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) < 3:
        return {"n": int(len(y))}
    ic = st.spearmanr(y, yhat).statistic
    return {"n": int(len(y)), "rmse": float(np.sqrt(((y - yhat) ** 2).mean())), "mae": float(np.abs(y - yhat).mean()),
            "r2_oos": float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "ic_spearman": float(ic), "hit_rate": float((np.sign(y) == np.sign(yhat)).mean())}


def quantile_loss(y, q_pred, q) -> float:
    e = np.asarray(y) - np.asarray(q_pred)
    return float(np.nanmean(np.maximum(q * e, (q - 1) * e)))
