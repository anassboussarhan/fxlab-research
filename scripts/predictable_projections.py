"""Search for projections of the price in which it is NOT a random walk, and test them out of sample.

A single FX rate is close to a martingale. Projections where that may fail:

  A. MAXIMALLY PREDICTABLE PORTFOLIO (Lo & MacKinlay 1997; rank-1 canonical correlation / reduced-rank
     regression). The linear combination w of the instruments' future h-bar returns that is best predicted by a
     linear function a of lagged returns of ALL instruments (1, 5, 20-bar sums). Fitted on the training window
     with shrinkage chosen on an inner hold-out, then traded out of sample: position = forecast sign/size on w.
  B. MOST MEAN-REVERTING PORTFOLIO (Box & Tiao 1977 canonical decomposition of a VAR(1) on log levels;
     d'Aspremont 2011). The combination of price levels that is least predictable from its own level (closest
     to white noise), i.e. the fastest mean-reverting basket. Traded by fading its z-score.
     CAUTION: with crosses in the universe (PP_UNIVERSE=all) this finds the triangular identity
     (log EURJPY - log EURUSD - log USDJPY = 0 up to quote noise): a "mean-reverting" residual made of mid-quote
     asynchrony inside the spread - large gross Sharpe, untradable. The default universe has USD pairs only.
  C. VOLATILITY PROJECTION |r|. HAR model of log realised variance (Corsi 2009): out-of-sample R^2. This is the
     projection that is known not to be random, but it carries no direction.
  D. SIGN PROJECTION: out-of-sample R^2 of the direction forecast of A, against a placebo.

Walk-forward 3-year training / 6-month test, development data only (common history of the local panel,
2007-2018). Costs: the platform's vector engine per instrument (bid/ask spread, slippage, commission, swap).
PLACEBOS fit the same pipeline after breaking the link between predictors and targets in TRAINING only
(targets circularly shifted by a random offset of at least 60 bars), then trade the real test data: they give
the out-of-sample result of a projection fitted to noise.
Output: results/reports/projections/{summary.json, streams.csv, folds.csv}"""
from __future__ import annotations

import functools
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# local instruments with common 2007-2018 history (AUDJPY ends 2015 and the metals start 2010-2011 locally)
UNIVERSES = {
    # 11 pairs; the crosses are (up to quotes) exact products of USD pairs: EURJPY = EURUSD x USDJPY etc.
    "all": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY", "EURGBP", "GBPJPY",
            "EURCHF"],
    # 7 USD pairs: no triangular identity links them, so every basket is a genuine currency combination
    "usd": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
}
UNIVERSE = UNIVERSES[os.environ.get("PP_UNIVERSE", "usd")]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reports",
                   "projections_" + os.environ.get("PP_UNIVERSE", "usd"))
N_PLACEBO = int(os.environ.get("PP_PLACEBOS", 20))


@functools.lru_cache(maxsize=2)
def load_panel(tf):
    """Union of bar times over the common history; a missing bar is a flat copy of the previous close (<= 5)."""
    from fxlab.orchestration.runners import cached_bars
    bars = {s: cached_bars(s, tf, "2007-01-01", "2025-12-31") for s in UNIVERSE}
    lo = max(b.index[0] for b in bars.values())
    hi = min(b.index[-1] for b in bars.values())
    idx = None
    for b in bars.values():
        i = b.index[(b.index >= lo) & (b.index <= hi)]
        idx = i if idx is None else idx.union(i)
    px = [c for c in next(iter(bars.values())).columns if c[:4] in ("bid_", "ask_")]
    out = {}
    for s, b in bars.items():
        x = b.reindex(idx)
        last = x[["bid_c", "ask_c"]].ffill(limit=5)
        for side in ("bid", "ask"):
            for k in "ohlc":
                x[f"{side}_{k}"] = x[f"{side}_{k}"].fillna(last[f"{side}_c"])
        x["rv_intrabar"] = x["rv_intrabar"].fillna(0.0)
        out[s] = x
    keep = np.all([out[s]["bid_c"].notna().to_numpy() for s in out], axis=0)
    bars = {s: x[keep] for s, x in out.items()}
    idx = idx[keep]
    mid = pd.DataFrame({s: (b["bid_c"] + b["ask_c"]) / 2 for s, b in bars.items()}, index=idx)
    rv = pd.DataFrame({s: b["rv_intrabar"] for s, b in bars.items()}, index=idx)
    return bars, np.log(mid), rv


def features(r: pd.DataFrame) -> pd.DataFrame:
    """Lagged information at the close of t: last 1, 5 and 20-bar returns of every instrument."""
    f = {}
    for L in (1, 5, 20):
        s = r.rolling(L).sum()
        for c in r:
            f[f"{c}_r{L}"] = s[c]
    return pd.DataFrame(f)


def _shrink(C, d):
    p = C.shape[0]
    return (1 - d) * C + d * np.trace(C) / p * np.eye(p)


def _inv_sqrt(C):
    e, V = np.linalg.eigh(C)
    e = np.clip(e, 1e-12, None)
    return V @ np.diag(e ** -0.5) @ V.T


def cca1(X, Y, d):
    """Rank-1 canonical pair with shrinkage d: returns (a, w) with forecast a'X of portfolio return w'Y."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    n = len(X)
    Cxx, Cyy, Cxy = _shrink(Xc.T @ Xc / n, d), _shrink(Yc.T @ Yc / n, d), Xc.T @ Yc / n
    Wy = _inv_sqrt(Cyy)
    M = Wy @ Cxy.T @ np.linalg.solve(Cxx, Cxy) @ Wy
    e, V = np.linalg.eigh(M)
    w = Wy @ V[:, -1]
    a = np.linalg.solve(Cxx, Cxy @ w)
    if (Xc @ a).std() > 0 and np.corrcoef(Xc @ a, Yc @ w)[0, 1] < 0:
        a = -a
    return a, w


def fit_mpp(X, Y, rng=None):
    """Inner hold-out (last 25% of the training window) picks the shrinkage; refit on the whole window."""
    k = int(len(X) * 0.75)
    best, best_ic = 0.5, -np.inf
    for d in (0.2, 0.5, 0.8):
        a, w = cca1(X[:k], Y[:k], d)
        f, y = X[k:] @ a, Y[k:] @ w
        ic = np.corrcoef(f, y)[0, 1] if f.std() > 0 and y.std() > 0 else -np.inf
        if ic > best_ic:
            best, best_ic = d, ic
    a, w = cca1(X, Y, best)
    return a, w, best, best_ic


def boxtiao_mr(P, ridge=1e-3):
    """Least predictable (most mean-reverting) combination of demeaned levels under a VAR(1)."""
    Pc = P - P.mean(0)
    X0, X1 = Pc[:-1], Pc[1:]
    G = X0.T @ X0 / len(X0)
    A = np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), X0.T @ X1 / len(X0))   # x_t' = x_{t-1}' A
    G0 = Pc.T @ Pc / len(Pc)
    B = A.T @ G0 @ A                                     # covariance of the predictable part
    Wi = _inv_sqrt(G0 + 1e-10 * np.eye(len(G0)))
    e, V = np.linalg.eigh(Wi @ B @ Wi)
    return Wi @ V[:, 0], float(e[0])                     # smallest predictability ratio


def portfolio_returns(bars, weights: pd.DataFrame):
    """Net and gross returns of a time-varying portfolio (weights decided at each close), per-instrument engine."""
    from fxlab.execution.simulator import vector_backtest
    from fxlab.execution.costs import BASE, ZERO
    from fxlab.data.instruments import get as instrument
    net = gross = 0.0
    for s in weights:
        b = bars[s]
        net = net + vector_backtest(b, weights[s].fillna(0.0), instrument(s).pip, BASE)["net"]
        gross = gross + vector_backtest(b, weights[s].fillna(0.0), instrument(s).pip, ZERO)["net"]
    return net, gross


def sharpe(r):
    from fxlab.validation.metrics import daily
    d = daily(r.dropna())
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 20 and d.std() > 0 else 0.0


def run(tf, h, placebo_seed=None):
    from fxlab.validation.splits import time_walk_forward
    bars, lp, rv = load_panel(tf)
    r = lp.diff()
    F = features(r)
    fut = r[::-1].rolling(h).sum()[::-1].shift(-1)          # r_{t+1..t+h}
    idx = lp.index
    folds = time_walk_forward(idx, 3, 6)
    sig = pd.Series(np.nan, index=idx)
    W = pd.DataFrame(np.nan, index=idx, columns=lp.columns)
    sig_bt = pd.Series(np.nan, index=idx)
    W_bt = pd.DataFrame(np.nan, index=idx, columns=lp.columns)
    rows = []
    rng = np.random.default_rng(placebo_seed) if placebo_seed is not None else None
    for f in folds:
        tr = idx[f.train]
        te = idx[f.test]
        tr_ok = tr[(F.loc[tr].notna().all(1)) & (fut.loc[tr].notna().all(1))]
        tr_ok = tr_ok[:-h]                                   # the last h targets reach into the test window
        X, Y = F.loc[tr_ok].to_numpy(), fut.loc[tr_ok].to_numpy()
        if rng is not None:                                  # placebo: break the X -> Y link in training only
            Y = np.roll(Y, int(rng.integers(60, len(Y) - 60)), axis=0)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xs = (X - mu) / sd
        a, w, d, ic_in = fit_mpp(Xs, Y)
        ftr = Xs @ a
        Xte = ((F.loc[te].to_numpy() - mu) / sd)
        fte = Xte @ a
        z = np.clip(np.nan_to_num((fte - ftr.mean()) / (ftr.std() + 1e-12)) / 2, -1, 1)
        sig.loc[te] = z
        W.loc[te] = np.tile(w / np.abs(w).sum(), (len(te), 1))
        # out-of-sample information coefficient of the projected forecast
        yte = fut.loc[te].to_numpy() @ w
        ok = np.isfinite(yte) & np.isfinite(fte)
        ic = float(np.corrcoef(fte[ok], yte[ok])[0, 1]) if ok.sum() > 10 else np.nan
        # Box-Tiao most mean-reverting basket of levels
        Ptr = lp.loc[tr].to_numpy()
        if rng is not None:                                  # placebo: random basket
            wb = rng.normal(size=Ptr.shape[1]); lam = np.nan
        else:
            wb, lam = boxtiao_mr(Ptr)
        ztr = Ptr @ wb
        zte = (lp.loc[te].to_numpy() @ wb - ztr.mean()) / (ztr.std() + 1e-12)
        sig_bt.loc[te] = np.clip(-zte / 2, -1, 1)
        W_bt.loc[te] = np.tile(wb / np.abs(wb).sum(), (len(te), 1))
        rows.append({"fold": f.fold, "test_start": str(te[0]), "shrink": d, "inner_ic": ic_in, "oos_ic": ic,
                     "bt_predictability": lam})
    # overlapping h-bar forecasts: hold the average of the last h signals
    s_mpp = sig.rolling(h, min_periods=1).mean()
    pos_mpp = W.mul(s_mpp, axis=0)
    pos_bt = W_bt.mul(sig_bt, axis=0)
    out = {}
    for name, pos in (("mpp", pos_mpp), ("boxtiao", pos_bt)):
        net, gross = portfolio_returns(bars, pos)
        m = pos.notna().any(axis=1)
        out[name] = {"net": net[m], "gross": gross[m]}
    return out, pd.DataFrame(rows)


def har_r2(tf="D1"):
    """Out-of-sample R^2 of a HAR forecast of next-bar log realised variance (the non-random projection)."""
    from fxlab.validation.splits import time_walk_forward
    bars, lp, rv = load_panel(tf)
    res = {}
    for s in rv:
        y = np.log(rv[s].clip(lower=1e-12))
        X = pd.DataFrame({"d": y, "w": y.rolling(5).mean(), "m": y.rolling(22).mean()})
        tgt = y.shift(-1)
        pr, ac, sign_pr = [], [], []
        for f in time_walk_forward(y.index, 3, 6):
            tr, te = y.index[f.train], y.index[f.test]
            ok = X.loc[tr].notna().all(1) & tgt.loc[tr].notna()
            A = np.c_[np.ones(ok.sum()), X.loc[tr][ok].to_numpy()]
            beta = np.linalg.lstsq(A, tgt.loc[tr][ok].to_numpy(), rcond=None)[0]
            okt = X.loc[te].notna().all(1) & tgt.loc[te].notna()
            pr.append(np.c_[np.ones(okt.sum()), X.loc[te][okt].to_numpy()] @ beta)
            ac.append(tgt.loc[te][okt].to_numpy())
            sign_pr.append(np.full(okt.sum(), tgt.loc[tr][ok].mean()))
        p, a, b = np.concatenate(pr), np.concatenate(ac), np.concatenate(sign_pr)
        res[s] = float(1 - ((a - p) ** 2).sum() / ((a - b) ** 2).sum())
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, streams, folds = {}, [], []
    for tf, hs in (("D1", (1, 5, 20)), ("H4", (1, 6, 30))):
        for h in hs:
            real, fr = run(tf, h)
            fr["tf"], fr["h"] = tf, h
            folds.append(fr)
            plac = {"mpp": [], "boxtiao": []}
            for k in range(N_PLACEBO):
                p, _ = run(tf, h, placebo_seed=1000 + k)
                for m in plac:
                    plac[m].append((sharpe(p[m]["gross"]), sharpe(p[m]["net"])))
            for m in ("mpp", "boxtiao"):
                if m == "boxtiao" and h != hs[0]:
                    continue                                  # levels basket does not depend on h
                g, n = sharpe(real[m]["gross"]), sharpe(real[m]["net"])
                pg = np.array([x[0] for x in plac[m]]); pn = np.array([x[1] for x in plac[m]])
                key = f"{m}_{tf}_h{h}" if m == "mpp" else f"{m}_{tf}"
                summary[key] = {"gross_sharpe": g, "net_sharpe": n, "placebo_gross_mean": float(pg.mean()),
                                "placebo_gross_sd": float(pg.std()), "placebo_net_mean": float(pn.mean()),
                                "share_placebos_beating_real_gross": float((pg >= g).mean()),
                                "median_oos_ic": float(fr["oos_ic"].median()) if m == "mpp" else None,
                                "share_folds_ic_positive": float((fr["oos_ic"] > 0).mean()) if m == "mpp" else None}
                streams.append(pd.DataFrame({"key": key, "net": real[m]["net"], "gross": real[m]["gross"]}))
                print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[key].items()}, flush=True)
    summary["har_oos_r2_log_rv_D1"] = har_r2("D1")
    print("HAR OOS R2 (log RV, D1):", {k: round(v, 3) for k, v in summary["har_oos_r2_log_rv_D1"].items()})
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    pd.concat(streams).to_csv(os.path.join(OUT, "streams.csv"))
    pd.concat(folds).to_csv(os.path.join(OUT, "folds.csv"), index=False)


if __name__ == "__main__":
    main()
