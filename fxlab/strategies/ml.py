"""Model-driven strategy families. Every signal is produced OUT-OF-SAMPLE by an
internal purged walk-forward: the model is trained on the trailing `train`
bars (labels whose horizon reaches into the next test block are purged and an
embargo is applied), then predicts the next `test` bars. Rows before the first
test block get no signal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.registry import strategy, CATALOG
from fxlab.strategies.base import latch, edge, out
from fxlab.labels import forward_mid_return, triple_barrier
from fxlab.validation.splits import walk_forward
from fxlab.features._util import atr

DEFAULT_FEATURES = ["F.PR.LOGRET_N", "F.MO.RSI", "F.TR.REG_SLOPE", "F.VO.HV", "F.VO.ATR_RATIO", "F.MS.TICK_IMBALANCE",
                    "F.PR.CLV", "F.DI.SKEW", "F.TR.EMA", "F.MS.SPREAD_Z", "F.TS.HOUR_CYCLE", "F.RG.VOL_PERCENTILE",
                    "F.EC.AUTOCORR", "F.PR.EFFICIENCY", "F.MO.MACD"]
WF = {"train": [3000, 6000], "test": [500, 1000]}


def feature_matrix(df, fids=DEFAULT_FEATURES, symbol="X", tf="H1", panel=None):
    from fxlab.feature_store import build_matrix
    first = {"F.PR.LOGRET_N": {"n": 1}}
    X, err = build_matrix(df, fids, symbol, tf, panel, params=first)
    return X


def oos_predictions(df, X, y, label_end, model_id, task, train=4000, test=1000, embargo=24, h=1, model_params=None,
                    min_train=None):
    """Purged walk-forward out-of-sample predictions. Returns Series aligned to df.index."""
    e = CATALOG[model_id]
    p = dict(e.params) | dict(model_params or {})
    Xv = X.to_numpy(float)
    yv = np.asarray(y, float)
    pred = np.full(len(df), np.nan)
    for sp in walk_forward(len(df), train, test, label_end=label_end, embargo=embargo, allow_partial=True):
        tr = sp.train[np.isfinite(yv[sp.train]) & np.isfinite(Xv[sp.train]).all(1)]
        if len(tr) < (train // 3 if min_train is None else min_train):
            continue
        m = e.fn(task=task, **(p | ({"h": h} if "h" in e.fn.__code__.co_varnames else {})))
        yt = (yv[tr] > 0).astype(int) if task == "clf" else yv[tr]
        if task == "clf" and len(np.unique(yt)) < 2:
            continue
        m.fit(Xv[tr], yt)
        te = sp.test
        Xt = np.nan_to_num(Xv[te])
        pred[te] = m.predict_proba(Xt) if task == "clf" else m.predict(Xt)
    return pd.Series(pred, index=df.index)


@strategy("S.ML.SUPERVISED_DIRECTION", "machine_learning", "supervised directional trading: walk-forward classifier "
          "P(forward h-bar mid return > 0); long when p > 0.5 + m, short when p < 0.5 - m",
          params={"model": "M.TREE.LGBM", "h": 5, "m": 0.03, "train": 4000, "test": 1000},
          space={"model": ["M.LIN.LOGISTIC", "M.TREE.LGBM", "M.NN.MLP"], "h": [1, 5, 20], "m": [0.02, 0.05]} | WF,
          extra={"mode": "state", "ml": True})
def s_ml_dir(df, model="M.TREE.LGBM", h=5, m=0.03, train=4000, test=1000, panel=None, symbol="X"):
    X = feature_matrix(df, symbol=symbol)
    y, le = forward_mid_return(df, h)
    p = oos_predictions(df, X, y, le, model, "clf", train, test, embargo=h + 1)
    return out(pd.Series(np.where(p > 0.5 + m, 1.0, np.where(p < 0.5 - m, -1.0, 0.0)), index=df.index), proba=p)


@strategy("S.ML.RETURN_REGRESSION", "machine_learning", "return-regression trading: walk-forward regressor of the h-bar "
          "return; trade when |prediction| exceeds k x the median quoted spread (cost hurdle)",
          params={"model": "M.LIN.RIDGE", "h": 5, "k": 1.0, "train": 4000, "test": 1000},
          space={"model": ["M.LIN.RIDGE", "M.TREE.LGBM", "M.LIN.HUBER"], "k": [0.5, 1.0, 2.0]} | WF,
          extra={"mode": "state", "ml": True})
def s_ml_reg(df, model="M.LIN.RIDGE", h=5, k=1.0, train=4000, test=1000, panel=None, symbol="X"):
    X = feature_matrix(df, symbol=symbol)
    y, le = forward_mid_return(df, h)
    p = oos_predictions(df, X, y, le, model, "reg", train, test, embargo=h + 1)
    hurdle = k * (df["spread_mean"] / df["close"]).rolling(500, min_periods=50).median()
    return out(np.sign(p) * (p.abs() > hurdle), pred=p)


@strategy("S.ML.META_LABEL", "machine_learning", "meta-labelling (López de Prado): primary EMA(20/100) crossover entries; a "
          "walk-forward classifier predicts whether each primary trade hits its 2-ATR target before its 2-ATR stop "
          "(triple barrier, horizon h); only trades with P(success) > q are taken",
          params={"model": "M.TREE.LGBM", "h": 48, "q": 0.55, "train": 4000, "test": 1000},
          space={"q": [0.5, 0.55, 0.6], "h": [24, 48]} | WF, extra={"mode": "trade", "ml": True,
          "risk": {"stop": "atr", "stop_params": {"mult": 2.0}, "target": "affine", "target_params": {"a": 0, "b": 1.0}}})
def s_meta(df, model="M.TREE.LGBM", h=48, q=0.55, train=4000, test=1000, panel=None, symbol="X"):
    from fxlab.features._util import ema
    prim = np.sign(ema(df["close"], 20) - ema(df["close"], 100))
    entries = edge(prim).fillna(0)
    a = 2 * atr(df, 14)
    lab_l, end_l = triple_barrier(df, a, a, h, side=1)
    lab_s, end_s = triple_barrier(df, a, a, h, side=-1)
    lab = np.where(entries > 0, lab_l["label"], np.where(entries < 0, lab_s["label"], np.nan))
    le = np.where(entries < 0, end_s, end_l)
    X = feature_matrix(df, symbol=symbol)
    X["primary_side"] = entries.astype("float32")
    y = pd.Series(np.where(np.isnan(lab), np.nan, (lab > 0).astype(float)), index=df.index).where(entries != 0)
    # train only on primary-entry rows: rows without an entry have NaN labels and are skipped
    p = oos_predictions(df, X, y, le, model, "clf", train, test, embargo=h + 1, min_train=40)
    take = (p > q) & (entries != 0)
    return out(entries.where(take, 0.0), proba=p)


@strategy("S.ML.TRIPLE_BARRIER", "machine_learning", "triple-barrier trading: two walk-forward classifiers estimate "
          "P(long hits +k ATR before -k ATR within h) and the same for shorts; take the side with the higher probability "
          "if it exceeds q", params={"model": "M.TREE.LGBM", "h": 24, "k": 1.5, "q": 0.55, "train": 4000, "test": 1000},
          space={"k": [1.0, 1.5, 2.0], "q": [0.5, 0.55, 0.6]} | WF, extra={"mode": "trade", "ml": True,
          "risk": {"stop": "atr", "stop_params": {"mult": 1.5}, "target": "affine", "target_params": {"a": 0, "b": 1.0}}})
def s_tb(df, model="M.TREE.LGBM", h=24, k=1.5, q=0.55, train=4000, test=1000, panel=None, symbol="X"):
    a = k * atr(df, 14)
    X = feature_matrix(df, symbol=symbol)
    ps = []
    for side in (1, -1):
        lab, end = triple_barrier(df, a, a, h, side=side)
        y = (lab["label"] > 0).astype(float).where(lab["label"].notna())
        ps.append(oos_predictions(df, X, y, end, model, "clf", train, test, embargo=h + 1))
    pl, ps_ = ps
    side = np.where((pl > ps_) & (pl > q), 1.0, np.where((ps_ > pl) & (ps_ > q), -1.0, 0.0))
    return out(edge(pd.Series(side, index=df.index)), p_long=pl, p_short=ps_)


@strategy("S.ML.RL_POLICY", "machine_learning", "reinforcement-learning policy: walk-forward trained policy (DQN/PPO/SAC/"
          "LinUCB) on 1-bar rewards net of costs; position = greedy action on the test block",
          params={"model": "M.RL.LINUCB", "train": 4000, "test": 1000},
          space={"model": ["M.RL.DQN", "M.RL.PPO", "M.RL.SAC", "M.RL.LINUCB"]} | WF, extra={"mode": "state", "ml": True})
def s_rl(df, model="M.RL.LINUCB", train=4000, test=1000, panel=None, symbol="X"):
    X = feature_matrix(df, symbol=symbol)
    y, le = forward_mid_return(df, 1)
    p = oos_predictions(df, X, y, le, model, "pol", train, test, embargo=2)
    return out(p.fillna(0))


@strategy("S.ML.REGIME_GATED_ENSEMBLE", "machine_learning", "regime-gated ensemble of strategy families: at each bar, "
          "weight a trend (EMA cross), a mean-reversion (Bollinger) and a breakout (Donchian) signal by their "
          "trailing realised Sharpe within the current volatility regime (tercile of vol percentile); weights use "
          "returns realised up to t only", params={"w": 500}, space={"w": [250, 500, 1000]}, extra={"mode": "state"})
def s_regime_ens(df, w=500, panel=None, symbol="X"):
    from fxlab.strategies.rules import s_ma_cross, s_bollinger, s_donchian
    from fxlab.features.interactions import compute
    sigs = pd.DataFrame({"trend": s_ma_cross(df)["side"], "mr": s_bollinger(df)["side"], "bo": s_donchian(df)["side"]})
    mo = (df["bid_o"] + df["ask_o"]) / 2
    r_next = mo.shift(-2) / mo.shift(-1) - 1        # return earned by a decision at t (open t+1 -> open t+2)
    realised = (sigs.mul(r_next, axis=0)).shift(2)  # known at t: decisions made at t-2 completed by open t
    reg = pd.cut(compute(df, "F.RG.VOL_PERCENTILE", n=1000), [-0.01, 1 / 3, 2 / 3, 1.01], labels=False)
    wts = pd.DataFrame(0.0, index=df.index, columns=sigs.columns)
    for k in range(3):
        m = (reg == k)
        rr = realised.where(m)
        sh = rr.rolling(w, min_periods=w // 4).mean() / (rr.rolling(w, min_periods=w // 4).std() + 1e-12)
        sh = sh.ffill()
        wts = wts.where(~m, sh.clip(lower=0))
    tot = wts.sum(axis=1).replace(0, np.nan)
    comb = (sigs * wts).sum(axis=1) / tot
    return out(comb.fillna(0).clip(-1, 1))
