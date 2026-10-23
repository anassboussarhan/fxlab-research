"""Experiment runners (executed inside campaign worker processes).

strategy_wf: nested walk-forward evaluation of one strategy family on one
instrument/timeframe:
  * candidate parameter sets = defaults + seeded random draws from the family's space;
  * every candidate is simulated ONCE over the full development series (valid because every
    strategy is tested for prefix-invariance: its signal at t does not depend on bars > t);
  * for each calendar walk-forward fold the candidate with the best TRAIN-window daily Sharpe
    (after costs, min trade count) is selected and its TEST-window returns are kept;
  * the concatenated test windows form the out-of-sample record; cost/latency stress tests
    re-simulate exactly the selected candidates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fxlab.config import CACHE_DIR, seed_everything
from fxlab.registry import CATALOG

BAR_CACHE = CACHE_DIR / "bars"
BAR_CACHE.mkdir(parents=True, exist_ok=True)


def cached_bars(symbol: str, tf: str, start=None, end=None) -> pd.DataFrame:
    from fxlab.data.store import load_bars
    from fxlab.data.acquire_fxdata_git import load_manifest
    man = load_manifest()
    ver = sorted((y, r.get("sha256", "")[:8]) for (p, y), r in man.items() if p == symbol and r.get("status") == "ok")
    import hashlib
    key = f"{symbol}_{tf}_{start}_{end}_{hashlib.sha1(json.dumps(ver).encode()).hexdigest()[:12]}"  # stable across processes
    path = BAR_CACHE / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    b = load_bars(symbol, tf, start, end)
    b.to_parquet(path)
    return b


PANEL_LAG_TF = "M5"   # event bars: other instruments are read from their M5 closes, as of the bar's start


def load_panel_closes(symbols, tf, index, start=None, end=None) -> pd.DataFrame:
    """Close prices of other instruments aligned to `index`.

    Time bars: exact timestamp alignment. Information-driven (event) bars have instrument-specific timestamps, so
    other instruments are taken as their last COMPLETED M5 close at or before the bar's start (causal; at most
    one bar stale). With FXLAB_PANEL_DIR set (cloud sweep), closes come from the pre-built panel_<tf>.parquet files
    so a job does not need every instrument's M1 history."""
    import os
    from fxlab.data.store import available, is_event_tf
    event = is_event_tf(tf)
    src_tf = PANEL_LAG_TF if event else tf
    pdir = os.environ.get("FXLAB_PANEL_DIR")
    if pdir and os.path.exists(os.path.join(pdir, f"panel_{src_tf}.parquet")):
        P = pd.read_parquet(os.path.join(pdir, f"panel_{src_tf}.parquet"))
        P = P[[s for s in symbols if s in P.columns]]
    else:
        cols = {}
        for s in symbols:
            if available(s):
                try:
                    cols[s] = cached_bars(s, src_tf, start, end)["close"]
                except FileNotFoundError:
                    pass
        P = pd.DataFrame(cols)
    if not event:
        return P.reindex(index)
    from fxlab.data.store import TF_MINUTES
    P = P.copy()
    P.index = P.index + pd.Timedelta(minutes=TF_MINUTES[src_tf])     # an M5 bar's close is known at its end
    return P.reindex(P.index.union(index)).ffill().reindex(index)


def sample_params(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    keys = [k for k, v in space.items() if isinstance(v, (list, tuple)) and len(v) > 0]
    out, seen = [], set()
    for _ in range(n * 5):
        p = {k: space[k][int(rng.integers(len(space[k])))] for k in keys}
        sig = json.dumps(p, sort_keys=True, default=str)
        if sig not in seen:
            seen.add(sig)
            out.append(p)
        if len(out) >= n:
            break
    return out


def daily_sharpe(r: pd.Series) -> float:
    from fxlab.validation.metrics import daily
    d = daily(r)
    return float(d.mean() / d.std() * np.sqrt(260)) if len(d) > 5 and d.std() > 0 else 0.0


def _trades_in(trades: pd.DataFrame, a, b) -> int:
    if trades is None or not len(trades):
        return 0
    return int(((trades["entry_time"] >= a) & (trades["entry_time"] < b)).sum())


def _breakdowns(r: pd.Series, bars: pd.DataFrame) -> dict:
    from fxlab.features.interactions import compute
    out = {}
    yr = (1 + r).groupby(r.index.year).prod() - 1
    out["by_year"] = {int(k): float(v) for k, v in yr.items()}
    try:
        vp = compute(bars, "F.RG.VOL_PERCENTILE", n=1000).reindex(r.index)
        reg = pd.cut(vp, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low_vol", "mid_vol", "high_vol"])
        out["by_vol_regime"] = {str(k): {"sum_ret": float(v.sum()), "sharpe_bars": float(v.mean() / (v.std() + 1e-12))}
                                for k, v in r.groupby(reg, observed=True)}
    except Exception as e:
        out["by_vol_regime"] = repr(e)
    hl = r.index.tz_convert("Europe/London").hour
    sess = np.where((hl >= 0) & (hl < 7), "asia", np.where(hl < 13, "london", np.where(hl < 17, "overlap", "ny_late")))
    out["by_session"] = {k: float(v.sum()) for k, v in r.groupby(sess)}
    return out


def strategy_wf(cfg: dict, outdir: Path) -> dict:
    if cfg.get("joint"):
        return strategy_wf_joint(cfg, outdir)
    from fxlab.evaluate import evaluate as _evaluate, run_signals

    def evaluate(*a, exec_cfg=None, **k):  # default: the experiment's configured execution style
        return _evaluate(*a, exec_cfg=exec_cfg if exec_cfg is not None else cfg.get("exec"),
                         financing=cfg.get("financing"), **k)

    def strip(p):  # strategy parameters only (candidate keys starting with __ are execution / risk settings)
        return {k: v for k, v in p.items() if not k.startswith("__")}
    from fxlab.execution.costs import BASE, STRESS, LATENCY
    from fxlab.validation.splits import time_walk_forward
    from fxlab.validation.metrics import performance
    from fxlab.validation import stats
    import fxlab.features, fxlab.models, fxlab.strategies  # noqa: F401

    seed_everything(cfg.get("seed", 0))
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    sid, sym, tf = cfg["target"], cfg["symbol"], cfg["tf"]
    e = CATALOG[sid]
    bars = cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))
    panel = load_panel_closes(cfg.get("panel", []), tf, bars.index, cfg.get("start"), cfg.get("end")) if e.multi_asset else None
    fixed = dict(cfg.get("fixed_params", {}))
    space = {k: v for k, v in e.space.items() if k not in fixed}
    if cfg.get("risk_grid"):  # risk-management comparison: candidates are risk configs, strategy params fixed
        cands = [dict(fixed) | {"__risk__": ro} for ro in cfg["risk_grid"]]
    else:
        cands = [dict(fixed)] + [dict(fixed) | c for c in sample_params(space, cfg.get("n_trials", 6) - 1, cfg.get("seed", 0))]
    if cfg.get("exec_grid"):  # entry order type chosen per fold on TRAINING data, jointly with the parameters
        cands = [c | {"__exec__": ex} for c in cands for ex in cfg["exec_grid"]]
    sims, errors = [], []
    sig_cache = {}
    for p in cands:
        try:
            ro = p.get("__risk__")
            sp_ = strip(p)
            key_ = json.dumps(sp_, sort_keys=True, default=str)
            if key_ not in sig_cache:
                sig_cache[key_] = run_signals(sid, bars, panel, sym, sp_)
            res = evaluate(sid, bars, sym, panel, sp_, BASE, risk_override=ro, signals=sig_cache[key_],
                           mode_override=cfg.get("mode_override"), exec_cfg=p.get("__exec__"))
            sims.append((p, res))
        except Exception as ex:
            errors.append({"params": p, "error": repr(ex)[:500]})
    if not sims:
        raise RuntimeError(f"all {len(cands)} candidates failed: {errors[:2]}")
    folds = time_walk_forward(bars.index, cfg.get("train_years", 3), cfg.get("test_months", 6))
    oos = pd.Series(np.nan, index=bars.index)
    fold_rows, chosen = [], []
    index_of = {json.dumps(p, sort_keys=True, default=str): i for i, (p, _) in enumerate(sims)}

    def sim_for(p):
        k = json.dumps(p, sort_keys=True, default=str)
        if k not in index_of:
            sig = run_signals(sid, bars, panel, sym, p)
            sims.append((p, evaluate(sid, bars, sym, panel, p, BASE, signals=sig)))
            index_of[k] = len(sims) - 1
        return index_of[k]

    def train_score(i, tr_idx):
        res = sims[i][1]
        ntr = _trades_in(res["trades"], tr_idx[0], tr_idx[-1])
        return daily_sharpe(res["returns"].loc[tr_idx[0]:tr_idx[-1]]) if ntr >= cfg.get("min_trades", 10) else -np.inf

    for f in folds:
        tr_idx, te_idx = bars.index[f.train], bars.index[f.test]
        best, best_s = None, -np.inf
        if cfg.get("optimizer") == "optuna" and not cfg.get("risk_grid"):
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            keys = [k for k, v in e.space.items() if isinstance(v, (list, tuple)) and v]

            def objective(trial):
                p = {k: e.space[k][trial.suggest_int(k, 0, len(e.space[k]) - 1)] for k in keys}
                try:
                    v = train_score(sim_for(p), tr_idx)
                except Exception:
                    return -10.0
                return v if np.isfinite(v) else -10.0
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 0) + f.fold))
            study.optimize(objective, n_trials=cfg.get("n_trials", 6))
            bp = {k: e.space[k][v] for k, v in study.best_params.items()}
            best, best_s = sim_for(bp), study.best_value
        for i, (p, res) in enumerate(sims):
            if best is not None and cfg.get("optimizer") == "optuna":
                break
            s = train_score(i, tr_idx)
            if s > best_s:
                best, best_s = i, s
        if best is None:
            best = 0
        p, res = sims[best]
        rte = res["returns"].loc[te_idx[0]:te_idx[-1]]
        oos.loc[rte.index] = rte
        chosen.append(best)
        fold_rows.append({"fold": f.fold, "train_start": tr_idx[0], "train_end": tr_idx[-1], "test_start": te_idx[0],
                          "test_end": te_idx[-1], "selected": p, "train_metrics": {"sharpe": best_s},
                          "test_metrics": {"sharpe": daily_sharpe(rte), "ret": float((1 + rte).prod() - 1),
                                           "trades": _trades_in(res["trades"], te_idx[0], te_idx[-1])}})
    oos_r = oos.dropna()
    if oos_r.empty:
        raise RuntimeError("no walk-forward folds (insufficient history)")
    # OOS trade log restricted to test windows of the selected candidate in each fold
    tl = []
    for f, b in zip(fold_rows, chosen):
        t = sims[b][1]["trades"]
        if t is not None and len(t):
            tl.append(t[(t["entry_time"] >= f["test_start"]) & (t["entry_time"] <= f["test_end"])])
    trades = pd.concat(tl) if tl else pd.DataFrame()
    m = performance(oos_r, trades if len(trades) else None)
    from fxlab.validation.metrics import daily
    d = daily(oos_r)
    m["psr_0"] = stats.psr(d.to_numpy())
    m["ttest_p"] = stats.ttest_mean(d.to_numpy())
    est, lo, hi = stats.sharpe_ci(d.to_numpy(), n_boot=500)
    m["sharpe_ci95"] = [lo, hi]
    m["n_candidates"] = len(sims)
    m["candidate_errors"] = len(errors)
    m["in_sample_default"] = {k: sims[0][1]["metrics"].get(k) for k in ("sharpe", "ann_return", "max_drawdown", "trades")}
    # stress tests: re-simulate the SELECTED candidate for each fold under adverse execution
    stress = {}
    for name, cm in list({f"cost_x{k}": v for k, v in STRESS.items()}.items()) + [("latency1", LATENCY[1])]:
        sr = pd.Series(np.nan, index=bars.index)
        try:
            for b in sorted(set(chosen)):  # simulate each selected candidate once, then slice its test windows
                p = sims[b][0]
                res = evaluate(sid, bars, sym, panel, strip(p), cm,
                               risk_override=p.get("__risk__"), signals=sims[b][1]["signals"],
                               mode_override=cfg.get("mode_override"), exec_cfg=p.get("__exec__"))
                for f, bb in zip(fold_rows, chosen):
                    if bb == b:
                        seg = res["returns"].loc[f["test_start"]:f["test_end"]]
                        sr.loc[seg.index] = seg
        except Exception as ex:
            stress[name] = repr(ex)
            continue
        s2 = sr.dropna()
        stress[name] = {"sharpe": daily_sharpe(s2), "ann_return": performance(s2).get("ann_return")}
    m["stress"] = stress
    # zero-cost diagnostic (never primary)
    from fxlab.execution.costs import ZERO
    try:
        z = evaluate(sid, bars, sym, panel, strip(sims[0][0]), ZERO,
                     risk_override=sims[0][0].get("__risk__"), signals=sims[0][1]["signals"],
                     exec_cfg=sims[0][0].get("__exec__"))
        m["diagnostic_zero_cost_default_sharpe"] = z["metrics"].get("sharpe")
    except Exception:
        pass
    m["breakdowns"] = _breakdowns(oos_r, bars)
    m["param_stability"] = float(len(set(chosen)) / max(1, len(chosen)))  # 1/len => always same params
    m["runtime_s"] = time.time() - t0
    oos_side = pd.Series(np.nan, index=bars.index)
    for f, b in zip(fold_rows, chosen):
        sd = sims[b][1]["signals"]["side"].loc[f["test_start"]:f["test_end"]]
        oos_side.loc[sd.index] = sd
    pd.DataFrame({"ret": oos_r, "side": oos_side.reindex(oos_r.index)}).to_parquet(outdir / "oos_returns.parquet")
    if len(trades):
        trades.to_parquet(outdir / "oos_trades.parquet")
    pd.DataFrame(fold_rows).astype({"selected": str, "train_metrics": str, "test_metrics": str}).to_parquet(
        outdir / "folds.parquet")
    return {"metrics": m, "folds": fold_rows,
            "artifacts": {"oos_returns": str(outdir / "oos_returns.parquet"),
                          "oos_trades": str(outdir / "oos_trades.parquet") if len(trades) else None,
                          "candidate_errors": errors}}


def strategy_wf_joint(cfg: dict, outdir: Path) -> dict:
    """JOINT walk-forward optimisation of strategy parameters x entry order type x exit (risk) configuration for
    one instrument and timeframe.

    cfg["joint"] = {"exec": [None, {...}, ...], "risk": [None, {...}, ...], "n_trials": new trials per fold,
                    "n_startup": random joint candidates before the first fold}
      exec None = market entry at the next open; risk None = the family's own exit (signal flat/reversal for
      target-position families, the family's default stop/target for trade-mode families); a risk dict runs the
      trade engine with that stop / target / trail / breakeven / partial / time-exit / sizing configuration.
    In every fold a TPE study searches the joint space using the TRAINING window only. It is warm-started with
    every candidate simulated so far (each re-scored on this fold's training window; earlier folds only ever saw
    earlier data) and then asks n_trials new candidates. The best training-window candidate (daily Sharpe, at
    least min_trades trades) trades the following test window. Selection is per instrument (one job = one pair).
    Diagnostic 'abstain' stream: folds whose best training Sharpe <= 0 stay flat (decided on training data)."""
    import collections
    import optuna
    from optuna.distributions import IntDistribution
    from fxlab.evaluate import evaluate as _evaluate, run_signals
    from fxlab.execution.costs import BASE, STRESS, LATENCY, ZERO
    from fxlab.validation.splits import time_walk_forward
    from fxlab.validation.metrics import performance, daily, trading_day
    from fxlab.validation import stats
    import fxlab.features, fxlab.models, fxlab.strategies  # noqa: F401

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    sid, sym, tf = cfg["target"], cfg["symbol"], cfg["tf"]
    e = CATALOG[sid]
    bars = cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))
    panel = load_panel_closes(cfg.get("panel", []), tf, bars.index, cfg.get("start"), cfg.get("end")) if e.multi_asset else None
    J = cfg["joint"]
    exec_opts = list(J.get("exec") or [None])
    risk_opts = list(J.get("risk") or [None])
    fixed = dict(cfg.get("fixed_params", {}))
    tune = not e.extra.get("ml")          # model-driven families: params fixed (retraining cost), exec x exit tuned
    pkeys = [k for k, v in e.space.items() if tune and k not in fixed and isinstance(v, (list, tuple)) and len(v) > 1]
    dims = {k: len(e.space[k]) for k in pkeys}
    dims.update({k: n for k, n in (("__exec__", len(exec_opts)), ("__risk__", len(risk_opts))) if n > 1})
    dists = {k: IntDistribution(0, n - 1) for k, n in dims.items()}
    base_mode = e.extra.get("mode", "state")
    min_trades = cfg.get("min_trades", 10)
    n = len(bars)
    td = trading_day(bars.index)
    codes = pd.factorize(td)[0]                         # increasing day codes (bars are sorted)
    day_last = np.flatnonzero(np.r_[codes[1:] != codes[:-1], True])   # last bar position of each trading day

    def decode(idx):
        p = dict(fixed) | {k: e.space[k][idx[k]] for k in pkeys if k in idx}
        return p, exec_opts[idx.get("__exec__", 0)], risk_opts[idx.get("__risk__", 0)]

    def describe(idx):
        p, ex, ro = decode(idx)
        return {"params": p, "exec": ex, "risk": ro}

    sig_lru = collections.OrderedDict()

    def signals_for(p):
        k = json.dumps(p, sort_keys=True, default=str)
        if k in sig_lru:
            sig_lru.move_to_end(k)
            return sig_lru[k]
        s = run_signals(sid, bars, panel, sym, p)
        sig_lru[k] = s
        if len(sig_lru) > 6:
            sig_lru.popitem(last=False)
        return s

    def simulate(idx, cost):
        p, ex, ro = decode(idx)
        mode = "trade" if (ro is not None and base_mode == "state") else cfg.get("mode_override")
        return _evaluate(sid, bars, sym, panel, p, cost, risk_override=ro, signals=signals_for(p),
                         mode_override=mode, exec_cfg=ex, financing=cfg.get("financing"))

    sims, by_key, failed, errors = [], {}, set(), []

    def run(idx):
        """Simulate a joint candidate over the full history at BASE costs; keep only compact arrays."""
        key = json.dumps(idx, sort_keys=True)
        if key in by_key:
            return by_key[key]
        if key in failed:
            return None
        try:
            res = simulate(idx, BASE)
        except Exception as ex:
            failed.add(key)
            errors.append({"candidate": describe(idx), "error": repr(ex)[:500]})
            return None
        r = res["returns"].reindex(bars.index).fillna(0.0).to_numpy(np.float64)
        clp = np.r_[0.0, np.cumsum(np.log1p(np.clip(r, -0.999, None)))]
        tr = res["trades"]
        ent = (np.sort(bars.index.get_indexer(pd.DatetimeIndex(tr["entry_time"]))) if tr is not None and len(tr)
               else np.zeros(0, np.int64))
        sims.append({"idx": dict(idx), "clp": clp, "entries": ent})
        by_key[key] = len(sims) - 1
        return by_key[key]

    def score(i, i0, i1):
        """Daily Sharpe of candidate i over bar positions [i0, i1] using only bars inside the window."""
        s = sims[i]
        ntr = int(np.searchsorted(s["entries"], i1, "right") - np.searchsorted(s["entries"], i0, "left"))
        if ntr < min_trades:
            return -np.inf
        c0, c1 = codes[i0], codes[i1]
        b = np.r_[i0 - 1, day_last[c0:c1], i1]
        d = np.expm1(np.diff(s["clp"][b + 1]))
        sd = d.std()
        return float(d.mean() / sd * np.sqrt(260)) if len(d) > 5 and sd > 0 else -np.inf

    rng = np.random.default_rng(seed)
    run({})                                             # the family's defaults, market entry, own exit
    for _ in range(J.get("n_startup", 8) if dims else 0):
        run({k: int(rng.integers(m)) for k, m in dims.items()})
    if not sims:
        raise RuntimeError(f"all initial candidates failed: {errors[:2]}")
    folds = time_walk_forward(bars.index, cfg.get("train_years", 3), cfg.get("test_months", 6))
    if not folds:
        raise RuntimeError("no walk-forward folds (insufficient history)")

    def select(i0, i1, fold_seed, n_new):
        """TPE over the joint space on bars [i0, i1], warm-started with all simulated candidates."""
        if dims and n_new > 0:
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=fold_seed, n_startup_trials=8))
            clip = lambda v: float(np.clip(v, -10, 10)) if np.isfinite(v) else -10.0   # noqa: E731
            study.add_trials([optuna.trial.create_trial(params={k: s["idx"][k] for k in dims if k in s["idx"]},
                                                        distributions={k: dists[k] for k in dims if k in s["idx"]},
                                                        value=clip(score(j, i0, i1)))
                              for j, s in enumerate(sims)])
            for _ in range(n_new):
                trial = study.ask(fixed_distributions=dists)
                j = run(dict(trial.params))
                study.tell(trial, -10.0 if j is None else clip(score(j, i0, i1)))
        sc = np.array([score(j, i0, i1) for j in range(len(sims))])
        best = int(np.argmax(sc))
        fin = sc[np.isfinite(sc)]
        # deflation benchmark: expected maximum training Sharpe of len(fin) zero-skill trials with the observed
        # cross-candidate Sharpe variance (Bailey & Lopez de Prado); used only for the 'deflated' diagnostic stream
        sr0 = stats.expected_max_sr(len(fin), float(fin.var())) if len(fin) > 1 else 0.0
        return best, float(sc[best]), float(sr0)

    fold_rows, chosen = [], []
    oos = np.full(n, np.nan)
    oos_abs = np.full(n, np.nan)
    oos_dfl = np.full(n, np.nan)
    for f in folds:
        i0, i1 = int(f.train[0]), int(f.train[-1])
        best, best_s, sr0 = select(i0, i1, seed + 1000 * f.fold, J.get("n_trials", 16))
        t0_, t1_ = int(f.test[0]), int(f.test[-1])
        rte = np.expm1(np.diff(sims[best]["clp"][t0_:t1_ + 2]))
        oos[t0_:t1_ + 1] = rte
        oos_abs[t0_:t1_ + 1] = rte if best_s > 0 else 0.0
        oos_dfl[t0_:t1_ + 1] = rte if best_s > sr0 else 0.0
        chosen.append(best)
        ntest = int(np.searchsorted(sims[best]["entries"], t1_, "right") - np.searchsorted(sims[best]["entries"], t0_, "left"))
        rs = pd.Series(rte, index=bars.index[t0_:t1_ + 1])
        fold_rows.append({"fold": f.fold, "train_start": bars.index[i0], "train_end": bars.index[i1],
                          "test_start": bars.index[t0_], "test_end": bars.index[t1_],
                          "selected": describe(sims[best]["idx"]),
                          "train_metrics": {"sharpe": best_s, "deflation_benchmark": sr0},
                          "test_metrics": {"sharpe": daily_sharpe(rs), "ret": float((1 + rs).prod() - 1),
                                           "trades": ntest, "abstain": bool(not best_s > 0),
                                           "deflated_abstain": bool(not best_s > sr0)},
                          "n_simulated": len(sims)})
    test_mask = ~np.isnan(oos)
    oos_r = pd.Series(oos[test_mask], index=bars.index[test_mask])
    oos_a = pd.Series(oos_abs[test_mask], index=bars.index[test_mask])
    oos_d = pd.Series(oos_dfl[test_mask], index=bars.index[test_mask])
    # re-simulate each selected candidate once: OOS trade log, executed sides, stress tests
    uniq = sorted(set(chosen))
    full = {}
    for b in uniq:
        full[b] = simulate(sims[b]["idx"], BASE)
    tl, side = [], pd.Series(np.nan, index=bars.index)
    for fr, b in zip(fold_rows, chosen):
        t = full[b]["trades"]
        if t is not None and len(t):
            tl.append(t[(t["entry_time"] >= fr["test_start"]) & (t["entry_time"] <= fr["test_end"])])
        sd = full[b]["signals"]["side"].loc[fr["test_start"]:fr["test_end"]]
        side.loc[sd.index] = sd
    trades = pd.concat(tl) if tl else pd.DataFrame()
    m = performance(oos_r, trades if len(trades) else None)
    d = daily(oos_r)
    m["psr_0"] = stats.psr(d.to_numpy())
    m["ttest_p"] = stats.ttest_mean(d.to_numpy())
    est, lo, hi = stats.sharpe_ci(d.to_numpy(), n_boot=500)
    m["sharpe_ci95"] = [lo, hi]
    m["n_candidates"] = len(sims)
    m["candidate_errors"] = len(errors)
    m["joint_space"] = {"param_dims": {k: dims[k] for k in pkeys if k in dims}, "exec_options": len(exec_opts),
                        "risk_options": len(risk_opts), "size": int(np.prod(list(dims.values()))) if dims else 1}
    pa = performance(oos_a)
    m["abstain_variant"] = {"sharpe": pa.get("sharpe"), "ann_return": pa.get("ann_return"),
                            "max_drawdown": pa.get("max_drawdown"),
                            "folds_active": int(sum(1 for fr in fold_rows if not fr["test_metrics"]["abstain"])),
                            "folds": len(fold_rows)}
    pdf = performance(oos_d)
    m["deflated_variant"] = {"sharpe": pdf.get("sharpe"), "ann_return": pdf.get("ann_return"),
                             "max_drawdown": pdf.get("max_drawdown"),
                             "folds_active": int(sum(1 for fr in fold_rows if not fr["test_metrics"]["deflated_abstain"])),
                             "folds": len(fold_rows)}
    cnt = lambda xs: dict(collections.Counter(json.dumps(x, sort_keys=True, default=str) for x in xs))   # noqa: E731
    m["choice_counts"] = {"exec": cnt(fr["selected"]["exec"] for fr in fold_rows),
                          "risk": cnt(fr["selected"]["risk"] for fr in fold_rows)}
    # the configuration that would be carried forward: best on the most recent training-length window
    last = bars.index[-1]
    j0 = int(bars.index.searchsorted(last - pd.DateOffset(months=int(cfg.get("train_years", 3) * 12))))
    lb, ls, l0 = select(j0, n - 1, seed + 999_999, J.get("n_trials", 16))
    m["latest_choice"] = describe(sims[lb]["idx"]) | {"train_sharpe": ls, "deflation_benchmark": l0,
                                                       "window_start": str(bars.index[j0]),
                                                       "window_end": str(last)}
    m["n_candidates"] = len(sims)
    stress = {}
    for name, cm in list({f"cost_x{k}": v for k, v in STRESS.items()}.items()) + [("latency1", LATENCY[1])]:
        sr = pd.Series(np.nan, index=bars.index)
        try:
            for b in uniq:
                res = simulate(sims[b]["idx"], cm)
                for fr, bb in zip(fold_rows, chosen):
                    if bb == b:
                        seg = res["returns"].loc[fr["test_start"]:fr["test_end"]]
                        sr.loc[seg.index] = seg
        except Exception as ex:
            stress[name] = repr(ex)
            continue
        s2 = sr.dropna()
        stress[name] = {"sharpe": daily_sharpe(s2), "ann_return": performance(s2).get("ann_return")}
    m["stress"] = stress
    try:
        m["diagnostic_zero_cost_default_sharpe"] = simulate({}, ZERO)["metrics"].get("sharpe")
    except Exception:
        pass
    m["breakdowns"] = _breakdowns(oos_r, bars)
    m["param_stability"] = float(len(set(chosen)) / max(1, len(chosen)))
    m["runtime_s"] = time.time() - t0
    pd.DataFrame({"ret": oos_r, "side": side.reindex(oos_r.index), "ret_abstain": oos_a,
                  "ret_deflated": oos_d}).to_parquet(
        outdir / "oos_returns.parquet")
    if len(trades):
        trades.to_parquet(outdir / "oos_trades.parquet")
    pd.DataFrame(fold_rows).astype({"selected": str, "train_metrics": str, "test_metrics": str}).to_parquet(
        outdir / "folds.parquet")
    return {"metrics": m, "folds": fold_rows,
            "artifacts": {"oos_returns": str(outdir / "oos_returns.parquet"),
                          "oos_trades": str(outdir / "oos_trades.parquet") if len(trades) else None,
                          "candidate_errors": errors[:20]}}


SCREEN_FEATURES = ["F.PR.LOGRET_N", "F.PR.EFFICIENCY", "F.PR.CLV", "F.PR.RET_TSTAT", "F.TR.EMA", "F.TR.REG_SLOPE",
                   "F.TR.ADX", "F.MO.RSI", "F.MO.MACD", "F.MO.CCI", "F.VO.HV", "F.VO.ATR_RATIO", "F.VO.YANG_ZHANG",
                   "F.VO.SEMIVAR", "F.DI.SKEW", "F.DI.KURT", "F.MS.TICK_IMBALANCE", "F.MS.SPREAD_Z",
                   "F.MS.QUOTE_INTENSITY", "F.MS.LIQ_IMBALANCE", "F.ST.BREAKOUT", "F.ST.SWING_DIST",
                   "F.SP.SUPER_SMOOTHER", "F.FR.HIGUCHI", "F.IT.PERMUTATION", "F.EC.AUTOCORR", "F.EC.VARIANCE_RATIO",
                   "F.TS.HOUR_CYCLE", "F.TS.SESSION_CLOCK", "F.RG.VOL_PERCENTILE", "F.RG.CHOPPINESS"]


def model_wf(cfg: dict, outdir: Path) -> dict:
    """Prediction-problem evaluation of one model family: purged walk-forward on a feature matrix built
    from the screening feature set; target = direction / return / volatility over horizon h."""
    from fxlab.feature_store import build_matrix
    from fxlab.labels import forward_mid_return, volatility_label
    from fxlab.validation.splits import walk_forward
    from fxlab.validation.metrics import classification_metrics, regression_metrics
    import fxlab.features, fxlab.models  # noqa: F401

    seed_everything(cfg.get("seed", 0))
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    mid, sym, tf, task, h = cfg["target"], cfg["symbol"], cfg["tf"], cfg["task"], cfg["h"]
    e = CATALOG[mid]
    bars = cached_bars(sym, tf, cfg.get("start"), cfg.get("end"))
    fids = cfg.get("features", SCREEN_FEATURES)
    X, ferr = build_matrix(bars, fids, sym, tf, None, params={"F.PR.LOGRET_N": {"n": 1}})
    if e.extra.get("input") == "series":  # series models need r_t first
        X = X[[c for c in X.columns if c.startswith("F.PR.LOGRET_N")] + [c for c in X.columns if not c.startswith("F.PR.LOGRET_N")]]
    if task == "vol":
        lab, le = volatility_label(bars, h)
        y = lab["rv"]
    else:
        y, le = forward_mid_return(bars, h)
    Xv, yv = X.to_numpy(float), y.to_numpy(float)
    warm = int(np.argmax(np.isfinite(Xv).all(1))) if np.isfinite(Xv).all(1).any() else len(Xv)
    pred = np.full(len(bars), np.nan)
    folds = []
    params = dict(e.params) | dict(cfg.get("params", {}))
    if "h" in e.fn.__code__.co_varnames:
        params["h"] = h
    for sp in walk_forward(len(bars), cfg.get("train", 10000), cfg.get("test", 5000), label_end=le, embargo=h + 1,
                           allow_partial=True):
        tr = sp.train[(sp.train >= warm) & np.isfinite(yv[sp.train]) & np.isfinite(Xv[sp.train]).all(1)]
        if len(tr) < 2000:
            continue
        m = e.fn(task=task, **params)
        yt = (yv[tr] > 0).astype(int) if task == "clf" else yv[tr]
        tf0 = time.time()
        m.fit(Xv[tr], yt)
        fit_s = time.time() - tf0
        te = sp.test
        Xt = np.nan_to_num(Xv[te])
        p = m.predict_proba(Xt) if task == "clf" else np.asarray(m.predict(Xt), float)
        pred[te] = p
        ok = np.isfinite(yv[te]) & np.isfinite(p)
        fm = classification_metrics((yv[te][ok] > 0).astype(int), p[ok]) if task == "clf" else \
            regression_metrics(yv[te][ok], p[ok])
        folds.append({"fold": sp.fold, "train_start": bars.index[tr[0]], "train_end": bars.index[tr[-1]],
                      "test_start": bars.index[te[0]], "test_end": bars.index[te[-1]], "selected": params,
                      "train_metrics": {"n": int(len(tr)), "fit_seconds": fit_s}, "test_metrics": fm})
    ok = np.isfinite(pred) & np.isfinite(yv)
    if ok.sum() < 100:
        raise RuntimeError("too few out-of-sample predictions")
    m_all = classification_metrics((yv[ok] > 0).astype(int), pred[ok]) if task == "clf" else regression_metrics(yv[ok], pred[ok])
    if task == "clf":  # economic translation (diagnostic, not the trading evaluation)
        sgn = np.sign(pred[ok] - 0.5)
        m_all["sign_ret_mean_bps"] = float(1e4 * np.mean(sgn * yv[ok]))
    per_year = {}
    yr = bars.index.year[ok]
    for Y in np.unique(yr):
        sel = yr == Y
        try:
            per_year[int(Y)] = (classification_metrics((yv[ok][sel] > 0).astype(int), pred[ok][sel]).get("auc")
                                if task == "clf" else regression_metrics(yv[ok][sel], pred[ok][sel]).get("ic_spearman"))
        except Exception:
            per_year[int(Y)] = None
    m_all["per_year"] = per_year
    m_all["n_folds"] = len(folds)
    m_all["feature_errors"] = ferr
    m_all["runtime_s"] = time.time() - t0
    pd.DataFrame({"pred": pred, "y": yv}, index=bars.index).to_parquet(outdir / "oos_predictions.parquet")
    return {"metrics": m_all, "folds": folds, "artifacts": {"oos_predictions": str(outdir / "oos_predictions.parquet")}}
