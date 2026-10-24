"""Projection lab: map the price path through non-standard mathematical projections and test, on one
instrument x one timeframe, whether the NEXT price move is predictable from the projected state.

Projections (all causal: the state at bar t uses data up to the close of t only):
  ordinal_d     Bandt-Pompe ordinal pattern of the last d closes (d = 3, 4, 5, 6; d! states). Symbolic
                dynamics: a random walk visits every pattern and is unpredictable from it; deterministic or
                nonlinear dynamics show forbidden / biased patterns.
  markov_L      returns quantised to {down, flat, up} (|r| vs 0.43 x causal volatility, ~terciles) and the last
                L symbols as a context (L = 1, 2, 3, 5; 3^L states): a variable-order Markov / entropy-rate
                projection.
  simplex_E     Sugihara-May simplex projection: delay embedding of volatility-standardised returns in E
                dimensions (E = 2, 4, 8); forecast = distance-weighted next move of the E+1 nearest neighbours
                in the training library (Takens / nonlinear-deterministic forecasting).
  dc_k          intrinsic (directional-change) time (Glattfelder-Dupuis-Olsen): the path is projected onto
                alternating up/down events of size k x volatility (k = 3, 10); state = current mode x
                overshoot since the last event (in units of the threshold).
  phase         instantaneous cycle phase of the detrended log price (band-pass EMA(10)-EMA(40) and its
                derivative, each scaled causally), 8 phase sectors.
  fracdiff      fractional differencing d = 0.4 of the log price (memory-preserving stationary projection),
                causal z-score in 10 bins.
  clock         the time projection: hour of the week (day of the week on D1).
  hvg           horizontal visibility graph (Luque et al. 2009): how many earlier bars the current close
                "sees", x sign of the last move (i.i.d. data have a universal degree law)
  levy_N        path signature of the last N standardised moves (rough-path theory): level-1 increment and
                Levy area (did the move happen early or late in the window), 10 states
  record_50     record statistics: bars since the last 50-bar high / low (arcsine law for a random walk)
For bin projections the training window estimates the mean next return of each state (shrunk towards zero);
the forecast for a test bar is the mean of its state.

Out-of-sample measures per projection (walk-forward 3-year train / 6-month test, development data only):
  ic            correlation between forecast and next return; t = ic x sqrt(n)
  hit           sign hit rate (non-zero forecasts and returns)
  info_bits     out-of-sample information gain in bits per bar of the state's up/down probability over the
                training base rate (log-loss), for bin projections
  edge_bps      gross return per bar of trading sign(forecast) on every bar, in basis points
  cost_rt_bps   median round-trip cost (spread x 1.25 + slippage + commission, both sides) in basis points
  gross/net Sharpe of the cost-aware rule (trade only when |forecast| > round-trip cost), platform engine
  SURROGATE control: the same pipeline on 3 sign-randomised copies of the series (|r_t| kept, signs random).
  This keeps volatility clustering, heavy tails and the clock, but destroys any directional structure. z_sur
  is the number of surrogate standard deviations by which the real ic exceeds the surrogates.

Usage: python scripts/projection_lab.py --pair EURUSD --tf H1 [--out DIR]
Writes DIR/<pair>_<tf>.json (one row per projection)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

START, END = "2005-01-01", "2025-12-31"
LAMBDA, MIN_N = 50.0, 30
N_SUR = int(os.environ.get("PL_SURROGATES", 3))
MAX_LIB, MAX_Q = 200_000, 60_000                     # simplex: library / query caps per fold (strided)


# ----------------------------------------------------------------------------------------------- data
def load(pair, tf):
    from fxlab.data.store import load_m1
    from fxlab.orchestration.runners import cached_bars
    if tf == "M1":
        cols = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
        b = load_m1(pair, START, END, columns=cols)       # bid/ask only: all the lab and the engine need
    else:
        b = cached_bars(pair, tf, START, END)
    return b


# ----------------------------------------------------------------------------------------------- states
@njit(cache=False)
def _ordinal(c, d):
    n = len(c)
    out = np.full(n, -1, np.int64)
    w = np.empty(d)
    for t in range(d - 1, n):
        for i in range(d):
            w[i] = c[t - d + 1 + i]
        code = 0
        for i in range(d):                         # Lehmer code: number of later elements smaller than w[i]
            cnt = 0
            for j in range(i + 1, d):
                if w[j] < w[i]:
                    cnt += 1
            code = code * (d - i) + cnt
        out[t] = code
    return out


@njit(cache=False)
def _markov(sym, L):
    n = len(sym)
    out = np.full(n, -1, np.int64)
    for t in range(L - 1, n):
        s = 0
        ok = True
        for j in range(L):
            v = sym[t - j]
            if v < 0:
                ok = False
                break
            s = s * 3 + v
        if ok:
            out[t] = s
    return out


@njit(cache=False)
def _dc(lp, delta):
    """Directional-change projection: mode (0 down, 1 up) x overshoot bucket since the last confirmed event."""
    n = len(lp)
    out = np.full(n, -1, np.int64)
    mode = 1; ext = lp[0]; conf = lp[0]
    for t in range(1, n):
        dl = delta[t]
        if dl != dl or dl <= 0:
            continue
        x = lp[t]
        if mode == 1:
            if x > ext:
                ext = x
            elif ext - x >= dl:
                mode = 0; ext = x; conf = x
        else:
            if x < ext:
                ext = x
            elif x - ext >= dl:
                mode = 1; ext = x; conf = x
        os_ = abs(x - conf) / dl
        b = 0 if os_ < 0.5 else (1 if os_ < 1.0 else (2 if os_ < 2.0 else 3))
        out[t] = mode * 4 + b
    return out


@njit(cache=False)
def _hvg_back_degree(x):
    """Horizontal visibility graph (Luque et al. 2009): number of EARLIER bars that bar t 'sees' (every bar in
    between is lower than both). Monotonic stack, O(n). For i.i.d. data the degree law is universal
    P(k) = (1/3)(2/3)^(k-2); correlated or deterministic dynamics deviate."""
    n = len(x)
    out = np.zeros(n, np.int64)
    stack = np.empty(n)
    top = 0
    for t in range(n):
        v = x[t]
        k = 0
        while top > 0 and stack[top - 1] < v:
            top -= 1
            k += 1
        if top > 0:
            k += 1
            if stack[top - 1] == v:
                top -= 1
        stack[top] = v
        top += 1
        out[t] = k
    return out


@njit(cache=False)
def _levy(z, N):
    """Signature projection of the last N standardised moves as a path (time, cumulative move): level-1
    increment and the Levy area (level-2 antisymmetric part) - whether the move happened early or late."""
    n = len(z)
    dx = np.full(n, np.nan)
    area = np.full(n, np.nan)
    for t in range(N, n):
        x = 0.0
        a = 0.0
        ok = True
        for k in range(N):
            v = z[t - N + 1 + k]
            if v != v:
                ok = False
                break
            tk = k / N
            a += 0.5 * (tk * v - x / N)          # 1/2 (t dx - x dt)
            x += v
        if ok:
            dx[t] = x
            area[t] = a
    return dx, area


@njit(cache=False)
def _record(lp, N):
    """Record-time projection: bars since the last N-bar high and since the last N-bar low (arcsine law for a
    random walk), each in 3 buckets -> 9 states."""
    n = len(lp)
    out = np.full(n, -1, np.int64)
    for t in range(N - 1, n):
        imax = t; imin = t
        for k in range(t - N + 1, t + 1):
            if lp[k] >= lp[imax]:
                imax = k
            if lp[k] <= lp[imin]:
                imin = k
        a = (t - imax) / N; b = (t - imin) / N
        ba = 0 if a < 0.1 else (1 if a < 0.5 else 2)
        bb = 0 if b < 0.1 else (1 if b < 0.5 else 2)
        out[t] = ba * 3 + bb
    return out


def ewm_abs(r, span):
    return pd.Series(np.abs(r)).ewm(span=span, min_periods=span // 2).mean().to_numpy() * math.sqrt(math.pi / 2)


def states(name, lp, r, index):
    """Causal state series (int, -1 = undefined) for a bin projection."""
    if name.startswith("ordinal_"):
        return _ordinal(lp, int(name.split("_")[1]))
    if name.startswith("markov_"):
        sd = ewm_abs(r, 100)
        sym = np.where(np.isnan(sd) | np.isnan(r), -1, np.where(r > 0.43 * sd, 2, np.where(r < -0.43 * sd, 0, 1)))
        return _markov(sym.astype(np.int64), int(name.split("_")[1]))
    if name.startswith("dc_"):
        k = float(name.split("_")[1])
        return _dc(lp, k * ewm_abs(r, 200))
    if name == "phase":
        s = pd.Series(lp)
        bp = (s.ewm(span=10).mean() - s.ewm(span=40).mean()).to_numpy()
        db = np.r_[np.nan, np.diff(bp)]
        a = bp / (pd.Series(np.abs(bp)).ewm(span=200, min_periods=50).mean().to_numpy() + 1e-15)
        b = db / (pd.Series(np.abs(db)).ewm(span=200, min_periods=50).mean().to_numpy() + 1e-15)
        ang = np.arctan2(b, a)
        st = np.floor((ang + np.pi) / (2 * np.pi) * 8).astype(np.int64)
        st[np.isnan(ang)] = -1
        return np.clip(st, -1, 7)
    if name == "fracdiff":
        d, K = 0.4, 100
        w = [1.0]
        for k in range(1, K):
            w.append(-w[-1] * (d - k + 1) / k)
        fd = np.convolve(lp, np.array(w), mode="full")[:len(lp)]
        fd[:K] = np.nan
        s = pd.Series(fd)
        z = ((s - s.rolling(500, min_periods=200).mean()) / s.rolling(500, min_periods=200).std()).to_numpy()
        edges = np.array([-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28])
        st = np.searchsorted(edges, z).astype(np.int64)
        st[np.isnan(z)] = -1
        return st
    if name == "hvg":                                   # visibility-graph degree x sign of the last move
        k = np.minimum(_hvg_back_degree(lp), 8)
        st = k * 2 + (np.nan_to_num(r) > 0)
        st[:2] = -1
        return st.astype(np.int64)
    if name.startswith("levy_"):
        N = int(name.split("_")[1])
        sd = ewm_abs(r, 100)
        dx, area = _levy(np.r_[np.nan, r[1:] / sd[:-1]], N)
        a_ = pd.Series(area)
        az = (area / (a_.abs().rolling(2000, min_periods=200).mean() * 1.25 + 1e-15)).to_numpy()
        edges = np.array([-0.84, -0.25, 0.25, 0.84])
        q = np.searchsorted(edges, az)
        st = q * 2 + (dx > 0)
        st[np.isnan(az) | np.isnan(dx)] = -1
        return st.astype(np.int64)
    if name.startswith("record_"):
        return _record(lp, int(name.split("_")[1]))
    if name == "clock":
        if index.freq is None and len(index) > 2 and (index[2] - index[1]) >= pd.Timedelta(hours=20):
            return index.dayofweek.to_numpy().astype(np.int64)
        return (index.dayofweek * 24 + index.hour).to_numpy().astype(np.int64)
    raise KeyError(name)


BIN_PROJ = ["ordinal_3", "ordinal_4", "ordinal_5", "ordinal_6", "markov_1", "markov_2", "markov_3", "markov_5",
            "dc_3", "dc_10", "phase", "fracdiff", "clock", "hvg", "levy_8", "levy_32", "record_50"]
KNN_PROJ = ["simplex_2", "simplex_4", "simplex_8"]
if os.environ.get("PL_SET") == "wave2":              # only the second-wave projections
    BIN_PROJ, KNN_PROJ = ["hvg", "levy_8", "levy_32", "record_50"], []


# ----------------------------------------------------------------------------------------------- fitting
def fit_bins(st, y, tr):
    """Shrunk mean next return and up-probability per state on training positions tr."""
    s, v = st[tr], y[tr]
    ok = (s >= 0) & np.isfinite(v)
    s, v = s[ok], v[ok]
    K = int(st.max()) + 1 if st.max() >= 0 else 1
    n = np.bincount(s, minlength=K).astype(float)
    sm = np.bincount(s, weights=v, minlength=K)
    up = np.bincount(s, weights=(v > 0).astype(float), minlength=K)
    dn = np.bincount(s, weights=(v < 0).astype(float), minlength=K)
    m = np.where(n >= MIN_N, sm / (n + LAMBDA), 0.0)
    p = (up + 1) / (up + dn + 2)
    U, D = (v > 0).sum(), (v < 0).sum()
    return m, p, (U + 1) / (U + D + 2)


def simplex_forecast(z, sd, y_z, E, tr, te):
    from scipy.spatial import cKDTree
    n = len(z)
    emb_ok = np.zeros(n, bool)
    emb_ok[E - 1:] = True
    emb_ok &= ~np.isnan(sd)
    zz = np.nan_to_num(z)

    def emb(ix):
        return np.stack([zz[ix - j] for j in range(E)], axis=1)
    lib = tr[emb_ok[tr] & np.isfinite(y_z[tr])]
    if len(lib) > MAX_LIB:
        lib = lib[-MAX_LIB:]
    q = te[emb_ok[te]]
    stride = max(1, len(q) // MAX_Q)
    q = q[::stride]
    f = np.full(n, np.nan)
    if len(lib) < 100 or len(q) == 0:
        return f, q
    tree = cKDTree(emb(lib))
    dist, nb = tree.query(emb(q), k=E + 1)
    d1 = np.maximum(dist[:, :1], 1e-12)
    w = np.exp(-dist / d1)
    zhat = (w * y_z[lib][nb]).sum(1) / w.sum(1)
    f[q] = zhat * sd[q]
    return f, q


# ----------------------------------------------------------------------------------------------- evaluation
def stats_oos(f, y, pos_eval):
    ok = pos_eval[np.isfinite(f[pos_eval]) & np.isfinite(y[pos_eval])]
    if len(ok) < 50:
        return {"n": int(len(ok)), "ic": np.nan, "t": np.nan, "hit": np.nan, "edge_bps": np.nan}
    a, b = f[ok], y[ok]
    ic = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    nz = (a != 0) & (b != 0)
    hit = float((np.sign(a[nz]) == np.sign(b[nz])).mean()) if nz.any() else np.nan
    edge = float((np.sign(a) * b).mean() * 1e4)
    return {"n": int(len(ok)), "ic": ic, "t": ic * math.sqrt(len(ok)), "hit": hit, "edge_bps": edge}


def evaluate_projection(name, lp, r, y, index, folds, sd, z, y_z):
    """Walk-forward forecast series and info gain for one projection on one (real or surrogate) series."""
    n = len(lp)
    f = np.full(n, np.nan)
    info_num, info_den = 0.0, 0
    eval_pos = []
    if name in KNN_PROJ:
        E = int(name.split("_")[1])
        for fo in folds:
            tr = fo.train[:-1]
            ff, q = simplex_forecast(z, sd, y_z, E, tr, fo.test)
            f[q] = ff[q]
            eval_pos.append(q)
        return f, np.concatenate(eval_pos) if eval_pos else np.zeros(0, int), np.nan
    st = states(name, lp, r, index)
    for fo in folds:
        tr = fo.train[:-1]
        m, p, p0 = fit_bins(st, y, tr)
        te = fo.test
        s = st[te]
        good = s >= 0
        good &= s < len(m)
        f[te[good]] = m[s[good]]
        yy = y[te[good]]
        pp = p[s[good]]
        nzm = yy != 0
        if nzm.any():
            upm = yy[nzm] > 0
            ll = np.where(upm, np.log2(pp[nzm]), np.log2(1 - pp[nzm]))
            l0 = np.where(upm, np.log2(p0), np.log2(1 - p0))
            info_num += float((ll - l0).sum())
            info_den += int(nzm.sum())
        eval_pos.append(te[good])
    return f, np.concatenate(eval_pos) if eval_pos else np.zeros(0, int), (info_num / info_den if info_den else np.nan)


def run(pair, tf):
    from fxlab.validation.splits import time_walk_forward
    from fxlab.execution.simulator import vector_backtest
    from fxlab.execution.costs import BASE, ZERO
    from fxlab.data.instruments import get as instrument
    from fxlab.validation.metrics import daily
    t0 = time.time()
    bars = load(pair, tf)
    index = bars.index
    mid = ((bars["bid_c"] + bars["ask_c"]) / 2).to_numpy(float)
    lp = np.log(mid)
    r = np.r_[np.nan, np.diff(lp)]
    y = np.r_[r[1:], np.nan]                          # next-bar return (target)
    sd = ewm_abs(r, 100)
    folds = time_walk_forward(index, 3, 6)
    pip = instrument(pair).pip
    spr = (bars["ask_c"] - bars["bid_c"]).to_numpy(float)
    c_rt = 2 * ((BASE.spread_mult / 2 + BASE.slip_spread_frac) * spr + BASE.slip_fixed_pips * pip) / mid \
        + 2 * BASE.commission_bps_per_side * 1e-4
    thr = float(np.nanmedian(c_rt[folds[0].train])) if folds else np.nan
    rng = np.random.default_rng(12345)
    sur = []
    for k in range(N_SUR):
        sgn = rng.choice([-1.0, 1.0], size=len(r))
        rs = np.abs(np.nan_to_num(r)) * sgn
        rs[0] = 0.0
        lps = lp[0] + np.cumsum(rs)
        sur.append((lps, np.r_[np.nan, rs[1:]]))
    rows = []
    for name in BIN_PROJ + KNN_PROJ:
        def zfun(rr):
            s_ = ewm_abs(rr, 100)
            return rr / s_, s_
        z, _ = zfun(r)
        y_z = np.r_[z[1:], np.nan]
        f, pos, info = evaluate_projection(name, lp, r, y, index, folds, sd, z, y_z)
        st = stats_oos(f, y, pos)
        # surrogate controls (statistics only)
        sur_ic, sur_info, sur_edge = [], [], []
        for lps, rs in sur:
            ys = np.r_[rs[1:], np.nan]
            zs, sds = zfun(rs)
            fs, ps, infs = evaluate_projection(name, lps, rs, ys, index, folds, sds, zs, np.r_[zs[1:], np.nan])
            ss = stats_oos(fs, ys, ps)
            sur_ic.append(ss["ic"]); sur_info.append(infs); sur_edge.append(ss["edge_bps"])
        si = np.array(sur_ic, float)
        row = {"pair": pair, "tf": tf, "projection": name, **st, "info_bits": info,
               "sur_ic_mean": float(np.nanmean(si)), "sur_ic_sd": float(np.nanstd(si)),
               "z_sur": float((st["ic"] - np.nanmean(si)) / (np.nanstd(si) + 1e-9)) if np.isfinite(st["ic"]) else np.nan,
               "sur_info_mean": float(np.nanmean(sur_info)) if name in BIN_PROJ else np.nan,
               "sur_edge_bps_mean": float(np.nanmean(sur_edge)), "cost_rt_bps": thr * 1e4}
        # economics: cost-aware rule, platform engine (positions only where a forecast exists)
        posn = np.where(np.isfinite(f), np.where(np.abs(f) > thr, np.sign(f), 0.0), 0.0)
        mask = np.zeros(len(f), bool)
        if len(pos):
            mask[pos] = True
        if name in KNN_PROJ:                           # strided queries: hold each forecast until the next one
            pser = pd.Series(np.where(mask, posn, np.nan), index=index).ffill(limit=max(1, len(index) // max(1, len(pos))))
            posn = pser.fillna(0.0).to_numpy()
        ps = pd.Series(posn, index=index)
        oos = np.zeros(len(f), bool)
        for fo in folds:
            oos[fo.test] = True
        out = {}
        for nm, cm in (("net", BASE), ("gross", ZERO)):
            bt = vector_backtest(bars, ps, pip, cm)
            rr = bt["net"][oos]
            dd = daily(rr)
            out[nm] = float(dd.mean() / dd.std() * math.sqrt(260)) if len(dd) > 20 and dd.std() > 0 else 0.0
            if nm == "net":
                row["trades_per_year"] = float((np.abs(np.diff(np.r_[0, posn[oos]])) > 0).sum()
                                               / max(1e-9, (index[oos][-1] - index[oos][0]).days / 365.25))
                row["net_ann"] = float(dd.mean() * 260)
        row["gross_sharpe"], row["net_sharpe"] = out["gross"], out["net"]
        rows.append(row)
        print(f"{pair} {tf} {name:10s} ic={st['ic']:+.4f} t={st['t']:+.1f} hit={st['hit']:.4f} "
              f"info={info if info == info else float('nan'):+.5f} z_sur={row['z_sur']:+.1f} edge={st['edge_bps']:+.3f}bp "
              f"cost={thr * 1e4:.2f}bp gross={out['gross']:+.2f} net={out['net']:+.2f}", flush=True)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "projection_lab"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, sec = run(a.pair, a.tf)
    json.dump({"pair": a.pair, "tf": a.tf, "seconds": sec, "rows": rows},
              open(os.path.join(a.out, f"{a.pair}_{a.tf}{'_w2' if os.environ.get('PL_SET') == 'wave2' else ''}.json"), "w"),
              indent=1, default=float)
    print(f"done {a.pair} {a.tf} in {sec / 60:.1f} min")


if __name__ == "__main__":
    main()
