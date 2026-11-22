"""Round 25 execution space and signal path (docs/round25_preregistration.md, part P).

N_EXEC Sobol dimensions per configuration, after the rule's own parameters:
- round 23: direction, mode, order type and its k / life, stop (kind, multiple, structural buffer), target, trailing,
  breakeven, partial exit, time stop;
- round 25:
  - confirmation and session window, combinable with the order type;
  - day-of-week, regime and higher-timeframe filters;
  - ATR length, realised-vol stop length and horizon, target intercept;
  - neighbourhood-strength and volatility sizing;
  - price source, moving-average type and volatility measure of the rule's own calls.

signal_path() turns a rule (or a placebo side) into the executed target:
  categorical swaps -> rule side x direction -> confirmation -> gate (session, day, regime, higher timeframe)
  -> size at entry (neighbourhood strength x volatility).
evaluate25() runs it through fxlab.evaluate. Everything is causal; scripts/r25_lookahead.py tests that.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from numba import njit

from fxlab.features import _util as U_

EX = ["dir", "mode", "otype", "ok", "ottl", "stop", "smult", "sbuf", "tgt", "tgt_b", "trail", "trail_k", "be", "be_r",
      "part", "part_r", "part_f", "tstop", "tstop_n",
      "conf", "conf_k", "sess", "sess_tz", "sess_start", "sess_len", "day", "day_mask",
      "reg", "reg_type", "reg_n", "reg_w", "reg_lo", "reg_width",
      "htf", "htf_tf", "htf_type", "htf_n", "htf_mode",
      "atr_n", "rv_n", "rv_h", "tgt_a",
      "size", "size_k", "size_w", "vt", "vt_n", "vt_c",
      "psrc", "ma", "vol", "vol_k"]
N_EXEC = len(EX)
IX = {k: i for i, k in enumerate(EX)}
TF_ORDER = ["H1", "H2", "H4", "H8", "H12", "D1", "D2", "W1"]
TZS = ["Asia/Tokyo", "Europe/London", "America/New_York"]
REGIMES = ["adx", "atr_pct", "trend", "spread"]
HTF_TYPES = ["slope", "level", "donchian"]
PSRC = ["close", "hl2", "hlc3", "ohlc4"]
MAS = ["native", "sma", "ema", "wma", "hma"]


def _pick(u, xs):
    return xs[min(int(u * len(xs)), len(xs) - 1)]


def _logr(u, lo, hi):
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def _ilog(u, lo, hi):
    return int(round(_logr(u, lo, hi)))


def day_masks(n_days):
    """Every non-empty proper subset of the n_days weekdays (Mon = 0), as tuples."""
    out = []
    for r in range(1, n_days):
        out += list(itertools.combinations(range(n_days), r))
    return out


MASKS5, MASKS7 = day_masks(5), day_masks(7)


def config(u, tf, crypto=False):
    """Map N_EXEC unit-interval draws to a configuration dict."""
    g = lambda k: float(u[IX[k]])                                          # noqa: E731
    on = lambda k: g(k) >= 0.5                                             # noqa: E731
    c = {"dir": -1 if g("dir") < 0.5 else 1, "mode": "native" if g("mode") < 0.5 else "custom"}
    c["otype"] = _pick(g("otype"), ["market", "limit_pullback", "stop_breakout"])
    c["ok"], c["ottl"] = 0.05 + g("ok") * 1.45, int(1 + g("ottl") * 23.999)
    sk = _pick(g("stop"), ["atr", "realized_vol", "structural", "none"])
    c["atr_n"] = _ilog(g("atr_n"), 5, 50)
    if sk == "atr":
        stop = ("atr", {"mult": 0.5 + g("smult") * 5.5})
    elif sk == "realized_vol":
        stop = ("realized_vol", {"mult": 0.5 + g("smult") * 5.5, "n": _ilog(g("rv_n"), 10, 200),
                                 "horizon": _ilog(g("rv_h"), 2, 40)})
    elif sk == "structural":
        stop = ("structural", {"k": int(2 + g("smult") * 8.999), "buffer_atr": g("sbuf")})
    else:
        stop = ("atr", {"mult": 100.0})
    c["risk"] = {"stop": stop[0], "stop_params": stop[1], "sizing": "none", "atr_n": c["atr_n"],
                 "target": "affine" if on("tgt") else "none",
                 "target_params": {"a": g("tgt_a") * 2.0, "b": 0.5 + g("tgt_b") * 5.5} if on("tgt") else {},
                 "trail_atr": (0.5 + g("trail_k") * 5.5) if on("trail") else None,
                 "breakeven_r": (0.3 + g("be_r") * 2.7) if on("be") else None,
                 "partial_r": (0.5 + g("part_r") * 3.5) if on("part") else None,
                 "partial_frac": 0.2 + g("part_f") * 0.6,
                 "max_hold": _ilog(g("tstop_n"), 2, 500) if on("tstop") else None}
    c["conf"] = int(1 + g("conf_k") * 4.999) if on("conf") else 0
    c["sess"] = (_pick(g("sess_tz"), TZS), int(g("sess_start") * 23.999), 2 + int(g("sess_len") * 20.999)) \
        if on("sess") else None
    masks = MASKS7 if crypto else MASKS5
    c["day"] = _pick(g("day_mask"), masks) if on("day") else None
    c["reg"] = (_pick(g("reg_type"), REGIMES), _ilog(g("reg_n"), 5, 200), _logr(g("reg_w"), 0.25, 5.0),
                g("reg_lo") * 0.8, min(1.0, g("reg_lo") * 0.8 + 0.2 + g("reg_width") * 0.8)) if on("reg") else None
    hi = [t for t in ("D1", "W1") if TF_ORDER.index(t) > TF_ORDER.index(tf)]
    c["htf"] = (_pick(g("htf_tf"), hi), _pick(g("htf_type"), HTF_TYPES), _ilog(g("htf_n"), 3, 200),
                "with" if g("htf_mode") < 0.5 else "against") if (on("htf") and hi) else None
    c["size"] = (2 + int(g("size_k") * 2.999), 0.02 + g("size_w") * 0.28) if on("size") else None
    c["vt"] = (_ilog(g("vt_n"), 20, 500), 0.3 + g("vt_c") * 0.7) if on("vt") else None
    c["psrc"], c["ma"] = _pick(g("psrc"), PSRC), _pick(g("ma"), MAS)
    c["vol"] = ("std", 0.8 + g("vol_k") * 1.7) if g("vol") >= 0.5 else ("native", None)
    return c


# ------------------------------------------------------------------------------------------- categorical swaps
def hma(x: pd.Series, n: int) -> pd.Series:
    n = max(2, int(n))
    return U_.wma(2.0 * U_.wma(x, max(1, n // 2)) - U_.wma(x, n), max(1, int(round(math.sqrt(n)))))


MA_FN = {"sma": U_.sma, "ema": U_.ema, "wma": U_.wma, "hma": hma}


def _std_vol(k):
    def vol(df, n=14):
        return k * df["close"].diff().rolling(int(n), min_periods=int(n)).std()
    return vol


class Swap:
    """Replace sma / ema / atr in the globals of the rule function and of its module (an exposed function runs in a
    copy of the module's globals, its helpers in the module's own) for one call."""

    def __init__(self, fns, cfg):
        self.gs = []
        for f in fns:
            g = getattr(f, "__globals__", None)
            if g is not None and all(g is not x for x in self.gs):
                self.gs.append(g)
        self.new = {}
        if cfg["ma"] != "native":
            self.new["sma"] = self.new["ema"] = MA_FN[cfg["ma"]]
        if cfg["vol"][0] == "std":
            self.new["atr"] = _std_vol(cfg["vol"][1])
        self.old = []

    def __enter__(self):
        for g in self.gs:
            keep = {k: g[k] for k in self.new if k in g}
            self.old.append((g, keep))
            for k in keep:
                g[k] = self.new[k]
        return self

    def __exit__(self, *a):
        for g, keep in self.old:
            g.update(keep)
        return False


def price_source(bars, src):
    if src == "close":
        return bars
    b = bars.copy()
    if src == "hl2":
        b["close"] = (bars["high"] + bars["low"]) / 2.0
    elif src == "hlc3":
        b["close"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    else:
        b["close"] = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4.0
    return b


# ------------------------------------------------------------------------------------------------ transforms
@njit(cache=True)
def _gate(side, al, ash):
    """Entries / additions only where allowed for that direction; exits and reductions always."""
    n = len(side)
    out = np.zeros(n)
    o = 0.0
    for t in range(n):
        s = side[t]
        if s == o:
            pass
        elif s == 0.0 or (o != 0.0 and s * o > 0 and abs(s) < abs(o)):
            o = s
        elif (s > 0 and al[t]) or (s < 0 and ash[t]):
            o = s
        elif o != 0.0 and s * o < 0:
            o = 0.0
        out[t] = o
    return out


@njit(cache=True)
def _confirm_trig(trig, close, k):
    """Entry triggers: a trigger is taken once a later bar closes beyond the trigger bar's close, within k bars."""
    n = len(trig)
    out = np.zeros(n)
    pend = 0.0; ref = 0.0; cnt = 0
    for t in range(n):
        s = trig[t]
        if pend != 0.0 and s == 0.0:
            if (close[t] - ref) * pend > 0:
                out[t] = pend
                pend = 0.0
            else:
                cnt -= 1
                if cnt <= 0:
                    pend = 0.0
        if s != 0.0:
            pend = s; ref = close[t]; cnt = k
    return out


@njit(cache=True)
def _latch(side, mag):
    """Magnitude fixed at entry: when the sign turns on or flips, the current magnitude is kept until it changes."""
    n = len(side)
    out = np.zeros(n)
    o = 0.0; m = 1.0
    for t in range(n):
        s = np.sign(side[t])
        if s != 0.0 and s != o:
            m = mag[t] if np.isfinite(mag[t]) else 1.0
        o = s
        out[t] = s * m
    return out


def _next_start(idx):
    step = pd.Series(idx[: min(len(idx), 200)]).diff().dropna().median() if len(idx) > 2 else pd.Timedelta(hours=1)
    return idx + step


def allow_session(idx, tz, start, length):
    nx = _next_start(idx).tz_convert(tz)
    h = nx.hour.to_numpy() + nx.minute.to_numpy() / 60.0
    return ((h - start) % 24.0) < length


def allow_days(idx, mask):
    nx = (_next_start(idx).tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7))
    return np.isin(nx.dayofweek.to_numpy(), np.asarray(mask))


def _bars_per_year(idx):
    """Bars in the instrument's first year (known once that year is over; causal for every later bar). With less
    than a year of data, the count so far scaled to a year."""
    if len(idx) < 2:
        return 1.0
    end = idx[0] + pd.Timedelta(days=365.25)
    k = int(pd.DatetimeIndex(idx).searchsorted(end, side="left"))       # (index units may be us or ns)
    if k < len(idx):
        return float(k)
    span = (idx[-1] - idx[0]).total_seconds() / (365.25 * 86400)
    return len(idx) / max(span, 1e-9)


def _adx(b, n):
    h, l, c = b["high"], b["low"], b["close"]
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = U_.true_range(b)
    atr_ = U_.rma(tr, n)
    pdi, mdi = 100 * U_.rma(pdm, n) / (atr_ + U_.EPS), 100 * U_.rma(mdm, n) / (atr_ + U_.EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + U_.EPS)
    return U_.rma(dx, n)


def regime_value(b, kind, n):
    if kind == "adx":
        return _adx(b, n)
    a = U_.atr(b, n)
    if kind == "atr_pct":
        return a / b["close"]
    if kind == "trend":
        return (b["close"] - b["close"].shift(n)).abs() / (a * math.sqrt(n) + U_.EPS)
    sp = b["spread_mean"] if "spread_mean" in b else (b["ask_c"] - b["bid_c"])
    return sp.rolling(n, min_periods=n).mean() / b["close"]


def allow_regime(b, kind, n, w_years, lo, hi):
    W = max(10, int(round(w_years * _bars_per_year(b.index))))
    x = regime_value(b, kind, n)
    rk = x.rolling(W, min_periods=W).rank(pct=True)
    return ((rk >= lo) & (rk <= hi)).to_numpy()


def htf_groups(idx, htf):
    """Group index of each bar: trading day (D1) or the week of the trading day (W1)."""
    ny = idx.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    d = ny.floor("D")
    if htf == "W1":
        iso = d.isocalendar()
        key = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    else:
        key = d.year.to_numpy() * 10000 + d.month.to_numpy() * 100 + d.day.to_numpy()
    _, g = np.unique(key, return_inverse=True)                             # keys increase with time
    return g


def htf_state(b, htf, kind, n, G=None):
    """+1 / -1 / 0 state of the higher timeframe as of the previous completed group (day or week)."""
    g = htf_groups(b.index, htf) if G is None else G
    df = pd.DataFrame({"g": g, "c": b["close"].to_numpy(), "h": b["high"].to_numpy(), "l": b["low"].to_numpy()})
    agg = df.groupby("g").agg(c=("c", "last"), h=("h", "max"), l=("l", "min"))
    C, H, L = agg["c"], agg["h"], agg["l"]
    if kind == "slope":
        m = C.rolling(n, min_periods=n).mean()
        st = np.sign(m - m.shift(1))
    elif kind == "level":
        m = C.rolling(n, min_periods=n).mean()
        st = np.sign(C - m)
    else:
        up = C > H.shift(1).rolling(n, min_periods=n).max()
        dn = C < L.shift(1).rolling(n, min_periods=n).min()
        st = pd.Series(np.where(up, 1.0, np.where(dn, -1.0, np.nan)), index=C.index).ffill()
    st = st.fillna(0.0).to_numpy()
    prev = np.r_[0.0, st[:-1]]                                             # the previous completed group only
    return prev[g]


def vol_scale(b, n, c):
    lr = np.log(b["close"]).diff()
    sig = lr.rolling(n, min_periods=n).std()
    med = sig.expanding(min_periods=1).median()
    v = (c * med / sig).clip(upper=1.0)
    return v.fillna(c).to_numpy()


def opinion(side, trade_mode):
    s = pd.Series(np.sign(np.asarray(side, float)))
    if trade_mode:
        s = s.replace(0.0, np.nan).ffill().fillna(0.0)
    return s.to_numpy()


def neighbour_strength(base_op, others):
    """max(1/(K+1), (agree - oppose)/(K+1)) where the base configuration counts as agreeing."""
    K = len(others)
    b = np.sign(base_op)
    agree = np.ones(len(b))
    oppose = np.zeros(len(b))
    for o in others:
        so = np.sign(o)
        agree += (so == b) & (b != 0)
        oppose += (so == -b) & (b != 0)
    return np.maximum(1.0 / (K + 1), (agree - oppose) / (K + 1))


def signal_path(cfg, bars, raw_side, trade_mode, neighbour_sides=None, cache=None, strength=None):
    """The executed target: (side, size_mult). raw_side is the rule's side on bars (already x direction).
    side: state rules -> sign x magnitude for the native state engines (vector / order), sign for the trade engine
    with size_mult; trade rules -> entry triggers with size_mult."""
    cache = {} if cache is None else cache
    idx = bars.index
    side = np.sign(np.nan_to_num(np.asarray(raw_side, float)))
    close = bars["close"].to_numpy(float)
    if cfg["conf"]:
        if trade_mode:
            side = _confirm_trig(side, close, int(cfg["conf"]))
        else:
            from fxlab.execution.entries import confirm_entries
            side = confirm_entries(pd.Series(side, index=idx), bars["close"], int(cfg["conf"])).to_numpy()
    n = len(idx)
    al = np.ones(n, bool)
    if cfg["sess"]:
        tz, st, ln = cfg["sess"]
        key = ("sess", tz, st, ln)
        if key not in cache:
            cache[key] = allow_session(idx, tz, st, ln)
        al &= cache[key]
    if cfg["day"]:
        key = ("day", cfg["day"])
        if key not in cache:
            cache[key] = allow_days(idx, cfg["day"])
        al &= cache[key]
    if cfg["reg"]:
        al &= allow_regime(bars, *cfg["reg"])
    ash = al.copy()
    if cfg["htf"]:
        htf, kind, hn, mode = cfg["htf"]
        gk = ("htfg", htf)
        if gk not in cache:
            cache[gk] = htf_groups(idx, htf)
        st = htf_state(bars, htf, kind, hn, cache[gk])
        if mode == "against":
            st = -st
        al &= st > 0
        ash &= st < 0
    if trade_mode:                                                         # triggers: an entry needs its bar allowed
        gated = np.where(((side > 0) & al) | ((side < 0) & ash), side, 0.0)
    else:
        gated = _gate(side, al, ash)
    mag = np.ones(n)
    if cfg["size"] and neighbour_sides:
        mag = mag * neighbour_strength(opinion(raw_side, trade_mode), [opinion(x, trade_mode) for x in neighbour_sides])
    elif cfg["size"] and strength is not None:                             # part CF: the vote share
        mag = mag * np.clip(np.nan_to_num(np.asarray(strength, float), nan=1.0), 0.0, 1.0)
    if cfg["vt"]:
        mag = mag * vol_scale(bars, *cfg["vt"])
    if trade_mode:
        return gated, np.where(gated != 0, mag, 1.0)
    held = _latch(gated, mag)
    return held, np.abs(held) + (held == 0)                                # size_mult = |magnitude| where held


def neighbours(u_params, K, w, seed):
    rng = np.random.default_rng(seed)
    return [np.clip(u_params + w * (2.0 * rng.random(len(u_params)) - 1.0), 0.0, 1.0 - 1e-9) for _ in range(K)]


def rule_side(e, fn, bars_sig, params, cfg):
    """The rule's raw signal frame on bars_sig with the categorical swaps of cfg."""
    with Swap([fn, getattr(e, "fn", None)], cfg):
        return fn(bars_sig.copy(), **params)


def evaluate25(e, fn, bars, sym, params, cfg, sps=None, u_params=None, seed=0, placebo_side=None, financing=None,
               cache=None, return_target=False, strength=None):
    """Per-bar net returns of one round-25 configuration (or the executed target when return_target)."""
    from fxlab.evaluate import evaluate
    trade_mode = bool(e is not None and e.extra.get("mode", "state") == "trade")
    if placebo_side is None:
        bsig = price_source(bars, cfg["psrc"])
        sig = rule_side(e, fn, bsig, params, cfg)
        raw = sig["side"].reindex(bars.index).fillna(0.0).to_numpy(float)
    else:
        sig = pd.DataFrame({"side": placebo_side}, index=bars.index)
        raw = np.asarray(placebo_side, float)
    raw = raw * cfg["dir"]
    nb = None
    if cfg["size"] and sps and u_params is not None and len(u_params) and placebo_side is None:
        from r23_LW import draw
        K, w = cfg["size"]
        nb = []
        for un in neighbours(np.asarray(u_params, float), K, w, seed):
            pn = {s["name"]: draw(s, un[j]) for j, s in enumerate(sps)}
            try:
                sn = rule_side(e, fn, price_source(bars, cfg["psrc"]), pn, cfg)
                nb.append(sn["side"].reindex(bars.index).fillna(0.0).to_numpy(float) * cfg["dir"])
            except Exception:                                              # noqa: BLE001  a failed neighbour abstains
                continue
    side, size_mult = signal_path(cfg, bars, raw, trade_mode, nb, cache, strength)
    if return_target:
        return side, size_mult
    sig = sig.copy()
    ent = None
    if cfg["otype"] != "market":
        ent = {"type": "order", "entry": cfg["otype"], "k": cfg["ok"], "ttl": cfg["ottl"], "atr_n": cfg["atr_n"]}
    sid = e.id if e is not None else "S.TREND.MA_CROSS"
    if cfg["mode"] == "native":
        if trade_mode:
            sig["side"] = side
            sig["size_mult"] = size_mult
            r = evaluate(sid, bars, sym, signals=sig, exec_cfg=ent, financing=financing,
                         risk_override={"atr_n": cfg["atr_n"]})
        else:
            sig["side"] = side                                             # sign x magnitude (state engines)
            r = evaluate(sid, bars, sym, signals=sig[["side"]], exec_cfg=ent, financing=financing,
                         mode_override="state" if e is None else None)
    else:
        s2 = pd.DataFrame({"side": np.sign(side), "size_mult": size_mult}, index=bars.index)
        r = evaluate(sid, bars, sym, signals=s2, mode_override="trade", risk_override=cfg["risk"], exec_cfg=ent,
                     financing=financing)
    return r["returns"]
