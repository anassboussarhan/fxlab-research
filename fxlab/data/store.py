"""Point-in-time data access with bid/ask semantics and an enforced 2026 holdout.

Bar convention (all timeframes): a bar is labelled by its OPEN time (UTC) and
contains M1 observations in [open, open + period). A bar's values are only
known at its close; strategies therefore act at the NEXT bar's open at the
earliest (enforced in fxlab.execution.simulator).

Higher timeframes are built only from validated M1 bars. Periods with no M1
observations produce no bar (nothing is interpolated). D1 bars use the FX
market convention of a 17:00 America/New_York rollover (DST-aware).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from fxlab.config import (FROZEN_CANDIDATES_FILE, HOLDOUT_START, HOLDOUT_UNLOCK_ENV, M1_DIR)

TIMEFRAMES = {"M1": "1min", "M2": "2min", "M3": "3min", "M5": "5min", "M10": "10min",
              "M15": "15min", "M30": "30min", "H1": "1h", "H2": "2h", "H4": "4h", "H6": "6h", "H8": "8h", "H12": "12h",
              "D1": "D1-NY17"}
TF_MINUTES = {"M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30, "H1": 60,
              "H2": 120, "H4": 240, "H6": 360, "H8": 480, "H12": 720, "D1": 1440}


class HoldoutViolation(RuntimeError):
    pass


def holdout_unlocked() -> bool:
    """The holdout opens only if candidates are frozen AND the unlock token matches
    the SHA-256 recorded in the frozen-candidates file (explicit authorization)."""
    token = os.environ.get(HOLDOUT_UNLOCK_ENV)
    if not token or not FROZEN_CANDIDATES_FILE.exists():
        return False
    import hashlib
    frozen = json.loads(FROZEN_CANDIDATES_FILE.read_text())
    return bool(frozen.get("frozen")) and hashlib.sha256(token.encode()).hexdigest() == frozen.get("unlock_token_sha256")


def available(symbol: str) -> list[int]:
    d = M1_DIR / symbol
    if not d.exists():
        return []
    return sorted(int(p.stem.split("_")[-1]) for p in d.glob(f"{symbol}_*.parquet"))


def load_m1(symbol: str, start: str | pd.Timestamp | None = None,
            end: str | pd.Timestamp | None = None, allow_holdout: bool = False,
            columns: list[str] | None = None) -> pd.DataFrame:
    """Load validated M1 bid/ask bars. Never returns holdout rows unless unlocked. `columns` restricts the
    columns read (no mid columns are added then), e.g. the bid/ask highs and lows for intrabar path replay."""
    years = available(symbol)
    if not years:
        raise FileNotFoundError(f"no M1 data for {symbol} in {M1_DIR}")
    unlocked = allow_holdout and holdout_unlocked()
    hs = pd.Timestamp(HOLDOUT_START)
    s = pd.Timestamp(start, tz="UTC") if start is not None and pd.Timestamp(start).tzinfo is None else (pd.Timestamp(start) if start is not None else None)
    e = pd.Timestamp(end, tz="UTC") if end is not None and pd.Timestamp(end).tzinfo is None else (pd.Timestamp(end) if end is not None else None)
    if e is not None and e >= hs and not unlocked:
        raise HoldoutViolation(f"requested end {e} reaches the locked holdout (>= {hs})")
    files = []
    for y in years:
        if y >= hs.year and not unlocked:
            continue
        if s is not None and y < s.year:
            continue
        if e is not None and y > e.year:
            continue
        files.append(str(M1_DIR / symbol / f"{symbol}_{y}.parquet"))
    if not files:
        raise FileNotFoundError(f"no M1 files for {symbol} in requested range")
    lf = pl.scan_parquet(files)
    if s is not None:
        lf = lf.filter(pl.col("ts") >= s.to_pydatetime())
    if e is not None:
        lf = lf.filter(pl.col("ts") <= e.to_pydatetime())
    if not unlocked:
        lf = lf.filter(pl.col("ts") < hs.to_pydatetime())
    if columns:
        lf = lf.select(["ts"] + list(columns))
    df = lf.sort("ts").collect().to_pandas()
    df = df.set_index("ts")
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    if not unlocked and len(df) and df.index.max() >= hs:
        raise HoldoutViolation("holdout rows leaked through loader")  # defensive
    return df if columns else add_mid(df)


M1_PATH_COLUMNS = ["bid_h", "bid_l", "ask_h", "ask_l"]


def add_mid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k in ("o", "h", "l", "c"):
        df[{"o": "open", "h": "high", "l": "low", "c": "close"}[k]] = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2.0
    # mid high/low as average of bid and ask extremes (bid/ask extremes may occur at different ticks)
    df["spread"] = df["spread_mean"]
    df["volume"] = df["n_ticks"].astype(float)  # tick count = only honest "volume" in OTC FX
    return df


def _agg_spec() -> dict:
    return {"bid_o": "first", "bid_h": "max", "bid_l": "min", "bid_c": "last",
            "ask_o": "first", "ask_h": "max", "ask_l": "min", "ask_c": "last",
            "n_ticks": "sum", "spread_max": "max", "spread_min": "min",
            "bid_liq": "sum", "ask_liq": "sum", "up_ticks": "sum", "down_ticks": "sum"}


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate M1 bid/ask bars to timeframe ``tf`` (see TIMEFRAMES)."""
    if tf == "M1":
        out = m1.copy()
        out["rv_intrabar"] = np.nan  # undefined for single M1 bar
        out["n_m1"] = 1
        return out
    base_cols = list(_agg_spec().keys())
    x = m1[base_cols + ["spread_mean"]].copy()
    x["_sw"] = m1["spread_mean"] * m1["n_ticks"]
    mid_c = (m1["bid_c"] + m1["ask_c"]) / 2
    lr = np.log(mid_c).diff()
    # intrabar realised variance: only M1 returns whose previous minute is in the same bar
    if tf == "D1":
        ny = x.index.tz_convert("America/New_York").tz_localize(None)  # naive NY wall clock
        open_local = (ny + pd.Timedelta(hours=7)).floor("D") - pd.Timedelta(hours=7)  # 17:00 NY rollover
        key = open_local.tz_localize("America/New_York").tz_convert("UTC")  # bar open in UTC
        grp = pd.Series(key, index=x.index)
    else:
        grp = pd.Series(x.index.floor(TIMEFRAMES[tf]), index=x.index)
    same = grp.eq(grp.shift(1))
    x["_r2"] = np.where(same, lr.values ** 2, 0.0)
    x["_n"] = 1
    spec = _agg_spec() | {"_sw": "sum", "_r2": "sum", "_n": "sum"}
    out = x.groupby(grp.values, sort=True).agg(spec)
    out.index = pd.DatetimeIndex(out.index, name="ts")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out["spread_mean"] = out["_sw"] / out["n_ticks"].clip(lower=1)
    out["rv_intrabar"] = out["_r2"]
    out["n_m1"] = out["_n"]
    out = out.drop(columns=["_sw", "_r2", "_n"])
    return add_mid(out)


EVENT_BAR_PREFIXES = ("TICK", "VOLA", "LIQ")


def is_event_tf(tf: str) -> bool:
    return tf.startswith(EVENT_BAR_PREFIXES)


def load_bars(symbol: str, tf: str, start=None, end=None) -> pd.DataFrame:
    m1 = load_m1(symbol, start, end)
    if is_event_tf(tf):
        return event_bars(m1, tf)
    return resample(m1, tf)


def adaptive_threshold(values: pd.Series, minutes: int, lookback_days: int = 20) -> np.ndarray:
    """Causal bar threshold: the average amount of `values` per `minutes` of active market over the PREVIOUS
    `lookback_days` (M1 rows strictly before the current one). Keeps the average bar duration near `minutes`
    even though tick counts / quoted liquidity grew several-fold between 2007 and 2025."""
    per_row = values.rolling(f"{lookback_days}D", closed="left", min_periods=500).mean()
    thr = (per_row * minutes).to_numpy()
    first = np.nanmean(values.iloc[:5000].to_numpy()) * minutes      # warm-up: first days use an early average
    return np.where(np.isfinite(thr), thr, first)


def _adaptive_breaks(v: np.ndarray, thr: np.ndarray) -> np.ndarray:
    from numba import njit

    @njit(cache=False)
    def run(v, thr):
        br = np.zeros(len(v), np.bool_)
        acc = 0.0
        for i in range(len(v)):
            acc += v[i]
            if acc >= thr[i]:
                br[i] = True
                acc = 0.0
        return br
    return run(v, thr)


def event_bars(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Information-driven bars with an adaptive (causal) threshold. tf = TICK<min> | VOLA<min> | LIQ<min>:
    a bar closes after the tick count / absolute mid move / quoted liquidity that the market produced, on
    average, in <min> minutes over the previous 20 days."""
    for p in EVENT_BAR_PREFIXES:
        if tf.startswith(p):
            kind, minutes = p, int(tf[len(p):])
    if kind == "TICK":
        v = m1["n_ticks"].astype(float)
    elif kind == "LIQ":
        v = (m1["bid_liq"] + m1["ask_liq"]).astype(float)
    else:
        mid = (m1["bid_c"] + m1["ask_c"]) / 2
        v = np.log(mid).diff().abs().fillna(0.0)
    thr = adaptive_threshold(v, minutes)
    br = _adaptive_breaks(v.to_numpy(np.float64), thr.astype(np.float64))
    return _bars_from_breaks(m1, br)


def load_panel(symbols: list[str], tf: str, start=None, end=None, field: str = "close",
               how: str = "inner") -> pd.DataFrame:
    """Aligned panel of one field across symbols (point-in-time; no forward fill
    across missing bars beyond the last *observed* value)."""
    cols = {}
    for s in symbols:
        try:
            cols[s] = load_bars(s, tf, start, end)[field]
        except FileNotFoundError:
            continue
    return pd.DataFrame(cols).dropna(how="all" if how == "outer" else "any")


# ---------------------------------------------------------------------------
# Alternative bar types (information-driven). Built causally from M1 bars.
# ---------------------------------------------------------------------------

def _bars_from_breaks(m1: pd.DataFrame, breaks: np.ndarray) -> pd.DataFrame:
    """breaks[i] = True if M1 row i closes a bar."""
    gid = np.concatenate([[0], np.cumsum(breaks[:-1])])
    x = m1[list(_agg_spec().keys())].copy()
    x["_sw"] = m1["spread_mean"] * m1["n_ticks"]
    x["_end"] = m1.index
    spec = _agg_spec() | {"_sw": "sum", "_end": "last"}
    g = x.groupby(gid)
    out = g.agg(spec)
    starts = np.r_[0, np.flatnonzero(np.asarray(breaks[:-1])) + 1]      # first M1 row of every bar
    out.index = pd.DatetimeIndex(m1.index[starts], name="ts")
    out["spread_mean"] = out["_sw"] / out["n_ticks"].clip(lower=1)
    out = out.rename(columns={"_end": "bar_end"}).drop(columns=["_sw"])
    out["rv_intrabar"] = np.nan
    out["n_m1"] = g.size().values
    return add_mid(out)


def tick_bars(m1: pd.DataFrame, ticks_per_bar: int) -> pd.DataFrame:
    c = m1["n_ticks"].to_numpy().astype(np.int64)
    acc, br = 0, np.zeros(len(c), bool)
    for i, v in enumerate(c):
        acc += v
        if acc >= ticks_per_bar:
            br[i], acc = True, 0
    return _bars_from_breaks(m1, br)


def volume_bars(m1: pd.DataFrame, liquidity_per_bar: float) -> pd.DataFrame:
    v = (m1["bid_liq"] + m1["ask_liq"]).to_numpy().astype(float)
    acc, br = 0.0, np.zeros(len(v), bool)
    for i, x in enumerate(v):
        acc += x
        if acc >= liquidity_per_bar:
            br[i], acc = True, 0.0
    return _bars_from_breaks(m1, br)


def volatility_bars(m1: pd.DataFrame, abs_move: float) -> pd.DataFrame:
    """Close a bar once cumulative |Δ log mid| since bar start exceeds abs_move."""
    mid = ((m1["bid_c"] + m1["ask_c"]) / 2).to_numpy()
    lr = np.abs(np.diff(np.log(mid), prepend=np.log(mid[0])))
    acc, br = 0.0, np.zeros(len(lr), bool)
    for i, x in enumerate(lr):
        acc += x
        if acc >= abs_move:
            br[i], acc = True, 0.0
    return _bars_from_breaks(m1, br)


def range_bars(m1: pd.DataFrame, price_range: float) -> pd.DataFrame:
    """Event bars: close when high-low of the forming bar exceeds price_range."""
    hi = ((m1["bid_h"] + m1["ask_h"]) / 2).to_numpy()
    lo = ((m1["bid_l"] + m1["ask_l"]) / 2).to_numpy()
    br = np.zeros(len(hi), bool)
    h, l = -np.inf, np.inf
    for i in range(len(hi)):
        h, l = max(h, hi[i]), min(l, lo[i])
        if h - l >= price_range:
            br[i] = True
            h, l = -np.inf, np.inf
    return _bars_from_breaks(m1, br)
