"""Data-quality audit of the M1 store (per instrument-year) and the verified
data inventory (actual first/last observation per instrument and timeframe)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl

from fxlab.config import M1_DIR, RESULTS_DIR
from fxlab.data.instruments import INSTRUMENTS


PLAUSIBLE = {"JPY": (60.0, 260.0), "XAUUSD": (200.0, 6000.0), "XAGUSD": (3.0, 80.0), "USDTRY": (1.0, 60.0)}


def plausible_range(sym: str) -> tuple[float, float]:
    if sym in PLAUSIBLE:
        return PLAUSIBLE[sym]
    if sym.endswith("JPY"):
        return PLAUSIBLE["JPY"]
    return (0.3, 25.0)  # every other FX pair in scope (incl. SEK/NOK quotes) trades between these levels


def audit_file(path) -> dict:
    df = pl.read_parquet(path)
    ts = df["ts"]
    mid = (df["bid_c"] + df["ask_c"]) / 2
    lr = (mid.log().diff()).to_numpy()
    sym = path.stem.split("_")[0]
    pip = INSTRUMENTS[sym].pip if sym in INSTRUMENTS else 1e-4
    spread_pips = (df["spread_mean"] / pip).to_numpy()
    t = ts.to_numpy().astype("datetime64[ms]").astype(np.int64)
    gaps_min = np.diff(t) / 60000.0
    wd = pd.DatetimeIndex(ts.to_numpy()).dayofweek
    # trading-hours gaps: gaps > 5 min that do not span a weekend (Fri 21:00 - Sun 21:00 UTC)
    starts = pd.DatetimeIndex(ts.to_numpy()[:-1])
    weekendish = (starts.dayofweek == 4) & (starts.hour >= 20) | (starts.dayofweek >= 5) | (gaps_min > 36 * 60)
    tg = gaps_min[(gaps_min > 5) & ~weekendish]
    sd = pd.Series(lr).rolling(1440, min_periods=200).std().shift(1).to_numpy()
    spikes = int(np.nansum(np.abs(lr) > 20 * sd))
    ohlc_bad = int(((df["bid_h"] < df["bid_l"]) | (df["ask_h"] < df["ask_l"]) | (df["ask_c"] < df["bid_c"])).sum())
    lo, hi = plausible_range(sym)
    med = float(mid.median())
    return {"symbol": sym, "year": int(path.stem.split("_")[1]), "minutes": df.height,
            "median_price": med, "price_level_plausible": bool(lo <= med <= hi),
            "first": str(ts.min()), "last": str(ts.max()),
            "weekend_minutes_sat": int((wd == 5).sum()),
            "gaps_gt5min_trading_hours": int(len(tg)), "max_gap_trading_hours_min": float(tg.max()) if len(tg) else 0.0,
            "spread_pips_p50": float(np.nanmedian(spread_pips)), "spread_pips_p99": float(np.nanpercentile(spread_pips, 99)),
            "spread_pips_max": float(np.nanmax(spread_pips)), "zero_spread_minutes": int((df["spread_min"] <= 0).sum()),
            "crossed_or_inconsistent_bars": ohlc_bad, "ticks_per_min_p50": float(df["n_ticks"].median()),
            "stale_minutes_share": float(((df["up_ticks"] + df["down_ticks"]) == 0).mean()),
            "spikes_gt20sd": spikes}


def run_audit() -> pd.DataFrame:
    rows = [audit_file(p) for p in sorted(M1_DIR.glob("*/*.parquet"))]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "data_quality_audit.csv", index=False)
    return out


def inventory(audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Actual first/last observation per instrument (M1). Higher timeframes are built from M1 and
    therefore share the same span (first/last bar = floor of these timestamps)."""
    a = audit if audit is not None else run_audit()
    g = a.groupby("symbol").agg(first_obs=("first", "min"), last_obs=("last", "max"), years=("year", "count"),
                                year_list=("year", lambda s: ",".join(map(str, sorted(s)))),
                                m1_bars=("minutes", "sum"))
    g["tradable"] = True
    g["timeframes"] = "tick-derived M1 -> M2,M3,M5,M10,M15,M30,H1,H2,H4,H8,D1 (+tick/volume/volatility/range bars)"
    g.to_csv(RESULTS_DIR / "data_inventory.csv")
    return g
