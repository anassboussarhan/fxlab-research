"""Point-in-time short rates for every currency of the FX instruments (round 24 R1), from OECD MEI via DBnomics.

File: data/rates/oecd_ir3m.csv (columns month YYYY-MM, ccy, rate in per cent), written by ci/r24_rates.py; the
3-month interbank rate (IR3TIB01), else call money (IRSTCI01); CNH uses the CNY series. The monthly average of
month m is used from the first day of month m + 1 (as fxlab/data/rates.py). Metals (XAU, XAG) yield nothing:
their rate is 0, so a long metal position pays the quote currency's rate."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fxlab.config import DATA_DIR

FILE = DATA_DIR / "rates" / "oecd_ir3m.csv"
ZERO = ("XAU", "XAG")


@lru_cache(maxsize=1)
def monthly() -> pd.DataFrame:
    """Monthly rates in decimal, index = first day of the month (UTC), one column per currency."""
    if not FILE.exists():
        return pd.DataFrame()
    d = pd.read_csv(FILE)
    d["month"] = pd.to_datetime(d["month"] + "-01", utc=True)
    m = d.pivot(index="month", columns="ccy", values="rate").sort_index() / 100.0
    for z in ZERO:
        m[z] = 0.0
    return m.ffill(limit=3)


def has_rates(symbol: str) -> bool:
    m = monthly()
    return len(symbol) == 6 and symbol[:3] in m.columns and symbol[3:] in m.columns


def known_at(index: pd.DatetimeIndex) -> pd.DataFrame:
    m = monthly().copy()
    m.index = m.index + pd.DateOffset(months=1)                        # month m known from the start of m + 1
    u = pd.DatetimeIndex(index).unique()
    return m.reindex(m.index.union(u)).ffill(limit=None).reindex(u).reindex(index)


def rate_diff(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """Annualised carry of a LONG position in `symbol` (base rate - quote rate), decimal, point-in-time."""
    if not has_rates(symbol):
        return pd.Series(np.nan, index=index)
    k = known_at(index)
    return (k[symbol[:3]] - k[symbol[3:]]).rename("rate_diff")
