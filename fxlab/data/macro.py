"""Point-in-time consumer price indices for the currency value (PPP) factor.

Source: data/macro/cpi_raw.txt (FRED series listed in the file header, 2006-2022). Publication lags are applied
conservatively: a monthly CPI for month m is used only from the start of month m+2; a quarterly CPI (AUD, NZD;
quarter starting q) only from q + 4 months. Between observations the last known value is carried forward."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fxlab.config import DATA_DIR

RAW = DATA_DIR / "macro" / "cpi_raw.txt"


@lru_cache(maxsize=1)
def cpi_available() -> pd.DataFrame:
    """CPI levels indexed by the date from which each value is KNOWN (UTC month starts)."""
    cols = {}
    for line in RAW.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        tag, year, vals = p[0], int(p[1]), p[2:]
        if tag.endswith("Q"):
            ccy, lag, months = tag[:-1], 4, (1, 4, 7, 10)
        else:
            ccy, lag, months = tag, 2, tuple(range(1, 13))
        for m, v in zip(months, vals):
            if v == "NA":
                continue
            t = pd.Timestamp(year=year, month=m, day=1, tz="UTC") + pd.DateOffset(months=lag)
            cols.setdefault(ccy, {})[t] = float(v)
    return pd.DataFrame(cols).sort_index()


def cpi_known_at(index: pd.DatetimeIndex) -> pd.DataFrame:
    a = cpi_available()
    u = pd.DatetimeIndex(index).unique()
    return a.reindex(a.index.union(u)).ffill().reindex(u).reindex(index)


def has_cpi(symbol: str) -> bool:
    a = cpi_available()
    return len(symbol) == 6 and symbol[:3] in a.columns and symbol[3:] in a.columns
