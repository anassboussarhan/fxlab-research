"""Point-in-time short-term interest rates (3-month interbank, OECD MEI via FRED) for carry and financing.

Source file: data/rates/ir3m_raw.txt (monthly averages, percent, 2006-01..2025-12; 2026 is not stored).
Point-in-time rule: the average of month m is only complete at the end of month m, so it is used from the
first day of month m+1 onwards (one-month lag). Rates were transcribed from the FRED table pages with a
web-fetch tool and rounded to 2 decimals (documented in docs/data_sources.md); for carry RANKING and
financing at this precision the rounding is immaterial."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from fxlab.config import DATA_DIR

RAW = DATA_DIR / "rates" / "ir3m_raw.txt"
CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
SWAP_MARKUP_PA = 0.0075   # broker financing mark-up per side (fraction p.a.) on top of the rate differential


@lru_cache(maxsize=1)
def monthly() -> pd.DataFrame:
    """Monthly average rates in DECIMAL (0.05 = 5%), index = first day of the month (UTC)."""
    rows = []
    for line in RAW.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        ccy, year = p[0], int(p[1])
        for m, v in enumerate(p[2:14], start=1):
            rows.append((ccy, pd.Timestamp(year=year, month=m, day=1, tz="UTC"), np.nan if v == "NA" else float(v) / 100))
    df = pd.DataFrame(rows, columns=["ccy", "month", "rate"]).pivot(index="month", columns="ccy", values="rate")
    return df.sort_index().ffill()          # an isolated NA month carries the previous month's value


def known_at(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Rates as KNOWN at each timestamp: month m's average becomes available at the start of month m+1."""
    m = monthly()
    avail = m.copy()
    avail.index = avail.index + pd.DateOffset(months=1)
    return avail.reindex(avail.index.union(index)).ffill().reindex(index)


def has_rates(symbol: str) -> bool:
    return len(symbol) == 6 and symbol[:3] in CURRENCIES and symbol[3:] in CURRENCIES


def rate_diff(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """Annualised carry of a LONG position in `symbol` (base rate - quote rate), decimal, point-in-time."""
    if not has_rates(symbol):
        return pd.Series(np.nan, index=index)
    k = known_at(index)
    return (k[symbol[:3]] - k[symbol[3:]]).rename("rate_diff")
