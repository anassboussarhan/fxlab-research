"""Offline astronomical calendars (PyEphem) for the esoteric families. Deterministic, known in advance.

moon_phase(index)     : days since the last new moon at the timestamp (0..29.5)
mercury_retro(index)  : True while Mercury's geocentric ecliptic longitude is decreasing (retrograde)"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _daily_tables():
    import ephem
    days = pd.date_range("2003-01-01", "2027-01-01", freq="D")
    m = ephem.Mercury()
    lon = np.empty(len(days))
    age = np.empty(len(days))
    for i, d in enumerate(days):
        ed = ephem.Date(d.to_pydatetime())
        m.compute(ed)
        lon[i] = float(ephem.Ecliptic(m).lon)
        age[i] = float(ed - ephem.previous_new_moon(ed))
    dl = np.diff(lon, prepend=lon[0])
    dl = (dl + np.pi) % (2 * np.pi) - np.pi             # unwrap the 360-degree crossing
    retro = dl < 0
    return pd.DataFrame({"retro": retro, "moon_age": age}, index=days.tz_localize("UTC"))


def _lookup(index: pd.DatetimeIndex, col: str) -> np.ndarray:
    t = _daily_tables()[col]
    d = pd.DatetimeIndex(index).tz_convert("UTC").normalize()
    return t.reindex(d).to_numpy()


def moon_age(index: pd.DatetimeIndex) -> np.ndarray:
    return _lookup(index, "moon_age").astype(float)


def mercury_retro(index: pd.DatetimeIndex) -> np.ndarray:
    return _lookup(index, "retro").astype(bool)
