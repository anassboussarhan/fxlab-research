"""Feature matrix construction with a content-addressed on-disk cache.

Cache key = sha1(symbol, timeframe, feature id, params, data fingerprint, code
version of the feature's module). Re-running an experiment never recomputes an
identical feature; editing a feature's source invalidates only its entries."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fxlab.config import CACHE_DIR
from fxlab.registry import CATALOG

FEATURE_CACHE = CACHE_DIR / "features"
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)


def data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha1()
    h.update(str((len(df), df.index[0], df.index[-1])).encode())
    h.update(np.ascontiguousarray(df["close"].to_numpy()[:: max(1, len(df) // 997)]).tobytes())
    return h.hexdigest()[:16]


def _code_version(fid: str) -> str:
    fn = CATALOG[fid].fn
    try:
        src = inspect.getsource(inspect.getmodule(fn))
    except Exception:
        src = fid
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def key(symbol, tf, fid, params, fp) -> str:
    payload = json.dumps([symbol, tf, fid, params, fp, _code_version(fid)], sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


def compute_feature(df, fid, symbol, tf, panel=None, params=None, use_cache=True) -> pd.DataFrame:
    e = CATALOG[fid]
    p = dict(e.params) | dict(params or {})
    fp = data_fingerprint(df)
    k = key(symbol, tf, fid, p, fp)
    path = FEATURE_CACHE / f"{k}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    out = e.fn(df, panel=panel, symbol=symbol, **p) if e.multi_asset else e.fn(df, **p)
    if isinstance(out, pd.Series):
        out = out.to_frame("v")
    out = out.astype("float32")
    out.columns = [f"{fid}:{c}" for c in out.columns]
    if use_cache:
        out.to_parquet(path)
    return out


def build_matrix(df, fids, symbol, tf, panel=None, params: dict | None = None, use_cache=True,
                 on_error="record") -> tuple[pd.DataFrame, dict]:
    cols, errors = [], {}
    for fid in fids:
        try:
            cols.append(compute_feature(df, fid, symbol, tf, panel, (params or {}).get(fid), use_cache))
        except Exception as ex:  # failures are recorded, never silently dropped
            if on_error == "raise":
                raise
            errors[fid] = repr(ex)
    X = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=df.index)
    return X, errors
