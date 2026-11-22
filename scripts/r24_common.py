"""Round 24 shared pieces (docs/round24_preregistration.md): Sobol draws, the walk-forward of a configuration matrix
(default and nested, scripts/wf23.py) and the row format the combine reads."""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf22  # noqa: E402
import wf23  # noqa: E402

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
DEV_END = pd.Timestamp("2025-12-31")


def sobol(dims, n=128, seed=2401):
    return qmc.Sobol(dims, scramble=True, seed=seed).random(n)


def lin(u, lo, hi):
    return lo + u * (hi - lo)


def ilin(u, lo, hi):
    return int(min(hi, lo + math.floor(u * (hi - lo + 1))))


def ilog(u, lo, hi):
    return int(round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))))


def cat(u, choices):
    return choices[min(int(u * len(choices)), len(choices) - 1)]


def get(url, tries=6, timeout=120, params=None, js=False, method="get", body=None):
    import requests
    last = None
    for k in range(tries):
        try:
            if method == "post":
                r = requests.post(url, json=body, headers=UA, timeout=timeout)
            else:
                r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.json() if js else r
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 403, 404, 451):
                break
            if r.status_code == 429:
                time.sleep(30 * (k + 1))
                continue
        except Exception as ex:                                            # noqa: BLE001
            last = repr(ex)[:160]
        time.sleep(min(120, 5 * 2 ** k))
    raise RuntimeError(f"{url}: {last}")


def units(part, family, symbol, W, placebo=False, desc=None, min_active=20, min_valid=40, unit=""):
    """W: DataFrame, index = dates (daily or weekly), columns = configurations, values = net returns of that period
    (0 = flat, NaN = no data). Returns rows (default and nested) and the two streams."""
    W = W[W.index <= DEV_END].sort_index()
    W = W.loc[:, W.notna().any()]
    rows, ser = [], {}
    if W.empty or W.shape[1] == 0:
        return rows, ser
    first = int(W.index[0].year) + 3
    ag = wf23.Agg(W.index, P=W.to_numpy().T)
    res = wf23.both(ag, first, min_active=min_active, min_valid=min_valid)
    for kind, (s, st, own, ch) in res.items():
        r = {"part": part, "family": family, "symbol": symbol, "unit": unit or family, "placebo": placebo,
             "selection": kind, "n_cfg": int(W.shape[1]), **st, "own": own}
        if desc is not None:
            last = list(ch.items())[-3:] if isinstance(ch, dict) else []
            r["choices"] = json.dumps({str(k): (v if isinstance(v, str) else [desc[i] for i in v][:3]) for k, v in last},
                                      default=str)[:1500]
        rows.append(r)
        ser[f"{family}|{symbol}|{kind}{'|placebo' if placebo else ''}"] = s
    return rows, ser


def write(out, tag, rows, ser, meta=None):
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, f"R24_rows_{tag}.csv"), index=False)
    if ser:
        pd.DataFrame(ser).to_parquet(os.path.join(out, f"R24_series_{tag}.parquet"))
    if meta is not None:
        json.dump(meta, open(os.path.join(out, f"R24_meta_{tag}.json"), "w"), indent=1, default=str)
    for r in rows:
        print(json.dumps({k: r.get(k) for k in ("family", "symbol", "selection", "placebo", "oos_first", "p1_mean", "p2_mean",
                                                 "p2_t", "own")}, default=str), flush=True)
