"""Cloud sweep, step 2: merge every instrument's close series into one panel file per timeframe
(data/panel/panel_<TF>.parquet), used through FXLAB_PANEL_DIR by cross-instrument features and strategies."""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci.data_job import PANEL_TFS  # noqa: E402

os.makedirs("data/panel", exist_ok=True)
for tf in PANEL_TFS:
    cols = {}
    for f in sorted(glob.glob(f"data/closes/*_{tf}.parquet")):
        sym = os.path.basename(f).rsplit("_", 1)[0]
        cols[sym] = pd.read_parquet(f)["close"]
    P = pd.DataFrame(cols).sort_index()
    P.astype("float64").to_parquet(f"data/panel/panel_{tf}.parquet")
    print(tf, P.shape, flush=True)
