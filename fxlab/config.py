"""Global configuration: paths, holdout protection, reproducibility."""
from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FXLAB_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
M1_DIR = DATA_DIR / "m1"
MANIFEST_DIR = DATA_DIR / "manifests"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"

# ---------------------------------------------------------------------------
# Holdout protection. Development data ends 2025-12-31 23:59:59 UTC.
# Anything at or after HOLDOUT_START is refused by every loader unless the
# holdout has been formally unlocked (see fxlab.data.store.unlock_holdout).
# ---------------------------------------------------------------------------
DEV_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
HOLDOUT_UNLOCK_ENV = "FXLAB_HOLDOUT_UNLOCK"
FROZEN_CANDIDATES_FILE = RESULTS_DIR / "frozen_candidates.json"

GLOBAL_SEED = 20251231


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass


for _d in (M1_DIR, MANIFEST_DIR, CACHE_DIR, RESULTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
