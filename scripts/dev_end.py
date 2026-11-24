"""Pre-registered 2026 checks only: move the data end and the last selection year of the round 22-25 code together.

apply("2026-10-31", 2026) sets every module's DEV_END and wf23.LAST_YEAR, and makes load_m1 read the holdout (it
opens only with the unlock token of fxlab.config). Nothing else changes: the code, seeds and configurations are those
of the rounds, so every stream reproduces before 2026 and its 2026 part uses selections made from earlier data only.
"""
from __future__ import annotations

import pandas as pd


def apply(end: str, last_year: int):
    import intraday17 as I17
    import r23_LW
    import r25_common
    import wf22
    import wf23
    import fxlab.data.store as ST
    ts = pd.Timestamp(f"{end} 23:59:59", tz="UTC")
    I17.DEV_END = ts
    r23_LW.DEV_END = ts
    r25_common.DEV_END = ts
    wf22.DEV_END = pd.Timestamp(end)
    wf23.LAST_YEAR = int(last_year)
    import sys
    mm = sys.modules.get("__main__")
    if mm is not None and hasattr(mm, "DEV_END"):                         # a runner started as a script
        mm.DEV_END = ts
    if not ST.holdout_unlocked():
        raise SystemExit("holdout locked: the 2026 check needs the unlock token")
    orig = ST.load_m1

    def load_m1(symbol, start=None, end=None, allow_holdout=True, columns=None):
        return orig(symbol, start, end, allow_holdout=True, columns=columns)
    ST.load_m1 = load_m1
