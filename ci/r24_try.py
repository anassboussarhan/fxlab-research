"""Round 24 addendum 6: the USD/TRY benchmark. Buy-and-hold long USD/TRY (D1 bars as R1 builds them, the R1 cost model)
with the BIS rate differential charged as in R1, and with it scaled 1.5x and 2x (TRY funding above the policy rate).
Prints P1 / P2 statistics only."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import intraday17 as I17  # noqa: E402
import wf22  # noqa: E402
import wf25  # noqa: E402
from fxlab.data.store import load_m1, resample  # noqa: E402
from fxlab.evaluate import evaluate  # noqa: E402
import fxlab.data.rates_oecd as RO  # noqa: E402

I17.strategies()
m1 = load_m1("USDTRY")
b = resample(m1, "D1")
b = b[b.index <= pd.Timestamp("2025-12-31 23:59", tz="UTC")]
day = I17.trading_day(b.index)
dd = pd.Index(sorted(set(day)))
sig = pd.DataFrame({"side": np.ones(len(b))}, index=b.index)
orig = RO.rate_diff
out = {}
for k in (0.0, 1.0, 1.5, 2.0):
    RO.rate_diff = (lambda s, i, k=k: orig(s, i) * k)
    r = evaluate("S.TREND.MA_CROSS", b, "USDTRY", signals=sig, financing="rates_oecd", mode_override="state")["returns"]
    s = pd.Series(I17.daily_sum(r, day, dd), index=pd.DatetimeIndex(pd.to_datetime(dd)))
    st = wf25.stats(s)
    out[f"carry_x{k}"] = {x: (round(float(st[x]), 6) if st.get(x) is not None and np.isfinite(st[x]) else None)
                          for x in ("p1_mean", "p1_t", "p2_mean", "p2_t", "p2_hac_t")}
    y = s.groupby(s.index.year).sum()
    out[f"carry_x{k}"]["annual_sum_2019_2025"] = {int(a): round(float(v), 3) for a, v in y[y.index >= 2019].items()}
print(json.dumps(out, indent=1))
