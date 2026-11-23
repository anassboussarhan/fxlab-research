"""Round 24 addendum 8: where the USD/TRY R1 returns come from (statistics only).
- the pair's D1 mid moves: days with |move| > 5% / 10%, the largest days with their spread;
- the counting-unit portfolio (carry x1 rerun): sum by year, share of 2019-2025 from its 10 / 20 best days, t without
  them, and without the pair's 10 largest-move days;
- hourly bars: hours whose bid / ask range or mid jump exceeds 5% (bad prints)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wf25  # noqa: E402
from fxlab.data.store import load_m1, resample  # noqa: E402

m = load_m1("USDTRY")
h1 = resample(m, "H1")
mid = (h1.bid_c + h1.ask_c) / 2
hr = np.log(mid).diff()
sp = (h1.ask_c - h1.bid_c) / mid
d1 = resample(m, "D1")
dm = np.log((d1.bid_c + d1.ask_c) / 2).diff().dropna()
dm.index = pd.DatetimeIndex(dm.index).tz_convert("UTC").tz_localize(None).normalize()
out = {"hours": int(len(h1)), "hours_abs_move_gt_5pct": int((hr.abs() > 0.05).sum()),
       "hours_spread_gt_1pct": int((sp > 0.01).sum()), "median_spread_bp": float(sp.median() * 1e4),
       "days_abs_move_gt_5pct": int((dm.abs() > 0.05).sum()), "days_abs_move_gt_10pct": int((dm.abs() > 0.10).sum()),
       "largest_days": [{"day": str(k.date()), "move": round(float(v), 4)} for k, v in dm.abs().sort_values().tail(12).items()]}
cnt = pd.read_csv("res/r24_streams.csv.gz", low_memory=False)
cnt = cnt[(cnt.counts == True) & (cnt.symbol == "USDTRY")]
keys = {f"{u}|{s}" for u, s in zip(cnt.unit, cnt.selection)}
ser = {}
for f in glob.glob("inp/r24TRS-x1.0-*/LW_series_USDTRY.parquet"):
    x = pd.read_parquet(f)
    for c in x.columns:
        if c in keys:
            ser[c] = x[c]
P = pd.DataFrame(ser)
P.index = pd.DatetimeIndex(pd.to_datetime(P.index))
port = P.mean(axis=1).dropna()
p2 = port[(port.index >= "2019-01-01") & (port.index <= "2025-12-31")]
out["portfolio_units"] = int(P.shape[1])
out["portfolio_by_year"] = {int(y): round(float(v), 3) for y, v in port.groupby(port.index.year).sum().items()}
tot = float(p2.sum())
for k in (10, 20):
    top = p2.sort_values().tail(k)
    rest = p2.drop(top.index)
    out[f"p2_share_from_best_{k}_days"] = round(float(top.sum()) / tot, 3) if tot else None
    out[f"p2_t_without_best_{k}_days"] = round(float(wf25.stats(rest)["p2_t"]), 3)
big = set(dm.abs().sort_values().tail(10).index)
nb = p2[~p2.index.normalize().isin(big)]
out["p2_t_without_pair_10_largest_days"] = round(float(wf25.stats(nb)["p2_t"]), 3)
out["p2_best_days"] = [{"day": str(k.date()), "ret": round(float(v), 4)} for k, v in p2.sort_values().tail(10).items()]
st = wf25.stats(port)
out["portfolio"] = {k: round(float(st[k]), 6) for k in ("p2_mean", "p2_t", "p2_hac_t")}
print(json.dumps(out, indent=1))
