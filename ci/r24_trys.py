"""Round 24 addendum 7: the USD/TRY R1 units rerun with carry x1, x1.5 and x2 (LW_series of r24TRS-* artifacts).
For every unit stream that counted in the combine: own rule, iid t and Newey-West t at each scale; the equal-weight
portfolio of those streams. Statistics only."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import wf22  # noqa: E402
import wf25  # noqa: E402

cnt = pd.read_csv("res/r24_streams.csv.gz", low_memory=False)
cnt = cnt[(cnt.counts == True) & (cnt.symbol == "USDTRY")]
keys = {f"{u}|{s}" for u, s in zip(cnt.unit, cnt.selection)}
out = {"counting_units": len(keys)}
for sc in ("1.0", "1.5", "2.0"):
    ser = {}
    for f in glob.glob(f"inp/r24TRS-x{sc}-*/LW_series_USDTRY.parquet"):
        x = pd.read_parquet(f)
        for c in x.columns:
            ser[c] = x[c]
    rows = []
    for k in sorted(keys):
        s = ser.get(k)
        if s is None:
            continue
        s = s.dropna(); s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
        st = wf25.stats(s)
        own22 = bool((st["p1_n"] == 0 or st["p1_mean"] > 0) and st["p2_mean"] > 0 and st["p2_t"] >= 1.645)
        rows.append({"k": k, "p2_t": st["p2_t"], "hac": st["p2_hac_t"], "own": own22, "own_hac": wf25.counts_rule(st)})
    R = pd.DataFrame(rows)
    port = pd.DataFrame({k: ser[k] for k in keys if k in ser}).mean(axis=1).dropna() if len(R) else pd.Series(dtype=float)
    pst = wf25.stats(port) if len(port) else {}
    out[f"x{sc}"] = {"found": len(R), "own_rule": int(R.own.sum()) if len(R) else 0,
                     "own_with_hac": int(R.own_hac.sum()) if len(R) else 0,
                     "median_t": float(R.p2_t.median()) if len(R) else None,
                     "median_hac_t": float(R.hac.median()) if len(R) else None,
                     "portfolio": {k: (round(float(v), 6) if isinstance(v, (int, float)) and np.isfinite(v) else v)
                                   for k, v in pst.items() if k in ("p1_mean", "p1_t", "p2_mean", "p2_t", "p2_hac_t")}}
print(json.dumps(out, indent=1))
json.dump(out, open("res/r24_try_stress.json", "w"), indent=1)
