"""2026 check: quality of the 2026 hour candles against 2025 (statistics only)."""
import glob, json
import numpy as np, pandas as pd
out = {}
for S in ("USDTRY", "EUSIDXEUR", "CHEIDXCHF"):
    r = {}
    for y in (2025, 2026):
        f = glob.glob(f"m1/{S}/{S}_{y}.parquet")
        if not f:
            continue
        x = pd.read_parquet(f[0])
        ts = pd.DatetimeIndex(x["ts"] if "ts" in x else x.index)
        mid = (x.bid_c + x.ask_c) / 2
        sp = (x.ask_c - x.bid_c) / mid
        lr = np.log(mid).diff()
        r[str(y)] = {"rows": int(len(x)), "first": str(ts.min()), "last": str(ts.max()), "median_step_min": float(pd.Series(ts).diff().dt.total_seconds().median() / 60),
                     "ask_below_bid": int((x.ask_c < x.bid_c).sum()), "spread_median_bp": float(sp.median() * 1e4),
                     "spread_p99_bp": float(sp.quantile(0.99) * 1e4), "abs_move_p99_bp": float(lr.abs().quantile(0.99) * 1e4),
                     "first_mid": float(mid.iloc[0]), "last_mid": float(mid.iloc[-1]),
                     "high_below_low": int((x.bid_h < x.bid_l).sum()), "close_outside_hl": int(((x.bid_c > x.bid_h + 1e-12) | (x.bid_c < x.bid_l - 1e-12)).sum()),
                     "columns": list(x.columns)[:14]}
    out[S] = r
print(json.dumps(out, indent=1))
