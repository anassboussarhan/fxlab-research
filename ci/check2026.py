"""2026 check of the round 24-25 counting streams (docs/check2026_r24r25_preregistration.md): reproduction of
2019-2025 and the 2026-01..10 statistics of each frozen stream and of the USD/TRY portfolios. Statistics only."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import wf22  # noqa: E402


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


def p2(s):
    b = s[(s.index >= "2019-01-01") & (s.index <= "2025-12-31")]
    return tstat(b.to_numpy())


def y26(s):
    b = s[(s.index >= "2026-01-01") & (s.index <= "2026-10-31")]
    return {"n": int(len(b)), "mean": float(b.mean()) if len(b) else None, "t": tstat(b.to_numpy()) if len(b) else None,
            "sum": float(b.sum()) if len(b) else None}


def load(pattern):
    ser = {}
    for f in glob.glob(pattern, recursive=True):
        x = pd.read_parquet(f)
        for c in x.columns:
            s = x[c].dropna(); s.index = pd.DatetimeIndex(pd.to_datetime(s.index)); ser[c] = s
    return ser


out = {}
r24 = pd.read_csv("res/r24_streams.csv.gz", low_memory=False)
r24 = r24[(r24.counts == True) & (r24.symbol == "USDTRY")]
r25 = pd.read_csv("res/r25_streams.csv.gz", low_memory=False)
r25 = r25[(r25.counts == True) & ~r25.family.astype(str).str.endswith("FIX_REVERSAL")]


def portfolio(name, units, ser, pub):
    rows, keep = [], {}
    for k, t_pub in zip(units, pub):
        s = ser.get(k)
        if s is None:
            rows.append({"k": k, "found": False}); continue
        t = p2(s)
        rep = True if t_pub is None else bool(np.isfinite(t) and abs(t - t_pub) <= 0.01)
        rows.append({"k": k, "found": True, "p2_t": t, "published": t_pub, "reproduced": rep, **{f"y26_{a}": b for a, b in y26(s).items()}})
        if rep:
            keep[k] = s
    P = pd.DataFrame(keep).mean(axis=1).dropna() if keep else pd.Series(dtype=float)
    R = pd.DataFrame(rows)
    out[name] = {"units": len(units), "found": int(R.found.sum()) if len(R) else 0,
                 "reproduced": int(R.get("reproduced", pd.Series(dtype=bool)).fillna(False).sum()),
                 "portfolio_p2_t": p2(P) if len(P) else None, "portfolio_2026": y26(P) if len(P) else None,
                 "units_2026_positive": int((R.get("y26_mean", pd.Series(dtype=float)) > 0).sum())}
    R.to_csv(f"res/check2026_{name}.csv", index=False)


for sc in ("x1.0", "x1.5"):
    ser = load(f"inp/c26-r1-{sc}-*/LW_series_USDTRY.parquet")
    portfolio(f"R1_USDTRY_{sc}", [f"{u}|{s}" for u, s in zip(r24.unit, r24.selection)], ser,
              list(r24.p2_t) if sc == "x1.0" else [None] * len(r24))
serP = load("inp/c26-p-*/P_series_*.parquet") | load("inp/c26-alt-*/P_series_*.parquet")
t = r25[r25.symbol == "USDTRY"]
portfolio("P_USDTRY", [f"{u}|{sy}|{se}" for u, sy, se in zip(t.unit, t.symbol, t.selection)], serP, list(t.p2_t))
singles = r25[(r25.symbol != "USDTRY") & ~r25.symbol.astype(str).str.startswith("PANEL")]
for r in singles.itertuples():
    k = f"{r.unit}|{r.symbol}|{r.selection}"
    s = serP.get(k)
    out[f"single {r.part} {r.symbol} {r.unit} {r.selection}"] = ({"published": r.p2_t, "p2_t": p2(s), "y2026": y26(s)}
                                                                  if s is not None else {"found": False})
for f in glob.glob("res/try_bh_*.json"):
    out[os.path.basename(f)[:-5]] = json.load(open(f))
print(json.dumps(out, indent=1, default=str))
json.dump(out, open("res/check2026_summary.json", "w"), indent=1, default=str)
