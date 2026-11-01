"""Seventh literature round: the AI forecasters people now claim an edge with, and the two gold rules still untested.
Daily Yahoo closes; rules fixed before any result; development data only (to 2025-12-31).

FIXED BEFORE ANY RESULT:
  AI1 Time-series foundation models, zero-shot (the "AI predicts the market" claim; Chronos-Bolt, Amazon 2024):
     each trading day t, the model sees the last 512 closes up to t and forecasts the close of t+1 (median); hold the
     instrument from close t to close t+1 in the sign of (forecast - close t). Models: chronos-bolt-small and
     chronos-bolt-base. Series: EURUSD=X, GBPUSD=X, JPY=X (USD/JPY), AUDUSD=X, GLD (gold), 2006-2025. Costs: 1 bp
     per unit of position change (FX), 2 bp (GLD); no carry (one day). Reported next to a naive benchmark (the sign
     of day t's return) and with 2025 separately: Chronos-Bolt was released in November 2024 and its training data
     may include these series before then, so 2025 is the only period it cannot have seen.
  G7 Gold and real yields (Erb & Harvey 2013 sceptical; the common "falling real yields -> buy gold" rule): long GLD
     for the next month when TIP (inflation-protected Treasuries, whose price rises as real yields fall) rose over
     the past 3 months, otherwise T-bills; 2 bp a switch; 2004-2025.
  G8 Gold vs gold miners pair (CXO's test, 2006-2016): weekly GLD / GDX ratio; if it is above its 13-week mean, long
     GDX and short GLD for the week, below: the reverse (the X = 0 variant, always in); 2 bp a leg on changes;
     2006-2025.
PRE-REGISTERED TESTS (one-sided, mean net > 0, Newey-West t with the variance floored at iid; Benjamini-Hochberg
  over the ten AI1 series and G7, G8).
Usage: python scripts/lit7_lab.py --out DIR [--only ai,gold]"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = pd.Timestamp("2025-12-31")
SERIES = {"EURUSD": ("EURUSD=X", 1e-4), "GBPUSD": ("GBPUSD=X", 1e-4), "USDJPY": ("JPY=X", 1e-4),
          "AUDUSD": ("AUDUSD=X", 1e-4), "GOLD": ("GLD", 2e-4)}
MODELS = ("amazon/chronos-bolt-small", "amazon/chronos-bolt-base")
CTX = 512


def yahoo(t, start="2003-01-01"):
    import yfinance as yf
    h = yf.download(t, start=start, end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    h = h.iloc[:, 0] if isinstance(h, pd.DataFrame) else h
    h.index = pd.DatetimeIndex(h.index).tz_localize(None)
    h = h[h.index <= END].dropna()
    return h[h > 0]


def nw_t(x, lags=5):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 12:
        return np.nan
    e = x - x.mean()
    s0 = (e @ e) / n
    s = s0
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    s = max(s, s0)
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else np.nan


def summ(name, x, per_year, **kw):
    x = pd.Series(x, dtype=float).dropna()
    t = nw_t(x)
    yrs = x.groupby(x.index.year).sum()
    y25 = x[x.index.year == 2025]
    return {"test": name, "n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": t,
            "p_one_sided": float(st.norm.sf(t)) if np.isfinite(t) else np.nan,
            "sharpe": float(x.mean() / x.std() * math.sqrt(per_year)) if x.std() > 0 else np.nan,
            "years_positive": f"{int((yrs > 0).sum())}/{len(yrs)}",
            "mean_bp_2025": float(y25.mean() * 1e4) if len(y25) else np.nan, "t_2025": nw_t(y25), **kw}


def positions_pnl(px, pos, cost):
    r = px.pct_change().shift(-1)                          # return from close t to close t+1
    turn = pos.diff().abs().fillna(pos.abs())
    return (pos * r - turn * cost).dropna()


def chronos_forecasts(px, model_id, batch=256):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    v = px.to_numpy(dtype=np.float32)
    idx = list(range(CTX, len(v)))                         # forecast for t+1 made at t = i - 1 ... use closes up to i-1
    out = np.full(len(v), np.nan)
    for a in range(0, len(idx), batch):
        chunk = idx[a:a + batch]
        ctx = torch.stack([torch.tensor(v[i - CTX:i]) for i in chunk])   # closes t-511..t where t = i - 1
        res = pipe.predict_quantiles(ctx, prediction_length=1, quantile_levels=[0.5])
        q = res[0]
        if isinstance(q, (list, tuple)):
            vals = np.array([float(np.asarray(x).reshape(-1)[0]) for x in q])
        else:
            vals = np.asarray(q).reshape(len(chunk), -1)[:, 0]
        out[np.array(chunk) - 1] = vals                     # forecast of close (t+1), stored at t
    return pd.Series(out, index=px.index)


def ai_tests(rows, series, info):
    for name, (tick, cost) in SERIES.items():
        px = yahoo(tick)
        px = px[px.index >= "2003-12-01"]
        info[f"{name} first"] = str(px.index.min())[:10]
        naive = np.sign(px.pct_change()).fillna(0)
        x = positions_pnl(px, naive, cost)
        x = x[x.index >= "2006-01-01"]
        rows.append(summ(f"AI1 benchmark: {name} sign of today's return", x, 252))
        for m in MODELS:
            f = chronos_forecasts(px, m)
            pos = np.sign(f - px).where(f.notna())
            x = positions_pnl(px, pos.fillna(0), cost)
            x = x[(x.index >= "2006-01-01") & f.reindex(x.index).notna()]
            hit = float((np.sign(f - px).shift(0).reindex(x.index) == np.sign(px.pct_change().shift(-1).reindex(x.index))).mean())
            label = m.split("/")[-1]
            rows.append(summ(f"AI1 {name} {label}", x, 252, direction_hit=hit))
            series[f"AI1 {name} {label}"] = x
            print(rows[-1], flush=True)


def gold_tests(rows, series):
    gld, tip, gdx, irx = yahoo("GLD", "2004-01-01"), yahoo("TIP", "2003-06-01"), yahoo("GDX", "2006-05-01"), yahoo("^IRX")
    # G7: monthly
    m_g = gld.resample("ME").last()
    m_t = tip.resample("ME").last()
    rf = (irx.resample("ME").last() / 100 / 12).reindex(m_g.index).ffill().fillna(0)
    sig = (m_t.pct_change(3) > 0).reindex(m_g.index).fillna(False)
    nxt = m_g.pct_change().shift(-1)
    pos = sig.astype(float)
    x = (pos * nxt + (1 - pos) * rf.shift(-1) - pos.diff().abs().fillna(pos) * 2e-4).dropna()
    x = x[x.index >= "2004-12-31"] - rf.shift(-1).reindex(x.index).fillna(0)      # excess over T-bills
    rows.append(summ("G7 gold when TIP rose over 3 months (excess over T-bills, monthly)", x, 12,
                     share_months_in=float(pos.reindex(x.index).mean())))
    series["G7"] = x
    # information: the same months for buy-and-hold gold, and timed minus buy-and-hold (is it timing or gold's drift?)
    bh = (nxt - rf.shift(-1)).reindex(x.index)
    rows.append(summ("G7 benchmark: buy-and-hold gold, excess over T-bills", bh, 12))
    rows.append(summ("G7 benchmark: timed minus buy-and-hold", x - bh, 12))
    pin = pos.reindex(x.index) > 0
    rows.append({"test": "G7 benchmark: gold excess in signal months minus other months (Welch t)",
                 "n": int(len(bh)), "mean_bp": float((bh[pin].mean() - bh[~pin].mean()) * 1e4),
                 "t": float(st.ttest_ind(bh[pin], bh[~pin], equal_var=False).statistic)})
    # G8: weekly pair
    w = pd.concat([gld.resample("W-FRI").last().rename("gld"), gdx.resample("W-FRI").last().rename("gdx")], axis=1).dropna()
    ratio = w.gld / w.gdx
    p = np.sign(ratio - ratio.rolling(13).mean())                  # +1: ratio high -> long GDX, short GLD
    rg, rx = w.gld.pct_change().shift(-1), w.gdx.pct_change().shift(-1)
    pnl = 0.5 * p * (rx - rg) - p.diff().abs().fillna(p.abs()) * 2e-4
    pnl = pnl.dropna()
    pnl = pnl[pnl.index >= "2006-09-01"]
    rows.append(summ("G8 GLD / GDX ratio pair, 13-week mean, always in (weekly)", pnl, 52))
    series["G8"] = pnl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="ai,gold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, series, info, errors = [], {}, {}, {}
    for key, fn in (("gold", lambda: gold_tests(rows, series)), ("ai", lambda: ai_tests(rows, series, info))):
        if key in a.only:
            try:
                fn()
            except Exception:                                               # noqa: BLE001
                import traceback
                errors[key] = traceback.format_exc()[-1500:]
                print(errors[key], flush=True)
                print("::error title=lit7 " + key + "::" + errors[key].replace("\n", " | ")[-800:], flush=True)
    R = pd.DataFrame(rows)
    from cross_market_ta_lab import bh_q
    if len(R):
        prim = ~R.test.str.contains("benchmark")
        R["q_bh"] = np.nan
        R.loc[prim, "q_bh"] = bh_q(R.loc[prim, "p_one_sided"].to_numpy())
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    for k, v in series.items():
        v.rename("net").to_csv(os.path.join(a.out, "series_" + k.replace(" ", "_").replace("/", "_") + ".csv"))
    json.dump({"info": info, "errors": errors}, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
