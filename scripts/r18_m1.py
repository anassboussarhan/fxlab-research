"""Round 18, parts G and C (docs/round18_preregistration.md): per symbol, on M1 bid / ask data 2007-2025.

- Metals (XAUUSD, XAGUSD): the round-2 literature tests unchanged (lit2_lab.metals): G1, G2, G3, A1, A2, S1, SEAS.
- USD pairs: local hours (L1), around the clock (K_ny, K_on), fix reversal (W1) unchanged (flows_lab), plus the legs
  of the overnight-to-intraday reversal (C1): the night return 17:00 New York -> 08:00 London (N_night, mid) and the
  day trade 08:00 London -> 16:55 New York long (D_long) and short (D_short) the foreign currency.
- Index CFDs: the VWAP trend rule (C2), one row per day.

Output: r18_SYM.parquet (rows: test, day, net, gross, swap_rate, swap_flat, ...) and r18_SYM.json.
Usage: python scripts/r18_m1.py --symbol XAUUSD --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = "2025-12-31"
USD_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
INDICES = ("USA500IDXUSD", "USATECHIDXUSD")
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]


def reversal_legs(FL, book, pair, start, end):
    base, quote = pair[:3], pair[3:]
    foreign = base if quote == "USD" else quote
    long_f = 1.0 if base == foreign else -1.0
    rows = []
    for d in pd.bdate_range(start, end):
        p = FL.prev_bday(d)
        i0 = book.at(FL.local_ts(p, "America/New_York", 17, 0), 65)
        t1, t2 = FL.local_ts(d, "Europe/London", 8, 0), FL.local_ts(d, "America/New_York", 16, 55)
        i1 = book.at(t1, 5)
        if i0 < 0 or i1 < 0 or i1 <= i0:
            continue
        m0, m1 = 0.5 * (book.bo[i0] + book.ao[i0]), 0.5 * (book.bo[i1] + book.ao[i1])
        if not (np.isfinite(m0) and np.isfinite(m1) and m0 > 0):
            continue
        rl, rs = book.trade(t1, t2, long_f), book.trade(t1, t2, -long_f)
        if rl is None or rs is None:
            continue
        night = long_f * (m1 / m0 - 1.0)
        rows.append({"test": "N_night", "day": d, "net": night, "gross": night, "ccy": foreign})
        rows.append(FL.rec("D_long", d, rl, ccy=foreign))
        rows.append(FL.rec("D_short", d, rs, ccy=foreign))
    return rows


def vwap_trend(m1, pip, sf, sp, comm):
    """C2: one row per session day. Signal at each minute's close (mid vs session VWAP), traded at the next minute's
    open; flat at 16:00 New York. Cost per unit of position change: half the quoted spread + slippage + commission."""
    ny = m1.index.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    sess = (mins >= 570) & (mins <= 960) & (ny.dayofweek < 5)
    x = m1[sess]
    nyx = ny[sess]
    has_ticks = "n_ticks" in x.columns and x["n_ticks"].notna().mean() > 0.9
    rows = []
    for day, g in x.groupby(nyx.normalize()):
        if len(g) < 300:                                                    # a session with holes is skipped
            continue
        mo = 0.5 * (g["bid_o"].to_numpy() + g["ask_o"].to_numpy())
        mc = 0.5 * (g["bid_c"].to_numpy() + g["ask_c"].to_numpy())
        spr = (g["ask_o"].to_numpy() - g["bid_o"].to_numpy())
        if not (np.isfinite(mo).all() and np.isfinite(mc).all() and (mo > 0).all()):
            continue
        w = g["n_ticks"].to_numpy(float) if has_ticks else np.ones(len(g))
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        tp = (0.5 * (g["bid_h"].to_numpy() + g["ask_h"].to_numpy()) + 0.5 * (g["bid_l"].to_numpy() + g["ask_l"].to_numpy()) + mc) / 3
        vwap = np.cumsum(tp * w) / np.cumsum(w)
        sig = np.sign(mc - vwap)                                           # known at the close of bar k
        n = len(g)
        pos = np.zeros(n)                                                  # position held from the open of bar k
        pos[1:] = sig[:-1]
        pos[-1] = 0.0                                                      # flat from the last bar's open
        hold = pos[:-1] * (mo[1:] / mo[:-1] - 1.0)                         # open k -> open k+1
        dpos = np.abs(np.diff(np.r_[0.0, pos]))
        cost = dpos * ((0.5 * spr + sf * spr + sp * pip) / mo + comm)
        rows.append({"test": "C2_vwap", "day": pd.Timestamp(day.date()), "gross": float(hold.sum()),
                     "net": float(hold.sum() - cost.sum()), "flips": float(dpos.sum()), "ticks": bool(has_ticks)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2007-01-01")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    import flows_lab as FL
    import lit2_lab as L2
    from fxlab.data.store import load_m1
    sym = a.symbol
    meta, rows, errors = {"symbol": sym}, [], {}
    cols = COLS + (["n_ticks"] if sym in INDICES else [])
    try:
        m1 = load_m1(sym, a.start, DEV_END, columns=cols)
    except Exception:                                                       # noqa: BLE001  (a file without n_ticks)
        m1 = load_m1(sym, a.start, DEV_END, columns=COLS)
    start, end = str(m1.index[0].date()), str(min(m1.index[-1], pd.Timestamp(DEV_END, tz="UTC")).date())
    meta.update(start=start, end=end, rows_m1=int(len(m1)))
    try:
        if sym in L2.METALS:
            from fxlab.data.instruments import get as instrument
            rows += L2.metals(L2.metal_book(m1, instrument(sym).pip, sym), sym, start, end)
        elif sym in USD_PAIRS:
            from fxlab.data.instruments import get as instrument
            book = FL.Book(m1, instrument(sym).pip, sym)
            for fn in (FL.local_hours, FL.around_clock, FL.fix_reversal):
                try:
                    rows += fn(book, sym, start, end)
                except Exception as ex:                                     # noqa: BLE001
                    errors[fn.__name__] = repr(ex)[:300]
            rows += reversal_legs(FL, book, sym, start, end)
        elif sym in INDICES:
            from fxlab.execution.costs import BASE
            pip = float(1e-4 * np.nanmedian(0.5 * (m1.bid_c + m1.ask_c)))
            rows += vwap_trend(m1, pip, BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4)
    except Exception as ex:                                                 # noqa: BLE001
        errors["lab"] = repr(ex)[:500]
        print(sym, "ERROR", repr(ex)[:500], flush=True)
    meta["errors"] = errors
    meta["seconds"] = round(time.time() - t0, 1)
    R = pd.DataFrame(rows)
    if len(R):
        R["pair"] = sym
        R["day"] = pd.to_datetime(R["day"])
        R.to_parquet(os.path.join(a.out, f"r18_{sym}.parquet"))
        print(R.groupby("test")[["net", "gross"]].agg(["count", "mean"]).to_string(), flush=True)
    json.dump(meta, open(os.path.join(a.out, f"r18_{sym}.json"), "w"), indent=1, default=str)
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    main()
