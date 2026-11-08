"""Round 14 data prep (docs/trend14_preregistration.md): one row per D1 bar, the day split at the 08:00 London fill.

Columns: trading date; bar open bid / ask; the pre-fill segment's bid / ask high and low; the fill minute's bid / ask
and slippage; the post-fill segment's bid / ask high and low; the close bid / ask; mid OHLC; rollovers into the bar;
swap per night for a long and a short; rate differential. Built with the round-13 data layer (long_lab.Mkt), which
only reads and arranges prices. Usage: python scripts/trend14_prep.py --symbol EURUSD --out DIR"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@njit(cache=True)
def segments(first, ex, n, bh, bl, ah, al):
    nb = len(first)
    out = np.full((nb, 8), np.nan)          # pre: bid_h bid_l ask_h ask_l ; post: bid_h bid_l ask_h ask_l
    for k in range(nb):
        a = first[k]
        b = first[k + 1] if k + 1 < nb else n
        x = ex[k] if ex[k] >= 0 else b
        if x > a:
            out[k, 0] = bh[a:x].max()
            out[k, 1] = bl[a:x].min()
            out[k, 2] = ah[a:x].max()
            out[k, 3] = al[a:x].min()
        if b > x:
            out[k, 4] = bh[x:b].max()
            out[k, 5] = bl[x:b].min()
            out[k, 6] = ah[x:b].max()
            out[k, 7] = al[x:b].min()
    return out


def build(sym: str):
    import edges_lab2 as E2
    import long_lab as LL
    bk = E2.book(sym)
    if bk is None:
        return None, None
    return from_mkt(LL.Mkt(bk.m, sym, bk.pip)), bk.pip


def from_mkt(M):
    nb = len(M.d1)
    seg = segments(M.first, M.exec, len(M.t), M.bh, M.bl, M.ah, M.al)
    last = np.r_[M.first[1:] - 1, len(M.t) - 1]
    ex = np.maximum(M.exec, 0)
    has = M.exec >= 0
    P = pd.DataFrame({
        "tday": M.tday.values, "o_bid": M.bo[M.first], "o_ask": M.ao[M.first],
        "pre_bh": seg[:, 0], "pre_bl": seg[:, 1], "pre_ah": seg[:, 2], "pre_al": seg[:, 3],
        "x_ok": has, "x_bid": np.where(has, M.bo[ex], np.nan), "x_ask": np.where(has, M.ao[ex], np.nan),
        "x_slip": np.where(has, M.slip[ex], np.nan),
        "post_bh": seg[:, 4], "post_bl": seg[:, 5], "post_ah": seg[:, 6], "post_al": seg[:, 7],
        "c_bid": M.bc[last], "c_ask": M.ac[last],
        "open": M.d1["open"].to_numpy(), "high": M.d1["high"].to_numpy(), "low": M.d1["low"].to_numpy(),
        "close": M.d1["close"].to_numpy(), "roll": M.roll, "swl": M.swl, "sws": M.sws, "rd": M.rd,
    }, index=M.d1.index)
    P.index.name = "t"
    assert len(P) == nb
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P, pip = build(a.symbol)
    if P is None:
        print("no data", a.symbol)
        return
    P.attrs["pip"] = pip
    P["pip"] = pip
    P.to_parquet(os.path.join(a.out, f"d1p_{a.symbol}.parquet"))
    print(a.symbol, len(P), P.tday.iloc[0], P.tday.iloc[-1], "fills", int(P.x_ok.sum()),
          "pre-seg missing", int(P.pre_bh.isna().sum()), "post-seg missing", int(P.post_bh.isna().sum()))


if __name__ == "__main__":
    main()
