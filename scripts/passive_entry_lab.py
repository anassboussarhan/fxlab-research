"""Cutting costs instead of hunting edge: the out-of-sample trades of the three ML labs (the policy optimisation,
the adaptive barriers and the fixed triple barriers; every trade was chosen without seeing its own year) are
re-simulated with a limit order at the touch instead of a market order. The ML models beat random entries before
costs by about 1-2 bp a trade, and a market entry pays the spread plus slippage; this asks whether entering
passively turns that skill into a net profit once the fills that never happen (the trades that ran away) are lost.
Development data only (to 2025-12-31). Nothing is tuned: the rules below are fixed before any result.

FIXED BEFORE ANY RESULT:
  Signals: every out-of-sample trade stored on the branches policy-opt (linear, mlp, gbm), ml-adaptive-barrier
    (+ ml-adaptive-barrier-d1) and ml-triple-barrier (B1, B2, B3): entry time, side, take-profit, stop (ATR(14)
    multiples of the decision bar) and time limit (bars). The decision bar is the timeframe bar before the entry.
  Market entry (baseline, re-simulated): the labs' rule, at the open of the next bar, paying the ask plus 25% of
    the spread plus 0.1 pip (long), commission 0.35 bp a side, stop first within a minute, gaps through the stop
    fill at the open, target is a limit, swap per night.
  P1 passive entry (primary): a buy limit at the bid (sell limit at the ask) of the entry minute, valid for one
    bar of the timeframe; filled at the limit price at the first minute whose ask low (bid high) reaches it, the
    way MT5 fills limit orders at the quoted price. No slippage on the fill; commission unchanged. Within the fill
    minute only the stop can trigger (conservative); from the next minute as the market rule. The time exit is at
    the same moment as the market trade's. Unfilled signals are not traded.
  P1s strict: the same, but the quote must trade 0.2 pip through the limit.
  P2 passive in and out: P1, and at the time limit a limit at the opposite touch (sell at the ask for a long),
    valid one more bar, with the stop and the target still working; else a market exit at the end of that bar.
  Adverse selection is measured, not assumed: the market-entry gross return of the filled and the unfilled signals.
PRE-REGISTERED TEST (one-sided): mean net return per filled P1 trade > 0 (t clustered by month), per source x
  instrument x timeframe; Benjamini-Hochberg over all of them in the combine stage.
Usage: python scripts/passive_entry_lab.py --symbol EURUSD --trades DIR --out DIR
       python scripts/passive_entry_lab.py --combine DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_triple_barrier_lab as TB  # noqa: E402
import ml_adaptive_barrier_lab as AB  # noqa: E402

END = "2025-12-31 23:59"
TFS = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
STRICT_PIPS = 0.2


@njit(cache=True)
def sim_passive(bo, ao, bh, bl, ah, al, i, w, j, wx, side, tp_d, sl_d, sf, sp, comm, rc, sw_long, sw_short,
                through, exit_passive):
    """Limit entry at the touch of minute i, valid until minute w (exclusive); barriers from the fill price; time
    exit at minute j at market, or (exit_passive) a limit at the opposite touch of minute j valid until wx, then
    market. Returns (net, gross, exit minute, fill minute); fill minute -1 = not filled."""
    L = bo[i] if side > 0 else ao[i]
    xf = -1
    for x in range(i, min(w, j)):
        if side > 0:
            if al[x] <= L - through:
                xf = x
                break
        else:
            if bh[x] >= L + through:
                xf = x
                break
    if xf < 0:
        return np.nan, np.nan, -1, -1
    E = L
    if side > 0:
        TP, SL = E + tp_d, E - sl_d
    else:
        TP, SL = E - tp_d, E + sl_d
    k = j
    px = 0.0
    hit = False
    for x in range(xf, j):
        sx = sf * (ao[x] - bo[x]) + sp
        if side > 0:
            if bl[x] <= SL:
                px = min(SL, bo[x]) - sx if x > xf else SL - sx
                k = x
                hit = True
                break
            if x > xf and bh[x] >= TP:
                px = TP
                k = x
                hit = True
                break
        else:
            if ah[x] >= SL:
                px = max(SL, ao[x]) + sx if x > xf else SL + sx
                k = x
                hit = True
                break
            if x > xf and al[x] <= TP:
                px = TP
                k = x
                hit = True
                break
    if not hit and exit_passive:
        Lx = ao[j] if side > 0 else bo[j]
        for x in range(j, wx):
            sx = sf * (ao[x] - bo[x]) + sp
            if side > 0:
                if bl[x] <= SL:
                    px = min(SL, bo[x]) - sx
                    k = x
                    hit = True
                    break
                if bh[x] >= TP:
                    px = TP
                    k = x
                    hit = True
                    break
                if bh[x] >= Lx + through:
                    px = Lx
                    k = x
                    hit = True
                    break
            else:
                if ah[x] >= SL:
                    px = max(SL, ao[x]) + sx
                    k = x
                    hit = True
                    break
                if al[x] <= TP:
                    px = TP
                    k = x
                    hit = True
                    break
                if al[x] <= Lx - through:
                    px = Lx
                    k = x
                    hit = True
                    break
        if not hit:
            k = wx
    if not hit:
        sk = sf * (ao[k] - bo[k]) + sp
        px = bo[k] - sk if side > 0 else ao[k] + sk
    net = (px / E - 1.0) if side > 0 else (1.0 - px / E)
    swap = (rc[k] - rc[xf]) * (sw_long[i] if side > 0 else sw_short[i])
    mi = 0.5 * (ao[i] + bo[i])
    gross = side * (0.5 * (ao[k] + bo[k]) / mi - 1.0)
    return net - 2 * comm + swap, gross, k, xf


def atr14(bars: pd.DataFrame) -> np.ndarray:
    """ATR(14) exactly as scripts/ml_triple_barrier_lab.py computes it (EWM of the mid true range)."""
    h, l, c = (0.5 * (bars[f"bid_{k}"] + bars[f"ask_{k}"]) for k in "hlc")
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()


class Light:
    """The adaptive-barrier lab's Market without its features: M1 path, bars, ATR, swap arrays."""

    def __init__(self, m1, bk, rc, sw_long, sw_short, tf):
        from fxlab.data.store import resample
        bars = resample(m1, tf) if tf != "M1" else m1.copy()
        self.bars = bars.dropna(subset=["bid_o", "ask_o", "bid_h", "ask_l"])
        self.first_m1 = bk.idx.searchsorted(self.bars.index).astype(np.int64)
        self.atr = atr14(self.bars)
        self.bk, self.rc, self.sw_long, self.sw_short = bk, rc, sw_long, sw_short


def load_book(symbol):
    import flows_lab as FL
    from fxlab.data.store import load_m1
    m1 = load_m1(symbol, "2007-01-01", END)
    m1 = m1[(m1.ask_o >= m1.bid_o) & (m1.bid_o > 0)].dropna(subset=["bid_o", "ask_o", "bid_c", "ask_c"])
    pip = TB.PIPS.get(symbol, 1e-4)
    cols = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
    bk = FL.Book(m1[cols], pip, symbol if len(symbol) == 6 and symbol not in TB.CFD else None)
    bk._hl()
    rc = AB.rollover_index(bk.idx)
    days = bk.idx.floor("D")
    if symbol in TB.CFD:
        from fxlab.data import rates as R
        usd = np.nan_to_num(R.known_at(pd.DatetimeIndex(days.unique()))["USD"].reindex(days).to_numpy())
        swl, sws = -(usd + AB.MARKUP) / 360.0, (usd - AB.MARKUP) / 360.0
    elif bk.rd is not None:
        from fxlab.data.rates import SWAP_MARKUP_PA
        rd = np.nan_to_num(bk.rd.reindex(days, method="ffill").to_numpy())
        swl, sws = (rd - SWAP_MARKUP_PA) / 365.0, (-rd - SWAP_MARKUP_PA) / 365.0
    else:
        swl = sws = -0.5 * pip / (0.5 * (bk.bo + bk.ao))
    return m1, bk, rc, np.asarray(swl, float), np.asarray(sws, float), pip


def sources(trades_dir, symbol, tf):
    """(source name, DataFrame[time, side, tp, sl, h]) for every stored trade file of symbol x tf."""
    out = []
    fs = (glob.glob(os.path.join(trades_dir, "**", f"trades_{symbol}_{tf}.csv.gz"), recursive=True)
          + glob.glob(os.path.join(trades_dir, "**", f"trades_{symbol}_{tf}_*.csv.gz"), recursive=True))
    for f in sorted(fs):
        name = os.path.basename(f)[len(f"trades_{symbol}_{tf}"):-len(".csv.gz")].strip("_")
        T = pd.read_csv(f)
        if "policy_opt" in f:
            src = f"po-{name}"
        elif "ml_triple_barrier" in f:
            src = f"tb-{name}"
            tp, sl, h = TB.BARRIERS[name]
            T["tp"], T["sl"], T["h"] = tp, sl, h
        else:
            src = "ab"
        if "val_ok" in T:
            T = T[T.val_ok.astype(str) == "True"]                          # the policy lab's primary (gated) set
        T = T[["time", "side", "tp", "sl", "h"]].copy()
        T["tp"] = pd.to_numeric(T.tp, errors="coerce").replace([np.inf], 1e9)
        T["sl"] = pd.to_numeric(T.sl, errors="coerce").replace([np.inf], 1e9)
        out.append((src, T))
    return out


def run(a):
    from scipy import stats as st
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    m1, bk, rc, swl, sws, pip = load_book(a.symbol)
    arr = tuple(x.astype(float) for x in (bk.bo, bk.ao, bk.bh, bk.bl, bk.ah, bk.al))
    sf, sp, comm = bk.sf, bk.sp * bk.pip, bk.comm
    nm = len(bk.idx)
    print(a.symbol, "M1 rows", nm, f"{time.time() - t0:.0f}s", flush=True)
    rows, trades_out = [], []
    for tf in TFS:
        srcs = sources(a.trades, a.symbol, tf)
        if not srcs:
            continue
        mk = Light(m1, bk, rc, swl, sws, tf)
        fm = mk.first_m1
        nb = len(fm)
        for src, T in srcs:
            ts = pd.to_datetime(T.time, utc=True)
            I = bk.idx.get_indexer(pd.DatetimeIndex(ts))
            pos = np.searchsorted(fm, I)                                  # bar t+1 starts at minute I
            ok = (I >= 0) & (pos < nb) & (pos >= 1)
            ok[ok] &= fm[pos[ok]] == I[ok]
            res = []
            for q in np.flatnonzero(ok):
                i, t = int(I[q]), int(pos[q]) - 1
                side = float(T.side.iloc[q])
                a_t = mk.atr[t]
                if not np.isfinite(a_t) or a_t <= 0:
                    continue
                h = int(T.h.iloc[q])
                j = int(min(fm[min(t + 1 + h, nb - 1)], nm - 1))
                w = int(min(fm[min(t + 2, nb - 1)], nm - 1))
                w = max(w, i + 1)
                wx = int(min(fm[min(t + 2 + h, nb - 1)], nm - 1))
                wx = max(wx, j + 1) if j + 1 < nm else j
                if j <= i:
                    continue
                tp_d, sl_d = float(T.tp.iloc[q]) * a_t, float(T.sl.iloc[q]) * a_t
                mn, mg, _ = AB.sim(*arr, i, j, side, tp_d, sl_d, sf, sp, comm, rc, swl, sws)
                p1 = sim_passive(*arr, i, w, j, wx, side, tp_d, sl_d, sf, sp, comm, rc, swl, sws, 0.0, False)
                p1s = sim_passive(*arr, i, w, j, wx, side, tp_d, sl_d, sf, sp, comm, rc, swl, sws,
                                  STRICT_PIPS * pip, False)
                p2 = sim_passive(*arr, i, w, j, wx, side, tp_d, sl_d, sf, sp, comm, rc, swl, sws, 0.0, True)
                res.append((bk.idx[i], side, mn, mg, p1[0], p1[1], p1[3] >= 0, p1s[0], p1s[3] >= 0, p2[0],
                            (p1[3] - i) if p1[3] >= 0 else -1))
            if not res:
                continue
            R = pd.DataFrame(res, columns=["time", "side", "mkt_net", "mkt_gross", "p1_net", "p1_gross", "p1_fill",
                                           "p1s_net", "p1s_fill", "p2_net", "fill_delay_min"]).set_index("time")
            f1 = R[R.p1_fill]
            x = f1.p1_net
            tt = TB.month_t(x) if len(x) >= 10 else np.nan
            row = {"symbol": a.symbol, "tf": tf, "source": src, "signals": int(len(T)), "matched": int(len(R)),
                   "fill_rate": float(R.p1_fill.mean()), "mkt_net_bp": float(R.mkt_net.mean() * 1e4),
                   "mkt_t": TB.month_t(R.mkt_net) if len(R) >= 10 else np.nan,
                   "mkt_gross_bp": float(R.mkt_gross.mean() * 1e4),
                   "p1_net_bp": float(x.mean() * 1e4) if len(x) else np.nan, "p1_t": tt,
                   "p_one_sided": float(1 - st.norm.cdf(tt)) if np.isfinite(tt) else np.nan,
                   "p1_per_signal_bp": float(R.p1_net.fillna(0).mean() * 1e4),
                   "mkt_gross_filled_bp": float(f1.mkt_gross.mean() * 1e4) if len(f1) else np.nan,
                   "mkt_gross_unfilled_bp": float(R[~R.p1_fill].mkt_gross.mean() * 1e4) if (~R.p1_fill).any() else np.nan,
                   "mkt_net_filled_bp": float(f1.mkt_net.mean() * 1e4) if len(f1) else np.nan,
                   "p1s_fill_rate": float(R.p1s_fill.mean()),
                   "p1s_net_bp": float(R[R.p1s_fill].p1s_net.mean() * 1e4) if R.p1s_fill.any() else np.nan,
                   "p1s_t": TB.month_t(R[R.p1s_fill].p1s_net) if R.p1s_fill.sum() >= 10 else np.nan,
                   "p2_net_bp": float(f1.p2_net.mean() * 1e4) if len(f1) else np.nan,
                   "p2_t": TB.month_t(f1.p2_net) if len(f1) >= 10 else np.nan,
                   "median_fill_delay_min": float(f1.fill_delay_min.median()) if len(f1) else np.nan,
                   "years_positive": (f"{int((x.groupby(x.index.year).sum() > 0).sum())}/{x.index.year.nunique()}"
                                      if len(x) else "")}
            rows.append(row)
            R["source"], R["tf"] = src, tf
            trades_out.append(R.reset_index())
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"tests_{a.symbol}.csv"), index=False)
    if trades_out:
        pd.concat(trades_out).to_csv(os.path.join(a.out, f"trades_{a.symbol}.csv.gz"), index=False)
    json.dump({"symbol": a.symbol, "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(a.out, f"meta_{a.symbol}.json"), "w"))


def combine(a):
    from cross_market_ta_lab import bh_q
    fs = glob.glob(os.path.join(a.combine, "**", "tests_*.csv"), recursive=True)
    R = pd.concat([pd.read_csv(f) for f in fs if os.path.getsize(f) > 5], ignore_index=True)
    R["q_bh"] = bh_q(R.p_one_sided.to_numpy())
    os.makedirs(a.out, exist_ok=True)
    R.sort_values("p_one_sided").to_csv(os.path.join(a.out, "tests_all.csv"), index=False)
    R["family"] = R.source.str.split("-").str[0]

    def wmean(g, col, wcol):
        w = g[wcol].astype(float)
        return float((g[col] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    rows = []
    for (fam, tf), g in R.groupby(["family", "tf"]):
        g = g.assign(filled=g.matched * g.fill_rate)
        rows.append({"family": fam, "tf": tf, "tests": len(g), "signals": int(g.matched.sum()),
                     "fill_rate": wmean(g, "fill_rate", "matched"),
                     "mkt_net_bp": wmean(g, "mkt_net_bp", "matched"), "mkt_gross_bp": wmean(g, "mkt_gross_bp", "matched"),
                     "p1_net_bp": wmean(g, "p1_net_bp", "filled"), "p1_per_signal_bp": wmean(g, "p1_per_signal_bp", "matched"),
                     "gross_filled_bp": wmean(g, "mkt_gross_filled_bp", "filled"),
                     "gross_unfilled_bp": wmean(g.assign(unf=g.matched - g.filled), "mkt_gross_unfilled_bp", "unf"),
                     "p1s_net_bp": wmean(g.assign(fs=g.matched * g.p1s_fill_rate), "p1s_net_bp", "fs"),
                     "p2_net_bp": wmean(g, "p2_net_bp", "filled"),
                     "share_p1_positive": float((g.p1_net_bp > 0).mean()),
                     "p1_t_gt_2": int((g.p1_t > 2).sum()), "p1_t_lt_m2": int((g.p1_t < -2).sum())})
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(a.out, "by_family_timeframe.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(G.round(2).to_string())
    print(R.sort_values("p_one_sided").head(40)[["symbol", "tf", "source", "matched", "fill_rate", "mkt_net_bp",
                                                 "p1_net_bp", "p1_t", "p1s_net_bp", "p2_net_bp", "p2_t",
                                                 "mkt_gross_filled_bp", "mkt_gross_unfilled_bp", "q_bh",
                                                 "years_positive"]].round(2).to_string())
    json.dump({"tests": int(len(R)), "q10": int((R.q_bh <= 0.10).sum()), "p1_t_gt_2": int((R.p1_t > 2).sum()),
               "p1_t_lt_m2": int((R.p1_t < -2).sum()), "p1_positive": int((R.p1_net_bp > 0).sum())},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--trades", default="in")
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", default="")
    a = ap.parse_args()
    combine(a) if a.combine else run(a)


if __name__ == "__main__":
    main()
