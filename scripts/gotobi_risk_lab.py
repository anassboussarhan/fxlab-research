"""Gotobi overnight with risk control: which exit rule cuts the losses and keeps the gains, and which hedge or
filter raises the share of winning trades. The trade is the frozen one (long 23:00 Tokyo the evening before ->
09:53 Tokyo on Gotobi days). M1 bid/ask path for every fill; costs as in the flows lab (market orders cross the
spread plus 25% of it and 0.1 pip; 0.35 bp commission a side; the rate swap over 17:00 New York). Development
data only (to 2025-12-31).

SELECTION PROTOCOL (fixed before any result): every variant is run on 2007-2025. The best variant is chosen on
2007-2018 ("train") only, then reported on 2019-2025 ("test"), which plays no part in the choice. Objective (the
user's request: cut losses, keep gains): the highest train score = mean net / CVaR5 of the open loss (the average
worst open loss of the worst 5% of trades), among variants whose train mean net is at least 75% of the no-stop
rule's. All variants are reported on both periods.

UNIT W: the mean high-low range of the 23:00 -> 09:53 window over the previous 20 Gotobi days of the same pair
(known before entry; at least 10 of them).

PART 1, EXITS (USDJPY, EURJPY, CHFJPY; each pair alone and the USDJPY + EURJPY basket):
  none                      the frozen rule
  stop k W                  k = 0.25, 0.5, 0.75, 1.0, 1.5, 2.0
  stop x bp                 x = 15, 25, 40, 60, 100
  stop k W + target t W     k = 0.5, 1.0; t = 0.5, 1.0
  break-even                stop 1.0 W, moved to the entry once the trade is +b W (b = 0.25, 0.5)
  trailing                  a stop c W below the highest bid since entry (c = 0.25, 0.5, 0.75, 1.0)
  time cut                  at 05:00 or 07:00 Tokyo, exit if the trade is below -g W (g = 0, 0.25, 0.5)
  Stops and targets are resting orders (stop checked first in a bar, a gap fills at the open; the fill bar only
  checks the stop); time cuts and the 09:53 exit are market orders.
PART 2, HEDGES AND FILTERS (no stop; USDJPY and EURJPY):
  hedge: short beta x a JPY cross that carries little or no Gotobi flow over the same window (NZDJPY, AUDJPY,
    GBPJPY, CADJPY), beta = 0.5 and 1.0; short beta x US500 (2019-2025 only). Both legs pay their own costs.
  filters (known at entry): skip nights with a scheduled FOMC announcement; skip when VIX closed above 25 the day
    before; skip when the pair moved more than +1 W or less than -1 W over the 24 hours before entry.
REPORTED per variant and period: trades, mean net (bp), t, win rate, average win / average loss, worst trade,
  share of trades losing more than 50 bp, worst open loss, CVaR5 of the open loss, the score.
Usage: python scripts/gotobi_risk_lab.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

LONG = ("USDJPY", "EURJPY", "CHFJPY")
HEDGES = ("NZDJPY", "AUDJPY", "GBPJPY", "CADJPY")
U.PIPS.update({p: 0.01 for p in LONG + HEDGES})
DAY = pd.Timedelta(days=1)
TRAIN_END = pd.Timestamp("2018-12-31")


def variants():
    V = [("none", {})]
    V += [(f"stop {k} W", {"sl_w": k}) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
    V += [(f"stop {x} bp", {"sl_bp": x}) for x in (15, 25, 40, 60, 100)]
    V += [(f"stop {k} W + target {t} W", {"sl_w": k, "tp_w": t}) for k in (0.5, 1.0) for t in (0.5, 1.0)]
    V += [(f"stop 1.0 W, break-even at +{b} W", {"sl_w": 1.0, "be_w": b}) for b in (0.25, 0.5)]
    V += [(f"trailing {c} W", {"trail_w": c}) for c in (0.25, 0.5, 0.75, 1.0)]
    V += [(f"time cut {h:02d}:00 if below -{g} W", {"cut_h": h, "cut_g": g}) for h in (5, 7) for g in (0, 0.25, 0.5)]
    return V


def slip(bk, k):
    return bk.sf * (bk.ao[k] - bk.bo[k]) + bk.sp * bk.pip


def swap_ret(bk, i, j, d, px):
    n, flat, rate = bk.swaps(i, j, d, px)
    return rate if np.isfinite(rate) else flat


def exit_sim(bk, i, j, W, rule, kcut=None):
    """Long from the open of bar i to bar j under `rule`. Returns (net, gross, open loss)."""
    E = bk.ao[i] + slip(bk, i)
    stop = -np.inf
    if "sl_w" in rule:
        stop = E - rule["sl_w"] * W
    if "sl_bp" in rule:
        stop = E * (1 - rule["sl_bp"] * 1e-4)
    tp = E + rule["tp_w"] * W if "tp_w" in rule else np.inf
    trail = rule.get("trail_w")
    be = E + rule["be_w"] * W if "be_w" in rule else None
    hi = -np.inf
    X, m, at_tp = None, j, False
    for q in range(i, j):
        if kcut is not None and q == kcut and bk.mo[q] - E < -rule["cut_g"] * W:
            X, m = bk.bo[q] - slip(bk, q), q
            break
        if bk.bl[q] <= stop:
            X, m = (min(stop, bk.bo[q]) if q > i else stop) - slip(bk, q), q
            break
        if q > i and bk.bh[q] >= tp:
            X, m, at_tp = tp, q, True
            break
        hi = max(hi, bk.bh[q])
        if trail is not None:
            stop = max(stop, hi - trail * W)
        if be is not None and bk.bh[q] >= be:
            stop = max(stop, E)
    if X is None:
        X = bk.bo[j] - slip(bk, j)
    net = X / E - 1.0 - 2 * bk.comm + swap_ret(bk, i, m, +1, E)
    e_mid = E - slip(bk, i) - 0.5 * (bk.ao[i] - bk.bo[i])
    x_mid = (X if at_tp else X + slip(bk, m)) + 0.5 * (bk.ao[m] - bk.bo[m])
    mae = min(float(np.min(bk.bl[i:m + 1]) / E - 1.0), net, 0.0)
    return float(net), float(x_mid / e_mid - 1.0), float(mae)


def window_units(bk, G):
    wr = {}
    for d in G:
        i, j = F.window(bk, d)
        if i >= 0:
            wr[d] = float(np.max(bk.mh[i:j + 1]) - np.min(bk.ml[i:j + 1]))
    return pd.Series(wr).sort_index().rolling(20, min_periods=10).mean().shift(1)


def run_pair(bk, G, W):
    out = {name: {} for name, _ in variants()}
    for d in G:
        w = float(W.get(d, np.nan))
        if not np.isfinite(w) or w <= 0:
            continue
        t0, t1 = U.jst(d - DAY, 23, 0), U.jst(d, 9, 53)
        i, j = bk.at(t0, 10), bk.at(t1, 65)
        if i < 0 or j <= i:
            continue
        if not (np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all()):
            continue
        for name, rule in variants():
            kc = None
            if "cut_h" in rule:
                kc = bk.at(U.jst(d, rule["cut_h"], 0), 10)
                if kc < 0 or kc <= i or kc >= j:
                    kc = None
            out[name][d] = exit_sim(bk, i, j, w, rule, kc)
    return {k: pd.DataFrame.from_dict(v, orient="index", columns=["net", "gross", "mae"]).sort_index()
            for k, v in out.items()}


def hedged(bkA, bkB, G, beta):
    """Long A, short beta x B over the same window, no stop. Open loss: A's bid low against B's ask high in the
    same minute (conservative)."""
    out = {}
    for d in G:
        t0, t1 = U.jst(d - DAY, 23, 0), U.jst(d, 9, 53)
        ia, ja = bkA.at(t0, 10), bkA.at(t1, 65)
        ib, jb = bkB.at(t0, 10), bkB.at(t1, 65)
        if min(ia, ja, ib, jb) < 0 or ja <= ia or jb <= ib:
            continue
        ra, rb = bkA.trade_ij(ia, ja, +1), bkB.trade_ij(ib, jb, -1)
        if ra is None or rb is None:
            continue
        na, nb = U.net_of(ra), U.net_of(rb)
        Ea = bkA.ao[ia] + slip(bkA, ia)
        Eb = bkB.bo[ib] - slip(bkB, ib)
        ta = bkA.idx[ia:ja]
        qb = np.clip(bkB.idx.searchsorted(ta, side="right") - 1, ib, jb)
        path = (bkA.bl[ia:ja] / Ea - 1.0) - beta * (bkB.ah[qb] / Eb - 1.0)
        net = na + beta * nb
        out[d] = (net, ra[1] + beta * rb[1], float(min(np.min(path), net, 0.0)))
    return pd.DataFrame.from_dict(out, orient="index", columns=["net", "gross", "mae"]).sort_index()


def stats(f):
    x = f.net.dropna()
    if len(x) < 10:
        return {"n": int(len(x))}
    m = f.mae.reindex(x.index)
    q = m.quantile(0.05)
    cvar = float(m[m <= q].mean())
    win, loss = x[x > 0], x[x < 0]
    return {"n": int(len(x)), "mean_bp": float(x.mean() * 1e4), "t": float(x.mean() / x.std() * np.sqrt(len(x))),
            "win_rate": float((x > 0).mean()), "avg_win_bp": float(win.mean() * 1e4) if len(win) else np.nan,
            "avg_loss_bp": float(loss.mean() * 1e4) if len(loss) else np.nan, "worst_bp": float(x.min() * 1e4),
            "share_loss_gt_50bp": float((x < -0.005).mean()), "worst_open_loss_bp": float(m.min() * 1e4),
            "cvar5_open_loss_bp": cvar * 1e4, "score": float(x.mean() / -cvar) if cvar < 0 else np.nan}


def periods(name, pair, f, rows):
    f.index = pd.DatetimeIndex(f.index)
    for per, g in (("2007-2018 train", f[f.index <= TRAIN_END]), ("2019-2025 test", f[f.index > TRAIN_END]),
                   ("all", f)):
        rows.append({"pair": pair, "variant": name, "period": per, **stats(g)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, books, Gs, errors = [], {}, {}, {}
    for p in LONG + HEDGES:
        bk = U.book(p)
        if bk is not None:
            bk._hl()
            books[p] = bk
            Gs[p], _, _ = F.calendar(bk)
    for p in ("USA500IDXUSD",):
        bk = U.book(p)
        if bk is not None:
            bk._hl()
            books[p] = bk
    print({p: (str(b.idx[0].date()), str(b.idx[-1].date())) for p, b in books.items()}, flush=True)
    res = {}
    for p in LONG:
        if p not in books:
            continue
        W = window_units(books[p], Gs[p])
        res[p] = run_pair(books[p], Gs[p], W)
        for name, f in res[p].items():
            periods(name, p, f, rows)
            f.to_csv(os.path.join(a.out, f"trades_{p}_{name.replace(' ', '_').replace('/', '_')}.csv"))
        print(p, "exits done", flush=True)
    if "USDJPY" in res and "EURJPY" in res:
        for name in res["USDJPY"]:
            A, B = res["USDJPY"][name], res["EURJPY"][name]
            j = A.join(B, lsuffix="_u", rsuffix="_e", how="outer")
            bsk = pd.DataFrame({"net": j[["net_u", "net_e"]].mean(axis=1), "gross": j[["gross_u", "gross_e"]].mean(axis=1),
                                "mae": j[["mae_u", "mae_e"]].mean(axis=1)}).dropna()
            periods(name, "basket UJ+EJ", bsk, rows)
            bsk.to_csv(os.path.join(a.out, f"trades_basket_{name.replace(' ', '_').replace('/', '_')}.csv"))
    # PART 2: hedges
    for p in ("USDJPY", "EURJPY"):
        if p not in books:
            continue
        for h in HEDGES + ("USA500IDXUSD",):
            if h not in books:
                continue
            for beta in (0.5, 1.0):
                try:
                    f = hedged(books[p], books[h], Gs[p], beta)
                    periods(f"hedge short {beta} x {h}", p, f, rows)
                    f.to_csv(os.path.join(a.out, f"trades_{p}_hedge_{h}_{beta}.csv"))
                except Exception as ex:                                     # noqa: BLE001
                    errors[f"{p} {h} {beta}"] = repr(ex)[:300]
        print(p, "hedges done", flush=True)
    # PART 2: filters on the no-stop rule
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start="2006-01-01", end="2026-01-01", progress=False)["Close"]
        vix = (vix.iloc[:, 0] if isinstance(vix, pd.DataFrame) else vix)
        vix.index = pd.DatetimeIndex(vix.index).tz_localize(None)
    except Exception as ex:                                                 # noqa: BLE001
        vix = pd.Series(dtype=float)
        errors["vix"] = repr(ex)[:200]
    fomc = set()
    for y, s in FL.FOMC_FALLBACK.items():
        fomc |= {pd.Timestamp(f"{y}-{x}") for x in s.split()}
    for p in ("USDJPY", "EURJPY"):
        if p not in res:
            continue
        bk, base = books[p], res[p]["none"].copy()
        W = window_units(bk, Gs[p])
        keep_f = [d for d in base.index if (pd.Timestamp(d) - DAY) not in fomc]
        periods("filter: skip FOMC nights", p, base.loc[keep_f], rows)
        if len(vix):
            prev = vix.reindex(pd.DatetimeIndex(base.index) - DAY, method="ffill").to_numpy()
            periods("filter: skip when VIX > 25", p, base[prev <= 25], rows)
        mv = {}
        for d in base.index:
            t0 = U.jst(pd.Timestamp(d) - DAY, 23, 0)
            i, i0 = bk.at(t0, 10), bk.at(t0 - DAY, 30)
            w = float(W.get(d, np.nan))
            if i >= 0 and i0 >= 0 and np.isfinite(w) and w > 0:
                mv[d] = (bk.mo[i] - bk.mo[i0]) / w
        mv = pd.Series(mv)
        periods("filter: skip after a fall of more than 1 W", p, base.loc[mv.index[mv >= -1]], rows)
        periods("filter: skip after a rise of more than 1 W", p, base.loc[mv.index[mv <= 1]], rows)
    R = pd.DataFrame(rows)
    # selection on train: best score among variants keeping >= 75% of the no-stop train mean (per pair / basket)
    picks = {}
    for pair, g in R[R.period == "2007-2018 train"].groupby("pair"):
        g = g[~g.variant.str.startswith(("hedge", "filter"))]
        base = g.loc[g.variant == "none", "mean_bp"]
        if not len(base):
            continue
        ok = g[g.mean_bp >= 0.75 * float(base.iloc[0])]
        if len(ok):
            best = ok.sort_values("score", ascending=False).iloc[0]
            picks[pair] = {"variant": best.variant, "train_score": best.score, "train_mean_bp": best.mean_bp}
    R.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    json.dump({"picks": picks, "errors": errors}, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    cols = ["pair", "variant", "period", "n", "mean_bp", "t", "win_rate", "avg_win_bp", "avg_loss_bp", "worst_bp",
            "share_loss_gt_50bp", "worst_open_loss_bp", "cvar5_open_loss_bp", "score"]
    print(R[R.period != "all"][cols].round(3).to_string())
    print(json.dumps({"picks": picks, "errors": errors}, indent=1, default=str))


if __name__ == "__main__":
    main()
