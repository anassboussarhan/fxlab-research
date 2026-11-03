"""M2 of docs/edges_batch_preregistration.md (addendum 3): cost-aware meta-labels on the rules that earn before
costs. 2007-2025 only.

Rules (their own code, unchanged): night fade I7 (AUD/NZD, EUR/GBP, GBP/CAD; scripts/untested_intraday_lab.py),
gold London breakout I9 (same file), end-of-day momentum E14 (gold and 7 majors; scripts/edges_lab2.py) and round
numbers E18 (EUR/USD, GBP/USD, USD/JPY, gold; same file; E18's gross is the same exit on the mid path from the limit
price). For every trade, features known at its entry bar (M1):
  spread / mid; spread over its 100-bar median; the rule's own size (I7 band width, I9 Asian range, E14 the
  London->NY move, E18 the distance from the day's open to the limit) in units of 20-bar M5 volatility; hour and
  weekday (UTC); 20-bar M5 volatility; distance to the UTC day's high and low so far (in the same volatility units,
  signed by the trade's side); the side.
Model: LightGBM regression (8.5 settings) of the trade's gross mid return, one model per rule pooled over its
markets, refitted every year on trades before 1 January of the test year (first test year 2010). A trade is kept only
if the prediction exceeds 1.2 times its round-trip cost at entry (2 x (spread x (1/2 + slippage fraction) + fixed
slippage) / mid + 2 x commission). Counts if the kept trades' net > 0 with t >= 1.645 (month-clustered) on 2019-2025.
Usage: python scripts/ml_meta_lab.py --out DIR"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edges_lab2 as E  # noqa: E402
import flows_lab as FL  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402
from ml_weekly_lab import month_t, LGBR  # noqa: E402

FEATS = ["spread", "spread_rel", "size", "hour", "wday", "vol20", "dist_hi", "dist_lo", "side"]


def capture(bk):
    """Record (entry bar, side, stop distance) of every bracket call that returned a trade."""
    log, orig = [], bk.bracket

    def br(i, j, d, tp, sl):
        r = orig(i, j, d, tp, sl)
        if r is not None:
            log.append((int(i), int(d), float(sl)))
        return r
    bk.bracket = br
    return log


class Feat:
    def __init__(self, bk):
        self.bk = bk
        mid = 0.5 * (bk.bo + bk.ao)
        self.mid = mid
        s = pd.Series((bk.ao - bk.bo) / mid, index=bk.idx)
        self.spread = s.to_numpy()
        self.sp_med = s.rolling(100, min_periods=50).median().shift(1).to_numpy()
        m5 = pd.Series(mid, index=bk.idx).resample("5min", label="left", closed="left").last().dropna()
        v = np.log(m5).diff().rolling(20, min_periods=15).std().shift(1)    # completed M5 bars only
        self.vol = v.reindex(bk.idx, method="ffill").to_numpy()
        day = bk.idx.floor("D")
        g = pd.Series(mid, index=bk.idx).groupby(day)
        self.dhi = g.cummax().shift(1).to_numpy()
        self.dlo = g.cummin().shift(1).to_numpy()
        self.day = day.to_numpy()

    def at(self, i, side, size):
        bk = self.bk
        if i <= 0 or i >= len(bk.idx):
            return None
        m, v = self.mid[i], self.vol[i]
        if not (np.isfinite(v) and v > 0):
            return None
        same = self.day[i] == self.day[i - 1]
        hi, lo = (self.dhi[i], self.dlo[i]) if same else (m, m)
        u = m * v
        c = 2 * (bk.sf * (bk.ao[i] - bk.bo[i]) + bk.sp * bk.pip) / m + 2 * bk.comm
        dh, dl = (hi - m) / u, (m - lo) / u
        return {"spread": self.spread[i], "spread_rel": self.spread[i] / self.sp_med[i] if self.sp_med[i] > 0 else np.nan,
                "size": size / u, "hour": bk.idx[i].hour, "wday": bk.idx[i].weekday(), "vol20": v,
                "dist_hi": dh if side > 0 else dl, "dist_lo": dl if side > 0 else dh, "side": side, "cost": c}


def from_brackets(rule, sym, bk, F, frame, log):
    rows = []
    if len(frame) != len(log):
        print(rule, sym, "frame/log mismatch", len(frame), len(log), flush=True)
        return rows
    for (t, r), (i, d, sl) in zip(frame.iterrows(), log):
        f = F.at(i, d, sl)
        if f is None:
            continue
        rows.append({"rule": rule, "market": sym, "time": bk.idx[i], "net": r.net, "gross": r.gross, **f})
    return rows


def i7(sym, bk, F):
    log = capture(bk)
    fr = U.night_scalp(sym, bk=bk)
    bk.bracket = type(bk).bracket.__get__(bk)
    return from_brackets("I7 night fade", sym, bk, F, fr, log) if fr is not None else []


def i9(bk, F):
    log = capture(bk)
    res = {}
    U.gold_london(res, bk=bk)
    bk.bracket = type(bk).bracket.__get__(bk)
    fr = res["I9 XAUUSD London breakout of the Asian range"][0]
    return from_brackets("I9 gold London breakout", "XAUUSD", bk, F, fr, log)


def e14(sym, bk, F):
    gold = sym == "XAUUSD"
    tr = ("America/New_York", 13, 0) if gold else ("America/New_York", 14, 30)
    te = ("America/New_York", 13, 30) if gold else ("America/New_York", 15, 0)
    mv, ev, ent = {}, {}, {}
    for d in E.us_bdays("2007-01-01", "2025-12-31"):
        t0 = FL.local_ts(d, "Europe/London", 8, 0)
        t1 = FL.local_ts(d, *tr)
        a, b = E.mid_at(bk, t0, 5), E.mid_at(bk, t1, 5)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        r = E.both(bk, t1, FL.local_ts(d, *te))
        if r is None:
            continue
        mv[d], ev[d], ent[d] = b / a - 1, r, bk.at(t1, 5)
    M = pd.Series(mv)
    cut = float(M[M.index <= E.TRAIN_END].abs().quantile(0.8))
    rows = []
    for d, m in M[M.abs() > cut].items():
        side = 1 if m > 0 else -1
        i = ent[d]
        f = F.at(i, side, abs(m) * F.mid[i])
        if f is None:
            continue
        r = ev[d]
        rows.append({"rule": "E14 end-of-day momentum", "market": sym, "time": bk.idx[i],
                     "net": r["long"] if side > 0 else r["short"], "gross": side * r["gross"], **f})
    return rows


def e18(sym, bk, F):
    pip = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 0.01}.get(sym)
    if pip is None:
        step, off, tgt = 50.0, 2.0, 5.0
    else:
        step, off, tgt = 100 * pip, 2 * pip, 10 * pip
    idx = bk.idx
    rows = []
    days = pd.bdate_range(str(idx[0].date()), str(min(idx[-1], pd.Timestamp(E.END).tz_localize("UTC")).date()))
    for d in days:
        a, e = int(idx.searchsorted(E.utc(d, 0))), int(idx.searchsorted(E.utc(d, 21)))
        if e - a < 300 or e >= len(idx):
            continue
        o = 0.5 * (bk.bo[a] + bk.ao[a])
        for side in (+1, -1):
            if side > 0:
                lim = np.floor(o / step) * step + off
                if bk.ao[a] <= lim:
                    continue
                f = np.flatnonzero(bk.al[a + 1:e] <= lim)
            else:
                lim = np.ceil(o / step) * step - off
                if bk.bo[a] >= lim:
                    continue
                f = np.flatnonzero(bk.bh[a + 1:e] >= lim)
            if not len(f):
                continue
            k = a + 1 + int(f[0])
            Ep = min(lim, bk.ao[k]) if side > 0 else max(lim, bk.bo[k])
            T, S = Ep + side * tgt, Ep - side * tgt
            X, Xm = None, None
            for q in range(k, e):
                if side > 0:
                    if bk.bl[q] <= S:
                        X = min(S, bk.bo[q]) - R.slip(bk, q) if q > k else S - R.slip(bk, q)
                        Xm = S
                        break
                    if bk.bh[q] >= T and q > k:
                        X, Xm = T, T
                        break
                else:
                    if bk.ah[q] >= S:
                        X = max(S, bk.ao[q]) + R.slip(bk, q) if q > k else S + R.slip(bk, q)
                        Xm = S
                        break
                    if bk.al[q] <= T and q > k:
                        X, Xm = T, T
                        break
            if X is None:
                X = bk.bo[e] - R.slip(bk, e) if side > 0 else bk.ao[e] + R.slip(bk, e)
                Xm = 0.5 * (bk.bo[e] + bk.ao[e])
            net = side * (X / Ep - 1.0) - 2 * bk.comm
            ft = F.at(k, side, abs(o - lim))
            if ft is None:
                continue
            rows.append({"rule": "E18 round numbers", "market": sym, "time": idx[k], "net": net,
                         "gross": side * (Xm / lim - 1.0), **ft})
    return rows


def model(T, out):
    import lightgbm as lgb
    T = T.sort_values("time").reset_index(drop=True)
    T["time"] = pd.to_datetime(T.time, utc=True)
    T["market_id"] = T.market.astype("category").cat.codes
    feats = FEATS + ["market_id"]
    res, kept_all = {}, []
    for rule, g in T.groupby("rule"):
        g = g.reset_index(drop=True)
        yr = g.time.dt.year.to_numpy()
        X, y = g[feats].to_numpy(np.float32), g.gross.to_numpy()
        pred = np.full(len(g), np.nan)
        for Y in range(2010, 2026):
            tr, te = yr < Y, np.flatnonzero(yr == Y)
            if tr.sum() < 300 or not len(te):
                continue
            m = lgb.LGBMRegressor(**{**LGBR, "min_child_samples": min(200, max(20, int(tr.sum() // 50)))}).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        g["pred"] = pred
        g["keep"] = g.pred > 1.2 * g.cost
        r = {}
        for per, msk in (("2010-2018", (yr >= 2010) & (yr <= 2018)), ("2019-2025", yr >= 2019)):
            h = g[msk & np.isfinite(pred)]
            k = h[h.keep]
            r[per] = {"all_trades": int(len(h)), "all_net_bp": float(h.net.mean() * 1e4) if len(h) else None,
                      "all_gross_bp": float(h.gross.mean() * 1e4) if len(h) else None,
                      "kept": int(len(k)), "kept_net_bp": float(k.net.mean() * 1e4) if len(k) else None,
                      "kept_gross_bp": float(k.gross.mean() * 1e4) if len(k) else None,
                      "kept_t": month_t(k.set_index("time").net) if len(k) >= 10 else None,
                      "pred_gross_corr": float(np.corrcoef(h.pred, h.gross)[0, 1]) if len(h) > 10 else None}
        c = r["2019-2025"]
        r["counts"] = bool((c["kept_net_bp"] or -1) > 0 and (c["kept_t"] or 0) >= 1.645)
        res[rule] = r
        kept_all.append(g)
    pd.concat(kept_all).to_csv(os.path.join(out, "m2_trades.csv.gz"), index=False)
    json.dump(res, open(os.path.join(out, "m2.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.table:
        T = pd.read_csv(a.table)
    else:
        plan = {"AUDNZD": ("i7",), "EURGBP": ("i7",), "GBPCAD": ("i7",), "XAUUSD": ("i9", "e14", "e18"),
                "EURUSD": ("e14", "e18"), "GBPUSD": ("e14", "e18"), "USDJPY": ("e14", "e18"), "USDCHF": ("e14",),
                "USDCAD": ("e14",), "AUDUSD": ("e14",), "NZDUSD": ("e14",)}
        if a.only:
            plan = {k: v for k, v in plan.items() if k in a.only.split(",")}
        rows = []
        for sym, rules in plan.items():
            bk = E.book(sym)
            if bk is None:
                print("no data", sym, flush=True)
                continue
            bk._hl()
            F = Feat(bk)
            for r in rules:
                try:
                    x = {"i7": lambda: i7(sym, bk, F), "i9": lambda: i9(bk, F), "e14": lambda: e14(sym, bk, F),
                         "e18": lambda: e18(sym, bk, F)}[r]()
                    rows += x
                    print(sym, r, len(x), flush=True)
                except Exception as ex:                                     # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(sym, r, "error", repr(ex)[:200], flush=True)
            E.BOOKS.clear()
            del bk, F
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(a.out, "m2_table.csv.gz"), index=False)
    model(T, a.out)


if __name__ == "__main__":
    main()
