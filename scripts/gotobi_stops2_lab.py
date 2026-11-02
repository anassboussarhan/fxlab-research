"""Gotobi overnight, third stop round: the stop families not yet tried. ATR stops, chandelier (ATR trailing),
swing-low and round-number stops, moving-average, Donchian and Parabolic-SAR exits, a stop on the whole basket, and
stops confirmed by a correlated pair. The trade: long USD/JPY and EUR/JPY 23:00 Tokyo the evening before -> 09:53
Tokyo on Gotobi days (the frozen trade). M1 bid/ask fills and costs as in scripts/gotobi_accuracy_lab.py.
Development data only (to 2025-12-31). Chosen on 2007-2018, reported on 2019-2025.

SIGNALS come from USD/JPY (full M1 history in both periods; EUR/JPY's 2019-2025 bars cover only the trade's hours, so
it has no pre-entry history there). Series "USDJPY": the rule on USD/JPY. Series "basket": both legs leave when the
rule fires (for the joint rules: the basket's own mark-to-market, or both legs / the confirming pair).
UNITS known at entry: ATR = USD/JPY's daily ATR 14 (daily bars cut at 14:00 UTC) in % of price, applied to each
leg; H1 ATR the same on 1-hour bars; W = each pair's mean 23:00-09:53 range over the previous 20 Gotobi days.
RULES:
  ATR stop k x daily ATR, k = 0.25, 0.5, 0.75, 1.0, 1.5, 2.0; k x 1-hour ATR, k = 2, 3, 4, 6
  chandelier: highest bid since entry minus k x daily ATR (initial stop entry minus k ATR), k = 0.5, 1.0, 1.5, 2.0
  swing low: below the lowest mid of the 24 hours / the London session (07:00-14:00 UTC) / the 4 hours before entry,
    buffer 0 or 0.25 ATR (at least 10 bp below entry)
  round number: 10 pips below the whole yen under the entry (at most 2 W below)
  moving averages: leave at the first 15-minute close below its SMA 20 / SMA 50 / EMA 100, or the first 1-hour
    close below its SMA 20 / EMA 50
  Donchian: first 15-minute close below the lowest low of the previous 16 / 32 bars; first 1-hour close below the
    previous 12 bars' low
  Parabolic SAR (0.02, 0.2) on 15-minute bars: leave at the first sell flip after entry
  basket stop: leave both legs when the basket's mark-to-market falls to -40 / -60 / -80 / -100 bp or -0.75 / -1.0 /
    -1.5 W (mean of the two pairs' W)
  confirmed stops: USD/JPY below -k W AND AUD/JPY (the yen risk barometer) below -k of its own W at the same minute,
    k = 0.75, 1.0, 1.5; and (basket only) both USD/JPY and EUR/JPY below -k W, k = 0.75, 1.0, 1.5
  references: stop 80 bp (frozen S1), stop 0.75 W
  Close-based exits fill at the next minute's bid; resting stops at the stop (or the open if it gaps through).
PICKS on the basket, 2007-2018, among rules keeping at least 90% of the no-stop mean per night (95% for P3):
  P1 the highest mean / -CVaR5 of the open loss; P2 the highest share of winners; P3 the smallest worst trade.
  Each rule against no stop on the same nights: paired difference, t, Benjamini-Hochberg on train.
Usage: python scripts/gotobi_stops2_lab.py --out DIR"""
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
import gotobi_accuracy_lab as A  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import gotobi_ta_lab as T  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

U.PIPS.update({"AUDJPY": 0.01})
MIN15 = pd.Timedelta(minutes=15)


def variants():
    V = [("none", {}), ("stop 80 bp", {"kind": "bp", "x": 80}), ("stop 0.75 W", {"kind": "w", "k": 0.75})]
    V += [(f"ATR stop {k} x daily ATR", {"kind": "atr", "k": k}) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)]
    V += [(f"ATR stop {k} x 1-hour ATR", {"kind": "atr_h1", "k": k}) for k in (2, 3, 4, 6)]
    V += [(f"chandelier {k} x daily ATR", {"kind": "chandelier", "k": k}) for k in (0.5, 1.0, 1.5, 2.0)]
    V += [(f"swing low {w}, buffer {b} ATR", {"kind": "swing", "win": w, "b": b})
          for w in ("24h", "London", "4h") for b in (0.0, 0.25)]
    V += [("round number below", {"kind": "round"})]
    V += [(f"15-min close below {m}", {"kind": "ma15", "ma": m}) for m in ("SMA20", "SMA50", "EMA100")]
    V += [(f"1-hour close below {m}", {"kind": "ma60", "ma": m}) for m in ("SMA20", "EMA50")]
    V += [(f"15-min Donchian {n}", {"kind": "don15", "n": n}) for n in (16, 32)]
    V += [("1-hour Donchian 12", {"kind": "don60", "n": 12}), ("15-min Parabolic SAR flip", {"kind": "psar"})]
    V += [(f"basket stop {x} bp", {"kind": "bmtm_bp", "x": x, "basket_only": True}) for x in (40, 60, 80, 100)]
    V += [(f"basket stop {k} W", {"kind": "bmtm_w", "k": k, "basket_only": True}) for k in (0.75, 1.0, 1.5)]
    V += [(f"stop {k} W confirmed by AUD/JPY", {"kind": "conf_aj", "k": k}) for k in (0.75, 1.0, 1.5)]
    V += [(f"stop {k} W on both legs", {"kind": "conf_both", "k": k, "basket_only": True}) for k in (0.75, 1.0, 1.5)]
    return V


# ------------------------------------------------------------------------------------------------ signal series
def signal_series(bk):
    """USD/JPY daily / 1-hour ATR (% of price) and 15-minute / 1-hour close-based exit signals, by bar end time."""
    d = T.bars(bk, "1D", "14h")
    h1 = T.bars(bk, "1h", "0h")
    m = bk.m
    mid = pd.DataFrame({"o": 0.5 * (m.bid_o + m.ask_o), "h": 0.5 * (m.bid_h + m.ask_h),
                        "l": 0.5 * (m.bid_l + m.ask_l), "c": 0.5 * (m.bid_c + m.ask_c)}, index=m.index)
    g = mid.resample("15min", origin="epoch", closed="left", label="right")
    m15 = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "n": g.c.count()})
    m15 = m15[m15.n >= 5].drop(columns="n")
    S = {"atr_d": (T.atr(d) / d.c), "atr_h1": (T.atr(h1) / h1.c)}
    c = m15.c
    S["ma15"] = {"SMA20": c < c.rolling(20).mean(), "SMA50": c < c.rolling(50).mean(),
                 "EMA100": c < c.ewm(span=100, adjust=False).mean()}
    c1 = h1.c
    S["ma60"] = {"SMA20": c1 < c1.rolling(20).mean(), "EMA50": c1 < c1.ewm(span=50, adjust=False).mean()}
    S["don15"] = {n: c < m15.l.rolling(n).min().shift(1) for n in (16, 32)}
    S["don60"] = {12: c1 < h1.l.rolling(12).min().shift(1)}
    ps = T.psar(m15)
    S["psar"] = (ps < 0) & (ps.shift(1) > 0)
    return S


def asof(series, t, max_age):
    k = series.index.searchsorted(t, side="right") - 1
    if k < 0 or (t - series.index[k]) > max_age:
        return np.nan
    return float(series.iloc[k])


# ------------------------------------------------------------------------------------------------ exits
def entry(bk, i):
    return bk.ao[i] + R.slip(bk, i)


def settle(bk, i, j, q, fill=None):
    """Long from bar i, out at bar q (q >= j: the time exit at j). fill: a resting stop's price."""
    E = entry(bk, i)
    stopped = q < j
    q = min(q, j)
    if fill is None or not stopped:
        X = bk.bo[q] - R.slip(bk, q)
    else:
        X = fill - R.slip(bk, q)
    net = X / E - 1.0 - 2 * bk.comm + R.swap_ret(bk, i, q, +1, E)
    lo = float(np.min(bk.bl[i:q]) / E - 1.0) if q > i else 0.0
    return {"net": float(net), "mae": float(min(lo, X / E - 1.0, net, 0.0)), "stopped": bool(stopped), "target": False}


def level_hit(bk, i, j, S):
    """First bar at or after i whose bid low reaches the resting stop S, and its fill."""
    k = np.flatnonzero(bk.bl[i:j] <= S)
    if not len(k):
        return j, None
    q = i + int(k[0])
    return q, (min(S, bk.bo[q]) if q > i else S)


def chandelier(bk, i, j, d):
    E = entry(bk, i)
    rm = np.maximum.accumulate(bk.bh[i:j])
    lev = np.empty(j - i)
    lev[0] = E - d
    lev[1:] = np.maximum(E - d, rm[:-1] - d)
    k = np.flatnonzero(bk.bl[i:j] <= lev)
    if not len(k):
        return j, None
    q = i + int(k[0])
    return q, (min(lev[q - i], bk.bo[q]) if q > i else lev[0])


def first_signal(bk, sig, i, j):
    """First bar-close signal strictly after entry and no later than the exit; returns the minute to leave at."""
    t0, t1 = bk.idx[i], bk.idx[j]
    s = sig[(sig.index > t0) & (sig.index <= t1)]
    s = s[s.astype(bool)]
    if not len(s):
        return j
    return int(min(bk.idx.searchsorted(s.index[0]), j))


def map_bars(src, dst, a, b, lo, hi):
    """For src bars a..b-1, the index of the last dst bar at or before each time, clipped to [lo, hi]."""
    k = dst.idx.searchsorted(src.idx[a:b], side="right") - 1
    return np.clip(k, lo, hi)


# ------------------------------------------------------------------------------------------------ one night
def night(books, Ws, d, sig, V):
    bu, be, ba = books["USDJPY"], books.get("EURJPY"), books.get("AUDJPY")
    iu, ju = F.window(bu, d)
    if iu < 0:
        return None
    wu = float(Ws["USDJPY"].get(d, np.nan))
    if not np.isfinite(wu) or wu <= 0 or not np.isfinite([bu.bo[iu], bu.ao[iu], bu.bo[ju], bu.ao[ju]]).all():
        return None
    ie, je = F.window(be, d) if be is not None else (-1, -1)
    we = float(Ws["EURJPY"].get(d, np.nan)) if be is not None else np.nan
    has_e = ie >= 0 and np.isfinite(we) and we > 0 and np.isfinite([be.bo[ie], be.ao[ie], be.bo[je], be.ao[je]]).all()
    t0 = bu.idx[iu]
    atr = asof(sig["atr_d"], t0.floor("h"), pd.Timedelta(hours=1))
    atr1 = asof(sig["atr_h1"], t0.floor("h"), pd.Timedelta(hours=1))
    Eu = entry(bu, iu)
    out = {"USDJPY": {}, "basket": {}}

    def both(q_u, fill_u=None):
        """UJ leaves at bar q_u (at fill_u for a resting stop); EJ at market at the same moment."""
        ru = settle(bu, iu, ju, q_u, fill_u)
        if not has_e:
            return ru, None
        qe = je if q_u >= ju else int(min(max(be.idx.searchsorted(bu.idx[q_u]), ie), je))
        return ru, settle(be, ie, je, qe)

    for name, r in V:
        kind = r.get("kind")
        res_u = res_b = None
        if kind is None:
            ru = settle(bu, iu, ju, ju)
            re_ = settle(be, ie, je, je) if has_e else None
            res_u, res_b = ru, (ru, re_)
        elif kind in ("bp", "w", "atr", "atr_h1", "swing", "round"):
            if kind == "bp":
                dist = Eu * r["x"] * 1e-4
            elif kind == "w":
                dist = r["k"] * wu
            elif kind == "atr":
                dist = r["k"] * atr * Eu if np.isfinite(atr) else np.nan
            elif kind == "atr_h1":
                dist = r["k"] * atr1 * Eu if np.isfinite(atr1) else np.nan
            elif kind == "swing":
                hrs = {"24h": (24, 0), "4h": (4, 0)}
                if r["win"] == "London":
                    a = int(bu.idx.searchsorted(t0.normalize() + pd.Timedelta(hours=7)))
                else:
                    a = int(bu.idx.searchsorted(t0 - pd.Timedelta(hours=hrs[r["win"]][0])))
                lo = float(np.min(bu.ml[a:iu])) if iu - a > 30 else np.nan
                lev = lo - r["b"] * atr * Eu if np.isfinite(lo) and np.isfinite(atr) else np.nan
                dist = max(Eu - lev, Eu * 0.001) if np.isfinite(lev) else np.nan
            else:
                lev = np.floor(Eu) - 0.10
                dist = min(Eu - lev, 2 * wu)
            if not np.isfinite(dist):
                continue
            S = Eu - dist
            q, fill = level_hit(bu, iu, ju, S)
            res_u = settle(bu, iu, ju, q, fill)
            if has_e:                                                    # EJ: same distance in % from its entry
                Ee = entry(be, ie)
                if kind == "w":
                    Se = Ee - r["k"] * we
                else:
                    Se = Ee * (1 - dist / Eu)
                if kind in ("swing", "round"):                           # USD/JPY's level drives both legs
                    res_b = both(q, fill)
                else:
                    qe, fe = level_hit(be, ie, je, Se)
                    res_b = (res_u, settle(be, ie, je, qe, fe))
            else:
                res_b = (res_u, None)
        elif kind == "chandelier":
            if not np.isfinite(atr):
                continue
            q, fill = chandelier(bu, iu, ju, r["k"] * atr * Eu)
            res_u = settle(bu, iu, ju, q, fill)
            if has_e:
                qe, fe = chandelier(be, ie, je, r["k"] * atr * entry(be, ie))
                res_b = (res_u, settle(be, ie, je, qe, fe))
            else:
                res_b = (res_u, None)
        elif kind in ("ma15", "ma60", "don15", "don60", "psar"):
            s = sig["psar"] if kind == "psar" else sig[kind][r.get("ma", r.get("n"))]
            q = first_signal(bu, s, iu, ju)
            res_u = settle(bu, iu, ju, q)
            res_b = both(q)
        elif kind in ("bmtm_bp", "bmtm_w", "conf_both"):
            if not has_e:
                continue
            Ee = entry(be, ie)
            ke = map_bars(bu, be, iu, ju, ie, je)
            mu = bu.bl[iu:ju] / Eu - 1.0
            me = be.bl[ke] / Ee - 1.0
            if kind == "bmtm_bp":
                hit = 0.5 * (mu + me) <= -r["x"] * 1e-4
            elif kind == "bmtm_w":
                hit = 0.5 * (mu + me) <= -r["k"] * 0.5 * (wu / Eu + we / Ee)
            else:
                hit = (mu <= -r["k"] * wu / Eu) & (me <= -r["k"] * we / Ee)
            k = np.flatnonzero(hit)
            q = iu + int(k[0]) + 1 if len(k) else ju
            res_b = both(min(q, ju))
        elif kind == "conf_aj":
            if ba is None:
                continue
            ia, ja = F.window(ba, d)
            wa = float(Ws["AUDJPY"].get(d, np.nan))
            if ia < 0 or not np.isfinite(wa) or wa <= 0:
                continue
            ka = map_bars(bu, ba, iu, ju, ia, ja)
            pa0 = ba.mo[ia]
            hit = (bu.bl[iu:ju] <= Eu - r["k"] * wu) & (ba.ml[ka] <= pa0 - r["k"] * wa)
            k = np.flatnonzero(hit)
            q = iu + int(k[0]) + 1 if len(k) else ju
            res_u = settle(bu, iu, ju, min(q, ju))
            res_b = both(min(q, ju))
        if res_u is not None and not r.get("basket_only"):
            out["USDJPY"][name] = res_u
        if res_b is not None:
            a_, b_ = res_b
            if b_ is None:
                out["basket"][name] = a_
            else:
                out["basket"][name] = {"net": 0.5 * (a_["net"] + b_["net"]), "mae": 0.5 * (a_["mae"] + b_["mae"]),
                                       "stopped": float(a_["stopped"] or b_["stopped"]), "target": 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    books, Ws, Gs, errors, info = {}, {}, {}, {}, {}
    for p in ("USDJPY", "EURJPY", "AUDJPY"):
        bk = U.book(p)
        if bk is None:
            errors[p] = "no data"
            continue
        bk._hl()
        G, _, _ = F.calendar(bk)
        books[p], Gs[p], Ws[p] = bk, G, R.window_units(bk, G)
        info[p] = [str(bk.idx[0].date()), str(bk.idx[-1].date())]
    sig = signal_series(books["USDJPY"])
    V = variants()
    rows = {"USDJPY": {n: {} for n, _ in V}, "basket": {n: {} for n, _ in V}}
    for d in Gs["USDJPY"]:
        try:
            o = night(books, Ws, d, sig, V)
        except Exception as ex:                                             # noqa: BLE001
            errors.setdefault("nights", []).append(f"{d.date()}: {repr(ex)[:120]}")
            continue
        if o is None:
            continue
        for s in o:
            for n, r in o[s].items():
                rows[s][n][d] = r
    E = []
    frames = {}
    for s in rows:
        fr = {n: pd.DataFrame.from_dict(v, orient="index").sort_index() for n, v in rows[s].items() if v}
        for f in fr.values():
            f.index = pd.DatetimeIndex(f.index)
        frames[s] = fr
        A.exit_rows(s, fr, E)
        for n in ("none", "stop 80 bp", "basket stop 80 bp", "chandelier 1.0 x daily ATR"):
            if n in fr:
                fr[n].to_csv(os.path.join(a.out, f"trades_{s}_{n.replace(' ', '_').replace('/', '_')}.csv"))
    E = pd.DataFrame(E)
    tr = E.period == "2007-2018 train"
    E["q_diff_train"] = np.nan
    from scipy import stats as st
    for s in E.series.unique():
        m = tr & (E.series == s) & (E.variant != "none")
        E.loc[m, "q_diff_train"] = A.bh(2 * st.norm.sf(np.abs(E.loc[m, "t_diff"].to_numpy(float))))
    E.to_csv(os.path.join(a.out, "exits.csv"), index=False)
    touch = []
    for s, fr in frames.items():
        A.touch_rows(s, fr, touch)
    pd.DataFrame(touch).to_csv(os.path.join(a.out, "touch.csv"), index=False)
    pk = A.picks(E)
    for k, v in pk.items():
        te = E[(E.series == "basket") & (E.variant == v["variant"]) & (E.period == "2019-2025 test")]
        if len(te):
            v["test"] = te.iloc[0][["mean_per_day_bp", "t", "win_rate", "worst_bp", "cvar5_open_loss_bp", "score",
                                    "diff_vs_none_bp", "t_diff"]].to_dict()
    json.dump({"picks": pk, "info": info, "errors": errors}, open(os.path.join(a.out, "summary.json"), "w"), indent=1,
              default=str)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    cols = ["series", "variant", "period", "n", "mean_per_day_bp", "t", "win_rate", "worst_bp", "cvar5_open_loss_bp",
            "score", "diff_vs_none_bp", "t_diff", "q_diff_train", "share_stopped"]
    print(E[E.period != "all"][cols].round(3).to_string())
    print(json.dumps({"picks": pk, "info": info, "errors": errors}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
