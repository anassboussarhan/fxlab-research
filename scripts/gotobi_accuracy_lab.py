"""Gotobi overnight, second risk round: more stops and targets, what the trade's winners and losers have in common,
and whether any of it raises the share of winners without cutting the gain. Long USD/JPY and EUR/JPY 23:00 Tokyo
the evening before -> 09:53 Tokyo on Gotobi days (the frozen trade). M1 bid/ask fills and costs as in
scripts/gotobi_risk_lab.py. Development data only (to 2025-12-31); 2026 is not loaded.

SELECTION PROTOCOL (fixed before any result): every rule is run on 2007-2025, chosen on 2007-2018 ("train") only
and reported on 2019-2025 ("test"). Each variant is compared with no stop on the same days (paired difference,
t); Benjamini-Hochberg over the variants on train. Three picks on the basket, each among variants whose train mean
per Gotobi day is at least 90% of the no-stop rule's (the 2026 check showed a 75% floor let a stop through that
lost the gain):
  P1 cut losses: the highest mean / -CVaR5 of the open loss;
  P2 accuracy:   the highest share of winning trades;
  P3 worst case: the smallest worst trade (requires 95% of the no-stop mean).

PART A, EXITS AND ENTRIES (W = mean range of the 23:00-09:53 window over the previous 20 Gotobi days):
  stops k W, k = 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0; stops x bp, x = 50, 60, 70, 80, 90, 100, 120,
    150, 200
  stops on a 15-minute close (exit at the next minute's bid): 0.75, 1.0, 1.5, 2.0 W; 60, 100 bp
  stops active only from 03:00 or 05:00 Tokyo (1.0, 1.5 W), or only until 05:00 or 07:00 (1.0, 1.5 W)
  stop then re-entry: stop 0.5 / 0.75 / 1.0 W, then buy again when the ask comes back to the entry price (until
    09:00), or buy at 08:00 if stopped before then; held to 09:53 without a stop
  targets without a stop: 0.5, 0.75, 1.0, 1.5, 2.0 W; 15, 25, 40 bp; targets active only from 08:00: 0.25, 0.5,
    1.0 W
  half at a target (+0.5 W, +1.0 W), the rest to 09:53; the same with a 100 bp stop
  100 bp stop + target 1.0 / 1.5 / 2.0 W
  scale-in: half at 23:00, half by a limit 0.5 or 1.0 W lower (until 08:00); with and without a 100 bp stop
  dip entry: the whole position by a limit 0.25 or 0.5 W below the 23:00 ask (until 07:00), else no trade
PART B, DIAGNOSTICS (no choice is made from them):
  B1 for each stop: of the trades it stopped, how many would have won without it, and the mean of (stopped net -
     held net) on them; the same for each target
  B2 the rest of the trade: at 01:00, 03:00, 05:00, 07:00, 08:00 and 09:00 Tokyo, the mean mid move from there to
     the exit, by how far the trade stands at that moment (in W)
  B3 does a stop help more in some regimes: (stop - no stop) by tercile of W / its yearly median and of the VIX
PART C, WHAT GOES WITH WINNING NIGHTS (features known at 23:00 Tokyo; US closes up to the day before):
  W / its median over the previous 60 Gotobi days; the pair's return over the previous 24 hours, 1 week and 4
  weeks (in W); its move since the previous Tokyo fix (09:55 -> 23:00); distance below the 4-week high (in W);
  distance from its 200-day mean; the previous night's range (in W); spread at entry / its recent median; the
  previous Gotobi trade's result; VIX and its 1-week change; S&P 500 and Nikkei 225 previous-day returns; US 10-year
  yield 1-week change; month-end, 10th, 25th, moved by a weekend or holiday, weekday, quarter-end month, March,
  days off before the Gotobi day; speculators' JPY futures positioning (z over 156 weeks; published reports only).
  Tests: Spearman correlation with the net return on train, BH over the features per series; the same on test;
  tercile spread (terciles cut on train). One model each (logistic regression, L2, C = 0.1; gradient boosting,
  depth 2, 150 trees, rate 0.05) fitted on train to predict a winning night; test AUC, the test mean by predicted
  tercile, and the rule "skip the lowest predicted tercile".
Usage: python scripts/gotobi_accuracy_lab.py --out DIR [--data-cot DIR] [--max-days N (debugging)]"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

PAIRS = ("USDJPY", "EURJPY")
DAY = pd.Timedelta(days=1)
TRAIN_END = R.TRAIN_END
HOURS = (1, 3, 5, 7, 8, 9)


# ------------------------------------------------------------------------------------------------ variants
def variants():
    V = [("none", {})]
    V += [(f"stop {k} W", {"sl_w": k}) for k in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)]
    V += [(f"stop {x} bp", {"sl_bp": x}) for x in (50, 60, 70, 80, 90, 100, 120, 150, 200)]
    V += [(f"stop {k} W on a 15-min close", {"sl_w": k, "close_n": 15}) for k in (0.75, 1.0, 1.5, 2.0)]
    V += [(f"stop {x} bp on a 15-min close", {"sl_bp": x, "close_n": 15}) for x in (60, 100)]
    V += [(f"stop {k} W from {h:02d}:00", {"sl_w": k, "sl_from": h}) for h in (3, 5) for k in (1.0, 1.5)]
    V += [(f"stop {k} W until {h:02d}:00", {"sl_w": k, "sl_until": h}) for h in (5, 7) for k in (1.0, 1.5)]
    V += [(f"stop {k} W, re-enter at the entry price", {"sl_w": k, "re": "entry", "re_until": 9}) for k in (0.5, 0.75, 1.0)]
    V += [(f"stop {k} W, re-enter at 08:00", {"sl_w": k, "re": "time", "re_at": 8}) for k in (0.5, 0.75, 1.0)]
    V += [(f"target {t} W", {"tp_w": t}) for t in (0.5, 0.75, 1.0, 1.5, 2.0)]
    V += [(f"target {x} bp", {"tp_bp": x}) for x in (15, 25, 40)]
    V += [(f"target {t} W from 08:00", {"tp_w": t, "tp_from": 8}) for t in (0.25, 0.5, 1.0)]
    V += [(f"half at +{t} W", {"tp_w": t, "part": 0.5}) for t in (0.5, 1.0)]
    V += [(f"half at +{t} W, stop 100 bp", {"tp_w": t, "part": 0.5, "sl_bp": 100}) for t in (0.5, 1.0)]
    V += [(f"stop 100 bp + target {t} W", {"sl_bp": 100, "tp_w": t}) for t in (1.0, 1.5, 2.0)]
    V += [(f"scale-in at -{k} W", {"add_w": k, "add_until": 8}) for k in (0.5, 1.0)]
    V += [(f"scale-in at -{k} W, stop 100 bp", {"add_w": k, "add_until": 8, "sl_bp": 100}) for k in (0.5, 1.0)]
    V += [(f"dip entry at -{k} W", {"dip_w": k, "dip_until": 7}) for k in (0.25, 0.5)]
    return V


# ------------------------------------------------------------------------------------------------ simulator
def _first(cond):
    k = np.flatnonzero(cond)
    return int(k[0]) if len(k) else -1


def sim(bk, i, j, W, rule, H):
    """One long trade from bar i to bar j under `rule`. H: Tokyo hour on the Gotobi day -> first bar at or after it
    (clipped to i+1..j). Returns dict(net, mae, stopped, target, filled) or None (dip entry not filled)."""
    bl, bh, al, ah, ao, bo = bk.bl, bk.bh, bk.al, bk.ah, bk.ao, bk.bo
    sl = lambda q: R.slip(bk, q)                                         # noqa: E731
    segs = []                                                             # [weight, entry px, bar in, exit px, bar out]
    ref = ao[i] + sl(i)
    if "dip_w" in rule:
        L = ref - rule["dip_w"] * W
        b = H[rule["dip_until"]]
        q = _first(al[i + 1:b] <= L)
        if q < 0:
            return None
        q += i + 1
        E0, q0, w0 = min(L, ao[q]), q, 1.0
    else:
        E0, q0, w0 = ref, i, (0.5 if "add_w" in rule else 1.0)
    open_legs = [[w0, E0, q0]]
    S = -np.inf
    if "sl_w" in rule:
        S = E0 - rule["sl_w"] * W
    if "sl_bp" in rule:
        S = E0 * (1 - rule["sl_bp"] * 1e-4)
    sa = max(q0, H[rule["sl_from"]]) if "sl_from" in rule else q0
    sb = H[rule["sl_until"]] if "sl_until" in rule else j
    T = np.inf
    if "tp_w" in rule:
        T = E0 + rule["tp_w"] * W
    if "tp_bp" in rule:
        T = E0 * (1 + rule["tp_bp"] * 1e-4)
    ta = max(q0 + 1, H[rule["tp_from"]]) if "tp_from" in rule else q0 + 1
    close_n = rule.get("close_n")
    if close_n:
        mins = (bk.idx.minute.to_numpy() if not hasattr(bk, "_mins") else bk._mins)
        bk._mins = mins
        bc = bk.m["bid_c"].to_numpy() if not hasattr(bk, "_bc") else bk._bc
        bk._bc = bc

    def stop_hit(a, b):
        """(bar, fill) of the first stop in bars a..b-1, else (-1, nan)."""
        a, b = max(a, sa), min(b, sb)
        if b <= a or not np.isfinite(S):
            return -1, np.nan
        if close_n:
            q = _first(((mins[a:b] + 1) % close_n == 0) & (bc[a:b] <= S))
            if q < 0:
                return -1, np.nan
            q += a
            q1 = min(q + 1, j)
            return q1, bo[q1] - sl(q1)
        q = _first(bl[a:b] <= S)
        if q < 0:
            return -1, np.nan
        q += a
        return q, (min(S, bo[q]) if q > i else S) - sl(q)

    def close_all(q, px):
        for w, E, a in open_legs:
            segs.append([w, E, a, px, q])
        open_legs.clear()

    stopped = target = filled = False
    t = q0                                                                # bars before t are settled
    # scale-in: the add fills before a stop in the same bar (the worse case for a long)
    if "add_w" in rule:
        L2 = E0 - rule["add_w"] * W
        b = H[rule["add_until"]]
        qa = _first(al[i + 1:b] <= L2)
        qa = qa + i + 1 if qa >= 0 else -1
        qs, fs = stop_hit(t, j)
        if qa >= 0 and (qs < 0 or qa <= qs):
            open_legs.append([0.5, min(L2, ao[qa]), qa])
            filled = True
    qs, fs = stop_hit(t, j)
    qt = -1
    if np.isfinite(T) and ta < j:
        qt = _first(bh[ta:j] >= T)
        qt = qt + ta if qt >= 0 else -1
    if qt >= 0 and (qs < 0 or qt < qs):                                   # target first
        target = True
        f = rule.get("part", 1.0)
        rest = []
        for w, E, a in open_legs:
            segs.append([w * f, E, a, T, qt])
            if f < 1:
                rest.append([w * (1 - f), E, a])
        open_legs[:] = rest
        if open_legs:
            qs, fs = stop_hit(qt + 1, j)
            if qs >= 0:
                stopped = True
                close_all(qs, fs)
    elif qs >= 0:
        stopped = True
        close_all(qs, fs)
        if rule.get("re") == "entry":
            b = H[rule["re_until"]]
            q3 = _first(ah[qs + 1:b] >= E0) if b > qs + 1 else -1
            if q3 >= 0:
                q3 += qs + 1
                open_legs.append([1.0, max(E0, ao[q3]) + sl(q3), q3])
        elif rule.get("re") == "time":
            kr = H[rule["re_at"]]
            if qs < kr < j:
                open_legs.append([1.0, ao[kr] + sl(kr), kr])
    if open_legs:
        close_all(j, bo[j] - sl(j))
    net = 0.0
    path = np.zeros(j - i + 1)
    for w, E, a, X, b in segs:
        net += w * (X / E - 1.0 - 2 * bk.comm + R.swap_ret(bk, a, b, +1, E))
        if b > a:
            path[a - i:b - i] += w * (bl[a:b] / E - 1.0)
        path[b - i:] += w * (X / E - 1.0)
    mae = float(min(path.min(), net, 0.0))
    return {"net": float(net), "mae": mae, "stopped": stopped, "target": target, "filled": filled}


def hours_map(bk, d, i, j):
    H = {}
    for h in HOURS:
        q = int(bk.idx.searchsorted(U.jst(d, h, 0)))
        H[h] = int(min(max(q, i + 1), j))
    return H


def run_pair(bk, G, W, max_days=None):
    V = variants()
    out = {n: {} for n, _ in V}
    diag = {}
    days = G if max_days is None else G[:max_days]
    for d in days:
        w = float(W.get(d, np.nan))
        if not np.isfinite(w) or w <= 0:
            continue
        i, j = F.window(bk, d)
        if i < 0:
            continue
        if not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all():
            continue
        H = hours_map(bk, d, i, j)
        for n, rule in V:
            r = sim(bk, i, j, w, rule, H)
            if r is not None:
                out[n][d] = r
        m0 = bk.mo[i]
        diag[d] = {"W_ret": w / m0, "final_mid": bk.mo[j] / m0 - 1.0,
                   **{f"run_{h:02d}": bk.mo[H[h]] / m0 - 1.0 for h in HOURS}}
    frames = {n: pd.DataFrame.from_dict(v, orient="index").sort_index() for n, v in out.items()}
    return frames, pd.DataFrame.from_dict(diag, orient="index").sort_index()


# ------------------------------------------------------------------------------------------------ statistics
def stats(f, n_days):
    x = f.net.dropna()
    if len(x) < 10:
        return {"n": int(len(x))}
    m = f.mae.reindex(x.index)
    q = m.quantile(0.05)
    cvar = float(m[m <= q].mean())
    win, loss = x[x > 0], x[x < 0]
    return {"n": int(len(x)), "mean_bp": x.mean() * 1e4, "mean_per_day_bp": x.sum() / max(n_days, 1) * 1e4,
            "t": float(x.mean() / x.std() * np.sqrt(len(x))), "win_rate": float((x > 0).mean()),
            "avg_win_bp": win.mean() * 1e4 if len(win) else np.nan, "avg_loss_bp": loss.mean() * 1e4 if len(loss) else np.nan,
            "worst_bp": x.min() * 1e4, "p5_bp": x.quantile(0.05) * 1e4, "worst_open_loss_bp": m.min() * 1e4,
            "cvar5_open_loss_bp": cvar * 1e4, "score": float(x.mean() / -cvar) if cvar < 0 else np.nan,
            "share_stopped": float(f.stopped.reindex(x.index).mean()) if "stopped" in f else np.nan,
            "share_target": float(f.target.reindex(x.index).mean()) if "target" in f else np.nan}


def split(idx):
    idx = pd.DatetimeIndex(idx)
    return {"2007-2018 train": idx <= TRAIN_END, "2019-2025 test": idx > TRAIN_END, "all": np.ones(len(idx), bool)}


def exit_rows(series, frames, rows):
    base = frames["none"].copy()
    base.index = pd.DatetimeIndex(base.index)
    days = split(base.index)
    for n, f in frames.items():
        f = f.copy()
        f.index = pd.DatetimeIndex(f.index)
        for per, msk in days.items():
            dsel = base.index[msk]
            g = f.loc[f.index.intersection(dsel)]
            s = stats(g, len(dsel))
            # paired difference vs no stop, per Gotobi day (a dip entry that did not fill counts 0)
            dd = g.net.reindex(dsel).fillna(0.0) - base.net.reindex(dsel)
            s["diff_vs_none_bp"] = dd.mean() * 1e4
            s["t_diff"] = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else np.nan
            rows.append({"series": series, "variant": n, "period": per, **s})


def basket(fa, fb, ia, ib):
    """Equal-weight basket per Gotobi day. ia / ib: the days each pair trades without a stop; on those days a leg
    whose order did not fill (dip entry) counts 0, on other days the basket is the one pair that traded."""
    cols = ["net", "mae", "stopped", "target"]

    def leg(f, idx):
        f = f.copy()
        f.index = pd.DatetimeIndex(f.index)
        g = f.reindex(pd.DatetimeIndex(idx))[cols].astype(float)
        return g.fillna(0.0)
    j = leg(fa, ia).join(leg(fb, ib), lsuffix="_u", rsuffix="_e", how="outer")
    return pd.DataFrame({c: j[[c + "_u", c + "_e"]].mean(axis=1) for c in cols}).dropna(subset=["net"]).sort_index()


def picks(R_):
    tr = R_[(R_.series == "basket") & (R_.period == "2007-2018 train")].set_index("variant")
    base = float(tr.loc["none", "mean_per_day_bp"])
    ok90 = tr[tr.mean_per_day_bp >= 0.90 * base]
    ok95 = tr[tr.mean_per_day_bp >= 0.95 * base]
    out = {}
    for key, g, col, asc in (("P1 cut losses (mean / -CVaR5)", ok90, "score", False),
                             ("P2 accuracy (win rate)", ok90, "win_rate", False),
                             ("P3 worst case (worst trade)", ok95, "worst_bp", False)):
        g = g.drop(index="none", errors="ignore")
        if len(g):
            v = g.sort_values(col, ascending=asc).index[0]
            out[key] = {"variant": v, "train": float(g.loc[v, col])}
    return out


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        from cross_market_ta_lab import bh_q
        q[ok] = bh_q(p[ok])
    return q


# ------------------------------------------------------------------------------------------------ diagnostics
def touch_rows(series, frames, rows):
    """B1: of the trades a stop (target) closed early, the share that would have won held, and the mean of (exit
    rule - held) on them."""
    base = frames["none"].copy()
    base.index = pd.DatetimeIndex(base.index)
    for n, f in frames.items():
        f = f.copy()
        f.index = pd.DatetimeIndex(f.index)
        if n == "none" or "re-enter" in n or "scale" in n or "dip" in n:
            continue
        for col in ("stopped", "target"):
            hit = f[f[col].astype(float) > 0].index.intersection(base.index)
            if not len(hit):
                continue
            for per, msk in split(hit).items():
                h = hit[msk]
                if len(h) < 5:
                    continue
                dd = f.net.reindex(h) - base.net.reindex(h)
                rows.append({"series": series, "variant": n, "event": col, "period": per, "n_hit": int(len(h)),
                             "share_of_trades": len(h) / max(int(split(base.index)[per].sum()), 1),
                             "share_would_win_held": float((base.net.reindex(h) > 0).mean()),
                             "mean_held_bp": base.net.reindex(h).mean() * 1e4,
                             "mean_rule_bp": f.net.reindex(h).mean() * 1e4,
                             "rule_minus_held_bp": dd.mean() * 1e4,
                             "t": float(dd.mean() / dd.std() * np.sqrt(len(dd))) if len(dd) > 2 and dd.std() > 0 else np.nan})


BUCKETS = [-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf]


def conditional_rows(series, D, rows):
    """B2: mean mid move from hour h to the exit, by the trade's standing at h in W."""
    D = D.copy()
    D.index = pd.DatetimeIndex(D.index)
    for h in HOURS:
        z = D[f"run_{h:02d}"] / D.W_ret
        rest = (D.final_mid - D[f"run_{h:02d}"]) * 1e4
        b = pd.cut(z, BUCKETS)
        for per, msk in split(D.index).items():
            for lab, g in rest[msk].groupby(b[msk], observed=True):
                if len(g) < 5:
                    continue
                rows.append({"series": series, "hour": f"{h:02d}:00", "standing_W": str(lab), "period": per,
                             "n": int(len(g)), "mean_rest_bp": g.mean(), "t": float(g.mean() / g.std() * np.sqrt(len(g))),
                             "share_rest_up": float((g > 0).mean())})


# ------------------------------------------------------------------------------------------------ features
def yahoo(t):
    import yfinance as yf
    h = yf.download(t, start="2005-01-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"]
    h = h.iloc[:, 0] if isinstance(h, pd.DataFrame) else h
    h.index = pd.DatetimeIndex(h.index).tz_localize(None)
    return h[h.index <= "2025-12-31"].dropna()


def nominal_map(G, hol):
    """Gotobi day -> set of nominal labels (5, 10, 15, 20, 25, 'end') and whether it was moved."""
    def is_bd(d):
        return d.weekday() < 5 and d not in hol
    lab = {}
    for m in sorted({(d.year, d.month) for d in G}):
        p = pd.Period(year=m[0], month=m[1], freq="M")
        last = p.end_time.normalize()
        for dom in (5, 10, 15, 20, 25, None):
            nom = last if dom is None else pd.Timestamp(year=m[0], month=m[1], day=dom)
            d = nom
            while not is_bd(d):
                d -= DAY
            e = lab.setdefault(d, {"labels": set(), "moved": False})
            e["labels"].add("end" if dom is None else dom)
            e["moved"] |= d != nom
    return lab


def features(bk, G, W, base, hol, mkt, cot):
    lab = nominal_map(G, hol)
    Wm = W.rolling(60, min_periods=20).median().shift(1)
    rows = {}
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(bk.mo))])
    prev_net = base.net.shift(1)
    spreads = {}
    for d in base.index:
        d = pd.Timestamp(d)
        i, j = F.window(bk, d)
        if i < 0:
            continue
        w = float(W.get(d, np.nan))
        m0 = bk.mo[i]
        t0 = bk.idx[i]
        wr = w / m0

        def px_before(ts, max_gap_days=4):
            k = int(bk.idx.searchsorted(ts, side="right")) - 1
            if k < 0 or (ts - bk.idx[k]) > pd.Timedelta(days=max_gap_days):
                return np.nan
            return bk.mo[k]
        r = {}
        r["W_vs_median"] = w / Wm.get(d, np.nan) if np.isfinite(Wm.get(d, np.nan)) else np.nan
        for name, lag in (("ret_24h_W", 1), ("ret_1w_W", 7), ("ret_4w_W", 28)):
            p = px_before(t0 - lag * DAY)
            r[name] = (m0 / p - 1.0) / wr if np.isfinite(p) else np.nan
        kf = bk.at(U.jst(d - DAY, 9, 55), 30)
        r["move_since_fix_W"] = (m0 / bk.mo[kf] - 1.0) / wr if kf >= 0 and kf < i else np.nan
        a = int(bk.idx.searchsorted(t0 - 28 * DAY))
        r["below_4w_high_W"] = (m0 / np.max(bk.mh[a:i]) - 1.0) / wr if i - a > 1000 else np.nan
        a2 = int(bk.idx.searchsorted(t0 - 290 * DAY))
        r["vs_200d_mean"] = m0 / ((csum[i] - csum[a2]) / (i - a2)) - 1.0 if i - a2 > 150_000 else np.nan
        pa, pb = bk.at(U.jst(d - 2 * DAY, 23, 0), 120), bk.at(U.jst(d - DAY, 9, 53), 30)
        r["prev_night_range_W"] = ((np.max(bk.mh[pa:pb + 1]) - np.min(bk.ml[pa:pb + 1])) / w
                                   if pa >= 0 and pb > pa + 200 else np.nan)
        spreads[d] = (bk.ao[i] - bk.bo[i]) / m0
        r["prev_trade_bp"] = prev_net.get(d, np.nan) * 1e4
        e = lab.get(d, {"labels": set(), "moved": False})
        r["month_end"] = float("end" in e["labels"])
        r["day_10"] = float(10 in e["labels"])
        r["day_25"] = float(25 in e["labels"])
        r["moved"] = float(e["moved"])
        r["weekday"] = float(d.weekday())
        r["quarter_end_month"] = float(d.month in (3, 6, 9, 12))
        r["march"] = float(d.month == 3)
        k, off = d - DAY, 0
        while k.weekday() >= 5 or k in hol:
            off += 1
            k -= DAY
        r["days_off_before"] = float(off)
        prev_day = (d - DAY).normalize()
        for name, s, same_day_ok, kind in mkt:
            ss = s[s.index <= prev_day] if same_day_ok else s[s.index < prev_day]
            if len(ss) < 7:
                r[name] = np.nan
            elif kind == "level":
                r[name] = float(ss.iloc[-1])
            elif kind == "ret1":
                r[name] = float(ss.iloc[-1] / ss.iloc[-2] - 1.0)
            elif kind == "chg5":
                r[name] = float(ss.iloc[-1] - ss.iloc[-6])
        if cot is not None and len(cot):
            c = cot[cot.index + pd.Timedelta(days=4) <= prev_day]
            r["cot_jpy_spec_z"] = float(c.iloc[-1]) if len(c) else np.nan
        rows[d] = r
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    sp = pd.Series(spreads).sort_index()
    X["spread_vs_median"] = sp / sp.rolling(60, min_periods=20).median().shift(1)
    return X


def feature_rows(series, X, y, rows):
    X = X.reindex(y.index)
    tr, te = y.index <= TRAIN_END, y.index > TRAIN_END
    out = []
    for c in X.columns:
        x = X[c]
        r = {"series": series, "feature": c}
        for per, msk in (("train", tr), ("test", te)):
            ok = msk & x.notna().to_numpy() & y.notna().to_numpy()
            if ok.sum() < 30 or x[ok].nunique() < 2:
                continue
            rho, p = st.spearmanr(x[ok], y[ok])
            r[f"{per}_n"], r[f"{per}_rho"], r[f"{per}_p"] = int(ok.sum()), float(rho), float(p)
        okt = tr & x.notna().to_numpy()
        if okt.sum() >= 30:
            if x[okt].nunique() <= 2:
                lo_v, hi_v = x[okt].min(), x[okt].max()
                lo_m, hi_m = (lambda s: s == lo_v), (lambda s: s == hi_v)
            else:
                c1, c2 = x[okt].quantile([1 / 3, 2 / 3])
                lo_m, hi_m = (lambda s, c1=c1: s <= c1), (lambda s, c2=c2: s > c2)
            for per, msk in (("train", tr), ("test", te)):
                yy, xx = y[msk], x[msk]
                a, b = yy[hi_m(xx).to_numpy() & xx.notna().to_numpy()], yy[lo_m(xx).to_numpy() & xx.notna().to_numpy()]
                if len(a) >= 10 and len(b) >= 10:
                    r[f"{per}_top_minus_bottom_bp"] = (a.mean() - b.mean()) * 1e4
                    r[f"{per}_t_top_bottom"] = float(st.ttest_ind(a, b, equal_var=False).statistic)
                    r[f"{per}_win_top"], r[f"{per}_win_bottom"] = float((a > 0).mean()), float((b > 0).mean())
        out.append(r)
    o = pd.DataFrame(out)
    if "train_p" in o:
        o["train_q_bh"] = bh(o["train_p"].to_numpy())
        o["confirmed_on_test"] = ((o.train_q_bh <= 0.10) & (np.sign(o.train_rho) == np.sign(o.get("test_rho")))
                                  & (o.get("test_p") <= 0.10))
    rows.extend(o.to_dict("records"))


def model_rows(series, X, y, rows):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = X.reindex(y.index)
    tr, te = y.index <= TRAIN_END, y.index > TRAIN_END
    keep = [c for c in X.columns if X.loc[tr, c].notna().mean() > 0.8]
    Xt = X[keep]
    med, mu = Xt[tr].median(), Xt[tr].mean()
    sd = Xt[tr].std().replace(0, 1.0)
    Z = ((Xt.fillna(med) - mu) / sd).to_numpy()
    yb = (y > 0).astype(int).to_numpy()
    for name, mdl in (("logistic L2 C=0.1", LogisticRegression(C=0.1, max_iter=2000)),
                      ("gradient boosting d2 n150", GradientBoostingClassifier(max_depth=2, n_estimators=150,
                                                                               learning_rate=0.05, subsample=0.8,
                                                                               random_state=0))):
        mdl.fit(Z[tr], yb[tr])
        p_tr, p_te = mdl.predict_proba(Z[tr])[:, 1], mdl.predict_proba(Z[te])[:, 1]
        c1, c2 = np.quantile(p_tr, [1 / 3, 2 / 3])
        yt = y[te].to_numpy()
        r = {"series": series, "model": name, "features": len(keep), "train_auc": roc_auc_score(yb[tr], p_tr),
             "test_auc": roc_auc_score(yb[te], p_te), "test_n": int(te.sum()), "test_mean_all_bp": yt.mean() * 1e4}
        for lab_, m in (("low", p_te <= c1), ("mid", (p_te > c1) & (p_te <= c2)), ("high", p_te > c2)):
            r[f"test_{lab_}_n"], r[f"test_{lab_}_mean_bp"] = int(m.sum()), yt[m].mean() * 1e4 if m.sum() else np.nan
            r[f"test_{lab_}_win"] = float((yt[m] > 0).mean()) if m.sum() else np.nan
        k = p_te > c1
        r["test_skip_low_mean_bp"], r["test_skip_low_t"] = yt[k].mean() * 1e4, float(yt[k].mean() / yt[k].std() * np.sqrt(k.sum()))
        r["test_skip_low_per_day_bp"] = yt[k].sum() / len(yt) * 1e4
        r["test_skip_low_win"] = float((yt[k] > 0).mean())
        rows.append(r)


def regime_rows(series, frames, X, rows):
    base = frames["none"].copy()
    base.index = pd.DatetimeIndex(base.index)
    for feat in ("W_vs_median", "vix"):
        if feat not in X:
            continue
        x = X[feat].reindex(base.index)
        tr = base.index <= TRAIN_END
        if x[tr].notna().sum() < 60:
            continue
        c1, c2 = x[tr].quantile([1 / 3, 2 / 3])
        terc = pd.Series(np.where(x <= c1, "low", np.where(x <= c2, "mid", "high")), index=base.index).where(x.notna())
        for n in ("stop 1.0 W", "stop 2.0 W", "stop 100 bp", "stop 0.75 W", "target 1.0 W", "half at +0.5 W"):
            if n not in frames:
                continue
            f = frames[n].copy()
            f.index = pd.DatetimeIndex(f.index)
            dd = f.net.reindex(base.index) - base.net
            for per, msk in (("train", tr), ("test", ~tr)):
                for tt in ("low", "mid", "high"):
                    m = msk & (terc == tt).to_numpy()
                    g = dd[m].dropna()
                    if len(g) < 10:
                        continue
                    rows.append({"series": series, "regime": feat, "tercile": tt, "variant": n, "period": per,
                                 "n": int(len(g)), "none_mean_bp": base.net[m].mean() * 1e4,
                                 "rule_minus_none_bp": g.mean() * 1e4,
                                 "t": float(g.mean() / g.std() * np.sqrt(len(g))) if g.std() > 0 else np.nan})


# ------------------------------------------------------------------------------------------------ main
def load_cot(d):
    if not d:
        return None
    p = os.path.join(d, "cot.csv")
    if not os.path.exists(p):
        return None
    import cot_lab as C
    c = pd.read_csv(p, parse_dates=["date"])
    c = c[c.market == "JPY"].set_index("date").sort_index()
    net = (c.nc_long - c.nc_short) / c.oi
    return C.zscore(net).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-cot", default="")
    ap.add_argument("--max-days", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    errors, info = {}, {}
    mkt = []
    for name, tick, same_day_ok, kind in (("vix", "^VIX", False, "level"), ("vix_chg_1w", "^VIX", False, "chg5"),
                                          ("spx_ret_1d", "^GSPC", False, "ret1"), ("n225_ret_1d", "^N225", True, "ret1"),
                                          ("us10y_chg_1w", "^TNX", False, "chg5")):
        try:
            mkt.append((name, yahoo(tick), same_day_ok, kind))
        except Exception as ex:                                             # noqa: BLE001
            errors[f"yahoo {tick}"] = repr(ex)[:200]
    try:
        cot = load_cot(a.data_cot)
    except Exception as ex:                                                 # noqa: BLE001
        cot, errors["cot"] = None, repr(ex)[:300]
    info["cot_reports"] = 0 if cot is None else int(len(cot))
    ex_rows, touch, cond, feat, models, regime = [], [], [], [], [], []
    frames, diags, Xs, books, Gs = {}, {}, {}, {}, {}
    for p in PAIRS:
        bk = U.book(p)
        if bk is None:
            errors[p] = "no data"
            continue
        bk._hl()
        G, _, hol = F.calendar(bk)
        W = R.window_units(bk, G)
        fr, D = run_pair(bk, G, W, a.max_days)
        frames[p], diags[p], books[p], Gs[p] = fr, D, bk, G
        info[p] = {"first": str(bk.idx[0].date()), "last": str(bk.idx[-1].date()), "trades": int(len(fr["none"]))}
        exit_rows(p, fr, ex_rows)
        touch_rows(p, fr, touch)
        conditional_rows(p, D, cond)
        base = fr["none"].copy()
        base.index = pd.DatetimeIndex(base.index)
        try:
            Xs[p] = features(bk, G, W, base, hol, mkt, cot)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            errors[f"features {p}"] = traceback.format_exc()[-800:]
        print(p, "done", info[p], flush=True)
    if all(p in frames for p in PAIRS):
        iu, ie = frames["USDJPY"]["none"].index, frames["EURJPY"]["none"].index
        fb = {n: basket(frames["USDJPY"][n], frames["EURJPY"][n], iu, ie) for n in frames["USDJPY"]}
        frames["basket"] = fb
        exit_rows("basket", fb, ex_rows)
        touch_rows("basket", fb, touch)
        Du, De = diags["USDJPY"], diags["EURJPY"]
        Db = (Du + De.reindex(Du.index)) / 2
        conditional_rows("basket", Db.dropna(), cond)
        if "USDJPY" in Xs:
            Xs["basket"] = Xs["USDJPY"]
        for n in ("none", "stop 100 bp", "stop 2.0 W"):
            fb[n].to_csv(os.path.join(a.out, f"trades_basket_{n.replace(' ', '_')}.csv"))
    for s, X in Xs.items():
        base = frames[s]["none"].copy()
        base.index = pd.DatetimeIndex(base.index)
        y = base.net
        try:
            feature_rows(s, X, y, feat)
            model_rows(s, X, y, models)
            regime_rows(s, frames[s], X, regime)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            errors[f"analysis {s}"] = traceback.format_exc()[-800:]
        X.to_csv(os.path.join(a.out, f"features_{s}.csv"))
    E = pd.DataFrame(ex_rows)
    tr = E.period == "2007-2018 train"
    E["q_diff_train"] = np.nan
    for s in E.series.unique():
        m = tr & (E.series == s) & (E.variant != "none")
        p_ = 2 * st.norm.sf(np.abs(E.loc[m, "t_diff"].to_numpy(float)))
        E.loc[m, "q_diff_train"] = bh(p_)
    E.to_csv(os.path.join(a.out, "exits.csv"), index=False)
    pk = picks(E) if "basket" in set(E.series) else {}
    for k, v in pk.items():
        te = E[(E.series == "basket") & (E.variant == v["variant"]) & (E.period == "2019-2025 test")]
        if len(te):
            v["test"] = te.iloc[0][["mean_per_day_bp", "t", "win_rate", "worst_bp", "cvar5_open_loss_bp", "score",
                                    "diff_vs_none_bp", "t_diff"]].to_dict()
    for name, rows in (("touch", touch), ("conditional", cond), ("features", feat), ("models", models),
                       ("regime", regime)):
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    json.dump({"picks": pk, "info": info, "errors": errors}, open(os.path.join(a.out, "summary.json"), "w"), indent=1,
              default=str)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 40)
    cols = ["series", "variant", "period", "n", "mean_per_day_bp", "t", "win_rate", "worst_bp", "cvar5_open_loss_bp",
            "score", "diff_vs_none_bp", "t_diff", "q_diff_train"]
    print(E[(E.series == "basket") & (E.period != "all")][cols].round(3).to_string())
    print(pd.DataFrame(feat).round(3).to_string())
    print(pd.DataFrame(models).round(3).to_string())
    print(json.dumps({"picks": pk, "info": info, "errors": errors}, indent=1, default=str))


if __name__ == "__main__":
    main()
