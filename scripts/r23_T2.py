"""Round 23 part T2 (docs/round23_preregistration.md): every half-hour time window with event-day filters, default and
nested walk-forward.

Grid per instrument: entry at the first minute of each half-hour 00:00-23:30 UTC (48) x hold {0.5,1,1.5,2,3,4,6,8,12,
16,24} h (11) x sign {long, short, fade 1h, fade 4h, follow 1h, follow 4h} (6) x day filter (FILTERS). Entry and exit at
the first-minute mid (within 5 minutes of the half-hour), costs from that minute's spread, as round 22 part T.
Usage: python scripts/r23_T2.py --symbol SYM --events DIR --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r22_T as T  # noqa: E402
import wf22  # noqa: E402
import wf23  # noqa: E402

HOLD_H = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24)
SIGNS = T.SIGNS
NEW = ("pre_hol_us", "pre_hol_uk", "pre_hol_jp", "opex", "quad_witching", "quarter_end", "msci_review", "boj", "ecb",
       "cpi_us")                                                         # no BoE filter (addendum 1)
FILTERS = T.FILTERS + NEW


def third_fridays(days, months=None):
    out = set()
    for m in pd.period_range(days.min(), days.max(), freq="M"):
        if months and m.month not in months:
            continue
        d = m.start_time.normalize()
        while d.weekday() != 4:
            d += pd.Timedelta(days=1)
        out.add(d + pd.Timedelta(days=14))
    return out


def last_bdays(days, months):
    out = set()
    for m in pd.period_range(days.min(), days.max(), freq="M"):
        if m.month in months:
            d = m.end_time.normalize()
            while d.weekday() >= 5:
                d -= pd.Timedelta(days=1)
            out.add(d)
    return out


def pre_holidays(hol):
    out = set()
    for h in hol:
        d = pd.Timestamp(h) - pd.Timedelta(days=1)
        while d.weekday() >= 5 or d in hol:
            d -= pd.Timedelta(days=1)
        out.add(d)
    return out


def event_days(days, evdir):
    import holidays
    yrs = list(range(days.min().year, days.max().year + 2))
    us = {pd.Timestamp(d) for d in holidays.financial_holidays("NYSE", years=yrs)}
    uk = {pd.Timestamp(d) for d in holidays.UK(years=yrs)}
    jp = {pd.Timestamp(d) for d in holidays.JP(years=yrs)}
    ev = {"pre_hol_us": pre_holidays(us), "pre_hol_uk": pre_holidays(uk), "pre_hol_jp": pre_holidays(jp),
          "opex": third_fridays(days), "quad_witching": third_fridays(days, {3, 6, 9, 12}),
          "quarter_end": last_bdays(days, {3, 6, 9, 12}), "msci_review": last_bdays(days, {2, 5, 8, 11})}
    rep = {}
    for k, fn in (("boj", "boj_dates.txt"), ("ecb", "ecb_dates.txt"), ("cpi_us", "cpi_us_dates.txt")):
        p = os.path.join(evdir, fn)
        ev[k] = {pd.Timestamp(x.strip()) for x in open(p)} if os.path.exists(p) else set()
        rep[k] = {"dates": len(ev[k]), "first": str(min(ev[k]).date()) if ev[k] else None,
                  "last": str(max(ev[k]).date()) if ev[k] else None}
    return {k: days.isin(sorted(v)) for k, v in ev.items()}, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--events", default="events")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mid, spr = T.load_mid(a.symbol, "")
    pip = T.pip_of(a.symbol, mid.to_numpy())
    hh = mid.index.floor("30min")
    hidx = pd.date_range(hh.min(), mid.index.max().ceil("30min"), freq="30min")
    first_min = pd.Series(mid.index, index=mid.index).groupby(hh).first()
    ok = (first_min - first_min.index).dt.total_seconds() <= 300
    P = mid.groupby(hh).first()[ok].reindex(hidx).to_numpy(float)
    Sp = spr.groupby(hh).first()[ok].reindex(hidx).to_numpy(float)
    C = T.side_cost(P, Sp, pip)
    slot = (hidx.hour * 2 + hidx.minute // 30).to_numpy()
    dates = hidx.tz_localize(None).normalize()
    has = pd.Series(np.isfinite(P), index=dates).groupby(level=0).any()
    days = pd.DatetimeIndex(has.index[has.to_numpy()])
    days = days[days <= wf22.DEV_END]
    dpos = pd.Series(np.arange(len(days)), index=days)
    masks, mrep = T.day_masks(days)
    em, erep = event_days(days, a.events)
    masks.update(em)
    MK = np.vstack([masks[f] for f in FILTERS])                            # filters x days
    nf = len(FILTERS)
    per = days.to_period("M")
    mk = np.asarray(per.year * 12 + per.month - 1)
    mstart = np.flatnonzero(np.r_[True, mk[1:] != mk[:-1]])
    bases = [(s, h, g) for s in range(48) for h in HOLD_H for g in SIGNS]
    B = np.full((len(bases), len(days)), np.nan, np.float32)
    for bi, (s, h, g) in enumerate(bases):
        k = int(round(2 * h))
        t = np.flatnonzero(slot == s)
        t = t[t + k < len(P)]
        gr = P[t + k] / P[t] - 1
        cost = C[t] + C[t + k]
        if g == "long":
            sg = np.ones(len(t))
        elif g == "short":
            sg = -np.ones(len(t))
        else:
            L = 2 if g.endswith("1") else 8
            prev = np.full(len(t), np.nan)
            okp = t - L >= 0
            prev[okp] = P[t[okp]] / P[t[okp] - L] - 1
            sg = np.sign(prev) * (-1 if g.startswith("fade") else 1)
        net = sg * gr - np.abs(sg) * cost
        ix = dpos.reindex(dates[t]).to_numpy()
        keep = np.isfinite(ix) & np.isfinite(net)
        B[bi, ix[keep].astype(int)] = net[keep]
    n_cfg = len(bases) * nf
    S = np.zeros((n_cfg, len(mstart)), np.float32)
    Q, A, V = S.copy(), S.copy(), S.copy()
    for bi in range(len(bases)):
        x = B[bi].astype(np.float64)
        fin = np.isfinite(x)
        X = np.where(MK, np.where(fin, x, 0.0), 0.0)                       # filter off: flat (0) where data exist
        Vd = np.broadcast_to(fin, MK.shape).astype(np.float64)
        Ad = (X != 0).astype(np.float64)
        sl = slice(bi * nf, (bi + 1) * nf)
        S[sl] = np.add.reduceat(X, mstart, axis=1)
        Q[sl] = np.add.reduceat(X ** 2, mstart, axis=1)
        A[sl] = np.add.reduceat(Ad, mstart, axis=1)
        V[sl] = np.add.reduceat(Vd, mstart, axis=1)

    def rows(idx, d0, d1):
        out = np.zeros((len(idx), d1 - d0))
        for j, i in enumerate(idx):
            bi, fi = divmod(i, nf)
            x = B[bi, d0:d1].astype(np.float64)
            out[j] = np.where(MK[fi, d0:d1] & np.isfinite(x), x, 0.0)
        return out
    ag = wf23.Agg(days, S=S.astype(np.float64), Q=Q.astype(np.float64), A=A.astype(np.float64), V=V.astype(np.float64),
                  rows=rows)
    first = max(2010, int(days.year.min()) + 3)
    res = wf23.both(ag, first, min_active=30, min_valid=50)
    out_rows, ser = [], {}
    cname = lambda i: "|".join(map(str, bases[i // nf])) + "|" + FILTERS[i % nf]  # noqa: E731
    for kind, (s, st, own, ch) in res.items():
        last = {k: (v if isinstance(v, str) else [cname(i) for i in v]) for k, v in list(ch.items())[-2:]}
        out_rows.append({"part": "T2", "family": f"T2_time_windows-{kind}", "symbol": a.symbol, "placebo": False,
                         "selection": kind, "n_cfg": n_cfg, **st, "own": own, "choices": json.dumps(last)[:2000]})
        ser[f"T2_{kind}"] = s
        print(a.symbol, kind, json.dumps({k: out_rows[-1][k] for k in ("oos_first", "p1_mean", "p2_mean", "p2_t", "own")},
                                         default=str), flush=True)
    pd.DataFrame(out_rows).to_csv(os.path.join(a.out, f"T2_rows_{a.symbol}.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, f"T2_series_{a.symbol}.parquet"))
    json.dump({"symbol": a.symbol, "days": len(days), "n_cfg": n_cfg, "filters": list(FILTERS),
               "filter_days": {f: int(masks[f].sum()) for f in FILTERS}, "events": erep, **mrep},
              open(os.path.join(a.out, f"T2_meta_{a.symbol}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
