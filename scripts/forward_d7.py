"""Forward test of Double Seven on weekly bars (docs/forward_d7_preregistration.md). Paper only.

Downloads ECB reference rates (or reads --ecb FILE), builds the 28 crosses, runs D7-base and D7-safe over the whole
history (the rules need 200 weeks), and reports the trades filled from 2026-11-16 on, the open positions and the
signals for the next fill. Usage: python scripts/forward_d7.py --out DIR [--ecb FILE]"""
from __future__ import annotations

import argparse
import io
import json
import math
import os

import numpy as np
import pandas as pd

CUR8 = ("EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY")
FX = sorted({a + b for i, a in enumerate(CUR8) for b in CUR8[i + 1:]})
COST = 3e-4
START = pd.Timestamp("2026-11-16")
DEV_MEAN = 0.40
URL = ("https://data-api.ecb.europa.eu/service/data/EXR/D.USD+JPY+GBP+CHF+CAD+AUD+NZD.EUR.SP00.A"
       "?format=csvdata&startPeriod=1999-01-01")


def load(ecb):
    if ecb:
        raw = pd.read_csv(ecb, usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    else:
        import time
        import requests
        r = None
        for k in range(8):                                   # the ECB API times out now and then (504)
            try:
                r = requests.get(URL, timeout=180, headers={"User-Agent": "fxlab-research/1.0 (research data collection)"})
                if r.status_code == 200:
                    break
            except Exception:                                # noqa: BLE001
                pass
            time.sleep(min(600, 30 * 2 ** k))
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text), usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    raw["TIME_PERIOD"] = pd.to_datetime(raw.TIME_PERIOD)
    X = raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
    X["EUR"] = 1.0
    return X


def weekly(X):
    W = {}
    for s in FX:
        px = (X[s[3:]] / X[s[:3]]).dropna()
        iso = px.index.isocalendar()
        key = (iso.year * 100 + iso.week).to_numpy()
        last = np.r_[key[1:] != key[:-1], True]
        w = px[last]
        nxt = pd.Series(px.index, index=px.index).shift(-1)
        W[s] = pd.DataFrame({"c": w, "sma": w.rolling(200).mean(), "lo7": w.rolling(7).min(), "hi7": w.rolling(7).max(),
                             "N": w.diff().abs().ewm(alpha=1 / 20, adjust=False, min_periods=20).mean(),
                             "fill_t": nxt[last].values, "fill_px": px.reindex(nxt[last].values).values}, index=w.index)
    return W


def sim(W, k_stop=0.0, cap=0, cur_cap=0):
    weeks = sorted(set().union(*[set(d.index) for d in W.values()]))
    open_, trades, signals = {}, [], []
    last_week = weeks[-1]
    for t in weeks:
        for s in list(open_):
            d = W[s]
            if t not in d.index:
                continue
            r, e = d.loc[t], open_[s]
            stop_hit = k_stop > 0 and r.c < e["px"] - k_stop * e["N"]
            if r.c >= r.hi7 or stop_hit:
                if pd.isna(r.fill_px):               # signal on the latest week: fill pending
                    signals.append({"week": t, "symbol": s, "action": "exit", "reason": "stop" if stop_hit and r.c < r.hi7 else "7-week high"})
                    continue
                ret = r.fill_px / e["px"] - 1 - COST
                trades.append({"entry": e["t"], "exit": r.fill_t, "symbol": s, "ret": ret, "R": ret / (2 * e["N"] / e["px"]),
                               "reason": "stop" if stop_hit and r.c < r.hi7 else "7-week high"})
                del open_[s]
        for s, d in W.items():
            if s in open_ or t not in d.index:
                continue
            r = d.loc[t]
            if not (np.isfinite(r.sma) and r.c > r.sma and r.c <= r.lo7 and np.isfinite(r.N)):
                continue
            if cap and len(open_) >= cap:
                break
            if cur_cap:
                b, q = s[:3], s[3:]
                if sum(1 for o in open_ if b in (o[:3], o[3:])) >= cur_cap or sum(1 for o in open_ if q in (o[:3], o[3:])) >= cur_cap:
                    continue
            if pd.isna(r.fill_px):
                signals.append({"week": t, "symbol": s, "action": "buy", "reason": "7-week low above the 200-week average"})
                continue
            open_[s] = {"t": r.fill_t, "px": r.fill_px, "N": r.N}
    opens = [{"symbol": s, "entry": v["t"], "entry_px": v["px"], "N": v["N"]} for s, v in open_.items()]
    return pd.DataFrame(trades), pd.DataFrame(opens), pd.DataFrame(signals), last_week


def stats(T, dev_mean=DEV_MEAN):
    if not len(T):
        return {"trades": 0}
    x = T.R
    n = len(x)
    out = {"trades": n, "mean_R": float(x.mean()), "sum_R": float(x.sum()), "win_rate": float((x > 0).mean()),
           "worst_R": float(x.min())}
    if n > 1:
        m = x.mean()
        g = (x - m).groupby(pd.to_datetime(T.exit).dt.to_period("M")).sum()
        se = math.sqrt((g ** 2).sum()) / n
        out["t_clustered_month"] = float(m / se) if se > 0 else None
        out["t_vs_dev"] = float((m - dev_mean) / se) if se > 0 else None
        eq = x.cumsum()
        out["max_dd_R"] = float((eq - eq.cummax()).min())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ecb", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = load(a.ecb)
    W = weekly(X)
    rep = {"data_last_fixing": str(X.index.max().date()), "forward_start": str(START.date())}
    for name, kw in (("D7-base", {}), ("D7-safe", {"k_stop": 5.0, "cap": 4, "cur_cap": 2})):
        T, O, S, lw = sim(W, **kw)
        T["entry"] = pd.to_datetime(T.entry) if len(T) else T.get("entry")
        F = T[T.entry >= START] if len(T) else T
        Ofw = O[pd.to_datetime(O.entry) >= START] if len(O) else O
        F.to_csv(os.path.join(a.out, f"{name}_closed.csv"), index=False)
        Ofw.to_csv(os.path.join(a.out, f"{name}_open.csv"), index=False)
        S.to_csv(os.path.join(a.out, f"{name}_next_signals.csv"), index=False)
        rep[name] = {"forward": stats(F), "open_positions": int(len(Ofw)), "latest_week": str(pd.Timestamp(lw).date()),
                     "next_signals": S.to_dict("records") if len(S) else [],
                     "history_2019_2025": stats(T[(T.entry >= "2019-01-01") & (pd.to_datetime(T.exit) <= "2025-12-31")]) if len(T) else {}}
        if len(O):
            O.to_csv(os.path.join(a.out, f"{name}_open_incl_pre_start.csv"), index=False)
    # D7-family (addendum 2026-11-09): the seven close-only Connors weekly rules, long only, equal weight
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import fxlab.strategies  # noqa: F401
        from fxlab.registry import CATALOG
        fam = ["S.DAILY.DOUBLE_SEVEN", "S.DAILY.CUM_RSI", "S.FAMOUS.CONNORS_RSI2", "S.BOOK.CONNORS_R3", "S.BOOK.CONNORS_PCTB",
               "S.BOOK.CONNORS_CRSI", "S.DAILY.THREE_DOWN"]
        rows, opens, sigs = [], [], []
        for s_, d in W.items():
            wv = d.c.to_numpy(float)
            bars = pd.DataFrame({"open": np.r_[wv[0], wv[:-1]], "high": wv, "low": wv, "close": wv},
                                index=pd.DatetimeIndex(d.index).tz_localize("UTC"))
            for rid in fam:
                e = CATALOG[rid]
                side = e.fn(bars.copy(), **e.params)["side"].clip(lower=0).to_numpy()
                state, ent = 0, None
                for i in range(len(side)):
                    if state == 0 and side[i] > 0 and np.isfinite(d.N.iloc[i]):
                        if pd.isna(d.fill_px.iloc[i]):
                            sigs.append({"week": d.index[i], "rule": rid, "symbol": s_, "action": "buy"})
                            break
                        state, ent = 1, (d.fill_t.iloc[i], d.fill_px.iloc[i], d.N.iloc[i] / wv[i])
                    elif state == 1 and side[i] <= 0:
                        if pd.isna(d.fill_px.iloc[i]):
                            sigs.append({"week": d.index[i], "rule": rid, "symbol": s_, "action": "exit"})
                            break
                        r = d.fill_px.iloc[i] / ent[1] - 1 - COST
                        rows.append({"rule": rid, "symbol": s_, "entry": ent[0], "exit": d.fill_t.iloc[i], "ret": r, "R": r / (2 * ent[2])})
                        state, ent = 0, None
                if state == 1 and ent is not None:
                    opens.append({"rule": rid, "symbol": s_, "entry": ent[0], "entry_px": ent[1]})
        TF = pd.DataFrame(rows)
        TF["entry"] = pd.to_datetime(TF.entry)
        Ffw = TF[TF.entry >= START]
        Ofw = pd.DataFrame(opens)
        if len(Ofw):
            Ofw = Ofw[pd.to_datetime(Ofw.entry) >= START]
        Ffw.to_csv(os.path.join(a.out, "D7-family_closed.csv"), index=False)
        Ofw.to_csv(os.path.join(a.out, "D7-family_open.csv"), index=False)
        rep["D7-family"] = {"forward": stats(Ffw, 0.24), "forward_without_double_seven": stats(Ffw[Ffw.rule != "S.DAILY.DOUBLE_SEVEN"], 0.20),
                            "open_positions": int(len(Ofw)), "next_signals": sigs,
                            "history_2019_2025": stats(TF[(TF.entry >= "2019-01-01") & (pd.to_datetime(TF.exit) <= "2025-12-31")], 0.24),
                            "dev_mean_R": 0.24}
    except Exception as ex:                                                  # noqa: BLE001
        rep["D7-family"] = {"error": repr(ex)[:300]}
    json.dump(rep, open(os.path.join(a.out, "forward_d7.json"), "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
