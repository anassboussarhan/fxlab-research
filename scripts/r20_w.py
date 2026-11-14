"""Round 20, part W (docs/round20_preregistration.md, addendum W): Warrior-style small-cap gap trading.

  fetch --out DIR --cap USD        Databento: consolidated daily bars (EQUS.SUMMARY, 2024-07 .. 2025-12), the gap
                                   events, then Nasdaq ITCH minute bars for the event stock-days only. Every request
                                   is priced first; nothing is bought beyond the cap.
  run   --data DIR --out DIR       the tests
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

START, END = "2024-07-01", "2026-01-01"
NY = "America/New_York"
SPLIT = pd.Timestamp("2025-04-01")


def events_from_daily(D):
    """D: symbol, date, open, high, low, close, volume (unadjusted). Gap events known at the open."""
    D = D.sort_values(["symbol", "date"]).copy()
    g = D.groupby("symbol")
    D["pc"] = g.close.shift()
    D["pv"] = g.volume.shift()
    D["pdate"] = g.date.shift()
    D["gap"] = D.open / D.pc - 1
    ok = (D.date - D.pdate).dt.days.between(1, 4)
    E = D[ok & (D.gap >= 0.10) & D.pc.between(1.0, 20.0) & (D.pv >= 100_000)].copy()
    return E[["symbol", "date", "pc", "pv", "gap", "open", "high", "low", "close", "volume"]]


def cmd_fetch(a):
    import databento as db
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if key and not key.startswith("db-"):
        key = "db-" + key
    c = db.Historical(key)
    os.makedirs(a.out, exist_ok=True)
    spent = 0.0
    log = []
    q = dict(dataset="EQUS.SUMMARY", schema="ohlcv-1d", symbols="ALL_SYMBOLS", start=START, end=END)
    cost = float(c.metadata.get_cost(**q))
    log.append({"item": "EQUS.SUMMARY ohlcv-1d", "usd": cost})
    print("daily cost", cost, flush=True)
    if cost > a.cap:
        print("ABORT daily above cap")
        json.dump(log, open(os.path.join(a.out, "spend.json"), "w"))
        return
    raw = os.path.join(a.out, "daily.dbn.zst")
    if a.reuse and os.path.exists(a.reuse):                              # a raw file bought earlier: no new purchase
        st = db.DBNStore.from_file(a.reuse)
        log[-1]["reused"] = a.reuse
    else:
        st = c.timeseries.get_range(**q, path=raw)                       # keep the raw file: never pay twice
        spent += cost
    D = st.to_df(map_symbols=False).reset_index()
    print("daily rows", len(D), "columns", list(D.columns), flush=True)
    # ALL_SYMBOLS requests carry no symbol mappings: resolve instrument ids -> tickers (free symbology call)
    ids = sorted(D.instrument_id.unique().astype(int))
    mp = {}
    for i in range(0, len(ids), 2000):
        r = c.symbology.resolve(dataset="EQUS.SUMMARY", symbols=[str(x) for x in ids[i:i + 2000]], stype_in="instrument_id",
                                stype_out="raw_symbol", start_date=START, end_date=END)
        for k, v in r.get("result", {}).items():
            mp[int(k)] = [(pd.Timestamp(x["d0"]), pd.Timestamp(x["d1"]), x["s"]) for x in v]
    D["date"] = pd.to_datetime(D.ts_event).dt.tz_convert(None).dt.normalize()

    def sym(row_id, d):
        for d0, d1, s_ in mp.get(int(row_id), []):
            if d0 <= d < d1:
                return s_
        return None
    D["symbol"] = [sym(i, d) for i, d in zip(D.instrument_id.to_numpy(), D.date)]
    print("symbol resolved share", float(D.symbol.notna().mean()), flush=True)
    D = D[["symbol", "instrument_id", "date", "open", "high", "low", "close", "volume"]]
    if len(D) == 0 or D.symbol.notna().mean() < 0.9:
        print("ABORT: daily file empty or symbols unresolved; minutes not bought")
        json.dump(log, open(os.path.join(a.out, "spend.json"), "w"))
        return
    D = D.dropna(subset=["symbol", "open", "close"])
    D.to_parquet(os.path.join(a.out, "daily.parquet"))
    E = events_from_daily(D)
    E.to_parquet(os.path.join(a.out, "events.parquet"))
    print("events", len(E), "days", E.date.nunique(), flush=True)
    # price the minute bars for every event day first
    per = []
    for d, g in E.groupby("date"):
        syms = sorted(g.symbol.unique())
        qd = dict(dataset="XNAS.ITCH", schema="ohlcv-1m", symbols=syms, start=d.strftime("%Y-%m-%d"),
                  end=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        try:
            per.append((d, syms, float(c.metadata.get_cost(**qd))))
        except Exception as ex:                                          # noqa: BLE001
            print("price fail", d, repr(ex)[:150], flush=True)
    total = sum(x[2] for x in per)
    log.append({"item": "XNAS.ITCH ohlcv-1m event days", "usd": total, "days": len(per)})
    print("minute cost", total, "cap left", a.cap - spent, flush=True)
    if spent + total > a.cap:
        print("ABORT minutes above cap; only daily bought")
        json.dump(log, open(os.path.join(a.out, "spend.json"), "w"))
        return
    parts = []
    for d, syms, _ in per:
        st = c.timeseries.get_range(dataset="XNAS.ITCH", schema="ohlcv-1m", symbols=syms, start=d.strftime("%Y-%m-%d"),
                                    end=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        x = st.to_df(map_symbols=True).reset_index()
        if len(x):
            x["date"] = d
            parts.append(x[["symbol", "date", "ts_event", "open", "high", "low", "close", "volume"]])
    spent += total
    pd.concat(parts).to_parquet(os.path.join(a.out, "minutes.parquet"))
    log.append({"spent_total_usd": spent})
    json.dump(log, open(os.path.join(a.out, "spend.json"), "w"), indent=1)
    print("spent", spent)


def tstat_day(v, days):
    s = pd.Series(v).groupby(np.asarray(days)).mean()
    return float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))) if len(s) > 20 and s.std() > 0 else np.nan


def cmd_run(a):
    E = pd.read_parquet(os.path.join(a.data, "events.parquet"))
    Mn = pd.read_parquet(os.path.join(a.data, "minutes.parquet"))
    Mn["t"] = pd.to_datetime(Mn.ts_event).dt.tz_convert(NY)
    Mn["m"] = Mn.t.dt.hour * 60 + Mn.t.dt.minute
    rows = []
    cov = {"events": len(E), "with_minutes": 0, "with_premarket": 0}
    G = {k: g.sort_values("m") for k, g in Mn.groupby(["symbol", "date"])}
    for ev in E.itertuples():
        g = G.get((ev.symbol, ev.date))
        if g is None:
            continue
        cov["with_minutes"] += 1
        pm = g[(g.m >= 240) & (g.m < 570)]
        rth = g[(g.m >= 570) & (g.m < 960)]
        if len(rth) < 30:
            continue
        o = float(ev.open)                                                 # consolidated official open
        m, hi, lo, cl, op = (rth[k].to_numpy(float) for k in ("m", "high", "low", "close", "open"))
        def px_at(mm):
            j = np.searchsorted(m, mm)
            return float(op[j]) if j < len(m) else float(cl[-1])
        last = float(cl[m < 955][-1]) if (m < 955).any() else float(cl[-1])
        base = {"symbol": ev.symbol, "day": ev.date, "gap": ev.gap, "open": o, "pc": ev.pc}
        # W2 / W4 base rates: open to 10:30, open to the close
        p1030 = px_at(630)
        rows.append({**base, "test": "W2_long_open_1030", "gross": p1030 / o - 1})
        if ev.gap >= 0.30 and o >= 2.0:
            rows.append({**base, "test": "W4_short_open_close", "gross": -(last / o - 1)})
        # W1 gap-and-go: break of the premarket high, 09:30-10:30
        if len(pm) and pm.volume.sum() >= 50_000:
            cov["with_premarket"] += 1
            pmh = float(pm.high.max())
            k = np.flatnonzero((m < 630) & (hi > pmh))
            if len(k):
                j = int(k[0])
                E_ = max(float(op[j]), pmh)
                stop = float(lo[max(0, j - 5):j + 1].min())
                if stop >= E_:
                    stop = E_ * 0.95
                stop = max(stop, E_ * 0.90)
                tgt = E_ + 2 * (E_ - stop)
                X = None
                for q in range(j + 1, len(m)):
                    if m[q] >= 660:
                        X = float(op[q])
                        break
                    if lo[q] <= stop:
                        X = min(float(op[q]), stop)
                        break
                    if hi[q] >= tgt:
                        X = max(float(op[q]), tgt)
                        break
                if X is None:
                    X = float(cl[-1])
                rows.append({**base, "test": "W1_gap_and_go", "gross": X / E_ - 1})
        # W3 Dux-style gap-up short: gap >= 50 %, open >= 3; short at 10:30 if below the high so far by 10 %+
        if ev.gap >= 0.50 and o >= 3.0:
            j = np.searchsorted(m, 630)
            if j < len(m):
                hs = float(hi[:j].max()) if j > 0 else o
                ent = float(op[j])
                if ent <= 0.90 * hs:
                    stop = hs
                    X = None
                    for q in range(j, len(m)):
                        if m[q] >= 955:
                            X = float(op[q])
                            break
                        if hi[q] >= stop:
                            X = max(float(op[q]), stop)
                            break
                    if X is None:
                        X = last
                    rows.append({**base, "test": "W3_dux_short", "gross": -(X / ent - 1)})
    R = pd.DataFrame(rows)
    R["day"] = pd.to_datetime(R.day)
    out = {"coverage": cov, "tests": []}
    for test, g in R.groupby("test"):
        short = "short" in test
        for lab, cost in (("base", 0.003), ("stress", 0.006)):
            net = g.gross - 2 * cost - (0.005 if short else 0.0)          # 0.5 % borrow for the day on shorts
            p1, p2 = g.day < SPLIT, g.day >= SPLIT
            out["tests"].append({"test": test, "cost": lab, "n": len(g), "gross_pct": g.gross.mean() * 100,
                                 "gross_t": tstat_day(g.gross, g.day), "p1_n": int(p1.sum()), "p1_net_pct": net[p1].mean() * 100,
                                 "p2_n": int(p2.sum()), "p2_net_pct": net[p2].mean() * 100, "p2_t": tstat_day(net[p2], g.day[p2]),
                                 "win": float((net > 0).mean())})
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "w_trades.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "w_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(cov))
    print(pd.DataFrame(out["tests"]).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=9.0)
    ap.add_argument("--reuse", default="")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
