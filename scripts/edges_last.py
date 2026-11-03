"""E25 and E26 of docs/edges_batch_preregistration.md (addendum 5). 2007-2025 only.

E26 (information only): Fed decisions 2017-2023 from the calendar dataset (calendar_repo/news_data, file
*_Fed_Interest_Rate_Decision.csv, timestamps in UTC). Expected move = sign(forecast - previous); none when equal or
missing. Long the dollar against EUR, JPY, GBP, AUD (equal weight) for an expected hike, short it for a cut, from
48 hours to 15 minutes before the decision. Fills bid/ask + slippage, commission, rate swaps.

E25: probe of the Ministry of Finance JGB auction history. The page and any spreadsheet it links are saved and
listed (--probe); the auction test itself runs only once the dates can be parsed (--e25 FILE with columns date,
tenor), USD/JPY 00:00 -> 03:00 UTC (before the 03:35 UTC result) and 03:40 -> 08:00 UTC (after), direction from the
fit period (60/40 by events if the dates start after 2008, else 2007-2018 / 2019-2025).
Usage: python scripts/edges_last.py --raw RAW --out DIR [--probe] [--e25 FILE]"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_weekly_lab as W  # noqa: E402

END = pd.Timestamp("2025-12-31 23:59", tz="UTC")


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else None


def e26(a):
    fs = glob.glob(os.path.join(a.raw, "**", "news_data", "*_Fed_Interest_Rate_Decision.csv"), recursive=True)
    if not fs:
        return {"status": "no calendar file"}
    d = pd.read_csv(fs[0])
    for c in ("Actual", "Forecast", "Previous"):
        d[c] = pd.to_numeric(d[c].astype(str).str.rstrip("%"), errors="coerce")
    d["ts"] = pd.to_datetime(d.Timestamp).dt.tz_localize("UTC")
    d = d.drop_duplicates("ts").sort_values("ts")
    d = d[(d.ts >= "2017-01-01") & (d.ts <= "2023-12-31")]
    d["exp"] = np.sign(d.Forecast - d.Previous)
    ev = d[d.exp.notna() & (d.exp != 0)]
    legs = {"EURUSD": -1, "USDJPY": 1, "GBPUSD": -1, "AUDUSD": -1}        # +1 = long the pair when long the dollar
    books = {p: W.book(p) for p in legs}
    rows = []
    for r in ev.itertuples():
        t0, t1 = r.ts - pd.Timedelta(hours=48), r.ts - pd.Timedelta(minutes=15)
        nets, gross = [], []
        for p, s in legs.items():
            bk = books[p]
            x = bk.trade(t0, t1, int(s * r.exp), 10) if bk is not None else None
            if x is None:
                continue
            net, g, _, _, n, flat, rate = x
            nets.append(net + (rate if np.isfinite(rate) else flat))
            gross.append(g)
        if nets:
            rows.append({"decision": str(r.ts), "expected": "hike" if r.exp > 0 else "cut", "forecast": r.Forecast,
                         "previous": r.Previous, "actual": r.Actual, "legs": len(nets),
                         "net_bp": float(np.mean(nets) * 1e4), "gross_bp": float(np.mean(gross) * 1e4)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "e26_events.csv"), index=False)
    if not len(R):
        return {"status": "no events priced", "decisions_with_expected_change": int(len(ev))}
    return {"test": "E26 dollar 48h -> 15 min before Fed decisions, in the expected direction (information only)",
            "decisions_2017_2023": int(len(d)), "with_expected_change": int(len(ev)), "priced": int(len(R)),
            "net_bp": float(R.net_bp.mean()), "t": tstat(R.net_bp), "gross_bp": float(R.gross_bp.mean()),
            "win": float((R.net_bp > 0).mean()),
            "hikes": {"n": int((R.expected == "hike").sum()), "net_bp": float(R[R.expected == "hike"].net_bp.mean())},
            "cuts": {"n": int((R.expected == "cut").sum()),
                     "net_bp": float(R[R.expected == "cut"].net_bp.mean()) if (R.expected == "cut").any() else None}}


def probe(a):
    import requests
    H = {"User-Agent": "Mozilla/5.0 (research script)"}
    out = {"pages": {}, "files": {}}
    d = os.path.join(a.out, "mof_probe")
    os.makedirs(d, exist_ok=True)
    pages = ["https://www.mof.go.jp/english/policy/jgbs/auction/past_auction_results/index.html",
             "https://www.mof.go.jp/jgbs/reference/appendix/index.html",
             "https://www.mof.go.jp/jgbs/auction/past_auction_results/index.html",
             "https://www.mof.go.jp/english/policy/jgbs/auction/calendar/0701e.htm",
             "https://www.mof.go.jp/english/policy/jgbs/auction/calendar/1501e.htm",
             "https://www.mof.go.jp/english/policy/jgbs/auction/calendar/2001e.htm",
             "https://www.mof.go.jp/english/policy/jgbs/auction/calendar/2501e.htm"]
    links = set()
    for u in pages:
        try:
            r = requests.get(u, headers=H, timeout=60)
            out["pages"][u] = {"status": r.status_code, "bytes": len(r.content)}
            if r.ok:
                open(os.path.join(d, re.sub(r"[^A-Za-z0-9]+", "_", u)[-80:] + ".html"), "wb").write(r.content)
                for h in re.findall(r'href="([^"]+)"', r.text):
                    if re.search(r"\.(xls|xlsx|csv)$", h, re.I):
                        links.add(requests.compat.urljoin(u, h))
        except Exception as ex:                                             # noqa: BLE001
            out["pages"][u] = {"error": repr(ex)[:150]}
    for u in sorted(links)[:40]:
        try:
            r = requests.get(u, headers=H, timeout=120)
            fn = os.path.join(d, os.path.basename(u))
            open(fn, "wb").write(r.content)
            info = {"status": r.status_code, "bytes": len(r.content)}
            try:
                X = pd.read_excel(fn, sheet_name=None, header=None) if not fn.lower().endswith(".csv") else \
                    {"csv": pd.read_csv(fn, header=None, encoding="cp932", on_bad_lines="skip")}
                info["sheets"] = {k: {"shape": list(v.shape), "head": v.head(8).astype(str).values.tolist()}
                                  for k, v in list(X.items())[:4]}
            except Exception as ex:                                         # noqa: BLE001
                info["parse"] = repr(ex)[:150]
            out["files"][u] = info
        except Exception as ex:                                             # noqa: BLE001
            out["files"][u] = {"error": repr(ex)[:150]}
    return out


def e25(a):
    D = pd.read_csv(a.e25, parse_dates=["date"], comment="#")
    D = D[D.date >= "2007-01-01"]
    D = D[(D.tenor.isin([10, 30])) & (D.date <= "2025-12-31")].drop_duplicates("date").sort_values("date")
    bk = W.book("USDJPY")
    res = {"auction_days": int(len(D)), "first": str(D.date.min().date()), "windows": {}}
    split60 = D.date.min() > pd.Timestamp("2008-12-31")
    for nm, (h0, m0, h1, m1) in (("before 00:00->03:00", (0, 0, 3, 0)), ("after 03:40->08:00", (3, 40, 8, 0))):
        rows = []
        for t in D.date:
            t = pd.Timestamp(t).tz_localize("UTC")
            # the 2007-2018 files have no 00:00-01:00 UTC bars: the 00:00 entry takes the first bar up to 01:05
            t0, t1 = t + pd.Timedelta(hours=h0, minutes=m0), t + pd.Timedelta(hours=h1, minutes=m1)
            i, j = bk.at(t0, 65 if h0 == 0 else 5), bk.at(t1, 5)
            L = bk.trade_ij(i, j, 1, t0, t1) if 0 <= i < j else None
            S = bk.trade_ij(i, j, -1, t0, t1) if 0 <= i < j else None
            if L and S:
                rows.append((t, L[0] + (L[6] if np.isfinite(L[6]) else L[5]), S[0] + (S[6] if np.isfinite(S[6]) else S[5]), L[1]))
        F = pd.DataFrame(rows, columns=["t", "long", "short", "gross"]).set_index("t")
        fit = np.arange(len(F)) < int(0.6 * len(F)) if split60 else (F.index.year <= 2018)
        if fit.sum() < 20:
            res["windows"][nm] = {"events": int(len(F)), "fit_events": int(fit.sum()), "status": "too few fit events"}
            continue
        dr = "long" if F.gross[fit].mean() > 0 else "short"
        chk = F[dr][~fit]
        res["windows"][nm] = {"events": int(len(F)), "direction_from_fit": dr, "fit_gross_bp": float(F.gross[fit].mean() * 1e4),
                              "check_n": int(len(chk)), "check_net_bp": float(chk.mean() * 1e4), "check_t": tstat(chk),
                              "counts": bool(chk.mean() > 0 and (tstat(chk) or 0) >= 1.645)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--e25", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = {"E26": e26(a)}
    if a.probe:
        out["E25_probe"] = probe(a)
    if a.e25:
        out["E25"] = e25(a)
    json.dump(out, open(os.path.join(a.out, "edges_last.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
