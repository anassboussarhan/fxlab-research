"""Round 24 R10 (docs/round24_preregistration.md): Fed "net liquidity" (WALCL - WTREGEN - RRPONTSYD, FRED) as a weekly
signal for the Nasdaq CFD (Dukascopy USATECHIDXUSD daily bars, FX-day close 17:00 New York) and BTC/USDT (Binance).
The Wednesday level is released on Thursday at 16:30 New York; positions are taken at the next daily open
(Thursday 17:00 New York for the CFD's FX-day bar, Friday 00:00 UTC for BTC) and held for one week.
Usage: python scripts/r24_liq.py --out DIR
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import r24_common as C  # noqa: E402


def fred(sid):
    """A FRED series, trying FRED's CSV, its dated CSV, its text file and ALFRED in turn (the first run timed out on
    the CSV endpoint; addendum 5). Still FRED's own data."""
    last = None
    for url in (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2002-01-01&coed=2025-12-31",
                f"https://fred.stlouisfed.org/data/{sid}.txt",
                f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}"):
        try:
            r = C.get(url, tries=3, timeout=90)
        except Exception as ex:                                            # noqa: BLE001
            last = ex
            print("FRED try failed", url, repr(ex)[:120], flush=True)
            continue
        txt = r.text
        if url.endswith(".txt"):
            lines = txt.splitlines()
            k = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("DATE"))
            d = pd.read_csv(io.StringIO("\n".join(lines[k:])), sep=r"\s+")
        else:
            d = pd.read_csv(io.StringIO(txt))
        d = d.iloc[:, :2]
        d.columns = ["date", "v"]
        s = pd.Series(pd.to_numeric(d.v, errors="coerce").to_numpy(), index=pd.to_datetime(d.date, errors="coerce"))
        s = s[s.index.notna()].dropna()
        if len(s) > 100:
            print("FRED", sid, "from", url, len(s), "observations", flush=True)
            return s
        last = ValueError(f"{sid}: {len(s)} observations from {url}")
    raise RuntimeError(f"FRED unreachable for {sid}: {last!r}"[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        walcl, tga, rrp = fred("WALCL"), fred("WTREGEN"), fred("RRPONTSYD")
    except Exception as ex:                                                # noqa: BLE001
        print("FRED unreachable:", ex, flush=True)
        C.write(a.out, "R10", [], {}, {"not_run": f"FRED unreachable: {ex}"[:300]})
        return
    wed = walcl.index[walcl.index.dayofweek == 2]
    rrp_w = rrp.reindex(rrp.index.union(wed)).ffill().reindex(wed).fillna(0.0)   # RRP was ~0 before 2013
    tga_w = tga.reindex(tga.index.union(wed)).ffill().reindex(wed)
    walcl_w = walcl.reindex(wed)
    if walcl_w.max() > 1e5 and tga_w.max() < 1e5:                         # WALCL in USD millions, WTREGEN in billions
        tga_w = tga_w * 1000
    if rrp_w.max() < 1e5:                                                  # RRPONTSYD in billions
        rrp_w = rrp_w * 1000
    NL = (walcl_w - tga_w - rrp_w).dropna()
    NL = NL[NL.index <= "2025-12-31"]
    from fxlab.data.store import load_m1, resample
    m1 = load_m1("USATECHIDXUSD")
    d1 = resample(m1, "D1")
    del m1
    nas_o = ((d1["bid_o"] + d1["ask_o"]) / 2) if "bid_o" in d1 else d1["open"]
    nas_o.index = pd.DatetimeIndex(nas_o.index).tz_convert("UTC").tz_localize(None)
    import r24_crypto as K
    btc = K.klines("spot", "BTCUSDT", "1d")["open"]
    btc.index = btc.index.tz_localize(None)
    nas_cost = 2 * 1.0e-4                                                  # the round-23 CFD cost: about 1 bp a side
    rows, ser = [], {}
    rng = np.random.default_rng(2410)
    for inst, px, cost, start in (("NAS100", nas_o, nas_cost, "2012-01-01"), ("BTC", btc, 2 * 5e-4, "2017-08-17")):
        px = px[(px.index >= start) & (px.index <= "2025-12-31")].dropna()
        # entry: the first daily open after the Thursday 16:30 New York release of each Wednesday level
        ent = []
        for w in NL.index:                                                 # Wednesday level, Thursday 16:30 NY release
            rel = (pd.Timestamp(w.date()) + pd.Timedelta(days=1, hours=16, minutes=30)).tz_localize("America/New_York")
            rel = rel.tz_convert("UTC").tz_localize(None)
            first = px.index[px.index >= rel]                              # bars are labelled by their open (UTC)
            ent.append(first[0] if len(first) else pd.NaT)
        E = pd.Series(ent, index=NL.index).dropna()
        E = E[~E.duplicated()]
        wk_ret = pd.Series(px.reindex(E.values).to_numpy()[1:] / px.reindex(E.values).to_numpy()[:-1] - 1,
                           index=E.values[:-1])
        nl = NL.reindex(E.index).to_numpy()
        for placebo in (False, True):
            sig_src = nl.copy()
            if placebo:
                sh = int(rng.integers(26, 105))
                sig_src = np.roll(sig_src, sh)
            U = C.sobol(4, 128, seed=2411)
            desc, cols = [], {}
            for ci, u in enumerate(U):
                p = {"L": C.ilin(u[0], 1, 26), "form": C.cat(u[1], ["sign", "z"]), "thr": C.lin(u[2], 0.0, 2.0),
                     "side": C.cat(u[3], ["long_only", "long_short"])}
                desc.append(p)
                x = pd.Series(sig_src)
                ch = x - x.shift(p["L"])
                if p["form"] == "z":
                    z = (ch - ch.rolling(52, min_periods=26).mean()) / ch.rolling(52, min_periods=26).std()
                    s = np.where(z > p["thr"], 1.0, np.where(z < -p["thr"], -1.0, 0.0))
                else:
                    s = np.sign(ch.to_numpy())
                s = np.nan_to_num(s)[:-1]
                if p["side"] == "long_only":
                    s = np.where(s > 0, 1.0, 0.0)
                turn = np.abs(np.diff(np.r_[0.0, s]))
                cols[ci] = s * wk_ret.to_numpy() - turn * cost / 2
            W = pd.DataFrame(cols, index=pd.DatetimeIndex(wk_ret.index))
            r, s_ = C.units("R10", f"R10-net-liquidity-{inst}", inst, W, placebo=placebo, desc=None if placebo else desc,
                            min_active=8, min_valid=12)
            rows += r
            ser.update(s_)
    C.write(a.out, "R10", rows, ser, {"weeks": int(len(NL)), "first": str(NL.index[0].date())})


if __name__ == "__main__":
    main()
