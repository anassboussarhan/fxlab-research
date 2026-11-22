"""Round 24 crypto parts (docs/round24_preregistration.md): R7 quarterly-futures basis, R8 perpetual funding across
venues, R9 the bitcoin "CME gap". Data: Binance public archive (data.binance.vision) and the venues' public APIs.
Usage: python scripts/r24_crypto.py {basis,funding,gap} --out DIR
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r24_common as C  # noqa: E402

ARCH = "https://data.binance.vision/"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"


def s3_list(prefix):
    out, marker = [], ""
    while True:
        r = C.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        txt = r.text
        pre = re.findall(r"<Prefix>([^<]+)</Prefix>", txt)[1:] if "<CommonPrefixes>" in txt else []
        keys = re.findall(r"<Key>([^<]+)</Key>", txt)
        out += pre + keys
        if "<IsTruncated>true</IsTruncated>" not in txt:
            return out
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", txt)
        marker = nm[0] if nm else (keys[-1] if keys else pre[-1])


def _zip_csv(key):
    r = C.get(ARCH + key, tries=4, timeout=180)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    d = pd.read_csv(io.StringIO(raw), header=None)
    if not str(d.iloc[0, 0]).replace(".", "").isdigit():
        d = d.iloc[1:]
    return d


def _ts(x):
    x = pd.to_numeric(x).astype("int64")
    return pd.to_datetime(np.where(x > 1e14, x // 1000, x), unit="ms", utc=True)


def klines(market, sym, interval, years=range(2017, 2026)):
    """Closes (and opens) of one symbol from the monthly kline files; market spot / um / cm."""
    base = {"spot": "data/spot/monthly/klines/", "um": "data/futures/um/monthly/klines/",
            "cm": "data/futures/cm/monthly/klines/"}[market]
    keys = [k for k in s3_list(f"{base}{sym}/{interval}/") if k.endswith(".zip")
            and int(re.search(r"-(\d{4})-\d\d\.zip$", k).group(1)) in years]
    with ThreadPoolExecutor(12) as ex:
        parts = list(ex.map(lambda k: _safe(_zip_csv, k), keys))
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    d = pd.concat(parts)
    out = pd.DataFrame({"open": pd.to_numeric(d[1]).to_numpy(), "high": pd.to_numeric(d[2]).to_numpy(),
                        "low": pd.to_numeric(d[3]).to_numpy(), "close": pd.to_numeric(d[4]).to_numpy()},
                       index=_ts(d[0]))
    out = out[~out.index.duplicated()].sort_index()
    return out[out.index <= "2025-12-31 23:59"]


def _safe(f, *a):
    try:
        return f(*a)
    except Exception as ex:                                                # noqa: BLE001
        print("skip", a[:1], repr(ex)[:100], flush=True)
        return None


# ------------------------------------------------------------------------------------------------ R7 basis
def cmd_basis(a):
    rows, ser, meta = [], {}, {}
    for coin in ("BTC", "ETH"):
        spot = klines("spot", f"{coin}USDT", "1h")["close"]
        S = spot.resample("1D").last().dropna()
        S.index = S.index.tz_localize(None)
        F = {}
        for mkt, pat in (("cm", rf"^{coin}USD_(\d{{6}})/$"), ("um", rf"^{coin}USDT_(\d{{6}})/$")):
            base = {"cm": "data/futures/cm/monthly/klines/", "um": "data/futures/um/monthly/klines/"}[mkt]
            syms = [p.rstrip("/").split("/")[-1] for p in s3_list(base)]
            syms = [s for s in syms if re.match(pat, s + "/")]
            for s in syms:
                k = klines(mkt, s, "1h")["close"]
                if len(k) < 24 * 20:
                    continue
                d = k.resample("1D").last().dropna()
                d.index = d.index.tz_localize(None)
                exp = pd.Timestamp("20" + s.split("_")[1][:2] + "-" + s.split("_")[1][2:4] + "-" + s.split("_")[1][4:6])
                F[(mkt, s)] = (d, exp)
        meta[coin] = {"spot_days": int(len(S)), "contracts": {f"{m}:{s}": [str(v[0].index[0].date()), str(v[1].date())]
                                                               for (m, s), v in F.items()}}
        print(coin, "contracts", len(F), flush=True)
        U = C.sobol(6, 128, seed=2470 + (coin == "ETH"))
        desc, cols = [], {}
        for i, u in enumerate(U):
            p = {"theta": C.lin(u[0], 0.0, 0.40), "exit": None, "win": C.cat(u[2], ["any"] + list(range(7, 91, 7))),
                 "which": C.cat(u[3], ["front", "next"]), "mkt": C.cat(u[4], ["cm", "um"])}
            p["exit"] = C.lin(u[1], -0.05, p["theta"])
            desc.append(p)
            cols[i] = basis_run(S, F, p, coin)
        W = pd.DataFrame(cols)
        r, s_ = C.units("R7", f"R7-basis-{coin}", coin, W, desc=desc)
        rows += r
        ser.update(s_)
        rng = np.random.default_rng(7 + (coin == "ETH"))
        Wp = pd.DataFrame({i: basis_run(S, F, dict(desc[i], theta=-1.0), coin, rng=rng) for i in range(len(desc))})
        r, s_ = C.units("R7", f"R7-basis-{coin}", coin, Wp, placebo=True)
        rows += r
        ser.update(s_)
    C.write(a.out, "R7", rows, ser, meta)


def basis_run(S, F, p, coin, rng=None, cost=5e-4):
    """Daily net return of long spot / short quarterly (equal coin quantity, per unit of spot at entry)."""
    cons = sorted([(exp, d.to_dict()) for (m, s), (d, exp) in F.items() if m == p["mkt"]], key=lambda z: z[0])
    days = S.index
    S = S.to_dict()
    ret = dict.fromkeys(days, 0.0)
    pos = None                                                            # (exp, series, entry_spot)
    for t_i in range(1, len(days)):
        t, y = days[t_i], days[t_i - 1]
        if pos is not None:
            exp, Fd, s0 = pos
            if t <= exp + pd.Timedelta(days=1) and y in Fd:
                f_t = Fd.get(t, np.nan)
                if t >= exp or np.isnan(f_t):                              # settles: the future converges to spot
                    f_t = S[t]
                ret[t] += ((S[t] - S[y]) - (f_t - Fd[y])) / s0
                b = (f_t / S[t] - 1) * 365 / max((exp - t).days, 1) if t < exp else 0.0
                if t >= exp or (b < p["exit"] and rng is None):
                    ret[t] -= 2 * cost * (0.5 if t >= exp else 1.0)        # at expiry only the spot leg is traded
                    pos = None
                continue
            pos = None
        avail = [(exp, d) for exp, d in cons if t in d and (exp - t).days > 1]
        if not avail:
            continue
        k = 0 if p["which"] == "front" or len(avail) < 2 else 1
        exp, Fd = avail[k]
        dte = (exp - t).days
        if p["win"] != "any" and dte > p["win"]:
            continue
        b = (Fd[t] / S[t] - 1) * 365 / dte
        go = (b >= p["theta"]) if rng is None else (rng.random() < 0.1)
        if go:
            pos = (exp, Fd, S[t])
            ret[t] -= 2 * cost
    return pd.Series(ret)


# ------------------------------------------------------------------------------------------------ R8 funding
def fund_binance(coin):
    keys = [k for k in s3_list(f"data/futures/um/monthly/fundingRate/{coin}USDT/") if k.endswith(".zip")]
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(lambda k: _safe(_zip_csv, k), keys) if p is not None]
    if not parts:
        return pd.Series(dtype=float)
    d = pd.concat(parts)
    s = pd.Series(pd.to_numeric(d[2]).to_numpy(), index=_ts(d[0]))
    return s[~s.index.duplicated()].sort_index()


def fund_deribit(coin):
    out, t = [], pd.Timestamp("2019-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    while t < end:
        t1 = min(t + pd.Timedelta(days=30), end)
        j = C.get("https://www.deribit.com/api/v2/public/get_funding_rate_history", js=True,
                  params={"instrument_name": f"{coin}-PERPETUAL", "start_timestamp": int(t.timestamp() * 1000),
                          "end_timestamp": int(t1.timestamp() * 1000)})
        out += [(x["timestamp"], x["interest_1h"]) for x in j.get("result", [])]
        t = t1
        time.sleep(0.2)
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series([v for _, v in out], index=pd.to_datetime([k for k, _ in out], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def fund_kraken(coin):
    sym = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD"}[coin]
    j = C.get("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates", js=True, params={"symbol": sym})
    r = j.get("rates", [])
    if not r:
        return pd.Series(dtype=float)
    s = pd.Series([x["relativeFundingRate"] for x in r], index=pd.to_datetime([x["timestamp"] for x in r], utc=True))
    return s[~s.index.duplicated()].sort_index()


def fund_hyperliquid(coin):
    out, t = [], int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)
    while t < end:
        j = C.get("https://api.hyperliquid.xyz/info", method="post", js=True,
                  body={"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": end})
        if not j:
            break
        out += [(x["time"], float(x["fundingRate"])) for x in j]
        nt = j[-1]["time"] + 1
        if nt <= t or len(j) < 2:
            break
        t = nt
        time.sleep(0.3)
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series([v for _, v in out], index=pd.to_datetime([k for k, _ in out], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def fund_dydx(coin):
    out, before = [], "2026-01-01T00:00:00.000Z"
    for _ in range(400):
        j = C.get(f"https://indexer.dydx.trade/v4/historicalFunding/{coin}-USD", js=True,
                  params={"limit": 100, "effectiveBeforeOrAt": before})
        h = j.get("historicalFunding", [])
        if not h:
            break
        out += [(x["effectiveAt"], float(x["rate"])) for x in h]
        nb = h[-1]["effectiveAt"]
        if nb == before:
            break
        before = nb
        time.sleep(0.2)
    if not out:
        return pd.Series(dtype=float)
    s = pd.Series([v for _, v in out], index=pd.to_datetime([k for k, _ in out], utc=True))
    return s[~s.index.duplicated()].sort_index()


def fund_gate(coin):
    j = C.get("https://api.gateio.ws/api/v4/futures/usdt/funding_rate", js=True,
              params={"contract": f"{coin}_USDT", "limit": 1000})
    if not j:
        return pd.Series(dtype=float)
    s = pd.Series([float(x["r"]) for x in j], index=pd.to_datetime([x["t"] for x in j], unit="s", utc=True))
    return s[~s.index.duplicated()].sort_index()


VENUES = {"binance": fund_binance, "deribit": fund_deribit, "kraken": fund_kraken, "hyperliquid": fund_hyperliquid,
          "dydx": fund_dydx, "gate": fund_gate}


def cmd_funding(a):
    rows, ser, meta = [], {}, {}
    for coin in ("BTC", "ETH"):
        D = {}
        for v, f in VENUES.items():
            s = _safe(f, coin)
            if s is None or len(s) < 30:
                meta[f"{coin}:{v}"] = "no data"
                continue
            s = s[s.index <= "2025-12-31 23:59"]
            if len(s) < 30:                                                # e.g. a venue that serves only its latest
                meta[f"{coin}:{v}"] = "no development-period data"         # records (all 2026, never read)
                continue
            d = s.groupby(s.index.tz_localize(None).normalize()).sum()     # funding paid per UTC day (per unit notional)
            D[v] = d
            meta[f"{coin}:{v}"] = {"first": str(d.index[0].date()), "last": str(d.index[-1].date()), "days": int(len(d))}
        print(coin, meta, flush=True)
        if len(D) < 2:
            continue
        X = pd.DataFrame(D).sort_index()
        names = list(X.columns)
        sets = [tuple(names)] + [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
        U = C.sobol(4, 128, seed=2480 + (coin == "ETH"))
        desc, cols, colsp = [], {}, {}
        rng = np.random.default_rng(81 + (coin == "ETH"))
        for i, u in enumerate(U):
            p = {"theta": C.lin(u[0], 0.0, 0.001), "n": C.ilin(u[2], 1, 21), "set": C.cat(u[3], sets)}
            p["exit"] = C.lin(u[1], -p["theta"], p["theta"])
            desc.append(p)
            cols[i] = funding_run(X, p)
            colsp[i] = funding_run(X, p, rng=rng)
        r, s_ = C.units("R8", f"R8-xvenue-{coin}", coin, pd.DataFrame(cols), desc=desc)
        rows += r
        ser.update(s_)
        r, s_ = C.units("R8", f"R8-xvenue-{coin}", coin, pd.DataFrame(colsp), placebo=True)
        rows += r
        ser.update(s_)
    C.write(a.out, "R8", rows, ser, meta)


def funding_run(X, p, rng=None, cost=5e-4):
    """Daily: short the perpetual where the expected funding (mean of the last n days) is highest, long where it is
    lowest, among venues with data that day; P&L = funding received - funding paid; four legs of costs on a switch."""
    Z = X[list(p["set"])]
    E = Z.rolling(p["n"], min_periods=1).mean().shift(1).to_numpy(float)   # known before the day starts
    Zv = Z.to_numpy(float)
    ret = np.zeros(len(Z))
    cur = None
    for t in range(len(Z)):
        ok = np.flatnonzero(np.isfinite(E[t]) & np.isfinite(Zv[t]))
        if len(ok) < 2:
            if cur is not None:
                ret[t] -= 2 * cost * 2
                cur = None
            continue
        e = E[t, ok]
        hi, lo = ok[int(np.argmax(e))], ok[int(np.argmin(e))]
        spread = E[t, hi] - E[t, lo]
        if rng is not None:
            hi, lo = (int(x) for x in rng.permutation(ok)[:2])
        want = (hi, lo) if (spread >= p["theta"] or (cur is not None and spread >= p["exit"])) else None
        if want != cur:
            if cur is not None:
                ret[t] -= 2 * cost * 2                                     # close two legs
            if want is not None:
                ret[t] -= 2 * cost * 2                                     # open two legs
            cur = want
        if cur is not None:
            ret[t] += Zv[t, cur[0]] - Zv[t, cur[1]]
    return pd.Series(ret, index=Z.index)


# ------------------------------------------------------------------------------------------------ R9 CME gap
def cmd_gap(a):
    m = klines("spot", "BTCUSDT", "1m")
    m.index = m.index.tz_convert("America/Chicago")
    print("minutes", len(m), m.index[0], m.index[-1], flush=True)
    close = m["close"]
    weeks = []
    for fri in pd.date_range(m.index[0].normalize(), m.index[-1].normalize(), freq="W-FRI"):
        f_t = fri + pd.Timedelta(hours=15, minutes=59)                    # the minute bar closing at 16:00
        s_t = fri + pd.Timedelta(days=2, hours=17)                         # Sunday 17:00 open
        if f_t not in close.index or s_t not in m.index:
            continue
        weeks.append((s_t, close[f_t], m.at[s_t, "open"]))
    print("weeks", len(weeks), flush=True)
    hi, lo, cl = m["high"].to_numpy(float), m["low"].to_numpy(float), close.to_numpy(float)
    opn = m["open"].to_numpy(float)
    pos_of = {t: i for i, t in enumerate(m.index)}
    U = C.sobol(5, 128, seed=2490)
    rng = np.random.default_rng(91)
    idx_all = np.arange(len(m))
    desc, cols, colsp = [], {}, {}
    days = pd.DatetimeIndex(sorted({t.tz_convert("UTC").tz_localize(None).normalize() for t in m.index[::1440]}))
    for i, u in enumerate(U):
        p = {"g": C.lin(u[0], 0.0, 0.03), "dir": C.cat(u[1], ["fade", "follow"]), "f": C.lin(u[2], 0.25, 1.0),
             "k": C.lin(u[3], 0.5, 5.0), "H": C.ilin(u[4], 1, 120)}
        desc.append(p)
        ev = [(pos_of[s_t], F, S) for s_t, F, S in weeks]
        cols[i] = gap_run(ev, p, hi, lo, cl, m.index, days)
        pl = []
        for s_i, F, S in ev:                                               # placebo: a random minute of the week
            j = int(rng.integers(max(0, s_i - 6 * 1440), s_i + 1440))
            j0 = max(0, j - 49 * 60)
            pl.append((j, cl[j0], opn[j]))
        colsp[i] = gap_run(pl, p, hi, lo, cl, m.index, days)
    rows, ser = [], {}
    r, s_ = C.units("R9", "R9-cme-gap", "BTC", pd.DataFrame(cols), desc=desc)
    rows += r
    ser.update(s_)
    r, s_ = C.units("R9", "R9-cme-gap", "BTC", pd.DataFrame(colsp), placebo=True)
    rows += r
    ser.update(s_)
    C.write(a.out, "R9", rows, ser, {"weeks": len(weeks)})


from numba import njit  # noqa: E402


@njit(cache=True)
def _scan(hi, lo, cl, s_i, end, d, tgt, stp):
    for j in range(s_i, end + 1):
        if d > 0:
            if lo[j] <= stp:
                return stp, j
            if hi[j] >= tgt:
                return tgt, j
        else:
            if hi[j] >= stp:
                return stp, j
            if lo[j] <= tgt:
                return tgt, j
    return cl[end], end


def gap_run(ev, p, hi, lo, cl, idx, days, cost=5e-4):
    """Event P&L booked on the UTC day of the exit (stop-first within a minute)."""
    ret = pd.Series(0.0, index=days)
    for s_i, F, S in ev:
        gap = S / F - 1
        if abs(gap) < max(p["g"], 1e-12):
            continue
        d = -np.sign(gap) if p["dir"] == "fade" else np.sign(gap)
        tgt = S + d * p["f"] * abs(S - F)
        stp = S - d * p["k"] * abs(S - F)
        end = min(len(cl) - 1, s_i + p["H"] * 60)
        px, j = _scan(hi, lo, cl, s_i, end, float(d), float(tgt), float(stp))
        r = d * (px / S - 1) - 2 * cost
        day = idx[j].tz_convert("UTC").tz_localize(None).normalize()
        if day in ret.index:
            ret[day] += r
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["basis", "funding", "gap"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"basis": cmd_basis, "funding": cmd_funding, "gap": cmd_gap}[a.cmd](a)


if __name__ == "__main__":
    main()
