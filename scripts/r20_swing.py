"""Round 20, part S (docs/round20_preregistration.md): Kullamaegi breakout, episodic pivot and the Minervini trend
template on point-in-time S&P 500 stocks, daily bars, 2005-2025.

  data --out DIR     S&P 500 membership (Wikipedia, rebuilt backwards) and Yahoo daily bars -> parquet panels
  run  --data DIR --out DIR
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time

import numpy as np
import pandas as pd

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
DEV_END = "2025-12-31"
COST = 5e-4                      # per side
FIN = 0.06 / 365                 # long financing per calendar day
SPLIT = pd.Timestamp("2015-01-01")
P1_START = pd.Timestamp("2005-01-01")


# ============================================================================================ data
def membership():
    """Point-in-time S&P 500 members from the public fja05680/sp500 history (date, comma-separated tickers)."""
    import requests
    lst = requests.get("https://api.github.com/repos/fja05680/sp500/contents", headers=UA, timeout=60).json()
    import re
    cand = []
    for f in lst:
        m = re.search(r"\((\d{2})-(\d{2})-(\d{4})\)\.csv$", f["name"])
        if f["name"].startswith("S&P 500 Historical Components") and m:
            cand.append((pd.Timestamp(f"{m.group(3)}-{m.group(1)}-{m.group(2)}"), f))
    print("candidates", [(str(d.date()), f["name"]) for d, f in cand], flush=True)
    f = max(cand, key=lambda x: x[0])[1]
    print("membership file", f["name"], flush=True)
    H = pd.read_csv(io.StringIO(requests.get(f["download_url"], headers=UA, timeout=120).text))
    H["date"] = pd.to_datetime(H["date"])
    H = H.sort_values("date")
    days = pd.bdate_range("2003-01-01", DEV_END)
    sets = [frozenset(t.strip().replace(".", "-") for t in str(x).split(",") if t.strip()) for x in H["tickers"]]
    tick = sorted(set().union(*sets))
    M = pd.DataFrame(False, index=days, columns=tick)
    pos = H["date"].searchsorted(days, side="right") - 1
    col = {t: i for i, t in enumerate(tick)}
    arr = np.zeros((len(days), len(tick)), bool)
    for i, p_ in enumerate(pos):
        if p_ >= 0:
            arr[i, [col[t] for t in sets[p_]]] = True
    M = pd.DataFrame(arr, index=days, columns=tick)
    return M, {"source": f["name"], "snapshots": int(len(H)), "first": str(H.date.min()), "last": str(H.date.max()),
               "tickers_ever": len(tick)}


def cmd_data(a):
    import yfinance as yf
    os.makedirs(a.out, exist_ok=True)
    M, meta = membership()
    M.to_parquet(os.path.join(a.out, "members.parquet"))
    tick = list(M.columns) + ["SPY"]
    P = {k: [] for k in ("Open", "High", "Low", "Close", "Volume")}
    got = []
    for i in range(0, len(tick), 80):
        chunk = tick[i:i + 80]
        for k in range(3):
            try:
                X = yf.download(chunk, start="2003-01-01", end="2026-01-01", auto_adjust=True, progress=False,
                                group_by="column", threads=True)
                break
            except Exception as ex:                                      # noqa: BLE001
                print("retry", i, repr(ex)[:100], flush=True)
                time.sleep(10)
        else:
            continue
        for f in P:
            if f in X.columns.get_level_values(0):
                P[f].append(X[f])
        print("chunk", i, flush=True)
    for f, parts in P.items():
        Y = pd.concat(parts, axis=1)
        Y = Y.loc[:, ~Y.columns.duplicated()]
        Y.to_parquet(os.path.join(a.out, f"{f.lower()}.parquet"))
        if f == "Close":
            got = [c for c in Y.columns if Y[c].notna().sum() > 50]
    meta["tickers_with_prices"] = len(got)
    ever = set(M.columns)
    meta["members_missing_prices"] = sorted(ever - set(got))[:2000]
    meta["n_missing"] = len(ever - set(got))
    # share of member-days covered by price data
    C = pd.read_parquet(os.path.join(a.out, "close.parquet")).reindex(index=M.index, columns=M.columns)
    meta["member_days_with_price_share"] = float((C.notna() & M).sum().sum() / M.sum().sum())
    json.dump(meta, open(os.path.join(a.out, "data_meta.json"), "w"), indent=1)
    print({k: v for k, v in meta.items() if k != "members_missing_prices"})


# ============================================================================================ rules
def load(d):
    O, H, L, C = (pd.read_parquet(os.path.join(d, f"{k}.parquet")) for k in ("open", "high", "low", "close"))
    M = pd.read_parquet(os.path.join(d, "members.parquet"))
    idx = C.index[(C.index >= "2003-01-01") & (C.index <= DEV_END)]
    O, H, L, C = (x.reindex(idx) for x in (O, H, L, C))
    M = M.reindex(index=idx, method="ffill").fillna(False).astype(bool)
    cols = [c for c in M.columns if c in C.columns]
    return O, H, L, C, M[cols]


def sim(rule, O, H, L, C, setup, level, stop_kind, exit_sma, max_days, spy):
    """One stock at a time per name. setup: boolean frame on day t (entries on t+1 or at t's open for 'open')."""
    trades = []
    sma = C.rolling(exit_sma).mean()
    adr = ((H / L) - 1).rolling(20).mean()
    pc = C.shift()
    tr = np.maximum(np.maximum(H - L, (H - pc).abs()), (L - pc).abs())
    atr = tr.rolling(20).mean()
    idx = C.index
    so, sc = spy["Open"].reindex(idx).to_numpy(), spy["Close"].reindex(idx).to_numpy()
    for name in setup.columns:
        s = setup[name].to_numpy()
        if not s.any():
            continue
        o, h, l, c = (X[name].to_numpy(float) for X in (O, H, L, C))
        lv = level[name].to_numpy(float) if level is not None else None
        sm, ad, at = sma[name].to_numpy(), adr[name].to_numpy(), atr[name].to_numpy()
        n = len(c)
        i = 0
        busy_until = -1
        for t in np.flatnonzero(s):
            if t <= busy_until:
                continue
            if rule == "EP":                                             # entry at the open of the signal day
                e = t
                if not np.isfinite(o[e]):
                    continue
                px = o[e]
                stop = px - at[e - 1] if np.isfinite(at[e - 1]) else np.nan
                spy_in = so[e]
                start_check = e                                          # the stop is live on the entry day
            else:
                e = t + 1
                if e >= n or not np.isfinite(h[e]) or not np.isfinite(lv[t]) or h[e] <= lv[t]:
                    continue
                px = max(o[e], lv[t])
                spy_in = so[e]
                if rule == "QB":
                    stop = l[e]
                    if np.isfinite(ad[t]) and (px - stop) / px > ad[t]:
                        stop = px * (1 - ad[t])
                else:
                    stop = px * 0.93
                start_check = e + 1
            xi, xp, spy_out, why = None, None, None, None
            for k in range(e, min(n, e + max_days + 1)):
                if k >= start_check and np.isfinite(l[k]) and l[k] <= stop:
                    xi, xp, why = k, min(o[k], stop) if k > e else stop, "stop"
                    spy_out = sc[k]
                    break
                min_day = e + 2 if rule == "QB" else e
                if k >= min_day and np.isfinite(sm[k]) and c[k] < sm[k]:
                    xi, xp, why, spy_out = k, c[k], "trail", sc[k]
                    break
            if xi is None:
                k = min(n - 1, e + max_days)
                xi, xp, why, spy_out = k, c[k], "time", sc[k]
            if not (np.isfinite(xp) and np.isfinite(px) and px > 0):
                busy_until = xi
                continue
            days = (idx[xi] - idx[e]).days
            r = xp / px - 1 - 2 * COST - FIN * days
            rs = spy_out / spy_in - 1 if np.isfinite(spy_out) and np.isfinite(spy_in) and spy_in > 0 else np.nan
            trades.append({"rule": rule, "symbol": name, "entry": idx[e], "exit": idx[xi], "ret": r, "spy": rs, "excess": r - rs,
                           "days": days, "why": why, "entry_px": px, "exit_px": xp})
            busy_until = xi
    return pd.DataFrame(trades)


def signals(O, H, L, C, M):
    r63 = C / C.shift(63) - 1
    r252 = C / C.shift(252) - 1
    elig = M & C.notna() & (C.rolling(252).count() >= 252)
    rk63 = r63.where(elig).rank(axis=1, pct=True)
    rk252 = r252.where(elig).rank(axis=1, pct=True)
    h10, l10 = H.rolling(10).max(), L.rolling(10).min()
    adr5, adr20 = ((H / L) - 1).rolling(5).mean(), ((H / L) - 1).rolling(20).mean()
    s10, s20, s50, s150, s200 = (C.rolling(k).mean() for k in (10, 20, 50, 150, 200))
    QB = elig & (rk63 >= 0.90) & (h10 <= 1.10 * l10) & (adr5 <= adr20) & (C > s10) & (s10 > s20)
    gap = O / C.shift() - 1
    EP = elig.shift().fillna(False).astype(bool) & (gap >= 0.10) & (r63.shift() <= 0.10)
    MV = elig & (C > s150) & (C > s200) & (s150 > s200) & (s200 > s200.shift(21)) & (s50 > s150) & (s50 > s200) & (C > s50) \
        & (C >= 1.30 * L.rolling(252).min()) & (C >= 0.75 * H.rolling(252).max()) & (rk252 >= 0.70) \
        & (H.rolling(10).max() <= 1.08 * L.rolling(10).min())
    h20 = H.rolling(20).max()
    base = {"QB_base": (elig, h10), "EP_base": (elig.shift().fillna(False).astype(bool) & (gap >= 0.10), None),
            "MV_base": (elig, h20)}
    return {"QB": (QB, h10), "EP": (EP, None), "MV": (MV, h20)}, base


def month_t(T, col="excess"):
    x = T[col].astype(float).dropna()
    if len(x) < 10:
        return np.nan
    m = x.mean()
    g = (x - m).groupby(pd.to_datetime(T.loc[x.index, "exit"]).dt.to_period("M")).sum()
    se = math.sqrt((g ** 2).sum()) / len(x)
    return float(m / se) if se > 0 else np.nan


def portfolio(T, C, spy):
    """Equal-weight daily return of open positions (cash when flat); returns stats."""
    idx = C.index
    pos = {}
    R = pd.Series(0.0, index=idx)
    N = pd.Series(0, index=idx)
    for tr in T.itertuples():
        c = C[tr.symbol]
        seg = c.loc[tr.entry:tr.exit]
        if len(seg) < 1:
            continue
        px = np.r_[tr.entry_px, seg.to_numpy()[:-1]] if len(seg) > 1 else np.array([tr.entry_px])
        nx = seg.to_numpy().copy()
        nx[-1] = tr.exit_px
        r = nx / px - 1
        r[0] -= COST
        r[-1] -= COST
        R.loc[seg.index] += r
        N.loc[seg.index] += 1
    d = (R / N.replace(0, np.nan)).fillna(0.0)
    d = d[d.index >= P1_START]
    out = {}
    for lab, m in (("2005-2014", d.index < SPLIT), ("2015-2025", d.index >= SPLIT), ("2005-2025", d.index >= P1_START)):
        x = d[m]
        sp = spy["Close"].reindex(x.index).pct_change().fillna(0.0)
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        out[lab] = {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(x.mean() / x.std() * math.sqrt(252)) if x.std() > 0 else None,
                    "maxdd": float((eq / eq.cummax() - 1).min()), "spy_cagr": float((1 + sp).prod() ** (1 / yrs) - 1),
                    "invested_share": float((N.reindex(x.index) > 0).mean()), "avg_positions": float(N.reindex(x.index).mean())}
    return out


def cmd_run(a):
    from scipy.stats import norm
    os.makedirs(a.out, exist_ok=True)
    O, H, L, C, M = load(a.data)
    spyO = pd.read_parquet(os.path.join(a.data, "open.parquet"))["SPY"]
    spyC = pd.read_parquet(os.path.join(a.data, "close.parquet"))["SPY"]
    spy = pd.DataFrame({"Open": spyO, "Close": spyC}).reindex(C.index)
    names = list(M.columns)
    O, H, L, C = (x[names] for x in (O, H, L, C))
    main_s, base_s = signals(O, H, L, C, M)
    spec = {"QB": (10, 60), "EP": (20, 60), "MV": (50, 120)}
    rows, allT = [], []
    for group, S in (("test", main_s), ("base", base_s)):
        for rid, (setup, level) in S.items():
            r = rid.split("_")[0]
            ex, mx = spec[r]
            setup = setup.copy()
            setup[setup.index < "2004-01-01"] = False
            T = sim(r, O, H, L, C, setup.astype(bool), level, None, ex, mx, spy)
            T["id"] = rid
            allT.append(T)
            T1, T2 = T[(T.entry >= P1_START) & (T.entry < SPLIT)], T[T.entry >= SPLIT]
            rec = {"id": f"S-{rid}", "group": group, "p1_trades": len(T1), "p1_excess_pct": T1.excess.mean() * 100,
                   "p1_t": month_t(T1), "p2_trades": len(T2), "p2_excess_pct": T2.excess.mean() * 100, "p2_t": month_t(T2),
                   "p2_ret_pct": T2.ret.mean() * 100, "p2_win": float((T2.ret > 0).mean()) if len(T2) else None,
                   "p2_avg_win_pct": T2.ret[T2.ret > 0].mean() * 100, "p2_avg_loss_pct": T2.ret[T2.ret <= 0].mean() * 100,
                   "p2_days": T2.days.mean(), "exits": T2.why.value_counts().to_dict()}
            if group == "test":
                rec["portfolio"] = portfolio(T[T.entry >= P1_START], C, spy)
            rows.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "portfolio"}, default=str), flush=True)
    R = pd.DataFrame(rows)
    F = R[R.group == "test"].copy()
    F["p"] = 1 - norm.cdf(F.p2_t.fillna(-9))
    o = np.argsort(F.p.to_numpy())
    thr = 0.10 * np.arange(1, len(F) + 1) / len(F)
    passed = np.flatnonzero(F.p.to_numpy()[o] <= thr)
    F["bh"] = False
    if len(passed):
        F.iloc[o[: passed.max() + 1], F.columns.get_loc("bh")] = True
    F["own"] = (F.p1_excess_pct > 0) & (F.p2_excess_pct > 0) & (F.p2_t >= 1.645)
    F["counts"] = F.own & F.bh
    pd.concat(allT).to_csv(os.path.join(a.out, "s_trades.csv.gz"), index=False)
    json.dump({"tests": F.drop(columns=["portfolio"]).to_dict("records"), "all": R.to_dict("records")},
              open(os.path.join(a.out, "s_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["portfolio", "exits"]).round(3).to_string())
    print(F[["id", "p2_t", "p", "bh", "own", "counts"]].to_string())
    for r in rows:
        if "portfolio" in r:
            print(r["id"], json.dumps(r["portfolio"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"data": cmd_data, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
