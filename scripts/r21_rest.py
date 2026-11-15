"""Round 21, parts M, S, K and the combine (docs/round21_preregistration.md).

  macro  --out DIR                 M1 real-rate carry, M2 curve slope (BoE spot, OECD via DBnomics), nominal carry
  stocks --data DIR --out DIR      S1-S7 on the round-20 S&P 500 panel
  cdata  --out DIR                 Binance spot daily klines for all USDT pairs (public files)
  crypto --data DIR --out DIR      K1 momentum, K2 reversal
  combine --inp DIR --out DIR      23 tests, BH
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = "2025-12-31"
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) >= 12 and x.std() > 0 else np.nan


# ============================================================================================ M
def oecd_series(subj, measure="ST", lag=1):
    import r18_long as RL
    out = {}
    for c, cc in RL.OECD.items():
        for code in ([cc] if c != "EUR" else ["EA19", "EA20", "EA"]):
            try:
                doc = RL.get(f"https://api.db.nomics.world/v22/series/OECD/MEI/{code}.{subj}.{measure}.M?observations=1").json()
                doc = doc["series"]["docs"][0]
                out[c] = pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").to_numpy(),
                                   index=pd.PeriodIndex(doc["period"], freq="M")).dropna()
                break
            except Exception:                                              # noqa: BLE001
                continue
    R = pd.DataFrame(out)
    if "DEM" in R:
        R["EUR"] = R["EUR"].combine_first(R["DEM"]) if "EUR" in R else R["DEM"]
    return R.sort_index().shift(lag)


def cmd_macro(a):
    import r18_long as RL
    os.makedirs(a.out, exist_ok=True)
    X = RL.boe_rates()
    Xm = X.resample("ME").last()
    Xm.index = Xm.index.to_period("M")
    P = pd.DataFrame({c: Xm["USD"] / Xm[c] for c in RL.CUR8})             # USD per unit of currency c
    months = pd.period_range("1975-01", "2025-12", freq="M")
    S, src = RL.oecd_rates(list(months.astype(str)))                     # short rate, lagged 1, fraction
    S.index = pd.PeriodIndex(S.index, freq="M")
    L = oecd_series("IRLTLT01", "ST", 1) / 100.0
    CPI = oecd_series("CPALTT01", "GY", 2) / 100.0
    L, CPI = (z.reindex(months).reindex(columns=list(RL.CUR8)) for z in (L, CPI))
    S = S.reindex(months)
    sig = {"M1_real_carry": S - CPI, "M2_curve_slope": -(L - S), "M0_nominal_carry": S}
    ret = (P.shift(-1) / P - 1).reindex(months) + (S.sub(S["USD"], axis=0) / 12.0)
    rows = []
    for name, Z in sig.items():
        w_prev = pd.Series(0.0, index=list(RL.CUR8))
        for m in months[:-1]:
            z = Z.loc[m].dropna()
            r = ret.loc[m]
            z = z[r.reindex(z.index).notna()]
            if len(z) < 6:
                continue
            o = z.sort_values()
            w = pd.Series(0.0, index=list(RL.CUR8))
            w[o.index[-3:]] = 1 / 3
            w[o.index[:3]] = -1 / 3
            turn = float((w - w_prev).abs().sum())
            w_prev = w
            rows.append({"test": name, "month": str(m + 1), "ret": float((w * r.fillna(0)).sum()) - 3e-4 * turn, "n": len(z)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "r21_macro.csv"), index=False)
    cov = {k: {c: str(v[c].first_valid_index()) for c in v} for k, v in (("long", L), ("cpi", CPI), ("short", S))}
    json.dump({"coverage": cov, "short_src": src}, open(os.path.join(a.out, "r21_macro_meta.json"), "w"), indent=1)
    print(R.groupby("test").ret.agg(["mean", "count"]))


# ============================================================================================ S
def cmd_stocks(a):
    os.makedirs(a.out, exist_ok=True)
    O, H, C = (pd.read_parquet(os.path.join(a.data, f"{k}.parquet")) for k in ("open", "high", "close"))
    Mb = pd.read_parquet(os.path.join(a.data, "members.parquet"))
    idx = C.index[(C.index >= "2003-01-01") & (C.index <= DEV_END)]
    spy = C["SPY"].reindex(idx)
    names = [c for c in Mb.columns if c in C.columns]
    O, H, C = (x.reindex(idx)[names] for x in (O, H, C))
    M = Mb.reindex(index=idx, method="ffill").fillna(False).astype(bool)[names]
    Cf = C.ffill(limit=25)
    r1 = C.pct_change(fill_method=None)
    sig = {"S1_mom_12_1": C.shift(21) / C.shift(252) - 1, "S2_reversal_1m": -(C / C.shift(21) - 1),
           "S3_52w_high": C / H.rolling(252, min_periods=200).max(), "S4_low_vol": -r1.rolling(252, min_periods=200).std(),
           "S5_overnight_mom": np.log(O / C.shift()).rolling(21, min_periods=15).sum()}
    me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last().to_numpy()
    elig = M & C.notna() & (C.rolling(252).count() >= 252)
    rows = []
    for name, Z in sig.items():
        w_prev = pd.Series(0.0, index=names)
        for t0, t1 in zip(me[:-1], me[1:]):
            if t0 < pd.Timestamp("2004-12-01"):
                continue
            e = elig.loc[t0]
            z = Z.loc[t0][e].dropna()
            if len(z) < 100:
                continue
            nxt = (Cf.loc[t1] / C.loc[t0] - 1)[z.index]
            nxt = nxt.fillna(0.0)
            top = z.sort_values().index[-max(1, len(z) // 10):]
            w = pd.Series(0.0, index=names)
            w[top] = 1.0 / len(top)
            turn = float((w - w_prev).abs().sum())
            w_prev = w
            port = float(nxt[top].mean())
            bench = float(nxt.mean())
            rows.append({"test": name, "month": str(pd.Timestamp(t1).to_period("M")), "ret": port - bench - 5e-4 * turn,
                         "port": port, "bench": bench})
    # S6 deletion rebound
    Mv = M.to_numpy()
    out_day = np.flatnonzero((~Mv[1:]) & Mv[:-1], )
    dele = []
    for flat in np.flatnonzero(((~Mv[1:]) & Mv[:-1]).ravel()):
        i, j = divmod(flat, Mv.shape[1])
        d = i + 1
        name = names[j]
        p0 = C.iloc[d, j]
        if not np.isfinite(p0):
            continue
        k = min(len(idx) - 1, d + 60)
        seg = C.iloc[d:k + 1, j].dropna()
        if len(seg) < 2:
            continue
        r = seg.iloc[-1] / p0 - 1 - 10e-4
        rs = spy.loc[seg.index[-1]] / spy.iloc[d] - 1
        dele.append({"test": "S6_deletion", "name": name, "entry": idx[d], "exit": seg.index[-1], "ret": r - rs})
    # S7 pairs
    pr = []
    fm = [t for t in me if t >= pd.Timestamp("2004-01-01")]
    pos_of = {t: idx.get_loc(t) for t in fm}
    Ca = C.to_numpy(float)
    for t in fm:
        p = pos_of[t]
        if p < 252 or p + 1 >= len(idx):
            continue
        cols = np.flatnonzero(Mv[p] & np.isfinite(Ca[p - 252:p + 1]).all(axis=0))
        if len(cols) < 100:
            continue
        W = Ca[p - 252:p + 1, cols]
        Xn = W / W[0]
        sq = (Xn ** 2).sum(axis=0)
        D = sq[:, None] + sq[None, :] - 2 * Xn.T @ Xn
        iu = np.triu_indices(len(cols), 1)
        best = np.argsort(D[iu])[:20]
        end = min(len(idx) - 1, p + 126)
        for bidx in best:
            i, j = cols[iu[0][bidx]], cols[iu[1][bidx]]
            base_i, base_j = Ca[p - 252, i], Ca[p - 252, j]
            sp_form = Xn[:, iu[0][bidx]] - Xn[:, iu[1][bidx]]
            sd = sp_form.std()
            if sd <= 0:
                continue
            open_, ent = 0, None
            for q in range(p + 1, end + 1):
                ci, cj = Ca[q, i], Ca[q, j]
                if not (np.isfinite(ci) and np.isfinite(cj)):
                    if open_:
                        pr.append(close_pair(ent, Ca, q - 1, i, j, open_, idx))
                        open_ = 0
                    break
                s = ci / base_i - cj / base_j
                if open_ == 0 and abs(s) > 2 * sd:
                    open_, ent = (-1 if s > 0 else 1), q              # -1: short i, long j
                elif open_ != 0 and ((open_ == -1 and s <= 0) or (open_ == 1 and s >= 0)):
                    pr.append(close_pair(ent, Ca, q, i, j, open_, idx))
                    open_ = 0
            if open_:
                pr.append(close_pair(ent, Ca, end, i, j, open_, idx))
    P_ = pd.DataFrame(pr)
    if len(P_):
        m = P_.groupby(pd.to_datetime(P_.exit).dt.to_period("M")).ret.sum() / (20 * 6)
        for k, v in m.items():
            rows.append({"test": "S7_pairs", "month": str(k), "ret": float(v)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "r21_stocks.csv"), index=False)
    pd.DataFrame(dele).to_csv(os.path.join(a.out, "r21_deletions.csv"), index=False)
    print(R.groupby("test").ret.agg(["mean", "count"]))
    print("deletions", len(dele), "of", int(((~Mv[1:]) & Mv[:-1]).sum()))


def close_pair(ent, Ca, q, i, j, side, idx):
    ri = Ca[q, i] / Ca[ent, i] - 1
    rj = Ca[q, j] / Ca[ent, j] - 1
    r = (rj - ri) if side == -1 else (ri - rj)
    return {"entry": idx[ent], "exit": idx[q], "ret": r - 4 * 5e-4}


# ============================================================================================ K
STABLE = {"USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "UST", "USTC", "SUSD", "EUR", "GBP", "AUD", "TRY", "BRL",
          "RUB", "BIDR", "IDRT", "BKRW", "NGN", "UAH", "ZAR", "USDS", "USDSB", "AEUR", "EURI", "PAXG", "XUSD", "USD1",
          "BFUSD", "RLUSD", "USDE", "PYUSD", "BVND", "JPY", "MXN", "ARS", "PLN", "RON", "CZK", "COP", "WBTC", "WBETH", "BETH"}


def s3_list(prefix):
    import requests
    out, marker = [], ""
    base = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    while True:
        r = requests.get(base, params={"delimiter": "/", "prefix": prefix, "marker": marker}, headers=UA, timeout=60)
        txt = r.text
        pre = re.findall(r"<Prefix>([^<]+)</Prefix>", txt)[1:] if "<CommonPrefixes>" in txt else []
        keys = re.findall(r"<Key>([^<]+)</Key>", txt)
        out += pre + keys
        if "<IsTruncated>true</IsTruncated>" not in txt:
            return out
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", txt)
        marker = nm[0] if nm else (keys[-1] if keys else pre[-1])


def cmd_cdata(a):
    import requests
    os.makedirs(a.out, exist_ok=True)
    pre = s3_list("data/spot/monthly/klines/")
    syms = sorted({p.rstrip("/").split("/")[-1] for p in pre})
    usdt = [s for s in syms if s.endswith("USDT")]
    bases = {s[:-4] for s in usdt}
    keep = []
    for s in usdt:
        b = s[:-4]
        if b in STABLE:
            continue
        lev = any(b.endswith(x) and b[: -len(x)] in bases for x in ("UP", "DOWN", "BULL", "BEAR"))
        if lev:
            continue
        keep.append(s)
    print("symbols", len(syms), "USDT", len(usdt), "kept", len(keep), flush=True)

    def files(s):
        try:
            return [k for k in s3_list(f"data/spot/monthly/klines/{s}/1d/") if k.endswith(".zip")]
        except Exception:                                                  # noqa: BLE001
            return []

    with ThreadPoolExecutor(16) as ex:
        F = dict(zip(keep, ex.map(files, keep)))
    jobs = [(s, k) for s, ks in F.items() for k in ks if re.search(r"-(201[89]|202[0-5])-\d\d\.zip$", k)]
    print("files", len(jobs), flush=True)

    def get(job):
        s, k = job
        for _ in range(3):
            try:
                r = requests.get("https://data.binance.vision/" + k, headers=UA, timeout=60)
                if r.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    raw = z.read(z.namelist()[0]).decode()
                    d = pd.read_csv(io.StringIO(raw), header=None)
                    if not str(d.iloc[0, 0]).isdigit():
                        d = d.iloc[1:]
                    d = d.iloc[:, [0, 4, 7]].astype(float)
                    d.columns = ["t", "close", "qvol"]
                    d["sym"] = s
                    return d
            except Exception:                                              # noqa: BLE001
                pass
        return None

    with ThreadPoolExecutor(24) as ex:
        parts = [x for x in ex.map(get, jobs) if x is not None]
    D = pd.concat(parts, ignore_index=True)
    D["date"] = pd.to_datetime(D.t, unit="us" if D.t.max() > 1e14 else "ms").dt.normalize()
    big = D.t > 1e14
    if big.any() and (~big).any():
        D.loc[big, "date"] = pd.to_datetime(D.t[big], unit="us").dt.normalize()
        D.loc[~big, "date"] = pd.to_datetime(D.t[~big], unit="ms").dt.normalize()
    D = D.drop(columns="t").drop_duplicates(["sym", "date"])
    D.to_parquet(os.path.join(a.out, "crypto_daily.parquet"))
    json.dump({"symbols_kept": len(keep), "files": len(jobs), "downloaded": len(parts), "coins_with_data": int(D.sym.nunique()),
               "first": str(D.date.min()), "last": str(D.date.max())}, open(os.path.join(a.out, "cdata_meta.json"), "w"), indent=1)
    print("rows", len(D), "coins", D.sym.nunique())


def cmd_crypto(a):
    os.makedirs(a.out, exist_ok=True)
    D = pd.read_parquet(glob.glob(os.path.join(a.data, "**", "crypto_daily.parquet"), recursive=True)[0])
    C = D.pivot(index="date", columns="sym", values="close").sort_index()
    Q = D.pivot(index="date", columns="sym", values="qvol").reindex_like(C)
    C = C[C.index <= DEV_END]
    Q = Q.reindex(C.index)
    Cf = C.ffill(limit=10)
    hist = C.notna().cumsum()
    qv30 = Q.rolling(30, min_periods=20).sum()
    sat = C.index[C.index.dayofweek == 5]
    rows = []
    for name, look, sign in (("K1_momentum", 28, 1), ("K2_reversal", 7, -1)):
        w_prev = pd.Series(0.0, index=C.columns)
        for t0, t1 in zip(sat[:-1], sat[1:]):
            if t0 < pd.Timestamp("2018-12-01"):
                continue
            ok = (hist.loc[t0] >= 60) & C.loc[t0].notna()
            v = qv30.loc[t0][ok].dropna()
            uni = v.sort_values().index[-50:]
            if len(uni) < 30:
                continue
            past = (C.loc[t0, uni] / Cf.shift(look).loc[t0, uni] - 1).dropna()
            if len(past) < 30:
                continue
            o = past.sort_values()
            q = len(o) // 5
            lo, hi = (o.index[:q], o.index[-q:]) if sign > 0 else (o.index[-q:], o.index[:q])
            longs, shorts = (hi, lo)
            nxt = (Cf.loc[t1] / C.loc[t0] - 1)
            w = pd.Series(0.0, index=C.columns)
            w[longs] = 1.0 / q
            w[shorts] = -1.0 / q
            turn = float((w - w_prev).abs().sum())
            w_prev = w
            r = float(nxt[longs].fillna(0).mean() - nxt[shorts].fillna(0).mean()) - 10e-4 * turn
            rows.append({"test": name, "week": t1, "ret": r, "n": len(past)})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "r21_crypto.csv"), index=False)
    print(R.groupby("test").ret.agg(["mean", "count"]))


# ============================================================================================ combine
def cmd_combine(a):
    from scipy.stats import norm
    os.makedirs(a.out, exist_ok=True)
    fam, rows = [], []

    def add(tid, s_net, split, start, s_gross=None, extra=None):
        s_net = s_net.dropna()
        p1 = s_net[(s_net.index >= start) & (s_net.index < split)]
        p2 = s_net[s_net.index >= split]
        rec = {"id": tid, "p1_n": len(p1), "p1_mean": p1.mean(), "p1_t": tstat(p1), "p2_n": len(p2), "p2_mean": p2.mean(),
               "p2_t": tstat(p2)}
        if s_gross is not None:
            g2 = s_gross.dropna()
            rec["p2_gross_mean"] = g2[g2.index >= split].mean()
            rec["p2_gross_t"] = tstat(g2[g2.index >= split])
            g1 = g2[(g2.index >= start) & (g2.index < split)]
            rec["p1_gross_mean"] = g1.mean()
        if extra:
            rec.update(extra)
        rows.append(rec)
        fam.append({"id": tid, "t": rec["p2_t"], "own": bool(rec["p1_mean"] > 0 and rec["p2_mean"] > 0 and rec["p2_t"] >= 1.645)})

    S19 = pd.Timestamp("2019-01-01")
    for f in glob.glob(os.path.join(a.inp, "**", "r21_idx.parquet"), recursive=True) + \
            glob.glob(os.path.join(a.inp, "**", "r21_fx.parquet"), recursive=True):
        T = pd.read_parquet(f)
        T["day"] = pd.to_datetime(T.day)
        for test, g in T.groupby("test"):
            per_sym = test.startswith("I")
            groups = [(f"{test}-{'US500' if s == 'USA500IDXUSD' else 'NAS100'}", g[g.sym == s]) for s in sorted(g.sym.unique())] \
                if per_sym else [(test, g)]
            for tid, x in groups:
                dly = x.groupby("day")[["net", "gross", "net_alt"]].mean()
                start = pd.Timestamp("2012-01-01") if ("I" == test[0] or test in ("X1b_smt_idx", "X2_ndog")) else pd.Timestamp("2007-01-01")
                add(tid, dly.net, S19, start, dly.gross, extra={"trades": len(x), "p2_net_alt_bp": dly.net_alt[dly.index >= S19].mean() * 1e4,
                                                               "symbols": ",".join(sorted(x.sym.unique()))})
    f = glob.glob(os.path.join(a.inp, "**", "r21_macro.csv"), recursive=True)
    if f:
        M = pd.read_csv(f[0])
        M["month"] = pd.PeriodIndex(M.month, freq="M").to_timestamp()
        for test, g in M.groupby("test"):
            s = g.set_index("month").ret
            if test == "M0_nominal_carry":
                rows.append({"id": test, "note": "reported, not a test", "p1_mean": s[(s.index >= "1985-01-01") & (s.index < "2005-01-01")].mean(),
                             "p2_mean": s[s.index >= "2005-01-01"].mean(), "p2_t": tstat(s[s.index >= "2005-01-01"])})
                continue
            add(test, s, pd.Timestamp("2005-01-01"), pd.Timestamp("1985-01-01"))
    f = glob.glob(os.path.join(a.inp, "**", "r21_stocks.csv"), recursive=True)
    if f:
        Sx = pd.read_csv(f[0])
        Sx["month"] = pd.PeriodIndex(Sx.month, freq="M").to_timestamp()
        for test, g in Sx.groupby("test"):
            add(test, g.set_index("month").ret, pd.Timestamp("2015-01-01"), pd.Timestamp("2005-01-01"))
    f = glob.glob(os.path.join(a.inp, "**", "r21_deletions.csv"), recursive=True)
    if f:
        Dl = pd.read_csv(f[0], parse_dates=["entry", "exit"])
        m = Dl.groupby(Dl.exit.dt.to_period("M").dt.to_timestamp()).ret.mean()
        add("S6_deletion", m, pd.Timestamp("2015-01-01"), pd.Timestamp("2005-01-01"), extra={"events": len(Dl)})
    f = glob.glob(os.path.join(a.inp, "**", "r21_crypto.csv"), recursive=True)
    if f:
        K = pd.read_csv(f[0], parse_dates=["week"])
        for test, g in K.groupby("test"):
            add(test, g.set_index("week").ret, pd.Timestamp("2022-01-01"), pd.Timestamp("2019-01-01"))
    F = pd.DataFrame(fam)
    F["p"] = 1 - norm.cdf(F.t.fillna(-9).astype(float))
    o = np.argsort(F.p.to_numpy())
    thr = 0.10 * np.arange(1, len(F) + 1) / max(len(F), 1)
    passed = np.flatnonzero(F.p.to_numpy()[o] <= thr)
    F["bh"] = False
    if len(passed):
        F.loc[F.index[o[: passed.max() + 1]], "bh"] = True
    F["counts"] = F.own & F.bh
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "r21_rows.csv"), index=False)
    F.to_csv(os.path.join(a.out, "r21_family.csv"), index=False)
    json.dump({"tests": int(len(F)), "own": int(F.own.sum()), "bh": int(F.bh.sum()), "counts": int(F.counts.sum()),
               "family": F.round(4).to_dict("records")}, open(os.path.join(a.out, "r21_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(5).to_string())
    print(F.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", default="")
    ap.add_argument("--inp", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"macro": cmd_macro, "stocks": cmd_stocks, "cdata": cmd_cdata, "crypto": cmd_crypto, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    main()
