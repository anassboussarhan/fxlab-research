"""Round 23 part C30 (docs/round23_preregistration.md): monthly currency signals on every ECB-quoted currency with an
OECD short rate, 1999-2025, Sobol 256, default and nested walk-forward.

Spot: ECB reference rates (USD per unit = EURUSD / EURxxx), last fixing of the month. Short rates: OECD MEI 3-month
interbank, else call money (DBnomics), per cent a year, used with a one-month lag as in round 22. CPI: OECD MEI index.
Return of holding currency c over month m+1: P_c(m+1) / P_c(m) - 1 + (r_c - r_USD) / 12.
Usage: python scripts/r23_C30.py --out DIR
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wf22  # noqa: E402
import wf23  # noqa: E402

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OECD = {"EUR": "EA19", "JPY": "JPN", "GBP": "GBR", "CHF": "CHE", "AUD": "AUS", "NZD": "NZL", "CAD": "CAN", "SEK": "SWE",
        "NOK": "NOR", "CZK": "CZE", "HUF": "HUN", "PLN": "POL", "RON": "ROU", "ISK": "ISL", "TRY": "TUR", "BRL": "BRA",
        "CNY": "CHN", "IDR": "IDN", "ILS": "ISR", "INR": "IND", "KRW": "KOR", "MXN": "MEX", "ZAR": "ZAF", "RUB": "RUS",
        "USD": "USA"}
G10 = {"EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"}
DEV_END = pd.Period("2025-12", freq="M")


def get(url, tries=6, timeout=180):
    import requests
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 300:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as ex:                                            # noqa: BLE001
            last = repr(ex)[:160]
        time.sleep(min(300, 10 * 2 ** k))
    raise RuntimeError(f"{url}: {last}")


def ecb_monthly():
    r = get("https://data-api.ecb.europa.eu/service/data/EXR/D..EUR.SP00.A?format=csvdata&startPeriod=1999-01-01"
            "&endPeriod=2025-12-31")
    raw = pd.read_csv(io.StringIO(r.text), usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    raw["TIME_PERIOD"] = pd.to_datetime(raw.TIME_PERIOD)
    X = raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
    Xm = X.resample("ME").last()
    Xm.index = Xm.index.to_period("M")
    P = pd.DataFrame({c: Xm["USD"] / Xm[c] for c in OECD if c in Xm and c != "USD"})
    P["EUR"] = Xm["USD"]
    return P


def oecd(subjects, measure, code):
    for subj in subjects:
        try:
            doc = get(f"https://api.db.nomics.world/v22/series/OECD/MEI/{code}.{subj}.{measure}.M?observations=1",
                      tries=3).json()["series"]["docs"][0]
            return pd.Series(pd.to_numeric(pd.Series(doc["value"]), errors="coerce").to_numpy(),
                             index=pd.PeriodIndex(doc["period"], freq="M")).dropna(), subj
        except Exception:                                                  # noqa: BLE001
            continue
    return None, None


def vix_monthly():
    r = get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS")
    v = pd.read_csv(io.StringIO(r.text))
    v.columns = ["d", "v"]
    v["d"] = pd.to_datetime(v.d)
    v["v"] = pd.to_numeric(v.v, errors="coerce")
    s = v.set_index("d").v.dropna().resample("ME").last()
    s.index = s.index.to_period("M")
    return s


def book(Z, R, n, reb, cost_c, flat=None):
    out, w_prev = [], pd.Series(0.0, index=Z.columns)
    w = w_prev.copy()
    for i, m in enumerate(Z.index[:-1]):
        if i % reb == 0:
            z = Z.loc[m].dropna()
            z = z[R.loc[m].reindex(z.index).notna()]
            w = pd.Series(0.0, index=Z.columns)
            if len(z) >= 2 * n:
                o = z.sort_values()
                w[o.index[-n:]] = 1.0 / n
                w[o.index[:n]] = -1.0 / n
        wx = w * 0.0 if (flat is not None and bool(flat.get(m, False))) else w
        turn = (wx - w_prev).abs()
        w_prev = wx
        out.append(float((wx * R.loc[m].fillna(0.0)).sum()) - float((turn * cost_c).sum()))
    return pd.Series(out, index=Z.index[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = ecb_monthly()
    S, CPI, src = {}, {}, {}
    for c, code in OECD.items():
        codes = [code] if c != "EUR" else ["EA19", "EA20", "EA"]
        for cc in codes:
            s, subj = oecd(("IR3TIB01", "IRSTCI01"), "ST", cc)
            if s is not None:
                S[c], src[c] = s / 100.0, subj
                break
        for cc in codes:
            x, _ = oecd(("CPALTT01",), "IXOB", cc)
            if x is not None:
                CPI[c] = x
                break
    S, CPI = pd.DataFrame(S), pd.DataFrame(CPI)
    months = pd.period_range("1999-01", "2025-12", freq="M")
    S, CPI, P = (z.reindex(months) for z in (S, CPI, P))
    cur = [c for c in P.columns if c in S and S[c].notna().sum() > 36 and "USD" in S]
    P, S1 = P[cur], S[cur].shift(1)
    Susd = S["USD"].shift(1)
    R = (P.shift(-1) / P - 1) + S1.sub(Susd, axis=0) / 12.0
    realP = np.log(P * (CPI[cur] / CPI["USD"].to_numpy()[:, None]).reindex(columns=cur))
    vix = vix_monthly().reindex(months)
    cost = pd.Series({c: 5e-4 if c in G10 else 15e-4 for c in cur})
    U = qmc.Sobol(9, scramble=True, seed=2301).random(256)
    M, desc = [], []
    for u in U:
        sig = ["carry", "real_carry", "mom", "value", "composite"][min(4, int(u[0] * 5))]
        lb, sk = int(1 + u[1] * 11.999), int(u[2] * 1.999)
        vl, n = int(36 + u[3] * 48.999), int(1 + u[4] * 4.999)
        reb, lag = int(1 + u[5] * 2.999), int(1 + u[6] * 1.999)
        thr = None if u[7] < 0.5 else 15 + (u[7] - 0.5) * 2 * 25
        g10 = u[8] < 0.5
        cols = [c for c in cur if (c in G10 or not g10)]
        Sx, Cx = S1[cols].shift(lag - 1), CPI[cols].pct_change(12).shift(lag + 1)
        mom = np.log(P[cols]).shift(sk) - np.log(P[cols]).shift(sk + lb)
        val = -(realP[cols] - realP[cols].shift(vl))
        comps = {"carry": Sx.sub(Susd.shift(lag - 1), axis=0), "real_carry": Sx - Cx, "mom": mom, "value": val}
        Z = (sum(v.rank(axis=1, pct=True) for v in comps.values()) / 4) if sig == "composite" else comps[sig]
        flat = (vix.shift(lag - 1) > thr) if thr is not None else None
        M.append(book(Z, R[cols], n, reb, cost[cols], flat))
        desc.append({"sig": sig, "lb": lb, "skip": sk, "vl": vl, "n": n, "reb": reb, "lag": lag, "vix": thr, "g10": g10})
    W = pd.concat(M, axis=1)
    W.index = W.index.to_timestamp(how="end").normalize()
    W = W[W.index <= wf22.DEV_END]
    ag = wf23.Agg(W.index, P=W.to_numpy().T)
    res = wf23.both(ag, 2005, min_active=24, min_valid=12)
    rows, ser = [], {}
    for kind, (s, st, own, ch) in res.items():
        rows.append({"part": "C30", "family": f"C30-fx-{kind}", "symbol": f"{len(cur)}cur", "placebo": False,
                     "selection": kind, "n_cfg": len(U), **st, "own": own,
                     "choices": json.dumps({k: (v if isinstance(v, str) else [desc[i] for i in v][:3]) for k, v in list(ch.items())[-3:]},
                                           default=str)[:2000]})
        ser[f"C30_{kind}"] = s
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "C30_rows.csv"), index=False)
    pd.DataFrame(ser).to_parquet(os.path.join(a.out, "C30_series.parquet"))
    json.dump({"currencies": cur, "rate_source": src, "first_month_with_rate": {c: str(S[c].first_valid_index()) for c in cur},
               "cpi": list(CPI.columns)}, open(os.path.join(a.out, "C30_meta.json"), "w"), indent=1, default=str)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("family", "oos_first", "p1_mean", "p2_mean", "p2_t", "own")}, default=str))


if __name__ == "__main__":
    main()
