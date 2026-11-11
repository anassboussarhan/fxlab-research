"""Round 18 combine (docs/round18_preregistration.md): statistics of parts G and C by period, the gold anomaly
test, and one Benjamini-Hochberg correction over the round-18 family (with the part-L tests).
Usage: python scripts/r18_combine.py --inp DIR --long DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

SPLIT = pd.Timestamp("2019-01-01")


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std() == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def bh(p, q=0.10):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    ok = np.flatnonzero(np.isfinite(p))
    o = ok[np.argsort(p[ok])]
    thr = q * np.arange(1, len(ok) + 1) / len(ok)
    passed = np.flatnonzero(p[o] <= thr)
    if len(passed):
        out[o[: passed.max() + 1]] = True
    return out


def periods(s_net, s_gross):
    """s_net, s_gross: daily Series (index = day). Stats in bp for 2007-2018 and 2019-2025."""
    out = {}
    for lab, m in (("p1", s_net.index < SPLIT), ("p2", s_net.index >= SPLIT)):
        n, g = s_net[m], s_gross[m]
        out[lab] = {"n": int(n.notna().sum()), "net_bp": float(n.mean() * 1e4) if len(n) else np.nan, "net_t": tstat(n),
                    "gross_bp": float(g.mean() * 1e4) if len(g) else np.nan, "gross_t": tstat(g)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--long", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.inp, "**", "r18_*.parquet"), recursive=True))],
                  ignore_index=True)
    R["day"] = pd.to_datetime(R["day"])
    for c in ("swap_rate", "swap_flat"):
        if c not in R:
            R[c] = np.nan
    swap = R["swap_rate"].where(R["swap_rate"].notna(), R["swap_flat"]).fillna(0.0)
    R["net_total"] = R["net"] + swap
    cover = R.groupby("pair").day.agg(["min", "max", "count"]).astype(str).to_dict("index")
    rows, fam = [], []

    def daily(test, pair=None, mask=None):
        x = R[R.test == test]
        if pair:
            x = x[x.pair == pair]
        if mask is not None:
            x = x[mask(x)]
        g = x.groupby("day")
        return g["net_total"].mean(), g["gross"].mean()

    def add(tid, label, net, gross, in_family=True, need_p1_gross_pos=True, extra=None):
        st = periods(net, gross)
        rec = {"id": tid, "label": label, **{f"{p}_{k}": v for p, d in st.items() for k, v in d.items()}}
        if extra:
            rec.update(extra)
        rows.append(rec)
        if in_family:
            fam.append({"id": tid, "t": st["p2"]["net_t"], "mean_pos": st["p2"]["net_bp"] > 0,
                        "p1_ok": (st["p1"]["gross_bp"] > 0) if need_p1_gross_pos else True})
        return st

    n, g = daily("A1", "XAUUSD")
    add("G-A1", "gold, Asian hours, long 18:00 New York to 08:00 London", n, g)
    n, g = daily("A1", "XAUUSD", lambda x: x.day.dt.dayofweek != 0)
    add("G-A1x", "the same without Sunday-evening entries", n, g)
    n2, g2 = daily("A2", "XAUUSD")
    add("G-A2", "gold, Western hours, short 08:00 London to 16:55 New York", n2, g2)
    # the anomaly: Asian-hours gross minus Western-hours long gross (= A1 gross + A2 gross, A2 being the short)
    for tid, mask in (("G-A1", None), ("G-A1x", lambda x: x.day.dt.dayofweek != 0)):
        na, ga = daily("A1", "XAUUSD", mask)
        d = (ga + g2.reindex(ga.index)).dropna()
        for r in rows:
            if r["id"] == tid:
                r["anomaly_p1_t"], r["anomaly_p2_t"] = tstat(d[d.index < SPLIT]), tstat(d[d.index >= SPLIT])
                r["anomaly_p2_bp"] = float(d[d.index >= SPLIT].mean() * 1e4)
                r["western_long_gross_p2_bp"] = float(-g2[g2.index >= SPLIT].mean() * 1e4)
    add("G-G2", "gold AM fix, short 10:15 to 10:45 London", *daily("G2", "XAUUSD"))
    add("G-G1", "gold PM fix, short 14:45 to 15:15 London", *daily("G1", "XAUUSD"))
    add("G-S1", "silver fix, short 11:45 to 12:15 London", *daily("S1", "XAGUSD"))
    add("G-L1", "sell each currency in its home hours, 7 USD pairs", *daily("L1"))
    kn, kg = daily("K_ny")
    on, og = daily("K_on")
    add("G-K", "around the clock: long foreign in New York hours, short overnight (both legs)",
        kn.add(on, fill_value=0.0), kg.add(og, fill_value=0.0))
    add("G-K_ny", "New York-hours leg alone", kn, kg, in_family=False)
    add("G-K_on", "overnight leg alone", on, og, in_family=False)
    add("G-W1", "fade the 15:30-16:00 London move after the fix", *daily("W1"))
    # C1: overnight-to-intraday reversal across the seven currencies
    N = R[R.test == "N_night"].pivot_table(index="day", columns="ccy", values="gross")
    DL = R[R.test == "D_long"].pivot_table(index="day", columns="ccy", values="net_total")
    DS = R[R.test == "D_short"].pivot_table(index="day", columns="ccy", values="net_total")
    DG = R[R.test == "D_long"].pivot_table(index="day", columns="ccy", values="gross")
    ok = (N.notna() & DL.notna() & DS.notna()).reindex_like(N)
    N = N.where(ok)
    dev = N.sub(N.mean(axis=1), axis=0)
    W = -dev.div(dev.abs().sum(axis=1), axis=0)
    W = W[N.notna().sum(axis=1) >= 5]
    net = (W.clip(lower=0) * DL.reindex_like(W)).sum(axis=1) + ((-W).clip(lower=0) * DS.reindex_like(W)).sum(axis=1)
    gross = (W * DG.reindex_like(W)).sum(axis=1)
    add("C1", "overnight-to-intraday reversal across 7 currencies", net, gross, need_p1_gross_pos=False,
        extra={"p1_net_pos_required": True})
    fam[-1]["p1_ok"] = bool(net[net.index < SPLIT].mean() > 0)
    S = -np.sign(N)
    S = S[N.notna().sum(axis=1) >= 5]
    cnt = S.abs().sum(axis=1)
    net_ts = ((S.clip(lower=0) * DL.reindex_like(S)).sum(axis=1) + ((-S).clip(lower=0) * DS.reindex_like(S)).sum(axis=1)) / cnt
    add("C1-ts", "each pair fades its own overnight move (reported)", net_ts, (S * DG.reindex_like(S)).sum(axis=1) / cnt,
        in_family=False)
    # C2: VWAP trend
    for sym, lab in (("USA500IDXUSD", "US500"), ("USATECHIDXUSD", "NAS100")):
        x = R[(R.test == "C2_vwap") & (R.pair == sym)]
        if len(x):
            x = x.set_index("day")
            add(f"C2-{lab}", f"VWAP trend, {lab} CFD", x["net"], x["gross"], need_p1_gross_pos=False,
                extra={"flips_per_day": float(x["flips"].mean()), "tick_weights": bool(x["ticks"].iloc[0])})
    F = pd.DataFrame(fam)
    F["part"] = "M1"
    # part L
    longf = glob.glob(os.path.join(a.long, "**", "long_summary.json"), recursive=True) if a.long else []
    L = json.load(open(longf[0])) if longf else None
    if L:
        for r in L["L2"]:
            F.loc[len(F)] = {"id": f"L2-{r['family']}", "t": r["verify_t_1981_1998"], "mean_pos": r["verify_mean"] > 0,
                             "p1_ok": True, "part": "L"}
        l3 = L["L3"]
        F.loc[len(F)] = {"id": "L3", "t": l3["1981-2025"]["timed_t"],
                         "mean_pos": l3["1981-1998"]["timed_mean"] > 0 and l3["1999-2025"]["timed_mean"] > 0, "p1_ok": True,
                         "part": "L"}
    F["p"] = 1 - norm.cdf(F.t.fillna(-9).astype(float))
    F["bh_pass"] = bh(F.p.to_numpy())
    F["own_rule"] = (F.t >= 1.645) & F.mean_pos.astype(bool) & F.p1_ok.astype(bool)
    F["counts"] = F.own_rule & F.bh_pass
    T = pd.DataFrame(rows)
    # extra conditions
    for i, r in F.iterrows():
        if r.id in ("G-A1", "G-A1x"):
            an = float(T.loc[T.id == r.id, "anomaly_p2_t"].iloc[0])
            F.loc[i, "anomaly_t"] = an
            F.loc[i, "counts"] = bool(r.counts and an >= 1.645)
        if L and r.id.startswith("L2-"):
            rr = [x for x in L["L2"] if f"L2-{x['family']}" == r.id][0]
            okx = rr["minus_b1_t"] >= 1.645
            if rr["family"] == "M3":
                okx = okx and bool(rr.get("P3"))
            if rr["family"] == "C1":
                okx = okx and rr["minus_always_trend_t"] >= 1.645
            F.loc[i, "counts"] = bool(r.counts and okx)
    T.to_csv(os.path.join(a.out, "r18_rows.csv"), index=False)
    F.to_csv(os.path.join(a.out, "r18_family.csv"), index=False)
    summ = {"family_tests": int(len(F)), "counts": int(F.counts.sum()), "bh_pass": int(F.bh_pass.sum()),
            "first_bh_threshold": 0.10 / max(len(F), 1), "counting": F[F.counts].to_dict("records"),
            "family": F.round(4).to_dict("records"), "rows": T.round(4).to_dict("records"), "coverage": cover}
    json.dump(summ, open(os.path.join(a.out, "round18_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(T[["id", "p1_n", "p1_net_bp", "p1_gross_bp", "p1_gross_t", "p2_n", "p2_net_bp", "p2_net_t", "p2_gross_bp", "p2_gross_t"]].round(2).to_string())
    print(F.round(4).to_string())


if __name__ == "__main__":
    main()
