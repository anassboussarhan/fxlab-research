"""Round 18 addendum 1 (docs/round18_preregistration.md).

  c2b  --symbol IDX --out DIR   the VWAP trend rule decided every N minutes (pre-registered grid), one row per day
  gold --out DIR                exploratory: gold spread by New York hour; the Asian-hours trade with later entries
  r0x  --r17 DIR --out DIR      exploratory: where the intraday units' persistence comes from
  c2b-combine --inp DIR --family r18_family.csv --out DIR   selection on 2012-2018, verification on 2019-2025, BH
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = "2025-12-31"
SPLIT = pd.Timestamp("2019-01-01")
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
GRID = [(n, k0) for n in (15, 30, 60) for k0 in (0, 30)]


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) >= 20 and x.std() > 0 else np.nan


def cmd_c2b(a):
    from fxlab.data.store import load_m1
    from fxlab.execution.costs import BASE
    sf, sp, comm = BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4
    try:
        m1 = load_m1(a.symbol, "2007-01-01", DEV_END, columns=COLS + ["n_ticks"])
    except Exception:                                                       # noqa: BLE001
        m1 = load_m1(a.symbol, "2007-01-01", DEV_END, columns=COLS)
    pip = float(1e-4 * np.nanmedian(0.5 * (m1.bid_c + m1.ask_c)))
    ny = m1.index.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    sess = (mins >= 570) & (mins <= 960) & (ny.dayofweek < 5)
    x, nyx = m1[sess], ny[sess]
    has_ticks = "n_ticks" in x.columns and x["n_ticks"].notna().mean() > 0.9
    rows = []
    for day, g in x.groupby(nyx.normalize()):
        if len(g) < 300:
            continue
        mo = 0.5 * (g["bid_o"].to_numpy() + g["ask_o"].to_numpy())
        mc = 0.5 * (g["bid_c"].to_numpy() + g["ask_c"].to_numpy())
        spr = g["ask_o"].to_numpy() - g["bid_o"].to_numpy()
        if not (np.isfinite(mo).all() and np.isfinite(mc).all() and (mo > 0).all()):
            continue
        w = g["n_ticks"].to_numpy(float) if has_ticks else np.ones(len(g))
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        tp = (0.5 * (g["bid_h"].to_numpy() + g["ask_h"].to_numpy()) + 0.5 * (g["bid_l"].to_numpy() + g["ask_l"].to_numpy()) + mc) / 3
        vwap = np.cumsum(tp * w) / np.cumsum(w)
        sig = np.sign(mc - vwap)
        k = (g.index.tz_convert("America/New_York").hour * 60 + g.index.tz_convert("America/New_York").minute - 570).to_numpy()
        n = len(g)
        unit_cost = (0.5 * spr + sf * spr + sp * pip) / mo + comm
        ret = mo[1:] / mo[:-1] - 1.0
        day_ret = mc[-1] / mo[0] - 1.0
        for N, k0 in GRID:
            dec = (k >= k0) & ((k - k0) % N == 0)
            tgt = np.where(dec, sig, np.nan)
            held = pd.Series(tgt).ffill().fillna(0.0).to_numpy()            # the latest decision
            pos = np.zeros(n)
            pos[1:] = held[:-1]
            pos[-1] = 0.0
            dpos = np.abs(np.diff(np.r_[0.0, pos]))
            gross = float((pos[:-1] * ret).sum())
            rows.append({"test": f"C2b_N{N}_K{k0}", "day": pd.Timestamp(day.date()), "gross": gross,
                         "net": gross - float((dpos * unit_cost).sum()), "flips": float(dpos.sum()), "day_ret": day_ret,
                         "long_gross": float((np.clip(pos[:-1], 0, None) * ret).sum()),
                         "short_gross": float((np.clip(pos[:-1], None, 0) * ret).sum())})
    R = pd.DataFrame(rows)
    R["pair"] = a.symbol
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, f"c2b_{a.symbol}.parquet"))
    print(R.groupby("test")[["net", "gross", "flips"]].mean().round(6).to_string(), flush=True)


def cmd_c2b_combine(a):
    from scipy.stats import norm
    import r18_combine as C
    os.makedirs(a.out, exist_ok=True)
    R = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(a.inp, "**", "c2b_*.parquet"), recursive=True)])
    R["day"] = pd.to_datetime(R["day"])
    rows, chosen = [], []
    for sym, lab in (("USA500IDXUSD", "US500"), ("USATECHIDXUSD", "NAS100")):
        x = R[R.pair == sym]
        best = None
        for t, g in x.groupby("test"):
            p1, p2 = g[g.day < SPLIT], g[g.day >= SPLIT]
            rec = {"index": lab, "setting": t, "p1_n": len(p1), "p1_net_bp": p1.net.mean() * 1e4, "p1_net_t": tstat(p1.net),
                   "p1_gross_bp": p1.gross.mean() * 1e4, "p2_n": len(p2), "p2_net_bp": p2.net.mean() * 1e4,
                   "p2_net_t": tstat(p2.net), "p2_gross_bp": p2.gross.mean() * 1e4, "p2_gross_t": tstat(p2.gross),
                   "flips": g.flips.mean(), "cost_bp_per_unit": (g.gross.mean() - g.net.mean()) / g.flips.mean() * 1e4,
                   "p2_corr_with_index": float(np.corrcoef(p2.net, p2.day_ret)[0, 1]) if len(p2) > 20 else np.nan,
                   "p2_corr_with_abs_index": float(np.corrcoef(p2.net, p2.day_ret.abs())[0, 1]) if len(p2) > 20 else np.nan,
                   "p2_long_gross_bp": p2.long_gross.mean() * 1e4, "p2_short_gross_bp": p2.short_gross.mean() * 1e4,
                   "p2_sharpe": float(p2.net.mean() / p2.net.std() * math.sqrt(252)) if len(p2) > 20 else np.nan,
                   "p2_years_positive": f"{int((p2.groupby(p2.day.dt.year).net.sum() > 0).sum())}/{p2.day.dt.year.nunique()}"}
            rows.append(rec)
            if best is None or rec["p1_net_t"] > best["p1_net_t"]:
                best = rec
        chosen.append(best)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out, "c2b_surface.csv"), index=False)
    F = pd.read_csv(a.family)
    for b in chosen:
        F.loc[len(F)] = {"id": f"C2b-{b['index']}", "t": b["p2_net_t"], "mean_pos": b["p2_net_bp"] > 0,
                         "p1_ok": b["p1_net_bp"] > 0, "part": "C2b"}
    F["p"] = 1 - norm.cdf(F.t.fillna(-9).astype(float))
    F["bh_pass"] = C.bh(F.p.to_numpy())
    m = F.part == "C2b"
    F.loc[m, "own_rule"] = (F.t[m] >= 1.645) & F.mean_pos[m].astype(bool) & F.p1_ok[m].astype(bool)
    F.loc[m, "counts"] = F.own_rule[m].astype(bool) & F.bh_pass[m]
    F.to_csv(os.path.join(a.out, "r18_family_with_c2b.csv"), index=False)
    json.dump({"chosen": chosen, "family_c2b": F[m].to_dict("records"), "tests": int(len(F)),
               "bh_pass_total": int(F.bh_pass.sum())}, open(os.path.join(a.out, "c2b_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(S.round(2).to_string())
    print(F[m].round(4).to_string())


def cmd_gold(a):
    import flows_lab as FL
    import lit2_lab as L2
    from fxlab.data.instruments import get as instrument
    from fxlab.data.store import load_m1
    m1 = load_m1("XAUUSD", "2011-01-01", DEV_END, columns=COLS)
    ny = m1.index.tz_convert("America/New_York")
    mid = 0.5 * (m1.bid_o + m1.ask_o)
    sbp = ((m1.ask_o - m1.bid_o) / mid * 1e4)
    df = pd.DataFrame({"sbp": sbp.to_numpy(), "hour": ny.hour, "minute": ny.minute, "p2": m1.index >= SPLIT.tz_localize("UTC")})
    out = {"spread_bp_median_by_ny_hour": {}}
    for lab, m in (("2011-2018", ~df.p2), ("2019-2025", df.p2)):
        g = df[m]
        out["spread_bp_median_by_ny_hour"][lab] = {
            "all_minutes": g.groupby("hour").sbp.median().round(2).to_dict(),
            "first_minute_of_hour": g[g.minute == 0].groupby("hour").sbp.median().round(2).to_dict()}
    book = L2.metal_book(m1, instrument("XAUUSD").pip, "XAUUSD")
    start, end = str(m1.index[0].date()), str(m1.index[-1].date())
    rows = []
    for d in pd.bdate_range(start, end):
        for h in (18, 19, 20, 21):
            r = book.trade(FL.local_ts(d - pd.Timedelta(days=1), "America/New_York", h, 0), FL.local_ts(d, "Europe/London", 8, 0),
                           +1, tol_min=65 if h == 18 else 10)
            if r is not None:
                rows.append({"entry": h, "day": d, "net": r[0], "gross": r[1]})
        r = book.trade(FL.local_ts(d, "Europe/London", 8, 0), FL.local_ts(d, "America/New_York", 16, 55), +1, tol_min=10)
        if r is not None:
            rows.append({"entry": "west_long", "day": d, "net": r[0], "gross": r[1]})
    T = pd.DataFrame(rows)
    res = {}
    for e, g in T.groupby("entry"):
        for lab, m in (("2011-2018", g.day < SPLIT), ("2019-2025", g.day >= SPLIT)):
            x = g[m]
            res[f"{e}|{lab}"] = {"n": int(len(x)), "net_bp": float(x.net.mean() * 1e4), "net_t": tstat(x.net),
                                 "gross_bp": float(x.gross.mean() * 1e4), "gross_t": tstat(x.gross),
                                 "cost_bp": float((x.gross.mean() - x.net.mean()) * 1e4)}
    out["asian_hours_by_entry"] = res
    # buy-and-hold for comparison: mean daily close-to-close return, 2019-2025
    dly = mid.resample("1D").last().dropna()
    r = dly.pct_change().dropna()
    out["buy_hold_daily_bp_2019_2025"] = float(r[r.index >= SPLIT.tz_localize("UTC")].mean() * 1e4)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "gold_costs.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str), flush=True)


def cmd_r0x(a):
    import r18_regime0 as R0
    import regime17 as G17
    D = G17.load(a.r17, "BOOK-UNIT", "D")
    eng = D.tier.desc["engine"].to_numpy()
    tf = D.tier.desc["unit"].str.split("|").str[-1].to_numpy()
    ptf = D.ptier.desc["unit"].str.split("|").str[-1].to_numpy()
    peng = D.ptier.desc["engine"].to_numpy()
    rng = np.random.default_rng(181)
    KD = (5, 21, 63, 126, 252)
    out = {}
    log = lambda s: print(s, flush=True)                                   # noqa: E731
    out["placebo_intraday"] = R0.test("intraday placebo units", D.ptier.z[:, peng == "I"], KD, 63, 21, 200, rng, log)
    out["placebo_daily_engine"] = R0.test("daily-engine placebo units", D.ptier.z[:, peng == "D"], KD, 63, 21, 200, rng, log)
    # remove the cost-regime proxy: the placebo units' mean daily z at the same timeframe
    zi = D.tier.z[:, eng == "I"].astype(np.float64)
    tfi = tf[eng == "I"]
    adj = np.full_like(zi, np.nan)
    betas = []
    for t in sorted(set(tfi)):
        mp = np.nanmean(D.ptier.z[:, (peng == "I") & (ptf == t)], axis=1)
        cols = np.flatnonzero(tfi == t)
        for c in cols:
            y = zi[:, c]
            ok = np.isfinite(y) & np.isfinite(mp)
            if ok.sum() < 500:
                continue
            b = np.cov(y[ok], mp[ok])[0, 1] / mp[ok].var()
            betas.append(b)
            adj[:, c] = y - b * mp
    out["intraday_minus_placebo_factor"] = R0.test("intraday units minus the placebo factor", adj, KD, 63, 21, 200, rng, log)
    out["median_beta_to_placebo_factor"] = float(np.median(betas))
    # the common factor of the intraday units and its states
    m = pd.Series(np.nanmean(zi, axis=1), index=D.ends)
    m = m[m.notna()]
    out["mean_z_by_year"] = m.groupby(m.index.year).mean().round(4).to_dict()
    out["share_of_days_positive"] = float((m > 0).mean())
    state = m.rolling(63, min_periods=50).mean().shift(1)
    q = pd.qcut(state.dropna(), 5, labels=False)
    nxt = m.reindex(q.index)
    out["next_day_mean_z_by_state_quintile"] = {int(k): round(float(v), 4) for k, v in nxt.groupby(q).mean().items()}
    out["state_quintile_bounds"] = [round(float(v), 4) for v in state.dropna().quantile([0, .2, .4, .6, .8, 1]).to_numpy()]
    # per-unit: the share of intraday units with a positive mean in their best state quintile
    zz = pd.DataFrame(zi, index=D.ends).reindex(q.index)
    best = zz[q == 4].mean()
    out["units_positive_in_best_state"] = {"share": float((best > 0).mean()), "share_t_ge_1645": float(
        ((best / (zz[q == 4].std() / math.sqrt((q == 4).sum()))) >= 1.645).mean()), "median_mean_z": float(best.median())}
    all_ = zz.mean()
    out["units_positive_overall"] = {"share": float((all_ > 0).mean()), "median_mean_z": float(all_.median())}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "r0x.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or "library" not in v}, indent=1, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["c2b", "c2b-combine", "gold", "r0x"])
    ap.add_argument("--symbol", default="")
    ap.add_argument("--inp", default="")
    ap.add_argument("--family", default="")
    ap.add_argument("--r17", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"c2b": cmd_c2b, "c2b-combine": cmd_c2b_combine, "gold": cmd_gold, "r0x": cmd_r0x}[a.cmd](a)


if __name__ == "__main__":
    main()
