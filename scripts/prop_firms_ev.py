"""Expected value of a prop-firm challenge per attempt, for every edge that has survived costs, alone and mixed, at
several leverage levels, under the rules of four 100k programmes. Extends scripts/prop_mix_sim.py (FTMO 2-step
only) to other firms and to the new flow edges. Development data only (2019-2025 daily outcomes; 2026 untouched).

RULES (100k accounts; third-party summaries of the firms' published terms, checked 2026-01-15 to 2026-09-01:
luxalgo.com prop-firm-sim dataset, tradetanto.com, proptradingvibes.com; confirm on the firms' own pages):
  FTMO 2-step:        fee EUR 540 (USD 585), refunded with the first payout; +10% then +5%; daily loss 5% and
                      maximum loss 10% of the initial balance (static); 4 trading days a phase; 80% split; payouts
                      every 10 trading days.
  FTMO 1-step:        fee EUR 499 (USD 540), not refunded; +10% in one phase; daily loss 3%; maximum loss 10%
                      trailing from the highest end-of-day balance (no lock assumed); best day at most 50% of the
                      positive days' profit when passing; 90% split; payouts every 10 trading days.
  FundedNext Stellar 2-step: fee USD 549, refunded with the first payout; +8% then +5%; daily 5%, maximum 10%
                      static; 80% split; first payout 15 trading days after funding, then every 10.
  The5ers High Stakes: fee USD 495, not refunded; +8% then +5%; daily 5%, maximum 10% static; at least 3 days a
                      phase with a closed profit of 0.5% or more; 80% split; payouts every 10 trading days.
  All: a trader who has not passed within 500 trading days is counted as not passed; funded for 250 trading days.
  Losses are checked after each strategy's daily outcome (the open-trade path inside a trade is not modelled,
  which slightly favours the trader).
EDGES (per unit of notional, net of costs and swap, daily outcomes 2019-2025):
  G   Gotobi USD/JPY 03:00-09:55 Tokyo (gotobi-oos branch, test F1);
  I3  Gotobi USD/JPY overnight 23:00-09:53 Tokyo with a 0.5-range stop (untested-intraday branch);
  E   Gotobi EUR/JPY (cost-tuning branch, out-of-sample walk-forward trades);
  Y   the yen-basket Gotobi overnight trade (flow-edges branch, T3), when available;
  O   Nasdaq-100 overnight on an index CFD (QQQ close-to-open proxy, Mon-Thu nights), as in prop_mix_sim.py.
Each of N simulated traders starts on a random day and walks the real calendar (wrapping around).
Usage: python scripts/prop_firms_ev.py --series DIR --out DIR"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-31")
FIRMS = {
    "FTMO 2-step": dict(fee=585.0, targets=(0.10, 0.05), daily=0.05, maxloss=0.10, trailing=False, min_days=4,
                        profit_day=0.0, min_profit_days=0, best_day=None, split=0.80, refund=True, first_pay=10,
                        pay_every=10),
    "FTMO 1-step": dict(fee=540.0, targets=(0.10,), daily=0.03, maxloss=0.10, trailing=True, min_days=0,
                        profit_day=0.0, min_profit_days=0, best_day=0.5, split=0.90, refund=False, first_pay=10,
                        pay_every=10),
    "FundedNext Stellar 2-step": dict(fee=549.0, targets=(0.08, 0.05), daily=0.05, maxloss=0.10, trailing=False,
                                      min_days=0, profit_day=0.0, min_profit_days=0, best_day=None, split=0.80,
                                      refund=True, first_pay=15, pay_every=10),
    "The5ers High Stakes": dict(fee=495.0, targets=(0.08, 0.05), daily=0.05, maxloss=0.10, trailing=False,
                                min_days=0, profit_day=0.005, min_profit_days=3, best_day=None, split=0.80,
                                refund=False, first_pay=10, pay_every=10),
}
INIT, MAX_DAYS, FUNDED_DAYS = 100_000.0, 500, 250


def simulate(A, rules, n=20_000, seed=0):
    """A: (days x strategies) daily account returns at the chosen leverage. Returns the outcome summary."""
    r = rules
    rng = np.random.default_rng(seed)
    T = len(A)
    pos = rng.integers(0, T, size=n)
    nph = len(r["targets"])
    phase = np.zeros(n, int)                 # 0..nph-1 evaluation, nph funded, -1 failed, -2 finished
    eq = np.zeros(n)
    peak = np.zeros(n)                       # highest end-of-day equity (trailing rule)
    days = np.zeros(n, int)
    pdays = np.zeros(n, int)
    best = np.zeros(n)
    possum = np.zeros(n)
    fday = np.zeros(n, int)
    paid = np.zeros(n)
    npay = np.zeros(n, int)
    fail = np.zeros(n, int)                  # 1 daily, 2 max loss
    passed = np.zeros(n, bool)
    for step in range(MAX_DAYS + FUNDED_DAYS):
        alive = phase >= 0
        if not alive.any():
            break
        day = A[(pos + step) % T]
        start = eq.copy()
        floor = np.where(r["trailing"], peak - r["maxloss"], -r["maxloss"])
        for c in range(day.shape[1]):
            eq = np.where(alive, eq + day[:, c], eq)
            bd = alive & (start - eq >= r["daily"])
            bm = alive & (eq <= floor)
            br = bd | bm
            fail = np.where(br & (fail == 0), np.where(bd, 1, 2), fail)
            phase = np.where(br, -1, phase)
            alive &= ~br
        dp = eq - start
        traded = np.abs(day).sum(axis=1) > 0
        ev = alive & (phase < nph)
        days = np.where(ev & traded, days + 1, days)
        pdays = np.where(ev & (dp >= max(r["profit_day"], 1e-12)) & (r["profit_day"] > 0), pdays + 1, pdays)
        best = np.where(ev, np.maximum(best, dp), best)
        possum = np.where(ev & (dp > 0), possum + dp, possum)
        peak = np.where(alive, np.maximum(peak, eq), peak)
        tgt = np.array(list(r["targets"]) + [np.inf])[np.clip(phase, 0, nph)]
        ok = ev & (eq >= tgt) & (days >= r["min_days"]) & (pdays >= r["min_profit_days"])
        if r["best_day"] is not None:
            ok &= best <= r["best_day"] * np.maximum(possum, 1e-12)
        phase = np.where(ok, phase + 1, phase)
        reset = ok
        eq = np.where(reset, 0.0, eq)
        peak = np.where(reset, 0.0, peak)
        days = np.where(reset, 0, days)
        pdays = np.where(reset, 0, pdays)
        best = np.where(reset, 0.0, best)
        possum = np.where(reset, 0.0, possum)
        newly = ok & (phase == nph)
        passed |= newly
        fday = np.where(newly, 0, fday)
        f = alive & (phase == nph) & ~newly
        fday = np.where(f, fday + 1, fday)
        due = f & (fday >= r["first_pay"]) & ((fday - r["first_pay"]) % r["pay_every"] == 0) & (eq > 0)
        paid = np.where(due, paid + r["split"] * eq * INIT, paid)
        npay = np.where(due, npay + 1, npay)
        eq = np.where(due, 0.0, eq)
        peak = np.where(due, 0.0, peak)
        phase = np.where(f & (fday >= FUNDED_DAYS), -2, phase)
        phase = np.where((phase >= 0) & (phase < nph) & (step + 1 >= MAX_DAYS), -2, phase)
    refund = np.where((npay > 0) & r["refund"], r["fee"], 0.0)
    net = paid + refund - r["fee"]
    return {"pass": float(passed.mean()), "paid": float((npay > 0).mean()), "mean_net": float(net.mean()),
            "median_net": float(np.median(net)), "p10_net": float(np.percentile(net, 10)),
            "p90_net": float(np.percentile(net, 90)), "share_positive": float((net > 0).mean()),
            "fail_daily": float((fail == 1).mean()), "fail_max": float((fail == 2).mean())}


def series(d):
    """Daily per-unit outcomes of each edge, 2019-2025 business days (0 on days without a trade)."""
    out = {}
    f = os.path.join(d, "gotobi_series.csv")
    if os.path.exists(f):
        G = pd.read_csv(f, parse_dates=["day"])
        out["G"] = G[(G.test == "F1") & G.net_sw.notna()].set_index("day").net_sw
    for key, pat in (("I3", "trades_I3_USDJPY*.csv"), ("E", "oos_eurjpy_gotobi_EURJPY.csv"),
                     ("Y", "trades_T3_yen_basket*.csv")):
        fs = glob.glob(os.path.join(d, "**", pat), recursive=True)
        if fs:
            s = pd.read_csv(fs[0], index_col=0, parse_dates=True)["net"]
            s.index = pd.DatetimeIndex(s.index).tz_localize(None) if s.index.tz is not None else s.index
            out[key] = s
    try:
        import prop_mix_sim as PM
        q, irx = PM.yahoo("QQQ"), PM.yahoo("^IRX")
        fq = q["Adj Close"] / q["Close"]
        o, c = q["Open"] * fq, q["Adj Close"]
        rf = (irx["Close"] / 100).reindex(q.index).ffill().fillna(0)
        night = o / c.shift(1) - 1
        wk = q.index.to_series().diff().dt.days == 1
        out["O"] = (night - 1e-4 - (rf.shift(1) + 0.025) / 360)[wk].dropna()
    except Exception as ex:                                               # noqa: BLE001
        print("no QQQ series:", repr(ex)[:200], flush=True)
    days = pd.bdate_range(START, END)
    X = pd.DataFrame({k: v.groupby(v.index.normalize()).sum().reindex(days) for k, v in out.items()}).fillna(0.0)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traders", type=int, default=20_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = series(a.series)
    stats = {k: {"events": int((X[k] != 0).sum()), "mean_bp": float(X[k][X[k] != 0].mean() * 1e4),
                 "worst_bp": float(X[k].min() * 1e4), "year_at_1x": float(X[k].sum() / 7)} for k in X}
    print(json.dumps(stats, indent=1), flush=True)
    levs = {"G": (3, 6, 9, 12), "I3": (3, 6, 9, 12), "E": (3, 6, 9, 12), "Y": (3, 6, 9, 12), "O": (0.5, 1, 1.5)}
    combos = []
    for k in X.columns:
        combos += [((k,), (L,)) for L in levs[k]]
    yen = [k for k in ("I3", "E", "Y") if k in X]
    for k in yen:
        if "O" in X:
            combos += [((k, "O"), (L, lo)) for L in (3, 6, 9) for lo in (0.5, 1)]
    if len(yen) >= 2:
        combos += [((yen[0], yen[1]), (L, L)) for L in (3, 5, 7)]
        if "O" in X:
            combos += [((yen[0], yen[1], "O"), (L, L, lo)) for L in (3, 5) for lo in (0.5, 1)]
    rows = []
    for (keys, lev), (fname, rules) in itertools.product(combos, FIRMS.items()):
        A = X[list(keys)].to_numpy() * np.asarray(lev)[None, :]
        res = simulate(A, rules, n=a.traders)
        rows.append({"mix": " + ".join(f"{k} {l:g}x" for k, l in zip(keys, lev)), "firm": fname, **res})
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "ev.csv"), index=False)
    best = R.sort_values("mean_net", ascending=False).groupby("firm").head(5)
    best.to_csv(os.path.join(a.out, "best_per_firm.csv"), index=False)
    json.dump(stats, open(os.path.join(a.out, "inputs.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(best.round(3).to_string())


if __name__ == "__main__":
    main()
