"""The Gotobi overnight trade (I3: long USD/JPY 23:00 Tokyo -> 09:53 Tokyo on Gotobi days, stop 0.5 x the 20-day
range; net of spread, slippage, commission and swap) played for maximum profit at high risk, two ways:
  A. A personal account compounding at constant leverage L (the position is L x current equity on every trade).
  B. Prop-firm challenges, where the most that can be lost is the fee: leverage while passing (Le) and once funded
     (Lf) chosen separately.
Every trade's worst open loss (MAE) is applied inside the trade: in A a trade that would take the account below
10% of its start is a wipe-out (margin call); in B the daily-loss and maximum-loss limits are checked against
the open loss, not only the close. Trades are drawn at random (with replacement) from the 697 trades of
2007-2018, the years whose minute paths are stored here; 2019-2025 had a similar mean (+4.2 bp against +4.6).
"Haircut" scenarios subtract 2 bp from every trade (an edge half as large: decay, worse costs or luck).
Development data only; 2026 is untouched. Usage: python scripts/gotobi_lottery.py --mae CSV --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prop_firms_ev import FIRMS, INIT  # noqa: E402

TRADES_PER_YEAR = 58          # Gotobi days with a trade (Mondays have no evening-before entry)
P_DAY = TRADES_PER_YEAR / 252


def account(net, mae, L, years, n=20_000, seed=0, start=10_000.0):
    rng = np.random.default_rng(seed)
    k = TRADES_PER_YEAR * years
    ix = rng.integers(0, len(net), size=(n, k))
    r, m = net[ix] * L, mae[ix] * L
    eq = np.full(n, start)
    peak = eq.copy()
    mdd = np.zeros(n)
    dead = np.zeros(n, bool)
    for t in range(k):
        low = eq * (1 + m[:, t])                                   # the open loss inside the trade
        new = ~dead & (low <= 0.10 * start)                         # margin call: closed at the low
        mdd = np.where(dead, mdd, np.maximum(mdd, 1 - low / np.maximum(peak, 1e-9)))
        eq = np.where(new, np.maximum(low, 0.0), np.where(dead, eq, eq * (1 + r[:, t])))
        dead |= new
        peak = np.maximum(peak, eq)
    mult = eq / start
    return {"L": L, "years": years, "median_x": float(np.median(mult)), "mean_x": float(mult.mean()),
            "p_wipeout": float(dead.mean()), "p_lose_half": float((mult < 0.5).mean()),
            "p_loss": float((mult < 1).mean()), "p_2x": float((mult >= 2).mean()), "p_10x": float((mult >= 10).mean()),
            "p_100x": float((mult >= 100).mean()), "p90_x": float(np.percentile(mult, 90)),
            "p99_x": float(np.percentile(mult, 99)), "median_max_dd": float(np.median(mdd))}


def prop(net, mae, firm, Le, Lf, n=20_000, seed=0, max_days=500, funded_days=250):
    r = FIRMS[firm]
    rng = np.random.default_rng(seed)
    nph = len(r["targets"])
    phase = np.zeros(n, int)
    eq = np.zeros(n)
    peak = np.zeros(n)
    days = np.zeros(n, int)
    pdays = np.zeros(n, int)
    best = np.zeros(n)
    possum = np.zeros(n)
    fday = np.zeros(n, int)
    paid = np.zeros(n)
    npay = np.zeros(n, int)
    passed = np.zeros(n, bool)
    t_pass = np.full(n, -1)
    for step in range(max_days + funded_days):
        alive = phase >= 0
        if not alive.any():
            break
        trade = rng.random(n) < P_DAY
        ix = rng.integers(0, len(net), size=n)
        L = np.where(phase >= nph, Lf, Le)
        dr = np.where(trade, net[ix] * L, 0.0)
        dm = np.where(trade, mae[ix] * L, 0.0)
        floor = np.where(r["trailing"], peak - r["maxloss"], -r["maxloss"])
        br = alive & ((-dm >= r["daily"]) | (eq + dm <= floor) | (-dr >= r["daily"]) | (eq + dr <= floor))
        phase = np.where(br, -1, phase)
        alive &= ~br
        eq = np.where(alive, eq + dr, eq)
        ev = alive & (phase < nph)
        days = np.where(ev & trade, days + 1, days)
        if r["profit_day"] > 0:
            pdays = np.where(ev & (dr >= r["profit_day"]), pdays + 1, pdays)
        best = np.where(ev, np.maximum(best, dr), best)
        possum = np.where(ev & (dr > 0), possum + dr, possum)
        peak = np.where(alive, np.maximum(peak, eq), peak)
        tgt = np.array(list(r["targets"]) + [np.inf])[np.clip(phase, 0, nph)]
        ok = ev & (eq >= tgt) & (days >= r["min_days"]) & (pdays >= r["min_profit_days"])
        if r["best_day"] is not None:
            ok &= best <= r["best_day"] * np.maximum(possum, 1e-12)
        phase = np.where(ok, phase + 1, phase)
        for a in (eq, peak, best, possum):
            a[ok] = 0.0
        days[ok] = 0
        pdays[ok] = 0
        newly = ok & (phase == nph)
        passed |= newly
        t_pass = np.where(newly, step, t_pass)
        fday = np.where(newly, 0, fday)
        f = alive & (phase == nph) & ~newly
        fday = np.where(f, fday + 1, fday)
        due = f & (fday >= r["first_pay"]) & ((fday - r["first_pay"]) % r["pay_every"] == 0) & (eq > 0)
        paid = np.where(due, paid + r["split"] * eq * INIT, paid)
        npay = np.where(due, npay + 1, npay)
        eq = np.where(due, 0.0, eq)
        peak = np.where(due, 0.0, peak)
        phase = np.where(f & (fday >= funded_days), -2, phase)
        phase = np.where((phase >= 0) & (phase < nph) & (step + 1 >= max_days), -2, phase)
    refund = np.where((npay > 0) & r["refund"], r["fee"], 0.0)
    net_usd = paid + refund - r["fee"]
    tp = t_pass[passed]
    return {"firm": firm, "Le": Le, "Lf": Lf, "pass": float(passed.mean()),
            "median_days_to_pass": float(np.median(tp)) if len(tp) else np.nan,
            "paid": float((npay > 0).mean()), "mean_net": float(net_usd.mean()), "median_net": float(np.median(net_usd)),
            "p_net_10k": float((net_usd >= 10_000).mean()), "p_net_50k": float((net_usd >= 50_000).mean()),
            "p99_net": float(np.percentile(net_usd, 99)), "max_net": float(net_usd.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M = pd.read_csv(a.mae)
    base = (M.net.to_numpy(), M.mae.to_numpy())
    cut = (base[0] - 2e-4, np.minimum(base[1], base[0] - 2e-4))
    A = []
    for scen, (net, mae) in (("as traded", base), ("haircut -2bp", cut)):
        for years in (1, 3, 7):
            for L in (5, 10, 15, 20, 30, 45, 60, 100):
                A.append({"scenario": scen, **account(net, mae, L, years)})
    A = pd.DataFrame(A)
    A.to_csv(os.path.join(a.out, "account.csv"), index=False)
    B = []
    for scen, (net, mae) in (("as traded", base), ("haircut -2bp", cut)):
        for firm in FIRMS:
            for Le in (5, 10, 15, 20, 30):
                for Lf in (3, 5, 8, 12):
                    B.append({"scenario": scen, **prop(net, mae, firm, Le, Lf)})
    B = pd.DataFrame(B)
    B.to_csv(os.path.join(a.out, "prop.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(A.round(3).to_string())
    print(B.sort_values("mean_net", ascending=False).groupby(["scenario", "firm"]).head(3).round(3).to_string())
    json.dump({"trades": int(len(M)), "mean_bp": float(M.net.mean() * 1e4), "mae_p1_bp": float(M.mae.quantile(0.01) * 1e4)},
              open(os.path.join(a.out, "inputs.json"), "w"))


if __name__ == "__main__":
    main()
