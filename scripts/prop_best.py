"""Which prop-firm programme, account size and leverage to use for the frozen Gotobi trade (long USD/JPY + EUR/JPY,
23:00 -> 09:53 Tokyo on Gotobi days, 80 bp disaster stop), judged on the expected net per attempt, the chance of
being paid, the time to the first payout and the fee. Every attempt is a simulated trader starting on a random night
and walking the real calendar of nights (net returns and the worst open loss inside each trade).

PROGRAMMES (terms as published, checked 2026-11-02; sources in docs/prop_best.md; fees in USD, FTMO's in EUR x 1.15):
  FTMO 2-Step       10% / 5%, daily 5% (midnight balance, equity monitored), max 10% static, 4 days a phase, 80% split,
                    fee refunded with the first reward, reward after 14 days then every 14.
  FTMO 1-Step       10%, daily 3%, max 10% trailing (end-of-day high), best day <= 50% of the total profit, 90% split,
                    no refund.
  FundedNext Stellar 2-Step  8% / 5%, daily 5%, max 10% static, 5 days, 80%, refund with the 1st reward, first reward
                    after 21 days then every 14; at most 3% risk on a trade (caps leverage at 3.5x with the 80 bp stop).
  FundedNext Stellar 1-Step  10%, daily 3%, max 6% static, 2 days, 80%, refund with the 3rd reward, rewards every 5
                    business days; 3% risk cap.
  FundedNext Stellar Lite    8% / 4%, daily 4%, max 8% static, 2 days, 80%, refund with the 3rd reward, rewards every
                    10 days; 3% risk cap.
  The5ers High Stakes        10% / 5%, daily 5%, max 10% static, 3 profitable days (>= 0.5%) a step, 80%, fee refunded
                    (modelled with the first payout), payouts after 14 days then every 14.
  The5ers High Stakes Classic 8% / 5%, otherwise as High Stakes (only the 2.5K and 100K prices are published).
LEVERAGE CAPS (x the account, both pairs together; with the 80 bp stop a stop-out costs about lev x 0.8%, a gap
  through it about lev x 1%): FTMO 2-Step 5x (5% daily); the 3%-daily programmes 3x; FundedNext 3.5x (3% risk a
  trade); The5ers 4x, because The5ers measures the daily loss from the higher of balance and equity at MT5 server
  midnight (17:00 New York), which falls inside the trade: on 2007-2018 nights that turns 0 breaches at 5x into 2
  (scripts/prop_checks.py), while FTMO and FundedNext measure from the balance.
SCENARIOS for the size of the edge: "2019-2025" (the basket's own nights, +4.8 bp a night with the stop), "2026"
  (every night shifted down by the same amount so the mean is the 2026 no-stop mean, +2.04 bp), "zero" (mean 0).
Usage: python scripts/prop_best.py --trades docs/gotobi_losses/trades_basket_stop_80_bp.csv --out DIR [--n 4000]"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

EUR = 1.15
MAX_DAYS, FUNDED_DAYS = 500, 250
P = {
    "FTMO 2-Step": dict(sizes={10: 89 * EUR, 25: 250 * EUR, 50: 345 * EUR, 100: 540 * EUR, 200: 1080 * EUR},
                        targets=(0.10, 0.05), daily=0.05, maxloss=0.10, trailing=False, min_days=4, profit_day=0.0,
                        min_profit_days=0, best_day=None, split=0.80, refund_after=1, first_pay=10, pay_every=10,
                        max_lev=5),
    "FTMO 1-Step": dict(sizes={10: 79 * EUR, 25: 199 * EUR, 50: 319 * EUR, 100: 499 * EUR, 200: 999 * EUR},
                        targets=(0.10,), daily=0.03, maxloss=0.10, trailing=True, min_days=0, profit_day=0.0,
                        min_profit_days=0, best_day=0.5, split=0.90, refund_after=None, first_pay=10, pay_every=10,
                        max_lev=3),
    "FundedNext Stellar 2-Step": dict(sizes={6: 59.99, 15: 119.99, 25: 199.99, 50: 299.99, 100: 549.99, 200: 1099.99},
                                      targets=(0.08, 0.05), daily=0.05, maxloss=0.10, trailing=False, min_days=5,
                                      profit_day=0.0, min_profit_days=0, best_day=None, split=0.80, refund_after=1,
                                      first_pay=15, pay_every=10, max_lev=3.5),
    "FundedNext Stellar 1-Step": dict(sizes={6: 65.99, 15: 129.99, 25: 219.99, 50: 329.99, 100: 569.99, 200: 1099.99},
                                      targets=(0.10,), daily=0.03, maxloss=0.06, trailing=False, min_days=2,
                                      profit_day=0.0, min_profit_days=0, best_day=None, split=0.80, refund_after=3,
                                      first_pay=5, pay_every=5, max_lev=3),
    "FundedNext Stellar Lite": dict(sizes={5: 32.99, 10: 59.99, 25: 139.99, 50: 229.99, 100: 399.99, 200: 798.99},
                                    targets=(0.08, 0.04), daily=0.04, maxloss=0.08, trailing=False, min_days=2,
                                    profit_day=0.0, min_profit_days=0, best_day=None, split=0.80, refund_after=3,
                                    first_pay=7, pay_every=7, max_lev=3.5),
    "The5ers High Stakes": dict(sizes={2.5: 19, 5: 35, 10: 69, 25: 176, 50: 249, 100: 405}, targets=(0.10, 0.05),
                                daily=0.05, maxloss=0.10, trailing=False, min_days=0, profit_day=0.005,
                                min_profit_days=3, best_day=None, split=0.80, refund_after=1, first_pay=10,
                                pay_every=10, max_lev=4),
    "The5ers High Stakes Classic": dict(sizes={2.5: 22, 100: 455}, targets=(0.08, 0.05), daily=0.05, maxloss=0.10,
                                        trailing=False, min_days=0, profit_day=0.005, min_profit_days=3, best_day=None,
                                        split=0.80, refund_after=1, first_pay=10, pay_every=10, max_lev=4),
}
LEV = (2, 3, 3.5, 4, 5, 6)


def simulate(A, M, r, n, seed, le, lf):
    """Per-attempt outcomes: payouts as a fraction of the account, number of payouts, first payout step, fail type.
    A, M: daily net return and worst open loss (<= 0) of one unit; le / lf leverage in evaluation / funded."""
    rng = np.random.default_rng(seed)
    T = len(A)
    pos = rng.integers(0, T, size=n)
    nph = len(r["targets"])
    phase = np.zeros(n, int)
    eq, peak, best, possum = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    days, pdays, fday, npay, fail = (np.zeros(n, int) for _ in range(5))
    paid = np.zeros(n)
    passed = np.zeros(n, bool)
    first = np.full(n, -1)
    for step in range(MAX_DAYS + FUNDED_DAYS):
        alive = phase >= 0
        if not alive.any():
            break
        sc = np.where(phase >= nph, lf, le)
        day = A[(pos + step) % T] * sc
        dip = M[(pos + step) % T] * sc
        start = eq.copy()
        floor = np.where(r["trailing"], peak - r["maxloss"], -r["maxloss"])
        bo = alive & ((-dip >= r["daily"]) | (eq + dip <= floor))
        fail = np.where(bo & (fail == 0), np.where(-dip >= r["daily"], 1, 2), fail)
        phase = np.where(bo, -1, phase)
        alive &= ~bo
        eq = np.where(alive, eq + day, eq)
        bd, bm = alive & (start - eq >= r["daily"]), alive & (eq <= floor)
        br = bd | bm
        fail = np.where(br & (fail == 0), np.where(bd, 1, 2), fail)
        phase = np.where(br, -1, phase)
        alive &= ~br
        dp = eq - start
        traded = day != 0
        ev = alive & (phase < nph)
        days = np.where(ev & traded, days + 1, days)
        if r["profit_day"] > 0:
            pdays = np.where(ev & (dp >= r["profit_day"]), pdays + 1, pdays)
        best = np.where(ev, np.maximum(best, dp), best)
        possum = np.where(ev & (dp > 0), possum + dp, possum)
        peak = np.where(alive, np.maximum(peak, eq), peak)
        tgt = np.array(list(r["targets"]) + [np.inf])[np.clip(phase, 0, nph)]
        ok = ev & (eq >= tgt) & (days >= r["min_days"]) & (pdays >= r["min_profit_days"])
        if r["best_day"] is not None:
            ok &= best <= r["best_day"] * np.maximum(eq, 1e-12)            # best day <= 50% of the total profit
        phase = np.where(ok, phase + 1, phase)
        for v in (eq, peak, best, possum):
            v[ok] = 0.0
        days[ok] = 0
        pdays[ok] = 0
        newly = ok & (phase == nph)
        passed |= newly
        fday = np.where(newly, 0, fday)
        f = alive & (phase == nph) & ~newly
        fday = np.where(f, fday + 1, fday)
        due = f & (fday >= r["first_pay"]) & ((fday - r["first_pay"]) % r["pay_every"] == 0) & (eq > 0)
        first = np.where(due & (npay == 0), step, first)
        paid = np.where(due, paid + r["split"] * eq, paid)
        npay = np.where(due, npay + 1, npay)
        eq = np.where(due, 0.0, eq)
        peak = np.where(due, 0.0, peak)
        phase = np.where(f & (fday >= FUNDED_DAYS), -2, phase)
        phase = np.where((phase >= 0) & (phase < nph) & (step + 1 >= MAX_DAYS), -2, phase)
    return paid, npay, passed, first, fail


def daily(B):
    days = pd.bdate_range(B.index.min(), B.index.max())
    X = B.net.groupby(B.index).sum().reindex(days).fillna(0.0).to_numpy()
    Mo = B.mae.groupby(B.index).sum().reindex(days).fillna(0.0).to_numpy()
    return X, np.minimum(Mo, np.minimum(X, 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = pd.read_csv(a.trades, index_col=0, parse_dates=True)
    B.index = pd.DatetimeIndex(B.index).normalize()
    B = B[(B.index >= "2019-01-01") & (B.index <= "2025-12-31")][["net", "mae"]].dropna()
    scen = {}
    for name, target in (("2019-2025", None), ("2026", 2.04e-4), ("zero", 0.0)):
        g = B.copy()
        if target is not None:
            d = g.net.mean() - target
            g["net"] = g.net - d
            g["mae"] = np.minimum(g.mae - d, np.minimum(g.net, 0.0))
        scen[name] = daily(g)
    rows = []
    for (pname, r), (sname, (X, M)) in itertools.product(P.items(), scen.items()):
        for le, lf in itertools.product(LEV, LEV):
            if le > r["max_lev"] or lf > r["max_lev"] or lf > le + 1:
                continue
            paid, npay, passed, first, fail = simulate(X, M, r, a.n, 1, le, lf)
            refunded = (npay >= r["refund_after"]) if r["refund_after"] else np.zeros(len(npay), bool)
            fp = first[first >= 0]
            for size, fee in r["sizes"].items():
                net = paid * size * 1000 + fee * refunded - fee
                rows.append({"programme": pname, "size_k": size, "fee": round(fee, 2), "scenario": sname,
                             "lev_eval": le, "lev_funded": lf, "pass": passed.mean(), "paid": (npay > 0).mean(),
                             "mean_net": net.mean(), "median_net": np.median(net), "p90_net": np.percentile(net, 90),
                             "share_net_positive": (net > 0).mean(),
                             "median_days_to_first_pay": float(np.median(fp)) if len(fp) else np.nan,
                             "paid_within_3m": ((first >= 0) & (first <= 63)).mean(),
                             "paid_within_6m": ((first >= 0) & (first <= 126)).mean(),
                             "fail_daily": (fail == 1).mean(), "fail_max": (fail == 2).mean(),
                             "mean_net_per_fee": net.mean() / fee,
                             "fees_per_payout": fee / max((npay > 0).mean(), 1e-9)})
        print(pname, sname, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, "prop_best.csv"), index=False)
    out = {}
    for s in scen:
        q = R[R.scenario == s]
        out[s] = {
            "highest_mean_net": q.sort_values("mean_net", ascending=False).head(5).to_dict("records"),
            "highest_chance_paid": q.sort_values(["paid", "mean_net"], ascending=False).head(5).to_dict("records"),
            "fastest_paid_3m": q.sort_values(["paid_within_3m", "mean_net"], ascending=False).head(5).to_dict("records"),
            "best_per_fee": q[q.fee <= 70].sort_values("mean_net_per_fee", ascending=False).head(5).to_dict("records")}
    json.dump(out, open(os.path.join(a.out, "picks.json"), "w"), indent=1, default=float)
    # per programme: the leverage with the highest mean net per fee at the 2026 edge, every size and scenario
    keep = []
    for pname in P:
        q = R[(R.programme == pname) & (R.scenario == "2026")].sort_values("mean_net_per_fee", ascending=False)
        le, lf = q.iloc[0].lev_eval, q.iloc[0].lev_funded
        keep.append(R[(R.programme == pname) & (R.lev_eval == le) & (R.lev_funded == lf)])
    pd.concat(keep).to_csv(os.path.join(a.out, "by_programme.csv"), index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    for s in scen:
        q = R[R.scenario == s]
        print(f"== {s}: highest mean net")
        print(q.sort_values("mean_net", ascending=False).head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
