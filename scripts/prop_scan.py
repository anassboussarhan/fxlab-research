"""Every CFD prop programme in LuxAlgo's public rules directory (27 firms, 520 programmes; fetched by
ci/misc_propfirms.sh) run with the frozen Gotobi basket's real 2019-2025 nights (net of costs and swap, with each
night's worst open loss), at the development edge and with every night shifted to the 2026 mean (+2.04 bp).

Mapping of the directory fields (the directory's own adapter is the model: github.com/LuxAlgo/prop-firm-sim):
- targets: profitTarget (percent, or amounts / account size); steps 0 = instant funding (no target).
- daily loss: dailyLoss (percent), none if empty. "Balance/Equity - highest at EOD": the reference is set at the
  server's end of day, which falls inside the Gotobi trade, so leverage is capped at daily / 1.2% (balance-based:
  daily / 1.0%), from the worst nights (-0.97% gap at 1x; The5ers check in docs/prop_best.md).
- max loss: maxLoss percent; "Trailing" -> end-of-day trailing (an intraday trailing floor is stricter: flagged).
- payouts: payoutIntervalDays / firstPayoutMinDays, else parsed from payoutFrequency text (on demand -> any day;
  weekly 5, bi-weekly / 14 days 10, 21 days 15, monthly / 30 days 21 business days; "first N days" for the first);
  a minimum profit for a payout ("at least 2% ...") is applied; funded consistency rules are not simulated (flagged).
- fee: price + activation fee; refund when isFeeRefundable.
Only programmes that allow overnight holding are kept. Leverage 2, 3, 4, 5, 6, 8x up to the cap; the best leverage
per programme is the one with the highest mean net at the 2026 edge.
Usage: python scripts/prop_scan.py --rules challenges.csv --nights vs1_nights.csv --out DIR [--n 2000]"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_sims as FS  # noqa: E402

INIT, MAX_DAYS, FUNDED_DAYS = 100_000.0, 500, 250
LEVS = (2, 3, 4, 5, 6, 8)


def simulate(A, M, r, n=2000, seed=1):
    """prop_firms_ev.simulate with a minimum profit per payout (pay_min) added."""
    rng = np.random.default_rng(seed)
    T = len(A)
    pos = rng.integers(0, T, size=n)
    nph = len(r["targets"])
    phase = np.zeros(n, int)
    eq = np.zeros(n)
    peak = np.zeros(n)
    days = np.zeros(n, int)
    best = np.zeros(n)
    possum = np.zeros(n)
    fday = np.zeros(n, int)
    paid = np.zeros(n)
    npay = np.zeros(n, int)
    fail = np.zeros(n, int)
    passed = np.zeros(n, bool) if nph else np.ones(n, bool)
    first = np.full(n, -1)
    for step in range(MAX_DAYS + FUNDED_DAYS):
        alive = phase >= 0
        if not alive.any():
            break
        day = A[(pos + step) % T]
        dip = M[(pos + step) % T]
        start = eq.copy()
        floor = np.where(r["trailing"], peak - r["maxloss"], -r["maxloss"])
        bo = alive & ((-dip >= r["daily"]) | (eq + dip <= floor))
        fail = np.where(bo & (fail == 0), np.where(-dip >= r["daily"], 1, 2), fail)
        phase = np.where(bo, -1, phase)
        alive &= ~bo
        eq = np.where(alive, eq + day, eq)
        bd = alive & (start - eq >= r["daily"])
        bm = alive & (eq <= floor)
        br = bd | bm
        fail = np.where(br & (fail == 0), np.where(bd, 1, 2), fail)
        phase = np.where(br, -1, phase)
        alive &= ~br
        dp = eq - start
        ev = alive & (phase < nph)
        days = np.where(ev & (day != 0), days + 1, days)
        best = np.where(ev, np.maximum(best, dp), best)
        possum = np.where(ev & (dp > 0), possum + dp, possum)
        peak = np.where(alive, np.maximum(peak, eq), peak)
        tgt = np.array(list(r["targets"]) + [np.inf])[np.clip(phase, 0, nph)]
        ok = ev & (eq >= tgt) & (days >= r["min_days"])
        if r["best_day"] is not None:
            ok &= best <= r["best_day"] * np.maximum(possum, 1e-12)
        phase = np.where(ok, phase + 1, phase)
        for arr in (eq, peak, best, possum):
            arr[ok] = 0.0
        days = np.where(ok, 0, days)
        newly = ok & (phase == nph)
        passed |= newly
        fday = np.where(newly, 0, fday)
        f = alive & (phase == nph) & ~newly
        fday = np.where(f, fday + 1, fday)
        due = f & (fday >= r["first_pay"]) & ((fday - r["first_pay"]) % r["pay_every"] == 0) & (eq > max(r["pay_min"], 0.0))
        first = np.where(due & (npay == 0), step, first)
        paid = np.where(due, paid + r["split"] * eq * INIT, paid)
        npay = np.where(due, npay + 1, npay)
        eq = np.where(due, 0.0, eq)
        peak = np.where(due, 0.0, peak)
        phase = np.where(f & (fday >= FUNDED_DAYS), -2, phase)
        phase = np.where((phase >= 0) & (phase < nph) & (step + 1 >= MAX_DAYS), -2, phase)
    fee = r["fee_frac"] * INIT
    net = paid + np.where((npay >= 1) & r["refund"], fee, 0.0) - fee
    fp = first[first >= 0]
    return {"paid": float((npay > 0).mean()), "pass": float(passed.mean()), "mean_net": float(net.mean()),
            "median_days_first_pay": float(np.median(fp)) if len(fp) else np.nan,
            "paid_3m": float(((first >= 0) & (first <= 63)).mean()), "paid_6m": float(((first >= 0) & (first <= 126)).mean()),
            "fail_daily": float((fail == 1).mean()), "fail_max": float((fail == 2).mean()),
            "share_positive": float((net > 0).mean())}


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def payout_terms(row):
    txt = str(row.get("payoutFrequency") or "").lower()
    every = num(row.get("payoutIntervalDays"))
    first = num(row.get("firstPayoutMinDays"))
    flags = []

    def cal_to_bd(d):
        return max(1, int(round(d * 5 / 7)))
    if every is None:
        if re.search(r"demand|any ?time|daily|instant", txt):
            every = 1
        elif re.search(r"bi.?week|14 ?day|every 2 week|fortnight", txt):
            every = 10
        elif re.search(r"21 ?day", txt):
            every = 15
        elif re.search(r"week|7 ?day", txt):
            every = 5
        elif re.search(r"month|30 ?day", txt):
            every = 21
        elif re.search(r"(\d+) ?business day", txt):
            every = int(re.search(r"(\d+) ?business day", txt).group(1))
        else:
            every, _ = 10, flags.append("payout frequency unknown: 14 days assumed")
    else:
        every = cal_to_bd(every)
    if first is None:
        m = re.search(r"first[^,;.]*?(\d+) ?(?:calendar )?days", txt)
        first = cal_to_bd(int(m.group(1))) if m else every
    else:
        first = cal_to_bd(first)
    m = re.search(r"(\d+(?:\.\d+)?) ?% (?:of )?(?:gross |net )?profit", txt)
    pay_min = float(m.group(1)) / 100 if m else 0.0
    if re.search(r"consisten|best day", txt) or num(row.get("fundedConsistencyPct")):
        flags.append("funded consistency rule not simulated")
    return int(first), int(every), pay_min, flags


def rules_of(row):
    acc = num(row["accountSize"]) or 100000
    tgt = json.loads(row["profitTarget"]) if isinstance(row["profitTarget"], str) and row["profitTarget"].startswith("[") else []
    if str(row.get("profitTargetIsPercent")).lower() == "false":
        tgt = [t / acc * 100 for t in tgt]
    if int(num(row["steps"]) or 0) == 0:
        tgt = []
    pct = str(row.get("lossIsPercent")).lower() != "false"
    dl, ml = num(row.get("dailyLoss")), num(row.get("maxLoss"))
    if ml is None:
        return None, ["no max loss"]
    if not pct:
        dl = dl / acc * 100 if dl is not None else None
        ml = ml / acc * 100
    mtype = str(row.get("maxLossType") or "").lower()
    dtype = str(row.get("dailyLossType") or "").lower()
    flags = []
    if "intraday" in mtype or "unrealized" in mtype or "real" in mtype:
        flags.append("intraday trailing max loss approximated end of day (optimistic)")
    first, every, pay_min, pf = payout_terms(row)
    flags += pf
    fee = (num(row.get("price")) or 0) + (num(row.get("activationFee")) or 0)
    bd = num(row.get("consistencyMaxBestDayPct"))
    r = dict(targets=tuple(t / 100 for t in tgt), daily=(dl / 100) if dl else 1.0, maxloss=ml / 100,
             trailing="trail" in mtype, min_days=int(num(row.get("minTradingDays")) or 0),
             best_day=bd / 100 if bd else None, split=(num(row.get("profitSplitPercent")) or 80) / 100,
             refund=str(row.get("isFeeRefundable")).lower() == "true", first_pay=first, pay_every=every,
             pay_min=pay_min, fee_frac=fee / acc)
    if dl:
        per_lev = 1.2 if ("equity" in dtype or "highest" in dtype) else 1.0
        cap = int(np.floor(dl / per_lev + 1e-9))
    else:
        cap = max(LEVS)
    lev_cap = num(row.get("maxLeverage"))
    if lev_cap:
        cap = min(cap, int(lev_cap))
    r["lev_cap"] = cap
    return r, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--nights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = pd.read_csv(a.rules)
    R = R[R.productTypes.astype(str).str.contains("CFD") & (R.overnightHolding.astype(str).str.lower() != "false")]
    N = pd.read_csv(a.nights, index_col=0, parse_dates=True)[["net", "mae"]]
    N = N[(N.index >= "2019-01-01") & (N.index <= "2025-12-31")]
    scen = {"dev": N, "2026": FS.shifted(N, FS.TARGET_2026)}
    D = {k: FS.daily(v) for k, v in scen.items()}
    rows = []
    for _, row in R.iterrows():
        r, flags = rules_of(row)
        if r is None:
            continue
        for L in [x for x in LEVS if x <= r["lev_cap"]]:
            out = {"firm": row["firm"], "program": row["challengeName"], "size": row["accountSize"],
                   "fee": r["fee_frac"] * row["accountSize"], "lev": L, "targets": r["targets"], "daily": r["daily"],
                   "maxloss": r["maxloss"], "trailing": r["trailing"], "first_pay_bd": r["first_pay"],
                   "pay_every_bd": r["pay_every"], "pay_min": r["pay_min"], "split": r["split"], "refund": r["refund"],
                   "news": row.get("newsTrading"), "trustpilot": row.get("trustpilot"), "founded": row.get("yearFounded"),
                   "flags": "; ".join(flags)}
            for s, (X, M) in D.items():
                res = simulate(X[:, 0] * L, M[:, 0] * L, r, n=a.n)
                scale = row["accountSize"] / INIT
                for k, v in res.items():
                    out[f"{s}_{k}"] = v * scale if k == "mean_net" else v
            rows.append(out)
        print(row["challengeName"], flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(a.out, "prop_scan_all.csv"), index=False)
    best = P.sort_values("2026_mean_net", ascending=False).groupby(["firm", "program"]).head(1)
    best["ev_per_fee_2026"] = best["2026_mean_net"] / best.fee
    best.sort_values("ev_per_fee_2026", ascending=False).to_csv(os.path.join(a.out, "prop_scan_best.csv"), index=False)
    print(best.sort_values("ev_per_fee_2026", ascending=False).head(40).to_string())


if __name__ == "__main__":
    main()
