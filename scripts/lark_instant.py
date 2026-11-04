"""Lark Funding Instant account run with the frozen Gotobi basket B0 (real 2019-2025 nights, net of costs, with each
night's worst open loss), at the development edge and with every night shifted to the 2026 mean (+2.04 bp).

Rules as published (checked 2026-11-04 on thetrustedprop.com, thegodfunded.com, propfirmmap.com, luxalgo.com;
Lark's own help centre could not be opened from here):
- no profit target, no minimum days, no consistency rule; first payout on demand, then every 30 days
  (21 business days here); minimum payout $100; split 90% (older pages: 80%); payout fee $40 (Rise, waivable).
- daily loss 5% of the initial balance, reset at 5 pm New York (inside the Gotobi trade): leverage <= 4 keeps the
  worst night below it (prop_scan.py: daily / 1.2% per 1x).
- maximum loss 8% trailing on the highest CLOSED balance (since 2026-03-02); the floor locks at the initial
  balance once the account is 8% up "or after the first payout".
- price: $5k $200, $10k $400, $25k $1,125, $50k $2,250 (one page: $2,750), $100k $4,500.
What the pages leave open is what happens to the floor after the first payout, so two readings are run:
- "strict": after the first payout the floor is the initial balance (the literal reading of "locks at initial
  balance after first payout"): the only cushion left is the profit kept in the account.
- "gentle": after the first payout the floor stops trailing and stays where it was (initial balance - 8% + peak,
  capped at the initial balance).
- "reset" (information only, the assumption of the earlier directory scan in prop_scan.py, not supported by the
  pages): a fresh 8% trailing floor from the balance left after each payout.
Payout policy (chosen on the development scenario only; the 2026-shifted scenario is a stress check): take the
first payout when the closed balance is T above the start, then every 21 business days withdraw everything above
K (kept in the account as cushion). Horizon one year (250 business days) from a random start in 2019-2025.
Usage: python scripts/lark_instant.py --nights vs1_nights.csv --out DIR"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_sims as FS  # noqa: E402

DAYS, PAY_EVERY, DAILY, MAXL = 250, 21, 0.05, 0.08
PRICES = {5_000: [200], 10_000: [400], 25_000: [1_125], 50_000: [2_250, 2_750], 100_000: [4_500]}
LEVS = (1.875, 2.0, 3.0, 4.0)          # 1.875x: the 80 bp stop risks 1.5% of the balance (Lark's guideline)
TS = (0.005, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
KS = (0.0, 0.02, 0.04, 0.06, 0.08)


def run(X, M, L, size, fee, split, reading, T, K, n=4000, seed=1, pay_fee=40.0):
    rng = np.random.default_rng(seed)
    Tn = len(X)
    pos = rng.integers(0, Tn, size=n)
    bal = np.zeros(n)                 # closed balance above the start, fraction of the account
    peak = np.zeros(n)
    floor = np.full(n, -MAXL)
    alive = np.ones(n, bool)
    paid_any = np.zeros(n, bool)
    since = np.zeros(n, int)
    cash = np.zeros(n)
    cash3 = np.zeros(n)
    first = np.full(n, -1)
    fail_daily = np.zeros(n, bool)
    pmin = 100.0 / size
    for d in range(DAYS):
        x = X[(pos + d) % Tn] * L
        m = M[(pos + d) % Tn] * L
        bd = alive & (-m >= DAILY)
        bm = alive & (bal + m <= floor)
        fail_daily |= bd & ~bm
        alive &= ~(bd | bm)
        bal = np.where(alive, bal + x, bal)
        dead = alive & (bal <= floor)
        alive &= ~dead
        pre = alive & (~paid_any | (reading == "reset"))
        peak = np.where(pre, np.maximum(peak, bal), peak)
        floor = np.where(pre, np.minimum(peak - MAXL, 0.0), floor)
        since = np.where(paid_any, since + 1, since)
        want_first = pre & ~paid_any & (bal >= T) & (bal - K >= pmin)
        want_next = alive & paid_any & (since >= PAY_EVERY) & (bal - K >= pmin)
        due = want_first | want_next
        w = np.where(due, bal - K, 0.0)
        got = np.maximum(split * w * size - pay_fee, 0.0) * due
        cash += got
        cash3 += np.where(d < 63, got, 0.0)
        first = np.where(want_first, d, first)
        bal = bal - w
        since = np.where(due, 0, since)
        if reading == "strict":
            floor = np.where(want_first, 0.0, floor)
        if reading == "reset":                       # the earlier directory scan: a fresh 8% trail after each payout
            peak = np.where(due, bal, peak)
            floor = np.where(due, np.minimum(peak - MAXL, 0.0), floor)
        paid_any |= want_first
    net = cash - fee
    fp = first[first >= 0]
    return {"lev": L, "size": size, "fee": fee, "split": split, "reading": reading, "T": T, "K": K,
            "paid": float(paid_any.mean()), "mean_net": float(net.mean()), "median_net": float(np.median(net)),
            "p_net_pos": float((net > 0).mean()), "p_lose_fee": float((cash == 0).mean()),
            "mean_cash_3m": float(cash3.mean()), "median_bd_first_pay": float(np.median(fp)) if len(fp) else np.nan,
            "alive_end": float(alive.mean()), "breach_daily": float(fail_daily.mean()),
            "ev_per_fee": float(net.mean() / fee)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    N = pd.read_csv(a.nights, index_col=0, parse_dates=True)[["net", "mae"]]
    N = N[(N.index >= "2019-01-01") & (N.index <= "2025-12-31")]
    scen = {"dev": N, "2026": FS.shifted(N, FS.TARGET_2026)}
    D = {k: tuple(z[:, 0] for z in FS.daily(v)) for k, v in scen.items()}
    rows = []
    # 1) policy grid on the development scenario, $100k, 90% split
    for reading, L, T, K in itertools.product(("strict", "gentle", "reset"), LEVS, TS, KS):
        if K >= T:
            continue
        r = run(*D["dev"], L, 100_000, 4_500, 0.9, reading, T, K, n=a.n)
        rows.append({"scenario": "dev", **r})
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(a.out, "lark_instant_grid_dev.csv"), index=False)
    best = G.sort_values("mean_net", ascending=False).groupby(["reading", "lev"]).head(1)
    # 2) the chosen policies, both scenarios, every size and price, 90% and 80% split
    out = []
    for _, b in best.iterrows():
        for s, (X, M) in D.items():
            for size, prices in PRICES.items():
                for fee in prices:
                    for split in (0.9, 0.8):
                        out.append({"scenario": s, **run(X, M, b["lev"], size, fee, split, b["reading"], b["T"], b["K"], n=a.n)})
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(a.out, "lark_instant.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["scenario", "reading", "lev", "T", "K", "size", "fee", "split", "paid", "mean_net", "median_net",
            "p_net_pos", "p_lose_fee", "mean_cash_3m", "median_bd_first_pay", "alive_end", "ev_per_fee"]
    print(best[["reading", "lev", "T", "K", "paid", "mean_net", "median_net", "p_net_pos", "alive_end"]].to_string())
    print(R[(R["size"] == 100_000) & (R.split == 0.9)][cols].round(3).to_string())
    print(R[(R.split == 0.9) & (R.lev == 4.0) & (R.reading != "reset")][cols].round(3).to_string())
    print(R[(R["size"] == 100_000) & (R.split == 0.8) & (R.lev == 4.0)][cols].round(3).to_string())


if __name__ == "__main__":
    main()
