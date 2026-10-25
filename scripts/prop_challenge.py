"""How often does a trader WITHOUT an edge pass a prop-firm challenge and get paid, compared with the real edges found
here? (Question raised: "people make money on prop firms".)

Rules modelled (FTMO 100k 2-step as published, verified by luxalgo.com on 2026-01-15): fee EUR 540 (taken as
USD 585), refunded with the first payout; phase 1 target +10%, phase 2 target +5%; maximum daily loss 5% of the
initial balance; maximum loss 10% of the initial balance (static floor at 90,000); at least 4 trading days per
phase; no time limit (a trader who has neither passed nor failed after MAX_DAYS trading days is counted as not
passed). Funded account: every 10 trading days, 80% of any profit is paid out and the account goes back to the
initial balance; the account is lost when either loss limit is breached; followed for FUNDED_DAYS trading days.
Losses are checked after every trade (the open-trade path within a trade is not modelled, which slightly favours
the trader).

Traders:
  * RANDOM (no edge): entries at random minutes 07:00-19:00 UTC on random business days of EURUSD M1 bid/ask data
    (2010-2018), coin-flip direction, stop = target = S pips, closed at 20:55 UTC if neither is hit. Real costs:
    buy at ask / sell at bid, slippage 0.25 x spread + 0.1 pip on market and stop orders, the target is a limit on
    its own side of the quote, commission 0.35 bp per side. Outcome in R (multiples of the stop). Each simulated
    trader takes consecutive trades from a random start of this chronological pool, N per day, risking a fixed
    fraction of the initial balance per trade.
  * GOTOBI (the surviving spot edge, run 19, net of costs and rate swap): the 700 per-trade returns in date order,
    one per Gotobi day, at leverage L; block start at a random point, wrapping around.
Development data only (EURUSD to 2018 here; no 2026 data).
Usage: python scripts/prop_challenge.py [--gotobi PATH_TO_series.csv] [--out DIR]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FEE = 585.0
INIT = 100_000.0
MAX_DAYS = 500            # about two years to pass the two phases
FUNDED_DAYS = 250         # one year funded
PAYOUT_EVERY = 10
SPLIT = 0.80


BRACKETS = ((10, 1.0), (20, 1.0), (20, 2.0), (20, 0.5))   # (stop in pips, target as a multiple of the stop)


def random_pool(n=40_000, seed=7, start="2010-01-01", end="2018-06-30", brackets=BRACKETS):
    """Chronological pool of random bracket trades on EURUSD M1, outcome in R (multiples of the stop) for each
    bracket."""
    from fxlab.data.store import load_m1
    from fxlab.execution.costs import BASE
    m = load_m1("EURUSD", start, end, columns=["bid_o", "bid_h", "bid_l", "ask_o", "ask_h", "ask_l"])
    idx = m.index
    bo, bh, bl = m.bid_o.to_numpy(), m.bid_h.to_numpy(), m.bid_l.to_numpy()
    ao, ah, al = m.ask_o.to_numpy(), m.ask_h.to_numpy(), m.ask_l.to_numpy()
    pip, sf, sp, comm = 1e-4, BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4
    hour = idx.hour.to_numpy()
    day = idx.floor("D")
    ok = np.flatnonzero((hour >= 7) & (hour < 19) & (idx.weekday < 5))
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(ok, size=n, replace=False))
    side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    out = {b: np.full(n, np.nan) for b in brackets}
    # end-of-day exit bar for each trade: first bar at or after 20:55 UTC of the same day
    ends = idx.searchsorted(day[pick] + pd.Timedelta(hours=20, minutes=55))
    for k, (i, d, j) in enumerate(zip(pick, side, ends)):
        j = min(int(j), len(idx) - 1)
        if j <= i or day[j] != day[i]:
            continue
        si = sf * (ao[i] - bo[i]) + sp * pip
        E = ao[i] + si if d > 0 else bo[i] - si
        for (s, tm) in brackets:
            dist = s * pip
            if d > 0:
                SL, TP = E - dist, E + tm * dist
                hs, ht = bl[i:j] <= SL, bh[i:j] >= TP
            else:
                SL, TP = E + dist, E - tm * dist
                hs, ht = ah[i:j] >= SL, al[i:j] <= TP
            hit = np.flatnonzero(hs | ht)
            if len(hit):
                kx = i + int(hit[0])
                sk = sf * (ao[kx] - bo[kx]) + sp * pip
                if hs[hit[0]]:
                    px = (min(SL, bo[kx]) - sk) if d > 0 else (max(SL, ao[kx]) + sk)
                else:
                    px = TP
            else:
                sj = sf * (ao[j] - bo[j]) + sp * pip
                px = bo[j] - sj if d > 0 else ao[j] + sj
            pnl = (px - E) * d - comm * 2 * E                    # price units, commission on both sides
            out[(s, tm)][k] = pnl / dist
    df = pd.DataFrame({"time": idx[pick], "side": side, **{f"R_{s}_{tm:g}": out[b] for b in brackets
                                                          for (s, tm) in [b]}})
    return df.dropna().reset_index(drop=True)


def simulate(day_returns_fn, n_traders, seed):
    """day_returns_fn(rng, n_traders) -> list over days of arrays (n_traders x trades_that_day) of account returns
    (fractions of the initial balance), produced lazily per day. Returns per-trader outcomes."""
    rng = np.random.default_rng(seed)
    gen = day_returns_fn(rng, n_traders)
    phase = np.zeros(n_traders, int)                # 0 = phase 1, 1 = phase 2, 2 = funded, 3 = failed, 4 = done
    eq = np.zeros(n_traders)                        # profit since the phase / payout reset, fraction of INIT
    days_in_phase = np.zeros(n_traders, int)
    pass1 = np.zeros(n_traders, bool)
    pass2 = np.zeros(n_traders, bool)
    funded_day = np.zeros(n_traders, int)
    paid = np.zeros(n_traders)
    n_pay = np.zeros(n_traders, int)
    fail_reason = np.zeros(n_traders, int)          # 1 daily, 2 max loss
    total_days = 0
    while total_days < MAX_DAYS + FUNDED_DAYS and (phase < 3).any():
        total_days += 1
        trades = next(gen)                          # n_traders x k
        alive = phase < 3
        start_eq = eq.copy()
        for c in range(trades.shape[1]):
            r = trades[:, c]
            eq = np.where(alive, eq + r, eq)
            breach_d = alive & (start_eq - eq >= 0.05)
            breach_m = alive & (eq <= -0.10)
            br = breach_d | breach_m
            fail_reason = np.where(br & (fail_reason == 0), np.where(breach_d, 1, 2), fail_reason)
            phase = np.where(br, 3, phase)
            alive = alive & ~br
        days_in_phase = np.where(alive, days_in_phase + 1, days_in_phase)
        # phase targets (checked at the end of the day, with the minimum of 4 trading days)
        p1 = alive & (phase == 0) & (eq >= 0.10) & (days_in_phase >= 4)
        p2 = alive & (phase == 1) & (eq >= 0.05) & (days_in_phase >= 4)
        pass1 |= p1
        pass2 |= p2
        phase = np.where(p1, 1, np.where(p2, 2, phase))
        eq = np.where(p1 | p2, 0.0, eq)
        days_in_phase = np.where(p1 | p2, 0, days_in_phase)
        funded_day = np.where(p2, 0, funded_day)
        # funded: payouts every PAYOUT_EVERY trading days
        f = (phase == 2) & ~p2
        funded_day = np.where(f, funded_day + 1, funded_day)
        pay = f & (funded_day % PAYOUT_EVERY == 0) & (eq > 0)
        paid = np.where(pay, paid + SPLIT * eq * INIT, paid)
        n_pay = np.where(pay, n_pay + 1, n_pay)
        eq = np.where(pay, 0.0, eq)
        # stop conditions: not passed in time, or funded period over
        timeout = (phase < 2) & (total_days >= MAX_DAYS)
        phase = np.where(timeout, 4, phase)
        over = f & (funded_day >= FUNDED_DAYS)
        phase = np.where(over, 4, phase)
    refund = np.where(n_pay > 0, FEE, 0.0)
    return {"pass_phase1": float(pass1.mean()), "pass_both": float(pass2.mean()),
            "any_payout": float((n_pay > 0).mean()), "mean_payouts_usd": float(paid.mean()),
            "mean_net_usd": float((paid + refund - FEE).mean()),
            "median_net_usd": float(np.median(paid + refund - FEE)),
            "share_net_positive": float(((paid + refund - FEE) > 0).mean()),
            "p90_net_usd": float(np.percentile(paid + refund - FEE, 90)),
            "p99_net_usd": float(np.percentile(paid + refund - FEE, 99)),
            "fail_daily_loss_share": float((fail_reason == 1).mean()),
            "fail_max_loss_share": float((fail_reason == 2).mean())}


def random_trader_days(R, risk, per_day):
    R = np.asarray(R)
    n = len(R)

    def gen(rng, k):
        pos = rng.integers(0, n, size=k)
        while True:
            idx = (pos[:, None] + np.arange(per_day)[None, :]) % n
            pos = pos + per_day
            yield risk * R[idx]
    return gen


def stream_trader_days(ret, lev, events_per_day):
    """A stream of per-event returns (one event roughly every 1 / events_per_day trading days)."""
    ret = np.asarray(ret)
    n = len(ret)

    def gen(rng, k):
        pos = rng.integers(0, n, size=k)
        while True:
            hit = rng.random(k) < events_per_day
            r = np.where(hit, lev * ret[pos % n], 0.0)
            pos = pos + hit.astype(int)
            yield r[:, None]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gotobi", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "reports", "prop_challenge"))
    ap.add_argument("--traders", type=int, default=20_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    pool = random_pool()
    pool.to_parquet(os.path.join(a.out, "random_pool.parquet"))
    rows = []
    for (s, tm) in BRACKETS:
        R = pool[f"R_{s}_{tm:g}"].to_numpy()
        rows.append({"trader": "pool", "stop_pips": s, "target_x_stop": tm, "trades": int(len(R)),
                     "mean_R": float(R.mean()), "win_rate": float((R > 0).mean()),
                     "t_mean": float(R.mean() / R.std() * np.sqrt(len(R)))})
    for (s, tm) in BRACKETS:
        R = pool[f"R_{s}_{tm:g}"].to_numpy()
        for risk in (0.0025, 0.005, 0.01, 0.02, 0.045):
            for per_day in (1, 3):
                if risk == 0.045 and per_day == 3:
                    continue                             # would breach the 5% daily limit after two losses
                res = simulate(random_trader_days(R, risk, per_day), a.traders, seed=11)
                rows.append({"trader": "random", "stop_pips": s, "target_x_stop": tm, "risk_per_trade": risk,
                             "trades_per_day": per_day, **res})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    # zero-cost coin flip with the same rules (R = +-1), for reference
    R0 = np.where(np.random.default_rng(5).random(200_000) < 0.5, 1.0, -1.0)
    for risk in (0.005, 0.01, 0.02):
        res = simulate(random_trader_days(R0, risk, 1), a.traders, seed=12)
        rows.append({"trader": "coin_flip_no_costs", "risk_per_trade": risk, "trades_per_day": 1, **res})
    if a.gotobi and os.path.exists(a.gotobi):
        S = pd.read_csv(a.gotobi, parse_dates=["day"])
        g = S[S.test == "F1"].sort_values("day")["net_sw"].to_numpy()
        per_day = len(g) / (len(pd.bdate_range(S[S.test == "F1"].day.min(), S[S.test == "F1"].day.max())))
        for lev in (3, 6, 10, 20):
            res = simulate(stream_trader_days(g, lev, per_day), a.traders, seed=13)
            rows.append({"trader": "gotobi", "leverage": lev, "events_per_day": per_day,
                         "mean_trade_account_pct": float(g.mean() * lev * 100), **res})
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "prop_challenge.csv"), index=False)
    json.dump({"seconds": round(time.time() - t0, 1), "rules": {"fee_usd": FEE, "phase1": 0.10, "phase2": 0.05,
               "daily_loss": 0.05, "max_loss": 0.10, "min_days": 4, "max_days_to_pass": MAX_DAYS,
               "funded_days": FUNDED_DAYS, "payout_every_days": PAYOUT_EVERY, "split": SPLIT}},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
