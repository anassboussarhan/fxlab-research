"""Break-even costs of the best edges found (runs 15-22), and whether cheaper execution would make them pay.

For every per-event test (flows lab run 19, round-2 lab run 22): gross mean, the modelled cost (gross - net, swap
excluded), swap, and
  BE0 = the round-trip cost at which the net mean (after swap) is zero
  BE2 = the round-trip cost at which the net t-statistic is 2 (gross mean + swap - 2 x sd / sqrt(n))
in basis points of notional and in pips of a typical price of the period.
The modelled cost per round trip (scripts/flows_lab.py Book.trade) is 1.5 x quoted spread + 0.2 pip + 0.7 bp
(spread paid, slippage 0.25 x spread + 0.1 pip per side, commission 0.35 bp per side), so the quoted spread at the
trade times is recovered as (cost - 0.2 pip - 0.7 bp) / 1.5 and four cost levels are compared:
  A  this platform's model: 1.5 x spread + 0.2 pip + 0.7 bp
  B  raw-spread ECN account (IC Markets Raw / Pepperstone Razor: USD 3.50 per side per 100k = 0.35 bp per side),
     no slippage beyond the quote: spread + 0.7 bp
  C  Interactive Brokers (0.20 bp of trade value per order, pass-through quotes): spread + 0.4 bp
  D  institutional / passive execution (fills at the mid on average half the time: about half the spread) + 0.1 bp
The spreads are the 2007-2018 quoted spreads of this data at the trade times. Spreads today are tighter at
liquid hours (EURUSD raw 0.1-0.2 pip), so B-D are upper bounds for today if the effect still exists.
Random-entry and harvest strategies (runs 15/16) report Sharpe at full, half and zero cost; their break-even
fraction of the modelled cost is interpolated linearly in cost.
Usage: python scripts/breakeven_edges.py --flows DIR19 --lit2 DIR22 --pool DIR16 [--out FILE]"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

PIP_BP = {"USDJPY": 0.01 / 100 * 1e4, "EURUSD": 1e-4 / 1.27 * 1e4, "XAUUSD": 0.01 / 1350 * 1e4,
          "XAGUSD": 0.001 / 20 * 1e4, "PORT9": np.mean([1e-4 / 1.27, 1e-4 / 1.5, 0.01 / 100, 1e-4 / 0.85, 1e-4 / 1.15,
                                                      1e-4 / 1.0, 1e-4 / 0.72, 1e-4 / 7.8, 1e-4 / 6.8]) * 1e4,
          "PORT3": np.mean([1e-4 / 1.27, 1e-4 / 1.5, 1e-4 / 1.0]) * 1e4}
FLOWS = {"F1": ("USDJPY", "Gotobi 03:00 -> Tokyo fix"), "F2": ("USDJPY", "Gotobi 08:00 -> fix"),
         "F3": ("USDJPY", "Gotobi reversal after the fix"), "M1": ("PORT9", "month-end fix 08:00 -> 16:00"),
         "M2": ("PORT9", "month-end fix, 24 h"), "L1": ("PORT9", "local hours, 8 currencies"),
         "L2": ("EURUSD", "local hours, EURUSD"), "K1": ("PORT9", "New York hours long foreign"),
         "K2": ("PORT9", "overnight short foreign"), "K3": ("PORT3", "around the clock, EUR/GBP/CHF (2 trades/day)"),
         "P1": ("PORT9", "FOMC day, dollar portfolio"), "P2": ("PORT9", "FOMC day, 3 high-yielders"),
         "W1": ("PORT9", "4 p.m. fix reversal")}
LIT2 = {"S1": ("XAGUSD", "silver fix"), "G1": ("XAUUSD", "gold PM fix"), "G2": ("XAUUSD", "gold AM fix"),
        "G3": ("XAUUSD", "gold PM fix, follow the opening"), "A1": ("XAUUSD", "gold Asian hours"),
        "A2": ("XAUUSD", "gold Western hours"), "PF_all": ("PORT9", "2 days before FOMC, long foreign"),
        "WM": ("PORT9", "weekly momentum (per 4 weeks)"), "FM1": ("PORT9", "dollar-factor momentum (per month)"),
        "DC": ("PORT9", "dollar carry (per month)")}


def event_rows(series_csv, spec, source):
    S = pd.read_csv(series_csv, parse_dates=["day"])
    rows = []
    for t, (inst, desc) in spec.items():
        g = S[S.test == t]
        if not len(g):
            continue
        n = len(g)
        gm, gsd = g.gross.mean() * 1e4, g.gross.std() * 1e4
        cost = (g.gross - g.net).mean() * 1e4                       # modelled round-trip cost, swap excluded
        swap = (g.net_sw - g.net).mean() * 1e4 if "net_sw" in g else 0.0
        se = g.net_sw.std() * 1e4 / np.sqrt(n) if "net_sw" in g else gsd / np.sqrt(n)
        be0, be2 = gm + swap, gm + swap - 2 * se
        pipbp = PIP_BP[inst]
        r = {"source": source, "test": t, "description": desc, "instrument": inst, "events": n,
             "gross_bp": gm, "gross_t": gm / gsd * np.sqrt(n), "swap_bp": swap, "cost_A_model_bp": cost,
             "net_A_bp": gm + swap - cost, "net_A_t": (gm + swap - cost) / se,
             "breakeven_cost_bp": be0, "breakeven_cost_t2_bp": be2, "breakeven_pips": be0 / pipbp,
             "breakeven_share_of_model_cost": be0 / cost if cost > 0 else np.nan}
        if cost > 0 and t not in ("WM", "FM1", "DC"):              # per-trade cost decomposition (single trades)
            ntr = 2 if t == "K3" else 1
            spread = max((cost / ntr - 0.2 * pipbp - 0.7) / 1.5, 0.0)
            for k, c in (("B_ecn", spread + 0.7), ("C_ibkr", spread + 0.4), ("D_inst", 0.5 * spread + 0.1)):
                c *= ntr
                r[f"cost_{k}_bp"] = c
                r[f"net_{k}_bp"] = gm + swap - c
                r[f"net_{k}_t"] = (gm + swap - c) / se
            r["quoted_spread_bp"] = spread
        rows.append(r)
    return rows


def pool_rows(rules_csv):
    R = pd.read_csv(rules_csv)
    out = []
    for _, x in R.iterrows():
        g, f, tg, tf_ = x.gross_sharpe, x.portfolio_sharpe, x.gross_t, x.portfolio_t
        if not np.isfinite(g) or g <= 0:
            continue
        out.append({"source": "run 16 random-entry pool", "test": f"{x.tf} {x.rule}", "description": "random entries, "
                    "mean-reversion harvest (pooled choice)", "instrument": "34 pairs", "gross_sharpe": g,
                    "gross_t": tg, "net_A_sharpe": f, "net_A_t": tf_,
                    "breakeven_share_of_model_cost": g / (g - f) if g > f else np.nan,
                    "t2_share_of_model_cost": (tg - 2) / (tg - tf_) if tg > 2 and tg > tf_ else np.nan})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--lit2", required=True)
    ap.add_argument("--pool", default="")
    ap.add_argument("--out", default="docs/breakeven_edges.csv")
    a = ap.parse_args()
    rows = event_rows(os.path.join(a.flows, "series.csv"), FLOWS, "run 19 flows")
    rows += event_rows(os.path.join(a.lit2, "series.csv"), LIT2, "run 22 round 2")
    if a.pool and os.path.exists(os.path.join(a.pool, "rules.csv")):
        rows += pool_rows(os.path.join(a.pool, "rules.csv"))
    T = pd.DataFrame(rows)
    T.to_csv(a.out, index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    ev = T[T.source != "run 16 random-entry pool"]
    cols = ["test", "instrument", "events", "gross_bp", "gross_t", "swap_bp", "cost_A_model_bp", "quoted_spread_bp",
            "breakeven_cost_bp", "breakeven_cost_t2_bp", "breakeven_pips", "breakeven_share_of_model_cost",
            "net_A_t", "net_B_ecn_bp", "net_B_ecn_t", "net_C_ibkr_bp", "net_C_ibkr_t", "net_D_inst_bp", "net_D_inst_t"]
    print(ev[[c for c in cols if c in ev]].round(2).to_string())
    po = T[T.source == "run 16 random-entry pool"].sort_values("gross_t", ascending=False).head(15)
    if len(po):
        print(po[["test", "gross_sharpe", "gross_t", "net_A_sharpe", "breakeven_share_of_model_cost",
                  "t2_share_of_model_cost"]].round(2).to_string())


if __name__ == "__main__":
    main()
