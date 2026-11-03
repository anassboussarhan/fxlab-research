"""Cost audit of the frozen Gotobi basket (USD/JPY + EUR/JPY, 23:00 -> 09:53 Tokyo = 14:00 -> 00:53 UTC, 80 bp stop),
2007-2025 only. For every night and leg: the quoted Dukascopy spread at the entry and exit bars (pips and bp), the
modelled round-trip cost (spread crossed at both ends + 25% of the spread + 0.1 pip slippage each side, 0.35 bp
commission each side), the stopped mid-path gross, and the net under alternative broker cost assumptions: a fixed
spread of S pips at both ends (no other slippage than 0.1 pip), commission $5 or $7 per round lot, swap 0 or the
modelled rate swap. Usage: python scripts/gotobi_costs.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

STOP = 0.008


def mid_leg(bk, i, j):
    """Mid-path return of a long from bar i to j with the 80 bp stop on the mid (gross, no costs)."""
    mo = 0.5 * (bk.bo + bk.ao)
    ml = 0.5 * (bk.bl + bk.al)
    E = mo[i]
    q = np.flatnonzero(ml[i:j] <= E * (1 - STOP))
    if len(q):
        k = i + int(q[0])
        X = min(E * (1 - STOP), mo[k]) if k > i else E * (1 - STOP)
        return X / E - 1.0, k
    return mo[j] / E - 1.0, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = []
    for p in ("USDJPY", "EURJPY"):
        bk = U.book(p)
        bk._hl()
        G, _, _ = F.calendar(bk)
        for d in G:
            if pd.Timestamp(d) > pd.Timestamp("2025-12-31"):
                continue
            i, j = F.window(bk, d)
            if i < 0:
                continue
            g, k = mid_leg(bk, i, j)
            sp_in, sp_out = bk.ao[i] - bk.bo[i], bk.ao[k] - bk.bo[k]
            mid_in = 0.5 * (bk.ao[i] + bk.bo[i])
            n, flat, rate = bk.swaps(i, k, +1, mid_in) if k > i else (0, 0.0, 0.0)
            R.append({"pair": p, "date": pd.Timestamp(d), "gross_mid": g, "stopped": k < j,
                      "spread_in_pips": sp_in / bk.pip, "spread_out_pips": sp_out / bk.pip,
                      "spread_in_bp": sp_in / mid_in * 1e4, "spread_out_bp": sp_out / mid_in * 1e4,
                      "pip_bp": bk.pip / mid_in * 1e4, "swap_nights": n, "swap_rate": rate if np.isfinite(rate) else flat,
                      "entry_hour_utc": bk.idx[i].hour, "exit_hour_utc": bk.idx[k].hour})
        print(p, len(R), flush=True)
    D = pd.DataFrame(R)
    D.to_csv(os.path.join(a.out, "night_costs.csv.gz"), index=False)
    # exact model: long pays ask + 0.25 spread + 0.1 pip in, receives bid - 0.25 spread - 0.1 pip out:
    D["model_cost_bp"] = 0.75 * D.spread_in_bp + 0.75 * D.spread_out_bp + 0.2 * D.pip_bp + 0.7
    out = {}
    for per, lo, hi in (("2007-2018", "2007", "2018-12-31"), ("2019-2025", "2019", "2025-12-31"), ("2023-2025", "2023", "2025-12-31")):
        X = D[(D.date >= lo) & (D.date <= hi)]
        if X.pair.nunique() < 2:
            continue
        o = {}
        for p, g in X.groupby("pair"):
            o[p] = {"nights": int(len(g)),
                    "spread_in_pips_median": float(g.spread_in_pips.median()), "spread_in_pips_p90": float(g.spread_in_pips.quantile(0.9)),
                    "spread_out_pips_median": float(g.spread_out_pips.median()), "spread_out_pips_p90": float(g.spread_out_pips.quantile(0.9)),
                    "pip_bp": float(g.pip_bp.mean()), "model_cost_bp_mean": float(g.model_cost_bp.mean()),
                    "swap_bp_mean": float(g.swap_rate.mean() * 1e4), "gross_mid_bp": float(g.gross_mid.mean() * 1e4)}
        # basket under scenarios
        B = X.pivot_table(index="date", columns="pair", values=["gross_mid", "model_cost_bp", "swap_rate", "pip_bp"])
        B = B.dropna()
        gross = B["gross_mid"].mean(1) * 1e4
        mc = B["model_cost_bp"].mean(1)
        sw = B["swap_rate"].mean(1) * 1e4
        pipbp = B["pip_bp"]
        sc = {"model (Dukascopy spread x1.5 crossed + slippage, 0.7 bp commission, rate swap)": gross - mc + sw}
        for S in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
            for com, cbp in (("$5", 0.5), ("$7", 0.7)):
                for swn, swv in (("rate swap", sw), ("swap 0", 0 * sw), ("swap -0.5 bp", 0 * sw - 0.5)):
                    cost = (S + 0.2) * pipbp.mean(1) + cbp
                    sc[f"spread {S} pip both ends, commission {com}, {swn}"] = gross - cost + swv
        o["basket"] = {"nights": int(len(B)), "gross_mid_bp": float(gross.mean()), "model_cost_bp": float(mc.mean()),
                       "swap_bp": float(sw.mean()),
                       "scenarios": {k: {"net_bp": float(v.mean()), "t": float(v.mean() / v.std() * np.sqrt(len(v)))} for k, v in sc.items()}}
        out[per] = o
    json.dump(out, open(os.path.join(a.out, "costs.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
