"""Round 14 A1 investigation (information only): the daily engine's same-segment rule bracketed. Turtle 1 / 2 with
the worst case (an entry or add and its stop in one segment -> stopped) and the best case (that unit is not
stop-checked until the next segment), against round 13's minute-exact results."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trend14 as T

ap = argparse.ArgumentParser(); ap.add_argument("--inp"); ap.add_argument("--r13"); ap.add_argument("--out"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
P = T.load_panels(a.inp); r13 = pd.read_csv(a.r13)
rows = []
for name, (ne, nx) in {"L1": (55, 20), "L2": (20, 10)}.items():
    per = {}
    for worst in (True, False):
        for sym, d in P.items():
            N = T.turtleN(d).shift(1).to_numpy()
            res = T.run_stop(d, N, el=T.prior_max(d.high, ne), es=T.prior_min(d.low, ne), xl=T.prior_min(d.low, nx),
                             xs=T.prior_max(d.high, nx), stop_mult=2, max_units=4, add_step=0.5, worst=worst)
            per.setdefault(worst, {})[sym] = T.monthly(d, res)
    for grp, syms in (("FX", T.FX), ("NONFX", T.NONFX)):
        o = r13[(r13.system == name) & r13.symbol.isin(syms)].groupby("month")[["pnl", "gpnl"]].sum()
        so, _ = T.pstats(o.pnl); sog, _ = T.pstats(o.gpnl)
        rec = {"system": name, "grp": grp, "r13_check_mean_pct": so["check_mean_pct"], "r13_check_t": so["check_t"],
               "r13_check_gross_t": sog["check_t"], "r13_sum_net": float(o.pnl.sum())}
        for worst in (True, False):
            m = T.portfolio([v for s, v in per[worst].items() if s in syms])
            st, _ = T.pstats(m.pnl); stg, _ = T.pstats(m.gpnl)
            g = T.month_grid()
            tag = "worst" if worst else "best"
            rec.update({f"{tag}_check_mean_pct": st["check_mean_pct"], f"{tag}_check_t": st["check_t"],
                        f"{tag}_check_gross_t": stg["check_t"], f"{tag}_sum_net": float(m.pnl.sum()),
                        f"{tag}_corr_gross": float(m.gpnl.reindex(g).fillna(0).corr(o.gpnl.reindex(g).fillna(0)))})
        rows.append(rec)
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "bracket.csv"), index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
