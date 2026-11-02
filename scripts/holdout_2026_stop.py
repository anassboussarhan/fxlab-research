"""Second, separately pre-registered use of the 2026 data (2026-01-01..2026-10-31): the exit rule chosen on
2007-2018 in scripts/gotobi_risk_lab.py (stop 0.75 W; runner-up stop 40 bp) against the frozen no-stop rule, on the
same 2026 Gotobi days as docs/holdout_2026_results.md. Nothing is chosen from this run. Code identical to the lab's
exit_sim / window_units. Verdict (fixed now): the stop "keeps the gains" if its 2026 basket mean is at least 75% of
the no-stop mean, and "cuts the losses" if its CVaR5 open loss and its worst trade are smaller in size.
Usage: python scripts/holdout_2026_stop.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gotobi_risk_lab as R  # noqa: E402
import holdout_2026 as H  # noqa: E402

KEEP = ("none", "stop 0.75 W", "stop 40 bp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import flow_edges_lab as F
    res, rows = {}, []
    for p in ("USDJPY", "EURJPY"):
        bk = H.book26(p)
        bk._hl()
        G, _, _ = F.calendar(bk)
        W = R.window_units(bk, G)
        G26 = [d for d in G if pd.Timestamp("2026-01-01") <= d <= pd.Timestamp("2026-10-31")]
        out = R.run_pair(bk, G26, W)
        res[p] = {k: out[k] for k in KEEP}
        for k in KEEP:
            rows.append({"pair": p, "variant": k, **R.stats(res[p][k])})
    for k in KEEP:
        j = res["USDJPY"][k].join(res["EURJPY"][k], lsuffix="_u", rsuffix="_e", how="outer")
        b = pd.DataFrame({"net": j[["net_u", "net_e"]].mean(axis=1), "mae": j[["mae_u", "mae_e"]].mean(axis=1)}).dropna()
        rows.append({"pair": "basket", "variant": k, **R.stats(b)})
        b.to_csv(os.path.join(a.out, f"basket_{k.replace(' ', '_')}.csv"))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.out, "tests.csv"), index=False)
    bb = T[T.pair == "basket"].set_index("variant")
    verdict = {}
    for k in ("stop 0.75 W", "stop 40 bp"):
        verdict[k] = {"keeps_gains": bool(bb.loc[k, "mean_bp"] >= 0.75 * bb.loc["none", "mean_bp"]),
                      "cuts_losses": bool(bb.loc[k, "cvar5_open_loss_bp"] > bb.loc["none", "cvar5_open_loss_bp"]
                                          and bb.loc[k, "worst_bp"] > bb.loc["none", "worst_bp"])}
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string())
    print(verdict)


if __name__ == "__main__":
    main()
