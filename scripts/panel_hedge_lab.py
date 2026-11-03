"""H and F of docs/panel_hedge_preregistration.md (rules fixed 2026-11-02, before the data was read).
B0 = the frozen Gotobi basket with the 80 bp stop, net (docs/gotobi_losses/trades_basket_stop_80_bp.csv).
Instruments: USTBONDTRUSD, BUNDTREUR, USA500IDXUSD, USATECHIDXUSD, JPNIDXJPY, XAUUSD. Fit 2019-2022, check 2023-2025.

H (12 tests): hedge position -beta x X over (a) 14:00 -> 00:53 UTC (the whole night) or (b) 14:00 -> 21:00 UTC;
  beta = OLS slope of B0 on X's mid return over the window, fit nights; the hedge leg pays its own costs
  (hedged = B0 + |beta| x net of the short X leg if beta > 0, else of the long leg). Counts if on 2023-2025:
  hedged t >= B0 t + 0.3, sd at least 10% lower, mean at most 0.5 bp lower; the t also higher on 2019-2022; and the
  Holm-adjusted p of the variance reduction (Pitman-Morgan test for paired variances, one-sided) <= 0.10.
F (12 tests): X's mid return 12:00 -> 14:00 and 08:00 -> 14:00 UTC on the entry day vs B0 (Spearman); counts if
  fit p < 0.05, same sign on check with one-sided p < 0.05, and BH q <= 0.10 over the 12.
Usage: python scripts/panel_hedge_lab.py --out DIR"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import edges_lab2 as E  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

INSTR = ("USTBONDTRUSD", "BUNDTREUR", "USA500IDXUSD", "USATECHIDXUSD", "JPNIDXJPY", "XAUUSD")
FIT_END, CHECK_END = pd.Timestamp("2022-12-31"), pd.Timestamp("2025-12-31")


def tval(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else np.nan


def pitman_morgan(a, b):
    """One-sided p that var(b) < var(a) for paired samples (correlation of a+b and a-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r, p = st.pearsonr(a + b, a - b)
    return float(p / 2 if r > 0 else 1 - p / 2)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = E.gotobi_nights()
    B0 = B.net[(B.index > "2018-12-31") & (B.index <= CHECK_END)]
    H, Fr, frames = [], [], {}
    for sym in INSTR:
        bk = E.book(sym)
        if bk is None:
            H.append({"instrument": sym, "status": "no data"})
            continue
        print(sym, bk.idx[0], bk.idx[-1], flush=True)
        rows = {}
        for d in B0.index:
            e0 = U.jst(d - pd.Timedelta(days=1), 23, 0)                     # 14:00 UTC the evening before
            r = {}
            for w, t1 in (("a", U.jst(d, 9, 53)), ("b", E.utc(e0, 21, 0))):
                x = E.both(bk, e0, t1, tol=10)
                if x is not None:
                    r[f"{w}_long"], r[f"{w}_short"], r[f"{w}_gross"] = x["long"], x["short"], x["gross"]
            for nm, h0 in (("f2h", 12), ("f6h", 8)):
                p0 = E.mid_at(bk, E.utc(e0, h0, 0), 10)
                p1 = E.mid_at(bk, e0, 10, before=True)
                r[nm] = p1 / p0 - 1 if np.isfinite(p0) and np.isfinite(p1) else np.nan
            rows[d] = r
        X = pd.DataFrame.from_dict(rows, orient="index")
        X.index = pd.DatetimeIndex(X.index)
        frames[sym] = X
        for w in ("a", "b"):
            if f"{w}_gross" not in X:
                H.append({"instrument": sym, "window": w, "status": "no trades"})
                continue
            j = pd.concat([B0.rename("b0"), X[[f"{w}_long", f"{w}_short", f"{w}_gross"]]], axis=1).dropna()
            fit, chk = j[j.index <= FIT_END], j[j.index > FIT_END]
            if len(fit) < 50 or len(chk) < 30:
                H.append({"instrument": sym, "window": w, "status": f"too few nights ({len(fit)} / {len(chk)})"})
                continue
            beta = float(np.polyfit(fit[f"{w}_gross"], fit.b0, 1)[0])
            leg = (j[f"{w}_short"] if beta > 0 else j[f"{w}_long"]) * abs(beta)
            hd = j.b0 + leg
            res = {"instrument": sym, "window": "14:00->00:53" if w == "a" else "14:00->21:00", "beta": beta,
                   "fit_corr": float(np.corrcoef(fit[f"{w}_gross"], fit.b0)[0, 1])}
            for nm, m in (("fit", j.index <= FIT_END), ("check", j.index > FIT_END)):
                res.update({f"{nm}_n": int(m.sum()), f"{nm}_B0_bp": j.b0[m].mean() * 1e4, f"{nm}_hedged_bp": hd[m].mean() * 1e4,
                            f"{nm}_B0_t": tval(j.b0[m]), f"{nm}_hedged_t": tval(hd[m]),
                            f"{nm}_B0_sd_bp": j.b0[m].std() * 1e4, f"{nm}_hedged_sd_bp": hd[m].std() * 1e4})
            m = j.index > FIT_END
            res["p_var"] = pitman_morgan(j.b0[m], hd[m])
            res["rules_met"] = bool(res["check_hedged_t"] >= res["check_B0_t"] + 0.3 and
                                    res["check_hedged_sd_bp"] <= 0.9 * res["check_B0_sd_bp"] and
                                    res["check_hedged_bp"] >= res["check_B0_bp"] - 0.5 and
                                    res["fit_hedged_t"] > res["fit_B0_t"])
            H.append(res)
        for nm in ("f2h", "f6h"):
            j = pd.concat([B0.rename("y"), X[nm].rename("x")], axis=1).dropna()
            fit, chk = j[j.index <= FIT_END], j[j.index > FIT_END]
            if len(fit) < 50 or len(chk) < 30:
                Fr.append({"instrument": sym, "signal": nm, "status": "too few nights"})
                continue
            r1, p1 = st.spearmanr(fit.x, fit.y)
            r2, p2 = st.spearmanr(chk.x, chk.y)
            p_one = float(p2 / 2 if np.sign(r2) == np.sign(r1) else 1 - p2 / 2)
            Fr.append({"instrument": sym, "signal": "12:00->14:00 UTC" if nm == "f2h" else "08:00->14:00 UTC",
                       "fit_n": int(len(fit)), "fit_rho": float(r1), "fit_p": float(p1), "check_n": int(len(chk)),
                       "check_rho": float(r2), "check_p_one_sided": p_one})
    Hd = pd.DataFrame(H)
    if "p_var" in Hd:
        ok = Hd.p_var.notna()
        Hd.loc[ok, "holm_p"] = holm(Hd.p_var[ok])
        Hd["counts"] = Hd.get("rules_met", False).fillna(False).astype(bool) & (Hd.holm_p <= 0.10)
    Fd = pd.DataFrame(Fr)
    if "check_p_one_sided" in Fd:
        from statsmodels.stats.multitest import multipletests
        ok = Fd.check_p_one_sided.notna()
        Fd.loc[ok, "bh_q"] = multipletests(Fd.check_p_one_sided[ok], method="fdr_bh")[1]
        Fd["counts"] = (Fd.fit_p < 0.05) & (np.sign(Fd.fit_rho) == np.sign(Fd.check_rho)) & \
            (Fd.check_p_one_sided < 0.05) & (Fd.bh_q <= 0.10)
    Hd.to_csv(os.path.join(a.out, "hedges.csv"), index=False)
    Fd.to_csv(os.path.join(a.out, "pre_entry.csv"), index=False)
    for k, f in frames.items():
        f.to_csv(os.path.join(a.out, f"nights_{k}.csv"))
    pd.set_option("display.width", 250)
    print(Hd.round(3).to_string())
    print(Fd.round(3).to_string())
    json.dump({"hedges_that_count": Hd[Hd.get("counts", False) == True].instrument.tolist() if "counts" in Hd else [],  # noqa: E712
               "signals_that_count": Fd[Fd.get("counts", False) == True].instrument.tolist() if "counts" in Fd else []},  # noqa: E712
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
