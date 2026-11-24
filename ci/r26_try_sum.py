"""USD/TRY on the fixed trade engine (round 24 addendum 9 / round 25 addendum 6): own-rule share, placebo share,
max t, and streams above the rounds' BH thresholds (round 24 t 4.71, round 25 t 4.80). Statistics only."""
import glob, json
import pandas as pd
out = {}
L = pd.concat([pd.read_csv(f) for f in glob.glob("inp/r26try-r1-*/LW_rows_USDTRY.csv")], ignore_index=True)
P = pd.concat([pd.read_csv(f) for f in glob.glob("inp/r26try-p-*/P_rows_*.csv")], ignore_index=True)
P = P[P.selection.isin(["default", "nested_plus"])]
for name, D, thr, ownc in (("R1_round24", L, 4.71, "own"), ("P_round25", P, 4.80, "own")):
    D = D.copy(); D["placebo"] = D.placebo.astype(bool)
    real, plc = D[~D.placebo], D[D.placebo]
    real = real[~real.unit.astype(str).str.split("|").str[0].isin(["S.SES.SESSION_MOMENTUM", "S.SES.NY_OPEN_REVERSAL", "S.FAMOUS.TTM_SQUEEZE", "S.SEAS.FIX_REVERSAL"] if name == "R1_round24" else ["S.SEAS.FIX_REVERSAL"])]
    hit = real[real[ownc].astype(bool) & (real.p2_t >= thr)]
    out[name] = {"real": len(real), "own_share": round(float(real[ownc].astype(bool).mean()), 4),
                 "placebo": len(plc), "placebo_own_share": round(float(plc[ownc].astype(bool).mean()), 4) if len(plc) else None,
                 "max_t": round(float(real.p2_t.max()), 3), "median_t": round(float(real.p2_t.median()), 3),
                 "above_bh_threshold": len(hit),
                 "top": real.sort_values("p2_t", ascending=False).head(8)[["unit", "selection", "p1_mean", "p2_mean", "p2_t"]].round(5).values.tolist()}
print(json.dumps(out, indent=1, default=str))
