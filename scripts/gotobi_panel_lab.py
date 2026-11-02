"""The Gotobi night on every instrument we can get: all FX pairs (majors, crosses, Scandinavian and exotic yen
pairs), gold and silver, index CFDs (Nikkei 225, S&P 500, Nasdaq 100, Dow, Hang Seng, ASX 200, DAX, China A50),
oil, copper and bond futures CFDs. Window as in the frozen trade: 23:00 Tokyo the evening before -> 09:53 Tokyo on
Gotobi days; control nights = the same window on every other Tuesday-Friday (not a Japanese holiday). M1 bid/ask
fills and costs as in the other labs (crossing the spread plus 25% of it and 0.1 pip; 0.35 bp commission a side;
swap). Development data only (to 2025-12-31). 2007-2018 train, 2019-2025 test (each instrument from its history
start).

FIXED BEFORE ANY RESULT:
  Gotobi effect of an instrument = mean gross (mid) return on Gotobi nights minus on control nights; its direction
  is the sign on train.
  CARRIER (screen on train): |Welch t of the Gotobi-minus-control difference| >= 2.0 and the net mean on Gotobi
    nights in that direction > 0. CONFIRMED: on test, the net mean in that direction > 0 with one-sided t >= 1.645
    and the Gotobi-minus-control difference of the same sign; Benjamini-Hochberg over the carriers (q <= 0.10).
  Correlation of each instrument's Gotobi-night return with the USD/JPY + EUR/JPY basket (B0); its alpha after
    the basket (OLS on train; the test residual uses the train beta).
  BASKETS on Gotobi nights, weights from train: B0 = USD/JPY + EUR/JPY; B1 = all train carriers, equal weight;
    B2 = all train carriers, inverse train variance; B3 = B0 plus the carriers whose train correlation with B0 is
    below 0.5, equal weight. A basket is a forward candidate only if its 2019-2025 t is at least 0.5 above B0's and
    its CVaR5 (net) is no more than 10% worse.
  HEDGES: for every non-carrier X with train |correlation| >= 0.3 with B0 on Gotobi nights: B0 - beta X (beta by
    OLS on train, X long net); a hedge counts only if it raises t by >= 0.5 on train and on test.
  USD OR YEN: the Gotobi effect on the USD pairs (signed so that + = dollar up) next to the yen pairs.
Usage: python scripts/gotobi_panel_lab.py --instrument SYM --out DIR      (one instrument's nights)
       python scripts/gotobi_panel_lab.py --aggregate DIR --out DIR       (the tables)"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

TRAIN_END = pd.Timestamp("2018-12-31")
DAY = pd.Timedelta(days=1)
FX3 = {"USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK", "CNH", "SGD", "ZAR", "MXN", "TRY", "HKD"}


def is_fx(sym):
    return len(sym) == 6 and sym[:3] in FX3 and sym[3:] in FX3


def load_book(sym):
    m = U.load(sym)
    if m is None or len(m) < 1000:
        return None
    if is_fx(sym):
        pip = 0.01 if sym.endswith("JPY") else 1e-4
    else:
        pip = 1e-4 * float(np.median(0.5 * (m.bid_c + m.ask_c)))       # about 1 bp of the price
    return FL.Book(m, pip, sym if is_fx(sym) else None)


def nights(sym):
    bk = load_book(sym)
    if bk is None:
        return None
    bk._hl()
    G, _, hol = F.calendar(bk)
    Gs = set(G)
    start, end = bk.idx[0].tz_convert(None).normalize(), bk.idx[-1].tz_convert(None).normalize()
    ctrl = [d for d in pd.bdate_range(start, end) if d not in Gs and d not in hol and d.weekday() != 0]
    rows = []
    for d, g in [(x, True) for x in G] + [(x, False) for x in ctrl]:
        i, j = F.window(bk, d)
        if i < 0:
            continue
        if not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all() or bk.ao[i] < bk.bo[i]:
            continue
        rl, rs = bk.trade_ij(i, j, +1), bk.trade_ij(i, j, -1)
        if rl is None or rs is None:
            continue
        E = bk.ao[i]
        rows.append({"date": d, "gotobi": g, "net_long": U.net_of(rl), "net_short": U.net_of(rs), "gross": float(rl[1]),
                     "spread_bp": (bk.ao[i] - bk.bo[i]) / bk.mo[i] * 1e4,
                     "mae_long": float(min(np.min(bk.bl[i:j]) / E - 1.0, 0.0))})
    return pd.DataFrame(rows).set_index("date").sort_index()


# ------------------------------------------------------------------------------------------------ aggregate
def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def welch(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    return float(st.ttest_ind(a, b, equal_var=False).statistic) if len(a) > 2 and len(b) > 2 else np.nan


def periods(idx):
    idx = pd.DatetimeIndex(idx)
    return {"train": idx <= TRAIN_END, "test": idx > TRAIN_END}


def cvar5(x):
    x = pd.Series(x).dropna()
    if len(x) < 20:
        return np.nan
    q = x.quantile(0.05)
    return float(x[x <= q].mean())


def aggregate(src, out):
    from cross_market_ta_lab import bh_q
    os.makedirs(out, exist_ok=True)
    N = {}
    for f in glob.glob(os.path.join(src, "**", "nights_*.csv"), recursive=True):
        sym = os.path.basename(f)[7:-4]
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df):
            N[sym] = df
    rows = []
    for sym, df in sorted(N.items()):
        r = {"instrument": sym, "first": str(df.index.min().date()), "last": str(df.index.max().date())}
        for per, m in periods(df.index).items():
            g, c = df[m & df.gotobi.to_numpy()], df[m & ~df.gotobi.to_numpy()]
            r[f"{per}_n"], r[f"{per}_n_ctrl"] = len(g), len(c)
            if len(g) < 20 or len(c) < 20:
                continue
            r[f"{per}_gross_bp"], r[f"{per}_ctrl_gross_bp"] = g.gross.mean() * 1e4, c.gross.mean() * 1e4
            r[f"{per}_diff_bp"], r[f"{per}_diff_t"] = (g.gross.mean() - c.gross.mean()) * 1e4, welch(g.gross, c.gross)
            r[f"{per}_net_long_bp"], r[f"{per}_net_long_t"] = g.net_long.mean() * 1e4, tstat(g.net_long)
            r[f"{per}_net_short_bp"], r[f"{per}_net_short_t"] = g.net_short.mean() * 1e4, tstat(g.net_short)
            r[f"{per}_spread_bp"] = g.spread_bp.median()
        d = np.sign(r.get("train_diff_bp", np.nan))
        r["direction"] = "long" if d > 0 else ("short" if d < 0 else "")
        if r["direction"]:
            col = "net_long" if d > 0 else "net_short"
            r["train_net_dir_bp"], r["train_net_dir_t"] = r.get(f"train_{col}_bp", np.nan), r.get(f"train_{col}_t", np.nan)
            r["test_net_dir_bp"], r["test_net_dir_t"] = r.get(f"test_{col}_bp", np.nan), r.get(f"test_{col}_t", np.nan)
            r["carrier"] = bool(abs(r.get("train_diff_t", 0) or 0) >= 2.0 and (r["train_net_dir_bp"] or -1) > 0)
        else:
            r["carrier"] = False
        rows.append(r)
    P = pd.DataFrame(rows).set_index("instrument")
    car = P[P.carrier.fillna(False)].index.tolist()
    p = [st.norm.sf(P.loc[s, "test_net_dir_t"]) if np.isfinite(P.loc[s].get("test_net_dir_t", np.nan)) else np.nan
         for s in car]
    P["test_q_bh"] = np.nan
    ok = [k for k, v in enumerate(p) if np.isfinite(v)]
    if ok:
        P.loc[[car[k] for k in ok], "test_q_bh"] = bh_q(np.array([p[k] for k in ok]))
    P["confirmed"] = (P.carrier.fillna(False) & (P.test_net_dir_t >= 1.645) &
                      (np.sign(P.test_diff_bp) == np.sign(P.train_diff_bp)) & (P.test_q_bh <= 0.10))

    # Gotobi-night returns in each instrument's direction (long net for those without a direction)
    def dir_series(sym):
        df = N[sym]
        g = df[df.gotobi]
        return g.net_short if P.loc[sym, "direction"] == "short" else g.net_long
    R = pd.DataFrame({s: dir_series(s) for s in N})
    B0 = R[[c for c in ("USDJPY", "EURJPY") if c in R]].mean(axis=1)
    trm = R.index <= TRAIN_END
    for s in R.columns:
        x = R[s]
        for per, m in (("train", trm), ("test", ~trm)):
            j = pd.concat([x[m], B0[m]], axis=1).dropna()
            P.loc[s, f"{per}_corr_B0"] = j.corr().iloc[0, 1] if len(j) > 20 else np.nan
        j = pd.concat([x[trm], B0[trm]], axis=1).dropna()
        if len(j) > 30:
            b, a = np.polyfit(j.iloc[:, 1], j.iloc[:, 0], 1)
            res_tr = j.iloc[:, 0] - b * j.iloc[:, 1]
            P.loc[s, "beta_B0"], P.loc[s, "train_alpha_bp"], P.loc[s, "train_alpha_t"] = b, res_tr.mean() * 1e4, tstat(res_tr)
            jt = pd.concat([x[~trm], B0[~trm]], axis=1).dropna()
            if len(jt) > 20:
                res_te = jt.iloc[:, 0] - b * jt.iloc[:, 1]
                P.loc[s, "test_alpha_bp"], P.loc[s, "test_alpha_t"] = res_te.mean() * 1e4, tstat(res_te)
    P.to_csv(os.path.join(out, "panel.csv"))
    # baskets
    var_tr = R[trm].var()
    low_corr = [s for s in car if s not in ("USDJPY", "EURJPY") and (P.loc[s, "train_corr_B0"] or 1) < 0.5]
    defs = {"B0 USD/JPY + EUR/JPY": {s: 1.0 for s in ("USDJPY", "EURJPY") if s in R},
            "B1 all train carriers, equal": {s: 1.0 for s in car},
            "B2 all train carriers, inverse variance": {s: 1.0 / var_tr[s] for s in car if var_tr.get(s, 0) > 0},
            "B3 B0 + low-correlation carriers": {**{s: 1.0 for s in ("USDJPY", "EURJPY") if s in R},
                                                 **{s: 1.0 for s in low_corr}}}
    brows, series = [], {}
    for name, w in defs.items():
        if not w:
            continue
        W_ = pd.Series(w)
        X = R[W_.index]
        num = (X * W_).sum(axis=1, min_count=1)
        den = X.notna().mul(W_).sum(axis=1)
        b = (num / den).dropna()
        series[name] = b
        r = {"basket": name, "members": ", ".join(W_.index)}
        for per, m in periods(b.index).items():
            x = b[m]
            r.update({f"{per}_n": len(x), f"{per}_mean_bp": x.mean() * 1e4, f"{per}_t": tstat(x),
                      f"{per}_win": float((x > 0).mean()), f"{per}_worst_bp": x.min() * 1e4, f"{per}_cvar5_bp": cvar5(x) * 1e4})
        brows.append(r)
    Bt = pd.DataFrame(brows).set_index("basket")
    ref = Bt.loc["B0 USD/JPY + EUR/JPY"]
    Bt["forward_candidate"] = (Bt.test_t >= ref.test_t + 0.5) & (Bt.test_cvar5_bp >= 1.1 * ref.test_cvar5_bp)
    Bt.to_csv(os.path.join(out, "baskets.csv"))
    pd.DataFrame(series).to_csv(os.path.join(out, "basket_nights.csv"))
    # hedges
    hrows = []
    for s in R.columns:
        if s in car or s in ("USDJPY", "EURJPY"):
            continue
        gx = N[s][N[s].gotobi]
        xg = gx.gross
        j = pd.concat([B0[trm], xg[xg.index <= TRAIN_END]], axis=1).dropna()
        if len(j) < 40 or abs(j.corr().iloc[0, 1]) < 0.3:
            continue
        beta = np.polyfit(j.iloc[:, 1], j.iloc[:, 0], 1)[0]
        leg = gx.net_short if beta > 0 else gx.net_long                  # the hedge position pays its own costs
        h = (B0 + abs(beta) * leg).dropna()
        r = {"hedge": s, "beta": beta, "train_corr": j.corr().iloc[0, 1]}
        for per, m in periods(h.index).items():
            b0 = B0.reindex(h.index)[m]
            r.update({f"{per}_n": int(m.sum()), f"{per}_B0_t": tstat(b0), f"{per}_hedged_t": tstat(h[m]),
                      f"{per}_B0_bp": b0.mean() * 1e4, f"{per}_hedged_bp": h[m].mean() * 1e4,
                      f"{per}_B0_sd_bp": b0.std() * 1e4, f"{per}_hedged_sd_bp": h[m].std() * 1e4})
        r["counts"] = bool((r.get("train_hedged_t", -9) - r.get("train_B0_t", 0)) >= 0.5 and
                           (r.get("test_hedged_t", -9) - r.get("test_B0_t", 0)) >= 0.5)
        hrows.append(r)
    pd.DataFrame(hrows).to_csv(os.path.join(out, "hedges.csv"), index=False)
    summ = {"instruments": len(N), "carriers_train": car, "confirmed": P[P.confirmed.fillna(False)].index.tolist(),
            "forward_candidate_baskets": Bt[Bt.forward_candidate].index.tolist(),
            "hedges_that_count": [r["hedge"] for r in hrows if r["counts"]]}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 40)
    cols = ["first", "train_n", "train_diff_bp", "train_diff_t", "direction", "train_net_dir_bp", "train_net_dir_t",
            "carrier", "test_n", "test_diff_bp", "test_diff_t", "test_net_dir_bp", "test_net_dir_t", "confirmed",
            "train_corr_B0", "test_corr_B0", "train_alpha_bp", "test_alpha_bp"]
    print(P[[c for c in cols if c in P]].round(2).to_string())
    print(Bt.round(2).to_string())
    print(pd.DataFrame(hrows).round(2).to_string() if hrows else "no hedges")
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="")
    ap.add_argument("--aggregate", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.instrument:
        try:
            df = nights(a.instrument)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            print(traceback.format_exc())
            df = None
            json.dump({"error": repr(ex)[:500]}, open(os.path.join(a.out, f"error_{a.instrument}.json"), "w"))
        if df is not None:
            df.to_csv(os.path.join(a.out, f"nights_{a.instrument}.csv"))
            print(a.instrument, len(df), "nights", df.index.min(), df.index.max(),
                  df.groupby([df.index.year, "gotobi"]).size().unstack().tail(8).to_dict())
        else:
            print(a.instrument, "no data")
    else:
        aggregate(a.aggregate, a.out)


if __name__ == "__main__":
    main()
