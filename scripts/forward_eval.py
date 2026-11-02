"""Monthly evaluation of the frozen forward test (docs/forward_test_preregistration.md): every Gotobi night from
2026-11-05 is traded (no stop, 80 bp stop, 100 bp stop), its features are computed with the frozen code and scored
with the frozen filters (results/frozen_forward_v1.json, results/frozen_forward_v2.json), and the pre-registered
tests FT1-FT5 are reported on all nights so far. Nothing is refitted.

  --dry YEAR   code check on development data: M1 for YEAR-2 to YEAR from the development store (no 2026 data),
               the same steps for the nights of YEAR, compared with the development files (trades and features in
               docs/gotobi_accuracy/ and docs/gotobi_losses/). It must match before forward nights are evaluated.
  (default)    M1 from 2025-07-01 (the late-2025 and 2026 holdout months serve only as indicator history; needs the
               holdout unlock) to the last complete month; nights from 2026-11-05.
Known limits of the frozen code, kept as they are: the repo's rate table ends in December 2025 and is carried
forward (rate features and swaps); its FOMC table ends in 2022 (fomc_night is 0 since 2023, as in development);
its intervention list ends in July 2024.
Usage: python scripts/forward_eval.py --out DIR [--dry 2025] [--cot-data DIR]"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flow_edges_lab as F  # noqa: E402
import flows_lab as FL  # noqa: E402
import freeze_forward as FZ  # noqa: E402
import gotobi_accuracy_lab as A  # noqa: E402
import gotobi_losses_lab as L  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = pd.Timestamp("2026-11-05")
DEV_MEAN_BP = 4.6
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
V1_MKT = (("vix", "^VIX", False, "level"), ("vix_chg_1w", "^VIX", False, "chg5"), ("spx_ret_1d", "^GSPC", False, "ret1"),
          ("n225_ret_1d", "^N225", True, "ret1"), ("us10y_chg_1w", "^TNX", False, "chg5"))


def book(sym, dry):
    from fxlab.data.store import holdout_unlocked, load_m1
    if dry:
        m = load_m1(sym, f"{dry - 2}-01-01", f"{dry}-12-31 23:59", columns=COLS)
    else:
        if not holdout_unlocked():
            raise SystemExit("the 2026 warm-up months need the holdout unlock (FXLAB_HOLDOUT_UNLOCK)")
        m = load_m1(sym, "2025-07-01", None, allow_holdout=True, columns=COLS)
    m = m.dropna()
    m = m[(m.ask_o >= m.bid_o) & (m.bid_o > 0)]
    return FL.Book(m, 0.01, sym)


def yahoo(tickers, end):
    out, err = {}, {}
    for t in sorted(set(tickers)):
        try:
            import yfinance as yf
            h = yf.download(t, start="2005-01-01", end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            auto_adjust=True, progress=False)
            c, o = h["Close"], h["Open"]
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            o = o.iloc[:, 0] if isinstance(o, pd.DataFrame) else o
            c.index = o.index = pd.DatetimeIndex(c.index).tz_localize(None)
            df = pd.DataFrame({"close": c, "open": o})
            out[t] = df[(df.index <= end) & (df.close > 0)].dropna(subset=["close"])
        except Exception as ex:                                             # noqa: BLE001
            err[t] = repr(ex)[:200]
    return out, err


def cot_series(cot_dir, end):
    """Speculators' JPY net share z-score, as in scripts/gotobi_accuracy_lab.py, from the CFTC legacy files."""
    import cot_lab as C
    p = os.path.join(cot_dir, "cot.csv") if cot_dir else ""
    if p and os.path.exists(p):
        raw = pd.read_csv(p, parse_dates=["date"])
    else:
        import requests
        frames = []
        for y in range(2016, end.year + 1):
            r = requests.get(f"https://www.cftc.gov/files/dea/history/deacot{y}.zip", timeout=120,
                             headers={"User-Agent": "Mozilla/5.0 research"})
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            frames += [pd.read_csv(z.open(n), low_memory=False) for n in z.namelist() if n.lower().endswith(".txt")]
        C.END = end
        raw = C.tidy_cot(pd.concat(frames, ignore_index=True))
    c = raw[(raw.market == "JPY") & (raw.date <= end)].set_index("date").sort_index()
    return C.zscore((c.nc_long - c.nc_short) / c.oi).dropna()


def pair_nights(bk):
    """Every Gotobi night in the loaded bars (earlier nights give the previous-trade feature and W)."""
    G, _, hol = F.calendar(bk)
    W = R.window_units(bk, G)
    out = {"none": {}, "stop 80 bp": {}, "stop 100 bp": {}}
    for d in G:
        w = float(W.get(d, np.nan))
        i, j = F.window(bk, d)
        if not np.isfinite(w) or w <= 0 or i < 0:
            continue
        if not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all():
            continue
        H = A.hours_map(bk, d, i, j)
        out["none"][d] = A.sim(bk, i, j, w, {}, H)
        out["stop 80 bp"][d] = A.sim(bk, i, j, w, {"sl_bp": 80}, H)
        out["stop 100 bp"][d] = A.sim(bk, i, j, w, {"sl_bp": 100}, H)
    fr = {}
    for k, v in out.items():
        f = pd.DataFrame.from_dict(v, orient="index").sort_index()
        f.index = pd.DatetimeIndex(f.index)
        fr[k] = f
    return fr, G, W, hol


def cvar5(m):
    m = m.dropna()
    if len(m) < 5:
        return float(m.min()) if len(m) else np.nan
    q = m.quantile(0.05)
    return float(m[m <= q].mean())


def verdict(mean, t, contra=None):
    if contra is not None and contra:
        return "contradicted"
    if not np.isfinite(mean):
        return "no data"
    if mean > 0 and t >= 1.645:
        return "confirmed"
    return "consistent" if mean > 0 else "not supported"


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(st.ttest_ind(a, b, equal_var=False).statistic)


def tests(N):
    """FT1-FT5 on the nights table N (basket, bp)."""
    out = {"nights": int(len(N)), "first": str(N.index.min().date()) if len(N) else None,
           "last": str(N.index.max().date()) if len(N) else None}
    for tag, col in (("FT1 forward-v1 filter", "keep_v1"), ("FT5 forward-v2 filter", "keep_v2")):
        k, s = N.loc[N[col], "net_none_bp"], N.loc[~N[col], "net_none_bp"]
        diff, t = (k.mean() - s.mean()) if len(k) and len(s) else np.nan, welch(k, s)
        out[tag] = {"kept": int(len(k)), "skipped": int(len(s)), "kept_mean_bp": k.mean(), "skipped_mean_bp": s.mean(),
                    "difference_bp": diff, "t": t, "verdict": verdict(diff, t if np.isfinite(t) else -np.inf,
                                                                    np.isfinite(t) and t <= -1.645)}
    k = N.loc[N.keep_v1, "net_none_bp"]
    tk = float(k.mean() / k.std() * np.sqrt(len(k))) if len(k) > 2 and k.std() > 0 else np.nan
    out["FT2 kept nights (v1)"] = {"n": int(len(k)), "mean_bp": k.mean(), "t": tk,
                                   "verdict": verdict(k.mean(), tk if np.isfinite(tk) else -np.inf)}
    for tag, col in (("FT3 stop 80 bp", "net_s80_bp"), ("S2 stop 100 bp", "net_s100_bp")):
        dd = N[col] - N.net_none_bp
        mcol = col.replace("net_", "mae_")
        out[tag] = {"mean_bp": N[col].mean(), "paired_difference_bp": dd.mean(),
                    "keeps_the_gain": bool(dd.mean() >= -0.5) if len(dd) else None,
                    "worst_bp": N[col].min(), "worst_no_stop_bp": N.net_none_bp.min(),
                    "cvar5_open_loss_bp": cvar5(N[mcol]), "cvar5_open_loss_no_stop_bp": cvar5(N.mae_none_bp),
                    "cuts_the_loss": bool(N[col].min() > N.net_none_bp.min() and cvar5(N[mcol]) > cvar5(N.mae_none_bp))
                    if len(N) else None}
    x = N.net_none_bp
    t = float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
    td = float((x.mean() - DEV_MEAN_BP) / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
    out["FT4 the trade"] = {"mean_bp": x.mean(), "t": t, "winners": float((x > 0).mean()) if len(x) else np.nan,
                            "verdict": verdict(x.mean(), t if np.isfinite(t) else -np.inf, np.isfinite(td) and td <= -1.645)}
    return out


def report_md(S, N, mode):
    lines = [f"# Forward test: results so far ({mode})", "",
             f"Nights {S['nights']} ({S['first']} to {S['last']}). Basket, bp a night, after costs. Rules and verdicts: "
             "`docs/forward_test_preregistration.md`. First verdict after 2027-10-29.", "",
             "| Test | Result | Verdict so far |", "|---|---|---|"]
    f1, f5, f2, f4 = S["FT1 forward-v1 filter"], S["FT5 forward-v2 filter"], S["FT2 kept nights (v1)"], S["FT4 the trade"]

    def g(x, nd=2):
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"
    lines.append(f"| FT1 filter v1 (primary) | kept {f1['kept']}: {g(f1['kept_mean_bp'])}; skipped {f1['skipped']}: "
                 f"{g(f1['skipped_mean_bp'])}; difference {g(f1['difference_bp'])} (t {g(f1['t'])}) | {f1['verdict']} |")
    lines.append(f"| FT2 kept nights (v1) | {g(f2['mean_bp'])} (t {g(f2['t'])}) | {f2['verdict']} |")
    for tag in ("FT3 stop 80 bp", "S2 stop 100 bp"):
        s = S[tag]
        lines.append(f"| {tag} | {g(s['mean_bp'])} vs {g(f4['mean_bp'])} without; worst {g(s['worst_bp'], 0)} vs "
                     f"{g(s['worst_no_stop_bp'], 0)}; CVaR5 open loss {g(s['cvar5_open_loss_bp'], 0)} vs "
                     f"{g(s['cvar5_open_loss_no_stop_bp'], 0)} | keeps the gain: {s['keeps_the_gain']}; cuts the loss: "
                     f"{s['cuts_the_loss']} |")
    lines.append(f"| FT4 the trade | {g(f4['mean_bp'])} (t {g(f4['t'])}), {g(100 * f4['winners'], 0)}% winners | "
                 f"{f4['verdict']} |")
    lines.append(f"| FT5 filter v2 (secondary) | kept {f5['kept']}: {g(f5['kept_mean_bp'])}; skipped {f5['skipped']}: "
                 f"{g(f5['skipped_mean_bp'])}; difference {g(f5['difference_bp'])} (t {g(f5['t'])}) | {f5['verdict']} |")
    lines += ["", "## Nights", "", "| Night | No stop | Stop 80 bp | Stop 100 bp | v1 p(win) | v1 | v2 p(win) | v2 |",
              "|---|---|---|---|---|---|---|---|"]
    for d, r in N.iterrows():
        lines.append(f"| {d.date()} | {r.net_none_bp:+.1f} | {r.net_s80_bp:+.1f} | {r.net_s100_bp:+.1f} | {r.p_v1:.3f} | "
                     f"{'trade' if r.keep_v1 else 'skip'} | {r.p_v2:.3f} | {'trade' if r.keep_v2 else 'skip'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", type=int, default=0)
    ap.add_argument("--cot-data", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F1 = json.load(open(os.path.join(ROOT, "results", "frozen_forward_v1.json")))
    F2 = json.load(open(os.path.join(ROOT, "results", "frozen_forward_v2.json")))
    errors = {}
    books = {p: book(p, a.dry) for p in ("USDJPY", "EURJPY")}
    last_bar = min(b.idx[-1] for b in books.values())
    if a.dry:
        keep_night = (lambda d: d.year == a.dry)
        end = pd.Timestamp(f"{a.dry}-12-31")
    else:
        last_day = (last_bar.tz_convert(None) - pd.Timedelta(hours=2)).normalize()
        keep_night = (lambda d: START <= d <= last_day)
        end = last_day
    tick = {t for _, t, _, _ in L.MKT} | {t for _, t, _ in L.NIGHT} | {"^VIX", "^VIX3M", "^TNX", "^IRX", "^N225", "^GSPC"}
    data, e = yahoo(tick, end)
    errors.update({f"yahoo {k}": v for k, v in e.items()})
    mkt_v1 = [(n, data[t].close, s, k) for n, t, s, k in V1_MKT if t in data]
    try:
        cot = cot_series(a.cot_data, end)
    except Exception as ex:                                                 # noqa: BLE001
        cot, errors["cot"] = None, repr(ex)[:300]
    import holidays
    us_hol = {pd.Timestamp(d) for d in holidays.financial_holidays("NYSE", years=range(2006, 2031))}
    fomc = set()
    for y, s in FL.FOMC_FALLBACK.items():
        fomc |= {pd.Timestamp(f"{y}-{x}") for x in s.split()}
    fr, X = {}, {}
    for p, bk in books.items():
        bk._hl()
        fr[p], G, W, hol = pair_nights(bk)
        if p == "USDJPY" and len(fr[p]["none"]):
            base = fr[p]["none"]
            X1 = A.features(bk, G, W, base, hol, mkt_v1, cot)
            X2 = L.more_features(bk, G, W, base, hol, data, us_hol, fomc)
            X[p] = X1.join(X2, how="outer")
    iu, ie = fr["USDJPY"]["none"].index, fr["EURJPY"]["none"].index
    B = {k: A.basket(fr["USDJPY"][k], fr["EURJPY"][k], iu, ie) for k in fr["USDJPY"]}
    B = {k: v[[keep_night(d) for d in v.index]] for k, v in B.items()}       # the nights under test
    if not len(B["none"]):
        json.dump({"nights": 0, "last_bar": str(last_bar), "errors": errors}, open(os.path.join(a.out, "summary.json"), "w"))
        print("no nights yet; last bar", last_bar)
        return
    Xu = X["USDJPY"].reindex(B["none"].index)
    s1, s2 = FZ.score(Xu, F1), FZ.score(Xu, F2)
    N = pd.DataFrame({"net_none_bp": B["none"].net * 1e4, "net_s80_bp": B["stop 80 bp"].net * 1e4,
                      "net_s100_bp": B["stop 100 bp"].net * 1e4, "mae_none_bp": B["none"].mae * 1e4,
                      "mae_s80_bp": B["stop 80 bp"].mae * 1e4, "mae_s100_bp": B["stop 100 bp"].mae * 1e4,
                      "net_usdjpy_bp": fr["USDJPY"]["none"].net.reindex(B["none"].index) * 1e4,
                      "net_eurjpy_bp": fr["EURJPY"]["none"].net.reindex(B["none"].index) * 1e4,
                      "p_v1": s1.p_win, "keep_v1": s1.keep, "p_v2": s2.p_win, "keep_v2": s2.keep})
    N.to_csv(os.path.join(a.out, "nights.csv"))
    Xu.to_csv(os.path.join(a.out, "features_usdjpy.csv"))
    S = tests(N)
    S.update({"mode": f"dry {a.dry}" if a.dry else "forward", "last_bar": str(last_bar), "errors": errors,
              "frozen": {"v1": F1["version"], "v2": F2["version"]}})
    if a.dry:
        S["dry_check"] = dry_compare(N, Xu, F1, F2)
    json.dump(S, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "report.md"), "w").write(report_md(S, N, S["mode"]))
    print(json.dumps(S, indent=1, default=float)[:6000])


def dry_compare(N, Xu, F1, F2):
    """Compare a development year with the development files."""
    out = {}
    T = pd.read_csv(os.path.join(ROOT, "docs", "gotobi_losses", "trades_basket_none.csv"), index_col=0, parse_dates=True)
    t = (T.net.reindex(N.index) * 1e4 - N.net_none_bp).abs()
    out["trades_max_abs_diff_bp"], out["trades_compared"] = float(t.max()), int(t.notna().sum())
    T80 = pd.read_csv(os.path.join(ROOT, "docs", "gotobi_losses", "trades_basket_stop_80_bp.csv"), index_col=0, parse_dates=True)
    out["stop80_max_abs_diff_bp"] = float((T80.net.reindex(N.index) * 1e4 - N.net_s80_bp).abs().max())
    for tag, path, Fz in (("v1", os.path.join(ROOT, "docs", "gotobi_accuracy", "features_basket.csv"), F1),
                          ("v2", os.path.join(ROOT, "docs", "gotobi_losses", "features_basket.csv"), F2)):
        D = pd.read_csv(path, index_col=0, parse_dates=True).reindex(N.index)
        cols = Fz["features"]
        diffs = {c: float((D[c] - Xu[c]).abs().max()) for c in cols if c in D and c in Xu}
        big = {c: v for c, v in diffs.items() if not (v <= 1e-6 * max(1.0, float(D[c].abs().max())))}
        sd, sx = FZ.score(D, Fz), FZ.score(Xu, Fz)
        out[f"{tag}_features_differing"] = {c: round(v, 6) for c, v in sorted(big.items(), key=lambda x: -x[1])}
        out[f"{tag}_p_max_abs_diff"] = float((sd.p_win - sx.p_win).abs().max())
        out[f"{tag}_same_decision_share"] = float((sd.keep == sx.keep).mean())
    return out


if __name__ == "__main__":
    main()
