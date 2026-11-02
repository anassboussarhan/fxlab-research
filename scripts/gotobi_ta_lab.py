"""Technical indicators and the Gotobi night: do the usual chart readings at entry (RSI, MACD, stochastics,
Bollinger bands, ATR, ADX, CCI, Williams %R, moving averages and crosses, Donchian, Ichimoku, Parabolic SAR, rate of
change, Keltner, pivot points, candle patterns, Heikin-Ashi) say anything about the night's result? The trade: long
USD/JPY and EUR/JPY 23:00 Tokyo the evening before -> 09:53 Tokyo on Gotobi days, no stop. Development data only
(to 2025-12-31); 2026 is not loaded. 2007-2018 train, 2019-2025 test.

BARS (mid prices from M1, every bar ending exactly at the entry time, so each reading is known at entry):
  daily bars cut at 14:00 UTC (23:00 Tokyo), 4-hour bars ending 14:00 UTC, 1-hour bars ending 14:00 UTC.
INDICATORS (standard settings):
  daily: RSI 14 and 2; MACD 12/26/9 histogram / ATR, MACD above signal, above zero; stochastic %K 14 (3) and %D;
    Bollinger 20/2 %B and width / its 120-bar median; ATR 14 / price / its 120-bar median; ADX 14 and +DI - -DI;
    CCI 20; Williams %R 14; distance to SMA 20, 50, 100, 200 in ATR; EMA 12 above EMA 26; SMA 50 above SMA 200;
    Donchian 20 position; Ichimoku (above / in / below the cloud, Tenkan above Kijun); Parabolic SAR state; ROC 10;
    Keltner position (EMA 20, 2 ATR); classic pivot position in ATR, above R1, below S1; last bar's body / range,
    up bar, streak of higher / lower closes, inside bar, outside bar, bullish / bearish engulfing, doji, Heikin-Ashi
    up; and the usual thresholds: RSI 14 > 70 / < 30, RSI 2 > 90 / < 10, stochastic > 80 / < 20, %B > 1 / < 0,
    ADX > 25, CCI > 100 / < -100, Williams > -20 / < -80.
  4-hour and 1-hour: RSI 14, MACD histogram / ATR, %B, stochastic %K, distance to EMA 50 in ATR, ADX; 1-hour RSI 2.
TESTS (as in scripts/gotobi_losses_lab.py): Spearman with the net; losing / big-loss / deep-open-loss rate, top vs
  bottom tercile or 1 vs 0; Benjamini-Hochberg on train; confirmed = q <= 0.10 on train and same sign, p <= 0.10 on
  test. REDUNDANCY: each indicator's largest absolute Spearman correlation with the 80 features of forward-v2.
MODELS (logistic L2 C = 0.1, win target, skip the lowest-rated third; fitted on train, and walk-forward 2012-2025):
  indicators only; forward-v2's 80 features; the 80 plus the indicators.
  forward-v3 rule, fixed now: the 80-plus-indicators model is frozen as forward-v3 (a third, tertiary filter) only if
  its walk-forward kept-minus-skipped gap is at least 1.0 bp above the 80-feature model's in this run AND both its
  2019-2025 AUC and its walk-forward AUC are above the 80-feature model's.
Usage: python scripts/gotobi_ta_lab.py --out DIR"""
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
import gotobi_accuracy_lab as A  # noqa: E402
import gotobi_losses_lab as L  # noqa: E402
import gotobi_risk_lab as R  # noqa: E402
import untested_intraday_lab as U  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = ("USDJPY", "EURJPY")
TRAIN_END = pd.Timestamp("2018-12-31")


# ------------------------------------------------------------------------------------------------ bars
def bars(bk, rule, offset):
    m = bk.m
    mid = pd.DataFrame({"o": 0.5 * (m.bid_o + m.ask_o), "h": 0.5 * (m.bid_h + m.ask_h),
                        "l": 0.5 * (m.bid_l + m.ask_l), "c": 0.5 * (m.bid_c + m.ask_c)}, index=m.index)
    g = mid.resample("24h" if rule == "1D" else rule, origin="epoch", offset=offset, closed="left", label="right")
    b = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "n": g.c.count()})
    b = b[b.n > 0]
    if rule == "1D":
        # the bar ending Saturday 14:00 holds only Friday's last hours: join it to the bar ending Monday 14:00
        wd = b.index.weekday
        lab = b.index + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
        b = b.groupby(lab).agg({"o": "first", "h": "max", "l": "min", "c": "last", "n": "sum"})
    need = {"1D": 120, "4h": 30, "1h": 10}[rule]
    return b[b.n >= need].drop(columns="n")


# ------------------------------------------------------------------------------------------------ indicators
def wilder(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(b, n=14):
    pc = b.c.shift(1)
    tr = pd.concat([b.h - b.l, (b.h - pc).abs(), (b.l - pc).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def adx(b, n=14):
    up, dn = b.h.diff(), -b.l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=b.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=b.index)
    a = atr(b, n)
    pdi, ndi = 100 * wilder(pdm, n) / a, 100 * wilder(ndm, n) / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return wilder(dx, n), pdi - ndi


def stoch(b, n=14, k=3, d=3):
    hh, ll = b.h.rolling(n).max(), b.l.rolling(n).min()
    fast = 100 * (b.c - ll) / (hh - ll).replace(0, np.nan)
    K = fast.rolling(k).mean()
    return K, K.rolling(d).mean()


def macd(c, a_=14):
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    s = m.ewm(span=9, adjust=False).mean()
    return m, s, m - s


def psar(b, step=0.02, mx=0.2):
    h, l = b.h.to_numpy(), b.l.to_numpy()
    n = len(h)
    st_ = np.ones(n)
    if n < 3:
        return pd.Series(st_, index=b.index)
    up, af, ep, sar = True, step, h[0], l[0]
    for t in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[t - 1], l[t - 2] if t > 1 else l[t - 1])
            if l[t] < sar:
                up, sar, ep, af = False, ep, l[t], step
            elif h[t] > ep:
                ep, af = h[t], min(af + step, mx)
        else:
            sar = max(sar, h[t - 1], h[t - 2] if t > 1 else h[t - 1])
            if h[t] > sar:
                up, sar, ep, af = True, ep, h[t], step
            elif l[t] < ep:
                ep, af = l[t], min(af + step, mx)
        st_[t] = 1.0 if up else -1.0
    return pd.Series(st_, index=b.index)


def streak(c):
    s = np.sign(c.diff()).fillna(0).to_numpy()
    out = np.zeros(len(s))
    for t in range(1, len(s)):
        out[t] = (out[t - 1] + s[t]) if (s[t] != 0 and np.sign(out[t - 1]) in (0, s[t])) else s[t]
    return pd.Series(out, index=c.index)


def daily_set(b):
    c, o, h, l = b.c, b.o, b.h, b.l
    A14 = atr(b)
    f = pd.DataFrame(index=b.index)
    f["rsi14"], f["rsi2"] = rsi(c, 14), rsi(c, 2)
    m, s, hist = macd(c)
    f["macd_hist_atr"], f["macd_above_signal"], f["macd_above_zero"] = hist / A14, (m > s).astype(float), (m > 0).astype(float)
    f["stoch_k"], f["stoch_d"] = stoch(b)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    bw = 4 * sd / mid
    f["bb_width_vs_median"] = bw / bw.rolling(120, min_periods=60).median()
    ar = A14 / c
    f["atr_vs_median"] = ar / ar.rolling(120, min_periods=60).median()
    f["adx14"], f["di_diff"] = adx(b)
    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad)
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["willr14"] = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    for n in (20, 50, 100, 200):
        f[f"dist_sma{n}_atr"] = (c - c.rolling(n).mean()) / A14
    f["ema12_above_ema26"] = (c.ewm(span=12, adjust=False).mean() > c.ewm(span=26, adjust=False).mean()).astype(float)
    f["sma50_above_sma200"] = (c.rolling(50).mean() > c.rolling(200).mean()).astype(float).where(c.rolling(200).count() == 200)
    h20, l20 = h.rolling(20).max(), l.rolling(20).min()
    f["donchian20_pos"] = (c - l20) / (h20 - l20).replace(0, np.nan)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa, sb = ((tenkan + kijun) / 2).shift(26), ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top, bot = pd.concat([sa, sb], axis=1).max(axis=1), pd.concat([sa, sb], axis=1).min(axis=1)
    f["ichimoku_cloud"] = np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0))
    f["ichimoku_cloud"] = f["ichimoku_cloud"].where(sb.notna())
    f["tenkan_above_kijun"] = (tenkan > kijun).astype(float)
    f["psar_long"] = psar(b)
    f["roc10"] = c / c.shift(10) - 1
    f["keltner_pos"] = (c - c.ewm(span=20, adjust=False).mean()) / (2 * A14)
    P = ((h + l + c) / 3).shift(1)
    R1, S1 = 2 * P - l.shift(1), 2 * P - h.shift(1)
    f["pivot_pos_atr"], f["above_r1"], f["below_s1"] = (c - P) / A14, (c > R1).astype(float), (c < S1).astype(float)
    rngb = (h - l).replace(0, np.nan)
    f["bar_body"] = (c - o) / rngb
    f["bar_up"] = (c > o).astype(float)
    f["close_streak"] = streak(c)
    f["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
    f["outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)
    po, pc = o.shift(1), c.shift(1)
    f["engulf_bull"] = ((c > o) & (pc < po) & (c >= po) & (o <= pc)).astype(float)
    f["engulf_bear"] = ((c < o) & (pc > po) & (c <= po) & (o >= pc)).astype(float)
    f["doji"] = ((c - o).abs() < 0.1 * (h - l)).astype(float)
    hac = (o + h + l + c) / 4
    hao = hac.copy()
    for t in range(1, len(hao)):
        hao.iloc[t] = (hao.iloc[t - 1] + hac.iloc[t - 1]) / 2
    f["heikin_ashi_up"] = (hac > hao).astype(float)
    f["rsi14_gt70"], f["rsi14_lt30"] = (f.rsi14 > 70).astype(float), (f.rsi14 < 30).astype(float)
    f["rsi2_gt90"], f["rsi2_lt10"] = (f.rsi2 > 90).astype(float), (f.rsi2 < 10).astype(float)
    f["stoch_gt80"], f["stoch_lt20"] = (f.stoch_k > 80).astype(float), (f.stoch_k < 20).astype(float)
    f["pctb_gt1"], f["pctb_lt0"] = (f.bb_pctb > 1).astype(float), (f.bb_pctb < 0).astype(float)
    f["adx_gt25"] = (f.adx14 > 25).astype(float)
    f["cci_gt100"], f["cci_lt_m100"] = (f.cci20 > 100).astype(float), (f.cci20 < -100).astype(float)
    f["willr_gt_m20"], f["willr_lt_m80"] = (f.willr14 > -20).astype(float), (f.willr14 < -80).astype(float)
    return f.iloc[200:]                                                  # warm-up


def intraday_set(b, tag):
    c = b.c
    A14 = atr(b)
    f = pd.DataFrame(index=b.index)
    f[f"{tag}_rsi14"] = rsi(c, 14)
    f[f"{tag}_macd_hist_atr"] = macd(c)[2] / A14
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f[f"{tag}_bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    f[f"{tag}_stoch_k"] = stoch(b)[0]
    f[f"{tag}_dist_ema50_atr"] = (c - c.ewm(span=50, adjust=False).mean()) / A14
    f[f"{tag}_adx14"] = adx(b)[0]
    if tag == "h1":
        f["h1_rsi2"] = rsi(c, 2)
    return f.iloc[60:]


def ta_features(bk, nights):
    D = daily_set(bars(bk, "1D", "14h")).add_prefix("d_")
    H4 = intraday_set(bars(bk, "4h", "2h"), "h4")
    H1 = intraday_set(bars(bk, "1h", "0h"), "h1")
    rows = {}
    for d in nights:
        i, j = F.window(bk, d)
        if i < 0:
            continue
        t = bk.idx[i].floor("h")                                         # the entry hour, 14:00 UTC
        r = {}
        for T in (D, H4, H1):
            k = T.index.searchsorted(t, side="right") - 1                # the last bar ending at or before entry
            if k >= 0 and (t - T.index[k]) <= pd.Timedelta(hours=1):
                r.update(T.iloc[k].to_dict())
        rows[d] = r
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().add_prefix("ta_")


# ------------------------------------------------------------------------------------------------ models
def compare_models(series, sets, y, rows, wf_rows, coefs):
    from scipy import stats as st
    from sklearn.metrics import roc_auc_score
    tr, te = np.asarray(y.index <= TRAIN_END), np.asarray(y.index > TRAIN_END)
    target = (y > 0).astype(float)
    for name, X in sets.items():
        X = X.reindex(y.index)
        p_fit, p_te, coef = L.fit_lr(X[tr], target[tr], X[te])
        coefs[f"{series} | {name}"] = coef.sort_values().round(3).to_dict()
        skip = (p_te <= np.quantile(p_fit, 1 / 3)).to_numpy()
        yt = y[te]
        r = {"series": series, "model": name, "features": len(coef), "test_auc": roc_auc_score(target[te], p_te)}
        r.update({f"kept_{k}": v for k, v in L.sstat(yt[~skip], len(yt)).items()})
        r.update({f"skipped_{k}": v for k, v in L.sstat(yt[skip], len(yt)).items()})
        kept, skipped, ps, tg = [], [], [], []
        for Y in range(2012, 2026):
            f = np.asarray(y.index < pd.Timestamp(f"{Y}-01-01"))
            pm = np.asarray(y.index.year == Y)
            if pm.sum() == 0:
                continue
            pf, pp, _ = L.fit_lr(X[f], target[f], X[pm])
            sk = (pp <= np.quantile(pf, 1 / 3)).to_numpy()
            yy = y[pm]
            kept.append(yy[~sk])
            skipped.append(yy[sk])
            ps.append(pp)
            tg.append(target[pm])
            wf_rows.append({"series": series, "model": name, "year": Y, "kept_bp": yy[~sk].mean() * 1e4,
                            "skipped_bp": yy[sk].mean() * 1e4})
        K, S = pd.concat(kept), pd.concat(skipped)
        r.update({"wf_kept_bp": K.mean() * 1e4, "wf_skipped_bp": S.mean() * 1e4, "wf_gap_bp": (K.mean() - S.mean()) * 1e4,
                  "wf_gap_t": float(st.ttest_ind(K, S, equal_var=False).statistic),
                  "wf_auc": roc_auc_score(pd.concat(tg), pd.concat(ps)), "wf_kept_win": float((K > 0).mean()),
                  "wf_years_kept_above": int(sum(k.mean() > s.mean() for k, s in zip(kept, skipped)))})
        rows.append(r)


def redundancy(T, X80):
    out = []
    J = T.join(X80, how="inner")
    C = J.corr(method="spearman")
    for c in T.columns:
        if c not in C:
            continue
        s = C.loc[c, X80.columns].abs().dropna()
        t = C.loc[c, T.columns].drop(c).abs().dropna()
        out.append({"indicator": c, "max_abs_corr_with_80": s.max() if len(s) else np.nan,
                    "most_correlated_feature": s.idxmax() if len(s) else "",
                    "max_abs_corr_other_indicator": t.max() if len(t) else np.nan,
                    "most_correlated_indicator": t.idxmax() if len(t) else ""})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    errors, info = {}, {}
    X80 = pd.read_csv(os.path.join(ROOT, "docs", "gotobi_losses", "features_basket.csv"), index_col=0, parse_dates=True)
    X80 = X80[[c for c in X80.columns if not c.startswith("night_")]]
    v2 = json.load(open(os.path.join(ROOT, "results", "frozen_forward_v2.json")))["features"]
    X80 = X80[[c for c in v2 if c in X80.columns]]
    frames, TA, Wr = {}, {}, {}
    for p in PAIRS:
        bk = U.book(p)
        if bk is None:
            errors[p] = "no data"
            continue
        bk._hl()
        G, _, _ = F.calendar(bk)
        W = R.window_units(bk, G)
        out, wr = {}, {}
        for d in G:
            w = float(W.get(d, np.nan))
            i, j = F.window(bk, d)
            if not np.isfinite(w) or w <= 0 or i < 0:
                continue
            if not np.isfinite([bk.bo[i], bk.ao[i], bk.bo[j], bk.ao[j]]).all():
                continue
            out[d] = A.sim(bk, i, j, w, {}, A.hours_map(bk, d, i, j))
            wr[d] = w / bk.mo[i]
        f = pd.DataFrame.from_dict(out, orient="index").sort_index()
        f.index = pd.DatetimeIndex(f.index)
        frames[p], Wr[p] = f, pd.Series(wr).sort_index()
        try:
            TA[p] = ta_features(bk, list(f.index))
        except Exception:                                                   # noqa: BLE001
            import traceback
            errors[f"ta {p}"] = traceback.format_exc()[-1500:]
        info[p] = {"nights": int(len(f)), "indicators": int(TA[p].shape[1]) if p in TA else 0}
        print(p, info[p], flush=True)
    if all(p in frames for p in PAIRS):
        iu, ie = frames["USDJPY"].index, frames["EURJPY"].index
        frames["basket"] = A.basket(frames["USDJPY"], frames["EURJPY"], iu, ie)
        Wr["basket"] = pd.concat([Wr["USDJPY"], Wr["EURJPY"]], axis=1).mean(axis=1)
        if "USDJPY" in TA:
            TA["basket"] = TA["USDJPY"]
    tests, mrows, wf, coefs = [], [], [], {}
    for s in ("basket", "USDJPY", "EURJPY"):
        if s not in TA:
            continue
        base = frames[s]
        try:
            L.feature_tests(s, TA[s], base.net, L.outcomes(base, Wr[s]), tests)
            if s in ("basket", "USDJPY"):
                y = base.net.reindex(X80.index).dropna()
                sets = {"indicators only": TA[s], "forward-v2's 80 features": X80,
                        "80 features + indicators": X80.join(TA[s], how="left")}
                compare_models(s, sets, y, mrows, wf, coefs)
        except Exception:                                                   # noqa: BLE001
            import traceback
            errors[f"analysis {s}"] = traceback.format_exc()[-1500:]
        TA[s].to_csv(os.path.join(a.out, f"ta_{s}.csv"))
    if "USDJPY" in TA:
        redundancy(TA["USDJPY"], X80).to_csv(os.path.join(a.out, "redundancy.csv"), index=False)
    T, M = pd.DataFrame(tests), pd.DataFrame(mrows)
    T.to_csv(os.path.join(a.out, "ta_tests.csv"), index=False)
    M.to_csv(os.path.join(a.out, "models.csv"), index=False)
    pd.DataFrame(wf).to_csv(os.path.join(a.out, "models_walk_forward_years.csv"), index=False)
    v3 = {}
    if len(M):
        b = M[M.series == "basket"].set_index("model")
        if {"forward-v2's 80 features", "80 features + indicators"} <= set(b.index):
            x, ref = b.loc["80 features + indicators"], b.loc["forward-v2's 80 features"]
            v3 = {"gap_bp": x.wf_gap_bp, "ref_gap_bp": ref.wf_gap_bp, "test_auc": x.test_auc, "ref_test_auc": ref.test_auc,
                  "wf_auc": x.wf_auc, "ref_wf_auc": ref.wf_auc,
                  "qualifies_for_forward_v3": bool(x.wf_gap_bp >= ref.wf_gap_bp + 1.0 and x.test_auc > ref.test_auc
                                                   and x.wf_auc > ref.wf_auc)}
    json.dump({"info": info, "errors": errors, "coefficients": coefs, "forward_v3_rule": v3},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    if len(T):
        print(T[T.confirmed.astype(bool)].round(3).to_string())
    print(M.round(3).T.to_string() if len(M) else "no models")
    print(json.dumps({"errors": errors, "forward_v3_rule": v3}, indent=1, default=str))


if __name__ == "__main__":
    main()
