"""The full technical-analysis catalog on markets the FX sweep never covered: daily bars of US equity ETFs, sector
and country ETFs, European and Asian stock indices, bond ETFs, commodity ETFs and crypto (Yahoo, free), 1993-2025.
Development data only: every series is cut at 2025-12-31 (2026 stays locked).

WHAT RUNS (fixed before any result):
  * Every catalog strategy family whose timeframe scope includes D1 and that works on one instrument: the FX
    catalog's single-instrument families (51 with the optional astronomy package installed for the lunar and
    Mercury rules) plus the 11 daily families of fxlab/strategies/daily.py (S.DAILY.*). Left out: model-driven
    families (S.ML.*: the ML entry is tested separately in scripts/ml_entry_lab.py), quote/tick microstructure
    families (S.MS.*: daily Yahoo bars have no quotes or ticks) and FX-panel families (S.XA.*, S.TREND.XS_MOMENTUM).
  * The project's own walk-forward (fxlab.orchestration.runners.strategy_wf): per family and instrument, the default
    parameters plus 11 random draws from the family's search space; each 6-month test block trades the candidate
    with the best Sharpe on the previous 3 years; nothing from a test block is used to choose it.
  * Prices are excess-return prices: dividend-adjusted OHLC divided by a cash index that accrues the 3-month
    T-bill (USD; ^IRX of the previous day) or the point-in-time 3-month rate (EUR for DAX, CAC, Euro Stoxx; JPY for
    the Nikkei). A long position therefore earns the asset's return over cash and a short pays it, as on a CFD.
  * Costs, one way, through the engine's quoted-spread model (the synthetic spread is set so that half-spread +
    25% slippage + 0.35 bp commission equals the figure): SPY, QQQ 1 bp; IWM, DIA 1.5 bp; other ETFs and indices
    2 bp; HYG, SLV, USO, DBC 3 bp; UNG, DBA 4 bp; BTC 10 bp; ETH 12 bp. Financing mark-up on any open position
    (long or short) per night: 2.5% a year (crypto 5%), on top of the cash rate already in the excess-return price.
  * Execution: decision at the close, fill at the next open (engine timing contract). Indices whose Yahoo opens
    equal the previous close on more than 10% of days are dropped (the FTSE 100 is not in the list for that reason).

PRE-REGISTERED TESTS (one-sided):
  P1 (primary, one per family): the equal-weight portfolio of a family's out-of-sample returns across the universe
     has a positive alpha against the equal-weight buy-and-hold of the same instruments on the same days (OLS on
     daily excess returns, Newey-West 5 lags). Benjamini-Hochberg over the families.
  P2 (secondary): alpha of each family on each instrument against holding that instrument; BH over all pairs.
  Also reported: raw out-of-sample Sharpe and t, Sharpe at 2x costs, parameter stability, results by asset class.
Usage: python scripts/cross_market_ta_lab.py --stage data --out DIR
       python scripts/cross_market_ta_lab.py --stage run --data DIR --symbols SPY,QQQ --out DIR
       python scripts/cross_market_ta_lab.py --stage combine --runs DIR1,DIR2,... --out DIR"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

END = pd.Timestamp("2025-12-31")
# symbol: (asset class, one-way cost in bp, rate currency, first date)
UNIVERSE = {
    "SPY": ("equity", 1.0, "USD", "1993-01-29"), "QQQ": ("equity", 1.0, "USD", "1999-03-10"),
    "IWM": ("equity", 1.5, "USD", "2000-05-26"), "DIA": ("equity", 1.5, "USD", "1998-01-20"),
    "XLK": ("equity", 2.0, "USD", "1998-12-22"), "XLF": ("equity", 2.0, "USD", "1998-12-22"),
    "XLE": ("equity", 2.0, "USD", "1998-12-22"), "XLV": ("equity", 2.0, "USD", "1998-12-22"),
    "XLI": ("equity", 2.0, "USD", "1998-12-22"), "XLY": ("equity", 2.0, "USD", "1998-12-22"),
    "XLP": ("equity", 2.0, "USD", "1998-12-22"), "XLU": ("equity", 2.0, "USD", "1998-12-22"),
    "XLB": ("equity", 2.0, "USD", "1998-12-22"), "EEM": ("equity", 2.0, "USD", "2003-04-14"),
    "EFA": ("equity", 2.0, "USD", "2001-08-27"), "EWJ": ("equity", 2.0, "USD", "1996-03-18"),
    "^GDAXI": ("index", 2.0, "EUR", "2006-02-01"), "^FCHI": ("index", 2.0, "EUR", "2006-02-01"),
    "^STOXX50E": ("index", 2.0, "EUR", "2007-04-01"), "^N225": ("index", 2.0, "JPY", "2006-02-01"),
    "^HSI": ("index", 2.0, "USD", "1993-01-04"),
    "TLT": ("bond", 2.0, "USD", "2002-07-30"), "IEF": ("bond", 2.0, "USD", "2002-07-30"),
    "LQD": ("bond", 2.0, "USD", "2002-07-30"), "HYG": ("bond", 3.0, "USD", "2007-04-11"),
    "GLD": ("commodity", 2.0, "USD", "2004-11-18"), "SLV": ("commodity", 3.0, "USD", "2006-04-28"),
    "USO": ("commodity", 3.0, "USD", "2006-04-10"), "UNG": ("commodity", 4.0, "USD", "2007-04-18"),
    "DBA": ("commodity", 4.0, "USD", "2007-01-05"), "DBC": ("commodity", 3.0, "USD", "2006-02-03"),
    "BTC-USD": ("crypto", 10.0, "USD", "2014-09-17"), "ETH-USD": ("crypto", 12.0, "USD", "2017-11-09"),
}
MARKUP = {"crypto": 0.05}
MARKUP_DEFAULT = 0.025
N_TRIALS = 12
EXCLUDE_PREFIX = ("S.ML.", "S.MS.", "S.XA.")
EXCLUDE = {"S.TREND.XS_MOMENTUM"}


def safe(s):
    return s.replace("^", "I_").replace("=", "_").replace("-", "_")


# ----------------------------------------------------------------------------------------------- data
def yahoo(symbols, start="1990-01-01"):
    import yfinance as yf
    out = {}
    for s in symbols:
        h = pd.DataFrame()
        for _ in range(4):
            try:
                h = yf.download(s, start=start, end="2026-01-01", progress=False, auto_adjust=False)
                if h is not None and len(h):
                    break
            except Exception:                                          # noqa: BLE001
                time.sleep(5)
        if h is None or h.empty:
            print("no data", s, flush=True)
            continue
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        h.index = pd.to_datetime(h.index)
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        h = h[h.index <= END]
        out[s] = h[[c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if c in h.columns]].astype(float)
        print(s, len(h), h.index[0].date(), h.index[-1].date(), flush=True)
    return out


def cash_index(dates: pd.DatetimeIndex, ccy: str, irx: pd.Series) -> pd.Series:
    """Cash account accruing the previous day's 3-month rate, actual/360."""
    if ccy == "USD":
        r = (irx.reindex(dates.union(irx.index)).ffill().reindex(dates) / 100.0).shift(1)
    else:
        from fxlab.data import rates as R
        k = R.known_at(pd.DatetimeIndex(dates).tz_localize("UTC"))
        r = pd.Series(k[ccy].to_numpy(), index=dates).shift(1)
    days = pd.Series(dates, index=dates).diff().dt.days.fillna(0).to_numpy()
    acc = 1.0 + r.fillna(0.0).to_numpy() * days / 360.0
    return pd.Series(np.cumprod(acc), index=dates)


def build_bars(h: pd.DataFrame, cash: pd.Series, cost_bp: float) -> pd.DataFrame:
    """Engine-format daily bars of the excess-return price, with a synthetic quoted spread for the cost model."""
    adj = (h["Adj Close"] / h["Close"]).where(h["Close"] > 0) if "Adj Close" in h else 1.0
    f = adj / cash.reindex(h.index)
    o, hi, lo, c = (h[k] * f for k in ("Open", "High", "Low", "Close"))
    hi = np.maximum(hi, np.maximum(o, c))
    lo = np.minimum(lo, np.minimum(o, c))
    d = pd.DataFrame({"o": o, "h": hi, "l": lo, "c": c}).dropna()
    d = d[(d > 0).all(1)]
    s = max(cost_bp - 0.35, 0.05) / 0.75 * 1e-4                  # full relative spread
    idx = pd.DatetimeIndex(d.index).tz_localize("UTC")
    b = pd.DataFrame(index=idx)
    for k in "ohlc":
        v = d[k].to_numpy()
        b[f"bid_{k}"] = v * (1 - s / 2)
        b[f"ask_{k}"] = v * (1 + s / 2)
    for k, name in zip("ohlc", ("open", "high", "low", "close")):
        b[name] = d[k].to_numpy()
    vol = h["Volume"].reindex(d.index).fillna(0).to_numpy() if "Volume" in h else np.zeros(len(d))
    b["spread"] = b["spread_mean"] = b["spread_max"] = b["spread_min"] = (b["ask_c"] - b["bid_c"]).to_numpy()
    b["volume"] = b["n_ticks"] = np.where(vol > 0, vol, 1.0)
    b["bid_liq"] = b["ask_liq"] = b["volume"] / 2
    b["up_ticks"] = b["down_ticks"] = b["volume"] / 2
    b["rv_intrabar"] = (np.log(b["high"] / b["low"]) ** 2 / (4 * math.log(2))).to_numpy()
    b["n_m1"] = 1440
    return b


def stage_data(a):
    os.makedirs(a.out, exist_ok=True)
    raw = yahoo(list(UNIVERSE) + ["^VIX", "^IRX", "^TNX"])
    irx = raw["^IRX"]["Close"]
    meta = {}
    for s, (cls, bp, ccy, first) in UNIVERSE.items():
        if s not in raw:
            meta[s] = {"status": "no data"}
            continue
        h = raw[s][raw[s].index >= first]
        same_open = float((h["Open"] == h["Close"].shift(1)).mean())
        if same_open > 0.10:
            meta[s] = {"status": f"dropped: open equals the previous close on {same_open:.0%} of days"}
            continue
        cash = cash_index(pd.DatetimeIndex(h.index), ccy, irx)
        b = build_bars(h, cash, bp)
        b.to_parquet(os.path.join(a.out, f"bars_{safe(s)}.parquet"))
        meta[s] = {"status": "ok", "class": cls, "cost_bp": bp, "ccy": ccy, "markup": MARKUP.get(cls, MARKUP_DEFAULT),
                   "bars": len(b), "first": str(b.index[0].date()), "last": str(b.index[-1].date()),
                   "same_open_share": same_open}
        print(s, meta[s], flush=True)
    vix = raw["^VIX"]["Close"]
    vix.index = pd.DatetimeIndex(vix.index).tz_localize("UTC")
    vix.rename("VIX").to_frame().to_parquet(os.path.join(a.out, "vix.parquet"))
    macro = pd.DataFrame({"VIX": raw["^VIX"]["Close"], "IRX": raw["^IRX"]["Close"], "TNX": raw["^TNX"]["Close"]})
    macro.to_parquet(os.path.join(a.out, "macro.parquet"))         # for scripts/ml_entry_lab.py (dates, no tz)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------- run
def families():
    import fxlab.features, fxlab.models, fxlab.strategies  # noqa: F401
    from fxlab.registry import by_kind, requirements_met
    order = ["M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1"]
    out = []
    for e in sorted(by_kind("strategy"), key=lambda e: e.id):
        lo, hi = getattr(e, "frequencies", "M1..D1").split("..")
        if order.index(hi) < order.index("D1") or e.id.startswith(EXCLUDE_PREFIX) or e.id in EXCLUDE:
            continue
        if e.multi_asset and e.id != "S.DAILY.VIX_SPIKE":
            continue
        if requirements_met(e)[0]:
            out.append(e.id)
    return out


class Patch:
    """Route the project's walk-forward runner to the daily bars of this lab (bars, panel, instrument, financing)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.vix = pd.read_parquet(os.path.join(data_dir, "vix.parquet"))
        self.bars = {}
        self.fin = {}

    def load(self, sym):
        if sym not in self.bars:
            b = pd.read_parquet(os.path.join(self.data_dir, f"bars_{safe(sym)}.parquet"))
            self.bars[sym] = b
            mid_o = 0.5 * (b["bid_o"] + b["ask_o"])
            pip = float(np.median(mid_o)) * 1e-4                           # "1 pip" = 1 bp of the median price
            from fxlab.execution.costs import BASE
            nights = np.r_[pd.Series(b.index).diff().dt.days.to_numpy()[1:], 0.0]   # held from open j to open j+1
            markup = self.meta[sym]["markup"]
            # the engine charges financing = roll x swap_pips x pip / price per bar; this roll makes it
            # nights x mark-up / 360 of the position's notional (and scales with the stressed cost models)
            roll = nights * markup / 360.0 * mid_o.to_numpy() / (BASE.swap_pips_per_night * pip)
            self.fin[sym] = (pip, roll, b.index)
        return self.bars[sym]

    def install(self, sym):
        import fxlab.evaluate as EV
        import fxlab.orchestration.runners as RN
        from fxlab.data import instruments as INS
        b = self.load(sym)
        pip, roll, idx = self.fin[sym]
        INS.INSTRUMENTS[sym.upper()] = INS.Instrument(sym.upper(), sym.upper(), "USD", pip, 1.0,
                                                      self.meta[sym]["class"])
        RN.cached_bars = lambda s, tf, start=None, end=None: self.bars[{k.upper(): k for k in self.bars}[s.upper()]]

        def panel(symbols, tf, index, start=None, end=None):
            return self.vix.reindex(self.vix.index.union(index)).ffill().reindex(index)
        RN.load_panel_closes = panel

        def rollovers(index):
            if len(index) == len(idx) and index[0] == idx[0] and index[-1] == idx[-1]:
                return roll.astype(float)
            pos = idx.get_indexer(index)
            return np.where(pos >= 0, roll[np.clip(pos, 0, None)], 0.0).astype(float)
        EV.rollover_counts = rollovers
        return b


def nw_t(y, x=None, lags=5):
    """OLS of y on [1, x] with Newey-West standard errors; returns (alpha, t_alpha, beta)."""
    y = np.asarray(y, float)
    X = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ S @ XtX
    t = beta[0] / math.sqrt(V[0, 0]) if V[0, 0] > 0 else np.nan
    return float(beta[0]), float(t), (float(beta[1]) if x is not None else np.nan)


def bh_returns(b):
    mid_o = 0.5 * (b["bid_o"] + b["ask_o"])
    return (mid_o.shift(-1) / mid_o - 1).rename("bh")          # open-to-open, like the engine's held position


def stage_run(a):
    from fxlab.orchestration.runners import strategy_wf
    from scipy import stats as st
    P = Patch(a.data)
    fams = families()
    if a.families:
        fams = [f for f in fams if f in a.families.split(",")]
    os.makedirs(a.out, exist_ok=True)
    print(len(fams), "families", flush=True)
    for sym in a.symbols.split(","):
        if P.meta.get(sym, {}).get("status") != "ok":
            print("skip", sym, P.meta.get(sym), flush=True)
            continue
        b = P.install(sym)
        bh = bh_returns(b)
        rows, series = [], {}
        for fid in fams:
            t0 = time.time()
            cfg = {"kind": "strategy", "target": fid, "symbol": sym.upper(), "tf": "D1", "start": None, "end": None,
                   "n_trials": N_TRIALS, "train_years": 3, "test_months": 6, "seed": 0, "panel": ["VIX"]}
            try:
                res = strategy_wf(cfg, __import__("pathlib").Path(a.out) / "wf" / safe(sym) / fid)
                m = res["metrics"]
                r = pd.read_parquet(res["artifacts"]["oos_returns"])["ret"]
                r = r.groupby(r.index.normalize()).sum()
                x = bh.reindex(r.index).fillna(0.0)
                alpha, ta, beta = nw_t(r.to_numpy(), x.to_numpy())
                _, tr, _ = nw_t(r.to_numpy())
                stress = m.get("stress", {}).get("cost_x2.0", {})
                rows.append({"symbol": sym, "class": P.meta[sym]["class"], "family": fid, "days": int(len(r)),
                             "first": str(r.index[0].date()), "ann_ret": float(r.mean() * 252),
                             "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else np.nan,
                             "t": tr, "alpha_ann": alpha * 252, "alpha_t": ta, "beta": beta,
                             "p_alpha": float(1 - st.norm.cdf(ta)) if np.isfinite(ta) else np.nan,
                             "bh_sharpe": float(x.mean() / x.std() * math.sqrt(252)) if x.std() > 0 else np.nan,
                             "trades": m.get("trades"), "sharpe_cost_x2": stress.get("sharpe") if isinstance(stress, dict) else None,
                             "param_stability": m.get("param_stability"), "seconds": round(time.time() - t0, 1)})
                series[fid] = r
            except Exception as ex:                                       # noqa: BLE001
                rows.append({"symbol": sym, "class": P.meta[sym]["class"], "family": fid,
                             "error": repr(ex)[:300], "trace": traceback.format_exc()[-800:]})
            print(sym, fid, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                             if k in ("sharpe", "alpha_t", "trades", "error", "seconds")}, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"rows_{safe(sym)}.csv"), index=False)
        if series:
            S = pd.DataFrame(series)
            S["__bh__"] = bh.reindex(S.index)
            S.to_parquet(os.path.join(a.out, f"oos_{safe(sym)}.parquet"))
        import shutil
        shutil.rmtree(os.path.join(a.out, "wf", safe(sym)), ignore_errors=True)


# ----------------------------------------------------------------------------------------------- combine
def bh_q(p):
    p = np.asarray(p, float)
    n = len(p)
    q = np.full(n, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    qq = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pv[i] * m / rank)
        qq[i] = prev
    q[ok] = qq
    return q


def stage_combine(a):
    from scipy import stats as st
    os.makedirs(a.out, exist_ok=True)
    rows, oos = [], {}
    for d in a.runs.split(","):
        for f in glob.glob(os.path.join(d, "**", "rows_*.csv"), recursive=True):
            rows.append(pd.read_csv(f))
        for f in glob.glob(os.path.join(d, "**", "oos_*.parquet"), recursive=True):
            oos[os.path.basename(f)[4:-8]] = pd.read_parquet(f)
    R = pd.concat(rows, ignore_index=True)
    ok = R[R.get("error").isna()] if "error" in R else R
    ok = ok.copy()
    ok["q_bh_all_pairs"] = bh_q(ok.p_alpha.to_numpy())
    ok.to_csv(os.path.join(a.out, "pairs.csv"), index=False)
    if "error" in R:
        R[R.error.notna()].to_csv(os.path.join(a.out, "errors.csv"), index=False)
    fam_rows = []
    for fid in sorted(ok.family.unique()):
        cols, bhs = [], []
        for sym, S in oos.items():
            if fid in S:
                cols.append(S[fid].rename(sym))
                bhs.append(S["__bh__"].where(S[fid].notna()).rename(sym))
        if not cols:
            continue
        X = pd.concat(cols, axis=1)
        B = pd.concat(bhs, axis=1)
        port, bench = X.mean(axis=1, skipna=True), B.mean(axis=1, skipna=True).fillna(0.0)
        port = port.dropna()
        bench = bench.reindex(port.index).fillna(0.0)
        alpha, ta, beta = nw_t(port.to_numpy(), bench.to_numpy())
        _, tr, _ = nw_t(port.to_numpy())
        by_class = {}
        for cls in sorted(ok["class"].unique()):
            syms = [s for s in X.columns if s in set(ok[ok["class"] == cls].symbol.map(safe))]
            if syms:
                pc = X[syms].mean(axis=1).dropna()
                bc = B[syms].mean(axis=1).reindex(pc.index).fillna(0.0)
                al, tc, _ = nw_t(pc.to_numpy(), bc.to_numpy())
                by_class[cls] = {"alpha_ann": al * 252, "alpha_t": tc}
        sub = ok[ok.family == fid]
        fam_rows.append({"family": fid, "instruments": int(X.shape[1]), "days": int(len(port)),
                         "ann_ret": float(port.mean() * 252),
                         "sharpe": float(port.mean() / port.std() * math.sqrt(252)) if port.std() > 0 else np.nan,
                         "t": tr, "alpha_ann": alpha * 252, "alpha_t": ta, "beta": beta,
                         "p_alpha": float(1 - st.norm.cdf(ta)) if np.isfinite(ta) else np.nan,
                         "median_pair_sharpe": float(sub.sharpe.median()), "share_pairs_alpha_t_gt_2": float((sub.alpha_t > 2).mean()),
                         "median_sharpe_cost_x2": float(pd.to_numeric(sub.sharpe_cost_x2, errors="coerce").median()),
                         **{f"alpha_t_{k}": v["alpha_t"] for k, v in by_class.items()}})
    F = pd.DataFrame(fam_rows)
    F["q_bh"] = bh_q(F.p_alpha.to_numpy())
    F = F.sort_values("p_alpha")
    F.to_csv(os.path.join(a.out, "families.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(F.round(3).to_string())
    top = ok.sort_values("p_alpha").head(30)
    print(top[["symbol", "family", "sharpe", "alpha_ann", "alpha_t", "beta", "q_bh_all_pairs", "sharpe_cost_x2"]].round(3).to_string())
    json.dump({"pairs": int(len(ok)), "errors": int(len(R) - len(ok)), "families": int(len(F)),
               "families_q10": int((F.q_bh <= 0.10).sum()), "pairs_q10": int((ok.q_bh_all_pairs <= 0.10).sum())},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "run", "combine"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--runs", default="")
    ap.add_argument("--families", default="", help="comma list to restrict the run (smoke tests)")
    a = ap.parse_args()
    {"data": stage_data, "run": stage_run, "combine": stage_combine}[a.stage](a)


if __name__ == "__main__":
    main()
