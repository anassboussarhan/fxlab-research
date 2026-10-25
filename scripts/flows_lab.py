"""Flow-driven FX effects from the literature and free option-implied data (run 17), tested in their published
form with rules fixed before the run. Development data only (FX data 2007-2018, EURUSD to 2022; external series
are cut at 2025-12-31).

PRE-REGISTERED PRIMARY TESTS (Benjamini-Hochberg over these 17 in scripts/flows_report.py):
  F1  Gotobi (Ito & Yamada 2017; arXiv 2301.13204): long USDJPY 03:00 -> 09:55 JST (fix) on Gotobi days
  F2  Gotobi: long USDJPY 08:00 -> 09:55 JST on Gotobi days
  F3  Gotobi reversal: short USDJPY 09:55 -> 12:00 JST on Gotobi days
      Gotobi days: the 5th, 10th, 15th, 20th, 25th and the last day of the month; a date on a weekend or a
      Japanese public holiday moves to the previous business day. Controls (not tested): same windows on all other
      business days.
  M1  Month-end London 4 p.m. fix (Melvin & Prins 2015), 9 currencies vs USD, equal-weight portfolio, window
      08:00 -> 16:00 London on the last UK business day of the month. Signal: local-currency return of the
      country's main equity index minus the S&P 500 over the month, using closes up to the PREVIOUS business day;
      foreign outperformance -> sell the foreign currency into the fix (hedge rebalancing), else buy it.
  M2  Same, window 16:00 London of the previous business day -> 16:00 London fix (24 hours).
  L1  Local trading hours (Breedon & Ranaldo): sell each currency vs USD at its home market's local open and buy it
      back at the local close, every business day; equal-weight portfolio of EUR, GBP, CHF, SEK, NOK (08:00-16:00
      Frankfurt / London / Zurich / Stockholm / Oslo local time), JPY (09:00-15:00 Tokyo), AUD (10:00-16:00
      Sydney), NZD (10:00-16:00 Auckland). CAD is excluded (its local hours coincide with US hours).
  L2  Same, EURUSD alone (the case the paper reports as profitable after spreads).
  V1  Sell one-month EURUSD variance each month at the option-implied level (Cboe EVZ, 30-day implied vol of
      EUR options, VIX method): monthly P&L in vol points (K^2 - RV^2) / (2K) with K = EVZ at the previous close
      and RV the realised volatility of the next month's daily EURUSD returns, minus 0.5 vol points of cost.
  V2  Volatility-risk-premium signal (Della Corte, Ramadorai & Sarno 2016), time-series version for EURUSD:
      insurance cost = EVZ - realised vol of the past month, z-scored on the past 24 months; expensive insurance
      -> short EUR for the next month, cheap -> long (monthly, spot P&L with spread/slippage/commission on changes).
  C1  Carry held only when VIX is calm (VIX at the previous close <= its trailing 252-day median), 7 USD pairs
      with point-in-time 3-month rates, long the higher-yielding currency, equal weight, daily, rate differential
      accrued minus a 0.75% p.a. retail swap mark-up, costs on every position change.
  C2  The same carry portfolio held all the time (baseline for C1).
  K1  FX premia around the clock (Krohn, Mueller & Whelan): long the foreign currencies vs USD during New York hours
      08:30 -> 17:00 ET, equal weight over the 9 currencies, every business day.
  K2  Same paper: short the foreign currencies vs USD overnight, 17:00 ET -> 08:30 ET next business day.
  K3  Both legs for the three most liquid currencies (EUR, GBP, CHF), the case reported profitable after spreads.
  P1  FOMC announcement days (Mueller, Tahbaz-Salehi & Vedolin): long the foreign currencies vs USD (dollar
      portfolio) from 17:00 ET the day before a scheduled FOMC announcement to 17:00 ET on the day.
  P2  Same, the three highest-yielding currencies (point-in-time 3-month rates) - the high-interest portfolio.
  W1  WMR 4 p.m. fix reversal (Evans: negative serial correlation around the fix): every business day, fade the
      15:30 -> 16:00 London move from 16:01 to 16:30 London, equal weight over the 9 currencies.
DATA NOTE: the M1 store of most pairs (all but EURUSD and XAGUSD locally) has no bars between 00:00 and 01:00 UTC,
which is 09:00-10:00 Tokyo. The Gotobi exits therefore use 10:00 JST, the first available bar after the 09:55 fix,
and the JPY / AUD local-hours legs fall back to the first bar within 65 minutes of the stated local time (recorded
per trade). The number of bars per UTC hour is stored for every pair.
Execution: entries and exits at the open of the M1 bar of the stated local time (first bar within 5 minutes,
otherwise no trade): buy at ask + slippage, sell at bid - slippage (slippage 0.25 x spread + 0.1 pip), commission
0.35 bp per side. Gross (mid, no costs) is reported alongside.
External data (fetched on the GitHub runner, cached in data/alt/): FRED VIXCLS and EVZCLS (fallbacks: FRED text
file, Yahoo ^VIX / ^EVZ, Cboe CSV); equity indices from Yahoo Finance (fallback: Stooq).

RUN 19 ADDITIONS (same tests, nothing re-tuned):
  * swap: every trade records the 17:00 New York rollovers it is held through (Wednesday x3) and two swap charges,
    the platform's flat 0.5 pip per night and the point-in-time rate differential minus a 0.75% p.a. mark-up
    (rate data exist for USD EUR GBP JPY CHF CAD AUD NZD; SEK / NOK use the flat charge). The registered net
    excludes swap; the report adds both versions.
  * placebo rows for the random-date test of the month-end rule (M1_placebo / M2_placebo: the same rule on every
    other business day, signal over the trailing month); Gotobi and FOMC already have controls on all other days.
  * random timing (B1-B6): on each event, 16 entry minutes drawn uniformly in the pre-event window, the edge's
    direction, exit at the event (B1 Gotobi, B3 month-end, B5 FOMC) or with a random bracket (B2, B4, B6);
    coin-flip-direction controls at the same entry minutes (*_control).
Usage: python scripts/flows_lab.py --pair USDJPY [--out DIR]"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEV_END = "2025-12-31"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALT = os.environ.get("FLOWS_ALT", os.path.join(ROOT, "data", "alt"))
# country equity index per currency: Yahoo symbol, Stooq fallbacks
EQUITY = {"USD": ("^GSPC", ["^spx"]), "EUR": ("^STOXX50E", ["^dax", "^stoxx50e"]), "GBP": ("^FTSE", ["^ukx"]),
          "JPY": ("^N225", ["^nkx"]), "AUD": ("^AXJO", ["^aor", "^xjo"]), "CAD": ("^GSPTSE", ["^tsx"]),
          "CHF": ("^SSMI", ["^smi"]), "SEK": ("^OMX", ["^omxs"]), "NOK": ("OSEBX.OL", ["^oseax", "^obx"]),
          "NZD": ("^NZ50", ["^nz50", "^nzx50"])}
LOCAL = {"EUR": ("Europe/Berlin", 8, 0, 16, 0), "GBP": ("Europe/London", 8, 0, 16, 0),
         "CHF": ("Europe/Zurich", 8, 0, 16, 0), "SEK": ("Europe/Stockholm", 8, 0, 16, 0),
         "NOK": ("Europe/Oslo", 8, 0, 16, 0), "JPY": ("Asia/Tokyo", 9, 0, 15, 0),
         "AUD": ("Australia/Sydney", 10, 0, 16, 0), "NZD": ("Pacific/Auckland", 10, 0, 16, 0)}
CARRY_PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")
MONTH_END_PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "USDSEK", "USDNOK", "NZDUSD")


# ------------------------------------------------------------------------------------------- external data
def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 fxlab-research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _cache(name):
    os.makedirs(ALT, exist_ok=True)
    return os.path.join(ALT, name)


FRED_SOURCES = []                  # (series, source) actually used, for the metadata


def _fred_csv(series, timeout):
    t = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2005-01-01&coed={DEV_END}",
             timeout=timeout)
    d = pd.read_csv(io.StringIO(t))
    d.columns = ["date", "value"]
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().set_index("date")["value"]


def _fred_txt(series, timeout):
    """FRED's plain-text download (header block, then 'DATE  VALUE' lines)."""
    t = _get(f"https://fred.stlouisfed.org/data/{series}.txt", timeout=timeout)
    rows = []
    started = False
    for line in t.splitlines():
        p = line.split()
        if not started:
            started = len(p) >= 2 and p[0].upper() == "DATE"
            continue
        if len(p) >= 2:
            try:
                rows.append((p[0], float(p[1])))
            except ValueError:
                pass
    return pd.Series([v for _, v in rows], index=[d for d, _ in rows], dtype=float)


def _yahoo(sym):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    import yfinance as yf
    h = yf.download(sym, start="2005-01-01", end="2026-01-01", progress=False, auto_adjust=False)
    c = h["Close"]
    c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
    c = c.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None) if getattr(c.index, "tz", None) else pd.to_datetime(c.index)
    return c


def _cboe(cb, timeout):
    t = _get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{cb}_History.csv", timeout=timeout)
    d = pd.read_csv(io.StringIO(t))
    d.columns = [c.strip().upper() for c in d.columns]
    return d.set_index("DATE")["CLOSE"].astype(float)


def fred(series):
    """Daily FRED series, cut at DEV_END. Sources tried in order (run 19: the EVZ download timed out in run 17):
    FRED CSV (3 attempts, 120 s), FRED text file, Yahoo Finance (^VIX / ^EVZ), Cboe history CSV."""
    path = _cache(f"{series}.csv")
    if not os.path.exists(path):
        s, src, err = None, None, []
        alt = {"VIXCLS": ("^VIX", "VIX"), "EVZCLS": ("^EVZ", "EVZ")}.get(series)
        tries = [(f"fred_csv_{k}", lambda k=k: _fred_csv(series, 120)) for k in range(3)]
        tries.append(("fred_txt", lambda: _fred_txt(series, 120)))
        if alt:
            tries.append((f"yahoo:{alt[0]}", lambda: _yahoo(alt[0])))
            tries.append((f"cboe:{alt[1]}", lambda: _cboe(alt[1], 120)))
        for name, fn in tries:
            try:
                x = fn()
                if x is not None and len(x) >= 100:
                    s, src = x, name
                    break
                err.append(f"{name}: only {0 if x is None else len(x)} rows")
            except Exception as ex:                                     # noqa: BLE001
                err.append(f"{name}: {str(ex)[:100]}")
                time.sleep(5)
        manual = os.path.join(ROOT, "data", "alt_manual", f"{series}_monthend.csv")
        if s is None and os.path.exists(manual):
            # month-end closes read from the FRED page in the research session (see the file header); enough for
            # the monthly tests V1 / V2, which only use the last close before each month
            x = pd.read_csv(manual, comment="#", parse_dates=["date"]).set_index("date")["value"]
            s, src = x, "manual_monthend:" + os.path.basename(manual)
            err.append("used the manual month-end file")
        if s is None:
            raise RuntimeError(f"{series}: no source worked: {err}")
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()].sort_index()
        s = s[s.index <= DEV_END]
        pd.DataFrame({"value": s, "source": src}).to_csv(path)
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    FRED_SOURCES.append((series, str(d["source"].iloc[0]) if "source" in d else "cache"))
    s = d["value"]
    return s[s.index <= DEV_END]


def equity(ccy):
    """Daily close of the currency's equity index (local date), cut at DEV_END; returns (series, source)."""
    ysym, stooq = EQUITY[ccy]
    path = _cache(f"eq_{ccy}.csv")
    if not os.path.exists(path):
        s, src, err = None, None, []
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            import yfinance as yf
            h = yf.download(ysym, start="2005-01-01", end="2026-01-01", progress=False, auto_adjust=False)
            if len(h):
                c = h["Close"]
                c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
                s, src = c.dropna(), f"yahoo:{ysym}"
        except Exception as ex:                                         # noqa: BLE001
            err.append(f"yahoo: {ex}")
        if s is None or len(s) < 500:
            for st in stooq:
                try:
                    t = _get(f"https://stooq.com/q/d/l/?s={st}&i=d")
                    d = pd.read_csv(io.StringIO(t))
                    if "Close" in d and len(d) > 500:
                        s, src = d.set_index("Date")["Close"].astype(float), f"stooq:{st}"
                        break
                except Exception as ex:                                 # noqa: BLE001
                    err.append(f"stooq {st}: {ex}")
        if s is None:
            raise RuntimeError(f"equity {ccy}: no source worked: {err}")
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
        s = s.sort_index()
        s = s[s.index <= DEV_END]
        pd.DataFrame({"close": s, "source": src}).to_csv(path)
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    return d["close"][d.index <= DEV_END], str(d["source"].iloc[0])


# ------------------------------------------------------------------------------------------- execution
class Book:
    """M1 bid/ask path of one pair with price lookups at local times."""

    def __init__(self, m1, pip, pair=None):
        from fxlab.execution.costs import BASE
        self.m = m1
        self.idx = m1.index
        self.bo, self.ao = m1["bid_o"].to_numpy(), m1["ask_o"].to_numpy()
        self.pip = pip
        self.sf, self.sp, self.comm = BASE.slip_spread_frac, BASE.slip_fixed_pips, BASE.commission_bps_per_side * 1e-4
        self.swap_pips = BASE.swap_pips_per_night
        self.rd = None                                  # point-in-time rate differential base - quote, by UTC day
        if pair is not None:
            from fxlab.data.rates import has_rates, rate_diff
            if has_rates(pair):
                days = pd.date_range(m1.index[0].floor("D"), m1.index[-1].floor("D"), freq="D")
                self.rd = rate_diff(pair, days)

    def at(self, ts_utc, tol_min=5):
        """Index of the first M1 bar at or after ts (within tol minutes), else -1."""
        i = int(self.idx.searchsorted(ts_utc))
        if i < len(self.idx) and (self.idx[i] - ts_utc) <= pd.Timedelta(minutes=tol_min):
            return i
        return -1

    @staticmethod
    def rollovers(t0, t1):
        """Swap nights charged for a position held from t0 to t1 (UTC): every weekday 17:00 New York instant strictly
        inside (t0, t1), the Wednesday one counted three times (spot value date skips the weekend). A position opened
        or closed at the open of the 17:00 bar itself is treated as flat over that rollover."""
        a, b = t0.tz_convert("America/New_York"), t1.tz_convert("America/New_York")
        n, day = 0, a.normalize()
        while day <= b.normalize():
            if day.weekday() < 5:
                inst = day + pd.Timedelta(hours=17)
                if a < inst < b:
                    n += 3 if day.weekday() == 2 else 1
            day += pd.Timedelta(days=1)
        return n

    def swaps(self, i, j, d, mid):
        """(nights, flat swap, rate swap) as returns of the position; flat = the platform default of 0.5 pip per night
        charged on either side, rate = (d x (r_base - r_quote) - 0.75% mark-up) / 365 per night (NaN without rates)."""
        n = self.rollovers(self.idx[i], self.idx[j])
        if n == 0:
            return 0, 0.0, 0.0
        flat = -n * self.swap_pips * self.pip / mid
        rate = np.nan
        if self.rd is not None:
            v = self.rd.asof(self.idx[i].floor("D"))                    # both UTC-aware
            if v is not None and np.isfinite(v):
                from fxlab.data.rates import SWAP_MARKUP_PA
                rate = n * (d * v - SWAP_MARKUP_PA) / 365.0
        return n, flat, rate

    def trade(self, t_in, t_out, d, tol_min=5):
        """Position d (+1 long the pair, -1 short) from t_in to t_out (UTC), or None. Returns (net without swap,
        gross mid return, entry delay min, exit delay min, swap nights, flat swap, rate swap)."""
        i, j = self.at(t_in, tol_min), self.at(t_out, tol_min)
        if i < 0 or j < 0 or j <= i:
            return None
        return self.trade_ij(i, j, d, t_in, t_out)

    def trade_ij(self, i, j, d, t_in=None, t_out=None):
        bi, ai, bj, aj = self.bo[i], self.ao[i], self.bo[j], self.ao[j]
        if not (np.isfinite([bi, ai, bj, aj]).all() and ai >= bi > 0 and aj >= bj > 0):
            return None
        si = self.sf * (ai - bi) + self.sp * self.pip
        sj = self.sf * (aj - bj) + self.sp * self.pip
        mi, mj = 0.5 * (ai + bi), 0.5 * (aj + bj)
        if d > 0:
            net = (bj - sj) / (ai + si) - 1.0
        else:
            net = 1.0 - (aj + sj) / (bi - si)
        delay_in = (self.idx[i] - t_in).total_seconds() / 60 if t_in is not None else 0.0
        delay_out = (self.idx[j] - t_out).total_seconds() / 60 if t_out is not None else 0.0
        n, flat, rate = self.swaps(i, j, d, mi)
        return net - 2 * self.comm, d * (mj / mi - 1.0), delay_in, delay_out, n, flat, rate

    def _hl(self):
        if not hasattr(self, "bh"):
            m = self.m
            self.bh, self.bl = m["bid_h"].to_numpy(), m["bid_l"].to_numpy()
            self.ah, self.al = m["ask_h"].to_numpy(), m["ask_l"].to_numpy()
            self.mo = 0.5 * (self.bo + self.ao)
            self.mh, self.ml = 0.5 * (self.bh + self.ah), 0.5 * (self.bl + self.al)

    def avg_range(self):
        """Mean daily mid high-low range of the previous 20 FX trading days, indexed by trading day (known at the
        start of that day)."""
        if not hasattr(self, "_rng"):
            from fxlab.validation.metrics import trading_day
            self._hl()
            td = trading_day(self.idx)
            h = pd.Series(self.mh, index=self.idx).groupby(td).max()
            lo = pd.Series(self.ml, index=self.idx).groupby(td).min()
            self._rng = (h - lo).rolling(20, min_periods=10).mean().shift(1)
        return self._rng

    @staticmethod
    def _first_exit(E, TP, SL, lo, hi, op, d):
        """First bar where the stop (checked first) or the target is touched: (offset, stop?) or (-1, None)."""
        hit_sl = (lo <= SL) if d > 0 else (hi >= SL)
        hit_tp = (hi >= TP) if d > 0 else (lo <= TP)
        ks = np.flatnonzero(hit_sl | hit_tp)
        if not len(ks):
            return -1, None
        return int(ks[0]), bool(hit_sl[ks[0]])

    def bracket(self, i, j, d, tp, sl):
        """Market entry at the open of bar i, take-profit tp and stop sl (price distances from the fill), time exit
        at the open of bar j. The stop wins when both are touched in one bar; a gap through the stop exits at that
        bar's open; the target is a limit order filled at its price on its own side of the quote (sell limits need
        the bid high, buy limits the ask low). Gross: the same rule on the mid path without costs."""
        self._hl()
        bi, ai = self.bo[i], self.ao[i]
        if not (np.isfinite([bi, ai, self.bo[j], self.ao[j]]).all() and ai >= bi > 0):
            return None
        si = self.sf * (ai - bi) + self.sp * self.pip
        mi = 0.5 * (ai + bi)
        if d > 0:
            E = ai + si
            k, stop = self._first_exit(E, E + tp, E - sl, self.bl[i:j], self.bh[i:j], self.bo[i:j], d)
        else:
            E = bi - si
            k, stop = self._first_exit(E, E - tp, E + sl, self.al[i:j], self.ah[i:j], self.ao[i:j], d)
        if k < 0:
            kx = j
            sj = self.sf * (self.ao[j] - self.bo[j]) + self.sp * self.pip
            px = self.bo[j] - sj if d > 0 else self.ao[j] + sj
        else:
            kx = i + k
            sk = self.sf * (self.ao[kx] - self.bo[kx]) + self.sp * self.pip
            if stop:
                px = min(E - sl, self.bo[kx]) - sk if d > 0 else max(E + sl, self.ao[kx]) + sk
            else:
                px = E + tp if d > 0 else E - tp
        net = (px / E - 1.0) if d > 0 else (1.0 - px / E)
        km, stopm = self._first_exit(mi, mi + d * tp, mi - d * sl, self.ml[i:j], self.mh[i:j], self.mo[i:j], d)
        if km < 0:
            pm = self.mo[j]
        else:
            pm = (min(mi - sl, self.mo[i + km]) if d > 0 else max(mi + sl, self.mo[i + km])) if stopm else mi + d * tp
        n, flat, rate = self.swaps(i, kx, d, mi)
        return net - 2 * self.comm, d * (pm / mi - 1.0), 0.0, 0.0, n, flat, rate


SEEDS = 16                      # run 19 random-timing draws per event
BRACKET = (0.1, 0.2, 0.4)       # take-profit / stop distances, x the mean daily range of the previous 20 days


def random_window(book, names, pair, day, w0, w1, t_exit, side, tol_exit=5, **kw):
    """Run-19 random-timing rows for one event: SEEDS entry bars drawn uniformly from the bars in [w0, w1), exit at
    t_exit. For each draw: the edge's direction and a coin-flip direction (control) at the same entry bar, each with a
    time exit (names[0]) and with a random bracket (names[1]; take-profit and stop each drawn from BRACKET x the
    previous 20 days' mean range). Draws are deterministic per (pair, day, seed)."""
    import zlib
    from fxlab.validation.metrics import trading_day
    i0, i1 = int(book.idx.searchsorted(w0)), int(book.idx.searchsorted(w1))
    j = book.at(t_exit, tol_exit)
    if j < 0 or i1 - i0 < 30 or j < i1:
        return []
    rg_s = book.avg_range()
    td = trading_day(pd.DatetimeIndex([book.idx[i0]]))[0]
    rg = float(rg_s.get(td, np.nan))
    out = []
    key = [zlib.crc32(pair.encode()), int(pd.Timestamp(day).toordinal())]
    for s in range(SEEDS):
        rng = np.random.default_rng([19, s] + key)
        i = int(rng.integers(i0, i1))
        coin = 1.0 if rng.random() < 0.5 else -1.0
        tpm, slm = float(rng.choice(BRACKET)), float(rng.choice(BRACKET))
        lead = (book.idx[j] - book.idx[i]).total_seconds() / 60
        for suffix, d in (("", side), ("_control", coin)):
            r = book.trade_ij(i, j, d)
            if r is not None:
                out.append(rec(names[0] + suffix, day, r, seed=s, lead_min=lead, **kw))
            if np.isfinite(rg) and rg > 0:
                r = book.bracket(i, j, d, tpm * rg, slm * rg)
                if r is not None:
                    out.append(rec(names[1] + suffix, day, r, seed=s, lead_min=lead, tp_mult=tpm, sl_mult=slm, **kw))
    return out


def rec(test, day, r, **kw):
    """Result row of one trade from Book.trade."""
    return {"test": test, "day": day, "net": r[0], "gross": r[1], "delay_in": r[2], "delay_out": r[3],
            "nights": r[4], "swap_flat": r[5], "swap_rate": r[6], **kw}


def local_ts(day, tz, hh, mm):
    """UTC timestamp of local wall time hh:mm on calendar date `day` in timezone tz."""
    return pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm).tz_localize(tz).tz_convert("UTC")


# ------------------------------------------------------------------------------------------- calendars
def jp_holidays(years):
    try:
        import holidays
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "holidays"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        import holidays
    return {pd.Timestamp(d) for d in holidays.country_holidays("JP", years=years)}


def gotobi_days(start, end, jp_hol):
    """Business days (Tokyo) on which a Gotobi settlement falls."""
    def is_bd(d):
        return d.weekday() < 5 and d not in jp_hol
    out = set()
    for m in pd.period_range(start, end, freq="M"):
        last = m.end_time.normalize()
        for dom in (5, 10, 15, 20, 25, None):
            d = last if dom is None else pd.Timestamp(year=m.year, month=m.month, day=dom)
            while not is_bd(d):
                d -= pd.Timedelta(days=1)
            out.add(d)
    return sorted(x for x in out if pd.Timestamp(start) <= x <= pd.Timestamp(end))


def month_end_days(start, end):
    """Last weekday of each month (UK fix day; UK bank holidays never fall on the last weekday except rare cases,
    which are skipped when no 16:00 London bar exists)."""
    out = []
    for m in pd.period_range(start, end, freq="M"):
        d = m.end_time.normalize()
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        out.append(d)
    return out


def prev_bday(d):
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


# ------------------------------------------------------------------------------------------- tests
def gotobi(book, start, end):
    days_all = pd.bdate_range(start, end)
    hol = jp_holidays(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))
    G = set(gotobi_days(start, end, hol))
    rows = []
    for d in days_all:
        if d in hol:
            continue
        g = d in G
        for name, t_in, t_out, side in (
                ("F1", local_ts(d, "Asia/Tokyo", 3, 0), local_ts(d, "Asia/Tokyo", 9, 55), +1),
                ("F2", local_ts(d, "Asia/Tokyo", 8, 0), local_ts(d, "Asia/Tokyo", 9, 55), +1),
                ("F3", local_ts(d, "Asia/Tokyo", 9, 55), local_ts(d, "Asia/Tokyo", 12, 0), -1)):
            if d.weekday() == 0 and name == "F1":
                continue                            # Monday 03:00 JST precedes the weekly open
            r = book.trade(t_in, t_out, side, tol_min=65)     # 09:55 JST falls in the missing 00:00-01:00 UTC hour
            if r is not None:
                rows.append(rec(name if g else name + "_control", d, r))
        if g:                                               # run 19: random entry minute 03:00-09:00 JST -> fix
            rows += random_window(book, ("B1", "B2"), "USDJPY", d, local_ts(d, "Asia/Tokyo", 3, 0),
                                  local_ts(d, "Asia/Tokyo", 9, 0), local_ts(d, "Asia/Tokyo", 9, 55), +1.0, tol_exit=65)
    return rows


def month_end(book, pair, start, end):
    base, quote = pair[:3], pair[3:]
    foreign = base if quote == "USD" else quote
    eq_f, src_f = equity(foreign)
    eq_u, src_u = equity("USD")
    rows = []
    for d in month_end_days(start, end):
        p = prev_bday(d)                                        # signal uses closes up to the previous business day
        m0 = (d.replace(day=1) - pd.Timedelta(days=1))          # last calendar day of the previous month
        def ret(s):
            a = s[s.index <= m0]
            b = s[s.index <= p]
            return b.iloc[-1] / a.iloc[-1] - 1.0 if len(a) and len(b) and b.index[-1] > a.index[-1] else np.nan
        sig = ret(eq_f) - ret(eq_u)
        if not np.isfinite(sig) or sig == 0:
            continue
        # foreign outperformed -> foreign currency depreciates into the fix: short foreign
        side_foreign = -1.0 if sig > 0 else 1.0
        side = side_foreign if base == foreign else -side_foreign
        for name, t_in in (("M1", local_ts(d, "Europe/London", 8, 0)), ("M2", local_ts(p, "Europe/London", 16, 0))):
            r = book.trade(t_in, local_ts(d, "Europe/London", 16, 0), side)
            if r is not None:
                rows.append(rec(name, d, r, signal=float(sig)))
        # run 19: random entry minute in the 24 h before the fix (16:00 London previous business day -> 15:30)
        rows += random_window(book, ("B3", "B4"), pair, d, local_ts(p, "Europe/London", 16, 0),
                              local_ts(d, "Europe/London", 15, 30), local_ts(d, "Europe/London", 16, 0), side,
                              ccy=foreign, signal=float(sig))
    # run 19 placebo for the random-date test: the same rule on every other business day, with the signal over the
    # trailing calendar month up to the previous business day (the month-end signal is month-to-date, the same span)
    ends = set(month_end_days(start, end))
    fi, ui = eq_f.index.to_numpy(), eq_u.index.to_numpy()
    fv, uv = eq_f.to_numpy(), eq_u.to_numpy()

    def ret_between(ix, v, a_, b_):
        ia, ib = np.searchsorted(ix, np.datetime64(a_), "right") - 1, np.searchsorted(ix, np.datetime64(b_), "right") - 1
        return v[ib] / v[ia] - 1.0 if ia >= 0 and ib > ia else np.nan
    for d in pd.bdate_range(start, end):
        if d in ends:
            continue
        p = prev_bday(d)
        a0 = d - pd.DateOffset(months=1)
        sig = ret_between(fi, fv, a0, p) - ret_between(ui, uv, a0, p)
        if not np.isfinite(sig) or sig == 0:
            continue
        side_foreign = -1.0 if sig > 0 else 1.0
        side = side_foreign if base == foreign else -side_foreign
        for name, t_in in (("M1", local_ts(d, "Europe/London", 8, 0)), ("M2", local_ts(p, "Europe/London", 16, 0))):
            r = book.trade(t_in, local_ts(d, "Europe/London", 16, 0), side)
            if r is not None:
                rows.append(rec(name + "_placebo", d, r, signal=float(sig)))
    return rows, {"equity_foreign": src_f, "equity_us": src_u}


def local_hours(book, pair, start, end):
    base, quote = pair[:3], pair[3:]
    foreign = base if quote == "USD" else quote
    if foreign not in LOCAL:
        return []
    tz, h0, m0, h1, m1_ = LOCAL[foreign]
    side = -1.0 if base == foreign else 1.0                     # short the foreign currency during its hours
    rows = []
    for d in pd.bdate_range(start, end):
        r = book.trade(local_ts(d, tz, h0, m0), local_ts(d, tz, h1, m1_), side, tol_min=65)
        if r is not None:
            rows.append(rec("L1", d, r))
            if pair == "EURUSD":
                rows.append(rec("L2", d, r))
    return rows


def daily_mid(bars):
    from fxlab.validation.metrics import trading_day
    c = 0.5 * (bars["bid_c"] + bars["ask_c"])
    return c.groupby(trading_day(c.index)).last()


def vrp(bars, book, start, end):
    """V1 (sell one-month variance at EVZ) and V2 (VRP direction signal), EURUSD, monthly."""
    evz = fred("EVZCLS")
    px = daily_mid(bars)
    r = np.log(px).diff().dropna()
    rows = []
    months = pd.period_range(max(pd.Timestamp(start), evz.index[0] + pd.Timedelta(days=60)), end, freq="M")
    ins = []
    for m in months:
        t0 = m.start_time
        e = evz[evz.index < t0]
        rp = r[r.index < t0].tail(21)
        rn = r[(r.index >= t0) & (r.index <= m.end_time)]
        if len(e) == 0 or len(rp) < 15 or len(rn) < 15 or (t0 - e.index[-1]).days > 7:
            continue
        K = float(e.iloc[-1])                                       # vol points
        rv_next = float(np.sqrt((rn ** 2).mean() * 252) * 100)
        rv_past = float(np.sqrt((rp ** 2).mean() * 252) * 100)
        pnl = (K ** 2 - rv_next ** 2) / (2 * K)
        rows.append({"test": "V1", "day": t0, "net": (pnl - 0.5) / 100, "gross": pnl / 100, "K": K, "rv": rv_next})
        ins.append((t0, K - rv_past, m))
    # V2: z-score of the insurance cost on the past 24 months -> direction for the month (a full round trip every
    # month, conservatively)
    I = pd.Series([x[1] for x in ins], index=[x[0] for x in ins])
    for k, (t0, cost, m) in enumerate(ins):
        hist = I.iloc[max(0, k - 24):k]
        if len(hist) < 12 or hist.std() == 0:
            continue
        z = (cost - hist.mean()) / hist.std()
        side = -1.0 if z > 0 else 1.0                               # expensive EUR insurance -> short EUR
        t_in = local_ts(t0, "America/New_York", 17, 5)               # first bar after the month's first rollover
        t_out = local_ts((m + 1).start_time, "America/New_York", 17, 5)
        res = book.trade(t_in, t_out, side, tol_min=4 * 1440)
        if res is not None:
            rows.append(rec("V2", t0, res, z=float(z)))
    return rows


def carry(bars, pair, start, end, pip):
    """Daily carry return of one pair (long the higher-yielder), unfiltered (C2) and VIX-filtered (C1)."""
    from fxlab.data.rates import rate_diff
    from fxlab.execution.costs import BASE
    from fxlab.validation.metrics import trading_day
    vix = fred("VIXCLS")
    d = bars
    dd = trading_day(d.index)
    mid_c = 0.5 * (d["bid_c"] + d["ask_c"])
    spr = (d["ask_c"] - d["bid_c"])
    day = pd.DataFrame({"mid": mid_c.groupby(dd).last(), "spr": spr.groupby(dd).median()})
    day = day[(day.index >= start) & (day.index <= end)]
    idx_utc = pd.DatetimeIndex(day.index).tz_localize("UTC")
    rd = rate_diff(pair, idx_utc).to_numpy()                      # base - quote, decimal p.a., point in time
    pos = np.sign(rd)
    v = vix.reindex(day.index, method="ffill").shift(1)           # VIX at the previous close
    med = vix.rolling(252, min_periods=126).median().reindex(day.index, method="ffill").shift(1)
    calm = (v <= med).to_numpy()
    ret = day["mid"].pct_change().to_numpy()
    out = []
    for name, p in (("C2", pos), ("C1", np.where(calm, pos, 0.0))):
        p_prev = np.r_[0.0, p[:-1]]                                 # position decided at the previous close
        accr = p_prev * (rd / 365.0) - np.abs(p_prev) * 0.0075 / 365.0
        trade = np.abs(np.diff(np.r_[0.0, p_prev]))
        cost = trade * (((0.5 + BASE.slip_spread_frac) * day["spr"].to_numpy() + BASE.slip_fixed_pips * pip)
                        / day["mid"].to_numpy() + BASE.commission_bps_per_side * 1e-4)
        gross = p_prev * np.nan_to_num(ret)
        net = gross + np.nan_to_num(accr) - np.nan_to_num(cost)
        for t, n_, g_ in zip(day.index, net, gross):
            out.append({"test": name, "day": t, "net": float(n_), "gross": float(g_)})
    return out


FOMC_FALLBACK = {          # scheduled FOMC statement days (second day of each meeting), used if the Fed pages fail
    2007: "01-31 03-21 05-09 06-28 08-07 09-18 10-31 12-11", 2008: "01-30 03-18 04-30 06-25 08-05 09-16 10-29 12-16",
    2009: "01-28 03-18 04-29 06-24 08-12 09-23 11-04 12-16", 2010: "01-27 03-16 04-28 06-23 08-10 09-21 11-03 12-14",
    2011: "01-26 03-15 04-27 06-22 08-09 09-21 11-02 12-13", 2012: "01-25 03-13 04-25 06-20 08-01 09-13 10-24 12-12",
    2013: "01-30 03-20 05-01 06-19 07-31 09-18 10-30 12-18", 2014: "01-29 03-19 04-30 06-18 07-30 09-17 10-29 12-17",
    2015: "01-28 03-18 04-29 06-17 07-29 09-17 10-28 12-16", 2016: "01-27 03-16 04-27 06-15 07-27 09-21 11-02 12-14",
    2017: "02-01 03-15 05-03 06-14 07-26 09-20 11-01 12-13", 2018: "01-31 03-21 05-02 06-13 08-01 09-26 11-08 12-19",
    2019: "01-30 03-20 05-01 06-19 07-31 09-18 10-30 12-11", 2020: "01-29 04-29 06-10 07-29 09-16 11-05 12-16",
    2021: "01-27 03-17 04-28 06-16 07-28 09-22 11-03 12-15", 2022: "01-26 03-16 05-04 06-15 07-27 09-21 11-02 12-14"}
MONTHS = "January February March April May June July August September October November December".split()


def fomc_dates(years):
    """Scheduled FOMC announcement days: parsed from the Fed's historical pages (meeting headings such as
    'January 24-25 Meeting'), checked against FOMC_FALLBACK; the fallback is used for years that fail to parse."""
    import re
    fb = {y: [pd.Timestamp(f"{y}-{x}") for x in FOMC_FALLBACK[y].split()] for y in years if y in FOMC_FALLBACK}
    out, report = [], {}
    pat = re.compile(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:\s*-\s*(?:(" + "|".join(MONTHS) +
                     r")\s+)?(\d{1,2}))?\s+Meeting", re.I)
    for y in years:
        got = []
        try:
            html = _get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
            for m in pat.finditer(html):
                mon1, d1, mon2, d2 = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                mon = MONTHS.index((mon2 or mon1).capitalize()) + 1
                got.append(pd.Timestamp(year=y, month=mon, day=int(d2) if d2 else d1))
            got = sorted(set(got))
        except Exception as ex:                                        # noqa: BLE001
            report[y] = f"fetch failed: {ex}"[:120]
        ref = fb.get(y, [])
        if len(got) >= 6:
            report.setdefault(y, f"parsed {len(got)}; not in fallback: {[str(d.date()) for d in got if d not in ref]}; "
                                 f"fallback not parsed: {[str(d.date()) for d in ref if d not in got]}")
            out += [d for d in got if d in ref] if ref else got     # scheduled meetings only
        else:
            out += ref
            report.setdefault(y, "fallback used")
    return sorted(set(out)), report


def around_clock(book, pair, start, end):
    """K: New York hours (08:30-17:00 ET) long foreign; overnight (17:00 ET prev business day -> 08:30 ET) short."""
    base, quote = pair[:3], pair[3:]
    foreign = base if quote == "USD" else quote
    long_f = 1.0 if base == foreign else -1.0                   # position in the pair that is long the foreign ccy
    rows = []
    for d in pd.bdate_range(start, end):
        p = prev_bday(d)
        r = book.trade(local_ts(d, "America/New_York", 8, 30), local_ts(d, "America/New_York", 17, 0), long_f)
        if r is not None:
            rows.append(rec("K_ny", d, r, ccy=foreign))
        r = book.trade(local_ts(p, "America/New_York", 17, 0), local_ts(d, "America/New_York", 8, 30), -long_f,
                       tol_min=65)
        if r is not None:
            rows.append(rec("K_on", d, r, ccy=foreign))
    return rows


def fomc(book, pair, start, end, meta):
    from fxlab.data.rates import rate_diff
    base, quote = pair[:3], pair[3:]
    foreign = base if quote == "USD" else quote
    long_f = 1.0 if base == foreign else -1.0
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    F, rep = fomc_dates(years)
    meta["fomc_source"] = rep
    F = set(F)
    days = pd.bdate_range(start, end)
    rd = rate_diff(pair, pd.DatetimeIndex(days).tz_localize("UTC")).to_numpy()
    rows = []
    for k, d in enumerate(days):
        p = prev_bday(d)
        r = book.trade(local_ts(p, "America/New_York", 17, 0), local_ts(d, "America/New_York", 17, 0), long_f,
                       tol_min=65)
        if r is None:
            continue
        diff_f = float(rd[k] * long_f) if np.isfinite(rd[k]) else np.nan        # foreign - USD rate
        rows.append(rec("P_dol" if d in F else "P_dol_control", d, r, ccy=foreign, rate_diff_foreign=diff_f))
        if d in F:                                          # run 19: random entry 17:00 ET previous day -> 16:30 ET
            rows += random_window(book, ("B5", "B6"), pair, d, local_ts(p, "America/New_York", 17, 0),
                                  local_ts(d, "America/New_York", 16, 30), local_ts(d, "America/New_York", 17, 0),
                                  long_f, tol_exit=65, ccy=foreign, rate_diff_foreign=diff_f)
    return rows


def fix_reversal(book, pair, start, end):
    rows = []
    for d in pd.bdate_range(start, end):
        i0 = book.at(local_ts(d, "Europe/London", 15, 30)); i1 = book.at(local_ts(d, "Europe/London", 16, 0))
        if i0 < 0 or i1 < 0:
            continue
        m0 = 0.5 * (book.bo[i0] + book.ao[i0]); m1 = 0.5 * (book.bo[i1] + book.ao[i1])
        if not (np.isfinite(m0) and np.isfinite(m1)) or m1 == m0:
            continue
        side = -1.0 if m1 > m0 else 1.0
        r = book.trade(local_ts(d, "Europe/London", 16, 1), local_ts(d, "Europe/London", 16, 30), side)
        if r is not None:
            rows.append(rec("W1", d, r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reports", "flows"))
    a = ap.parse_args()
    from fxlab.data.store import load_m1
    from fxlab.data.instruments import get as instrument
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.pair not in MONTH_END_PAIRS:                       # the tests are defined for the 9 USD pairs only
        json.dump({"pair": a.pair, "skipped": "not a USD pair of the flows universe"},
                  open(os.path.join(a.out, f"{a.pair}.json"), "w"))
        print(f"{a.pair}: skipped (not in the flows universe)")
        return
    m1 = load_m1(a.pair, a.start, a.end, columns=["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"])
    book = Book(m1, instrument(a.pair).pip, a.pair)
    start, end = str(m1.index[0].date()), str(min(m1.index[-1], pd.Timestamp(DEV_END, tz="UTC")).date())
    rows, meta, errors = [], {"pair": a.pair, "start": start, "end": end}, {}
    jobs = []
    if a.pair == "USDJPY":
        jobs.append(("gotobi", lambda: gotobi(book, start, end)))
    if a.pair in MONTH_END_PAIRS:
        def me():
            r, src = month_end(book, a.pair, start, end)
            meta.update(src)
            return r
        jobs.append(("month_end", me))
    jobs.append(("local_hours", lambda: local_hours(book, a.pair, start, end)))
    if a.pair == "EURUSD":
        jobs.append(("vrp", lambda: vrp(m1, book, start, end)))
    if a.pair in CARRY_PAIRS:
        jobs.append(("carry", lambda: carry(m1, a.pair, start, end, instrument(a.pair).pip)))
    if a.pair in MONTH_END_PAIRS:
        jobs.append(("around_clock", lambda: around_clock(book, a.pair, start, end)))
        jobs.append(("fomc", lambda: fomc(book, a.pair, start, end, meta)))
        jobs.append(("fix_reversal", lambda: fix_reversal(book, a.pair, start, end)))
    h = m1.groupby(m1.index.hour).size()
    meta["bars_by_utc_hour"] = {int(k): int(v) for k, v in h.items()}
    for name, fn in jobs:
        try:
            r = fn()
            rows += r
            meta[f"{name}_rows"] = len(r)
        except Exception as ex:                                         # noqa: BLE001
            errors[name] = repr(ex)[:500]
            print(f"{a.pair} {name}: ERROR {ex}", flush=True)
    meta["errors"] = errors
    meta["external_sources"] = sorted(set(FRED_SOURCES))
    meta["seconds"] = round(time.time() - t0, 1)
    R = pd.DataFrame(rows)
    if len(R):
        R["pair"] = a.pair
        R.to_parquet(os.path.join(a.out, f"{a.pair}.parquet"))
        print(R.groupby("test")[["net", "gross"]].agg(["count", "mean"]).to_string())
    import shutil
    for f in os.listdir(ALT) if os.path.isdir(ALT) else []:          # keep the external series with the results
        meta.setdefault("alt_files", []).append(f)
        shutil.copy(os.path.join(ALT, f), os.path.join(a.out, f"alt__{a.pair}__{f}"))
    json.dump(meta, open(os.path.join(a.out, f"{a.pair}.json"), "w"), indent=1, default=str)
    print(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
