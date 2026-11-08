"""Round 12 per-instrument job (docs/anatomy_preregistration.md): every single-instrument catalog strategy plus the
nine new systems plus five placebos, turned into trades, priced on the M1 bid / ask path in every entry / exit
variant of the fix menu, with MFE / MAE and the pre-entry features. Output: one parquet of trades per instrument.
Usage: python scripts/anatomy_lab.py --symbol EURUSD --out DIR [--only ID,ID]"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import signal
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = pd.Timestamp("2025-12-31 23:59", tz="UTC")
TFS = ["M1", "M5", "M15", "M30", "H1", "H4", "H8", "H12", "D1"]
TFD = {"H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}
EXITS = {"X05": 0.5, "X1": 1.0, "X2": 2.0}
ENTRIES = ("E0", "E1", "E2", "E3")
CAP = 5000
EXCL_FAMILY = {"machine_learning", "retail_mm", "esoteric"}
KEEP_MULTI = {"S.MS.ROUND_NUMBER", "S.EVT.GOLD_FIX"}
ROUND = {"JPY": 1.0, "XAU": 10.0}
LOG = {}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


# ------------------------------------------------------------------------------------------------ market arrays
class Mkt:
    def __init__(self, bk, sym):
        self.bk, self.sym = bk, sym
        bk._hl()
        self.idx = bk.idx
        self.t = bk.idx.as_unit("ns").asi8
        self.bo, self.ao, self.bh, self.bl, self.ah, self.al = bk.bo, bk.ao, bk.bh, bk.bl, bk.ah, bk.al
        self.mo, self.mh, self.ml = bk.mo, bk.mh, bk.ml
        self.spr = self.ao - self.bo
        self.slip = bk.sf * self.spr + bk.sp * bk.pip
        self.comm = bk.comm
        self.pip, self.swap_pips = bk.pip, bk.swap_pips
        # rollover instants (weekday 17:00 New York; Wednesday counts 3) for fast night counts
        days = pd.date_range(self.idx[0].tz_convert("America/New_York").normalize().tz_localize(None),
                             self.idx[-1].tz_convert("America/New_York").normalize().tz_localize(None) + pd.Timedelta(days=1))
        days = days[days.weekday < 5]
        inst = (days + pd.Timedelta(hours=17)).tz_localize("America/New_York", nonexistent="shift_forward",
                                                            ambiguous="NaT").tz_convert("UTC")
        ok = ~inst.isna()
        self.inst = inst[ok].as_unit("ns").asi8
        self.wcum = np.r_[0, np.cumsum(np.where(days[ok].weekday == 2, 3, 1))]
        self.rd = None
        if bk.rd is not None and len(bk.rd):
            r = bk.rd.dropna()
            self.rd_t, self.rd_v = pd.DatetimeIndex(r.index).as_unit("ns").asi8, r.to_numpy(float)
            self.rd = True
        from fxlab.data.rates import SWAP_MARKUP_PA
        self.markup = SWAP_MARKUP_PA
        hrs = self.idx.hour
        fit = self.idx <= pd.Timestamp("2018-12-31 23:59", tz="UTC")
        med = pd.Series(self.spr[fit] if fit.any() else self.spr).groupby(hrs[fit] if fit.any() else hrs).median()
        self.spr_med = med.reindex(range(24)).ffill().bfill().to_numpy()
        self.zero = np.zeros(len(self.slip))

    def at(self, ts, tol_min):
        i = int(np.searchsorted(self.t, ts))
        if i < len(self.t) and self.t[i] - ts <= tol_min * 60_000_000_000:
            return i
        return -1

    def nights(self, k0, k1):
        a, b = self.t[k0], self.t[k1]
        n_le_a = np.searchsorted(self.inst, a, "right")
        n_lt_b = np.searchsorted(self.inst, b, "left")
        return max(0, int(self.wcum[n_lt_b] - self.wcum[n_le_a]))

    def swap(self, k0, k1, d, mid):
        n = self.nights(k0, k1)
        if n == 0:
            return 0.0
        if self.rd:
            p = np.searchsorted(self.rd_t, self.t[k0], "right") - 1
            if p >= 0 and np.isfinite(self.rd_v[p]):
                return n * (d * self.rd_v[p] - self.markup) / 365.0
        return -n * self.swap_pips * self.pip / mid

    # ---------------------------------------------------------------------------------------- one trade
    def fill(self, spec, ent, atr_h1):
        """Fill index and price for the entry variant; None if not filled."""
        d, et = spec["d"], spec["etype"]
        t0 = spec["t0"]
        if et == "mkt":
            if ent == "E0":
                i = self.at(t0, 30)
                if i < 0:
                    return None
                return i, (self.ao[i] + self.slip[i]) if d > 0 else (self.bo[i] - self.slip[i]), self.mo[i], i
            if ent == "E1":
                i = self.at(t0 + 15 * 60_000_000_000, 30)
                if i < 0:
                    return None
                return i, (self.ao[i] + self.slip[i]) if d > 0 else (self.bo[i] - self.slip[i]), self.mo[i], i
            i0 = self.at(t0, 30)
            if i0 < 0 or not np.isfinite(atr_h1) or atr_h1 <= 0:
                return None
            j0 = min(int(np.searchsorted(self.t, t0 + 60 * 60_000_000_000)), len(self.t) - 1)
            ref = self.mo[i0]
            if ent == "E2":                                              # limit 0.25 ATR better
                L = ref - d * 0.25 * atr_h1
                hit = (self.al[i0:j0] <= L) if d > 0 else (self.bh[i0:j0] >= L)
                if not hit.any():
                    return None
                k = i0 + int(np.argmax(hit))
                return k, (min(L, self.ao[k]) if d > 0 else max(L, self.bo[k])), L, k + 1
            S = ref + d * 0.25 * atr_h1                                  # E3 stop entry 0.25 ATR in the direction
            hit = (self.ah[i0:j0] >= S) if d > 0 else (self.bl[i0:j0] <= S)
            if not hit.any():
                return None
            k = i0 + int(np.argmax(hit))
            E = (max(S, self.ao[k]) + self.slip[k]) if d > 0 else (min(S, self.bo[k]) - self.slip[k])
            return k, E, S, k + 1
        if ent != "E0":
            return None
        i0 = self.at(t0, 30)
        if i0 < 0:
            return None
        j0 = min(int(np.searchsorted(self.t, spec["t_valid"])), len(self.t) - 1)
        if j0 <= i0:
            return None
        px = spec["px"]
        if et == "lim":
            hit = (self.al[i0:j0] <= px) if d > 0 else (self.bh[i0:j0] >= px)
            if not hit.any():
                return None
            k = i0 + int(np.argmax(hit))
            E = min(px, self.ao[k]) if d > 0 else max(px, self.bo[k])
            return k, E, (px if E == px else self.mo[k]), k + 1
        hit = (self.ah[i0:j0] >= px) if d > 0 else (self.bl[i0:j0] <= px)       # stp
        if not hit.any():
            return None
        k = i0 + int(np.argmax(hit))
        through = (self.ao[k] > px) if d > 0 else (self.bo[k] < px)        # opened through the stop
        E = (max(px, self.ao[k]) + self.slip[k]) if d > 0 else (min(px, self.bo[k]) - self.slip[k])
        return k, E, (self.mo[k] if through else px), k + 1

    def end_index(self, spec, kf, xm):
        tf_ = self.t[kf]
        if spec.get("t_end") is not None and not pd.isna(spec.get("t_end")):
            te = spec["t_end"]
        else:
            te = tf_ + spec["hold"]
        te = int(tf_ + (te - tf_) * xm) if xm != 1.0 else int(te)
        j = int(np.searchsorted(self.t, te))
        return min(j, len(self.t) - 1)

    def run_exit(self, ks, j, d, E, sl, tp, lo, hi, op, slip):
        """Exit index and price: stop first, target as a limit, else time exit at the open of bar j."""
        if j <= ks:
            j = ks
        if np.isfinite(sl) or np.isfinite(tp):
            SL = E - d * sl if np.isfinite(sl) else (-np.inf if d > 0 else np.inf)
            TP = E + d * tp if np.isfinite(tp) else (np.inf if d > 0 else -np.inf)
            lo_, hi_ = lo[ks:j], hi[ks:j]
            hs = (lo_ <= SL) if d > 0 else (hi_ >= SL)
            ht = (hi_ >= TP) if d > 0 else (lo_ <= TP)
            any_ = hs | ht
            if any_.any():
                k = int(np.argmax(any_))
                kx = ks + k
                if hs[k]:
                    px = (min(SL, op[kx]) - slip[kx]) if d > 0 else (max(SL, op[kx]) + slip[kx])
                else:
                    px = TP
                return kx, px
        return j, None

    def price(self, spec, ent, xm, atr_h1, flip=False):
        sp = dict(spec)
        if flip:
            # the mirror trade: opposite side, stop and target swapped; a resting limit at a level becomes a stop
            # order at the same level (and vice versa), so the reversed trade fills when the level is reached, as
            # the original does, never earlier
            sp["d"] = -spec["d"]
            sp["sl"], sp["tp"] = spec["tp"], spec["sl"]
            if spec["etype"] in ("lim", "stp"):
                sp["etype"] = "stp" if spec["etype"] == "lim" else "lim"
        f = self.fill(sp, ent, atr_h1)
        if f is None:
            return None
        kf, E, Em, ks = f
        d = sp["d"]
        j = self.end_index(sp, kf, xm)
        if j <= kf:
            return None
        sl, tp = float(sp["sl"]) if sp["sl"] is not None else np.nan, float(sp["tp"]) if sp["tp"] is not None else np.nan
        if d > 0:
            kx, px = self.run_exit(ks, j, d, E, sl, tp, self.bl, self.bh, self.bo, self.slip)
            if px is None:
                px = self.bo[kx] - self.slip[kx]
            net = px / E - 1.0
        else:
            kx, px = self.run_exit(ks, j, d, E, sl, tp, self.al, self.ah, self.ao, self.slip)
            if px is None:
                px = self.ao[kx] + self.slip[kx]
            net = 1.0 - px / E
        net = net - 2 * self.comm + self.swap(kf, kx, d, self.mo[kf])
        kgx, gpx = self.run_exit(ks, j, d, Em, sl, tp, self.ml, self.mh, self.mo, self.zero)
        if gpx is None:
            gpx = self.mo[kgx]
        gross = d * (gpx / Em - 1.0)
        return net, gross, kf, kx, Em


# ------------------------------------------------------------------------------------------------ specs
def specs_from_state(side: pd.Series, bars: pd.DataFrame) -> pd.DataFrame:
    s = np.sign(np.nan_to_num(side.to_numpy(float)))
    idx = bars.index
    n = len(s)
    rows = []
    k = 0
    while k < n - 1:
        if s[k] != 0 and (k == 0 or s[k] != s[k - 1]):
            m = k + 1
            while m < n and s[m] == s[k]:
                m += 1
            if m + 1 <= n - 1:
                rows.append((idx[k + 1], int(s[k]), "mkt", np.nan, pd.NaT, np.nan, np.nan, pd.NaT, idx[m + 1]))
            k = m
        else:
            k += 1
    return pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold", "t_end"])


def specs_from_trade(sig: pd.DataFrame, bars: pd.DataFrame, extra: dict, tf: str) -> pd.DataFrame:
    from fxlab.features._util import atr
    risk = extra.get("risk") or {}
    side = sig["side"].fillna(0.0)
    ent = side.where(side != side.shift(1), 0.0)
    a = atr(bars, 14)
    mult = (risk.get("stop_params") or {}).get("mult", 2.0)
    tgt, tpp = risk.get("target", "affine"), risk.get("target_params") or {"a": 0.0, "b": 1.5}
    mh = risk.get("max_hold", 48)
    idx = bars.index
    rows = []
    for k in np.flatnonzero(ent.to_numpy() != 0):
        if k + 1 >= len(idx):
            continue
        d = int(np.sign(ent.iloc[k]))
        sl = sig["stop_dist"].iloc[k] if "stop_dist" in sig and np.isfinite(sig["stop_dist"].iloc[k]) else mult * a.iloc[k]
        if "tp_dist" in sig and np.isfinite(sig["tp_dist"].iloc[k]):
            tp = sig["tp_dist"].iloc[k]
        elif tgt == "none":
            tp = np.nan
        else:
            tp = tpp.get("a", 0.0) + tpp.get("b", 1.5) * sl
        hold = sig["max_hold"].iloc[k] if "max_hold" in sig and np.isfinite(sig["max_hold"].iloc[k]) else mh
        if not np.isfinite(sl) or sl <= 0:
            continue
        rows.append((idx[k + 1], d, "mkt", np.nan, pd.NaT, sl, tp, int(hold) * TFD[tf], pd.NaT))
    return pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold", "t_end"])


def placebo_specs(h1: pd.DataFrame, seed: int):
    from fxlab.features._util import atr
    rng = np.random.default_rng(seed)
    a = atr(h1, 14).to_numpy()
    idx = h1.index
    out = {}
    for name, hold, slm, tpm in (("P1_HOLD1", 1, None, None), ("P2_HOLD6", 6, None, None), ("P3_HOLD24", 24, None, None),
                                 ("P4_BRACKET_1_1", 24, 1.0, 1.0), ("P5_BRACKET_2_1", 24, 2.0, 1.0)):
        ks = np.flatnonzero(rng.random(len(idx) - 1) < 1 / 24)
        rows = []
        for k in ks:
            if not np.isfinite(a[k]):
                continue
            d = 1 if rng.random() < 0.5 else -1
            rows.append((idx[k + 1], d, "mkt", np.nan, pd.NaT, slm * a[k] if slm else np.nan, tpm * a[k] if tpm else np.nan,
                         hold * TFD["H1"], pd.NaT))
        out[name] = pd.DataFrame(rows, columns=["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold", "t_end"])
    return out


# ------------------------------------------------------------------------------------------------ features
class Feat:
    def __init__(self, h1: pd.DataFrame, d1: pd.DataFrame, sym: str):
        from fxlab.features._util import atr
        self.h1t = h1.index.as_unit("ns").asi8
        self.c1 = h1["close"].to_numpy()
        self.atr1 = atr(h1, 14).to_numpy()
        self.atr_ratio = (pd.Series(self.atr1) / pd.Series(self.atr1).rolling(6000, min_periods=500).median()).to_numpy()
        self.d1t = d1.index.as_unit("ns").asi8
        c = d1["close"]
        self.ret20 = (c / c.shift(20) - 1).to_numpy()
        self.vol20 = (np.log(c).diff().rolling(20).std() * math.sqrt(20)).to_numpy()
        self.step = ROUND.get(sym[:3], ROUND.get(sym[3:], 0.01)) if ("JPY" in sym or "XAU" in sym) else 0.01
        if "XAU" in sym:
            self.step = 10.0
        elif "JPY" in sym:
            self.step = 1.0

    def at(self, t0, d):
        k = int(np.searchsorted(self.h1t, t0 - 3_600_000_000_000, "right")) - 1
        q = int(np.searchsorted(self.d1t, t0 - 86_400_000_000_000, "right")) - 1
        out = {}
        if k >= 24 and np.isfinite(self.atr1[k]) and self.atr1[k] > 0:
            a = self.atr1[k]
            out["atr"] = a
            out["f_atr"] = self.atr_ratio[k]
            out["f_ext"] = d * (self.c1[k] / self.c1[k - 24] - 1) / (a / self.c1[k])
            r = self.c1[k] / self.step
            out["f_round"] = abs(r - round(r)) * self.step / a
        if q >= 0 and np.isfinite(self.ret20[q]):
            out["f_with"] = float(np.sign(d * self.ret20[q]))
            out["f_tstr"] = abs(self.ret20[q]) / self.vol20[q] if self.vol20[q] > 0 else np.nan
        ts = pd.Timestamp(t0, tz="UTC")
        h = ts.hour
        out["f_sess"] = 0 if (h >= 22 or h < 7) else (1 if h < 12 else (2 if h < 17 else 3))
        wd = ts.weekday()
        out["f_wday"] = 0 if wd in (0, 6) else (2 if wd == 4 else 1)
        tk = ts.tz_convert("Asia/Tokyo")
        last_bd = (tk.normalize() + pd.offsets.BMonthEnd(0)).day
        out["f_cal"] = 1.0 if (tk.day in (5, 10, 15, 20, 25) or tk.day >= last_bd - 1) else 0.0
        return out


# ------------------------------------------------------------------------------------------------ units
def catalog_units():
    import fxlab.strategies.daily  # noqa: F401
    import fxlab.strategies.extra  # noqa: F401
    import fxlab.strategies.famous  # noqa: F401
    import fxlab.strategies.rules  # noqa: F401
    from fxlab.registry import CATALOG
    units = []
    for e in CATALOG.values():
        if e.kind != "strategy" or e.family in EXCL_FAMILY:
            continue
        if e.multi_asset and e.id not in KEEP_MULTI:
            continue
        lo, hi = (e.frequencies.split("..") + [None])[:2]
        hi = hi or lo
        allowed = [tf for tf in ("H1", "H4", "D1") if TFS.index(lo) <= TFS.index(tf) <= TFS.index(hi)]
        for tf in allowed:
            units.append((e.id, tf, e))
    return units


def run_symbol(sym, out_dir, only=None, per_unit_s=240, budget_s=17400):
    import edges_lab2 as E2
    from fxlab.data.store import resample
    import anatomy_systems as AS
    bk = E2.book(sym)
    if bk is None:
        print("no data", sym)
        return
    M = Mkt(bk, sym)
    m1 = bk.m
    bars = {tf: resample(m1, tf) for tf in ("H1", "H4", "D1")}
    bars = {tf: b[(b.index <= DEV_END)] for tf, b in bars.items()}
    F = Feat(bars["H1"], bars["D1"], sym)
    units = [(f"{name}|{tf}", "new", tf, fn) for name, (fn, tfs) in AS.SYSTEMS.items() for tf in tfs]
    units += [(f"{name}|H1", "placebo", "H1", name) for name in ("P1_HOLD1", "P2_HOLD6", "P3_HOLD24", "P4_BRACKET_1_1", "P5_BRACKET_2_1")]
    units += [(f"{eid}|{tf}", "cat", tf, e) for eid, tf, e in catalog_units()]
    if only:
        units = [u for u in units if u[0].split("|")[0] in only]
    pl = placebo_specs(bars["H1"], seed=abs(hash(sym)) % 2**31)
    signal.signal(signal.SIGALRM, _alarm)
    all_rows = []
    rng = np.random.default_rng(7)
    t_job = time.time()
    for uid, kind, tf, obj in units:
        t_start = time.time()
        if t_start - t_job > budget_s:
            LOG[uid] = {"error": "job time budget reached"}
            continue
        try:
            signal.alarm(per_unit_s)
            b = bars[tf]
            if kind == "cat":
                fn = obj.fn
                kw = dict(obj.params)
                if "symbol" in inspect.signature(fn).parameters:
                    kw["symbol"] = sym
                if obj.id == "S.EVT.GOLD_FIX" and not sym.startswith("XAU"):
                    signal.alarm(0)
                    continue
                sig = fn(b.copy(), **kw)
                S = specs_from_trade(sig, b, obj.extra, tf) if obj.extra.get("mode") == "trade" else specs_from_state(sig["side"], b)
            elif kind == "new":
                k_, res = obj(b.copy(), tf)
                S = specs_from_state(res, b) if k_ == "state" else res
            else:
                S = pl[obj]
            S = S[(S.t0 <= DEV_END)].reset_index(drop=True)
            if len(S) > CAP:
                S = S.iloc[np.sort(rng.choice(len(S), CAP, replace=False))].reset_index(drop=True)
            rows = price_specs(M, F, S, uid, sym, deadline=t_start + per_unit_s)
            signal.alarm(0)
            all_rows.extend(rows)
            LOG[uid] = {"specs": int(len(S)), "trades": len(rows), "s": round(time.time() - t_start, 1)}
        except Timeout:
            LOG[uid] = {"error": "timeout"}
        except Exception as ex:                                          # noqa: BLE001
            signal.alarm(0)
            LOG[uid] = {"error": repr(ex)[:200]}
            traceback.print_exc(limit=1)
        print(sym, uid, LOG[uid], flush=True)
    D = pd.DataFrame(all_rows)
    os.makedirs(out_dir, exist_ok=True)
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D.to_parquet(os.path.join(out_dir, f"trades_{sym}.parquet"), index=False)
    pd.Series(LOG).to_json(os.path.join(out_dir, f"log_{sym}.json"))
    print("saved", sym, len(D))


SPEC_COLS = ["t0", "d", "etype", "px", "t_valid", "sl", "tp", "hold", "t_end"]


def price_specs(M, F, S, uid, sym, deadline):
    rows = []
    S = S.reindex(columns=SPEC_COLS)
    S["px"], S["sl"], S["tp"] = S["px"].astype(float), S["sl"].astype(float), S["tp"].astype(float)
    for r in S.itertuples(index=False):
        if time.time() > deadline:
            raise Timeout()
        spec = {"t0": pd.Timestamp(r.t0).value, "d": int(r.d), "etype": r.etype,
                "px": float(r.px) if np.isfinite(r.px) else np.nan,
                "t_valid": pd.Timestamp(r.t_valid).value if not pd.isna(r.t_valid) else None,
                "sl": float(r.sl) if np.isfinite(r.sl) else np.nan, "tp": float(r.tp) if np.isfinite(r.tp) else np.nan,
                "hold": pd.Timedelta(r.hold).value if not pd.isna(r.hold) else None,
                "t_end": pd.Timestamp(r.t_end).value if not pd.isna(r.t_end) else None}
        if spec["hold"] is None and spec["t_end"] is None:
            continue
        f = F.at(spec["t0"], spec["d"])
        a = f.get("atr", np.nan)
        base = M.price(spec, "E0", 1.0, a)
        if base is None:
            continue
        net, gross, kf, kx, Em = base
        row = {"unit": uid, "sym": sym, "t0": pd.Timestamp(M.t[kf], tz="UTC"), "d": spec["d"], "native": spec["etype"],
               "n_E0X1": net, "g_E0X1": gross}
        d = spec["d"]
        seg_h, seg_l = M.mh[kf:kx + 1], M.ml[kf:kx + 1]
        cost = (M.spr[kf] + 2 * M.slip[kf]) / M.mo[kf] + 2 * M.comm
        mfe = (seg_h.max() / Em - 1) if d > 0 else (1 - seg_l.min() / Em)
        mae = (seg_l.min() / Em - 1) if d > 0 else (1 - seg_h.max() / Em)
        row.update({"cost": cost, "mfe_c": mfe / cost if cost > 0 else np.nan, "mae_c": mae / cost if cost > 0 else np.nan,
                    "hold_min": (M.t[kx] - M.t[kf]) / 6e10, "f_spread": M.spr[kf] / max(M.spr_med[M.idx[kf].hour], 1e-12)})
        row.update({k: v for k, v in f.items() if k != "atr"})
        for e in ENTRIES:
            for xn, xm in EXITS.items():
                if e == "E0" and xn == "X1":
                    continue
                p = M.price(spec, e, xm, a)
                row[f"n_{e}{xn}"] = p[0] if p else np.nan
                row[f"g_{e}{xn}"] = p[1] if p else np.nan
        p = M.price(spec, "E0", 1.0, a, flip=True)
        row["n_FLIP"], row["g_FLIP"] = (p[0], p[1]) if p else (np.nan, np.nan)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    run_symbol(a.symbol, a.out, only=set(a.only.split(",")) - {""} or None)


if __name__ == "__main__":
    main()
