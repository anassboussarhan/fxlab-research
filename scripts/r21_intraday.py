"""Round 21, parts I and X (docs/round21_preregistration.md): ICT rules on indices and FX.

  idx   --out DIR      I1-I3 on US500 and NAS100, X1b (SMT indices), X2 (NDOG fill)
  fx    --out DIR      X1a (SMT EUR/USD vs GBP/USD), X3 (OTE), X4 (CBDR), X5 (breaker)
Each writes trade rows: test, sym, t_in, day, gross, net (primary cost), net_alt (CFD cost for index tests).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV_END = "2025-12-31"
NY = "America/New_York"
COLS = ["bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"]
FUT_PTS = {"USA500IDXUSD": 0.25 + 2.5 / 50, "USATECHIDXUSD": 0.25 + 2.5 / 20}
FX5 = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
CBDR5 = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD"]


def load(sym, start="2007-01-01"):
    from fxlab.data.store import load_m1
    m = load_m1(sym, start, DEV_END, columns=COLS)
    mid = pd.DataFrame({k: 0.5 * (m[f"bid_{k}"] + m[f"ask_{k}"]) for k in "ohlc"})
    mid["spr"] = (m.ask_o - m.bid_o).clip(lower=0)
    mid = mid[np.isfinite(mid.o) & (mid.o > 0)]
    if "IDX" in sym:
        pip = float(1e-4 * np.nanmedian(mid.c))
    elif sym.startswith("XAU"):
        pip = 0.1
    elif sym.startswith("XAG"):
        pip = 0.01
    else:
        pip = 0.01 if sym.endswith("JPY") else 1e-4
    return mid, pip


def side_cost(px, spr, pip):
    from fxlab.execution.costs import BASE
    return (0.5 * spr + BASE.slip_spread_frac * spr + BASE.slip_fixed_pips * pip) / px + BASE.commission_bps_per_side * 1e-4


def bars(mid, rule):
    g = mid.resample(rule, label="left", closed="left")
    b = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "spr": g.spr.mean()}).dropna()
    return b


def sim_limit(o, h, l, c, i0, i1, level, d, stop, tgt, max_hold, cancel=None):
    """Limit entry between bars i0..i1; stop first on the entry bar and after; target from the next bar.
    Returns (i_in, px_in, i_out, px_out) or None."""
    n = len(o)
    for j in range(i0, min(i1 + 1, n)):
        if cancel is not None and ((d > 0 and h[j] >= cancel) or (d < 0 and l[j] <= cancel)):
            return None
        hit = l[j] <= level if d > 0 else h[j] >= level
        if not hit:
            continue
        E = min(o[j], level) if d > 0 else max(o[j], level)
        if (d > 0 and l[j] <= stop) or (d < 0 and h[j] >= stop):
            return j, E, j, stop
        for k in range(j + 1, min(n, j + 1 + max_hold)):
            if d > 0:
                if l[k] <= stop:
                    return j, E, k, min(o[k], stop)
                if h[k] >= tgt:
                    return j, E, k, max(o[k], tgt)
            else:
                if h[k] >= stop:
                    return j, E, k, max(o[k], stop)
                if l[k] <= tgt:
                    return j, E, k, min(o[k], tgt)
        k = min(n - 1, j + max_hold)
        return j, E, k, c[k]
    return None


# ============================================================================================ indices
def day_slices(idx):
    ny = idx.tz_convert(NY)
    mod = (ny.hour * 60 + ny.minute).to_numpy()
    day = ny.normalize().tz_localize(None)
    pos = pd.Series(np.arange(len(idx)), index=day)
    return mod, {d: (int(v.iloc[0]), int(v.iloc[-1]) + 1) for d, v in pos.groupby(level=0)}


def cmd_idx(a):
    rows = []
    data = {}
    for sym in FUT_PTS:
        mid, pip = load(sym, "2012-01-01")
        data[sym] = (mid, pip)
        o, h, l, c, spr = (mid[k].to_numpy(float) for k in ("o", "h", "l", "c", "spr"))
        mod, S = day_slices(mid.index)
        fut = FUT_PTS[sym]
        days = sorted(S)
        prev = None

        def rec(test, d, iin, pin, iout, pout, dirn):
            g = dirn * (pout / pin - 1)
            fu = fut / pin + fut / pout
            cf = side_cost(pin, spr[iin], pip) + side_cost(pout, spr[iout], pip)
            rows.append({"test": test, "sym": sym, "t_in": mid.index[iin], "day": d, "gross": g, "net": g - fu, "net_alt": g - cf})

        for d in days:
            a0, a1 = S[d]
            mm = mod[a0:a1]
            at = {int(m): a0 + k for k, m in enumerate(mm)}
            need = (0, 570, 599, 600, 959)
            if not all(x in at for x in need):
                prev = None
                continue
            # I1 Silver Bullet
            rng = (h - l)[at[570]:at[599] + 1]
            med = float(np.median(rng)) if len(rng) else np.nan
            for k in range(at[600], at[599] + 61):
                if k >= a1 or mod[k] > 659:
                    break
                if k - 2 < a0:
                    continue
                if l[k] > h[k - 2] and l[k] - h[k - 2] >= 0.5 * med:
                    dd, lvl, stp = 1, l[k], l[k - 2]
                elif h[k] < l[k - 2] and l[k - 2] - h[k] >= 0.5 * med:
                    dd, lvl, stp = -1, h[k], h[k - 2]
                else:
                    continue
                last = max([at[m] for m in range(659, 599, -1) if m in at] or [k])
                ex = at.get(720)
                if ex is None:
                    break
                R = abs(lvl - stp)
                if R <= 0:
                    break
                tgt = lvl + dd * 2 * R
                r = sim_limit(o, h, l, c, k + 1, last, lvl, dd, stp, tgt, max_hold=ex - k)
                if r:
                    iin, pin, iout, pout = r
                    if iout > ex:
                        iout, pout = ex, o[ex]
                    rec("I1_silver_bullet", d, iin, pin, iout, pout, dd)
                break
            # I2 Judas swing
            op9, mo = o[at[570]], o[at[0]]
            hi, lo = h[at[570]:at[599] + 1].max(), l[at[570]:at[599] + 1].min()
            c959 = c[at[599]]
            sh = hi > max(op9, mo) and c959 < op9
            lg = lo < min(op9, mo) and c959 > op9
            if sh != lg:
                dd = -1 if sh else 1
                stp = hi if sh else lo
                iin, pin = at[600], o[at[600]]
                iout, pout = at[959], c[at[959]]
                for k in range(iin, at[959] + 1):
                    if (dd < 0 and h[k] >= stp) or (dd > 0 and l[k] <= stp):
                        iout, pout = k, (max(o[k], stp) if dd < 0 else min(o[k], stp))
                        break
                rec("I2_judas", d, iin, pin, iout, pout, dd)
            # I3 midnight-open bias
            if prev is not None:
                bias = np.sign(prev[1] - prev[0])
                if bias > 0 and op9 < mo:
                    rec("I3_midnight_bias", d, at[570], op9, at[959], c[at[959]], 1)
                elif bias < 0 and op9 > mo:
                    rec("I3_midnight_bias", d, at[570], op9, at[959], c[at[959]], -1)
            prev = (op9, c[at[959]])
        # X2 NDOG fill: 16:59 close -> 18:00 reopen, exit at the fill or 09:30 next day
        for j, d in enumerate(days[:-1]):
            a0, a1 = S[d]
            mm = mod[a0:a1]
            at = {int(m): a0 + k for k, m in enumerate(mm)}
            pre = [m for m in range(1019, 989, -1) if m in at]              # last bar 16:30-16:59
            post = [m for m in range(1080, 1110) if m in at]                # first bar 18:00-18:29
            if not pre or not post:
                continue
            nd = days[j + 1]
            b0, b1 = S[nd]
            at2 = {int(m): b0 + k for k, m in enumerate(mod[b0:b1])}
            if 570 not in at2:
                continue
            cl, op = c[at[pre[0]]], o[at[post[0]]]
            gap = op / cl - 1
            if abs(gap) < 0.0005:
                continue
            dd = -1 if gap > 0 else 1
            iin, pin = at[post[0]], op
            iout, pout = at2[570], o[at2[570]]
            for k in range(iin, at2[570]):
                if (dd < 0 and l[k] <= cl) or (dd > 0 and h[k] >= cl):
                    iout, pout = k, cl
                    break
            rec("X2_ndog", d, iin, pin, iout, pout, dd)
    # X1b SMT indices on 5-minute bars (09:30-15:55)
    B = {}
    for sym, (mid, pip) in data.items():
        b = bars(mid, "5min")
        ny = b.index.tz_convert(NY)
        m = ny.hour * 60 + ny.minute
        B[sym] = b[(m >= 570) & (m <= 955)]
    common = B["USA500IDXUSD"].index.intersection(B["USATECHIDXUSD"].index)
    A_, Bb = B["USA500IDXUSD"].reindex(common), B["USATECHIDXUSD"].reindex(common)
    dkey = common.tz_convert(NY).normalize()
    for sA, X, Y in (("USA500IDXUSD", A_, Bb), ("USATECHIDXUSD", Bb, A_)):
        pip = data[sA][1]
        o, h, l, c, spr = (X[k].to_numpy(float) for k in ("o", "h", "l", "c", "spr"))
        hy, ly = Y.h.to_numpy(float), Y.l.to_numpy(float)
        busy = -1
        pos = pd.Series(np.arange(len(common)), index=dkey)
        for d, v in pos.groupby(level=0):
            ii = v.to_numpy()
            for q in range(12, len(ii) - 1):
                b = ii[q]
                if b <= busy:
                    continue
                w = ii[q - 12:q]
                up = h[b] > h[w].max() and hy[b] <= hy[w].max()
                dn = l[b] < l[w].min() and ly[b] >= ly[w].min()
                if up == dn:
                    continue
                dd = -1 if up else 1
                iin = b + 1
                iout = ii[min(q + 12, len(ii) - 1)]
                pin, pout = o[iin], (o[iout] if iout != ii[-1] else c[iout])
                g = dd * (pout / pin - 1)
                fu = FUT_PTS[sA] / pin + FUT_PTS[sA] / pout
                cf = side_cost(pin, spr[iin], pip) + side_cost(pout, spr[iout], pip)
                rows.append({"test": "X1b_smt_idx", "sym": sA, "t_in": common[iin], "day": d.tz_localize(None), "gross": g,
                             "net": g - fu, "net_alt": g - cf})
                busy = iout
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, "r21_idx.parquet"))
    print(R.groupby(["test", "sym"])[["gross", "net", "net_alt"]].agg(["mean", "count"]).to_string())


# ============================================================================================ FX
def atr(b, n=20):
    pc = b.c.shift()
    tr = np.maximum(np.maximum(b.h - b.l, (b.h - pc).abs()), (b.l - pc).abs())
    return tr.rolling(n).mean().to_numpy(float)


def cmd_fx(a):
    rows = []
    H = {}
    M = {}
    for sym in sorted(set(FX5 + CBDR5)):
        try:
            mid, pip = load(sym, "2007-01-01")
        except Exception as ex:                                            # noqa: BLE001
            print("no data", sym, repr(ex)[:100])
            continue
        M[sym] = (mid, pip)
        H[sym] = bars(mid, "1h")
        print(sym, "minutes", len(mid), mid.index.min(), mid.index.max(), flush=True)

    def rec(test, sym, b, iin, pin, iout, pout, dd, pip):
        g = dd * (pout / pin - 1)
        net = g - side_cost(pin, b.spr.iloc[iin], pip) - side_cost(pout, b.spr.iloc[iout], pip)
        t = b.index[iin]
        rows.append({"test": test, "sym": sym, "t_in": t, "day": t.tz_convert(NY).normalize().tz_localize(None), "gross": g,
                     "net": net, "net_alt": net})

    # X1a SMT EUR/USD vs GBP/USD
    if "EURUSD" in H and "GBPUSD" in H:
        idx = H["EURUSD"].index.intersection(H["GBPUSD"].index)
        E, G = H["EURUSD"].reindex(idx), H["GBPUSD"].reindex(idx)
        for sA, X, Y in (("EURUSD", E, G), ("GBPUSD", G, E)):
            hx, lx, hy, ly = X.h.to_numpy(), X.l.to_numpy(), Y.h.to_numpy(), Y.l.to_numpy()
            mxh, mnl = pd.Series(hx).rolling(24).max().shift().to_numpy(), pd.Series(lx).rolling(24).min().shift().to_numpy()
            myh, myl = pd.Series(hy).rolling(24).max().shift().to_numpy(), pd.Series(ly).rolling(24).min().shift().to_numpy()
            o, c = X.o.to_numpy(), X.c.to_numpy()
            busy = -1
            for b in range(25, len(idx) - 13):
                if b <= busy:
                    continue
                up = hx[b] > mxh[b] and hy[b] <= myh[b]
                dn = lx[b] < mnl[b] and ly[b] >= myl[b]
                if up == dn:
                    continue
                if (idx[b + 12] - idx[b + 1]) > pd.Timedelta(hours=30):      # across a weekend: skip
                    continue
                dd = -1 if up else 1
                rec("X1a_smt_fx", sA, X, b + 1, o[b + 1], b + 12, o[b + 12], dd, M[sA][1])
                busy = b + 12
    # X3 OTE and X5 breaker on H1
    for sym in FX5:
        if sym not in H:
            continue
        b = H[sym]
        o, h, l, c = (b[k].to_numpy(float) for k in "ohlc")
        A = atr(b)
        pip = M[sym][1]
        busy = -1
        for d in range(21, len(b) - 60):
            if d <= busy or not np.isfinite(A[d - 1]):
                continue
            rg = h[d] - l[d]
            if rg < 2 * A[d - 1] or rg <= 0:
                continue
            pos = (c[d] - l[d]) / rg
            if pos >= 0.75:
                L, Hh, dd = l[d - 10:d + 1].min(), h[d], 1
            elif pos <= 0.25:
                L, Hh, dd = h[d - 10:d + 1].max(), l[d], -1
            else:
                continue
            Elim = Hh - 0.705 * (Hh - L)
            r = sim_limit(o, h, l, c, d + 1, d + 10, Elim, dd, L, Hh, 48, cancel=Hh)
            if r:
                rec("X3_ote", sym, b, r[0], r[1], r[2], r[3], dd, pip)
                busy = r[2]
        # X5 breaker
        win = 3
        sh = [s for s in range(win, len(b) - win) if h[s] == h[s - win:s + win + 1].max()]
        sl = [s for s in range(win, len(b) - win) if l[s] == l[s - win:s + win + 1].min()]
        busy = -1
        for kind, P, Q in (("bear", sh, sl), ("bull", sl, sh)):
            Q = np.array(Q)
            for i in range(1, len(P)):
                s, sp = P[i], P[i - 1]
                swept = h[s] > h[sp] if kind == "bear" else l[s] < l[sp]
                if not swept:
                    continue
                qs = Q[(Q < s) & (Q > sp)]
                if not len(qs):
                    continue
                q = int(qs[-1])
                conf = s + win
                lvl = l[q] if kind == "bear" else h[q]
                brk = None
                for t in range(conf, min(len(b) - 1, conf + 48)):
                    if (kind == "bear" and c[t] < lvl) or (kind == "bull" and c[t] > lvl):
                        brk = t
                        break
                if brk is None or brk <= busy or not np.isfinite(A[brk]):
                    continue
                dd = -1 if kind == "bear" else 1
                stp = (h[q] + 0.5 * A[brk]) if kind == "bear" else (l[q] - 0.5 * A[brk])
                R = abs(stp - lvl)
                tgt = lvl + dd * 2 * R
                r = sim_limit(o, h, l, c, brk + 1, brk + 24, lvl, dd, stp, tgt, 48)
                if r:
                    rec("X5_breaker", sym, b, r[0], r[1], r[2], r[3], dd, pip)
                    busy = r[2]
    # X4 CBDR on minute bars
    for sym in CBDR5:
        if sym not in M:
            continue
        mid, pip = M[sym]
        o, h, l, c, spr = (mid[k].to_numpy(float) for k in ("o", "h", "l", "c", "spr"))
        mod, S = day_slices(mid.index)
        days = sorted(S)
        dr = {d: h[S[d][0]:S[d][1]].max() - l[S[d][0]:S[d][1]].min() for d in days}
        drs = pd.Series(dr)
        adr = drs.rolling(20).mean().shift()
        for j, d in enumerate(days[:-1]):
            a0, a1 = S[d]
            mm = mod[a0:a1]
            sel = np.flatnonzero((mm >= 840) & (mm < 1200)) + a0
            if len(sel) < 200 or not np.isfinite(adr.get(d, np.nan)):
                continue
            CH, CL = h[sel].max(), l[sel].min()
            rg = CH - CL
            if rg <= 0 or rg > 0.4 * adr[d]:
                continue
            nd = days[j + 1]
            if (nd - d).days > 3:
                continue
            b0, b1 = S[nd]
            mm2 = mod[b0:b1]
            w = np.flatnonzero(mm2 < 720) + b0
            if len(w) < 100:
                continue
            ex = int(w[-1])
            mid_ = 0.5 * (CH + CL)
            for dd, lvl, stp in ((-1, CH + 2 * rg, CH + 4 * rg), (1, CL - 2 * rg, CL - 4 * rg)):
                r = sim_limit(o, h, l, c, int(w[0]), ex, lvl, dd, stp, mid_, max_hold=ex - int(w[0]))
                if r:
                    iin, pin, iout, pout = r
                    if iout > ex:
                        iout, pout = ex, c[ex]
                    g = dd * (pout / pin - 1)
                    net = g - side_cost(pin, spr[iin], pip) - side_cost(pout, spr[iout], pip)
                    rows.append({"test": "X4_cbdr", "sym": sym, "t_in": mid.index[iin], "day": nd, "gross": g, "net": net,
                                 "net_alt": net})
    R = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    R.to_parquet(os.path.join(a.out, "r21_fx.parquet"))
    print(R.groupby(["test", "sym"])[["gross", "net"]].agg(["mean", "count"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"idx": cmd_idx, "fx": cmd_fx}[a.cmd](a)


if __name__ == "__main__":
    main()
