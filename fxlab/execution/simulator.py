"""Execution simulators with bid/ask semantics.

TIMING CONTRACT (enforced here, not by strategies): a decision taken with the
information of bar t (i.e. at its close) is executed at the OPEN of bar
t + 1 + latency_bars. Nothing in a bar's own high/low/close is ever used to
fill an order decided on that same bar.

Two engines:
* ``vector_backtest``  - position-target engine for fast screening: holds a
  continuous position in [-1, 1] (fraction of max notional), open-to-open P&L,
  real quoted half-spread + slippage + commission on every change, financing
  on rollovers.
* ``simulate_trades``  - event engine for discrete trades with stop-loss,
  take-profit, trailing stop, time stop and signal exits. Intrabar ordering is
  ambiguous on OHLC bars, so the CONSERVATIVE rule is used: if both the stop
  and the target are touched inside the same bar, the stop is assumed first.
  Gaps through a stop fill at the (worse) open; gaps through a target fill at
  the (better) open. Long positions are valued/exited on the bid, shorts on the
  ask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from fxlab.execution.costs import CostModel, BASE

REASONS = {0: "open", 1: "stop", 2: "target", 3: "time", 4: "signal", 5: "trail", 6: "end_of_data"}


_ROLL_CACHE: dict = {}


def rollover_counts(index: pd.DatetimeIndex) -> np.ndarray:
    """Number of NY-17:00 rollovers in [open_j, open_{j+1}) for each bar j, with the rollover that closes the
    Wednesday trading day counted 3x (weekend financing). Vectorised; cached per index."""
    key = (len(index), index[0] if len(index) else None, index[-1] if len(index) else None)
    if key in _ROLL_CACHE:
        return _ROLL_CACHE[key]
    ny = index.tz_convert("America/New_York").tz_localize(None)
    d0 = np.asarray((ny + pd.Timedelta(hours=7)).floor("D").values.astype("datetime64[D]"))
    d1 = np.r_[d0[1:], d0[-1:]]
    if len(d0) == 0:
        return np.zeros(0)
    base = d0.min()
    days = np.arange(base, d1.max() + np.timedelta64(1, "D"), dtype="datetime64[D]")
    wd = (days.view("int64") - 4) % 7          # 0 = Monday
    w = np.where(wd == 2, 3, np.where(wd < 5, 1, 0))
    C = np.r_[0, np.cumsum(w)]                  # C[k] = sum of weights of days[0..k-1]
    i0 = (d0 - base).astype(np.int64)
    i1 = (d1 - base).astype(np.int64)
    cnt = (C[i1] - C[i0]).astype(float)        # trading days d0 .. d1-1 end inside the interval
    if len(_ROLL_CACHE) > 64:
        _ROLL_CACHE.clear()
    _ROLL_CACHE[key] = cnt
    return cnt


def _arrays(bars: pd.DataFrame):
    return tuple(bars[c].to_numpy(np.float64) for c in
                 ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"))


# ---------------------------------------------------------------------------------------------- vector engine
def vector_backtest(bars: pd.DataFrame, position: pd.Series, pip: float, cost: CostModel = BASE,
                    rollovers: np.ndarray | None = None, carry: np.ndarray | None = None) -> pd.DataFrame:
    """position[t] = target exposure decided at close of bar t, in [-1, 1].

    Returns per-bar frame: held position, gross return, spread/slippage cost,
    commission, financing and net return (fractions of capital at 1x notional).
    carry: optional point-in-time annualised rate differential (base - quote, decimal) per bar. When given,
    financing per rollover = (|position| x mark-up - position x carry) / 360, i.e. a long position EARNS the
    differential minus the broker mark-up (fxlab.data.rates.SWAP_MARKUP_PA, scaled with cost stress) and a
    short pays it; otherwise the flat swap_pips_per_night charge on both sides is used.
    """
    pos = position.reindex(bars.index).fillna(0.0).clip(-1, 1).to_numpy(float)
    L = 1 + cost.latency_bars
    held = np.r_[np.zeros(L), pos[:-L]] if len(pos) > L else np.zeros(len(pos))
    bo, _, _, bc, ao, _, _, ac = _arrays(bars)
    mid_o = (bo + ao) / 2
    mid_c = (bc + ac) / 2
    nxt_o = np.r_[mid_o[1:], mid_c[-1]]
    gross = held * (nxt_o / mid_o - 1.0)
    dh = np.abs(np.diff(np.r_[0.0, held]))
    half_spread = cost.spread_mult * (ao - bo) / 2.0
    slip = cost.slip_spread_frac * (ao - bo) + cost.slip_fixed_pips * pip
    spread_cost = dh * (half_spread + slip) / mid_o
    commission = dh * cost.commission_bps_per_side * 1e-4
    if rollovers is None:
        rollovers = rollover_counts(bars.index)
    if carry is not None:
        from fxlab.data.rates import SWAP_MARKUP_PA
        markup = SWAP_MARKUP_PA * (cost.swap_pips_per_night / BASE.swap_pips_per_night if BASE.swap_pips_per_night else 1.0)
        cr = np.nan_to_num(np.asarray(carry, float))
        financing = rollovers / 360.0 * (np.abs(held) * markup - held * cr)
    else:
        financing = np.abs(held) * rollovers * cost.swap_pips_per_night * pip / mid_o
    net = gross - spread_cost - commission - financing
    return pd.DataFrame({"position": held, "gross": gross, "spread_slip": spread_cost, "commission": commission,
                         "financing": financing, "net": net}, index=bars.index)


# ----------------------------------------------------------------------------------------------- event engine
@njit(cache=True)
def _simulate(bo, bh, bl, bc, ao, ah, al, ac, roll, side_sig, stop_d, tp_d, max_hold, trail_d, size_f,
              latency, slip_frac, slip_pips, pip, spread_mult, comm, swap_pips, exit_on_flat, be_m, part_m, part_fr):
    n = len(bo)
    max_tr = n
    t_entry = np.full(max_tr, -1); t_exit = np.full(max_tr, -1); t_side = np.zeros(max_tr)
    t_epx = np.zeros(max_tr); t_xpx = np.zeros(max_tr); t_reason = np.zeros(max_tr, np.int64)
    t_size = np.zeros(max_tr); t_fin = np.zeros(max_tr); t_mae = np.zeros(max_tr); t_mfe = np.zeros(max_tr)
    t_spread = np.zeros(max_tr)
    bar_ret = np.zeros(n)
    k = -1
    pos = 0.0; epx = 0.0; stop = np.nan; tgt = np.nan; trail = np.nan; ebar = -1; f = 0.0
    best = 0.0; worst = 0.0; fin = 0.0; mh = 0; last_val = 0.0
    sd0 = 0.0; be_k = np.nan; pm_k = np.nan; pf_k = 0.0; part_done = False
    for j in range(n):
        mid_o = (bo[j] + ao[j]) / 2.0
        hs = spread_mult * (ao[j] - bo[j]) / 2.0
        slip = slip_frac * (ao[j] - bo[j]) + slip_pips * pip
        buy_o = mid_o + hs + slip   # market buy at open
        sell_o = mid_o - hs - slip  # market sell at open
        d = j - 1 - latency          # decision bar executed at this open
        # ---------------------------------------------------------- 1) open: time stop and signal-driven actions
        if pos != 0:
            exit_now = False; reason = 0
            if mh > 0 and j - ebar >= mh:
                exit_now = True; reason = 3
            elif d >= 0 and ((exit_on_flat and side_sig[d] == 0) or side_sig[d] == -pos):
                exit_now = True; reason = 4
            if exit_now:
                px = sell_o if pos > 0 else buy_o
                r = pos * (px - epx) / epx
                bar_ret[j] += f * (r - last_val)
                t_exit[k] = j; t_xpx[k] = px; t_reason[k] = reason; t_fin[k] = fin
                t_mae[k] = worst; t_mfe[k] = best
                bar_ret[j] -= f * comm
                pos = 0.0
        if pos == 0 and d >= 0 and side_sig[d] != 0 and not np.isnan(stop_d[d]) and size_f[d] > 0:
            k += 1
            pos = side_sig[d]
            epx = buy_o if pos > 0 else sell_o
            stop = epx - pos * stop_d[d]
            tgt = epx + pos * tp_d[d] if not np.isnan(tp_d[d]) else np.nan
            trail = trail_d[d]
            mh = int(max_hold[d]) if not np.isnan(max_hold[d]) else 0
            f = size_f[d]; ebar = j; best = 0.0; worst = 0.0; fin = 0.0
            sd0 = stop_d[d]; be_k = be_m[d]; pm_k = part_m[d]; pf_k = part_fr[d]; part_done = False
            t_entry[k] = j; t_side[k] = pos; t_epx[k] = epx; t_size[k] = f
            t_spread[k] = (hs + slip) / mid_o
            bar_ret[j] -= f * comm
            # mark entry cost relative to the mid at the open
            last_val = pos * (mid_o - epx) / epx
            bar_ret[j] += f * last_val
        # ---------------------------------------------------------- 2) intrabar stop / target
        if pos != 0:
            done = False
            if pos > 0:
                lo, hi = bl[j], bh[j]
                if bo[j] <= stop and ebar != j:
                    px = bo[j] - slip; reason = 1 if np.isnan(trail) else 5; done = True
                elif not np.isnan(tgt) and bo[j] >= tgt and ebar != j:
                    px = bo[j]; reason = 2; done = True
                elif lo <= stop:
                    px = stop - slip; reason = 1 if np.isnan(trail) else 5; done = True
                elif not np.isnan(tgt) and hi >= tgt:
                    px = tgt; reason = 2; done = True
                worst = min(worst, (lo - epx) / epx); best = max(best, (hi - epx) / epx)
            else:
                lo, hi = al[j], ah[j]
                if ao[j] >= stop and ebar != j:
                    px = ao[j] + slip; reason = 1 if np.isnan(trail) else 5; done = True
                elif not np.isnan(tgt) and ao[j] <= tgt and ebar != j:
                    px = ao[j]; reason = 2; done = True
                elif hi >= stop:
                    px = stop + slip; reason = 1 if np.isnan(trail) else 5; done = True
                elif not np.isnan(tgt) and lo <= tgt:
                    px = tgt; reason = 2; done = True
                worst = min(worst, (epx - hi) / epx); best = max(best, (epx - lo) / epx)
            if done:
                r = pos * (px - epx) / epx
                bar_ret[j] += f * (r - last_val)
                bar_ret[j] -= f * comm
                t_exit[k] = j; t_xpx[k] = px; t_reason[k] = reason; t_fin[k] = fin
                t_mae[k] = worst; t_mfe[k] = best
                pos = 0.0
            elif not part_done and pm_k == pm_k and pf_k > 0:
                # partial take-profit at pm_k x initial stop distance (checked after the stop: stop-first rule)
                lvl = epx + pos * pm_k * sd0
                hit = False
                if pos > 0:
                    if bo[j] >= lvl and ebar != j:
                        px = bo[j]; hit = True
                    elif bh[j] >= lvl:
                        px = lvl; hit = True
                else:
                    if ao[j] <= lvl and ebar != j:
                        px = ao[j]; hit = True
                    elif al[j] <= lvl:
                        px = lvl; hit = True
                if hit:
                    fp = f * pf_k
                    r = pos * (px - epx) / epx
                    bar_ret[j] += fp * (r - last_val) - fp * comm
                    f -= fp
                    part_done = True
                    if be_k == be_k:          # with breakeven enabled, the remainder is protected at entry
                        stop = max(stop, epx) if pos > 0 else min(stop, epx)
        # ---------------------------------------------------------- 3) close: mark-to-market, trailing, financing
        if pos != 0:
            val = pos * (((bc[j] if pos > 0 else ac[j]) - epx) / epx)
            bar_ret[j] += f * (val - last_val)
            last_val = val
            if roll[j] > 0:
                fc = roll[j] * swap_pips * pip / epx
                fin += fc
                bar_ret[j] -= f * fc
            if not np.isnan(trail):
                if pos > 0:
                    stop = max(stop, bh[j] - trail)
                else:
                    stop = min(stop, al[j] + trail)
            if be_k == be_k:                  # breakeven: once price has moved be_k x stop distance in favour
                if pos > 0 and bh[j] - epx >= be_k * sd0:
                    stop = max(stop, epx)
                elif pos < 0 and epx - al[j] >= be_k * sd0:
                    stop = min(stop, epx)
    if pos != 0:
        t_exit[k] = n - 1; t_xpx[k] = bc[n - 1] if pos > 0 else ac[n - 1]; t_reason[k] = 6; t_fin[k] = fin
        t_mae[k] = worst; t_mfe[k] = best
    m = k + 1
    return (bar_ret, t_entry[:m], t_exit[:m], t_side[:m], t_epx[:m], t_xpx[:m], t_reason[:m], t_size[:m],
            t_fin[:m], t_mae[:m], t_mfe[:m], t_spread[:m])


@dataclass
class TradeResult:
    bars: pd.DataFrame       # per-bar net return (fraction of equity)
    trades: pd.DataFrame     # trade log


def simulate_trades(bars: pd.DataFrame, signals: pd.DataFrame, pip: float, cost: CostModel = BASE,
                    exit_on_flat: bool = True, rollovers: np.ndarray | None = None) -> TradeResult:
    """signals (indexed like bars, decided at bar close) columns:
       side (-1/0/1), stop_dist (price, required), tp_dist (price or NaN), max_hold (bars or NaN),
       trail_dist (price or NaN), size (fraction of equity notional, e.g. from risk sizing)."""
    s = signals.reindex(bars.index)
    side = s["side"].fillna(0).to_numpy(np.float64)
    stop_d = s["stop_dist"].to_numpy(np.float64)
    tp_d = s.get("tp_dist", pd.Series(np.nan, index=s.index)).to_numpy(np.float64)
    mh = s.get("max_hold", pd.Series(np.nan, index=s.index)).to_numpy(np.float64)
    tr = s.get("trail_dist", pd.Series(np.nan, index=s.index)).to_numpy(np.float64)
    size = s.get("size", pd.Series(1.0, index=s.index)).fillna(0).to_numpy(np.float64)
    nanc = pd.Series(np.nan, index=s.index)
    be = s.get("be_mult", nanc).to_numpy(np.float64)
    pm = s.get("part_mult", nanc).to_numpy(np.float64)
    pf = s.get("part_frac", pd.Series(0.0, index=s.index)).fillna(0.0).to_numpy(np.float64)
    if rollovers is None:
        rollovers = rollover_counts(bars.index)
    out = _simulate(*_arrays(bars), rollovers.astype(np.float64), side, stop_d, tp_d, mh, tr, size,
                    int(cost.latency_bars), cost.slip_spread_frac, cost.slip_fixed_pips, pip, cost.spread_mult,
                    cost.commission_bps_per_side * 1e-4, cost.swap_pips_per_night, exit_on_flat, be, pm, pf)
    bar_ret, te, tx, ts, ep, xp, rs, sz, fin, mae, mfe, spr = out
    idx = bars.index
    gross = ts * (xp - ep) / ep
    trades = pd.DataFrame({
        "entry_time": idx[te] if len(te) else pd.DatetimeIndex([], tz="UTC"),
        "exit_time": idx[np.clip(tx, 0, None)] if len(tx) else pd.DatetimeIndex([], tz="UTC"),
        "side": ts, "entry_px": ep, "exit_px": xp, "reason": [REASONS[int(r)] for r in rs], "size": sz,
        "ret_price": gross, "commission": 2 * cost.commission_bps_per_side * 1e-4, "financing": fin,
        "entry_cost_frac": spr, "bars_held": tx - te, "mae": mae, "mfe": mfe})
    trades["net_ret"] = trades["ret_price"] - trades["commission"] - trades["financing"]
    trades["pnl_frac_equity"] = trades["net_ret"] * trades["size"]
    return TradeResult(pd.DataFrame({"net": bar_ret}, index=idx), trades)


# --------------------------------------------------------------------------------------------- passive engine
@njit(cache=True)
def _passive(bo, bh, bl, bc, ao, ah, al, ac, roll, target, latency, offset, max_wait, queue_mult, slip_frac,
             slip_pips, pip, comm, swap_pips):
    n = len(bo)
    held = np.zeros(n); gross = np.zeros(n); sslip = np.zeros(n); commission = np.zeros(n)
    financing = np.zeros(n); net = np.zeros(n)
    placed = 0; filled = 0; fallback = 0
    h = 0.0; wait = 0
    for j in range(1, n):
        mid_c = (bc[j] + ac[j]) / 2.0
        mid_pc = (bc[j - 1] + ac[j - 1]) / 2.0
        pnl = h * (mid_c / mid_pc - 1.0)          # carried position, close-to-close at mid
        d = j - 1 - latency
        tgt = target[d] if d >= 0 else 0.0
        dq = tgt - h
        cost_j = 0.0; com_j = 0.0
        if dq != 0.0:
            spr = ac[d] - bc[d] if d >= 0 else ac[j - 1] - bc[j - 1]
            mid_d = (bc[d] + ac[d]) / 2.0 if d >= 0 else mid_pc
            if wait >= max_wait and max_wait > 0:
                # give up waiting: cross the spread at this bar's open (market order)
                if dq > 0:
                    F = ao[j] + slip_frac * (ao[j] - bo[j]) + slip_pips * pip
                else:
                    F = bo[j] - slip_frac * (ao[j] - bo[j]) - slip_pips * pip
                ref = (bo[j] + ao[j]) / 2.0
                fallback += 1
                fill = True
            else:
                placed += 1
                if dq > 0:   # buy limit resting at (bid + offset*spread) of the decision bar
                    F = bc[d] + offset * spr if d >= 0 else bc[j - 1]
                    fill = al[j] < F - (queue_mult - 1.0) * spr   # strict trade-through by the ask
                else:        # sell limit at (ask - offset*spread)
                    F = ac[d] - offset * spr if d >= 0 else ac[j - 1]
                    fill = bh[j] > F + (queue_mult - 1.0) * spr
                ref = mid_d
                if fill:
                    filled += 1
            if fill:
                pnl += dq * (mid_c - F) / F
                cost_j = abs(dq) * (F - ref) / ref * (1.0 if dq > 0 else -1.0)   # >0 paid, <0 earned
                com_j = abs(dq) * comm
                h = tgt
                wait = 0
            else:
                wait += 1
        else:
            wait = 0
        fin = abs(h) * roll[j] * swap_pips * pip / mid_c
        held[j] = h
        net[j] = pnl - com_j - fin
        sslip[j] = cost_j
        commission[j] = com_j
        financing[j] = fin
        gross[j] = net[j] + cost_j + com_j + fin
    return held, gross, sslip, commission, financing, net, placed, filled, fallback


def passive_backtest(bars: pd.DataFrame, position: pd.Series, pip: float, cost: CostModel = BASE,
                     offset: float = 0.0, max_wait: int = 0, rollovers: np.ndarray | None = None) -> pd.DataFrame:
    """Limit-order execution of a target-position series (decided at close of bar t).

    Every change of target places a limit order at the decision bar's best bid (buys) / best ask (sells),
    shifted towards mid by ``offset`` x spread. It fills in a later bar only if the opposite quote trades
    strictly THROUGH the limit price (no queue priority assumed); ``cost.spread_mult`` > 1 demands an extra
    (spread_mult-1) x spread of trade-through (stress). Unfilled orders are re-quoted each bar at the new
    best price; after ``max_wait`` unfilled bars (0 = never) the change is executed at market at the next
    open with full spread + slippage. Fills pay commission; there is no spread cost on passive fills (the
    spread is earned), but adverse selection is fully captured because P&L is marked on subsequent mids.
    Columns match vector_backtest; fill statistics are in ``.attrs``."""
    pos = position.reindex(bars.index).fillna(0.0).clip(-1, 1).to_numpy(float)
    if rollovers is None:
        rollovers = rollover_counts(bars.index)
    out = _passive(*_arrays(bars), rollovers.astype(np.float64), pos, int(cost.latency_bars), float(offset),
                   int(max_wait), float(max(cost.spread_mult, 1.0)), cost.slip_spread_frac, cost.slip_fixed_pips, pip,
                   cost.commission_bps_per_side * 1e-4, cost.swap_pips_per_night)
    held, gross, sslip, com, fin, net, placed, filled, fb = out
    df = pd.DataFrame({"position": held, "gross": gross, "spread_slip": sslip, "commission": com, "financing": fin,
                       "net": net}, index=bars.index)
    df.attrs.update({"orders_placed": int(placed), "orders_filled": int(filled), "market_fallbacks": int(fb),
                     "fill_rate": float(filled / placed) if placed else float("nan")})
    return df


# ------------------------------------------------------------------------------------- entry-order-type engine
ENTRY_TYPES = {"market": 0, "limit_touch": 1, "limit_pullback": 2, "stop_breakout": 3}


@njit(cache=True)
def _orders(bo, bh, bl, bc, ao, ah, al, ac, atr_, roll, target, latency, etype, k, ttl, slip_frac, slip_pips, pip,
            spread_mult, comm, swap_pips):
    n = len(bo)
    held = np.zeros(n); gross = np.zeros(n); sslip = np.zeros(n); commission = np.zeros(n)
    financing = np.zeros(n); net = np.zeros(n)
    n_att = 0; n_fill = 0; n_exp = 0
    h = 0.0
    pend = False; plevel = 0.0; pdir = 0.0; page = 0; ptgt = 0.0; pref = 0.0
    skip_tgt = np.nan
    for j in range(1, n):
        mid_c = (bc[j] + ac[j]) / 2.0
        mid_pc = (bc[j - 1] + ac[j - 1]) / 2.0
        mid_o = (bo[j] + ao[j]) / 2.0
        hs = spread_mult * (ao[j] - bo[j]) / 2.0
        slip = slip_frac * (ao[j] - bo[j]) + slip_pips * pip
        pnl = h * (mid_c / mid_pc - 1.0)
        cost_j = 0.0; com_j = 0.0
        d = j - 1 - latency
        tgt = target[d] if d >= 0 else 0.0
        # ---- 1) exits / reductions: always market at this bar's open
        if h != 0.0 and (tgt == 0.0 or np.sign(tgt) != np.sign(h) or abs(tgt) < abs(h)):
            new_h = tgt if np.sign(tgt) == np.sign(h) else 0.0
            dq = new_h - h
            F = mid_o + hs + slip if dq > 0 else mid_o - hs - slip
            pnl += dq * (mid_c - F) / F
            cost_j += abs(dq) * abs(F - mid_o) / mid_o
            com_j += abs(dq) * comm
            h = new_h
        if tgt == 0.0 or tgt != skip_tgt:
            if tgt != skip_tgt:
                skip_tgt = np.nan
        # ---- 2) entries / additions via the chosen order type
        wants = tgt != 0.0 and (h == 0.0 or np.sign(tgt) == np.sign(h)) and abs(tgt) > abs(h) and not (tgt == skip_tgt)
        if not wants:
            pend = False
        else:
            dirn = np.sign(tgt)
            if (not pend) or ptgt != tgt:
                pend = True; ptgt = tgt; page = 0; pdir = dirn; n_att += 1
                mid_d = (bc[d] + ac[d]) / 2.0 if d >= 0 else mid_pc
                a = atr_[d] if d >= 0 else np.nan
                pref = mid_d
                if etype == 2:
                    plevel = mid_d - dirn * k * a
                elif etype == 3:
                    plevel = mid_d + dirn * k * a
            if etype == 1:  # re-quote at the touch every bar
                if d >= 0:
                    spr = ac[d] - bc[d]
                    plevel = bc[d] + k * spr if dirn > 0 else ac[d] - k * spr
                    pref = (bc[d] + ac[d]) / 2.0
            filled = False; F = 0.0
            if etype == 0:
                F = mid_o + hs + slip if dirn > 0 else mid_o - hs - slip
                filled = True
            elif etype == 1 or etype == 2:
                if not np.isnan(plevel):
                    if dirn > 0 and al[j] < plevel - (spread_mult - 1.0) * (ac[j] - bc[j]):
                        F = min(plevel, ao[j]); filled = True
                    elif dirn < 0 and bh[j] > plevel + (spread_mult - 1.0) * (ac[j] - bc[j]):
                        F = max(plevel, bo[j]); filled = True
            else:
                if not np.isnan(plevel):
                    if dirn > 0 and ah[j] >= plevel:
                        F = max(plevel, ao[j]) + slip; filled = True
                    elif dirn < 0 and bl[j] <= plevel:
                        F = min(plevel, bo[j]) - slip; filled = True
            if filled:
                dq = tgt - h
                pnl += dq * (mid_c - F) / F
                cost_j += abs(dq) * (F - pref) / pref * dirn
                com_j += abs(dq) * comm
                h = tgt
                pend = False
                n_fill += 1
            else:
                page += 1
                if ttl > 0 and page >= ttl:
                    pend = False
                    skip_tgt = tgt      # give up on this signal until the target changes
                    n_exp += 1
        fin = abs(h) * roll[j] * swap_pips * pip / mid_c
        held[j] = h
        net[j] = pnl - com_j - fin
        sslip[j] = cost_j
        commission[j] = com_j
        financing[j] = fin
        gross[j] = net[j] + cost_j + com_j + fin
    return held, gross, sslip, commission, financing, net, n_att, n_fill, n_exp


def order_backtest(bars: pd.DataFrame, position: pd.Series, pip: float, cost: CostModel = BASE,
                   entry: str = "market", k: float = 0.0, ttl: int = 0, rollovers: np.ndarray | None = None,
                   atr_n: int = 14) -> pd.DataFrame:
    """Target-position backtest with a chosen ENTRY order type (exits/reductions are market orders):
      market          cross the spread at the next open (+ slippage)
      limit_touch     resting limit at the best bid (buy) / ask (sell) shifted k x spread toward mid,
                      re-quoted each bar; fills only on a strict trade-through
      limit_pullback  limit k x ATR better than the decision mid (fixed level); fills on trade-through
      stop_breakout   stop k x ATR beyond the decision mid in the signal direction; fills when traded
                      through, at the worse of level and open, plus slippage
    ``ttl`` > 0: an unfilled entry is cancelled after ttl bars and the signal is skipped until the target
    changes. Columns match vector_backtest; order statistics in ``.attrs``."""
    from fxlab.features._util import atr as _atr
    pos = position.reindex(bars.index).fillna(0.0).clip(-1, 1).to_numpy(float)
    if rollovers is None:
        rollovers = rollover_counts(bars.index)
    a = _atr(bars, atr_n).to_numpy(float)
    out = _orders(*_arrays(bars), a, rollovers.astype(np.float64), pos, int(cost.latency_bars), ENTRY_TYPES[entry],
                  float(k), int(ttl), cost.slip_spread_frac, cost.slip_fixed_pips, pip, float(max(cost.spread_mult, 1.0)),
                  cost.commission_bps_per_side * 1e-4, cost.swap_pips_per_night)
    held, gross, sslip, com, fin, net, att, fill, exp = out
    df = pd.DataFrame({"position": held, "gross": gross, "spread_slip": sslip, "commission": com, "financing": fin,
                       "net": net}, index=bars.index)
    df.attrs.update({"entry_attempts": int(att), "entry_fills": int(fill), "entry_expired": int(exp),
                     "fill_rate": float(fill / att) if att else float("nan")})
    return df
