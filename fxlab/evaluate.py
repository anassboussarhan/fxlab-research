"""Run a catalog strategy on bars through the realistic execution layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.registry import CATALOG
from fxlab.data.instruments import get as instrument
from fxlab.execution.costs import CostModel, BASE
from fxlab.execution.risk import RiskConfig, build_orders
from fxlab.execution.simulator import vector_backtest, passive_backtest, order_backtest, simulate_trades, rollover_counts
from fxlab.validation.metrics import performance


def run_signals(sid: str, bars: pd.DataFrame, panel=None, symbol="EURUSD", params=None) -> pd.DataFrame:
    e = CATALOG[sid]
    p = dict(e.params) | dict(params or {})
    if e.multi_asset:
        return e.fn(bars, panel=panel, symbol=symbol, **p)
    if e.extra.get("ml"):
        return e.fn(bars, panel=panel, symbol=symbol, **p)
    return e.fn(bars, **p)


def risk_config(e, override: dict | None = None) -> RiskConfig:
    r = dict(e.extra.get("risk", {})) | dict(override or {})
    return RiskConfig(**r)


def evaluate(sid: str, bars: pd.DataFrame, symbol: str, panel=None, params=None, cost: CostModel = BASE,
             risk_override: dict | None = None, signals: pd.DataFrame | None = None,
             mode_override: str | None = None, exec_cfg: dict | None = None, financing: str | None = None,
             m1: pd.DataFrame | None = None) -> dict:
    """Returns {'returns': per-bar net returns, 'trades': trade log or None, 'metrics': dict, 'signals': frame}.
    financing='rates': overnight financing from point-in-time interest-rate differentials (vector engine).
    m1: the instrument's M1 bars; when given, the trade engine replays the minute path of any bar whose range
    contains both the stop and the target (otherwise stop-first)."""
    e = CATALOG[sid]
    pip = instrument(symbol).pip
    sig = signals if signals is not None else run_signals(sid, bars, panel, symbol, params)
    mode = mode_override or e.extra.get("mode", "state")
    state_with_stops = mode == "trade" and e.extra.get("mode", "state") == "state"
    roll = rollover_counts(bars.index)
    if mode == "state":
        if exec_cfg and exec_cfg.get("type") in ("confirm", "session"):
            from fxlab.execution.entries import confirm_entries, session_entries
            side_in = sig["side"]
            if exec_cfg["type"] == "confirm":
                side_x = confirm_entries(side_in, bars["close"], exec_cfg.get("k", 1))
            else:
                side_x = session_entries(side_in, exec_cfg.get("tz", "Europe/London"), exec_cfg.get("start", 7.0),
                                         exec_cfg.get("end", 17.0))
            sig_exec = sig.copy()          # executed positions; the returned `signals` stay untransformed so that
            sig_exec["side"] = side_x      # re-simulations (stress tests) do not apply the transform twice
            exec_cfg = None
        else:
            sig_exec = sig
        if exec_cfg and exec_cfg.get("type") == "scale_in":
            # half the position at market, half via a pullback limit (k x ATR, ttl bars); engines are linear in size
            b1 = vector_backtest(bars, sig_exec["side"], pip, cost, roll)
            b2 = order_backtest(bars, sig_exec["side"], pip, cost, entry="limit_pullback", k=exec_cfg.get("k", 0.5),
                                ttl=exec_cfg.get("ttl", 6), rollovers=roll)
            bt = (b1[["position", "gross", "spread_slip", "commission", "financing", "net"]] * 0.5
                  + b2[["position", "gross", "spread_slip", "commission", "financing", "net"]] * 0.5)
            bt.attrs.update({f"pullback_{k}": v for k, v in b2.attrs.items()})
        elif exec_cfg and exec_cfg.get("type") == "order":
            bt = order_backtest(bars, sig_exec["side"], pip, cost, entry=exec_cfg.get("entry", "market"),
                                k=exec_cfg.get("k", 0.0), ttl=exec_cfg.get("ttl", 0), rollovers=roll)
        elif exec_cfg and exec_cfg.get("type") == "passive":
            bt = passive_backtest(bars, sig_exec["side"], pip, cost, offset=exec_cfg.get("offset", 0.0),
                                  max_wait=exec_cfg.get("max_wait", 0), rollovers=roll)
        else:
            carry = None
            if financing == "rates":
                from fxlab.data.rates import has_rates, rate_diff
                if has_rates(symbol):
                    carry = rate_diff(symbol, bars.index).to_numpy()
            bt = vector_backtest(bars, sig_exec["side"], pip, cost, roll, carry=carry)
        r = bt["net"]
        turnover = bt["position"].diff().abs()
        costs = bt["spread_slip"] + bt["commission"] + bt["financing"]
        trades = _state_trades(bt)
        m = performance(r, trades, turnover, costs)
        m["gross_sharpe_contrib"] = float(bt["gross"].sum())
        m.update({k: v for k, v in bt.attrs.items()})
        return {"returns": r, "trades": trades, "metrics": m, "signals": sig, "detail": bt}
    rc = risk_config(e, risk_override)
    et = (exec_cfg or {}).get("type")
    side_in = sig["side"]
    if et in ("confirm", "session"):      # entry filters act on the decision signal before orders are built
        from fxlab.execution.entries import confirm_entries, session_entries
        if et == "confirm":
            side_in = confirm_entries(side_in, bars["close"], exec_cfg.get("k", 1))
        else:
            side_in = session_entries(side_in, exec_cfg.get("tz", "Europe/London"), exec_cfg.get("start", 7.0),
                                      exec_cfg.get("end", 17.0))
    orders = build_orders(bars, side_in, rc, pip)
    for col in ("stop_dist", "tp_dist", "max_hold", "trail_dist"):
        if col in sig and sig[col].notna().any():
            orders[col] = sig[col].where(sig[col].notna(), orders[col])
    if "stop_dist" in sig and sig["stop_dist"].notna().any():
        from fxlab.execution.risk import size_fixed_fractional
        orders["size"] = size_fixed_fractional(orders["stop_dist"], bars["close"], rc.risk_per_trade, rc.max_leverage)
    if state_with_stops:  # a continuous-position family run with stops: a flat/opposite signal also exits
        orders["side"] = np.sign(orders["side"])
    def sim(entry="market", k=0.0, ttl=0):
        return simulate_trades(bars, orders, pip, cost, exit_on_flat=state_with_stops, rollovers=roll,
                               entry=entry, k=k, ttl=ttl, m1=m1)
    if et == "order":
        res = sim(exec_cfg.get("entry", "market"), exec_cfg.get("k", 0.0), exec_cfg.get("ttl", 0))
        net, trades = res.bars["net"], res.trades
    elif et == "passive":                 # resting limit at the decision close (trade engine equivalent)
        res = sim("limit_touch", 0.0, exec_cfg.get("max_wait", 0) or 3)
        net, trades = res.bars["net"], res.trades
    elif et == "scale_in":                # half at market, half via a pullback limit; the engine is linear in size
        r1, r2 = sim(), sim("limit_pullback", exec_cfg.get("k", 0.5), exec_cfg.get("ttl", 6))
        net = 0.5 * r1.bars["net"] + 0.5 * r2.bars["net"]
        t1, t2 = r1.trades.assign(leg="market"), r2.trades.assign(leg="pullback")
        for t in (t1, t2):
            t["size"] *= 0.5
            t["pnl_frac_equity"] *= 0.5
        trades = pd.concat([t1, t2]).sort_values("entry_time").reset_index(drop=True)
    else:
        res = sim()
        net, trades = res.bars["net"], res.trades
    m = performance(net, trades)
    return {"returns": net, "trades": trades, "metrics": m, "signals": sig, "orders": orders}


def _state_trades(bt: pd.DataFrame) -> pd.DataFrame:
    """Derive round trips (contiguous non-zero same-sign holdings) from a vector backtest."""
    pos = np.sign(bt["position"].to_numpy())
    net = bt["net"].to_numpy()
    rows, start = [], None
    for i in range(len(pos)):
        if start is None and pos[i] != 0:
            start = i
        elif start is not None and (pos[i] != pos[start]):
            rows.append((start, i - 1))
            start = i if pos[i] != 0 else None
    if start is not None:
        rows.append((start, len(pos) - 1))
    idx = bt.index
    out = [{"entry_time": idx[a], "exit_time": idx[b], "side": pos[a], "bars_held": b - a + 1,
            "pnl_frac_equity": float(np.prod(1 + net[a:b + 1]) - 1)} for a, b in rows]
    return pd.DataFrame(out)
