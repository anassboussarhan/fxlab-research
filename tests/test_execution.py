"""Execution simulator correctness on hand-constructed bars."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxlab.execution.costs import CostModel, ZERO
from fxlab.execution.simulator import vector_backtest, simulate_trades, rollover_counts

PIP = 0.0001
NOCOST = CostModel(spread_mult=1.0, slip_spread_frac=0.0, slip_fixed_pips=0.0, commission_bps_per_side=0.0,
                   swap_pips_per_night=0.0, latency_bars=0, name="spread_only")


def bars_from(mids, spread=0.0002, highs=None, lows=None, start="2015-01-06 00:00"):
    n = len(mids)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    o = np.r_[mids[0], mids[:-1]]
    c = np.asarray(mids, float)
    h = np.maximum(o, c) if highs is None else np.asarray(highs, float)
    l = np.minimum(o, c) if lows is None else np.asarray(lows, float)
    df = pd.DataFrame(index=idx)
    for k, v in zip("ohlc", (o, h, l, c)):
        df[f"bid_{k}"] = v - spread / 2
        df[f"ask_{k}"] = v + spread / 2
    df["close"] = c
    return df


def test_vector_long_hold_equals_open_to_open_minus_spread():
    mids = np.linspace(1.10, 1.12, 50)
    b = bars_from(mids)
    pos = pd.Series(0.0, index=b.index); pos.iloc[5:30] = 1.0
    r = vector_backtest(b, pos, PIP, NOCOST, rollovers=np.zeros(len(b)))
    mo = (b["bid_o"] + b["ask_o"]) / 2
    gross_expected = sum(mo.iloc[j + 1] / mo.iloc[j] - 1 for j in range(6, 31))  # held from open 6 to open 31
    assert r["gross"].sum() == pytest.approx(gross_expected, rel=1e-9)
    # one entry + one exit, each paying half the 2-pip spread relative to mid
    assert r["spread_slip"].sum() == pytest.approx(2 * 0.0001 / mo.iloc[6], rel=0.02)
    assert r.loc[r.index[5], "position"] == 0 and r.loc[r.index[6], "position"] == 1  # decision at 5 executes at 6


def test_vector_latency_delays_execution():
    b = bars_from(np.linspace(1.1, 1.2, 30))
    pos = pd.Series(0.0, index=b.index); pos.iloc[10:] = 1
    c = CostModel(latency_bars=2)
    r = vector_backtest(b, pos, PIP, c, rollovers=np.zeros(len(b)))
    assert r["position"].iloc[12] == 0 and r["position"].iloc[13] == 1


def _orders(b, side_at, side, stop, tp=np.nan, max_hold=np.nan, trail=np.nan):
    o = pd.DataFrame({"side": 0.0, "stop_dist": stop, "tp_dist": tp, "max_hold": max_hold, "trail_dist": trail,
                      "size": 1.0}, index=b.index)
    o.iloc[side_at, 0] = side
    return o


def test_long_entry_at_ask_and_stop_fill():
    mids = [1.1000] * 5 + [1.0990, 1.0950, 1.0950]
    lows = np.array(mids) - 0.0001
    lows[6] = 1.0940
    b = bars_from(mids, lows=lows)
    res = simulate_trades(b, _orders(b, 1, 1, 0.0030), PIP, NOCOST, exit_on_flat=False, rollovers=np.zeros(len(b)))
    t = res.trades.iloc[0]
    assert t["entry_px"] == pytest.approx(b["ask_o"].iloc[2])       # decided at bar 1 -> ask at open of bar 2
    assert t["reason"] == "stop"
    assert t["exit_px"] == pytest.approx(t["entry_px"] - 0.0030)    # no slippage in NOCOST
    # telescoping: sum of bar returns equals realised trade return
    assert res.bars["net"].sum() == pytest.approx(t["ret_price"], abs=1e-12)


def test_gap_through_stop_fills_at_worse_open():
    mids = [1.1000] * 5 + [1.0900, 1.0900]
    b = bars_from(mids)  # bar 5 opens at 1.1000? -> force a gap: set bar 5 open below stop
    b.loc[b.index[5], ["bid_o", "ask_o"]] = [1.0899, 1.0901]
    b.loc[b.index[5], ["bid_l", "ask_l"]] = [1.0895, 1.0897]
    res = simulate_trades(b, _orders(b, 1, 1, 0.0030), PIP, NOCOST, exit_on_flat=False, rollovers=np.zeros(len(b)))
    t = res.trades.iloc[0]
    assert t["reason"] == "stop" and t["exit_px"] == pytest.approx(1.0899)


def test_stop_and_target_same_bar_is_conservative():
    mids = [1.1] * 6
    highs = np.array(mids) + 0.0001; lows = np.array(mids) - 0.0001
    highs[3], lows[3] = 1.1100, 1.0900  # touches both
    b = bars_from(mids, highs=highs, lows=lows)
    res = simulate_trades(b, _orders(b, 1, 1, 0.0050, tp=0.0050), PIP, NOCOST, exit_on_flat=False,
                          rollovers=np.zeros(len(b)))
    assert res.trades.iloc[0]["reason"] == "stop"


def test_short_target_on_ask_and_time_stop():
    mids = [1.1000] * 3 + [1.0990, 1.0980, 1.0970, 1.0960, 1.0950]
    b = bars_from(mids)
    res = simulate_trades(b, _orders(b, 1, -1, 0.0050, tp=0.0025), PIP, NOCOST, exit_on_flat=False,
                          rollovers=np.zeros(len(b)))
    t = res.trades.iloc[0]
    assert t["entry_px"] == pytest.approx(b["bid_o"].iloc[2])
    assert t["reason"] == "target" and t["exit_px"] == pytest.approx(t["entry_px"] - 0.0025)
    res2 = simulate_trades(b, _orders(b, 1, -1, 0.0050, max_hold=2), PIP, NOCOST, exit_on_flat=False,
                           rollovers=np.zeros(len(b)))
    t2 = res2.trades.iloc[0]
    assert t2["reason"] == "time" and t2["bars_held"] == 2 and t2["exit_px"] == pytest.approx(b["ask_o"].iloc[4])


def test_trailing_stop_locks_profit():
    mids = [1.10, 1.10, 1.101, 1.105, 1.110, 1.108, 1.100]
    b = bars_from(mids, spread=0.00002)
    res = simulate_trades(b, _orders(b, 0, 1, 0.0050, trail=0.0030), PIP, NOCOST, exit_on_flat=False,
                          rollovers=np.zeros(len(b)))
    t = res.trades.iloc[0]
    assert t["reason"] == "trail" and t["exit_px"] > t["entry_px"]


def test_commission_and_financing_charged():
    mids = [1.1] * 30
    b = bars_from(mids, spread=0.0)
    c = CostModel(spread_mult=0, slip_spread_frac=0, slip_fixed_pips=0, commission_bps_per_side=1.0,
                  swap_pips_per_night=1.0)
    roll = np.zeros(len(b)); roll[10] = 3
    res = simulate_trades(b, _orders(b, 1, 1, 0.01, max_hold=20), PIP, c, exit_on_flat=False, rollovers=roll)
    t = res.trades.iloc[0]
    assert t["financing"] == pytest.approx(3 * 0.0001 / 1.1)
    assert res.bars["net"].sum() == pytest.approx(-(2e-4 + 3 * 0.0001 / 1.1), rel=1e-6)


def test_rollover_counts_wednesday_triple_and_weekend():
    idx = pd.date_range("2015-01-05 20:00", "2015-01-12 20:00", freq="h", tz="UTC")  # Mon..next Mon (before its rollover)
    idx = idx[~((idx.dayofweek == 5) | ((idx.dayofweek == 4) & (idx.hour >= 22)) | ((idx.dayofweek == 6) & (idx.hour < 22)))]
    cnt = rollover_counts(idx)
    assert cnt.sum() == 7  # Mon, Tue, Wed(x3), Thu, Fri rollovers within the week
    ny = idx.tz_convert("America/New_York")
    wed = (ny.dayofweek == 2) & (ny.hour == 16)
    assert cnt[np.nonzero(wed)[0][0]] == 3


# ------------------------------------------------------------------------------------------- passive (limit) engine
from fxlab.execution.simulator import passive_backtest  # noqa: E402

PCOST = CostModel(spread_mult=1.0, slip_spread_frac=0.0, slip_fixed_pips=0.0, commission_bps_per_side=0.0,
                  swap_pips_per_night=0.0, latency_bars=0, name="passive_test")


def test_passive_buy_fills_only_on_trade_through():
    mids = [1.1000, 1.1000, 1.1000, 1.0995, 1.1005]
    sp = 0.0002
    b = bars_from(mids, spread=sp)
    pos = pd.Series(0.0, index=b.index); pos.iloc[1:] = 1.0     # decide long at close of bar 1
    # bar 2: ask low = 1.1000-0.0001 = 1.0999 > limit 1.0999? limit = bid_c[1] = 1.0999 -> not strictly through
    r = passive_backtest(b, pos, PIP, PCOST, rollovers=np.zeros(len(b)))
    assert r["position"].iloc[2] == 0.0
    # bar 3 closes at 1.0995: ask low 1.0996 < re-quoted limit bid_c[2] = 1.0999 -> filled at 1.0999
    assert r["position"].iloc[3] == 1.0
    mid3 = 1.0995
    assert r["net"].iloc[3] == pytest.approx((mid3 - 1.0999) / 1.0999, rel=1e-9)
    assert r["spread_slip"].iloc[3] < 0          # earned half the spread relative to decision mid
    assert r.attrs["orders_filled"] == 1 and r.attrs["orders_placed"] == 2


def test_passive_never_filled_without_fallback_and_market_fallback():
    mids = [1.1000, 1.1001, 1.1002, 1.1003, 1.1004, 1.1005]   # rising: a buy limit at the bid is never hit
    b = bars_from(mids, spread=0.0002)
    pos = pd.Series(0.0, index=b.index); pos.iloc[0:] = 1.0
    r = passive_backtest(b, pos, PIP, PCOST, rollovers=np.zeros(len(b)))
    assert (r["position"] == 0).all() and r.attrs["orders_filled"] == 0
    r2 = passive_backtest(b, pos, PIP, PCOST, max_wait=2, rollovers=np.zeros(len(b)))
    j = int(np.argmax(r2["position"].to_numpy() != 0))
    assert j == 3 and r2.attrs["market_fallbacks"] == 1       # waited bars 1-2, crossed at open of bar 3
    assert r2["spread_slip"].iloc[3] == pytest.approx(0.0001 / ((b["bid_o"].iloc[3] + b["ask_o"].iloc[3]) / 2), rel=1e-6)


def test_passive_stress_requires_deeper_trade_through():
    mids = [1.1000, 1.1000, 1.09975]   # ask low of bar 2 = 1.09985: 0.5 pip through the 1.0999 bid, < 1 spread
    b = bars_from(mids, spread=0.0002)
    pos = pd.Series(0.0, index=b.index); pos.iloc[1:] = 1.0
    assert passive_backtest(b, pos, PIP, PCOST, rollovers=np.zeros(3))["position"].iloc[2] == 1.0
    stressed = CostModel(spread_mult=2.0, slip_spread_frac=0, slip_fixed_pips=0, commission_bps_per_side=0,
                         swap_pips_per_night=0)
    assert passive_backtest(b, pos, PIP, stressed, rollovers=np.zeros(3))["position"].iloc[2] == 0.0


def test_passive_round_trip_accounting():
    mids = [1.1000, 1.1000, 1.0990, 1.1010, 1.1030, 1.1000]
    b = bars_from(mids, spread=0.0002)
    pos = pd.Series([0, 1, 1, 0, 0, 0], index=b.index, dtype=float)
    c = CostModel(spread_mult=1.0, slip_spread_frac=0, slip_fixed_pips=0, commission_bps_per_side=0.5,
                  swap_pips_per_night=0)
    r = passive_backtest(b, pos, PIP, c, rollovers=np.zeros(len(b)))
    # buy limit at bid_c[1]=1.0999 fills in bar 2 (ask low 1.0991); target 0 decided at close of bar 3 ->
    # sell limit at ask_c[3]=1.1011 fills in bar 4 (bid high 1.1029)
    assert r["position"].tolist() == [0, 0, 1, 1, 0, 0]
    expected = (1.1011 - 1.0999) / 1.0999 - 2 * 0.5e-4
    assert r["net"].sum() == pytest.approx(expected, abs=5e-6)  # bar-relative marking: 2nd-order approximation


# ------------------------------------------------------------------------------------------- entry-type engine
from fxlab.execution.simulator import order_backtest  # noqa: E402
from fxlab.data.store import add_mid  # noqa: E402


def _ob(mids, spread=0.0002, highs=None, lows=None):
    b = bars_from(mids, spread=spread, highs=highs, lows=lows)
    b["spread_mean"] = spread
    b["n_ticks"] = 100
    return add_mid(b)


def test_limit_pullback_fills_at_level_or_expires():
    mids = [1.1000] * 20 + [1.1000, 1.0990, 1.0985, 1.1000]
    lows = np.array(mids) - 0.0002
    b = _ob(mids, lows=lows)
    pos = pd.Series(0.0, index=b.index); pos.iloc[20:] = 1.0
    from fxlab.features._util import atr
    a = atr(b, 14).iloc[20]
    lvl = 1.1000 - 0.5 * a
    r = order_backtest(b, pos, PIP, PCOST, entry="limit_pullback", k=0.5, ttl=10, rollovers=np.zeros(len(b)))
    j = int(np.argmax(r["position"].to_numpy() != 0))
    assert b["ask_l"].iloc[j] < lvl and (b["ask_l"].iloc[21:j] >= lvl).all()
    assert r.attrs["entry_fills"] == 1
    r2 = order_backtest(b, pos, PIP, PCOST, entry="limit_pullback", k=50.0, ttl=2, rollovers=np.zeros(len(b)))
    assert (r2["position"] == 0).all() and r2.attrs["entry_expired"] == 1   # level never reached -> skipped


def test_stop_breakout_triggers_and_gap_fills_at_open():
    mids = [1.1000] * 20 + [1.1000, 1.1000, 1.1050, 1.1060]
    b = _ob(mids, highs=np.array(mids) + 0.0005, lows=np.array(mids) - 0.0005)    # ATR ~ 10 pips
    b.loc[b.index[22], ["bid_o", "ask_o"]] = [1.1039, 1.1041]                 # gap open above the stop
    b.loc[b.index[22], ["bid_h", "ask_h"]] = [1.1051, 1.1053]
    b = add_mid(b.drop(columns=["open", "high", "low", "close", "spread", "volume"]))
    pos = pd.Series(0.0, index=b.index); pos.iloc[20:] = 1.0
    c = CostModel(spread_mult=1.0, slip_spread_frac=0.0, slip_fixed_pips=0.1, commission_bps_per_side=0.0,
                  swap_pips_per_night=0.0)
    r = order_backtest(b, pos, PIP, c, entry="stop_breakout", k=1.0, ttl=5, rollovers=np.zeros(len(b)))
    j = int(np.argmax(r["position"].to_numpy() != 0))
    assert j == 22
    F = 1.1041 + 0.00001                                                       # worse of level and open + slip
    assert r["net"].iloc[22] == pytest.approx((b["close"].iloc[22] - F) / F, rel=1e-6)


def test_order_engine_exits_at_market_and_market_entry():
    mids = [1.1000] * 20 + [1.1010, 1.1020, 1.1030, 1.1030]
    b = _ob(mids)
    pos = pd.Series(0.0, index=b.index); pos.iloc[19:21] = 1.0
    r = order_backtest(b, pos, PIP, PCOST, entry="market", rollovers=np.zeros(len(b)))
    assert r["position"].iloc[20] == 1 and r["position"].iloc[21] == 1 and r["position"].iloc[22] == 0
    entry_px, exit_px = b["ask_o"].iloc[20], b["bid_o"].iloc[22]
    assert r["net"].sum() == pytest.approx((exit_px - entry_px) / entry_px, abs=3e-6)


def test_rate_financing_sign_and_markup():
    from fxlab.execution.costs import BASE
    """A long position earns (carry - markup)/360 per rollover; a short pays (carry + markup)/360."""
    from fxlab.data.rates import SWAP_MARKUP_PA
    idx = pd.date_range("2012-01-02 22:00", periods=48, freq="1h", tz="UTC")
    b = pd.DataFrame(index=idx)
    for side, sgn in (("bid", -1), ("ask", 1)):
        for k in "ohlc":
            b[f"{side}_{k}"] = 1.0 + sgn * 5e-5
    carry = np.full(len(idx), 0.03)
    roll = rollover_counts(idx)
    for pos, expect in ((1.0, (0.03 - SWAP_MARKUP_PA) / 360), (-1.0, -(0.03 + SWAP_MARKUP_PA) / 360)):
        bt = vector_backtest(b, pd.Series(pos, index=idx), 1e-4, BASE, roll, carry=carry)
        held_rolls = (roll * np.abs(bt["position"].to_numpy())).sum()
        assert held_rolls >= 1
        assert np.isclose(-bt["financing"].sum(), expect * held_rolls)


def test_rates_are_point_in_time():
    from fxlab.data.rates import rate_diff, monthly
    m = monthly()
    idx = pd.DatetimeIndex(["2015-01-15", "2015-01-31 23:00", "2015-02-01 00:00"], tz="UTC")
    d = rate_diff("AUDJPY", idx)
    dec, jan = m.loc["2014-12-01"], m.loc["2015-01-01"]
    assert np.isclose(d.iloc[0], dec["AUD"] - dec["JPY"]) and np.isclose(d.iloc[1], dec["AUD"] - dec["JPY"])
    assert np.isclose(d.iloc[2], jan["AUD"] - jan["JPY"])          # January's average only from 1 February
    assert m.index.max() < pd.Timestamp("2026-01-01", tz="UTC")     # no holdout-period rates stored


def _trend_bars(path, spread=1e-4):
    idx = pd.date_range("2012-01-02", periods=len(path), freq="1h", tz="UTC")
    b = pd.DataFrame(index=idx)
    mid = np.asarray(path, float)
    for side, sgn in (("bid", -1), ("ask", 1)):
        b[f"{side}_o"] = mid + sgn * spread / 2
        b[f"{side}_h"] = mid + sgn * spread / 2 + 2e-4
        b[f"{side}_l"] = mid + sgn * spread / 2 - 2e-4
        b[f"{side}_c"] = mid + sgn * spread / 2
    return b


def test_breakeven_stop_turns_a_round_trip_into_a_scratch():
    from fxlab.execution.costs import BASE
    # price rises 1.5R in favour, then declines steadily (no gaps) far below the entry
    path = [1.1000] * 3 + [1.1000 + 0.0005 * i for i in range(1, 8)] + [1.1035 - 0.0005 * i for i in range(1, 20)]
    b = _trend_bars(path)
    sig = pd.DataFrame({"side": 0.0, "stop_dist": 0.002, "tp_dist": np.nan, "max_hold": np.nan,
                        "trail_dist": np.nan, "size": 1.0}, index=b.index)
    sig.iloc[1, 0] = 1.0
    plain = simulate_trades(b, sig, 1e-4, BASE, exit_on_flat=False)
    be = simulate_trades(b, sig.assign(be_mult=1.0), 1e-4, BASE, exit_on_flat=False)
    assert plain.trades["net_ret"].iloc[0] < -0.0015          # full stop loss
    assert abs(be.trades["net_ret"].iloc[0]) < 0.0005          # stopped near entry (costs only)


def test_partial_take_profit_banks_half_at_1r():
    from fxlab.execution.costs import BASE
    path = [1.1000] * 3 + [1.1000 + 0.0005 * i for i in range(1, 6)] + [1.1025 - 0.0005 * i for i in range(1, 20)]
    b = _trend_bars(path)
    base = pd.DataFrame({"side": 0.0, "stop_dist": 0.002, "tp_dist": np.nan, "max_hold": np.nan,
                         "trail_dist": np.nan, "size": 1.0}, index=b.index)
    base.iloc[1, 0] = 1.0
    full = simulate_trades(b, base, 1e-4, BASE, exit_on_flat=False).bars["net"].sum()
    part = simulate_trades(b, base.assign(part_mult=1.0, part_frac=0.5), 1e-4, BASE, exit_on_flat=False).bars["net"].sum()
    # half closed at +1R (+0.002) instead of riding to the stop: improvement of about 0.5 x (0.002 + 0.002)/1.1
    assert part - full == pytest.approx(0.5 * 0.004 / 1.1, rel=0.25)


def test_confirmation_and_session_entries_are_causal():
    from fxlab.execution.entries import confirm_entries, session_entries
    rng = np.random.default_rng(5)
    idx = pd.date_range("2013-03-04", periods=800, freq="1h", tz="UTC")
    close = pd.Series(1.3 + np.cumsum(rng.normal(0, 1e-3, len(idx))), index=idx)
    side = pd.Series(np.sign(np.sin(np.arange(len(idx)) / 9.0)), index=idx)
    for f in (lambda s, c: confirm_entries(s, c, 1), lambda s, c: session_entries(s)):
        full = f(side, close)
        cut = 500
        part = f(side.iloc[:cut], close.iloc[:cut])
        assert np.array_equal(full.iloc[:cut].to_numpy(), part.to_numpy())
        assert (full.abs() <= 1).all()
    c = confirm_entries(side, close, 1)
    # an entry is never taken on the signal bar itself
    starts = (side != side.shift(1)) & (side != 0)
    assert not ((c == side) & starts & (c != 0)).any()
    sess = session_entries(side)
    nxt = (idx + pd.Timedelta(hours=1)).tz_convert("Europe/London")
    new_entries = (sess != 0) & (sess != sess.shift(1))
    assert ((nxt.hour[new_entries.to_numpy()] >= 7) & (nxt.hour[new_entries.to_numpy()] < 17)).all()
