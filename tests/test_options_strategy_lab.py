import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import options_strategy_lab as L  # noqa: E402


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _bs(cp, S, K, T, v):
    sd = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    if cp == "C":
        return S * _ncdf(d1) - K * _ncdf(d1 - sd), _ncdf(d1)
    return K * _ncdf(sd - d1) - S * _ncdf(-d1), _ncdf(d1) - 1


def _chain(S=100.0, iv=0.2, half=0.05, path=None):
    days = pd.bdate_range("2024-01-02", "2024-03-15")
    exp = pd.Timestamp("2024-03-15")                     # third Friday of March 2024
    rows = []
    for i, d in enumerate(days):
        if d >= exp:
            continue
        s = path[i] if path is not None else S
        T = (exp - d).days / 365
        for K in range(70, 131):
            for cp in ("P", "C"):
                p, dl = _bs(cp, s, K, T, iv)
                if p < 0.06:
                    continue
                rows.append({"date": d, "expiration": exp, "strike": float(K), "cp": cp, "bid": p - half,
                             "ask": p + half, "iv": iv, "delta": dl})
    C = pd.DataFrame(rows)
    C["mid"] = (C.bid + C.ask) / 2
    C["half"] = (C.ask - C.bid) / 2
    px = pd.Series([path[i] if path is not None else S for i in range(len(days))], index=days)
    und = L.Underlying(px, px)
    return C, und, exp


def test_short_put_held_expires_worthless():
    C, und, exp = _chain()
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    legs, front = L.build_position(bk, d, L.STRUCTS["SP30"], 45, 100.0)
    assert front == exp
    k = [l for l in legs if l[0] == "O"][0][3]
    assert k < 100
    T = L.cycle_paths(bk, d, legs, front)
    r = L.evaluate(bk, d, legs, front, T, "H", 0.75)
    row = C[(C.date == d) & (C.strike == k) & (C.cp == "P")].iloc[0]
    expect = (row.mid - 0.75 * row.half - L.COMM) / 100
    assert abs(r["pnl"] - expect) < 1e-9
    assert not r["early"]


def test_take_profit_closes_before_expiry():
    C, und, exp = _chain()
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    legs, front = L.build_position(bk, d, L.STRUCTS["SP30"], 45, 100.0)
    T = L.cycle_paths(bk, d, legs, front)
    r = L.evaluate(bk, d, legs, front, T, "T50", 0.75)
    assert r["early"] and r["exit"] < exp
    assert r["pnl"] >= 0.5 * r["credit"] - 1e-12


def test_spread_wings_ordered_and_loss_capped():
    path = list(np.linspace(100, 75, 55))               # a steady crash
    C, und, exp = _chain(path=path)
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    S0 = float(und.px(d))
    legs, front = L.build_position(bk, d, L.STRUCTS["PCS30_16"], 45, S0)
    ks = {l[4]: l[3] for l in legs}
    assert ks[1] < ks[-1]                                # long wing below the short put
    T = L.cycle_paths(bk, d, legs, front)
    r = L.evaluate(bk, d, legs, front, T, "H", 1.0)
    width = (ks[-1] - ks[1]) / S0
    assert r["pnl"] >= -width - 1e-9                     # loss never beyond the width


def test_bh_and_nw():
    q = L.bh([0.001, 0.02, 0.5, np.nan])
    assert q[0] <= q[1] <= q[2] and np.isnan(q[3])
    x = np.r_[np.ones(20), -np.ones(20)] * 0.01 + 0.002
    assert np.isfinite(L.nw_t(x))


def test_split_day_is_a_corporate_action():
    idx = pd.bdate_range("2024-01-01", periods=30)
    raw = pd.Series(np.r_[np.full(15, 1000.0), np.full(15, 100.0)], index=idx)      # 10-for-1 split on day 16
    adj = pd.Series(100.0, index=idx)
    u = L.Underlying(raw, adj)
    assert list(u.events) == [idx[15]]
    assert u.has_event(idx[10], idx[20]) and not u.has_event(idx[16], idx[29])


def test_mark_to_market_adds_up_to_realised():
    path = list(np.linspace(100, 90, 55))
    C, und, exp = _chain(path=path)
    bk = L.Book(C, und)
    d = pd.Timestamp("2024-01-31")
    S0 = float(und.px(d))
    for name in ("SP30", "IC16_05", "CC30"):
        legs, front = L.build_position(bk, d, L.STRUCTS[name], 45, S0)
        T = L.cycle_paths(bk, d, legs, front)
        for mg in ("H", "T50"):
            r = L.evaluate(bk, d, legs, front, T, mg, 0.75)
            inc = L.mtm_series(bk, d, legs, T, r["exit"], S0, r["pnl"])
            assert abs(inc.sum() - r["pnl"]) < 1e-9
