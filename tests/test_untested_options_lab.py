import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import options_strategy_lab as L  # noqa: E402
import untested_options_lab as U  # noqa: E402


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _bs(cp, S, K, T, v):
    sd = v * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sd * sd) / sd
    if cp == "C":
        return S * _ncdf(d1) - K * _ncdf(d1 - sd), _ncdf(d1)
    return K * _ncdf(sd - d1) - S * _ncdf(-d1), _ncdf(d1) - 1


def _chain(path_fn=lambda i: 100.0, iv=0.2, half=0.02, start="2024-01-02", end="2024-07-19"):
    days = pd.bdate_range(start, end)
    exps = [e for e in L.third_fridays(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=70))]
    rows, px = [], []
    for i, d in enumerate(days):
        s = path_fn(i)
        px.append(s)
        for e in exps:
            if not (d < e <= d + pd.Timedelta(days=95)):
                continue
            T = (e - d).days / 365
            for K in range(60, 141):
                for cp in ("P", "C"):
                    p, dl = _bs(cp, s, K, T, iv)
                    if p < 0.05:
                        continue
                    rows.append({"date": d, "expiration": e, "strike": float(K), "cp": cp, "bid": p - half,
                                 "ask": p + half, "iv": iv, "delta": dl})
    C = pd.DataFrame(rows)
    C["mid"] = (C.bid + C.ask) / 2
    C["half"] = (C.ask - C.bid) / 2
    s = pd.Series(px, index=days)
    und = L.Underlying(s, s)
    return C[C.expiration <= pd.Timestamp(end)], und


def test_continuous_hold_redeploys_on_expiry_and_increments_add_up():
    C, und = _chain()
    bk = L.Book(C, und)
    vix = pd.Series(18.0, index=pd.bdate_range("2023-12-01", "2024-08-01"))
    cyc, incs = U.continuous_book(bk, U.BOOK_STRUCTS["STRG16"], "H", vix)
    assert len(cyc) >= 2
    for j in range(1, len(cyc)):          # next entry on the expiry close, or within 5 days if no 35-57-day expiry
        assert 0 <= (cyc.entry.iloc[j] - cyc.exit.iloc[j - 1]).days <= 7
    for j, s in enumerate(incs):
        assert abs(s.sum() - cyc.pnl.iloc[j]) < 1e-9
    assert (cyc.pnl > 0).all()                                    # flat price: both shorts expire worthless
    assert abs(cyc.w_vix.iloc[0] - 1.0) < 1e-12                    # VIX 18 -> 30% = the base allocation


def test_continuous_take_profit_exits_early_and_redeploys_same_close():
    C, und = _chain()
    bk = L.Book(C, und)
    vix = pd.Series(25.0, index=pd.bdate_range("2023-12-01", "2024-08-01"))
    cyc, _ = U.continuous_book(bk, U.BOOK_STRUCTS["STRG16"], "T50", vix)
    assert cyc.early.iloc[0]
    assert cyc.entry.iloc[1] == cyc.exit.iloc[0]
    assert abs(cyc.w_vix.iloc[0] - 0.35 / 0.30) < 1e-12


def test_roll_untested_rolls_the_call_down_when_puts_are_tested():
    C, und = _chain(path_fn=lambda i: 100.0 - 0.25 * min(i, 30))     # falls 7.5% over six weeks, then flat
    bk = L.Book(C, und)
    cyc, incs = U.roll_untested_book(bk, U.BOOK_STRUCTS["STRG16"], "H")
    assert len(cyc) >= 1 and cyc.rolls.iloc[0] >= 1
    for j, s in enumerate(incs):
        assert abs(s.sum() - cyc.pnl.iloc[j]) < 1e-9


def test_earnings_short_straddle_accounting():
    C, und = _chain()
    bk = L.Book(C, und)
    E = pd.DatetimeIndex([pd.Timestamp("2024-02-14")])
    r = pd.DataFrame(U.earnings_trades("XYZ", bk, und, E))
    s = r[r.test == "E1_short_straddle"].iloc[0]
    assert s.entry < s.earnings < s.exit
    # flat price and constant IV: the straddle decays, so the short earns (credit minus buy-back, minus costs)
    assert s.pnl > 0 and s.credit > 0
    assert set(r.test) >= {"E1_short_straddle", "E2_short_strangle16"}


def test_vix_weights():
    assert abs(U.vix_weight(12) - 0.25 / 0.30) < 1e-12
    assert abs(U.vix_weight(45) - 0.50 / 0.30) < 1e-12
    assert U.vix_weight(np.nan) == 1.0
