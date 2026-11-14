import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r19  # noqa: E402


def day(path, spr=0.5):
    p = np.asarray(path, float)
    n = r19.SESSION
    o = np.r_[p[0], p[:-1]]
    return {"o": o, "h": np.maximum(o, p) + 0.1, "l": np.minimum(o, p) - 0.1, "c": p, "spr": np.full(n, spr), "w": np.ones(n)}


def test_orb_long_stop_and_target():
    n = r19.SESSION
    p = np.full(n, 100.0)
    p[:5] = [100.2, 100.4, 100.6, 100.8, 101.0]          # up candle -> long at the 09:35 open
    p[5:] = 101.0
    A = day(p)
    t = r19.orb5_day(A)
    d, E, X, ki, ko = t[0]
    assert d == 1 and E == A["o"][5] and ko == n - 1 and X == A["c"][-1]
    p2 = p.copy()
    p2[50:] = 95.0                                         # crash through the stop
    A2 = day(p2)
    d, E, X, ki, ko = r19.orb5_day(A2)[0]
    stop = float(np.min(A2["l"][:5]))
    assert ko == 50 and X == min(A2["o"][50], stop)


def test_nb_breakout_and_vwap_exit():
    n = r19.SESSION
    p = np.full(n, 100.0)
    p[10:] = 102.0                                          # above the band from 09:40
    p[200:] = 100.3                                         # back below VWAP / UB
    A = day(p)
    sigma = np.full(n, 0.005)
    tr = r19.nb_day(A, sigma, prev_close=100.0)
    assert tr[0][0] == 1 and tr[0][3] == 30                # long at 10:00
    assert tr[0][4] == 210                                  # exit at the first half-hour after the drop (13:00)
    g, fu, cf = r19.costs_of(tr, A, 0.3, 0.01)
    assert abs(g - (tr[0][2] / tr[0][1] - 1)) < 1e-12 and fu > 0 and cf > 0


def test_nb_no_trade_inside_band():
    p = np.full(r19.SESSION, 100.0)
    assert r19.nb_day(day(p), np.full(r19.SESSION, 0.01), 100.0) == []


def test_session_days_roundtrip():
    idx = pd.date_range("2024-03-05 14:30", periods=390, freq="min", tz="UTC")   # 09:30-15:59 New York (EST)
    m = pd.DataFrame({c: 100.0 for c in r19.COLS}, index=idx)
    m["ask_o"] = m["ask_c"] = m["ask_h"] = m["ask_l"] = 100.5
    D = r19.session_days(m)
    assert len(D) == 1
    A = list(D.values())[0]
    assert np.allclose(A["o"], 100.25) and np.allclose(A["spr"], 0.5)


def test_bh_and_kind():
    assert r19.bh([0.001, 0.5, 0.9]).tolist() == [True, False, False]
    assert r19.kind_of("USTBONDTRUSD") == "bond" and r19.kind_of("BTCUSDT") == "crypto" and r19.kind_of("DEUIDXEUR") == "other"


def test_lib_combine(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2013-01-01", "2025-12-31")
    units = ["S.X|H1", "S.X|H1", "PLACEBO17_00|H1", "PLACEBO17_01|H1"]
    pnl = rng.normal(0, 1e-3, (4, len(days))).astype(np.float32)
    pnl[0] += 5e-4
    pnl[1] += 5e-4
    np.savez_compressed(tmp_path / "daily17_DEUIDXEUR.npz", unit=np.array(units), cfg=np.array(["a", "b", "p", "p"]),
                        dir=np.array([1, 1, 1, 1], np.int8), days=np.array(list(days)), pnl=pnl)
    out = tmp_path / "o"
    A = type("A", (), {"inp": str(tmp_path), "out": str(out)})
    r19.cmd_lib(A)
    R = pd.read_csv(out / "lib_units.csv")
    u = R[R.unit == "S.X|H1"].iloc[0]
    assert u.settings == 2 and u.p2_t > 3
