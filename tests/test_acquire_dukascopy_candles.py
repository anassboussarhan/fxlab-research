import lzma
from datetime import date

import numpy as np

from fxlab.data import acquire_dukascopy_candles as C


def _blob(rows):
    a = np.array(rows, dtype=C.DT)
    return lzma.compress(a.tobytes())


def test_decode_and_merge(monkeypatch):
    day = date(2024, 3, 13)
    bid = _blob([(0, 109271, 109272, 109268, 109273, 80.8), (60, 109272, 109270, 109265, 109275, 0.0),
                 (120, 109270, 109280, 109269, 109281, 12.0)])
    ask = _blob([(0, 109273, 109274, 109270, 109275, 80.8), (60, 109274, 109272, 109267, 109277, 0.0),
                 (120, 109272, 109282, 109271, 109283, 9.0)])
    monkeypatch.setattr(C, "fetch_day", lambda pair, d, side: bid if side == "BID" else ask)
    df = C.day_frame("EURUSD", day)
    assert df.height == 2                                       # the zero-volume minute is dropped
    r = df.row(0, named=True)
    assert abs(r["bid_o"] - 1.09271) < 1e-12 and abs(r["bid_c"] - 1.09272) < 1e-12
    assert abs(r["bid_l"] - 1.09268) < 1e-12 and abs(r["bid_h"] - 1.09273) < 1e-12
    assert abs(r["ask_c"] - 1.09274) < 1e-12 and str(r["ts"])[:19] == "2024-03-13 00:00:00"
    assert str(df.row(1, named=True)["ts"])[:19] == "2024-03-13 00:02:00"
    j = C.decode_day(_blob([(0, 151234, 151240, 151200, 151300, 1.0)]), day, "USDJPY")
    assert abs(j["c"][0] - 151.240) < 1e-9
