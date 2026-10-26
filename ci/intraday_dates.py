"""Dates file for a Dukascopy download: gotobi = Tokyo Gotobi days 2019-2025 except Mondays (the I4 rule skips
them); gotobi_prev = the calendar day before each of those (03:00 Tokyo is 18:00 UTC the previous day)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import flows_lab as FL  # noqa: E402

mode = sys.argv[1]
hol = FL.jp_holidays(range(2019, 2026))
G = [d for d in FL.gotobi_days("2019-01-01", "2025-12-31", hol) if d.weekday() != 0]
days = G if mode == "gotobi" else [d - pd.Timedelta(days=1) for d in G]
for d in days:
    print(d.date().isoformat())
