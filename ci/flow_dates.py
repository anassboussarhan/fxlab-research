"""Dates files for the flow-edges downloads (scripts/flow_edges_lab.py), one year at a time:
  g    Tokyo Gotobi days (all, Mondays included)              -> fetched 00-01 UTC (09:00-10:59 Tokyo)
  gp   the calendar day before each Gotobi day                -> fetched 14-23 UTC (23:00 Tokyo onwards)
  c    control days: every third Tokyo business day, Tue-Fri, that is not a Gotobi day or a Japanese holiday
  cp   the calendar day before each control day
  me   the last weekday of each month (the London month-end fix day) -> fetched 00-16 UTC
  mep  the weekday before it                                   -> fetched 14-23 UTC (from 16:00 London)
Usage: python ci/flow_dates.py MODE YEAR"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import flows_lab as FL  # noqa: E402

mode, yr = sys.argv[1], int(sys.argv[2])
start, end = f"{yr}-01-01", f"{yr}-12-31"
hol = FL.jp_holidays(range(yr - 1, yr + 2))


def controls():
    G = set(FL.gotobi_days(start, end, hol))
    c = [d for d in pd.bdate_range(start, end) if d not in G and d not in hol and d.weekday() != 0]
    return c[::3]


if mode in ("g", "gp"):
    days = FL.gotobi_days(start, end, hol)
elif mode in ("c", "cp"):
    days = controls()
else:
    days = FL.month_end_days(start, end)
if mode in ("gp", "cp"):
    days = [d - pd.Timedelta(days=1) for d in days]
if mode == "mep":
    days = [d - pd.offsets.BDay(1) for d in days]
for d in days:
    print(pd.Timestamp(d).date().isoformat())
