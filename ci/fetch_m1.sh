#!/usr/bin/env bash
# Usage: ci/fetch_m1.sh PANEL_RUN SYM [SYM ...]  -> data/m1/SYM/SYM_YEAR.parquet for 2007-2025 (mirror + Dukascopy + panel)
set -u
PD=$1; shift
for S in "$@"; do
  mkdir -p dl_$S
  for R in 40403976401 40567306171; do gh run download $R -p "mm-$S-*" -D dl_$S/mm_$R >/dev/null 2>&1 || true; done
  gh run download 40946488168 -p "fd-$S-*" -D dl_$S/fd >/dev/null 2>&1 || true
  gh run download 41332786946 -p "fd2-$S-*" -D dl_$S/fd2 >/dev/null 2>&1 || true
  [ $S = USDJPY ] && { gh run download 40227329900 -p "m1-USDJPY-*" -D dl_$S/uj >/dev/null 2>&1 || true; }
  [ $S = EURJPY ] && { gh run download 40563102528 -p "dk-EURJPY-*" -D dl_$S/ej >/dev/null 2>&1 || true; }
  gh run download $PD -p "pd-$S-*" -D dl_$S/pd >/dev/null 2>&1 || true
  python ci/place_m1.py dl_$S data/m1 | tail -3 || true
  rm -rf dl_$S
  echo "$S years: $(ls data/m1/$S 2>/dev/null | tr '\n' ' ')"
done
