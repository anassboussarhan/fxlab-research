#!/usr/bin/env bash
# Usage: ci/fetch_m1.sh PANEL_RUN SYM [SYM ...]  -> data/m1/SYM/SYM_YEAR.parquet for 2007-2025.
# With STAGE_RUN set (a run of the m1-stage workflow), one consolidated artifact m1all-SYM is downloaded instead of
# the pieces of six runs (far fewer API calls; the pieces route hit the token's rate limit with many parallel jobs).
set -u
PD=$1; shift
get() {  # gh run download with retries; prints the last error
  for k in 1 2 3 4 5; do
    out=$(gh run download "$@" 2>&1) && return 0
    echo "$out" | grep -q -i "no artifact" && return 1
    echo "retry $k: $(echo "$out" | tail -1)"; sleep $((20 * k))
  done
  return 1
}
for S in "$@"; do
  mkdir -p dl_$S
  if [ -n "${STAGE_RUN:-}" ]; then
    get $STAGE_RUN -n "m1all-$S" -D dl_$S/st || echo "no staged artifact for $S"
  else
    for R in 40403976401 40567306171; do get $R -p "mm-$S-*" -D dl_$S/mm_$R || true; done
    get 40946488168 -p "fd-$S-*" -D dl_$S/fd || true
    get 41332786946 -p "fd2-$S-*" -D dl_$S/fd2 || true
    [ $S = USDJPY ] && { get 40227329900 -p "m1-USDJPY-*" -D dl_$S/uj || true; }
    [ $S = EURJPY ] && { get 40563102528 -p "dk-EURJPY-*" -D dl_$S/ej || true; }
    get $PD -p "pd-$S-*" -D dl_$S/pd || true
  fi
  python ci/place_m1.py dl_$S data/m1 | tail -2 || true
  rm -rf dl_$S
  echo "$S years: $(ls data/m1/$S 2>/dev/null | sed 's/.*_//; s/.parquet//' | tr '\n' ' ')"
done
