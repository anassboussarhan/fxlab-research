#!/usr/bin/env bash
# Run campaigns sequentially within ONE global deadline (minutes), so a pass always ends gracefully
# before the 2-hour background limit. Usage: scripts/lane.sh <deadline-min> <campaign> [<campaign> ...]
cd "$(dirname "$0")/.."
PY=/home/claude/venv/bin/python
export POLARS_MAX_THREADS=1
DEADLINE_MIN=$1; shift
START=$(date +%s)
for C in "$@"; do
  [ -f results/manifests/$C.json ] || continue
  LEFT=$(( DEADLINE_MIN - ( $(date +%s) - START ) / 60 ))
  [ "$LEFT" -gt 5 ] || break
  echo "== $C start $(date -u +%T) budget ${LEFT}m"
  $PY scripts/campaign.py run --name "$C" --workers "${WORKERS:-1}" --time-budget-min "$LEFT" >> logs/lane_$C.out 2>&1
  $PY scripts/campaign.py status --name "$C"
done
echo "== pass finished $(date -u +%T)"
