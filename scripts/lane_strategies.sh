#!/usr/bin/env bash
# Lane A: strategy campaigns + analyses (resumable; safe to re-run)
cd "$(dirname "$0")/.."
PY=/home/claude/venv/bin/python
export POLARS_MAX_THREADS=1
for C in screen_h1_a screen_tf_eurusd risk_eurusd_h1 screen_h1_b; do
  [ -f results/manifests/$C.json ] || continue
  echo "== $C start $(date -u +%T)"
  $PY scripts/campaign.py run --name "$C" --workers "${WORKERS:-1}" --time-budget-min "${BUDGET:-110}" >> logs/lane_strategies.out 2>&1
  $PY scripts/campaign.py status --name "$C"
done
echo "== lane A pass finished $(date -u +%T)"
