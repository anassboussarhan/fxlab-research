#!/usr/bin/env bash
# Resumable local research sequence for a small machine. Safe to re-run: completed experiments are skipped.
# Usage: scripts/run_local_campaigns.sh [workers] [minutes-per-campaign-chunk]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
W=${1:-1}
BUDGET=${2:-100}
export POLARS_MAX_THREADS=1
for C in screen_h1_a models_eurusd_h1 screen_tf_eurusd risk_eurusd_h1 screen_h1_b; do
  [ -f results/manifests/$C.json ] || continue
  echo "== $C $(date -u +%FT%TZ)"
  $PY scripts/campaign.py run --name "$C" --workers "$W" --time-budget-min "$BUDGET"
  $PY scripts/campaign.py status --name "$C"
done
