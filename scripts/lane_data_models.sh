#!/usr/bin/env bash
# Lane B: finish data acquisition, then the model campaign and feature analysis (resumable)
cd "$(dirname "$0")/.."
PY=/home/claude/venv/bin/python
export POLARS_MAX_THREADS=1
while pgrep -f "fxlab.data.acquire_fxdata_git" >/dev/null; do sleep 30; done
echo "== acquisition resume $(date -u +%T)"
FXLAB_SCRATCH=/home/claude/rawdata timeout 5400 $PY -u -m fxlab.data.acquire_fxdata_git >> logs/acquisition.log 2>&1
echo "== models $(date -u +%T)"
$PY scripts/campaign.py run --name models_eurusd_h1 --workers 1 --time-budget-min "${BUDGET:-100}" >> logs/lane_models.out 2>&1
$PY scripts/campaign.py status --name models_eurusd_h1
echo "== lane B pass finished $(date -u +%T)"
