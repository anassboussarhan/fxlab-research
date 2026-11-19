#!/usr/bin/env bash
# Usage: ci/r23_place.sh SYM -> data/m1/SYM/*.parquet. Round-22 instruments via ci/r22_place.sh; the round-23
# downloads from the r23dl run in ci/r23_dl_run.txt, rescaled by the power of ten that matches ci/r23_expected_2024.json.
set -u
S=$1
if python -c "import json,sys; d=json.load(open('ci/r22_syms.json')); sys.exit(0 if '$S' in d['store'] or '$S' in d['d2'] or '$S' in ('BTCUSDT','ETHUSDT') else 1)"; then
  bash ci/r22_place.sh $S
  exit $?
fi
R=$(cat ci/r23_dl_run.txt)
mkdir -p data/m1/$S dl_$S
bash ci/dl_each.sh $R "^r23[ch]-$S(-|$)" dl_$S || { echo "data download incomplete for $S"; exit 1; }
find dl_$S -name "*.parquet" -exec mv {} data/m1/$S/ \;
rm -rf dl_$S
python ci/r23_rescale.py $S
echo "$S: $(ls data/m1/$S | tr '\n' ' ')"
