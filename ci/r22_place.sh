#!/usr/bin/env bash
# Usage: ci/r22_place.sh SYM  -> data/m1/SYM/*.parquet from the store (33), the round-15 D2 downloads (13) or the
# round-19 crypto files (2).
set -u
S=$1
if python -c "import json,sys; d=json.load(open('ci/r22_syms.json')); sys.exit(0 if '$S' in d['store'] else 1)"; then
  bash ci/fetch_m1.sh 41447353092 $S
elif python -c "import json,sys; d=json.load(open('ci/r22_syms.json')); sys.exit(0 if '$S' in d['d2'] else 1)"; then
  mkdir -p data/m1/$S dl_$S
  gh run download 42610347008 -p "d2c-$S-*" -D dl_$S || true
  find dl_$S -name "*.parquet" -exec mv {} data/m1/$S/ \;
  rm -rf dl_$S
else
  mkdir -p data/m1/$S
  gh run download 43601520485 -n cdata -D cd_$S
  cp cd_$S/m1/$S/*.parquet data/m1/$S/
  rm -rf cd_$S
fi
echo "$S: $(ls data/m1/$S | tr '\n' ' ')"
