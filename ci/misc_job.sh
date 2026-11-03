set -u
for R in 41507595752 41553423279; do gh run download $R -n edges-data -D raw/$R || true; done
ci/fetch_m1.sh 41447353092 USDJPY EURJPY
mkdir -p out
python -W ignore scripts/prop_checks.py --raw raw --out out/practical
