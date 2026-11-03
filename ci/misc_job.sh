set -u
for R in 41507595752 41553423279; do gh run download $R -n edges-data -D raw/$R || true; done
mkdir -p out
TAG=$(cut -d' ' -f1 ci/MISC)
case "$TAG" in
  practical)
    ci/fetch_m1.sh 41447353092 USDJPY EURJPY
    python -W ignore scripts/prop_checks.py --raw raw --out out/practical ;;
  weekly)
    ci/fetch_m1.sh 41447353092 $(python -c "import sys;sys.path.insert(0,'scripts');import ml_weekly_lab as W;print(' '.join(W.PAIRS))") USA500IDXUSD
    python -W ignore scripts/ml_weekly_lab.py --raw raw --out out/m1 ;;
  *)
    bash ci/misc_$TAG.sh ;;
esac
