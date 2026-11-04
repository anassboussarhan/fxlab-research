# P3 / P4 of docs/push_preregistration.md: Dukascopy M1 for USD/JPY and EUR/JPY, May 2003 - December 2006 (never
pip install -q openpyxl xlrd
# used before; not committed), the five other yen crosses 2019-2025 from the panel run, then the lab.
python -m fxlab.data.acquire_dukascopy_candles --pairs USDJPY EURJPY --start 2003-05-01 --end 2007-01-01 \
  --workers 6 --deadline-min 150 2>&1 | tail -5
mkdir -p data/m1_early
for P in USDJPY EURJPY; do
  mkdir -p data/m1_early/$P
  for f in data/m1/$P/${P}_200[3-6]*.parquet; do [ -f "$f" ] && mv "$f" data/m1_early/$P/; done
  echo "$P early files: $(ls data/m1_early/$P | tr '\n' ' ')"
done
ci/fetch_m1.sh 41447353092 GBPJPY AUDJPY CHFJPY CADJPY NZDJPY USDJPY EURJPY
python -W ignore scripts/push_lab.py --p34 --early data/m1_early --out out/push
