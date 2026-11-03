# E4 / E16 rerun with the re-downloaded emerging-market years (em-redownload run 41652052176), rules unchanged
EXTRA_RUNS=41652052176 ci/fetch_m1.sh 41447353092 USDCNH USDMXN USDZAR USDSGD AUDUSD
python -W ignore scripts/edges_lab2.py --tests E4 E16 --raw raw --out out/em 2>&1 | tee out/em_log.txt
