# E25 JGB auctions (dates in data/jgb_auctions.csv, from the MoF spreadsheet saved by the misc-last run)
ci/fetch_m1.sh 41447353092 USDJPY EURUSD GBPUSD AUDUSD
python -W ignore scripts/edges_last.py --raw raw --out out/e25 --e25 data/jgb_auctions.csv
