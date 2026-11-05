# Round 9, M1 part (docs/round9_preregistration.md): G1, AT1, AT2, AT4, S5.
ci/fetch_m1.sh 41447353092 USDJPY EURJPY XAUUSD XAGUSD EURUSD GBPUSD AUDUSD USDCHF
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --m1 --out out/r9
