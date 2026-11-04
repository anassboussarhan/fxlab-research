# Queue batch Q, part A (docs/queue_preregistration.md): VS1, VS2, VS4, VS5 on M1 2007-2025.
ci/fetch_m1.sh 41447353092 EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURJPY XAUUSD AUDJPY NZDJPY
mkdir -p out/q
python -W ignore scripts/queue_lab.py --part-a --out out/q
