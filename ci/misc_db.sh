# Databento tests F1 and K1 (docs/databento_preregistration.md).
pip install -q databento==0.90.0
python ci/fomc_dates.py fomc.csv
gh run download 41846466839 -n databento-data -D dbn && ls -la dbn | head -30
ci/fetch_m1.sh 41447353092 EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD USDMXN XAUUSD XAGUSD
mkdir -p out/db && cp fomc.csv out/db/
python -W ignore scripts/databento_lab.py --dbn dbn --fomc fomc.csv --out out/db --f1 --k1
