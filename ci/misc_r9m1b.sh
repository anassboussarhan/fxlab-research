# Round 9, M1 part 2: N1-N3, S2, S5 (rerun), AT7, AT8.
bash ci/r9_getdata.sh
ci/fetch_m1.sh 41447353092 USDJPY EURUSD GBPUSD AUDUSD NZDUSD USDCAD USDCHF XAUUSD XAGUSD
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --m1 --only s5,s2,n_tests,at7,at8 --out out/r9 --r9data r9data
