# V1-V9 flow / volume / positioning batch (docs/flow_volume_preregistration.md)
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
echo "flow store from run $R"
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD USDMXN XAUUSD XAGUSD EURCHF
python -W ignore scripts/flow_volume_lab.py --store flow_store --out out/flowvol
