# Queue batch Q, part B (docs/queue_preregistration.md): T1, BS1, C5E, VS3, P1 with Australia.
pip install -q openpyxl xlrd
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURJPY XAUUSD XAGUSD
NO_STAGE=1 ci/fetch_m1.sh 41447353092 USA500IDXUSD USATECHIDXUSD JPNIDXJPY BRENTCMDUSD USTBONDTRUSD BUNDTREUR
mkdir -p out/q
python -W ignore scripts/queue_lab.py --part-b --vs3 --store flow_store --out out/q
