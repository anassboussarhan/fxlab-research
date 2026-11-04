# P1 rerun with the Australian 2-year yield read correctly (parser fix), then the BH combination again.
pip install -q openpyxl xlrd
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 GBPUSD USDCAD AUDUSD USDCHF NZDUSD
mkdir -p out/push
python -W ignore scripts/push_lab.py --p1 --store flow_store --out out/push
git fetch -q origin misc-push12 && for f in p2.json p5_v2_v9.csv; do git show origin/misc-push12:misc/push/$f > out/push/$f; done
git fetch -q origin misc-push34 && git show origin/misc-push34:misc/push/p34.json > out/push/p34.json 2>/dev/null || true
python -W ignore scripts/push_lab.py --combine --out out/push
