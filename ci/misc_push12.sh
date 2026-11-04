# P1 / P2 of docs/push_preregistration.md: the latest flow store (with the push probe files), M1 for the five
# dollar pairs, then the lab and the BH combination with the P3 / P4 results of the misc-push34 branch.
pip install -q openpyxl xlrd
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
echo "flow store from run $R"
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 GBPUSD USDCAD AUDUSD USDCHF NZDUSD XAUUSD
mkdir -p out/push
python -W ignore scripts/push_lab.py --fetch-extra --p1 --p2 --p5 --store flow_store --out out/push
git fetch -q origin misc-push34 && git show origin/misc-push34:misc/push/p34.json > out/push/p34.json 2>/dev/null || true
python -W ignore scripts/push_lab.py --combine --out out/push
