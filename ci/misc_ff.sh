# Free retail-flow and daily positioning tests (docs/free_flow_preregistration.md): FF1, FF2, FD1.
pip install -q openpyxl xlrd
mkdir -p ffaj
for f in trading_vol_and_position.xls open_position_with_mc.xls trading_vol_by_cp.xls; do
  curl -sS -L -m 120 -A "fxlab-research/1.0 (research data collection)" -o ffaj/$f "https://www.ffaj.or.jp/wp-content/uploads/2026/10/$f" || true
  echo "ffaj $f $(stat -c %s ffaj/$f 2>/dev/null) $(file -b ffaj/$f 2>/dev/null | cut -c1-60)"
done
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD XAUUSD XAGUSD EURJPY GBPJPY AUDJPY NZDJPY
mkdir -p out/ff
python -W ignore scripts/free_flow_lab.py --store flow_store --ffaj ffaj --out out/ff
