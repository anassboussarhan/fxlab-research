# FF2 rerun (pandas 3 text-conversion fix in the FFAJ parser).
pip install -q openpyxl xlrd
mkdir -p ffaj
for f in trading_vol_and_position.xls open_position_with_mc.xls trading_vol_by_cp.xls; do
  curl -sS -L -m 120 -A "fxlab-research/1.0 (research data collection)" -o ffaj/$f "https://www.ffaj.or.jp/wp-content/uploads/2026/10/$f" || true
done
ci/fetch_m1.sh 41447353092 USDJPY EURJPY GBPJPY AUDJPY NZDJPY
mkdir -p out/ff
python -W ignore scripts/free_flow_lab.py --only FF2 --ffaj ffaj --out out/ff
