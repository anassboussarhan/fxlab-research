# Round 9, M1 part 3: S4 (BoJ JGB purchase days, archive parsed here) and AT8 rerun.
pip install -q lxml html5lib openpyxl xlrd || true
bash ci/r9_getdata.sh
python ci/r9_boj.py r9data && cp r9data/boj_rinban_dates.csv out/ 2>/dev/null
ci/fetch_m1.sh 41447353092 USDJPY
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --m1 --only s4 --out out/r9 --r9data r9data
