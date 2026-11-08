# Round 9: S3 (Norges Bank, releases found by URL) and S4 (BoJ scheduled purchase days from the quarterly schedules).
bash ci/r9_getdata.sh
git fetch -q --depth 1 origin misc-r9data5 && git show FETCH_HEAD:misc/r9data/boj_rinban_dates.csv > r9data/boj_rinban_dates.csv
python ci/r9_norges.py r9data && cp r9data/norges_fx.csv out/
ci/fetch_m1.sh 41447353092 USDNOK
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --m1 --only s3 --out out/r9 --r9data r9data
