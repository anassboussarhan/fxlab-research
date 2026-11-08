# Round 9, daily part: S1, AT3, AT5.
R9ONLY=at5
bash ci/r9_getdata.sh
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --daily --only $R9ONLY --out out/r9 --r9data r9data
