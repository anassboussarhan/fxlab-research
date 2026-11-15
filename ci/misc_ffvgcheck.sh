set -u
bash ci/fetch_m1.sh 41447353092 USATECHIDXUSD
python -W ignore scripts/fvg_forward_check.py --out out
