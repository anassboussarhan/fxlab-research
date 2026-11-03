# M2 meta-labels, E26 Fed expected direction, E25 MoF probe (ci/misc_job.sh, tag "last")
ci/fetch_m1.sh 41447353092 AUDNZD EURGBP GBPCAD XAUUSD EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD
python -W ignore scripts/edges_last.py --raw raw --out out/last --probe || echo "edges_last failed"
python -W ignore scripts/ml_meta_lab.py --out out/m2 || echo "m2 failed"
