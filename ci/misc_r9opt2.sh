# Round 9, options part 2: F2 (CME 6J signed order flow, 2023-2025) and AT9 (option-implied Hurst).
pip install -q databento==0.90.0
gh run download 41846466839 -n databento-data -D dbn
gh run download 41854224685 -n databento-compact -D compact
gh run download 42496139194 -n databento-compact -D compact2 && cp compact2/* compact/ && ls compact
ci/fetch_m1.sh 41447353092 USDJPY
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --opt --only f2,at9 --out out/r9 --dbn dbn --compact compact
