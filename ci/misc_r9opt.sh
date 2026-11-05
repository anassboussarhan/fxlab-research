# Round 9, options part (docs/round9_preregistration.md): O3, O4, O5, AT6 on the Databento CME options.
pip install -q databento==0.90.0
gh run download 41846466839 -n databento-data -D dbn && ls dbn | wc -l
gh run download 41854224685 -n databento-compact -D compact && ls compact
ci/fetch_m1.sh 41447353092 EURUSD USDJPY
mkdir -p out/r9
python -W ignore scripts/round9_lab.py --opt --out out/r9 --dbn dbn --compact compact
