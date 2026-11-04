# O1 of docs/databento_preregistration.md: selling rich euro / yen volatility (Databento CME options, compact files).
pip install -q databento==0.90.0
gh run download 41846466839 -n databento-data -D dbn && ls dbn | wc -l
gh run download 41854224685 -n databento-compact -D compact && ls -la compact
mkdir -p out/db
python -W ignore scripts/databento_lab.py --dbn dbn --compact compact --out out/db --o1
