# Round 15 part D: rerun D1b (index alignment fixed) and D5 (RBA parser fixed) - code fixes, rules unchanged.
pip install -q yfinance requests >/dev/null 2>&1
curl -sSL -o ff.csv "https://huggingface.co/datasets/Tropstan/Forex_Factory_Calendar/resolve/main/forex_factory_cache.csv"
gh run download 42534301711 -p "d1p-*" -D d1p_in
mkdir -p out/data15b
for P in d1b d5; do python -W ignore scripts/data15_tests.py --part $P --ff ff.csv --panels d1p_in --out out/data15b 2>&1 | tail -30; done
