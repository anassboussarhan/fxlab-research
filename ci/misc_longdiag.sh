# Round 13 diagnostic (information only): where the top part-B units earn their money.
gh run download 42525289851 -p "long-*" -D long_in && ls long_in | head
mkdir -p out/longdiag
python -W ignore scripts/long_diag.py --inp long_in --out out/longdiag
