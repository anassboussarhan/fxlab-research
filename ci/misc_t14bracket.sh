# Round 14 A1 investigation: bracket the daily engine's same-segment rule for Turtle 1 / 2 (information only).
gh run download 42534301711 -p "d1p-*" -D d1p_in
git fetch -q origin long:refs/remotes/origin/long 2>/dev/null || git fetch -q -f origin +long:refs/remotes/origin/long
mkdir -p out/t14bracket ref && git show origin/long:long/A_monthly_by_symbol.csv > ref/r13.csv
python -W ignore scripts/trend14_bracket.py --inp d1p_in --r13 ref/r13.csv --out out/t14bracket
