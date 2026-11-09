# Round 16 exploratory follow-ups (scripts/r16_explore.py; not pre-registered, nothing counts).
gh run download 42669526702 -n r16data -D r16data && ls r16data
python -W ignore scripts/r16_explore.py --data r16data --out out/r16x
