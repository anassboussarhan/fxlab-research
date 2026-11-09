# Round 17 exploratory follow-ups (scripts/r17_explore.py; not pre-registered, nothing counts).
gh run download 42696792601 -n r17data -D r17data && du -sh r17data
python -W ignore scripts/r17_explore.py --data r17data --out out/r17x
