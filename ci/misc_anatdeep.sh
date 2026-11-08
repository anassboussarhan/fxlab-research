# Round 12, deeper look across units (information only).
gh run download 42520497617 -p "anat-*" -D anat_in && ls anat_in
git fetch -q --depth 1 origin anatomy && git show FETCH_HEAD:anatomy/anatomy_units.csv > anatomy_units.csv
mkdir -p out/anatdeep
python -W ignore scripts/anatomy_deep.py --inp anat_in --units anatomy_units.csv --out out/anatdeep
