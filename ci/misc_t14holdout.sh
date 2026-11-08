# Round 14 part D: Double Seven on weekly bars, 2026 (third use of the holdout; docs/trend14_preregistration.md).
git fetch -q -f origin +misc-r9data4:refs/remotes/origin/misc-r9data4
mkdir -p out/t14holdout ref && git show origin/misc-r9data4:misc/r9data/ecb/exr.csv > ref/ecb.csv
python -W ignore scripts/trend14_holdout.py --ecb ref/ecb.csv --out out/t14holdout
