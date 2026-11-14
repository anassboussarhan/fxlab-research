# Inspect the Databento daily file bought in run 43629956478 (no new purchase).
set -u
mkdir -p out
gh run download 43629956478 -n r20w-data -D wd
ls -la wd
python - <<'PY'
import pandas as pd
D = pd.read_parquet("wd/daily.parquet")
print(D.dtypes); print(D.head(10).to_string()); print(len(D), D.symbol.nunique(), D.date.min(), D.date.max())
print(D.describe().to_string())
print("symbols null:", D.symbol.isna().mean())
PY
