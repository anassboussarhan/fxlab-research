# Diagnostic for the V1-V9 batch: event counts by year, priced share, raw file heads. No test results.
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 EURUSD USDJPY
mkdir -p out/flowdiag
python -W ignore - <<'EOF' 2>&1 | tee out/flowdiag/diag.txt
import sys, os, glob
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import flow_volume_lab as L
st = "flow_store"
print("files:", len(glob.glob(st + "/**/*", recursive=True)))
for f in sorted(glob.glob(st + "/**/*", recursive=True))[:400]:
    if os.path.isfile(f):
        print(f"{os.path.getsize(f):>10} {f}")
F = L.futures(st)
for name, d in F.items():
    v = np.log(d.Volume.where(d.Volume > 0))
    z = (v - v.rolling(60, min_periods=40).mean().shift(1)) / v.rolling(60, min_periods=40).std().shift(1)
    yr = d.index.year
    t = pd.DataFrame({"vol_med": d.Volume.groupby(yr).median(), "zero_or_nan": d.Volume.isna().groupby(yr).sum() + (d.Volume == 0).groupby(yr).sum(),
                      "z_ge_2": (z >= 2).groupby(yr).sum(), "rows": d.Volume.groupby(yr).size()})
    print("\n==", name, "\n", t.to_string())
    print(d.tail(3).to_string())
for sym, name, sg in (("EURUSD", "EUR", 1), ("USDJPY", "JPY", -1)):
    d = F[name]
    v = np.log(d.Volume.where(d.Volume > 0))
    z = (v - v.rolling(60, min_periods=40).mean().shift(1)) / v.rolling(60, min_periods=40).std().shift(1)
    days = [x for x in z[z >= 2].index if pd.Timestamp("2007-01-01") <= x <= L.END]
    ok = {}
    for dd in days:
        r = L.daily_trade(sym, dd, 1, 1)
        ok.setdefault(dd.year, [0, 0])[0 if r is None else 1] += 1
    print("\npriced", sym, "year: [unpriced, priced]", ok)
for pat in ("GLD_holdings_archive.csv", "mof_en.csv", "mof_ja.csv"):
    for f in glob.glob(st + "/**/" + pat, recursive=True):
        raw = open(f, "rb").read()
        print("\n==", f, len(raw))
        for enc in ("utf-8", "cp932"):
            try:
                print("\n".join(raw.decode(enc).splitlines()[:25])); break
            except Exception as ex:
                print(enc, ex)
EOF
