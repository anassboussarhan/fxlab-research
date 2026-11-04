# Diagnostic 2 for the V1-V9 batch: M1 hour coverage near 17:05 New York, why daily trades go unpriced, file heads.
R=$(gh run list --workflow flow_collect.yml --status success -L 1 --json databaseId -q '.[0].databaseId')
gh run download $R -n flow-store -D flow_store
ci/fetch_m1.sh 41447353092 EURUSD USDJPY
mkdir -p out/flowdiag
python -W ignore - <<'EOF' 2>&1 | tee out/flowdiag/diag2.txt
import sys, glob
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import flow_volume_lab as L
for sym in ("EURUSD", "USDJPY"):
    bk = L.book(sym)
    ix = bk.idx
    print("\n==", sym, ix[0], ix[-1], len(ix))
    h = pd.DataFrame({"y": ix.year, "h": ix.hour})
    print(pd.crosstab(h.y, h.h).to_string())
    nyix = ix.tz_convert("America/New_York")
    near = (nyix.hour == 17) & (nyix.minute < 40)
    print("bars 17:00-17:39 NY by year:", pd.Series(near).groupby(ix.year).sum().to_dict())
    for d in pd.bdate_range("2012-03-05", periods=6).tolist() + pd.bdate_range("2021-03-08", periods=6).tolist():
        t_in = L.ny(L.bday_after(d, 1)); t_out = L.ny(L.bday_after(d, 2))
        i, j = bk.at(t_in, 30), bk.at(t_out, 30)
        k = int(ix.searchsorted(t_in))
        nxt = ix[k] if k < len(ix) else None
        print(d.date(), "t_in", t_in, "at", i, "next bar", nxt, "| t_out", t_out, "at", j)
for f in glob.glob("flow_store/**/GLD_holdings_archive.csv", recursive=True):
    t = open(f, "rb").read().decode("utf-8", "replace").splitlines()
    print("\n== GLD lines", len(t))
    for l in t[:12] + t[-3:]:
        print(repr(l[:300]))
EOF
