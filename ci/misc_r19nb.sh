# Round 19, exploratory (not a test): the noise-area breakout by year, after the paper's sample (May 2024 on),
# and its vol-sized drawdown. Reads the part-A outputs of run 43601331590.
set -u
mkdir -p out
for S in USA500IDXUSD USATECHIDXUSD; do gh run download 43601331590 -n r19-idx-$S -D in/$S; done
python - <<'PY'
import glob, json, math
import numpy as np, pandas as pd
A = pd.concat([pd.read_parquet(f) for f in glob.glob("in/**/idx_*.parquet", recursive=True)])
A["day"] = pd.to_datetime(A.day)
out = {}
def t(x):
    x = x.dropna(); return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))) if len(x) > 20 else None
for sym in A.pair.unique():
    g = A[(A.pair == sym) & (A.test == "NB")].set_index("day").sort_index()
    y = g.groupby(g.index.year).agg(days=("net_fut", "size"), net_fut_bp=("net_fut", lambda v: v.mean() * 1e4),
                                    gross_bp=("gross", lambda v: v.mean() * 1e4), trades=("trades", "mean"))
    post = g[g.index >= "2024-05-01"]
    v = (g.net_fut * g.lev).dropna()
    eq = v[v.index >= "2019-01-01"].cumsum()
    out[sym] = {"by_year": y.round(3).reset_index().to_dict("records"),
                "post_paper_2024_05_2025_12": {"days": int(len(post)), "net_fut_bp": float(post.net_fut.mean() * 1e4),
                                               "t": t(post.net_fut), "gross_bp": float(post.gross.mean() * 1e4)},
                "volsized_2019_2025_maxdd_pct": float((eq - eq.cummax()).min() * 100),
                "volsized_2019_2025_ann_ret_pct": float(v[v.index >= "2019-01-01"].mean() * 252 * 100),
                "days_with_trade_share": float((g.trades > 0).mean())}
    print(sym, json.dumps(out[sym], indent=1))
json.dump(out, open("out/nb_explore.json", "w"), indent=1)
PY
