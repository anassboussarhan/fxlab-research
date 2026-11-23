"""Round 25 combine (docs/round25_preregistration.md).

- Rows of parts P / CF (P_rows), ALT (P_rows with part ALT) and AD (AD_rows); when the same file exists in two runs the
  later run's copy is used.
- Split cells (P_pack_*.npz of jobs that ran part of a group): the panel streams of every rule x timeframe x class whose
  instruments are all present, as the jobs make them for full groups.
- Counting: real streams of the counted kinds (default, nested_plus, panel_default, panel_nested, ad_top5); own rule
  (with the Newey-West condition, AD without P1); Benjamini-Hochberg at q = 0.10 on the iid t over all of them.
- Placebo rates per part; streams with 2019-2025 t above 8 flagged.
- Part HAC: the reruns of the round-23 passes (LW_series of r25HAC artifacts) against their published t, with the
  Newey-West t.
Usage: python scripts/r25_combine.py --inp DIR --out DIR [--hac ci/r25_hac.json] [--groups ci/r25_groups.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf22  # noqa: E402
import wf25  # noqa: E402

COUNTED = {"default", "nested_plus", "panel_default", "panel_nested", "ad_top5"}


def latest(inp, pat):
    """[(file, artifact)] with the later run's copy where a file exists in two runs (inp/RUN/ARTIFACT/...)."""
    best = {}
    for f in sorted(glob.glob(os.path.join(inp, "**", pat), recursive=True)):
        rel = os.path.relpath(f, inp).split(os.sep)
        run, rest = (rel[0], os.sep.join(rel[1:])) if rel[0].isdigit() else ("0", os.sep.join(rel))
        if rest not in best or int(run) > int(best[rest][0]):
            best[rest] = (run, f)
    return sorted((f, rest.split(os.sep)[0]) for rest, (_, f) in best.items())


def agg_from_pack(path):
    z = np.load(path)
    a = object.__new__(wf25.Agg25)
    a.dates = pd.DatetimeIndex(pd.to_datetime(z["dates"], unit="ns"))
    per = a.dates.to_period("M")
    mk = np.asarray(per.year * 12 + per.month - 1)
    a.mstart = np.flatnonzero(np.r_[True, mk[1:] != mk[:-1]])
    a.mkey = mk[a.mstart]
    a.X = z["X"].astype(np.float64)
    a.S, a.Q, a.A, a.V = (z[k].astype(np.float64) for k in ("S", "Q", "A", "V"))
    a.n = a.S.shape[0]
    zc = np.zeros((a.n, 1))
    a.cS, a.cQ = np.hstack([zc, np.cumsum(a.S, 1)]), np.hstack([zc, np.cumsum(a.Q, 1)])
    a.cA, a.cV = np.hstack([zc, np.cumsum(a.A, 1)]), np.hstack([zc, np.cumsum(a.V, 1)])
    a.nbr = z["nbr"] if z["nbr"].size else None
    a.rows = None
    return a


def panel_from_packs(inp, groups, members=None):
    """Panel rows of split cells: packs grouped by (key, tf); a class's panel only when every instrument is there."""
    packs = {}
    for f, art in latest(inp, "P_pack_*.npz"):
        b = os.path.basename(f)[len("P_pack_"):-4]
        key, tf, sym = b.rsplit("_", 2)
        prt = "P" if sym in groups["class"] else "ALT"
        packs.setdefault((key, tf, prt), {})[sym] = f
    rows, notes = [], []
    for (key, tf, part), byi in sorted(packs.items()):
        if part == "ALT":
            klass_of = {s: "ALT" for s in byi}
            need = {"ALT": sorted((members or {}).keys())}
        else:
            klass_of = {s: groups["class"][s] for s in byi}
            need = {}
            for g, syms in groups["groups"].items():
                for s in syms:
                    need.setdefault(groups["class"][s], []).append(s)
        for klass in sorted(set(klass_of.values())):
            want = need.get(klass, [])
            have = [s for s in want if s in byi]
            if len(have) < len(want):
                notes.append({"key": key, "tf": tf, "klass": klass, "missing": sorted(set(want) - set(have))})
                continue
            aggs = [agg_from_pack(byi[s]) for s in have]
            first = max(2010, min(int(a.dates.year.min()) for a in aggs) + 3)
            kw = {"min_active": 30, "min_valid": 50}
            if part == "ALT":
                mem = [list(members.get(s, [])) for s in have]
                pool_d, per_d, _ = wf25.panel_pit(aggs, mem, first, wf25.wf23.DEFAULT, **kw)
                pool_n, _ = wf25.panel_pit_nested(aggs, mem, first, wf25.SETTINGS23, **kw)
            else:
                pool_d, per_d, _ = wf25.panel(aggs, first, wf25.wf23.DEFAULT, **kw)
                pool_n, _ = wf25.panel_nested(aggs, first, wf25.SETTINGS23, **kw)
            base = {"part": part, "family": f"{part}-{key}", "unit": f"{key}|{tf}|{klass}", "tf": tf,
                    "symbol": f"PANEL:{klass}", "klass": klass, "placebo": "PLACEBO" in key, "n_inst": len(aggs),
                    "from_packs": True}
            for kind, s in (("panel_default", pool_d), ("panel_nested", pool_n)):
                st = wf25.stats(s)
                rows.append({**base, "selection": kind, "counted_kind": True, **st,
                             "own": wf25.counts_rule(st) if len(s) else False})
    return rows, notes


def hac_part(inp, hac_file):
    if not hac_file or not os.path.exists(hac_file):
        return []
    H = json.load(open(hac_file))
    ser = {}
    for f, art in latest(inp, "LW_series_*.parquet"):
        if not art.startswith("r25HAC-"):
            continue
        sym = os.path.basename(f)[len("LW_series_"):-8]
        x = pd.read_parquet(f)
        for c in x.columns:
            ser[(sym, c)] = x[c]
    out = []
    for u in H["units"]:
        s = ser.get((u["symbol"], f"{u['rule']}|{u['tf']}|{u['selection']}"))
        row = {"symbol": u["symbol"], "tf": u["tf"], "rule": u["rule"], "selection": u["selection"],
               "published_p2_t": u["p2_t"], "rerun": s is not None}
        if s is not None:
            s = s.dropna()
            s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
            st = wf25.stats(s)
            row.update({"rerun_p2_t": st["p2_t"], "p2_hac_t": st["p2_hac_t"],
                        "reproduced": bool(abs(st["p2_t"] - u["p2_t"]) <= 0.01)})
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hac", default="ci/r25_hac.json")
    ap.add_argument("--groups", default="ci/r25_groups.json")
    ap.add_argument("--universe", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = json.load(open(a.groups))
    members = None
    if a.universe and os.path.exists(a.universe):
        members = {s: list(map(int, ys)) for s, ys in json.load(open(a.universe))["members"].items()}
    parts = []
    for pat in ("P_rows_*.csv", "AD_rows_*.csv"):
        for f, art in latest(a.inp, pat):
            x = pd.read_csv(f)
            if len(x):
                x["artifact"] = art
                parts.append(x)
    R = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    prow, notes = panel_from_packs(a.inp, groups, members)
    if prow:
        R = pd.concat([R, pd.DataFrame(prow)], ignore_index=True)
    # one row per stream: a unit can appear in two jobs only through a rerun (the later copy is kept above)
    R = R.drop_duplicates(["part", "unit", "symbol", "selection", "placebo"], keep="last")
    R["placebo"] = R["placebo"].fillna(False).astype(bool)
    R["counted_kind"] = R["selection"].isin(COUNTED)
    real = R[~R.placebo & R.counted_kind & R.p2_t.notna()].copy()
    real["p"] = 1 - norm.cdf(real.p2_t.astype(float))
    p = real.p.to_numpy()
    o = np.argsort(p)
    m = max(len(p), 1)
    ok = np.flatnonzero(p[o] <= 0.10 * np.arange(1, len(p) + 1) / m)
    real["bh"] = False
    if len(ok):
        real.iloc[o[: ok.max() + 1], real.columns.get_loc("bh")] = True
    real["counts"] = real.own.astype(bool) & real.bh
    real["flag_t8"] = real.p2_t > 8
    R = R.merge(real[["p", "bh", "counts", "flag_t8"]], left_index=True, right_index=True, how="left")
    R.to_csv(os.path.join(a.out, "r25_streams.csv.gz"), index=False)
    keep = [c for c in ("part", "family", "unit", "tf", "symbol", "klass", "selection", "oos_first", "p1_mean", "p2_mean",
                        "p2_t", "p2_hac_t", "own", "bh") if c in R]
    R["sub"] = np.where(R.family.astype(str).str.contains("-CF"), "CF", R.part)
    by = []
    for prt, grp in R[R.p2_t.notna()].groupby("sub"):
        rl = grp[~grp.placebo & grp.counted_kind]
        pl = grp[grp.placebo & grp.counted_kind]
        by.append({"part": prt, "real_streams": int(len(rl)), "real_own": int(rl.own.sum()),
                   "real_own_share": float(rl.own.mean()) if len(rl) else None,
                   "real_max_t": float(rl.p2_t.max()) if len(rl) else None,
                   "placebo_streams": int(len(pl)), "placebo_own": int(pl.own.sum()),
                   "placebo_own_share": float(pl.own.mean()) if len(pl) else None,
                   "placebo_max_t": float(pl.p2_t.max()) if len(pl) else None,
                   "counts": int(real[real.index.isin(rl.index)].counts.sum())})
    H = hac_part(a.inp, a.hac)
    meta = {}
    for f, art in latest(a.inp, "P_meta_*.json") + latest(a.inp, "AD_meta_*.json"):
        d = json.load(open(f))
        meta[art] = {k: d.get(k) for k in ("units", "done", "final", "missing_bars", "seconds")} | \
            {"lost": len(d.get("lost", [])), "incomplete_keys": d.get("incomplete_keys", [])[:30]}
    summ = {"streams": int(len(R)), "real_counted_streams": int(len(real)), "own_rule": int(real.own.sum()),
            "bh": int(real.bh.sum()), "counts": int(real.counts.sum()),
            "first_bh_threshold_t": float(norm.ppf(1 - 0.10 / m)), "by_part": by,
            "flag_t8": real[real.flag_t8][keep].round(6).to_dict("records")[:100],
            "top_real": real.sort_values("p2_t", ascending=False).head(60)[keep].round(6).to_dict("records"),
            "counting": real[real.counts][keep].round(6).to_dict("records"),
            "split_cells_not_complete": notes, "jobs": meta,
            "hac_round23": {"units": len(H), "rerun": sum(h["rerun"] for h in H),
                            "reproduced": sum(bool(h.get("reproduced")) for h in H),
                            "median_ratio_hac_to_iid": float(np.nanmedian([h["p2_hac_t"] / h["rerun_p2_t"] for h in H
                                                                           if h.get("rerun") and h.get("rerun_p2_t")]))
                            if any(h.get("rerun") for h in H) else None,
                            "still_above_1645": sum(1 for h in H if h.get("reproduced") and h["p2_hac_t"] >= 1.645)}}
    pd.DataFrame(H).to_csv(os.path.join(a.out, "r25_hac_round23.csv"), index=False)
    json.dump(summ, open(os.path.join(a.out, "r25_summary.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(json.dumps({k: summ[k] for k in ("streams", "real_counted_streams", "own_rule", "bh", "counts",
                                           "first_bh_threshold_t", "hac_round23")}, indent=1, default=str))
    print(pd.DataFrame(by).to_string())
    print(real.sort_values("p2_t", ascending=False)[keep].head(30).to_string())


if __name__ == "__main__":
    main()
