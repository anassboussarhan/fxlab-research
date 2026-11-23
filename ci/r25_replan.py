"""Round 25 reruns (addendum 3): jobs that reached their time budget left rules unfinished. From every job's meta
(P_meta / AD_meta), each unfinished rule x group x timeframe (P, ALT) or unit (AD) is rerun, sized from the seconds
per unit actually observed:
- P / ALT: the rule's observed seconds per unit in that job (or, with none finished, the job's slowest finished unit
  x 2); the group's instruments are split into parts (--insts) so each part fits a job; split parts save aggregates
  and the combine makes the panels.
- AD: the lost units by rule, split by instruments the same way.
Usage: python ci/r25_replan.py --inp DIR --matrix-p ci/r25_P.json --matrix-ad ci/r25_AD.json --out DIR
"""
import argparse
import glob
import json
import math
import os

CAP = 4 * 19500 * 0.6


def parts_of(syms, per_unit):
    k = max(1, math.ceil(len(syms) * per_unit / CAP))
    return [syms[j::k] for j in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--groups", default="ci/r25_groups.json")
    ap.add_argument("--universe", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    GR = json.load(open(a.groups))
    alt = sorted(json.load(open(a.universe))["members"]) if a.universe and os.path.exists(a.universe) else []
    P, AD, rep = [], [], []
    for f in sorted(glob.glob(os.path.join(a.inp, "**", "P_meta_*.json"), recursive=True)):
        d = json.load(open(f))
        inc = d.get("incomplete_keys", [])
        if not inc:
            continue
        um = d.get("unit_meta", {})
        secs = {}
        for k, v in um.items():
            secs.setdefault(k.split("|")[0], []).append(v.get("seconds", 0) + v.get("select_seconds", 0))
        slow = max((max(v) for v in secs.values()), default=600.0)
        g, tf = d["group"], d["tf"]
        syms = alt if g == "ALT" else list(GR["groups"][g])
        part = "ALT" if g == "ALT" else "P"
        for k in inc:
            per = (sorted(secs[k])[len(secs[k]) // 2] * 1.3) if k in secs else 2.0 * slow
            plc = k.startswith("PLACEBO") or k in ("CF", "CF_PLACEBO")
            pp = parts_of(syms, per)
            for j, sub in enumerate(pp):
                P.append({"part": part, "group": g, "tf": tf, "rules": "NONE" if plc else k,
                          "insts": "" if len(pp) == 1 else ",".join(sub),
                          "placebos": 10 if k.startswith("PLACEBO") else 0, "cf": 1 if k.startswith("CF") else 0,
                          "only_keys": k if plc else "", "tag": f"{d['tag']}-r-{k.split('.')[-1]}-{j}",
                          "est_core_s": round(per * len(sub))})
            rep.append({"job": d["tag"], "key": k, "sec_per_unit": round(per), "parts": len(parts_of(syms, per))})
    for f in sorted(glob.glob(os.path.join(a.inp, "**", "AD_meta_*.json"), recursive=True)):
        d = json.load(open(f))
        lost = d.get("lost", [])
        if not lost:
            continue
        um = d.get("unit_meta", {})
        secs = {}
        for k, v in um.items():
            secs.setdefault(k.split("|")[0], []).append(v.get("seconds", 0))
        slow = max((max(v) for v in secs.values()), default=3600.0)
        by = {}
        for u in lost:
            k, s = u.split("|")
            by.setdefault(k, []).append(s)
        for k, syms in by.items():
            per = (max(secs[k]) * 1.2) if k in secs else 2.0 * slow
            for j, sub in enumerate(parts_of(sorted(syms), per)):
                AD.append({"tf": d["tf"], "rules": "NONE" if k == "PLACEBO_AD" else k, "placebo": 1 if k == "PLACEBO_AD" else 0,
                           "insts": ",".join(sub), "tag": f"AD-{d['tf']}-r-{k.split('.')[-1]}-{j}", "est_core_s": round(per * len(sub))})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"include": P}, open(os.path.join(a.out, "r25_P2.json"), "w"), indent=0)
    json.dump({"include": AD}, open(os.path.join(a.out, "r25_AD2.json"), "w"), indent=0)
    json.dump(rep, open(os.path.join(a.out, "replan_report.json"), "w"), indent=0)
    print("P/ALT rerun jobs", len(P), "AD rerun jobs", len(AD))
    print(json.dumps({"P": P, "AD": AD}))


if __name__ == "__main__":
    main()
