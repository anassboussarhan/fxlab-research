"""Round 25 job layout (docs/round25_preregistration.md, "Jobs"): part P rule chunks per class group and timeframe,
the placebo + CF jobs, and the part AD jobs, sized from the round-23 unit timings (ci/r25_unit_cost.json: median
seconds of a 128-configuration LW unit per rule and timeframe) times the round-25 cost ratio measured on synthetic
bars (1.6), plus the selection time per unit.

A rule x group cell above one job's budget is split by instruments (--insts); those jobs save the units' aggregates
and the combine makes the panel streams.
Usage: python ci/r25_matrix.py [--alt-universe FILE] -> ci/r25_P.json, ci/r25_AD.json (and ci/r25_ALT.json)
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

RATIO = 1.6                       # round-25 / round-23 seconds per configuration (synthetic bars, 192 configurations)
SELECT = 8.0                      # walk-forward selection per unit (nested-plus, diagnostics), seconds
PROCS, BUDGET, FILL = 4, 19500, 0.72
CAP = PROCS * BUDGET * FILL       # core-seconds a job is planned for
TFS = ["H1", "H2", "H4", "H8", "H12", "D1", "D2", "W1"]
N_PLACEBO = 10


def rules_of(tf):
    import intraday17 as I17
    return [e.id for e in I17.strategies() if not (e.id == "S.RG.HMM" and tf == "H1")]


def unit_cost(cost, tf, key, hist=1.0):
    c = cost.get(tf, {})
    med = sorted(v for k, v in c.items() if not k.startswith("PLACEBO"))
    base = c.get(key, med[len(med) // 2] if med else 30.0)
    return RATIO * base * hist + SELECT


def pack(items, cap):
    """First-fit decreasing: items [(key, cost)] -> bins [[keys], cost]."""
    bins = []
    for k, c in sorted(items, key=lambda x: -x[1]):
        for b in bins:
            if b[1] + c <= cap:
                b[0].append(k)
                b[1] += c
                break
        else:
            bins.append([[k], c])
    return bins


def p_jobs(cost, groups, part="P", hist=None):
    inc = []
    for g, syms in groups.items():
        for tf in TFS if part == "P" else ["H1", "H2", "H4", "H8", "H12", "D1"]:
            h = 1.0 if hist is None else hist
            cells, big = [], []
            for k in rules_of(tf):
                c = unit_cost(cost, tf, k, h) * len(syms)
                (big if c > CAP else cells).append((k, c))
            for i, (keys, c) in enumerate(pack(cells, CAP)):
                inc.append({"part": part, "group": g, "tf": tf, "rules": ",".join(keys), "insts": "", "placebos": 0,
                            "cf": 0, "tag": f"{g}-{tf}-{i:02d}", "est_core_s": round(c)})
            for k, c in big:                                               # split by instruments
                nparts = math.ceil(c / CAP)
                for j in range(nparts):
                    sub = syms[j::nparts]
                    inc.append({"part": part, "group": g, "tf": tf, "rules": k, "insts": ",".join(sub), "placebos": 0,
                                "cf": 0, "tag": f"{g}-{tf}-{k.split('.')[-1]}-s{j}", "est_core_s": round(c / nparts)})
            pc = unit_cost(cost, tf, "PLACEBO_00", h) * N_PLACEBO * len(syms)
            cf = (sum(cost.get(tf, {}).get(k, 30.0) for k in rules_of(tf)) / 128.0 + 128 * 0.15 + SELECT) * 2 * len(syms)
            inc.append({"part": part, "group": g, "tf": tf, "rules": "NONE", "insts": "", "placebos": N_PLACEBO,
                        "cf": 1 if part == "P" else 0, "tag": f"{g}-{tf}-plc", "est_core_s": round(pc + (cf if part == "P" else 0))})
    return inc


def ad_jobs(cost, ad_rules, insts):
    inc = []
    for tf in ("H1", "H4", "D1"):
        items = [(k, (unit_cost(cost, tf, k) - SELECT) * 576 / 128 * len(insts) + 60 * len(insts)) for k in ad_rules]
        items.append(("PLACEBO_AD", (unit_cost(cost, tf, "PLACEBO_00") - SELECT) * 4.5 * len(insts)))
        for i, (keys, c) in enumerate(pack(items, CAP)):
            plc = "PLACEBO_AD" in keys
            inc.append({"tf": tf, "rules": ",".join(k for k in keys if k != "PLACEBO_AD") or "NONE",
                        "placebo": 1 if plc else 0, "insts": ",".join(insts), "tag": f"AD-{tf}-{i:02d}",
                        "est_core_s": round(c)})
    return inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt-universe", default="")
    a = ap.parse_args()
    cost = json.load(open(os.path.join(ROOT, "ci", "r25_unit_cost.json")))
    GR = json.load(open(os.path.join(ROOT, "ci", "r25_groups.json")))
    if not a.alt_universe:
        P = p_jobs(cost, GR["groups"])
        json.dump({"include": sorted(P, key=lambda j: -j["est_core_s"])}, open(os.path.join(ROOT, "ci", "r25_P.json"), "w"),
                  indent=0)
        ad_rules = json.load(open(os.path.join(ROOT, "ci", "r25_ad_rules.json")))["rules"]
        g10 = [s for s in GR["groups"]["FX"] if s[:3] in "AUDCADCHFEURGBPJPYNZDUSD" and len(s) == 6
               and s[:3] in ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD")
               and s[3:] in ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD")] + ["XAUUSD", "XAGUSD"]
        AD = ad_jobs(cost, ad_rules, g10)
        json.dump({"include": sorted(AD, key=lambda j: -j["est_core_s"])}, open(os.path.join(ROOT, "ci", "r25_AD.json"), "w"),
                  indent=0)
        tot = sum(j["est_core_s"] for j in P)
        print("P jobs", len(P), "core-hours", round(tot / 3600), "split cells",
              sorted({j["rules"] for j in P if j["insts"]}), "| AD jobs", len(AD), "instruments", len(g10),
              "core-hours", round(sum(j["est_core_s"] for j in AD) / 3600))
    else:
        U = json.load(open(a.alt_universe))
        syms = sorted(U["members"])
        # coins have about 4 years of history against about 19 for the FX pairs
        A = p_jobs(cost, {"ALT": syms}, part="ALT", hist=0.25)
        json.dump({"include": sorted(A, key=lambda j: -j["est_core_s"])}, open(os.path.join(ROOT, "ci", "r25_ALT.json"), "w"),
                  indent=0)
        print("ALT jobs", len(A), "coins", len(syms), "core-hours", round(sum(j["est_core_s"] for j in A) / 3600))


if __name__ == "__main__":
    main()
