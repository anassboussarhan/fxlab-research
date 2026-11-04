# Prop-firm rules from LuxAlgo's public, keyless directory API (the data behind luxalgo.com/prop-firms; MIT-licensed
# simulator github.com/LuxAlgo/prop-firm-sim). Only the factual rule fields are kept, as a CSV.
mkdir -p out/propfirms
python - <<'PY'
import json, csv, urllib.request
req = urllib.request.Request("https://app.luxalgo.com/api/propfirms/list",
                             headers={"User-Agent": "fxlab-research/1.0 (research data collection)"})
d = json.loads(urllib.request.urlopen(req, timeout=120).read())
firms = d.get("data", {}).get("propfirms", d.get("propfirms", []))
F = ["challengeId", "challengeName", "accountSize", "steps", "profitTarget", "profitTargetIsPercent", "minTradingDays",
     "dailyLoss", "dailyLossType", "dailyLossBasis", "dailyLossLimitBasis", "dailyLossIncludesOpenPnl",
     "dailyLossEvaluation", "dailyLossIsPercent", "maxLoss", "maxLossType", "maxLossMode", "maxLossLocksAtInitial",
     "maxLossLockOffset", "maxLossIsPercent", "lossIsPercent", "price", "interval", "activationFee", "resetFee",
     "profitSplitPercent", "payoutFrequency", "payoutIntervalDays", "firstPayoutMinDays", "payoutMinWinningDays",
     "payoutMaxPct", "payoutMaxAmount", "consistencyMaxBestDayPct", "fundedConsistencyPct", "isFeeRefundable",
     "sourceUrl", "lastVerifiedAt"]
rows = []
for f in firms:
    for c in f.get("challenges", []) or []:
        r = {"propfirmId": f.get("propfirmId"), "firm": f.get("name"), "productTypes": "|".join(f.get("productTypes") or []),
             "currency": f.get("currency")}
        for k in F:
            v = c.get(k)
            r[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        rows.append(r)
with open("out/propfirms/challenges.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
extra = sorted({k for f in firms for c in (f.get("challenges") or []) for k in c.keys()} - set(F))
json.dump({"firms": len(firms), "challenges": len(rows), "other_challenge_fields": extra,
           "firm_fields": sorted({k for f in firms for k in f.keys()})}, open("out/propfirms/summary.json", "w"), indent=1)
print(len(firms), "firms,", len(rows), "challenges")
PY
