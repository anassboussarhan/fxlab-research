"""Norges Bank monthly FX transaction announcements by URL guessing (the listing does not filter by year).
Writes norges_fx.csv: release date, month it applies to, signed daily amount (NOK million; + = Norges Bank buys
foreign currency, i.e. sells NOK). Usage: python ci/r9_norges.py OUTDIR"""
import html as H, os, re, sys, time
import pandas as pd, requests
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
NB = "https://www.norges-bank.no/en/news-events/news/Press-releases"
MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
S = requests.Session(); S.headers.update(UA)
def text(h):
    h = re.sub(r"<script.*?</script>|<style.*?</style>", " ", h, flags=re.S)
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", h)))
rows = []
for y in range(2008, 2026):
    for m in range(1, 13):
        days = [d for d in pd.bdate_range(pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) - pd.Timedelta(days=6),
                                          pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0))][::-1][:4]
        hit = None
        for d in days:
            for slug in (f"{d:%Y-%m-%d}-fx", f"{d:%Y-%m-%d}-fx-transactions", f"Press-release-{d.day}-{MON[m-1]}-{y}",
                         f"Press-release-{d.day}-{MON[m-1]}"):
                try:
                    r = S.get(f"{NB}/{y}/{slug}/", timeout=30)
                except Exception:
                    continue
                if r.status_code == 200 and re.search(r"foreign exchange|foreign currency", r.text, re.I):
                    hit = (d, slug, text(r.text)); break
            if hit: break
        if not hit:
            print("miss", y, m, flush=True); continue
        d, slug, t = hit
        am = re.search(r"(purchas\w*|buy\w*|sell\w*)\s+(foreign exchange|foreign currency|currency|NOK|kroner)[^.]{0,200}?NOK\s*([\d,\.]+)\s*(million|billion)[^.]{0,40}?(?:per|each|a)\s*(?:trading\s*)?day", t, re.I)
        none = re.search(r"(will not (purchase|buy|sell|make)|no (foreign exchange|currency) (purchases|transactions|sales)|not (purchase|buy|sell) any)", t, re.I)
        if am:
            v = float(am.group(3).replace(",", "")) * (1000 if am.group(4).lower() == "billion" else 1)
            buy = am.group(1).lower().startswith(("purchas", "buy"))
            fc = not am.group(2).lower().startswith(("nok", "kroner"))
            sign = 1 if (buy == fc) else -1
            amt = sign * v
        elif none:
            amt = 0.0
        else:
            amt = None
        rows.append({"release": d.date(), "applies_to": (d + pd.offsets.MonthBegin(1)).strftime("%Y-%m"), "amount": amt,
                     "slug": slug, "snippet": (am.group(0) if am else t[t.find("Norges Bank will"):][:250] if "Norges Bank will" in t else t[200:450])[:250]})
        print(y, m, slug, amt, flush=True)
        time.sleep(0.2)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "norges_fx.csv"), index=False)
print("norges rows", len(rows))
