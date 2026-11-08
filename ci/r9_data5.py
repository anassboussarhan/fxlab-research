"""Round 9 data, fifth pass: Norges Bank monthly FX-purchase press releases (S3) and BoJ quarterly schedules of
outright JGB purchases (S4). Writes norges_fx.csv and boj_rinban_dates.csv. Usage: python ci/r9_data5.py OUTDIR"""
import io, os, re, subprocess, sys, tempfile, time
import pandas as pd, requests
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
def get(u, t=60):
    for k in range(3):
        try:
            r = requests.get(u, headers=UA, timeout=t)
            if r.status_code == 200: return r
            if r.status_code == 404: return None
        except Exception: pass
        time.sleep(3)
    return None
def text(h):
    h = re.sub(r"<script.*?</script>|<style.*?</style>", " ", h, flags=re.S)
    import html as H
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", h)))
# ---------------- Norges Bank
NB = "https://www.norges-bank.no"
links = set()
for y in range(2008, 2026):
    for p in range(1, 30):
        r = get(f"{NB}/en/news-events/news/?tab=newslist&newstype=0&year={y}&p={p}")
        if r is None: break
        new = set(re.findall(rf'href="(/en/news-events/news[^"]*/Press-releases/{y}/[^"]+)"', r.text)) - links
        if not new: break
        links |= new
        time.sleep(0.3)
print("norges press releases", len(links), flush=True)
rows = []
for l in sorted(links):
    r = get(NB + l)
    if r is None: continue
    t = text(r.text)
    if not re.search(r"foreign exchange (purchases|transactions)|purchase[s]? of (foreign currency|NOK)|sell(ing)? (foreign currency|NOK)", t, re.I):
        continue
    title = re.search(r"Norges Bank'?s? (?:foreign exchange|FX) (?:purchases|transactions)[^.|]{0,60}", t, re.I)
    am = re.search(r"(purchas\w*|buy\w*|sell\w*)\s+(?:foreign currency|NOK|the equivalent of NOK|currency)[^.]{0,160}?NOK\s*([\d,\.]+)\s*(million|billion)\s*(?:per|each|a)\s*(?:trading )?day", t, re.I)
    zero = re.search(r"no (?:foreign exchange|currency) (?:purchases|transactions)|will not (?:purchase|buy|sell)", t, re.I)
    rows.append({"url": l, "title": title.group(0)[:120] if title else "", "verb": am.group(1) if am else ("none" if zero else ""),
                 "amount": am.group(2) if am else "", "unit": am.group(3) if am else "", "snippet": (am.group(0) if am else t[:300])[:300]})
    time.sleep(0.2)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "norges_fx_raw.csv"), index=False)
print("norges fx releases", len(rows), flush=True)
# ---------------- BoJ quarterly schedules
BJ = "https://www.boj.or.jp"
r = get(BJ + "/en/mopo/measures/mkt_ope/ope_f/index.htm")
pdfs = []
if r is not None:
    for m in re.finditer(r'<a[^>]+href="([^"]+\.pdf)"[^>]*>(.*?)</a>', r.text, re.S):
        lab = text(m.group(2))
        if re.search(r"schedule", lab, re.I):
            pdfs.append((m.group(1), lab))
print("boj schedule pdfs", len(pdfs), flush=True)
MON = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
dates = []
for href, lab in pdfs:
    u = href if href.startswith("http") else BJ + href
    r = get(u)
    if r is None: continue
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(r.content); f.flush()
        txt = subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True).stdout
    yr = re.search(r"(20\d\d)", lab)
    q = re.search(r"\((January|April|July|October)-(March|June|September|December) (20\d\d)\)", lab)
    if not q: continue
    y = int(q.group(3)); m0 = MON[q.group(1)]; months = [m0, m0 + 1, m0 + 2]
    for mm, dd in re.findall(r"\b(\d{1,2})/(\d{1,2})\b", txt):
        mm, dd = int(mm), int(dd)
        if mm in months:
            try: dates.append({"date": pd.Timestamp(y, mm, dd), "src": u})
            except Exception: pass
    for mon, dd in re.findall(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})\b", txt):
        mm = [v for k, v in MON.items() if k.startswith(mon)][0]
        if mm in months:
            try: dates.append({"date": pd.Timestamp(y, mm, int(dd)), "src": u})
            except Exception: pass
    time.sleep(0.3)
D = pd.DataFrame(dates)
if len(D):
    D = D.drop_duplicates("date").sort_values("date")
    D.to_csv(os.path.join(OUT, "boj_rinban_dates.csv"), index=False)
    print("boj dates", len(D), D.date.min(), D.date.max()); print(D.groupby(D.date.dt.year).size().to_string())
