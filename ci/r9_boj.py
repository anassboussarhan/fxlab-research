"""BoJ market operations archive -> dates of JGB outright purchase offers (S4). Usage: python ci/r9_boj.py OUTDIR"""
import io, os, re, sys, time, zipfile
import pandas as pd, requests
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
BASE = "https://www.boj.or.jp"
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
def get(u):
    for k in range(3):
        try:
            r = requests.get(u, headers=UA, timeout=60)
            if r.status_code == 200: return r.content
        except Exception: pass
        time.sleep(3)
    return None
links = []
for y in range(2007, 2026):
    if y <= 2010:
        links += [(y, f"/en/statistics/boj/fm/ope/m_release/{y}/ope{y % 100:02d}{m:02d}.htm") for m in range(1, 13)]
        continue
    h = get(f"{BASE}/en/statistics/boj/fm/ope/m_release/{y}/index.htm")
    if h:
        links += [(y, l) for l in sorted(set(re.findall(rf'href="(/en/statistics/boj/fm/ope/m_release/{y}/ope\d{{4}}\.(?:zip|xlsx|xls|htm))"', h.decode(errors="ignore"))))]
print("links", len(links), flush=True)
def frames(name, data):
    out = []
    if name.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data))
        for n in z.namelist():
            out += frames(n.lower(), z.read(n))
    elif name.endswith((".xlsx", ".xls")):
        try: out += list(pd.read_excel(io.BytesIO(data), sheet_name=None, header=None).values())
        except Exception as ex: print("xl", name, repr(ex)[:100])
    elif name.endswith((".htm", ".html")):
        try: out += pd.read_html(io.BytesIO(data))
        except Exception as ex: print("html", name, repr(ex)[:100])
    return out
rows = []
for y, l in links:
    data = get(BASE + l)
    if data is None:
        continue
    mm = int(re.search(r"ope\d{2}(\d{2})", l).group(1))
    for df in frames(l.lower(), data):
        cur_md = None
        for vals in df.astype(str).values.tolist():
            line = " ".join(vals)
            md = re.search(r"\b(\d{1,2})/(\d{1,2})\b", line)
            if md: cur_md = md
            if re.search(r"outright purchases? of (jgbs|japanese government bonds)", line, re.I) or \
               (re.search(r"outright", line, re.I) and re.search(r"JGB", line) and not re.search(r"T-Bill|Treasury Discount|T-bills", line, re.I)):
                if md:
                    m_, d_ = int(md.group(1)), int(md.group(2))
                    yy = y if not (mm == 1 and m_ == 12) else y - 1
                    try: rows.append({"date": pd.Timestamp(yy, m_, d_), "src": l, "text": line[:160]})
                    except Exception: pass
    time.sleep(0.3)
R = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
R.to_csv(os.path.join(OUT, "boj_rinban_dates.csv"), index=False)
print("rinban dates", len(R), R.date.min() if len(R) else None, R.date.max() if len(R) else None)
print(R.groupby(R.date.dt.year).size().to_string() if len(R) else "")
