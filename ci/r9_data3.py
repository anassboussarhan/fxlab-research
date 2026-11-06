"""Round 9 data, third pass: FRED (longer timeout), GDELT (slower, retry on 429), BoJ archive index probe."""
import json, os, sys, time
import requests
UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
OUT = sys.argv[1]; LOG = {}
def get(url, tries=4, wait=60, **kw):
    err = ""
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=120, **kw)
            if r.status_code == 429:
                time.sleep(wait); continue
            return r
        except Exception as ex:
            err = repr(ex)[:200]; time.sleep(10)
    class R: status_code, content = -1, err.encode()
    return R()
def save(rel, c):
    p = os.path.join(OUT, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(c); return len(c)
for s in ["DEXJPUS", "DEXUSUK", "DEXSZUS", "DEXCAUS", "DEXUSAL", "DEXUSNZ", "DEXSDUS", "DEXNOUS", "DEXDNUS", "DEXGEUS", "DEXUSEU", "DEXCHUS"]:
    r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": s})
    LOG.setdefault("fred", {})[s] = [r.status_code, save(f"fred/{s}.csv", r.content) if r.status_code == 200 else r.content[:200].decode(errors="ignore")]
    print("fred", s, LOG["fred"][s][:1], flush=True)
G = {"US": "US", "DE": "GM", "FR": "FR", "JP": "JA", "UK": "UK", "CH": "SZ", "CA": "CA", "AU": "AS", "NZ": "NZ"}
for name, code in G.items():
    q = f'(economy OR inflation OR "interest rates" OR "central bank") sourcecountry:{code}'
    time.sleep(20)
    r = get("https://api.gdeltproject.org/api/v2/doc/doc", params={"query": q, "mode": "timelinetone", "format": "csv",
            "startdatetime": "20170101000000", "enddatetime": "20251231235959"})
    LOG.setdefault("gdelt", {})[name] = [r.status_code, save(f"gdelt/{name}.csv", r.content)]
    print("gdelt", name, LOG["gdelt"][name], flush=True)
for k, u in {"index": "https://www.boj.or.jp/en/statistics/boj/fm/ope/m_release/index.htm",
             "y2009": "https://www.boj.or.jp/en/statistics/boj/fm/ope/m_release/2009/ope0904.htm",
             "y2016": "https://www.boj.or.jp/en/statistics/boj/fm/ope/m_release/2016/index.htm",
             "ope_index": "https://www.boj.or.jp/en/statistics/boj/fm/ope/index.htm"}.items():
    r = get(u, tries=1)
    LOG.setdefault("boj", {})[k] = [r.status_code, save(f"boj/{k}.htm", r.content)]
json.dump(LOG, open(os.path.join(OUT, "r9_data3_log.json"), "w"), indent=1)
