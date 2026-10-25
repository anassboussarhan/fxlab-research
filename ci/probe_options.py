"""Probe free historical options-chain sources from the GitHub runner (no account, no payment).

1. DoltHub post-no-preference/options through the public SQL API: tables, columns, symbols, dates, sampling.
2. Hugging Face dataset search for SPY / options chain datasets.
3. Reachability of a few other free sample pages.
Writes probe_results/options/*.txt|json."""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

OUT = "probe_results/options"
os.makedirs(OUT, exist_ok=True)
API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
LOG = []


def get(url, timeout=90):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fxlab-research-probe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, body, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return None, repr(e).encode(), time.time() - t0


def sql(name, q, timeout=120):
    st, body, dt = get(API + "?q=" + urllib.parse.quote(q), timeout)
    try:
        js = json.loads(body)
    except Exception:  # noqa: BLE001
        js = {"raw": body[:2000].decode("utf-8", "replace")}
    rec = {"name": name, "query": q, "http": st, "seconds": round(dt, 1),
           "status": js.get("query_execution_status"), "message": js.get("query_execution_message"),
           "schema": js.get("schema"), "rows": (js.get("rows") or [])[:200]}
    if "raw" in js:
        rec["raw"] = js["raw"]
    LOG.append(rec)
    with open(f"{OUT}/dolt_{name}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(name, st, rec["status"], rec["message"], f"{dt:.1f}s", str(rec["rows"][:3])[:400], flush=True)
    return rec


sql("tables", "SHOW TABLES")
tabs = [list(r.values())[0] for r in LOG[-1]["rows"]]
for t in tabs:
    sql(f"describe_{t}", f"DESCRIBE `{t}`")
    sql(f"sample_{t}", f"SELECT * FROM `{t}` LIMIT 5")
if "option_chain" in tabs:
    sql("spy_first", "SELECT * FROM option_chain WHERE act_symbol='SPY' ORDER BY date LIMIT 3")
    sql("spy_last", "SELECT * FROM option_chain WHERE act_symbol='SPY' ORDER BY date DESC LIMIT 3")
    sql("spy_dates_2019", "SELECT DISTINCT date FROM option_chain WHERE act_symbol='SPY' AND date < '2020-01-01' "
        "ORDER BY date LIMIT 200")
    sql("spy_dates_2023", "SELECT DISTINCT date FROM option_chain WHERE act_symbol='SPY' AND date >= '2023-01-01' "
        "AND date < '2023-04-01' ORDER BY date LIMIT 200")
    sql("spy_one_day", "SELECT * FROM option_chain WHERE act_symbol='SPY' AND date = "
        "(SELECT MIN(date) FROM option_chain WHERE act_symbol='SPY' AND date >= '2023-03-01') LIMIT 200")
    sql("spy_expiries_one_day", "SELECT expiration, COUNT(*) n, MIN(strike), MAX(strike) FROM option_chain "
        "WHERE act_symbol='SPY' AND date = (SELECT MIN(date) FROM option_chain WHERE act_symbol='SPY' AND "
        "date >= '2023-03-01') GROUP BY expiration ORDER BY expiration")
    for sym in ("QQQ", "IWM", "GLD", "TLT", "AAPL", "XSP", "SPX"):
        sql(f"has_{sym}", f"SELECT date, COUNT(*) n FROM option_chain WHERE act_symbol='{sym}' "
            "AND date >= '2023-03-01' GROUP BY date ORDER BY date LIMIT 3")
if "volatility_history" in tabs:
    sql("vol_spy_first", "SELECT * FROM volatility_history WHERE act_symbol='SPY' ORDER BY date LIMIT 3")

# Hugging Face dataset search
for term in ("spy options", "options chain", "option chain", "spx options", "options eod"):
    st, body, dt = get("https://huggingface.co/api/datasets?search=" + urllib.parse.quote(term) + "&limit=50")
    with open(f"{OUT}/hf_{term.replace(' ', '_')}.json", "wb") as f:
        f.write(body)
    try:
        ids = [d.get("id") for d in json.loads(body)]
    except Exception:  # noqa: BLE001
        ids = body[:200]
    print("HF", term, st, ids, flush=True)

# other free sample pages (reachability only)
for url in ("https://www.dolthub.com/repositories/post-no-preference/options",
            "https://historicaloptiondata.com/free-data/",
            "https://www.optionsdx.com/",
            "https://datashop.cboe.com/",
            "https://www.cboe.com/us/options/market_statistics/historical_data/"):
    st, body, dt = get(url, 30)
    print("URL", url, st, len(body), f"{dt:.1f}s", flush=True)
    LOG.append({"name": "url", "url": url, "http": st, "bytes": len(body)})

with open(f"{OUT}/summary.json", "w") as f:
    json.dump([{k: v for k, v in r.items() if k != "rows"} for r in LOG], f, indent=1)
