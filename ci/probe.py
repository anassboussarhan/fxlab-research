"""Reachability probe for free statistics sources from the GitHub runner (FRED timed out on every request in runs
18 and 19). Fetches each URL once (30 s timeout) and stores status, size, time and the first 1,500 characters in
probe_results/probe.json, published to the `probe` branch. No data is used for research from here."""
import json
import os
import time
import urllib.request

URLS = [
    ("fred_csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"),
    ("dbn_oecd_mei_dataset", "https://api.db.nomics.world/v22/datasets/OECD/MEI"),
    ("dbn_oecd_mei_series3", "https://api.db.nomics.world/v22/series/OECD/MEI?limit=3&observations=0"),
    ("dbn_oecd_mei_usa_ip_a", "https://api.db.nomics.world/v22/series/OECD/MEI/USA.PRINTO01.IXOBSA.M?observations=1"),
    ("dbn_oecd_mei_usa_ip_b", "https://api.db.nomics.world/v22/series/OECD/MEI/PRINTO01.USA.IXOBSA.M?observations=1"),
    ("dbn_imf_ifs_series3", "https://api.db.nomics.world/v22/series/IMF/IFS?limit=3&observations=0"),
    ("dbn_imf_ifs_us_cpi", "https://api.db.nomics.world/v22/series/IMF/IFS/M.US.PCPI_IX?observations=1"),
    ("dbn_imf_ifs_us_ip", "https://api.db.nomics.world/v22/series/IMF/IFS/M.US.AIP_IX?observations=1"),
    ("dbn_imf_ifs_us_unemp", "https://api.db.nomics.world/v22/series/IMF/IFS/M.US.LUR_PT?observations=1"),
    ("dbn_eurostat_hicp", "https://api.db.nomics.world/v22/series/Eurostat/prc_hicp_midx?limit=2&observations=0"),
    ("wb_gem_series", "https://api.worldbank.org/v2/sources/15/series?format=json&per_page=50"),
    ("wb_gem_usa_cpi", "https://api.worldbank.org/v2/sources/15/country/USA/series/CPTOTSAXN/time/all?format=json&per_page=5"),
    ("wb_wdi_check", "https://api.worldbank.org/v2/country/USA/indicator/FP.CPI.TOTL?format=json&per_page=2"),
    ("oecd_sdmx_stes_flows", "https://sdmx.oecd.org/public/rest/dataflow/OECD.SDD.STES/all/latest?references=none"),
    ("ecb_hicp", "https://data-api.ecb.europa.eu/service/data/ICP/M.U2.N.000000.4.INX?format=csvdata&startPeriod=2024-01"),
    ("imf_new_api", "https://api.imf.org/external/sdmx/2.1/dataflow"),
    ("cboe_evz", "https://cdn.cboe.com/api/global/us_indices/daily_prices/EVZ_History.csv"),
]
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "*/*"}


def main():
    out = []
    for name, url in URLS:
        t0 = time.time()
        r = {"name": name, "url": url}
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=30) as f:
                body = f.read()
                r.update(status=f.status, bytes=len(body), head=body[:1500].decode("utf-8", "replace"))
        except Exception as ex:                                             # noqa: BLE001
            r.update(status=getattr(ex, "code", None), error=str(ex)[:300])
        r["seconds"] = round(time.time() - t0, 1)
        out.append(r)
        print(f"::notice title={name}::status={r.get('status')} bytes={r.get('bytes')} s={r['seconds']} "
              f"{(r.get('error') or '')[:120]}", flush=True)
    os.makedirs("probe_results", exist_ok=True)
    json.dump(out, open("probe_results/probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
