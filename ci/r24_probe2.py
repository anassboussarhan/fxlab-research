"""Round 24 R10 source probe: the original publishers of the three FRED series (FRED times out on the runners).
Records status codes, sizes and the first header lines only (no data rows)."""
import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
FED = "https://www.federalreserve.gov/datadownload/Output.aspx"
C = [
    ("h41_walcl_mnemonic", FED + "?rel=H41&series=RESPPANWW&from=01/01/2002&to=12/31/2025&filetype=csv&label=include&layout=seriescolumn"),
    ("h41_walcl_alt", FED + "?rel=H41&series=RESPPANWW&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package"),
    ("h41_tga_wed", FED + "?rel=H41&series=RESPODTGA_N.WW&from=01/01/2002&to=12/31/2025&filetype=csv&label=include&layout=seriescolumn"),
    ("h41_tga_wk", FED + "?rel=H41&series=RESPODTGA_N.WW&filetype=csv&label=include&layout=seriescolumn"),
    ("h41_package", "https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H41"),
    ("dts_cash", "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance?filter=record_date:gte:2025-12-01&page[size]=3"),
    ("dts_cash_old", "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance?filter=record_date:eq:2015-06-03&page[size]=5"),
    ("nyfed_rrp", "https://markets.newyorkfed.org/api/rp/reverserepo/propositions/search.json?startDate=2025-12-01&endDate=2025-12-05"),
    ("nyfed_rrp_all", "https://markets.newyorkfed.org/api/rp/reverserepo/all/results/search.json?startDate=2025-12-01&endDate=2025-12-05"),
    ("fred_api_nokey", "https://api.stlouisfed.org/fred/series/observations?series_id=WALCL&file_type=json"),
]
for name, url in C:
    try:
        r = requests.get(url, headers=UA, timeout=60)
        txt = r.text
        head = " | ".join(l.strip()[:120] for l in txt.splitlines()[:6]) if "json" not in r.headers.get("content-type", "") else txt[:300]
        print(f"{name}: HTTP {r.status_code} bytes {len(r.content)} type {r.headers.get('content-type', '')[:40]}")
        print("   head:", head[:700])
    except Exception as ex:  # noqa: BLE001
        print(f"{name}: {ex!r}"[:300])
