"""Round 24 access probe: HTTP status and size of one request to each candidate free source. Prints only status codes,
byte counts and content types: no data values are read. Usage: python ci/r24_probe.py"""
import json
import time

import requests

UA = {"User-Agent": "fxlab-research/1.0 (research data collection)"}
T = [
    ("sec_full_index", "https://www.sec.gov/Archives/edgar/full-index/2020/QTR1/form.idx"),
    ("sec_daily_index", "https://www.sec.gov/Archives/edgar/daily-index/2020/QTR1/"),
    ("sec_submissions_api", "https://data.sec.gov/submissions/CIK0000320193.json"),
    ("sec_insider_bulk", "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2020q1_form345.zip"),
    ("finra_short_volume", "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20200102.txt"),
    ("binance_cm_delivery", "https://data.binance.vision/data/futures/cm/monthly/klines/BTCUSD_201225/1h/BTCUSD_201225-1h-2020-10.zip"),
    ("binance_um_delivery", "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT_230331/1h/BTCUSDT_230331-1h-2023-01.zip"),
    ("binance_spot_1m", "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2021-01.zip"),
    ("okx_funding", "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=10"),
    ("okx_history_files", "https://www.okx.com/cdn/okex/traderecords/swaprate/monthly/202101/allswaprate-swaprate-2021-01.zip"),
    ("deribit_funding", "https://www.deribit.com/api/v2/public/get_funding_rate_history?instrument_name=BTC-PERPETUAL&start_timestamp=1609459200000&end_timestamp=1609545600000"),
    ("hyperliquid_funding", "POST https://api.hyperliquid.xyz/info"),
    ("bitmex_funding", "https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=10&reverse=false"),
    ("dydx_funding", "https://indexer.dydx.trade/v4/historicalFunding/BTC-USD?limit=10"),
    ("kraken_futures_funding", "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol=PF_XBTUSD"),
    ("gateio_funding", "https://api.gateio.ws/api/v4/futures/usdt/funding_rate?contract=BTC_USDT&limit=10"),
    ("wikipedia_sp500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    ("house_watcher", "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"),
    ("senate_watcher", "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"),
    ("fred_walcl", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL"),
    ("fred_wtregen", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN"),
    ("fred_rrp", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD"),
    ("dbnomics_sek_rate", "https://api.db.nomics.world/v22/series/OECD/MEI/SWE.IR3TIB01.ST.M?observations=1"),
    ("stooq_daily", "https://stooq.com/q/d/l/?s=spy.us&i=d"),
    ("yahoo_chart", "https://query1.finance.yahoo.com/v8/finance/chart/XLK?range=5d&interval=1d"),
]
out = {}
for name, url in T:
    t0 = time.time()
    try:
        if url.startswith("POST "):
            r = requests.post(url[5:], json={"type": "fundingHistory", "coin": "BTC", "startTime": 1672531200000},
                              headers=UA, timeout=60)
        else:
            r = requests.get(url, headers=UA, timeout=60, stream=True)
            body = r.raw.read(200000, decode_content=True)
        n = len(r.content) if url.startswith("POST ") else len(body)
        out[name] = {"status": r.status_code, "bytes_read": n, "type": r.headers.get("content-type", "")[:40],
                     "seconds": round(time.time() - t0, 1)}
    except Exception as ex:                                                # noqa: BLE001
        out[name] = {"error": repr(ex)[:160]}
    print(name, json.dumps(out[name]), flush=True)
    time.sleep(1)
json.dump(out, open("r24_probe.json", "w"), indent=1)
