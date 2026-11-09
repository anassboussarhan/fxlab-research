# Round 16: why the r16data web job hung (second probe): redirects, FRED errors, alternative sources.
mkdir -p out/probe16
UA="fxlab-research/1.0 (research data collection)"
probe() { s=$(date +%s.%N); r=$(curl -sS -L -o /tmp/p.out -w "%{http_code} %{size_download} %{url_effective} redirects=%{num_redirects}" -A "$UA" --max-time 40 "$1" 2>&1); e=$(date +%s.%N); echo "$2 | $r | $(echo "$e - $s" | bc)s | head: $(head -c 100 /tmp/p.out | tr '\n' ' ') | tail: $(tail -n 1 /tmp/p.out | head -c 100)"; }
probe "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv" cboeVIX
probe "https://cdn.cboe.com/api/global/us_indices/daily_prices/EVZ_History.csv" cboeEVZ
probe "https://cdn.cboe.com/api/global/us_indices/daily_prices/BXM_History.csv" cboeBXM
curl -sS -v -A "$UA" --max-time 20 -o /dev/null "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv" 2>&1 | grep -i "^< location\|^< HTTP" | head -5
curl -sS -v -A "$UA" --max-time 20 -o /dev/null "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2" 2>&1 | tail -8
probe "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2" fredDGS2
probe "https://www.chicagofed.org/-/media/publications/nfci/nfci-data-series-csv.csv" chicagoNFCI
probe "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv" EPUdaily
probe "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2015/all?type=daily_treasury_yield_curve&field_tdr_date_value=2015&page&_format=csv" treasury2015
python - <<'PY'
import yfinance as yf
for t in ["^VIX", "^VIX3M", "^VVIX", "^SKEW", "^GVZ", "^OVX", "^EVZ"]:
    try:
        h = yf.Ticker(t).history(start="2005-01-01", end="2026-01-01", auto_adjust=False)
        print("yahoo", t, len(h), h.index.min(), h.index.max())
    except Exception as ex:
        print("yahoo", t, "error", repr(ex)[:150])
PY
