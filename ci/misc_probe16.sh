# Round 16: why the r16data web job hung. One request per source, 30 s hard limit, timings printed.
mkdir -p out/probe16
UA="fxlab-research/1.0 (research data collection)"
for k in VIX VIX3M VVIX SKEW GVZ OVX EVZ; do
  s=$(date +%s.%N); code=$(curl -s -o out/probe16/cboe_$k.csv -w "%{http_code} %{size_download}" -A "$UA" --max-time 30 "https://cdn.cboe.com/api/global/us_indices/daily_prices/${k}_History.csv"); e=$(date +%s.%N)
  echo "cboe $k $code $(echo "$e - $s" | bc) s; head: $(head -c 120 out/probe16/cboe_$k.csv | tr '\n' ' '); tail: $(tail -n 1 out/probe16/cboe_$k.csv)"
done
for k in VIXCLS VXVCLS GVZCLS OVXCLS EVZCLS DGS2 DGS10 NFCI USEPUINDXD; do
  s=$(date +%s.%N); code=$(curl -s -o out/probe16/fred_$k.csv -w "%{http_code} %{size_download}" -A "$UA" --max-time 30 "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$k"); e=$(date +%s.%N)
  echo "fred $k $code $(echo "$e - $s" | bc) s; head: $(head -c 80 out/probe16/fred_$k.csv | tr '\n' ' '); tail: $(tail -n 1 out/probe16/fred_$k.csv)"
done
rm -f out/probe16/*.csv
