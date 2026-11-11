# Round 18 data probe: long daily FX history (1971 on) and long 3-month interest rates, reachable from the runner?
mkdir -p out/probe18
UA="fxlab-research/1.0 (research data collection)"
probe() { s=$(date +%s.%N); r=$(curl -sS -L -o /tmp/p.out -w "%{http_code} %{size_download}" -A "$UA" --max-time 60 "$1" 2>&1); e=$(date +%s.%N); echo "$2 | $r | $(echo "$e - $s" | bc)s | head: $(head -c 260 /tmp/p.out | tr '\n' ' ') | tail: $(tail -c 160 /tmp/p.out | tr '\n' ' ')"; }
probe "https://stooq.com/q/d/l/?s=usdjpy&i=d" stooq_usdjpy
probe "https://stooq.com/q/d/l/?s=gbpusd&i=d" stooq_gbpusd
probe "https://api.db.nomics.world/v22/series/FED/H10?limit=50&observations=0" dbnomics_fed_h10_list
probe "https://api.db.nomics.world/v22/datasets/FED" dbnomics_fed_datasets
probe "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H10&series=60f32914ab61dfab590e0e470153e3ae&lastobs=&from=01/01/1971&to=12/31/2025&filetype=csv&label=include&layout=seriescolumn" fed_ddp_h10
probe "https://www.federalreserve.gov/releases/h10/hist/dat00_ja.htm" fed_h10_hist_ja
probe "https://www.federalreserve.gov/releases/h10/hist/dat89_ja.htm" fed_h10_hist_ja89
probe "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1975&Dateto=31/Dec/2025&SeriesCodes=XUDLUSS,XUDLJYS,XUDLDMS,XUDLERS,XUDLSFS,XUDLCDS,XUDLADS,XUDLNDS&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N" boe_iadb
probe "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXJPUS" fred_dexjpus
probe "https://api.db.nomics.world/v22/series/OECD/MEI/USA.IR3TIB01.ST.M?observations=1" dbnomics_oecd_mei_us3m
probe "https://api.db.nomics.world/v22/series/OECD/MEI/JPN.IR3TIB01.ST.M?observations=1" dbnomics_oecd_mei_jp3m
probe "https://api.db.nomics.world/v22/series/OECD/MEI?dimensions=%7B%22SUBJECT%22%3A%5B%22IR3TIB01%22%5D%2C%22FREQUENCY%22%3A%5B%22M%22%5D%7D&limit=60&observations=0" dbnomics_oecd_mei_ir3_list
probe "https://api.db.nomics.world/v22/series/OECD/MEI?dimensions=%7B%22SUBJECT%22%3A%5B%22CCUSMA02%22%5D%2C%22FREQUENCY%22%3A%5B%22M%22%5D%7D&limit=60&observations=0" dbnomics_oecd_mei_fx_list
probe "https://api.db.nomics.world/v22/series/BIS/WS_XRU?dimensions=%7B%22FREQ%22%3A%5B%22D%22%5D%2C%22REF_AREA%22%3A%5B%22JP%22%5D%7D&limit=5&observations=0" dbnomics_bis_xru_daily
probe "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.JP.JPY.A?format=csv&startPeriod=1971-01-01" bis_xru_jpy
