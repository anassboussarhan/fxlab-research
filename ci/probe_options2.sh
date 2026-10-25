#!/usr/bin/env bash
# Inventory of the DoltHub options database after a local clone: symbols, dates, chain shape; SPY / DIA exported.
set -u
O=$GITHUB_WORKSPACE/probe_results/options2
mkdir -p $O
cd /tmp/d/options
q() { timeout 1500 dolt sql -q "$2" -r csv > $O/$1.csv 2> $O/$1.err; echo "$1 exit $? rows $(wc -l < $O/$1.csv)"; }
q vol_symbols "SELECT act_symbol, COUNT(*) n, MIN(date) first, MAX(date) last FROM volatility_history GROUP BY act_symbol"
q spy_dates "SELECT date, COUNT(*) n, COUNT(DISTINCT expiration) n_exp, MIN(strike) kmin, MAX(strike) kmax FROM option_chain WHERE act_symbol='SPY' GROUP BY date ORDER BY date"
q spy_all "SELECT * FROM option_chain WHERE act_symbol='SPY' AND date <= '2025-12-31'"
q dia_all "SELECT * FROM option_chain WHERE act_symbol='DIA' AND date <= '2025-12-31'"
q spy_vol "SELECT * FROM volatility_history WHERE act_symbol IN ('SPY','DIA','AAPL') AND date <= '2025-12-31'"
q chain_symbols "SELECT act_symbol, COUNT(DISTINCT date) d, MIN(date) first, MAX(date) last, COUNT(*) n FROM option_chain WHERE act_symbol IN ('SPY','QQQ','IWM','DIA','GLD','SLV','TLT','EEM','XLE','XLF','XLK','USO','HYG','EFA','FXI','EWZ','SMH','XBI','KRE','GDX','VXX','UVXY','ARKK','TQQQ','AAPL','MSFT','AMZN','NVDA','TSLA','META','GOOGL','JPM','XOM','KO','BAC','WMT','JNJ','PG','V','UNH','HD','DIS','NFLX','AMD','INTC','CSCO','PFE','T','VZ','BRK.B') GROUP BY act_symbol"
gzip -f $O/spy_all.csv $O/dia_all.csv
ls -la $O
