# Data-source comparison and selection

## Evidence gathered in the build environment (2026-10-21)

Outbound network access from the build environment goes through an allow-list proxy. Every
candidate market-data host was probed directly; the proxy log recorded a policy denial (HTTP 403 on
CONNECT) for each of them:

| Host probed | Result |
|---|---|
| datafeed.dukascopy.com, www.dukascopy.com, freeserv.dukascopy.com | denied |
| www.histdata.com | denied |
| huggingface.co, cdn-lfs.huggingface.co, hf-mirror.com | denied |
| www.kaggle.com | denied |
| query1.finance.yahoo.com | denied |
| api-fxpractice.oanda.com | denied |
| www.truefx.com | denied |
| api.polygon.io, api.tiingo.com, data.nasdaq.com | denied |
| fred.stlouisfed.org, ecb.europa.eu, stooq.com, investing.com | denied |
| drive.google.com, storage.googleapis.com, s3.amazonaws.com, zenodo.org, figshare.com, archive.org | denied |
| github.com (anonymous git reads), pypi.org, conda | **allowed** |

The only reachable route to real intraday market data is therefore public GitHub repositories.
The `FX-Data` organisation publishes **original Dukascopy tick data** (one branch per
instrument-year, hourly CSV files with millisecond UTC timestamps, bid, ask, bid volume, ask volume)
produced by the `fx31337/fx-data-download-action`. Column order was verified on the data itself
(ask >= bid on 100% of ticks; UTC; Friday ~21:00 close / Sunday ~22:00 open pattern).

## Comparison against the selection criteria

Ratings use public documentation (Tickstory's Dukascopy date-range table; a 2024 vendor
comparison by NYC Servers) plus the probes above. "Reachable here" decides what this environment
could actually use.

| Criterion | Dukascopy (direct) | Dukascopy via FX-Data mirror (GitHub) | TrueFX | HistData | OANDA v20 | Polygon / Tiingo | Kaggle / HF datasets |
|---|---|---|---|---|---|---|---|
| Historical depth | majors from 2003, exotics ~2007 | 2007-2018 (EURUSD to 2022) | May 2009 -> | ~2000-2010 -> (pair-dependent) | candles from ~2005 | varies, paid tiers | arbitrary snapshots |
| Granularity | tick | tick | tick | tick + M1 | S5..M (no ticks) | aggregates | usually M1-D1 |
| Bid/ask | yes | yes | yes | M1 bars are single-sided | bid/ask/mid candles | plan dependent | rarely |
| Spread history | yes (from ticks) | yes | yes | tick files only | candle-level | limited | rarely |
| Instruments | 30+ FX, metals, CFDs | 34 of the requested 38 (no MXN, ZAR, TRY, CNH) | ~15 majors/crosses | 60+ FX + some CFDs | account instruments | broad | arbitrary |
| Timestamp reliability | UTC ms | UTC ms (verified) | UTC ms | EST without DST (needs conversion) | UTC | UTC | unknown provenance |
| Licensing | free for research; no redistribution | same underlying terms; mirror is third party | free with registration | free personal use | demo/live account | paid | per dataset |
| Cost | free | free | free | free (FTP paid) | free with account | paid | free |
| Reachable here | **no** | **yes** | no | no | no | no | no |

## Selection

1. **Primary (used):** Dukascopy tick bid/ask via the FX-Data GitHub mirror, aggregated in-house
   to M1 bid/ask bars with tick counts, up/down quote ticks, spread statistics and quoted liquidity
   (`fxlab/data/acquire_fxdata_git.py`). Real tick-level bid/ask beats every reachable alternative on
   granularity, spread realism and timestamp quality.
2. **Recommended next step on an unrestricted machine:** run `fxlab/data/acquire_dukascopy.py`
   (direct bi5 downloader, same output format) to extend coverage to 2003-2025 and to the four missing
   instruments, then pull TrueFX ticks (2009 ->) as the independent second source required for the
   data-source sensitivity checks.
3. **Not used:** daily-only sources (Yahoo, ECB, FRED), single-sided M1 sources (HistData M1) as the
   primary dataset, and opaque Kaggle/HF re-uploads.

## Consequences that the research results inherit

* Development data that exists here: 2007-2018 for most instruments, 2007-2022 for EURUSD, 2011-2018
  gold, 2010-2018 silver. **2019-2025 are not available in this environment.**
* USDMXN, USDZAR, USDTRY, USDCNH are not in the mirror.
* No interest-rate, bond, equity-index or volatility-index data is reachable, so carry
  (S.XA.CARRY_FILTER) is implemented but blocked, swaps are modelled with a conservative constant,
  and macro explanatory inputs are limited to the FX/metal panel itself.
* The 2026 holdout cannot be evaluated here at all (the data does not exist in this environment);
  the loaders additionally refuse any 2026 timestamp until candidates are frozen and unlocked.

## Interest rates (added in round 2)
* **Series:** OECD Main Economic Indicators 3-month interbank rates, monthly averages, percent, as published by
  FRED (`IR3TIB01{US,EZ,GB,JP,CH,CA,AU,NZ}M156N`), 2006-01 to 2025-12. 2026 values exist at the source and were
  deliberately **not stored** (holdout).
* **Access:** the shell cannot reach fred.stlouisfed.org (proxy 403). The FRED table pages
  (`fred.stlouisfed.org/data/<series>.txt`) were read with the web-fetch tool, which returns a model-written
  summary; values were requested as a year x month table rounded to 2 decimals and stored verbatim in
  `data/rates/ir3m_raw.txt`. Spot checks against well-known levels (e.g. euro 3M rate 3.29% in Dec-2008, CHF
  negative from Jan-2015, US ~0.1-0.3% in 2010-2015) are consistent, but a transcription error in an individual
  month cannot be ruled out; carry ranking and financing are insensitive to 0.01% rounding.
* **Point in time:** month m's average is used only from the first day of month m+1 (`fxlab.data.rates.known_at`,
  tested in `tests/test_execution.py::test_rates_are_point_in_time`).
* **Use:** (1) carry signals; (2) optional real overnight financing (`financing="rates"`): a long position earns
  base-minus-quote differential minus a 0.75% p.a. broker mark-up per side (scaled with cost stress), a short pays
  it; metals keep the flat 0.5 pip/night charge.
