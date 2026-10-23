# Research report

Generated 2026-10-22T11:47:57.978076+00:00 from files under `results/`.

## 1. Data actually used

| symbol | first_obs | last_obs | years | m1_bars |
|---|---|---|---|---|
| AUDJPY | 2007-03-30 18:00:00+00:00 | 2015-12-31 21:59:00+00:00 | 9 | 3143716 |
| AUDUSD | 2007-03-30 18:00:00+00:00 | 2018-05-23 09:59:00+00:00 | 12 | 3842372 |
| EURCHF | 2007-03-30 18:00:00+00:00 | 2018-06-01 20:59:00+00:00 | 12 | 3999252 |
| EURGBP | 2007-03-30 18:00:00+00:00 | 2018-06-11 03:59:00+00:00 | 12 | 4017100 |
| EURJPY | 2007-03-30 18:00:00+00:00 | 2018-06-01 20:59:00+00:00 | 12 | 3927632 |
| EURUSD | 2007-03-30 17:00:00+00:00 | 2022-07-08 20:59:00+00:00 | 16 | 5696627 |
| GBPJPY | 2007-03-30 18:00:00+00:00 | 2018-06-06 10:59:00+00:00 | 12 | 4014244 |
| GBPUSD | 2007-03-30 18:00:00+00:00 | 2018-06-06 10:59:00+00:00 | 12 | 4013277 |
| NZDUSD | 2007-03-30 18:00:00+00:00 | 2018-06-06 01:59:00+00:00 | 12 | 4007313 |
| USDCAD | 2007-03-30 18:00:00+00:00 | 2018-06-06 12:59:00+00:00 | 12 | 4011352 |
| USDCHF | 2007-03-30 18:00:00+00:00 | 2018-06-06 17:59:00+00:00 | 12 | 4006651 |
| USDJPY | 2007-03-30 18:00:00+00:00 | 2018-06-17 22:59:00+00:00 | 12 | 3985693 |
| XAGUSD | 2010-11-11 17:00:00+00:00 | 2018-06-19 09:59:00+00:00 | 9 | 2489027 |
| XAUUSD | 2011-05-10 09:00:00+00:00 | 2018-06-19 10:59:00+00:00 | 8 | 2407951 |

## 2. Engineering completeness

| kind | registered | by state |
|---|---|---|
| feature | 266 | {'implemented_tested': 266} |
| strategy | 67 | {'implemented_tested': 66, 'blocked_missing_dependency': 1} |
| model | 69 | {'implemented_tested': 69} |

Leakage tests: {'feature': {'passed': 266, 'failed': 0, 'not_run': 0, 'blocked': 0}, 'strategy': {'passed': 66, 'failed': 0, 'not_run': 0, 'blocked': 1}, 'model': {'passed': 69, 'failed': 0, 'not_run': 0, 'blocked': 0}}

## 3. Compute

Machine: {'cpus': 2, 'ram_gb': 8.4, 'gpu': 'none'}. All 266 features on 12501 H1 bars: 378.0 s; all 69 model families, one fit on 10k x 20: 459.4 s. Full-campaign estimate: see `docs/compute_plan.md`.

## 4. Strategy campaign `screen_h1_a`

Experiments: 462 (failed 0).

Configurations tried (for DSR): 2030. White Reality Check p = 1.0; Hansen SPA (consistent) p = 0.952.

Top 15 by out-of-sample net Sharpe:

| target | symbol | tf | sharpe | ann_return | max_drawdown | trades | psr_0 | dsr | p_bh | stress.cost_x2.0.sharpe | diagnostic_zero_cost_default_sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S.SEAS.WEEKEND_GAP_FADE | AUDUSD | H1 | 0.750 | 0.008 | -0.062 | 123.000 | 0.997 | 0.000 | 1.000 | 0.687 | 0.138 |
| S.VOL.EXPANSION | XAUUSD | H1 | 0.709 | 0.012 | -0.031 | 55.000 | 0.932 | 0.000 | 1.000 | 0.633 | 0.886 |
| S.XA.STAT_ARB | GBPUSD | H1 | 0.498 | 0.007 | -0.028 | 40.000 | 0.920 | 0.000 | 1.000 | 0.389 | 0.518 |
| S.XA.PAIRS | XAUUSD | H1 | 0.488 | 0.039 | -0.151 | 41.000 | 0.839 | 0.000 | 1.000 | 0.438 | -0.463 |
| S.TREND.REGRESSION | USDJPY | H1 | 0.485 | 0.018 | -0.036 | 75.000 | 0.926 | 0.000 | 1.000 | 0.370 | 0.400 |
| S.BO.RANGE_EXPANSION | EURUSD | H1 | 0.465 | 0.031 | -0.148 | 918.000 | 0.949 | 0.000 | 1.000 | 0.056 | 0.600 |
| S.VOL.EXPANSION | GBPUSD | H1 | 0.443 | 0.014 | -0.083 | 230.000 | 0.900 | 0.000 | 1.000 | 0.246 | 0.884 |
| S.BO.VOL_BREAKOUT | USDCHF | H1 | 0.394 | 0.020 | -0.045 | 268.000 | 0.878 | 0.000 | 1.000 | 0.040 | 0.067 |
| S.XA.GOLD_FX | XAUUSD | H1 | 0.366 | 0.026 | -0.222 | 29.000 | 0.771 | 0.000 | 1.000 | 0.326 | 0.277 |
| S.XA.STAT_ARB | XAUUSD | H1 | 0.357 | 0.013 | -0.056 | 47.000 | 0.764 | 0.000 | 1.000 | 0.218 | -0.361 |
| S.XA.RELATIVE_VALUE | USDCHF | H1 | 0.339 | 0.022 | -0.178 | 29.000 | 0.849 | 0.000 | 1.000 | 0.238 | 0.282 |
| S.XA.COINTEGRATION | XAUUSD | H1 | 0.330 | 0.004 | -0.017 | 17.000 | 0.737 | 0.000 | 1.000 | 0.183 | -0.832 |
| S.GEO.HARMONIC | GBPUSD | H1 | 0.321 | 0.007 | -0.073 | 188.000 | 0.823 | 0.000 | 1.000 | 0.114 | 0.197 |
| S.RG.CHANGE_POINT | USDCHF | H1 | 0.292 | 0.012 | -0.095 | 413.000 | 0.801 | 0.000 | 1.000 | -0.211 | 0.736 |
| S.VOL.EXPANSION | EURUSD | H1 | 0.291 | 0.007 | -0.052 | 299.000 | 0.848 | 0.000 | 1.000 | 0.098 | 0.633 |

Share of experiments with OOS net Sharpe > 0: 8.7% of 462; median OOS Sharpe -0.91; median zero-cost diagnostic Sharpe -0.06.

### Family comparison

| family | n | median_sharpe | best_sharpe | share_positive | median_x2cost_sharpe |
|---|---|---|---|---|---|
| BO | 35 | -0.861 | 0.465 | 0.114 | -1.546 |
| GEO | 28 | -0.591 | 0.321 | 0.071 | -1.096 |
| ML | 42 | -1.634 | 0.205 | 0.024 | -3.631 |
| MR | 49 | -0.737 | 0.049 | 0.041 | -1.218 |
| MS | 21 | -1.274 | -0.297 | 0.000 | -2.403 |
| RG | 35 | -1.503 | 0.292 | 0.143 | -2.809 |
| SEAS | 35 | -0.970 | 0.750 | 0.029 | -2.759 |
| SES | 35 | -1.662 | -0.016 | 0.000 | -2.892 |
| SP | 14 | -0.643 | -0.119 | 0.000 | -1.334 |
| ST | 42 | -1.321 | -0.070 | 0.000 | -2.396 |
| TREND | 49 | -0.621 | 0.485 | 0.102 | -1.174 |
| VOL | 21 | -0.840 | 0.709 | 0.286 | -1.412 |
| XA | 56 | -0.326 | 0.498 | 0.250 | -0.572 |

### Instrument comparison

| symbol | n | median_sharpe | best_sharpe | share_positive |
|---|---|---|---|---|
| AUDUSD | 66 | -1.043 | 0.750 | 0.076 |
| EURUSD | 66 | -0.747 | 0.465 | 0.061 |
| GBPUSD | 66 | -0.877 | 0.498 | 0.121 |
| USDCAD | 66 | -1.147 | 0.228 | 0.045 |
| USDCHF | 66 | -1.006 | 0.394 | 0.076 |
| USDJPY | 66 | -0.828 | 0.485 | 0.106 |
| XAUUSD | 66 | -0.933 | 0.709 | 0.121 |

### Timeframe comparison

| tf | n | median_sharpe | share_positive |
|---|---|---|---|
| H1 | 462 | -0.911 | 0.087 |

### Survivors of the pre-declared bar

**None.**

### Portfolio / sizing comparison

| Unnamed: 0 | net_return | ann_return | sharpe | sortino | calmar | max_drawdown | dd_duration_bars | cvar_5_daily | skew_daily | kurt_daily | n_days | years |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alloc:equal_weight | -0.183 | -0.016 | -1.253 | -1.716 | -0.082 | -0.199 | 2839.000 | -0.002 | 0.556 | 12.528 | 3195.000 | 12.274 |
| alloc:inverse_vol | -0.058 | -0.005 | -0.733 | -1.000 | -0.068 | -0.071 | 2839.000 | -0.001 | 0.684 | 74.157 | 3195.000 | 12.274 |
| sizing:vol_target_10pct | -0.262 | -0.024 | -0.733 | -1.000 | -0.078 | -0.313 | 2839.000 | -0.005 | 0.684 | 74.157 | 3195.000 | 12.274 |
| sizing:drawdown_derisk | -0.057 | -0.005 | -0.741 | -1.007 | -0.068 | -0.070 | 2839.000 | -0.001 | 0.572 | 72.541 | 3195.000 | 12.274 |
| sizing:fractional_kelly_0.25 | -0.022 | -0.002 | -0.168 | -0.267 | -0.031 | -0.058 | 2839.000 | -0.001 | 8.094 | 260.273 | 3195.000 | 12.274 |
| alloc:currency_netted_cap0.5 | -0.057 | -0.005 | -0.720 | -0.987 | -0.067 | -0.071 | 2839.000 | -0.001 | 0.769 | 76.044 | 3195.000 | 12.274 |

### Hedging comparison

| stream | unhedged_sharpe | hedged_sharpe | unhedged_mdd | hedged_mdd | unhedged_ann_vol | hedged_ann_vol | hedge_cost_drag_ann | extra_gross_exposure_avg |
|---|---|---|---|---|---|---|---|---|
| S.SEAS.WEEKEND_GAP_FADE|AUDUSD|H1 | 0.750 | 0.661 | -0.062 | -0.088 | 0.018 | 0.018 | 0.012 | 0.002 |
| S.VOL.EXPANSION|XAUUSD|H1 | 0.709 | -0.020 | -0.031 | -0.044 | 0.021 | 0.019 | 0.024 | 0.001 |
| S.XA.STAT_ARB|GBPUSD|H1 | 0.498 | 0.421 | -0.028 | -0.034 | 0.016 | 0.012 | 0.053 | 0.005 |
| S.XA.PAIRS|XAUUSD|H1 | 0.488 | -0.541 | -0.151 | -0.206 | 0.099 | 0.064 | 1.375 | 0.125 |
| S.TREND.REGRESSION|USDJPY|H1 | 0.485 | 0.362 | -0.036 | -0.039 | 0.034 | 0.025 | 0.191 | 0.013 |
| S.BO.RANGE_EXPANSION|EURUSD|H1 | 0.465 | 0.004 | -0.148 | -0.207 | 0.052 | 0.050 | 0.051 | 0.001 |
| S.VOL.EXPANSION|GBPUSD|H1 | 0.443 | 0.322 | -0.083 | -0.073 | 0.031 | 0.030 | 0.016 | 0.001 |
| S.BO.VOL_BREAKOUT|USDCHF|H1 | 0.394 | 0.229 | -0.045 | -0.050 | 0.038 | 0.038 | 0.017 | 0.001 |
| S.XA.GOLD_FX|XAUUSD|H1 | 0.366 | -0.060 | -0.222 | -0.156 | 0.109 | 0.071 | 1.730 | 0.157 |
| S.XA.STAT_ARB|XAUUSD|H1 | 0.357 | 0.034 | -0.056 | -0.065 | 0.043 | 0.029 | 0.266 | 0.020 |
| S.XA.RELATIVE_VALUE|USDCHF|H1 | 0.339 | 0.188 | -0.178 | -0.117 | 0.066 | 0.052 | 0.575 | 0.065 |
| S.XA.COINTEGRATION|XAUUSD|H1 | 0.330 | -0.802 | -0.017 | -0.022 | 0.010 | 0.006 | 0.024 | 0.002 |
| S.GEO.HARMONIC|GBPUSD|H1 | 0.321 | 0.069 | -0.073 | -0.095 | 0.029 | 0.028 | 0.017 | 0.001 |
| S.RG.CHANGE_POINT|USDCHF|H1 | 0.292 | -0.165 | -0.095 | -0.165 | 0.044 | 0.042 | 0.038 | 0.002 |
| S.VOL.EXPANSION|EURUSD|H1 | 0.291 | 0.096 | -0.052 | -0.081 | 0.030 | 0.029 | 0.015 | 0.000 |
| S.RG.CHANGE_POINT|GBPUSD|H1 | 0.239 | 0.070 | -0.097 | -0.110 | 0.043 | 0.042 | 0.021 | 0.002 |
| S.XA.PAIRS|USDCHF|H1 | 0.229 | 0.186 | -0.169 | -0.130 | 0.096 | 0.080 | 1.103 | 0.119 |
| S.TREND.REGRESSION|USDCAD|H1 | 0.228 | -0.011 | -0.064 | -0.040 | 0.028 | 0.023 | 0.146 | 0.010 |
| S.RG.CHANGE_POINT|EURUSD|H1 | 0.225 | -0.172 | -0.117 | -0.207 | 0.049 | 0.048 | 0.036 | 0.001 |
| S.ML.META_LABEL|USDJPY|H1 | 0.205 | 0.039 | -0.043 | -0.048 | 0.017 | 0.017 | 0.006 | 0.000 |
| S.VOL.EXPANSION|USDJPY|H1 | 0.205 | -0.090 | -0.047 | -0.063 | 0.027 | 0.027 | 0.016 | 0.000 |
| S.XA.PAIRS|GBPUSD|H1 | 0.158 | 0.025 | -0.110 | -0.138 | 0.073 | 0.056 | 0.877 | 0.113 |
| S.TREND.MA_CROSS|USDCHF|H1 | 0.152 | -0.469 | -0.331 | -0.454 | 0.116 | 0.092 | 1.873 | 0.207 |
| S.RG.CHANGE_POINT|AUDUSD|H1 | 0.145 | -0.265 | -0.113 | -0.134 | 0.039 | 0.038 | 0.033 | 0.005 |
| S.TREND.MA_CROSS|USDJPY|H1 | 0.115 | -0.243 | -0.241 | -0.312 | 0.098 | 0.074 | 1.604 | 0.107 |

## 4. Strategy campaign `screen_h1_b`

Experiments: 462 (failed 0).

Configurations tried (for DSR): 2030. White Reality Check p = 1.0; Hansen SPA (consistent) p = 0.999.

Top 15 by out-of-sample net Sharpe:

| target | symbol | tf | sharpe | ann_return | max_drawdown | trades | psr_0 | dsr | p_bh | stress.cost_x2.0.sharpe | diagnostic_zero_cost_default_sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S.XA.PAIRS | XAGUSD | H1 | 0.598 | 0.073 | -0.284 | 43.000 | 0.905 | 0.000 | 1.000 | 0.382 | -0.439 |
| S.SEAS.WEEKEND_GAP_FADE | EURJPY | H1 | 0.576 | 0.001 | -0.044 | 114.000 | 0.948 | 0.000 | 1.000 | 0.413 | -0.033 |
| S.TREND.REGRESSION | NZDUSD | H1 | 0.459 | 0.014 | -0.072 | 57.000 | 0.911 | 0.000 | 1.000 | 0.302 | -0.293 |
| S.TREND.REGRESSION | AUDJPY | H1 | 0.428 | 0.027 | -0.071 | 67.000 | 0.859 | 0.000 | 1.000 | 0.304 | 0.259 |
| S.RG.CHANGE_POINT | GBPJPY | H1 | 0.416 | 0.018 | -0.061 | 362.000 | 0.887 | 0.000 | 1.000 | 0.083 | 0.740 |
| S.BO.RANGE_EXPANSION | GBPJPY | H1 | 0.327 | 0.019 | -0.136 | 677.000 | 0.826 | 0.000 | 1.000 | -0.042 | 0.630 |
| S.TREND.REGRESSION | XAGUSD | H1 | 0.308 | 0.011 | -0.129 | 23.000 | 0.748 | 0.000 | 1.000 | 0.066 | -0.049 |
| S.VOL.EXPANSION | EURGBP | H1 | 0.280 | 0.007 | -0.051 | 119.000 | 0.790 | 0.000 | 1.000 | 0.125 | 0.639 |
| S.SES.NY_OPEN_REVERSAL | NZDUSD | H1 | 0.273 | 0.004 | -0.035 | 199.000 | 0.788 | 0.000 | 1.000 | -0.318 | 0.300 |
| S.XA.RELATIVE_VALUE | XAGUSD | H1 | 0.244 | 0.004 | -0.268 | 19.000 | 0.701 | 0.000 | 1.000 | 0.123 | -0.013 |
| S.TREND.TSMOM | EURJPY | H1 | 0.242 | 0.013 | -0.239 | 1229.000 | 0.756 | 0.000 | 1.000 | -0.207 | 0.662 |
| S.VOL.EXPANSION | GBPJPY | H1 | 0.237 | 0.009 | -0.102 | 286.000 | 0.753 | 0.000 | 1.000 | -0.034 | 0.073 |
| S.TREND.KALMAN | GBPJPY | H1 | 0.233 | 0.010 | -0.303 | 918.000 | 0.746 | 0.000 | 1.000 | -0.160 | 0.649 |
| S.XA.STAT_ARB | XAGUSD | H1 | 0.225 | 0.004 | -0.131 | 55.000 | 0.693 | 0.000 | 1.000 | -0.251 | 0.471 |
| S.BO.RANGE_EXPANSION | EURJPY | H1 | 0.221 | 0.031 | -0.087 | 1059.000 | 0.737 | 0.000 | 1.000 | -0.283 | 0.680 |

Share of experiments with OOS net Sharpe > 0: 8.2% of 462; median OOS Sharpe -1.14; median zero-cost diagnostic Sharpe -0.09.

### Family comparison

| family | n | median_sharpe | best_sharpe | share_positive | median_x2cost_sharpe |
|---|---|---|---|---|---|
| BO | 35 | -1.170 | 0.327 | 0.086 | -2.074 |
| GEO | 28 | -1.184 | 0.051 | 0.071 | -1.679 |
| ML | 42 | -2.087 | 0.142 | 0.024 | -4.698 |
| MR | 49 | -0.901 | 0.030 | 0.020 | -1.604 |
| MS | 21 | -1.561 | -0.642 | 0.000 | -2.748 |
| RG | 35 | -1.795 | 0.416 | 0.086 | -3.290 |
| SEAS | 35 | -1.313 | 0.576 | 0.086 | -3.194 |
| SES | 35 | -1.757 | 0.273 | 0.057 | -3.517 |
| SP | 14 | -1.049 | -0.212 | 0.000 | -2.205 |
| ST | 42 | -1.818 | -0.201 | 0.000 | -2.940 |
| TREND | 49 | -0.521 | 0.459 | 0.224 | -1.056 |
| VOL | 21 | -1.077 | 0.280 | 0.095 | -1.654 |
| XA | 56 | -0.424 | 0.598 | 0.179 | -0.607 |

### Instrument comparison

| symbol | n | median_sharpe | best_sharpe | share_positive |
|---|---|---|---|---|
| AUDJPY | 66 | -0.979 | 0.428 | 0.061 |
| EURCHF | 66 | -1.410 | 0.104 | 0.030 |
| EURGBP | 66 | -1.261 | 0.280 | 0.061 |
| EURJPY | 66 | -0.675 | 0.576 | 0.121 |
| GBPJPY | 66 | -0.910 | 0.416 | 0.152 |
| NZDUSD | 66 | -1.294 | 0.459 | 0.061 |
| XAGUSD | 66 | -3.734 | 0.598 | 0.091 |

### Timeframe comparison

| tf | n | median_sharpe | share_positive |
|---|---|---|---|
| H1 | 462 | -1.137 | 0.082 |

### Survivors of the pre-declared bar

**None.**

### Portfolio / sizing comparison

| Unnamed: 0 | net_return | ann_return | sharpe | sortino | calmar | max_drawdown | dd_duration_bars | cvar_5_daily | skew_daily | kurt_daily | n_days | years |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alloc:equal_weight | -0.222 | -0.030 | -1.622 | -2.168 | -0.134 | -0.224 | 1996.000 | -0.003 | 0.180 | 5.719 | 2145.000 | 8.222 |
| alloc:inverse_vol | -0.062 | -0.008 | -0.908 | -1.108 | -0.111 | -0.070 | 1975.000 | -0.001 | -4.882 | 106.171 | 2145.000 | 8.222 |
| sizing:vol_target_10pct | -0.269 | -0.037 | -0.891 | -1.094 | -0.125 | -0.299 | 1975.000 | -0.006 | -4.677 | 106.725 | 2145.000 | 8.222 |
| sizing:drawdown_derisk | -0.061 | -0.008 | -0.913 | -1.108 | -0.111 | -0.069 | 1975.000 | -0.001 | -5.135 | 110.578 | 2145.000 | 8.222 |
| sizing:fractional_kelly_0.25 | -0.047 | -0.006 | -0.434 | -0.491 | -0.095 | -0.062 | 1975.000 | -0.001 | -21.559 | 947.809 | 2145.000 | 8.222 |
| alloc:currency_netted_cap0.5 | -0.062 | -0.008 | -0.906 | -1.105 | -0.111 | -0.069 | 1975.000 | -0.001 | -4.884 | 106.222 | 2145.000 | 8.222 |

## 4. Strategy campaign `screen_tf_eurusd`

Experiments: 122 (failed 0).

Configurations tried (for DSR): 547. White Reality Check p = 0.995; Hansen SPA (consistent) p = 0.202.

Top 15 by out-of-sample net Sharpe:

| target | symbol | tf | sharpe | ann_return | max_drawdown | trades | psr_0 | dsr | p_bh | stress.cost_x2.0.sharpe | diagnostic_zero_cost_default_sharpe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S.VOL.EXPANSION | EURUSD | M15 | 0.732 | 0.027 | -0.079 | 613 | 0.996 | 0.000 | 0.630 | 0.386 | 0.457 |
| S.TREND.XS_MOMENTUM | EURUSD | H4 | 0.257 | 0.011 | -0.103 | 307 | 0.817 | 0.000 | 1.000 | 0.100 | 0.319 |
| S.TREND.REGRESSION | EURUSD | M15 | 0.185 | 0.003 | -0.051 | 198 | 0.743 | 0.000 | 1.000 | 0.016 | -0.054 |
| S.BO.FALSE_BREAKOUT | EURUSD | H4 | 0.129 | 0.002 | -0.098 | 870 | 0.675 | 0.000 | 1.000 | -0.180 | 0.048 |
| S.XA.RELATIVE_VALUE | EURUSD | M15 | 0.119 | 0.005 | -0.111 | 157 | 0.661 | 0.000 | 1.000 | -0.002 | 0.118 |
| S.XA.TRIANGULAR | EURUSD | H4 | 0.110 | -0.006 | -0.209 | 258 | 0.651 | 0.000 | 1.000 | -0.079 | 0.215 |
| S.GEO.FIB_EXTENSION | EURUSD | H4 | 0.074 | 0.001 | -0.122 | 410 | 0.602 | 0.000 | 1.000 | -0.116 | 0.035 |
| S.SP.WAVELET | EURUSD | H4 | 0.031 | -0.002 | -0.278 | 477 | 0.544 | 0.000 | 1.000 | -0.189 | 0.032 |
| S.GEO.FIB_RETRACE | EURUSD | H4 | 0.012 | -0.003 | -0.099 | 469 | 0.517 | 0.000 | 1.000 | -0.180 | -0.146 |
| S.XA.GOLD_FX | EURUSD | H4 | 0.008 | -0.004 | -0.174 | 10 | 0.512 | 0.000 | 1.000 | -0.104 | 0.108 |
| S.ML.SUPERVISED_DIRECTION | EURUSD | H4 | 0.002 | -0.003 | -0.191 | 2868 | 0.502 | 0.000 | 1.000 | -0.665 | 0.172 |
| S.XA.STAT_ARB | EURUSD | H4 | -0.012 | -0.002 | -0.109 | 79 | 0.483 | 0.000 | 1.000 | -0.092 | 0.091 |
| S.SEAS.TURN_OF_MONTH | EURUSD | M15 | -0.057 | -0.005 | -0.129 | 293 | 0.421 | 0.000 | 1.000 | -0.204 | 0.055 |
| S.MR.BOLLINGER | EURUSD | H4 | -0.069 | -0.009 | -0.227 | 492 | 0.405 | 0.000 | 1.000 | -0.282 | 0.126 |
| S.XA.CURRENCY_STRENGTH | EURUSD | H4 | -0.120 | -0.006 | -0.204 | 543 | 0.337 | 0.000 | 1.000 | -0.324 | -0.008 |

Share of experiments with OOS net Sharpe > 0: 9.0% of 122; median OOS Sharpe -0.62; median zero-cost diagnostic Sharpe -0.03.

### Family comparison

| family | n | median_sharpe | best_sharpe | share_positive | median_x2cost_sharpe |
|---|---|---|---|---|---|
| BO | 9 | -0.773 | 0.129 | 0.111 | -1.799 |
| GEO | 8 | -0.498 | 0.074 | 0.250 | -0.830 |
| ML | 11 | -0.568 | 0.002 | 0.091 | -1.080 |
| MR | 14 | -0.613 | -0.069 | 0.000 | -1.073 |
| MS | 6 | -1.243 | -0.710 | 0.000 | -1.870 |
| RG | 10 | -0.815 | -0.158 | 0.000 | -1.881 |
| SEAS | 8 | -0.401 | -0.057 | 0.000 | -0.874 |
| SES | 5 | -1.718 | -0.548 | 0.000 | -3.002 |
| SP | 4 | -0.612 | 0.031 | 0.250 | -1.219 |
| ST | 11 | -1.196 | -0.318 | 0.000 | -2.199 |
| TREND | 14 | -0.505 | 0.257 | 0.143 | -0.923 |
| VOL | 6 | -0.671 | 0.732 | 0.167 | -0.945 |
| XA | 16 | -0.307 | 0.119 | 0.188 | -0.493 |

### Instrument comparison

| symbol | n | median_sharpe | best_sharpe | share_positive |
|---|---|---|---|---|
| EURUSD | 122 | -0.619 | 0.732 | 0.090 |

### Timeframe comparison

| tf | n | median_sharpe | share_positive |
|---|---|---|---|
| H4 | 58 | -0.357 | 0.138 |
| M15 | 64 | -1.518 | 0.047 |

### Survivors of the pre-declared bar

**None.**

## 5. Risk-management comparison `risk_eurusd_h1`

Each family re-run with 11 stop/target/sizing configurations as walk-forward candidates (selection on training windows); see `results/artifacts/<id>/folds.parquet` for the configuration chosen per fold.

| target | sharpe | ann_return | max_drawdown | trades | stress.cost_x2.0.sharpe |
|---|---|---|---|---|---|
| S.RG.CHANGE_POINT | 0.186 | 0.005 | -0.099 | 656 | -0.266 |
| S.XA.COINTEGRATION | 0.129 | 0.001 | -0.025 | 78 | -0.098 |
| S.VOL.EXPANSION | 0.117 | 0.005 | -0.072 | 344 | -0.095 |
| S.ST.PIVOT_POINTS | -0.028 | -0.006 | -0.250 | 2022 | -0.644 |
| S.MR.OU | -0.043 | -0.001 | -0.094 | 644 | -0.363 |
| S.SEAS.TURN_OF_MONTH | -0.203 | -0.008 | -0.184 | 716 | -0.486 |
| S.ST.SWING_STRUCTURE | -0.221 | -0.010 | -0.261 | 3181 | -0.961 |
| S.GEO.HARMONIC | -0.281 | -0.007 | -0.120 | 227 | -0.458 |
| S.MS.TICK_IMBALANCE | -0.319 | -0.001 | -0.009 | 15 | -0.558 |
| S.XA.PAIRS | -0.321 | -0.014 | -0.230 | 629 | -0.585 |
| S.TREND.XS_MOMENTUM | -0.477 | -0.013 | -0.196 | 1664 | -1.161 |
| S.BO.VOL_BREAKOUT | -0.481 | -0.025 | -0.420 | 1653 | -1.103 |
| S.BO.RANGE_EXPANSION | -0.498 | -0.028 | -0.483 | 2379 | -1.259 |
| S.SEAS.WEEKEND_GAP_FADE | -0.509 | -0.019 | -0.224 | 273 | -0.480 |
| S.GEO.GANN | -0.523 | -0.021 | -0.257 | 1684 | -0.981 |
| S.XA.TRIANGULAR | -0.552 | -0.018 | -0.218 | 1240 | -1.361 |
| S.SES.NY_OPEN_REVERSAL | -0.566 | -0.010 | -0.141 | 370 | -0.968 |
| S.MR.ZSCORE | -0.584 | -0.029 | -0.324 | 1477 | -1.012 |
| S.ST.CHANNEL | -0.600 | -0.028 | -0.351 | 1950 | -1.122 |
| S.TREND.TSMOM | -0.678 | -0.036 | -0.422 | 3732 | -1.436 |
| S.BO.RETEST | -0.694 | -0.035 | -0.413 | 1879 | -1.258 |
| S.BO.FALSE_BREAKOUT | -0.702 | -0.049 | -0.504 | 3859 | -1.525 |
| S.SEAS.FIX_REVERSAL | -0.723 | -0.014 | -0.179 | 1594 | -2.009 |
| S.GEO.FIB_EXTENSION | -0.726 | -0.022 | -0.249 | 599 | -0.978 |
| S.MR.BOLLINGER | -0.743 | -0.035 | -0.378 | 3672 | -1.615 |
| S.SP.WAVELET | -0.757 | -0.046 | -0.449 | 2474 | -1.346 |
| S.VOL.COMPRESSION | -0.806 | -0.036 | -0.421 | 1333 | -1.250 |
| S.MR.STOCH_REVERSAL | -0.826 | -0.052 | -0.499 | 2822 | -1.487 |
| S.TREND.MA_CROSS | -0.829 | -0.045 | -0.499 | 2708 | -1.450 |
| S.MR.TWAP_REVERSION | -0.833 | -0.039 | -0.408 | 3462 | -1.621 |

## 6. Prediction campaign `models_eurusd_h1`

Experiments: 198 (failed 0).

Direction classification, top 20 by out-of-sample AUC:

| target | h | auc | brier | ece | base_rate | sign_ret_mean_bps | p_holm | n |
|---|---|---|---|---|---|---|---|---|
| M.TREE.EXTRA | 1 | 0.539 | 0.249 | 0.009 | 0.499 | 0.304 | 0.000 | 85141 |
| M.EVO.GP | 1 | 0.538 | 0.283 | 0.161 | 0.499 | 0.278 | 0.000 | 85141 |
| M.ENS.MIXTURE_OF_EXPERTS | 1 | 0.538 | 0.249 | 0.015 | 0.499 | 0.300 | 0.000 | 85141 |
| M.TREE.RF | 1 | 0.536 | 0.249 | 0.002 | 0.499 | 0.246 | 0.000 | 85141 |
| M.TREE.XGB | 1 | 0.536 | 0.250 | 0.026 | 0.499 | 0.283 | 0.000 | 85141 |
| M.LIN.BAYES_RIDGE | 1 | 0.535 | 0.265 | 0.123 | 0.499 | -0.028 | 0.000 | 85141 |
| M.KER.LINEAR_SVM | 1 | 0.535 | 0.249 | 0.003 | 0.499 | 0.296 | 0.000 | 85141 |
| M.TREE.ADABOOST | 1 | 0.535 | 0.259 | 0.083 | 0.499 | 0.224 | 0.000 | 85141 |
| M.LIN.LOGISTIC | 1 | 0.535 | 0.250 | 0.021 | 0.499 | 0.295 | 0.000 | 85141 |
| M.LIN.LASSO | 1 | 0.535 | 0.250 | 0.021 | 0.499 | 0.291 | 0.000 | 85141 |
| M.LIN.ELASTICNET | 1 | 0.534 | 0.250 | 0.022 | 0.499 | 0.280 | 0.000 | 85141 |
| M.LIN.RIDGE | 1 | 0.534 | 0.249 | 0.002 | 0.499 | 0.274 | 0.000 | 85141 |
| M.TREE.CATBOOST | 1 | 0.534 | 0.250 | 0.023 | 0.499 | 0.288 | 0.000 | 85141 |
| M.EVO.GRAMMATICAL | 1 | 0.534 | 0.286 | 0.170 | 0.499 | 0.259 | 0.000 | 85141 |
| M.LIN.OLS | 1 | 0.534 | 0.265 | 0.123 | 0.499 | -0.028 | 0.000 | 85141 |
| M.LIN.HUBER | 1 | 0.534 | 0.265 | 0.123 | 0.499 | -0.028 | 0.000 | 85141 |
| M.ENS.DYNAMIC_SELECTION | 1 | 0.534 | 0.250 | 0.017 | 0.499 | 0.328 | 0.000 | 85141 |
| M.TREE.GBM | 1 | 0.533 | 0.251 | 0.032 | 0.499 | 0.256 | 0.000 | 85141 |
| M.ENS.STACKING | 1 | 0.532 | 0.249 | 0.002 | 0.499 | 0.267 | 0.000 | 85141 |
| M.TREE.LGBM | 1 | 0.532 | 0.252 | 0.043 | 0.499 | 0.251 | 0.000 | 85141 |

Volatility forecasting:

| target | h | ic_spearman | r2_oos | rmse |
|---|---|---|---|---|
| M.TS.EGARCH | 1 | 0.313 | -0.826 | 0.001 |
| M.TS.GARCH | 1 | 0.297 | -0.133 | 0.001 |
| M.TS.GJR_GARCH | 1 | 0.298 | -0.136 | 0.001 |
| M.TS.HAR | 1 | 0.295 | -0.131 | 0.001 |
| M.TS.EGARCH | 5 | 0.437 | -1.097 | 0.002 |
| M.TS.GARCH | 5 | 0.392 | 0.022 | 0.002 |
| M.TS.GJR_GARCH | 5 | 0.400 | -0.027 | 0.002 |
| M.TS.HAR | 5 | 0.485 | 0.114 | 0.002 |
| M.TS.EGARCH | 20 | 0.475 | -1.809 | 0.004 |
| M.TS.GARCH | 20 | 0.442 | 0.131 | 0.002 |
| M.TS.GJR_GARCH | 20 | 0.454 | 0.024 | 0.002 |
| M.TS.HAR | 20 | 0.656 | 0.340 | 0.002 |

Trading policies (RL / evolutionary), IC of position with realised return:

| target | h | ic_spearman | hit_rate |
|---|---|---|---|
| M.EVO.DE_SHARPE | 1 | 0.005 | 0.501 |
| M.RL.DQN | 1 | 0.018 | 0.261 |
| M.RL.LINUCB | 1 | 0.007 | 0.075 |
| M.RL.PPO | 1 | -0.002 | 0.358 |
| M.RL.SAC | 1 | 0.024 | 0.264 |
| M.EVO.DE_SHARPE | 5 | 0.015 | 0.506 |
| M.RL.DQN | 5 | 0.009 | 0.301 |
| M.RL.LINUCB | 5 | 0.004 | 0.138 |
| M.RL.PPO | 5 | 0.004 | 0.398 |
| M.RL.SAC | 5 | 0.015 | 0.290 |
| M.EVO.DE_SHARPE | 20 | -0.002 | 0.495 |
| M.RL.DQN | 20 | -0.005 | 0.346 |
| M.RL.LINUCB | 20 | -0.019 | 0.207 |
| M.RL.PPO | 20 | 0.002 | 0.403 |
| M.RL.SAC | 20 | -0.014 | 0.306 |

Horizon comparison: [{'h': 1, 'median': 0.5282938922637288, 'max': 0.5387126051259722}, {'h': 5, 'median': 0.5136267869218551, 'max': 0.5212752464309961}, {'h': 20, 'median': 0.49853458001164086, 'max': 0.5116135180461783}]

## 7. Feature relevance / redundancy (EURUSD H1)

Feature columns: 421; BH-significant (q<0.10) real columns: 163; BH-significant placebo columns: 5; redundant pairs |rho|>=0.9: 276.

| column | h | mean_ic | ic_t | p_bh | sign_stability | beats_placebo_p95 |
|---|---|---|---|---|---|---|
| F.PR.CLV:v | 1 | -0.054 | -14.611 | 0.000 | 1.000 | True |
| F.VO.LEE_MYKLAND:v | 1 | -0.043 | -12.462 | 0.000 | 1.000 | True |
| F.PR.INTRABAR_RET:v | 1 | -0.042 | -12.445 | 0.000 | 1.000 | True |
| F.MO.CONNORS_RSI:v | 1 | -0.049 | -12.164 | 0.000 | 1.000 | True |
| F.MS.CHAIKIN_AD:v | 1 | -0.042 | -12.108 | 0.000 | 1.000 | True |
| F.EC.AR_RESID:v | 1 | -0.042 | -12.102 | 0.000 | 1.000 | True |
| F.PR.STREAK:v | 1 | -0.044 | -11.054 | 0.000 | 1.000 | True |
| F.ST.BREAKOUT:v | 1 | -0.026 | -9.431 | 0.000 | 1.000 | True |
| F.MO.CFO:v | 1 | -0.027 | -8.591 | 0.001 | 1.000 | True |
| F.TR.MAMA:c_mama | 1 | -0.040 | -8.478 | 0.001 | 1.000 | True |
| F.ST.HEIKIN_ASHI:streak | 1 | -0.033 | -8.317 | 0.001 | 1.000 | True |
| F.SP.SUPER_SMOOTHER:v | 1 | -0.037 | -8.167 | 0.001 | 1.000 | True |
| F.MO.STOCHRSI:v | 1 | -0.030 | -7.964 | 0.001 | 1.000 | True |
| F.MO.RWI:rwi_lo | 1 | 0.028 | 7.948 | 0.001 | 1.000 | True |
| F.MS.MFI_TICK:v | 1 | -0.024 | -7.720 | 0.001 | 1.000 | True |

## 8. Scientific conclusions

All figures below are out-of-sample walk-forward results with costs: real quoted bid/ask, slippage, commission and financing. They come from `results/reports/*/summary.json` and `strategy_results.csv` / `model_results.csv`.

### 8.1 Headline
No strategy family, instrument, timeframe, risk configuration or portfolio construction produced a trading edge that survives realistic costs and data-snooping control. The pre-declared survival bar (OOS Sharpe > 0, BH-adjusted p < 0.10, DSR > 0.95, Sharpe > 0 at 2x costs) was met by **zero** of 1,107 strategy experiments, covering 5,278 parameter/risk configurations. The engineering system is complete and working; the research answer on this data is **negative**.

### 8.2 Strategies (H1, 14 instruments, 2007-2018; EURUSD to 2022)
* 924 family x instrument experiments: 8.4% had positive OOS net Sharpe; median OOS Sharpe -1.02.
* White's Reality Check p = 1.0 in both screens; Hansen SPA p = 0.95 and 0.999. Every BH-adjusted p-value = 1.0; max Deflated Sharpe = 0.00.
* Costs are not the whole story. Even at zero cost (diagnostic only), only 42% of default configurations were profitable; median zero-cost Sharpe was -0.07. Most rules carry no gross information at H1.
* Families by median OOS Sharpe: cross-asset (-0.35, 21% positive) > trend (-0.60) > geometry (-0.72) > mean reversion (-0.81) > ... > session (-1.70) > model-driven (-1.87). Model-driven strategies are positive before costs (median zero-cost Sharpe +0.14) but the worst after costs, because they trade frequently.
* Isolated good-looking results did not replicate across instruments. Weekend-gap fade: Sharpe 0.75 on AUDUSD and 0.58 on EURJPY, but positive on only 3 of 14 instruments. Volatility expansion: positive on 8 of 14, median 0.08. Best single results rest on 40-120 trades. They are treated as multiple-testing artefacts, not candidates.
* Timeframes (EURUSD): H4 is least bad (median -0.36, 14% positive), H1 -0.75, M15 -1.52. Higher frequency loses more to costs. Best M15 result (volatility expansion, 0.73) is not significant after correction (SPA p = 0.20, WRC p = 0.995).

### 8.3 Risk management, portfolio, hedging (OOS streams only)
* 11 stop/target/sizing designs (fixed, ATR, realised-vol, structural, quantile stops; affine, power, log, piecewise, polynomial targets; trailing; time exit; vol-target sizing; wide catastrophic stop + time exit) selected per fold on EURUSD: best 0.19, 3 of 61 positive, none positive at 2x costs. Risk shaping does not create an edge where the entry signal has none.
* Portfolios of the 20 best streams by trailing OOS Sharpe, re-selected monthly, lose money under every allocation tested (equal, inverse-vol, currency-netted; Sharpe -0.72 to -1.62). Fractional Kelly comes closest (-0.17 / -0.43). Out-of-sample performance does not persist.
* Min-variance hedging with the most-correlated pair reduced volatility in 29 of 30 streams but improved Sharpe in 0 of 30 (median Sharpe 0.27 -> 0.00). Hedge-leg costs and basis risk outweigh the variance reduction.

### 8.4 Prediction (EURUSD H1, 198 model runs, 2007-2022)
* **Direction is weakly but genuinely predictable at 1 hour.** Best AUC 0.539 (extremely randomised trees), with 92 of 171 classifier runs significant after Holm correction. It is stable in every year 2008-2022 (per-year AUC about 0.52-0.56 for the leading models). It decays fast: median AUC 0.528 at 1 hour, 0.514 at 5 hours, 0.499 at 20 hours.
* Complexity does not help. Linear/logistic models (about 0.535) match trees, ensembles and genetic programming. Neural nets (MLP 0.532 best, LSTM/GRU/TCN/Transformers lower) and time-series models (median 0.499) do not beat them.
* The edge is economically too small. A sign-trading proxy earns about 0.3 bp per hour gross, against about 1.35 bp modelled round-trip cost on EURUSD (median spread 0.4 pip; median |1h return| 4.7 bp).
* **Volatility is strongly predictable.** HAR-RV rank correlation with realised volatility is 0.30 / 0.49 / 0.66 at 1 / 5 / 20 hours (R^2_oos 0.34 at 20h); GARCH-family models are close behind. This is useful for position sizing and risk limits, not as a directional signal.
* RL and evolutionary trading policies (DQN, PPO, discrete SAC, LinUCB, differential evolution) learned nothing usable: correlation of position with realised return |IC| <= 0.024 at every horizon.

### 8.5 Data findings
* The GitHub mirror stores JPY-quoted pairs and metals at 1/100 of the true price. This was detected (USDJPY financing came out at about 1% per night) and corrected in the store; 132 affected results were discarded and re-run. A price-plausibility check is now part of the audit.
* The audit after correction: 0 inconsistent bars, 0 implausible price levels, no Saturday data. The last months of each series (June 2018) and the first months (March 2007, Nov 2010-Feb 2011 for silver) are partial.

## 9. Failures, exclusions and limitations
* **Data depth:** 2007-2018 for most instruments (EURUSD to 2022-07, gold 2011-05 to 2018-06, silver 2010-11 to 2018-06, AUDJPY to 2015). 2019-2025 is not available here, so the development period ends in 2018/2022, not 2025.
* **Instruments:** 14 of the 38 requested were acquired. USDMXN, USDZAR, USDTRY and USDCNH are not in the mirror. The other 20 (remaining crosses and SEK/NOK pairs) are available in the mirror but were not acquired locally (time budget).
* **No macro inputs:** interest rates, yields, equity indices and VIX are unreachable, so S.XA.CARRY_FILTER is blocked and swaps use a conservative constant (0.5 pip/night both sides).
* **2026 holdout:** not evaluated. The data does not exist in this environment, and candidates cannot be frozen because none survived.
* **Scope run locally:** H1 for all 14 instruments; M15/H4 for EURUSD only; models on EURUSD H1 only with 3 horizons (1, 5, 20). The remaining horizons (2, 3, 10, 30, 50, 100), M1-M10/H2/H8/D1, alternative bar types, Optuna re-optimisation and cross-market model transfer are implemented but not executed at scale (2 CPUs).
* **Excluded experiments:** S.ML.RL_POLICY on M15 (compute) and S.XA.CARRY_FILTER (data), recorded in the manifests. No experiment failed (0 of 1,305).
* **Simplifications:** a reduced TFT (no static covariates or multi-horizon decoder); discrete-action SAC; Kelly sizing on portfolio streams only; intrabar ambiguity resolved stop-first (conservative).
* **Statistical caveat:** the DSR uses the cross-sectional variance of OOS Sharpe across all configurations; with many strongly negative strategies this variance is large, which makes DSR conservative. Conclusions do not depend on it: BH-adjusted p-values, White RC and SPA all agree.

## 10. What would change the conclusion
1. Full 2003-2025 Dukascopy history and TrueFX as an independent second source (run `fxlab/data/acquire_dukascopy.py` on an unrestricted machine) to test the post-2018 regime and source sensitivity.
2. The full directive-scale campaign on rented CPU (about 3,500 core-hours; see `docs/compute_plan.md`). Not authorised; nothing has been spent.
3. Using the one real predictive signal differently. The 1-hour directional information and the strong volatility forecasts could be used as filters on lower-turnover entries, or for execution timing, rather than as stand-alone hourly trading signals.

