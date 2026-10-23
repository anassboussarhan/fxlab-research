# Pre-flight audit

Generated 2026-10-22T11:46:19.380498+00:00 at commit `59342eb`.

| check | status | evidence |
|---|---|---|
| 1 data sources selected on evidence | PASS | docs/data_sources.md: 25+ hosts probed; only GitHub reachable; Dukascopy ticks via FX-Data mirror |
| 2 historical coverage verified | PASS | 14 instruments; span 2007-03-30 17:00:00+00:00 .. 2022-07-08 20:59:00+00:00; missing: ['EURAUD', 'EURCAD', 'EURNZD', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'AUDCHF', 'AUDCAD', 'AUDNZD', 'CADJPY', 'CADCHF', 'CHFJPY', 'NZDJPY', 'NZDCHF', 'NZDCAD', 'USDSEK', 'USDNOK', 'USDMXN', 'USDZAR', 'USDTRY', 'USDCNH', 'EURSEK', 'EURNOK'] |
| 3 data quality checks pass | PASS | 162 instrument-years audited; inconsistent bars=0; spikes>20sd total=3189; median spread p50 range 0.24..44.58 pips |
| 4 instrument mappings correct | PASS | pip size / contract per instrument in fxlab/data/instruments.py; covered by bi5 decoder and pip tests |
| 5 timeframes correctly constructed | PASS | 12 timeframes + tick/volume/volatility/range bars; resample tests |
| 6 >= 260 feature methods implemented & tested | PASS | registered 266, tested-passing 266, states {'implemented_tested': 266} |
| 7 >= 63 strategy methods implemented & tested | PASS | registered 67, tested-passing 66, states {'implemented_tested': 66, 'blocked_missing_dependency': 1} |
| 8 >= 57 model methods implemented & tested | PASS | registered 69, tested-passing 69, states {'implemented_tested': 69} |
| 9 every implementation has tests | PASS | untested: [] |
| 10 numerical and statistical tests pass | PASS | failing catalog ids: [] |
| 11 no lookahead leakage detected | PASS | prefix-invariance tests for every feature and strategy; prediction-causality for every model |
| 12 walk-forward splits valid | PASS | tests/test_validation.py: 11 passed in 1.10s |
| 13 training folds have sufficient observations | PASS | runners require >=2000 rows per model fold and >=10 train trades for strategy selection |
| 14 execution costs modelled | PASS | bid/ask fills + slippage + commission + financing + latency; tests/test_execution.py: 9 passed in 0.53s |
| 15 model dependencies available | PASS | {'torch': True, 'xgboost': True, 'lightgbm': True, 'catboost': True, 'arch': True, 'hmmlearn': True, 'pywt': True, 'statsmodels': True, 'optuna': True, 'duckdb': True} |
| 16 GPU/CPU compatibility | CPU only | no GPU in this environment; all neural/RL code runs on CPU (torch CPU); GPU optional |
| 17-19 checkpoint recovery / error handling / persistence | PASS | tests/test_pipeline.py::test_campaign_resume_and_failure_capture; DuckDB results/results.duckdb |
| 20 2026 excluded from development | PASS | latest stored year 2022; loaders raise HoldoutViolation for >= 2026 (tested) |
| 21-22 compute benchmarked, runtime/cost estimated | PASS | results/benchmark.json + docs/compute_plan.md |
| 23 all planned families in the manifest | PASS | manifests: ['risk_eurusd_h1.json', 'screen_h1_b.json', 'screen_h1_a.json', 'models_eurusd_h1.json', 'screen_tf_eurusd.json'] |
| 24 no critical placeholder implementation | PASS | all 402 entries executable; data-blocked: ['S.XA.CARRY_FILTER'] |
| 25 end-to-end representative experiment | PASS | {'experiment': 'S.ML.SUPERVISED_DIRECTION EURUSD H1 2007-2013 (features->LightGBM walk-forward->signals->bid/ask execution->nested walk-forward selection->stress)', 'seconds': 33.7, 'oos_folds': 10, 'oos_sharpe': -1.5902893046533975, 'oos_ann_return': -0.08103453931010529, 'max_drawdown': -0.3460587047860655, 'trades': 1954, 'psr_0': 0.0002140374453857935, 'stress': {'cost_x1.5': {'sharpe': -2.315271246304666, 'ann_return': -0.1181777961067455}, 'cost_x2.0': {'sharpe': -3.0205468911680082, 'ann_return': -0.15382207678925774}, 'cost_x3.0': {'sharpe': -4.345730232133784, 'ann_return': -0.22085288328472508}, 'latency1': {'sharpe': -1.5590840640000712, 'ann_return': -0.08469870953665426}}, 'zero_cost_diag': 0.2526465987085427, 'artifacts': '/home/claude/fxlab/results/artifacts/e2e/oos_returns.parquet'} |
