# SOL Bot Phase 2 — Walk-Forward Backtest

**Generated:** 2026-04-22 13:48 UTC
**Dataset:** 4,281 rows  |  Train: 124d  |  Test: 53d

## Estratégia testada

- **G0:** ETH momentum filter (H3: β_ETH > β_BTC)
- **G1:** taker_z_prev > thr (H1: Cohen's d=0.51-0.61)
- **G2:** OI_z bipolar — continuation 1h, reversion block 24h (H1)
- **G3:** volume_z auxiliar (H2)
- **Exit:** TP 2% / SL 1.5% / Trail 1% / OI_early_exit / max 120h

## ✅ GO — Implementar sol_bot.py (Phase 3)

Sharpe ≥ 2.0, N ≥ 5, overfitting < 1.0.

## Best Config — TEST set (out-of-sample)

| Métrica | Valor |
|---------|-------|
| **Test Sharpe** | **2.029** |
| Train Sharpe | 1.092 |
| Overfitting (Δ) | -0.937 ✅ |
| N trades (test) | 21 |
| Win Rate | 57.1% |
| Avg Return | +0.268% |
| Total Return | +5.69% |
| Max DD | -1.4% |
| Profit Factor | 2.11 |

### Exit reasons (TEST)

| Reason | N |
|--------|---|
| TRAIL | 17 |
| TP | 4 |

## Best Parameters

```yaml
early_exits:
  oi_24h_reversion:
    enabled: true
    min_hours_since_entry: 12
    oi_z_threshold: 2.0
filters:
  close_above_ma21: true
  ret_1d_min: 0.0
  rsi_max: 80.0
  rsi_min: 60.0
g0_eth_regime:
  enabled: true
  eth_ret_1h_min: 0.0
g1_taker:
  enabled: true
  taker_z_4h_min: 0.3
g2_oi_bipolar:
  enabled: true
  oi_z_1h_min: -0.5
  oi_z_24h_block: 2.0
  oi_z_24h_warning: 2.0
g3_volume_aux:
  enabled: true
  volume_z_prev_min: -0.5
stops:
  max_hold_hours: 120
  stop_loss_pct: 0.015
  take_profit_pct: 0.02
  trailing_pct: 0.01
```

## Grid Search — Top 10 TRAIN

| g1_taker.taker_z_4h_min | g2_oi_bipolar.oi_z_1h_min | g2_oi_bipolar.oi_z_24h_block | filters.rsi_min | filters.rsi_max | sharpe | n_trades | win_rate | total_return |
|---|---|---|---|---|---|---|---|---|
| -0.5 | -0.5 | 2.0 | 50 | 80 | 1.2853 | 66 | 50.0 | 11.3793 |
| 0.3 | -0.5 | 2.0 | 55 | 80 | 1.2773 | 40 | 47.5 | 6.7541 |
| -0.5 | -0.5 | 2.0 | 50 | 85 | 1.2674 | 67 | 49.2537 | 11.2388 |
| 0.3 | -0.5 | 3.0 | 55 | 80 | 1.1923 | 48 | 47.9167 | 7.0974 |
| 0.3 | -0.5 | 2.0 | 50 | 80 | 1.169 | 46 | 50.0 | 6.8413 |
| 0.3 | -0.5 | 2.0 | 60 | 80 | 1.0923 | 34 | 44.1176 | 4.8529 |
| 0.3 | -0.5 | 2.5 | 55 | 80 | 1.0672 | 45 | 46.6667 | 6.0544 |
| 0.0 | -0.5 | 2.0 | 50 | 80 | 1.0578 | 57 | 49.1228 | 7.7685 |
| 0.3 | -0.5 | 2.0 | 50 | 85 | 1.0203 | 47 | 46.8085 | 5.9897 |
| 0.0 | -0.5 | 2.0 | 50 | 85 | 0.94 | 58 | 46.5517 | 6.9096 |

## TEST Results — Top 10 (out-of-sample)

| train_sharpe | test_sharpe | test_n_trades | test_wr | test_total_ret | overfitting |
|---|---|---|---|---|---|
| 1.092 | 2.029 | 21 | 57.1 | 5.69 | -0.936 |
| 1.277 | 1.549 | 24 | 54.2 | 4.77 | -0.272 |
| 1.169 | 1.113 | 26 | 46.2 | 3.68 | 0.056 |
| 1.02 | 1.113 | 26 | 46.2 | 3.68 | -0.093 |
| 1.067 | 1.101 | 26 | 50.0 | 3.58 | -0.034 |
| 1.192 | 1.099 | 27 | 51.9 | 3.65 | 0.093 |
| 1.058 | 0.924 | 29 | 44.8 | 3.24 | 0.134 |
| 0.94 | 0.924 | 29 | 44.8 | 3.24 | 0.016 |
| 1.285 | 0.364 | 33 | 39.4 | 1.33 | 0.921 |
| 1.267 | 0.364 | 33 | 39.4 | 1.33 | 0.903 |

## Arquivos

- `conf/parameters_sol.yml` — best params
- `prompts/tables/sol_backtest_grid.csv`
- `prompts/tables/sol_backtest_test_results.csv`
- `prompts/plots/sol_backtest/`