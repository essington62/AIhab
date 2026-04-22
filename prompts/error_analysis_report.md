# Error Analysis — Losers do Bot 2

**Generated:** 2026-04-21 13:29 UTC
**Sinais analisados:** 136
**Winners:** 91 (66.9%)
**Losers:** 45 (33.1%)
**Período:** 2026-02-24 → 2026-04-16

## Top Features — Cohen's d (Losers vs Winners)

| Feature | Winner μ | Loser μ | Diff | Cohen's d | p-value | Sig |
|---------|---------|--------|------|-----------|---------|-----|
| taker_z | 0.279 | -0.486 | +0.765 | +0.635 | 0.002 | ✅ |
| close_vs_ma200 | 0.020 | 0.034 | -0.014 | -0.591 | 0.002 | ✅ |
| funding_z | 0.180 | -0.538 | +0.718 | +0.588 | 0.002 | ✅ |
| oi_z | 0.843 | 0.225 | +0.617 | +0.564 | 0.003 | ✅ |
| trend_slope_3h | 0.001 | -0.001 | +0.001 | +0.436 | 0.026 | ✅ |
| bb_pct | 0.772 | 0.724 | +0.048 | +0.370 | 0.070 |  |
| hour_of_day | 10.516 | 13.089 | -2.572 | -0.369 | 0.056 |  |
| lower_wick | 0.254 | 0.311 | -0.057 | -0.275 | 0.276 |  |
| ret_1h | 0.001 | -0.001 | +0.001 | +0.274 | 0.191 |  |
| atr_pct | 0.009 | 0.009 | -0.001 | -0.241 | 0.400 |  |
| candle_body | 0.053 | -0.060 | +0.113 | +0.228 | 0.197 |  |
| ret_3h | 0.002 | 0.001 | +0.002 | +0.223 | 0.271 |  |
| ret_1d | 0.021 | 0.024 | -0.003 | -0.216 | 0.526 |  |
| is_weekend | 0.253 | 0.178 | +0.075 | +0.178 | 0.330 |  |
| rsi_14 | 63.238 | 61.984 | +1.254 | +0.161 | 0.307 |  |

## Clustering dos Losers

**K ótimo:** 2 (silhouette=0.253)
**Features usadas:** rsi_14, bb_pct, ret_1d, ret_3h, ret_1h, stablecoin_z, atr_pct, volume_z, oi_z, funding_z, taker_z, hour_of_day, day_of_week, is_weekend, candle_body, upper_wick, lower_wick, close_vs_ma21, close_vs_ma200, trend_slope_3h

### Cluster 0 — n=6 (13%)

| Feature | Média |
|---------|-------|
| rsi_14 | 75.996 |
| hour_of_day | 7.667 |
| day_of_week | 5.833 |
| stablecoin_z | 1.408 |
| funding_z | 1.100 |
| taker_z | -1.006 |
| is_weekend | 1.000 |
| bb_pct | 0.839 |

### Cluster 1 — n=39 (87%)

| Feature | Média |
|---------|-------|
| rsi_14 | 59.829 |
| hour_of_day | 13.923 |
| stablecoin_z | 1.977 |
| day_of_week | 1.949 |
| funding_z | -0.790 |
| bb_pct | 0.706 |
| taker_z | -0.406 |
| lower_wick | 0.330 |

## Hipóteses de Filtro

| Feature | Threshold | Direction | N bloqueados | Winners perdidos | WR antes | WR depois | Ganho |
|---------|-----------|-----------|-------------|-----------------|----------|-----------|-------|
| taker_z | -0.96 | block_below | 34 | 14 | 66.9% | 75.5% | +8.6pp |
| funding_z | -1.37 | block_below | 34 | 15 | 66.9% | 74.5% | +7.6pp |
| close_vs_ma200 | 0.04 | block_above | 34 | 16 | 66.9% | 73.5% | +6.6pp |
| trend_slope_3h | -0.00 | block_below | 34 | 18 | 66.9% | 71.6% | +4.7pp |
| oi_z | -0.10 | block_below | 34 | 20 | 66.9% | 69.6% | +2.7pp |

## Conclusão

**Padrão encontrado:** `taker_z` é o discriminador mais forte (Cohen's d=+0.63, p=0.002).
Filtro mais promissor: block `taker_z` block below -0.96 → +8.6pp WR, bloqueando 34 sinais (14 winners perdidos).

## Plots

- `prompts/plots/error_analysis/feature_comparison.png`
- `prompts/plots/error_analysis/clusters.png`
- `prompts/plots/error_analysis/temporal_patterns.png`
- `prompts/plots/error_analysis/hypothesis_impact.png`