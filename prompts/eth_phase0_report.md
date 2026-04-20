# 🔬 ETH Phase 0 — Statistical Study Report

**Generated:** 2026-04-20 21:26 UTC
**Window:** 180 days  |  **Method:** Pearson + Spearman + HMM regime

## 🎯 Decisão Final

### 🔴 RECALIBRATE_DEEPLY

**ETH diferente. Recalibração profunda de gates e pesos necessária.**

| Métrica | Valor | Threshold |
|---------|-------|-----------|
| Model Alignment | `0.495` | >0.70 copy / >0.40 adapt |
| Strong gates | `0` | ≥3 preferred |
| Moderate gates | `1` | — |
| Weak gates | `2` | — |
| Regime separates | `❌` | Bull>Sideways>Bear |

## H1: Correlações ETH vs Forward Returns

### Horizonte 1d

| Feature | Pearson r | p-value | Spearman r | n | Sig |
|---------|-----------|---------|------------|---|-----|
| volume | -0.142 | 0.069 | -0.100 | 165 | — |
| g9_taker | +0.058 | 0.460 | -0.000 | 165 | — |
| rsi | +0.039 | 0.606 | +0.015 | 179 | — |
| g10_funding | +0.016 | 0.837 | -0.004 | 165 | — |
| bb_pct | -0.016 | 0.832 | -0.055 | 179 | — |
| g4_oi_coin_margin | +0.015 | 0.849 | -0.034 | 165 | — |
| ret_7d | +0.005 | 0.945 | -0.054 | 179 | — |
| ret_1d | +0.002 | 0.983 | +0.040 | 179 | — |

### Horizonte 3d

| Feature | Pearson r | p-value | Spearman r | n | Sig |
|---------|-----------|---------|------------|---|-----|
| volume | -0.305 | 0.000 | -0.294 | 163 | ✅ |
| ret_1d | -0.091 | 0.231 | -0.044 | 177 | — |
| rsi | +0.076 | 0.318 | +0.056 | 177 | — |
| g4_oi_coin_margin | +0.054 | 0.494 | +0.003 | 163 | — |
| g10_funding | +0.052 | 0.508 | -0.034 | 163 | — |
| bb_pct | -0.023 | 0.762 | -0.043 | 177 | — |
| ret_7d | -0.021 | 0.785 | -0.036 | 177 | — |
| g9_taker | -0.012 | 0.875 | +0.014 | 163 | — |

### Horizonte 7d

| Feature | Pearson r | p-value | Spearman r | n | Sig |
|---------|-----------|---------|------------|---|-----|
| volume | -0.323 | 0.000 | -0.269 | 159 | ✅ |
| g4_oi_coin_margin | +0.168 | 0.035 | +0.129 | 159 | ✅ |
| rsi | +0.142 | 0.063 | +0.096 | 173 | — |
| g10_funding | +0.076 | 0.341 | +0.027 | 159 | — |
| ret_7d | +0.074 | 0.334 | +0.060 | 173 | — |
| bb_pct | +0.068 | 0.374 | +0.072 | 173 | — |
| ret_1d | +0.009 | 0.911 | -0.019 | 173 | — |
| g9_taker | -0.005 | 0.946 | +0.013 | 159 | — |

### Horizonte 14d

| Feature | Pearson r | p-value | Spearman r | n | Sig |
|---------|-----------|---------|------------|---|-----|
| volume | -0.387 | 0.000 | -0.401 | 152 | ✅ |
| ret_7d | +0.107 | 0.170 | +0.069 | 166 | — |
| g4_oi_coin_margin | +0.100 | 0.218 | +0.067 | 152 | — |
| g10_funding | +0.073 | 0.369 | +0.010 | 152 | — |
| bb_pct | +0.059 | 0.448 | +0.050 | 166 | — |
| g9_taker | +0.045 | 0.579 | +0.086 | 152 | — |
| rsi | +0.020 | 0.801 | -0.012 | 166 | — |
| ret_1d | +0.011 | 0.891 | +0.008 | 166 | — |

## H2: Poder Preditivo dos Gates (7d forward return)

| Gate | BTC ref | ETH actual | |Δ| | Power | Action | Sig |
|------|---------|------------|-----|-------|--------|-----|
| g4_oi_coin_margin | -0.472 | +0.168 | 0.640 | 🟡 MODERATE | `keep_reduced_weight` | ✅ |
| g5_stablecoin | +0.326 | N/A | — | ⚪ NO_DATA | `skip` | — |
| g7_etf | +0.263 | N/A | — | ⚪ NO_DATA | `skip` | — |
| g6_bubble | -0.345 | N/A | — | ⚪ NO_DATA | `skip` | — |
| g3_dgs10 | -0.315 | N/A | — | ⚪ NO_DATA | `skip` | — |
| g3_curve | -0.280 | N/A | — | ⚪ NO_DATA | `skip` | — |
| g9_taker | +0.060 | -0.005 | 0.065 | 🔴 WEAK | `discard` | — |
| g10_funding | +0.023 | +0.076 | 0.053 | 🔴 WEAK | `discard` | — |
| g8_fg | +0.150 | N/A | — | ⚪ NO_DATA | `skip` | — |

## H3: R5C HMM aplicado ao ETH

*R5C não disponível ou falhou.*

## H4: Model Alignment

**Alignment = 0.495**

🟡 Parcialmente similar → adaptar

## 🎯 Próximos Passos

1. Retreinar R5C HMM com dados ETH (dados acumulam ao longo do tempo)
2. Re-examinar correlações em janelas diferentes (90d, 360d)
3. Considerar features ETH-específicas (staking yield, ETH/BTC ratio)

## 📊 Plots

- [Correlações BTC vs ETH](plots/fase0/correlations_eth_vs_btc.png)
- [Regimes R5C em ETH](plots/fase0/regimes_eth.png)
- [Matriz de correlação ETH](plots/fase0/correlation_matrix_eth.png)