# 📊 MFE/MAE Study — Bot 2 Entry Signals

**Generated:** 2026-04-20 14:49 UTC
**Period:** 2026-02-24 → 2026-04-15
**Signals analyzed:** 131

## 🎯 Sumário Executivo

Em 131 sinais históricos do Bot 2, nas 24h seguintes à entrada:
- **MFE P50:** +2.91%  |  **MFE P75:** +4.10%  |  **MFE P90:** +4.92%
- **MAE P50:** -1.28%  |  **MAE P25:** -2.50%  |  **MAE P10:** -3.37%

**Interpretação:** o TP atual de +2.0% é atingido em apenas 63% dos trades.
O SL atual de -1.5% é atingido em 40% dos trades nas primeiras 24h.

## ⏱️ Percentis por Janela de Tempo

### MFE (ganho máximo atingido)

| Janela | n | P10 | P25 | P50 | P75 | P90 |
|--------|---|-----|-----|-----|-----|-----|
| 4h | 131 | +0.16% | +0.37% | +0.83% | +1.66% | +2.30% |
| 12h | 131 | +0.29% | +0.72% | +2.10% | +2.91% | +3.88% |
| 24h | 131 | +0.39% | +1.05% | +2.91% | +4.10% | +4.92% |
| 48h | 131 | +0.62% | +2.49% | +4.23% | +5.85% | +6.80% |
| 72h | 131 | +1.41% | +3.36% | +4.63% | +5.87% | +7.02% |
| 120h | 131 | +1.45% | +4.36% | +6.13% | +7.79% | +9.77% |

### MAE (perda máxima atingida)

| Janela | n | P10 | P25 | P50 | P75 | P90 |
|--------|---|-----|-----|-----|-----|-----|
| 4h | 131 | -1.56% | -1.09% | -0.62% | -0.24% | -0.12% |
| 12h | 131 | -2.31% | -1.56% | -0.95% | -0.37% | -0.18% |
| 24h | 131 | -3.37% | -2.50% | -1.28% | -0.52% | -0.20% |
| 48h | 131 | -5.32% | -3.00% | -1.56% | -0.62% | -0.23% |
| 72h | 131 | -7.67% | -3.80% | -2.11% | -0.97% | -0.40% |
| 120h | 131 | -8.37% | -4.69% | -3.05% | -1.67% | -0.84% |

## 📈 Análise Condicional por BB% na Entrada

| BB% Range | n | MFE P50 24h | MAE P50 24h | MFE P50 48h | MAE P50 48h |
|-----------|---|-------------|-------------|-------------|-------------|
| BB 0.3-0.6 (mid) | 24 | +1.18% | -2.15% | +1.46% | -2.41% |
| BB>0.6 (topo) | 107 | +3.31% | -0.99% | +4.48% | -1.44% |

## 🎯 Grid TP × SL — Top 10 por Expectancy (janela 24h)

| # | TP | SL | R:R | Win Rate | Loss Rate | Timeout | Expectancy |
|---|-----|-----|-----|----------|-----------|---------|------------|
| 1 | +2.0% | -1.5% | 1.33:1 | 61.1% | 34.4% | 4.6% | +0.706% |
| 2 | +2.0% | -1.0% | 2.00:1 | 56.5% | 43.5% | 0.0% | +0.695% |
| 3 | +2.0% | -2.0% | 1.00:1 | 62.6% | 29.0% | 8.4% | +0.672% |
| 4 | +2.0% | -0.8% | 2.50:1 | 49.6% | 50.4% | 0.0% | +0.589% |
| 5 | +1.5% | -1.0% | 1.50:1 | 60.3% | 39.7% | 0.0% | +0.508% |
| 6 | +1.5% | -1.5% | 1.00:1 | 64.9% | 31.3% | 3.8% | +0.504% |
| 7 | +1.5% | -2.0% | 0.75:1 | 66.4% | 26.0% | 7.6% | +0.477% |
| 8 | +2.0% | -0.5% | 4.00:1 | 38.9% | 61.1% | 0.0% | +0.473% |
| 9 | +1.5% | -0.8% | 1.88:1 | 53.4% | 46.6% | 0.0% | +0.429% |
| 10 | +1.5% | -0.5% | 3.00:1 | 42.7% | 57.3% | 0.0% | +0.355% |

## ⚖️ Config Atual vs Melhor Encontrada

### Atual: TP +2.0% / SL -1.5%
- Win Rate: **61.1%**
- Loss Rate: 34.4%
- Timeout: 4.6%
- Expectancy: **+0.706%**

### Melhor: TP +2.0% / SL -1.5%
- Win Rate: **61.1%**
- Loss Rate: 34.4%
- Timeout: 4.6%
- Expectancy: **+0.706%**

**Δ Expectancy: +0.000pp** | **Δ Win Rate: +0.0pp**

## 📊 Plots

- [Distribuição MFE/MAE 24h](plots/mfe_mae_distribution.png)
- [Scatter MFE vs MAE](plots/mfe_mae_scatter.png)
- [Heatmap Expectancy + WR](plots/tp_sl_expectancy.png)
- [Percentis por janela](plots/mfe_mae_by_window.png)

## 💡 Interpretação

**Regra de ouro:**
- TP ideal ≈ P25–P50 do MFE → ganhar o que o mercado dá na metade dos trades
- SL ideal ≈ P40–P50 do MAE → aceitar a perda que já acontece na metade dos casos
- Timeout alto (> 30%) indica que TP muito distante → mais trades ficam presos sem direção

## 🎯 Próximos Passos

1. Se melhor config tiver Expectancy significativamente maior → ajustar `parameters.yml`
2. Se BB% condicional mostrar grande variação → considerar stops por zona (BB%)
3. Rodar `backtest_bot2_v2.py` com nova config para confirmar