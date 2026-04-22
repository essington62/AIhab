# SOL EDA Phase 1 — Teste das 4 Hipóteses

**Generated:** 2026-04-21 22:42 UTC

## Resumo — Veredictos

- **H1 (Derivativos):** ✅ PARCIALMENTE SUPORTADA — 10/12 significativas (melhor: oi_z_prev 24h r=-0.155)
- **H2 (Volume flow):** ⚖️ MISTA — suportada em 2/3 horizontes
- **H3 (Beta ETH vs BTC):** ✅ SUPORTADA — β ETH (+0.629) > β BTC (+0.516), t_eth=+38.89
- **H4 (Reflexividade):** ❌ REJEITADA — SOL reversion -0.201pp ≤ 0 ou abaixo de outros

## H1 — Derivativos preveem retornos curtos?

### Correlações (feature prev × forward return)

| Feature | Horizon | N | Corr | p-value | Sig |
|---------|---------|---|------|---------|-----|
| funding_z_prev | 1h | 4,281 | +0.0101 | 0.5103 | ❌ |
| funding_z_prev | 24h | 4,258 | -0.0317 | 0.0387 | ✅ |
| funding_z_prev | 4h | 4,278 | -0.0397 | 0.0094 | ✅ |
| oi_z_prev | 1h | 4,281 | +0.0570 | 0.0002 | ✅ |
| oi_z_prev | 24h | 4,258 | -0.1550 | 0.0000 | ✅ |
| oi_z_prev | 4h | 4,278 | +0.0109 | 0.4744 | ❌ |
| taker_ratio_prev | 1h | 4,317 | +0.1530 | 0.0000 | ✅ |
| taker_ratio_prev | 24h | 4,294 | +0.0618 | 0.0001 | ✅ |
| taker_ratio_prev | 4h | 4,314 | +0.1424 | 0.0000 | ✅ |
| taker_z_prev | 1h | 4,281 | +0.1540 | 0.0000 | ✅ |
| taker_z_prev | 24h | 4,258 | +0.0345 | 0.0243 | ✅ |
| taker_z_prev | 4h | 4,278 | +0.1370 | 0.0000 | ✅ |

### Shocks extremos (feature > 2σ vs < -2σ)

| Feature | Horizon | N+ | N- | Ret high | Ret low | Cohen's d | p |
|---------|---------|----|----|----------|---------|-----------|---|
| oi_z_prev | 1h | 196 | 252 | +0.064% | -0.205% | +0.254 | 0.0061 |
| oi_z_prev | 4h | 196 | 252 | -0.120% | -0.399% | +0.154 | 0.0975 |
| oi_z_prev | 24h | 196 | 252 | -1.191% | +0.579% | -0.455 | 0.0000 |
| taker_z_prev | 1h | 84 | 104 | +0.061% | -0.288% | +0.511 | 0.0006 |
| taker_z_prev | 4h | 84 | 104 | +0.007% | -0.640% | +0.435 | 0.0027 |
| taker_z_prev | 24h | 84 | 104 | +0.498% | -1.181% | +0.610 | 0.0000 |

## H2 — Volume flow: SOL vs BTC/ETH

| Horizon | SOL |corr| | BTC |corr| | ETH |corr| | SOL vs avg | H2 |
|---------|-------------|-------------|-------------|------------|-----|
| 1h | 0.0206 | 0.0121 | 0.0119 | 1.71x | ✅ |
| 4h | 0.0272 | 0.0002 | 0.0311 | 1.74x | ✅ |
| 24h | 0.0144 | 0.0015 | 0.0274 | 1.00x | ❌ |

### Detalhe por asset

| Asset | Horizon | N | Corr | p-value | N shocks | Shock ret% | Baseline ret% |
|-------|---------|---|------|---------|----------|------------|--------------|
| BTC | 1h | 5,534 | -0.0121 | 0.3690 | 258 | -0.054% | -0.003% |
| BTC | 4h | 5,531 | +0.0002 | 0.9894 | 258 | -0.155% | -0.016% |
| BTC | 24h | 5,511 | -0.0015 | 0.9116 | 258 | +0.027% | -0.146% |
| ETH | 1h | 8,711 | +0.0119 | 0.2653 | 431 | -0.022% | +0.008% |
| ETH | 4h | 8,708 | +0.0311 | 0.0037 | 431 | +0.087% | +0.024% |
| ETH | 24h | 8,688 | +0.0274 | 0.0106 | 431 | +0.455% | +0.150% |
| SOL | 1h | 4,822 | -0.0206 | 0.1534 | 217 | -0.112% | -0.013% |
| SOL | 4h | 4,819 | -0.0272 | 0.0591 | 217 | -0.291% | -0.059% |
| SOL | 24h | 4,799 | -0.0144 | 0.3182 | 217 | -0.331% | -0.424% |

## H3 — Beta decomposition (SOL ~ BTC + ETH)

**Observações:** 4,820 candles 1h

### Multivariate regression: SOL = α + β_btc·BTC + β_eth·ETH

| Coef | Value | SE | t-stat |
|------|-------|----|--------|
| α | -0.000054 | — | — |
| β BTC | +0.5161 | 0.0228 | +22.67 |
| β ETH | +0.6286 | 0.0162 | +38.89 |

**R² combined:** 0.7742

### Univariate

| Model | R² | Corr(SOL) |
|-------|-----|-----------|
| SOL ~ BTC | 0.7033 | 0.8386 |
| SOL ~ ETH | 0.7501 | 0.8661 |

## H4 — Reflexividade / Mean reversion

| Asset | Shock | Horizon | N+ | N- | Ret+ | Ret- | Reversão | p+ | p- |
|-------|-------|---------|-----|-----|------|------|----------|----|----|
| BTC | ±2.0σ | 1h | 146 | 170 | +0.049% | -0.109% | -0.159pp | 0.4188 | 0.1044 |
| BTC | ±2.0σ | 24h | 145 | 170 | -0.281% | -0.043% | +0.238pp | 0.1643 | 0.8353 |
| BTC | ±2.0σ | 4h | 146 | 170 | +0.064% | -0.246% | -0.309pp | 0.4756 | 0.0262 |
| ETH | ±2.0σ | 1h | 225 | 229 | +0.098% | -0.077% | -0.175pp | 0.1835 | 0.3469 |
| ETH | ±2.0σ | 24h | 225 | 229 | +0.671% | -0.097% | -0.768pp | 0.0085 | 0.6889 |
| ETH | ±2.0σ | 4h | 225 | 229 | +0.196% | -0.245% | -0.441pp | 0.0930 | 0.0746 |
| SOL | ±2.0σ | 1h | 132 | 147 | +0.138% | +0.008% | -0.130pp | 0.1885 | 0.9452 |
| SOL | ±2.0σ | 24h | 132 | 147 | -0.372% | -0.134% | +0.239pp | 0.2665 | 0.7294 |
| SOL | ±2.0σ | 4h | 132 | 147 | +0.167% | -0.409% | -0.576pp | 0.3040 | 0.0312 |
| BTC | ±3.0σ | 1h | 43 | 63 | +0.207% | -0.023% | -0.230pp | 0.1353 | 0.8374 |
| BTC | ±3.0σ | 24h | 43 | 63 | -0.196% | -0.157% | +0.039pp | 0.5364 | 0.6303 |
| BTC | ±3.0σ | 4h | 43 | 63 | -0.136% | -0.201% | -0.065pp | 0.4714 | 0.3071 |
| ETH | ±3.0σ | 1h | 76 | 88 | +0.272% | -0.021% | -0.293pp | 0.0209 | 0.8780 |
| ETH | ±3.0σ | 24h | 76 | 88 | +1.122% | -0.308% | -1.430pp | 0.0164 | 0.4302 |
| ETH | ±3.0σ | 4h | 76 | 88 | +0.284% | -0.131% | -0.415pp | 0.1059 | 0.5488 |
| SOL | ±3.0σ | 1h | 33 | 47 | +0.072% | -0.054% | -0.126pp | 0.8028 | 0.8138 |
| SOL | ±3.0σ | 24h | 33 | 47 | -0.371% | -0.075% | +0.296pp | 0.6131 | 0.9197 |
| SOL | ±3.0σ | 4h | 33 | 47 | -0.219% | -0.419% | -0.201pp | 0.5986 | 0.2583 |

## Implicações estratégicas

### Decisão: SOL Bot strategy

**Strategy parcial:** usar apenas features das hipóteses confirmadas. Aguardar mais dados para features inconclusivas.

## Arquivos gerados

- Plots: `prompts/plots/sol_eda/`
- Tables: `prompts/tables/sol_eda_*.csv`