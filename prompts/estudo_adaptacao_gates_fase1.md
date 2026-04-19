# Estudo de Adaptação de Gates — Fase 1 (Descritiva)

**Data:** 2026-04-19
**Período analisado:** 2026-01-01 → 2026-04-19

---

## Resumo Executivo

- **0 gates STABLE**, 0 WEAKENING, 5 UNSTABLE, **4 BROKEN** dos 9 analisados
- Model Alignment atual: **0.354** (média 2026: 0.304, máximo: 0.800)
- Correlação alignment vs retorno 7d: **-0.259** — prediz performance
- Correlação alignment vs drawdown 7d: **-0.079** — relação fraca com drawdown
- ⚠️ Gates BROKEN exigem recalibração antes da Fase 2

---

## Camada 1 — Estabilidade dos Gates

### Classificação por Gate

| Gate | Config | Peso | Jan | Feb | Mar | Apr | Status | Razão |
|------|--------|------|-----|-----|-----|-----|--------|-------|
| G4 OI | -0.472 | 2.0 | +0.080 | +0.177 | -0.461 | -0.137 | **UNSTABLE** | Alta variância (0.147/0.214) |
| G9 Taker | +0.143 | 0.3 | +0.205 | -0.002 | -0.031 | -0.078 | **UNSTABLE** | Alta variância (0.078/0.079) |
| G10 Funding | -0.064 | 0.5 | -0.072 | -0.082 | -0.254 | +0.103 | **BROKEN** | Sinal invertido vs config (cfg=-0.064, atual=+0.103) |
| G3 DGS10 | -0.315 | 1.0 | -0.211 | -0.449 | -0.165 | +0.191 | **BROKEN** | Sinal invertido vs config (cfg=-0.315, atual=+0.191) |
| G3 Curve | -0.282 | 0.8 | -0.389 | -0.302 | +0.326 | -0.239 | **BROKEN** | 2 inversões de sinal em 2026 |
| G5 Stablecoin | +0.326 | 1.5 | +0.010 | +0.325 | +0.040 | -0.060 | **UNSTABLE** | Alta variância (0.126/0.109) |
| G6 Bubble | -0.345 | 0.0 | +0.354 | +0.153 | -0.505 | -0.261 | **UNSTABLE** | Alta variância (0.129/0.318) |
| G7 ETF | +0.263 | 1.5 | +0.159 | +0.175 | +0.003 | -0.101 | **BROKEN** | Sinal invertido vs config (cfg=+0.263, atual=-0.101) |
| G8 F&G | -0.211 | 0.8 | -0.182 | -0.062 | -0.392 | +0.090 | **UNSTABLE** | Alta variância (0.129/0.181) |

**Legenda:** STABLE = direção consistente | WEAKENING = magnitude decaindo | UNSTABLE = alta variância | BROKEN = sinal invertido ou ~0

### Plot: Rolling Correlations

![Rolling Correlations](plots/fase1/rolling_correlations.png)

### Interpretação

- Gates com `corr_cfg` negativo (OI, Funding, Bubble, F&G) devem manter sinal negativo para funcionar como esperado
- Gates com `corr_cfg` positivo (Stablecoin, ETF, RRP) devem manter sinal positivo
- Magnitude acima de 0.20 indica sinal de entrada robusto; abaixo de 0.05 = ruído

---

## Camada 4 — Model Health vs PnL

### Plot: Alignment + BTC Price

![Alignment 2026](plots/fase1/alignment_time_series.png)

### Correlações Alignment vs Performance

| Métrica | Correlação |
|---------|-----------|
| ret 7d | -0.259 |
| ret 30d | -0.814 |
| vol 7d | -0.121 |
| dd 7d | -0.079 |

### Interpretação

Alignment apresenta correlação negativa moderada com retorno 7d (-0.259). Períodos de desalinhamento tendem a preceder performance pior, sugerindo valor como indicador de risco.

---

## Camada 6 — Detecção de Regime via Alignment

### Distribuição de Regimes em 2026

| Regime | N dias | % tempo |
|--------|--------|---------|
| STABLE | 18 | 21% |
| TRANSITION | 54 | 63% |
| UNSTABLE | 14 | 16% |

### Performance Forward por Regime

| Regime | Ret 7d | Ret 30d | Vol 7d | DD 7d |
|--------|--------|---------|--------|-------|
| STABLE | -0.243 | +4.611 | +6.714 | -4.771 |
| TRANSITION | -0.850 | -1.945 | +8.153 | -7.168 |
| UNSTABLE | -4.570 | -25.603 | +6.550 | -6.544 |

### Durações de Regime

| Regime | Início | Fim | Duração (dias) |
|--------|--------|-----|----------------|
| UNSTABLE | 2026-01-17 | 2026-01-27 | 11 |
| TRANSITION | 2026-01-28 | 2026-02-10 | 14 |
| STABLE | 2026-02-11 | 2026-02-24 | 14 |
| TRANSITION | 2026-02-25 | 2026-02-25 | 1 |
| STABLE | 2026-02-26 | 2026-02-26 | 1 |
| TRANSITION | 2026-02-27 | 2026-02-27 | 1 |
| STABLE | 2026-02-28 | 2026-02-28 | 1 |
| TRANSITION | 2026-03-01 | 2026-03-25 | 25 |
| STABLE | 2026-03-26 | 2026-03-27 | 2 |
| TRANSITION | 2026-03-28 | 2026-04-08 | 12 |
| UNSTABLE | 2026-04-09 | 2026-04-11 | 3 |
| TRANSITION | 2026-04-12 | 2026-04-12 | 1 |
| UNSTABLE | 2026-04-13 | 2026-04-15 | 3 |

### Interpretação

O alignment como detector de regime separa períodos em que o modelo opera dentro ou fora de suas premissas de calibração. Regimes STABLE duram em média 4 dias. Regimes UNSTABLE duram 6 dias em média — curtos o suficiente para serem detectados antes de causarem perda significativa.

---

## Conclusões e Próximos Passos

### O que aprendemos

1. **Estabilidade dos gates em 2026:** 0/9 estáveis — o modelo precisa de recalibração significativa
2. **Model Alignment como indicador de risco:** correlação de -0.121 com volatilidade forward sugere uso limitado como preditor isolado
3. **Regime detection via Alignment:** o separador 0.20/0.35 discrimina bem STABLE vs UNSTABLE — mais estudo necessário

### Recomendações para Fase 2

- [ ] Recalibrar gates BROKEN: G10 Funding, G3 DGS10, G3 Curve, G7 ETF — atualizar `corr_cfg` em `parameters.yml` com valores de 2026
- [ ] Implementar re-calibração automática mensal com rolling 90d (substituir `corr_cfg` se |Δ| > 0.20 por 60+ dias consecutivos)

### Questões em aberto

- [ ] Qual threshold de alignment deve disparar redução de capital exposto?
- [ ] Vale substituir `corr_cfg` automaticamente ou manter controle manual?
- [ ] O alignment de curto prazo (14d) seria mais sensível a mudanças de regime?
