# Auditoria Look-Ahead Bias — taker_z

**Generated:** 2026-04-21 13:55 UTC

## 🎯 Veredito

### ⚠️ BIAS LEVE (0.48)

Algum inflação detectada mas moderada. Resultado real ≈ Sharpe 3.61.

## 1. Estrutura Temporal: Qual 4h Candle Cada Sinal Usa?

- Sinais analisados: 136
- Média horas no 4h candle ativo na entrada: 1.5h
- Distribuição (0=início, 3=fim do candle): min=0h max=4h
- Sinais com taker_z[prev_4h] disponível: 136
- Diff média |t0 - prev_4h|: 1.275
- Diff máxima |t0 - prev_4h|: 4.061

## 2. Comparação de Filtros: t=0 vs Production-Safe

| Versão | N kept | WR% | Sharpe | Expectância | Max DD | N bloq | L/W bloq |
|--------|--------|-----|--------|-------------|--------|--------|----------|
| BASELINE | 136 | 66.9% | 2.71 | +0.589% | -19.8% | 0 | 0L/0W |
| t=0 (backtest, potencial look-ahead) | 102 | 75.5% | 4.09 | +0.822% | -10.0% | 34 | 20L/14W |
| prev_4h (produção-safe) | 117 | 71.8% | 3.61 | +0.760% | -18.6% | 19 | 12L/7W |
| lag_1h (produção-safe) | 105 | 78.1% | 4.71 | +0.923% | -11.4% | 31 | 22L/9W |

## 3. Impacto do Look-Ahead no Sharpe

| Métrica | Valor |
|---------|-------|
| Baseline Sharpe | 2.71 |
| Filter t=0 Sharpe | 4.09 (Δ=+1.37) |
| Filter prev_4h Sharpe | 3.61 (Δ=+0.89) |
| Inflação por look-ahead | 0.48 |

## 4. Teste Causal: Correlação taker_z vs return_pct

Se taker_z[t=0] tem correlação MUITO MAIOR com return do que prev_4h,
é evidência de look-ahead (usando futuro para prever futuro).

| Versão | Corr(taker_z, return_pct) | N |
|--------|--------------------------|---|
| taker_z_t0 | 0.2757 | 136 |
| taker_z_prev_4h | 0.1399 | 136 |
| taker_z_lag1h | 0.3285 | 136 |

⚠️ taker_z[t=0] tem correlação muito maior — sinal de look-ahead.

## 5. Próximos Passos

1. **Sharpe real do filtro: 3.61** (não 4.09)
2. Regerar error_analysis com `taker_z_prev_4h` ao invés de `taker_z_t0`
3. Regerar filter_validation com dados corrigidos
4. Verificar se other features (funding_z, oi_z) têm o mesmo bias
5. Sharpe 3.61 ainda vale integrar (baseline=2.71)