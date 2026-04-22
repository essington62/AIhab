# Mini-Backtest: Validação Filtro Taker_Z

**Generated:** 2026-04-21 13:40 UTC
**Dataset:** 136 trades, 2026-02-24 → 2026-04-16

## 1. Comparação de Cenários

| Cenário | N | WR% | Expectância | Sharpe | Total Ret% | Max DD% | Bloq |
|---------|---|-----|-------------|--------|------------|---------|------|
| BASELINE | 136 | 66.9% | +0.589% | 2.71 | +118.70% | -19.8% | 0 |
| taker_z < -0.5 | 93 | 75.3% | +0.863% | 4.24 | +120.08% | -10.0% | 43 |
| taker_z < -0.8 | 101 | 76.2% | +0.845% | 4.23 | +131.60% | -10.0% | 35 |
| taker_z < -0.96 | 102 | 75.5% | +0.822% | 4.09 | +128.13% | -10.0% | 34 |
| taker_z < -1.0 | 102 | 75.5% | +0.822% | 4.09 | +128.13% | -10.0% | 34 |
| taker_z < -1.2 | 107 | 73.8% | +0.779% | 3.80 | +126.82% | -11.4% | 29 |
| taker_z < -1.5 | 116 | 72.4% | +0.753% | 3.62 | +135.74% | -11.4% | 20 |
| funding_z < -0.8 | 93 | 75.3% | +0.782% | 3.90 | +104.46% | -16.0% | 43 |
| funding_z < -1.0 | 93 | 75.3% | +0.782% | 3.90 | +104.46% | -16.0% | 43 |
| funding_z < -1.37 | 101 | 74.3% | +0.775% | 3.81 | +115.74% | -15.6% | 35 |
| funding_z < -1.5 | 115 | 70.4% | +0.681% | 3.22 | +115.29% | -15.6% | 21 |
| taker_z<-0.96 AND funding_z<-1.37 | 121 | 71.1% | +0.704% | 3.35 | +130.40% | -15.6% | 15 |
| taker_z<-0.96 OR funding_z<-1.37 | 82 | 80.5% | +0.939% | 4.99 | +113.61% | -7.5% | 54 |
| taker_z<-1.0 OR funding_z<-1.0 (robusto) | 74 | 82.4% | +0.966% | 5.31 | +102.44% | -7.5% | 62 |
| close_vs_ma200 > 0.04 | 98 | 73.5% | +0.828% | 3.98 | +121.86% | -9.9% | 38 |
| TRIPLE AND | 133 | 68.4% | +0.636% | 2.96 | +128.85% | -17.4% | 3 |

## 2. Sensibilidade ao Threshold (Overfit Check)

| Threshold | Sharpe | Δ Sharpe | Expectância | Δ Expect | Bloq (L/W) |
|-----------|--------|----------|-------------|----------|------------|
| taker_z < -1.5 | 3.62 | +0.90 | +0.753% | +0.164pp | 20 (13L/7W) |
| taker_z < -1.2 | 3.80 | +1.09 | +0.779% | +0.190pp | 29 (17L/12W) |
| taker_z < -1.0 | 4.09 | +1.37 | +0.822% | +0.233pp | 34 (20L/14W) |
| taker_z < -0.96 | 4.09 | +1.37 | +0.822% | +0.233pp | 34 (20L/14W) |
| taker_z < -0.8 | 4.23 | +1.52 | +0.845% | +0.256pp | 35 (21L/14W) |
| taker_z < -0.5 | 4.24 | +1.52 | +0.863% | +0.273pp | 43 (22L/21W) |

## 3. Robustez em Sub-Períodos

| Período | N | WR% | Sharpe | Expectância | Bloqueados |
|---------|---|-----|--------|-------------|------------|
| T1 (trades 1-45) — BASELINE | 45 | 60.0% | 1.25 | +0.269% | 0 |
| T1 (trades 1-45) — taker_z<-0.96 | 35 | 71.4% | 3.05 | +0.624% | 10 (8L/2W) |
| T2 (trades 46-90) — BASELINE | 45 | 75.6% | 4.69 | +0.960% | 0 |
| T2 (trades 46-90) — taker_z<-0.96 | 33 | 78.8% | 5.17 | +1.006% | 12 (4L/8W) |
| T3 (trades 91-136) — BASELINE | 46 | 65.2% | 2.45 | +0.540% | 0 |
| T3 (trades 91-136) — taker_z<-0.96 | 34 | 76.5% | 4.23 | +0.848% | 12 (8L/4W) |

## 4. Veredito

### ✅ INTEGRAR

**Filtro principal:** `taker_z < -0.96`
- Sharpe: 2.71 → 4.09 (+1.37)
- Expectância: +0.589% → +0.822% (+0.233pp)
- Total Return: +118.70% → +128.13%
- Max DD: -19.8% → -10.0% (+9.8pp)
- Bloqueados: 34 (20L / 14W)

**Avaliação:**
Filtro melhora Sharpe (+1.37), expectância (+0.233pp), é robusto ao threshold (range Sharpe=0.43 entre -0.8 e -1.2) e consistente em todos os sub-períodos (CV=26%).

**Filtro robusto alternativo:** `taker_z<-1.0 OR funding_z<-1.0`
- Sharpe: 5.31 (+2.59)
- Expectância: +0.966% (+0.377pp)

**Melhor filtro geral (Sharpe):** `taker_z<-1.0 OR funding_z<-1.0 (robusto)`
- Sharpe: 5.31 (+2.59)

**Próximo passo:** Implementar `taker_z < -1.0` em `paper_trader.py` (threshold round, mais robusto que -0.96).

## 5. Análise de Robustez

- **Threshold stability (range Sharpe -0.8→-1.2):** 0.431 ✅ robusto
- **Consistência sub-períodos (CV):** 25.5% ✅ consistente
- **Melhora em todos os tercis:** ✅ sim

## 6. Visualizações

- `prompts/plots/filter_validation/pnl_comparison.png`
- `prompts/plots/filter_validation/threshold_sensitivity.png`
- `prompts/plots/filter_validation/subperiod_robustness.png`
- `prompts/plots/filter_validation/equity_curves.png`