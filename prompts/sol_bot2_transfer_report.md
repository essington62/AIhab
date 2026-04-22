# SOL Bot 5 — Bot 2 BTC Strategy Transfer (2026 ONLY)

**Data:** 2026-04-22
**Objetivo:** Testar se Bot 2 BTC strategy transfere para SOL em 2026

---

## 1. Dados

| Asset | Período | Rows |
|-------|---------|------|
| BTC (clean) | Jan-Abr 2026 | 2675 |
| SOL (raw) | Jan-Abr 2026 | 2679 |
| stablecoin_z | Oct 2025 - Abr 15 2026 | 2679 |

> stablecoin_z com forward-fill para Abr 22 (última atualização: Abr 15)

---

## 2. Resultado BTC Bot 2 (2026 — baseline)

| Metric | BTC Bot 2 |
|--------|-----------|
| N trades | 25 |
| Sharpe | -0.90 |
| Win Rate | 24% |
| Avg Return | -0.130% |
| Total Return | -3.32% |
| Max DD | -7.06% |
| Profit Factor | 0.73 |

> Reference: live performance BTC Bot 2 Mar-Abr 2026 — WR 80%, PF 2.07, +1.83%

---

## 3. Resultado SOL Bot 2 Transfer (2026)

| Metric | SOL Bot 2 |
|--------|-----------|
| N trades | 26 |
| Sharpe | **-2.16** |
| Win Rate | 27% |
| Avg Return | -0.290% |
| Total Return | -7.39% |
| Max DD | -10.09% |
| Profit Factor | 0.49 |

---

## 4. Comparação BTC vs SOL

| Metric | BTC Bot 2 | SOL Bot 2 | Δ |
|--------|-----------|-----------|---|
| N trades | 25 | 26 | — |
| Sharpe | -0.90 | -2.16 | -1.26 |
| Win Rate | 24% | 27% | +3% |
| Avg Return | -0.130% | -0.290% | -0.160% |
| Max DD | -7.06% | -10.09% | — |
| Profit Factor | 0.73 | 0.49 | — |

---

## 5. Sub-períodos (SOL)

| Período | BTC N | BTC Sharpe | BTC WR | SOL N | SOL Sharpe | SOL WR |
|---------|-------|------------|--------|-------|------------|--------|
| Jan-Feb 2026    |  4 | -15.62 | 0% |  3 | 1.51 | 33% |
| Mar-Abr 2026    | 20 | -0.10 | 30% | 22 | -3.05 | 23% |

**SOL períodos com Sharpe > 1.0:** 1/2

---

## 6. Critérios de Decisão

| Critério | Meta | Resultado | Status |
|----------|------|-----------|--------|
| Sharpe > 1.5 | > 1.5 | -2.16 | ❌ |
| N ≥ 15 trades | ≥ 15 | 26 | ✅ |
| WR > 50% | > 50% | 27% | ❌ |
| Max DD < 5% | < 5% | -10.09% | ❌ |
| Robusto 2/2 | ≥ 2/2 | 1/2 | ❌ |

**Critérios atendidos:** 1/5

---

## 7. **VEREDITO: REJEITADO**

**Ação:** SOL não tem edge com Bot 2 strategy em 2026. Abandonar SOL por agora.

### Interpretação

**Hipótese REJEITADA:** Bot 2 strategy não transfere para SOL. O problema é o regime SOL em 2026, não as features específicas. Qualquer strategy de momentum enfrenta dificuldades.


---

## 9. Plots gerados

- `plots/sol_bot2_transfer/equity_curves.png`
- `plots/sol_bot2_transfer/returns_distribution.png`
- `plots/sol_bot2_transfer/subperiods_comparison.png`
- `plots/sol_bot2_transfer/feature_scatter_btc.png`
- `plots/sol_bot2_transfer/feature_scatter_sol.png`

## 10. Tables geradas

- `tables/sol_bot2_transfer/comparison_summary.csv`
- `tables/sol_bot2_transfer/trades_btc_vs_sol.csv`
- `tables/sol_bot2_transfer/subperiods.csv`
