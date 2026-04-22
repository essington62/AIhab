# SOL Bot 4 v2 "Sweet Spot" — Backtest Report

**Data:** 2026-04-22
**Objetivo:** Validar strategy v2 antes de reativar Bot 4 (pausado desde 22/04)

---

## 1. v1 Baseline (replicação)

| Metric | v1 Full History |
|--------|----------------|
| Sharpe | 0.76 |
| Trades | 85 |
| Win Rate | 38% |
| Avg Return | 0.117% |
| Max DD | -6.90% |

> Filter study confirmado: Sharpe 0.76 (Phase 2 reportou 2.03 em test set N=21 — lucky sampling)

---

## 2. v2 Sweet Spot Central

Config: `close/MA21 ∈ [1.03, 1.05]` e `volume_z ∈ [0.0, 0.5]`

| Metric | v2 Central |
|--------|-----------|
| Sharpe | **0.26** |
| Trades | 9 |
| Win Rate | 44% |
| Avg Return | 0.043% |
| Max DD | -2.97% |
| Total Return | +0.33% |
| Profit Factor | 1.10 |
| Δ vs v1 | -0.50 |

---

## 3. Grid Search — Variações

| Config | N | WR | Avg Ret | Sharpe | Max DD | Total Ret | PF |
|--------|---|-----|---------|--------|--------|-----------|----|
| v1_baseline            |  85 | 38% | +0.117% | 0.76 | -6.90% | +9.84% | 1.30 |
| v2_permissivo          |  12 | 33% | -0.173% | -1.12 | -4.90% | -2.13% | 0.68 |
| v2_central             |   9 | 44% | +0.043% | 0.26 | -2.97% | +0.33% | 1.10 |
| v2_restritivo          |   5 | 40% | -0.482% | -5.52 | -2.79% | -2.40% | 0.14 |
| v2_cm_permissivo       |  10 | 50% | +0.069% | 0.43 | -2.97% | +0.63% | 1.17 |
| v2_vz_restritivo       |  10 | 30% | -0.135% | -0.81 | -3.94% | -1.40% | 0.76 |

---

## 4. Train / Test Split (70/30 temporal)

- **Train:** 2025-10-01 → 2026-02-20
- **Test:**  2026-02-20 → 2026-04-22
- **Best config (train):** `v2_cm_permissivo` — Sharpe -4.13 (N=7)
- **Test Sharpe:** 4.71 (N=3)
- **Overfitting Δ:** -8.83

🔴 Over-fit provável (Δ ≥ 1.0)

---

## 5. Robustez por Sub-período

| Período | v1 N | v1 Sharpe | v1 WR | v2 N | v2 Sharpe | v2 WR |
|---------|------|-----------|-------|------|-----------|-------|
| Out-Dez 2025    | 29 | 2.23 | 45% |  5 | -4.25 | 40% |
| Jan-Feb 2026    | 22 | -0.02 | 36% |  2 | 1.70 | 50% |
| Mar-Abr 2026    | 32 | -0.26 | 31% |  1 | 0.00 | 0% |

**Períodos v2 com Sharpe > 1.0:** 1/3

⚠️ Regime-dependent

---

## 6. Trade 22/04 Validation

| Feature | Valor real | Range v2 central | Passa? |
|---------|-----------|-------------------|--------|
| close/MA21 | 1.0101 | [1.03, 1.05] | 🛡️ BLOQUEADO |
| volume_z | 0.218 | [0.0, 0.5] | ✅ |

**v2 teria bloqueado a trade 22/04?** ✅ SIM

---

## 7. Critérios de Decisão

| Critério | Meta | Resultado | Status |
|----------|------|-----------|--------|
| Sharpe > 1.5 | > 1.5 | 0.26 | ❌ |
| N trades ≥ 15 | ≥ 15 | 9 | ❌ |
| Max DD < 5% | < 5% | -2.97% | ✅ |
| Overfitting Δ < 0.5 | < 0.5 | -8.83 | ❌ |
| Robusto 2/3 períodos | ≥ 2/3 | 1/3 | ❌ |

**Critérios atendidos:** 1/5

---

## **VEREDITO: REJEITADO**

**Ação:** Não reativar Bot 4 v2. Considerar re-EDA ou abandono SOL.


---

## 9. Plots gerados

- `plots/sol_v2_sweet_spot/equity_curve.png`
- `plots/sol_v2_sweet_spot/trades_distribution.png`
- `plots/sol_v2_sweet_spot/threshold_heatmap.png`
- `plots/sol_v2_sweet_spot/subperiods_comparison.png`

## 10. Tables geradas

- `tables/sol_v2_sweet_spot/grid_search.csv`
- `tables/sol_v2_sweet_spot/trades_v1_vs_v2.csv`
- `tables/sol_v2_sweet_spot/subperiods_results.csv`
