# Prompt: SOL v2 "Sweet Spot" — Backtest Rigoroso

## Contexto

**SOL Bot 4 está PAUSADO** desde 22/04/2026 (crontab /etc/cron.d/aihab-cron com 3 linhas comentadas).

**Motivo do pause:**
- Phase 2 backtest reportou Sharpe 2.03 em test set (53d, N=21)
- Filter study 22/04 com histórico completo (178d, N=55) revelou Sharpe 0.08
- Overfitting Δ = -0.937 (test > train) foi red flag ignorado
- Phase 2 teve "lucky sampling" — strategy não tem edge general

**Descoberta do filter study (pistas para v2):**

Buckets univariados promissores:
- `close_ma21_ratio 1.03-1.05` (n=8): **Sharpe 5.55**, WR 62%, Avg Return +1.036%
- `volume_z 0-0.5` (n=10): **Sharpe 2.84**, WR 60%, Avg Return +0.456%

Buckets ruins (evitar):
- `close_ma21_ratio 1.02-1.03` (n=13): Sharpe -4.23
- `dist_to_resistance < 0.5%` (n=11): Sharpe -5.78
- `velocity_ratio 1-2` (n=4): Sharpe -11.55

## Hipótese a testar

**"Sweet Spot Strategy" (v2):**

Manter todos os hard gates v1 + adicionar dois filtros conservadores:

```python
# Hard gates v1 (mantém):
taker_z_4h > 0.3
oi_z_24h < 2.0
eth_ret_1h > 0
stablecoin_z > 1.3
ret_1d > 0
60 < RSI < 80
close > MA21

# NOVOS filtros v2:
1.03 <= close / MA21 <= 1.05   # Sweet spot de trend
0 <= volume_z <= 0.5            # Volume saudável, não extremo
```

**Pergunta a responder:** v2 tem edge real ou o "sweet spot" era coincidência do N=55?

## Objetivo

Validar estatisticamente se v2 tem:
- Sharpe > 1.5 no histórico completo
- N ≥ 15 trades (estatística mínima)
- Max DD < 5%
- Regime-robust (não depende de período específico)

**Critérios de decisão:**
- ✅ APROVAR v2 se: Sharpe > 1.5, N ≥ 15, DD < 5%, consistente em sub-períodos
- ❌ REJEITAR v2 se: Sharpe < 1.0 OU N < 10 OU DD > 10%
- ⚠️ INCONCLUSIVO se: resultados borderline — mais dados necessários

## Metodologia

### Fase 1: Replicar baseline v1 (validar setup)

```python
import pandas as pd
from pathlib import Path
import yaml

# Load data
sol_data = pd.read_parquet("data/01_raw/spot/sol_1h.parquet")
sol_data["timestamp"] = pd.to_datetime(sol_data["timestamp"], utc=True)
sol_data = sol_data.set_index("timestamp").sort_index()

# Load params v1
with open("conf/parameters_sol.yml") as f:
    params_v1 = yaml.safe_load(f)

# Replicar lógica Bot 4 v1 + medir Sharpe full history
v1_results = run_sol_backtest(
    data=sol_data,
    params=params_v1,
    mode="full_history"  # todos dados disponíveis
)

print(f"v1 Baseline Sharpe (full history): {v1_results['sharpe']:.2f}")
# Esperado: ~0.08 (confirma filter study)
```

### Fase 2: Implementar v2 Sweet Spot

```python
def apply_sweet_spot_filters(row, sol_df, i):
    """Filtros v2 adicionais — aplicados APÓS hard gates v1."""
    
    # Close/MA21 ratio
    ma21 = sol_df["close"].iloc[i-20:i+1].mean()
    close_ma21_ratio = row["close"] / ma21
    
    if not (1.03 <= close_ma21_ratio <= 1.05):
        return False, "outside_trend_sweet_spot"
    
    # Volume z-score (rolling 7d)
    if i < 168:
        return False, "insufficient_volume_history"
    
    vol_window = sol_df["volume"].iloc[i-168:i+1]
    vol_z = (row["volume"] - vol_window.mean()) / vol_window.std()
    
    if not (0 <= vol_z <= 0.5):
        return False, "outside_volume_sweet_spot"
    
    return True, None


# Strategy v2 = v1 gates + sweet spot filters
def run_v2_backtest(sol_data, params_v1):
    trades = []
    for i, row in enumerate(sol_data.itertuples()):
        # Hard gates v1 (replicar)
        if not pass_v1_hard_gates(row, sol_data, i, params_v1):
            continue
        
        # NOVO: Sweet spot filters v2
        passed, reason = apply_sweet_spot_filters(row, sol_data, i)
        if not passed:
            continue
        
        # Entry + exit simulation (mesma lógica v1)
        trade = simulate_entry_exit(row, sol_data, i, params_v1)
        trades.append(trade)
    
    return compute_metrics(trades)
```

### Fase 3: Testar variações de thresholds

Grid search pequeno (evitar over-fit):

```python
# Variações ao redor do sweet spot central
variations = [
    # (close_ma21_min, close_ma21_max, vol_z_min, vol_z_max)
    (1.025, 1.055, -0.1, 0.6),   # mais permissivo
    (1.03, 1.05, 0, 0.5),         # CENTRAL (filter study)
    (1.035, 1.045, 0, 0.3),       # mais restritivo
    (1.03, 1.06, -0.2, 0.7),      # close permissivo, vol permissivo
    (1.025, 1.05, 0.1, 0.5),      # close permissivo, vol restritivo
]

results = {}
for (cm_min, cm_max, vz_min, vz_max) in variations:
    v2_custom = run_v2_with_thresholds(
        sol_data, params_v1,
        close_ma21_range=(cm_min, cm_max),
        volume_z_range=(vz_min, vz_max)
    )
    key = f"cm{cm_min}-{cm_max}_vz{vz_min}-{vz_max}"
    results[key] = v2_custom
```

### Fase 4: Validação out-of-sample

**Split temporal (não walk-forward complexo):**

```python
# Split 70/30 temporal (não random)
split_date = sol_data.index[int(len(sol_data) * 0.7)]

train = sol_data[sol_data.index < split_date]
test = sol_data[sol_data.index >= split_date]

# Encontrar melhor config em TRAIN
best_config = None
best_sharpe = -999
for (cm_min, cm_max, vz_min, vz_max) in variations:
    result_train = run_v2_with_thresholds(
        train, params_v1,
        close_ma21_range=(cm_min, cm_max),
        volume_z_range=(vz_min, vz_max)
    )
    if result_train["sharpe"] > best_sharpe and result_train["n_trades"] >= 5:
        best_sharpe = result_train["sharpe"]
        best_config = (cm_min, cm_max, vz_min, vz_max)

# Aplicar best config em TEST
test_result = run_v2_with_thresholds(
    test, params_v1,
    close_ma21_range=best_config[0:2],
    volume_z_range=best_config[2:4]
)

print(f"Best config: {best_config}")
print(f"Train Sharpe: {best_sharpe:.2f}")
print(f"Test Sharpe: {test_result['sharpe']:.2f}")
print(f"Overfitting Δ: {best_sharpe - test_result['sharpe']:.2f}")
```

**Critério IMPORTANTE:**
- Overfitting Δ absoluto < 0.5 = boa generalização
- Overfitting Δ > 1.0 = over-fit provável
- Overfitting Δ < 0 (test > train) = RED FLAG "lucky sampling" (igual Phase 2!)

### Fase 5: Comparação final

```python
comparison = pd.DataFrame({
    "config": ["v1 baseline", "v2 sweet spot"] + [f"v2 var{i}" for i in range(len(variations))],
    "n_trades": [...],
    "sharpe": [...],
    "win_rate": [...],
    "avg_return": [...],
    "max_dd": [...],
    "total_return": [...],
})

print(comparison.to_string())
```

### Fase 6: Análise de robustez

Testar em SUB-PERÍODOS para detectar regime dependency:

```python
# Split em 3 períodos
periods = [
    ("Out-Dez 2025", "2025-10-01", "2025-12-31"),
    ("Jan-Feb 2026", "2026-01-01", "2026-02-28"),
    ("Mar-Abr 2026", "2026-03-01", "2026-04-22"),
]

for name, start, end in periods:
    sub = sol_data[(sol_data.index >= start) & (sol_data.index <= end)]
    result = run_v2_with_thresholds(
        sub, params_v1,
        close_ma21_range=best_config[0:2],
        volume_z_range=best_config[2:4]
    )
    print(f"{name}: Sharpe={result['sharpe']:.2f}, N={result['n_trades']}, WR={result['win_rate']:.1%}")
```

**Interpretação:**
- Sharpe consistente (> 1.0 em 2/3 períodos): strategy genuína ✅
- Sharpe forte só em 1 período: regime-dependent ⚠️
- Sharpe fraco em todos: sem edge real ❌

### Fase 7: Aplicar ao trade real 22/04

Verificar se v2 teria bloqueado:

```python
# Trade 22/04 16:15 UTC, entry $88.23
trade_22_04 = {
    "close": 88.33,
    "ma21": 87.45,
    "volume_z": 0.218,  # do filter study
}

close_ma21 = 88.33 / 87.45  # = 1.0101
volume_z = 0.218

# v2 check:
in_close_ma21 = (1.03 <= close_ma21 <= 1.05)  # False (1.010)
in_volume_z = (0 <= volume_z <= 0.5)           # True (0.218)

v2_would_block = not (in_close_ma21 and in_volume_z)

print(f"v2 bloquearia trade 22/04? {v2_would_block}")
# Esperado: True (falha em close_ma21)
```

## Output

Gerar relatório `prompts/sol_v2_sweet_spot_report.md`:

```markdown
# SOL Bot 4 v2 "Sweet Spot" — Backtest Report

**Data:** 2026-04-22
**Objetivo:** validar strategy v2 antes de reativar Bot 4

## 1. v1 Baseline (replicação)

Sharpe full history: X.XX (esperado ~0.08)
N trades: XX
→ Confirma filter study

## 2. v2 Sweet Spot (filtros centrais)

Config: close_ma21 1.03-1.05, volume_z 0-0.5

| Metric | Value |
|--------|-------|
| Sharpe | X.XX |
| N trades | XX |
| WR | XX% |
| Avg Return | X.XX% |
| Total Return | X.XX% |
| Max DD | X.XX% |
| Profit Factor | X.XX |

## 3. Grid Search — Variações

| Config | N | Sharpe | WR | DD |
|--------|---|--------|-----|-----|
| v2 central | XX | X.XX | XX% | X.X% |
| v2 permissivo | XX | X.XX | XX% | X.X% |
| v2 restritivo | XX | X.XX | XX% | X.X% |
| ... | | | | |

## 4. Train/Test Split

Best config (train): [thresholds]
Train Sharpe: X.XX (N=XX)
Test Sharpe: X.XX (N=XX)
Overfitting Δ: X.XX

[Interpretação]

## 5. Robustez (sub-períodos)

| Período | N | Sharpe | WR |
|---------|---|--------|-----|
| Out-Dez 2025 | XX | X.XX | XX% |
| Jan-Feb 2026 | XX | X.XX | XX% |
| Mar-Abr 2026 | XX | X.XX | XX% |

[Regime-dependent ou não?]

## 6. Trade 22/04 validation

v2 teria bloqueado? [Sim/Não]
Razão: [qual filtro falhou]

## 7. Recomendação

**APROVAR v2 se:**
- Sharpe > 1.5 no histórico completo
- N ≥ 15 trades
- Max DD < 5%
- Overfitting Δ < 0.5
- Consistente em 2/3 sub-períodos

**REJEITAR v2 se:**
- Sharpe < 1.0
- N < 10
- Regime-dependent forte

**Ação sugerida:**
[Aprovar + deploy | Rejeitar + abandonar | Inconclusivo + acumular dados]

## 8. Implementação (se aprovado)

```python
# Em src/trading/sol_bot4.py, adicionar em check_entry_signal():

def check_sweet_spot_filters(row, df, i):
    """v2 Sweet Spot filters."""
    if i < 21:
        return False, "insufficient_history"
    
    ma21 = df["close"].iloc[i-20:i+1].mean()
    close_ma21 = row["close"] / ma21
    if not (1.03 <= close_ma21 <= 1.05):
        return False, f"close_ma21_outside_{close_ma21:.3f}"
    
    if i < 168:
        return False, "insufficient_volume_history"
    
    vol_window = df["volume"].iloc[i-168:i+1]
    vol_z = (row["volume"] - vol_window.mean()) / vol_window.std()
    if not (0 <= vol_z <= 0.5):
        return False, f"volume_z_outside_{vol_z:.2f}"
    
    return True, None
```

## 9. Plots esperados

- `plots/sol_v2_sweet_spot/equity_curve.png` — curva capital v1 vs v2
- `plots/sol_v2_sweet_spot/trades_distribution.png` — histograma retornos
- `plots/sol_v2_sweet_spot/threshold_heatmap.png` — Sharpe por (close_ma21, volume_z)
- `plots/sol_v2_sweet_spot/subperiods_comparison.png` — robustez por período

## 10. Tables

- `tables/sol_v2_sweet_spot/grid_search.csv`
- `tables/sol_v2_sweet_spot/trades_v1_vs_v2.csv`
- `tables/sol_v2_sweet_spot/subperiods_results.csv`
```

## Tarefas resumo

1. ✅ Script `scripts/sol_v2_sweet_spot_backtest.py`
2. ✅ Replicar baseline v1 (validar Sharpe ~0.08)
3. ✅ Implementar v2 sweet spot
4. ✅ Grid search (5 variações)
5. ✅ Train/Test split com overfitting check
6. ✅ Análise por sub-períodos
7. ✅ Validar trade 22/04
8. ✅ Relatório final com recomendação clara
9. ✅ Plots + tables

## Critérios de qualidade

- ✅ Código limpo, comentado
- ✅ Overfitting check explícito
- ✅ Múltiplas validações (grid, train/test, sub-períodos)
- ✅ Recomendação BINÁRIA clara (aprovar/rejeitar)
- ✅ Se aprovar: código de implementação pronto

## Contexto importante

**Aprendizados do Phase 2 (não repetir):**
- N=21 trades foi insuficiente
- Overfitting Δ=-0.937 (test > train) foi IGNORADO — big mistake
- "Lucky sampling" do test set foi confundido com edge real

**Para v2 não cometer mesmo erro:**
- Exigir N ≥ 15 trades
- Overfitting Δ dentro de ±0.5
- Consistência em sub-períodos
- Validar em trade real 22/04

## Dados disponíveis

```
data/01_raw/spot/sol_1h.parquet  # 202 dias, OHLCV 1h
data/01_raw/futures/sol_oi_4h.parquet
data/01_raw/futures/sol_taker_4h.parquet
data/01_raw/spot/eth_1h.parquet  # para ETH context
conf/parameters_sol.yml  # v1 params
```

## Tempo estimado

- Setup + replicação baseline: 20 min
- v2 implementação: 30 min
- Grid search: 30 min
- Train/test + sub-períodos: 30 min
- Plots + relatório: 30 min
- **Total: ~2h30min**

## Se v2 APROVADO — Passos seguintes

1. Implementar em `src/trading/sol_bot4.py`
2. Atualizar `conf/parameters_sol.yml` com thresholds v2
3. Testar local
4. Commit + deploy AWS (rebuild)
5. Reativar crontab (remover comentário "PAUSED")
6. Monitorar próximas 10 trades

## Se v2 REJEITADO — Decisões possíveis

1. Abandonar SOL (focar em BTC onde há edge real)
2. Re-EDA com abordagem completamente diferente
3. Acumular mais dados e revisitar em 1-2 meses

## Output final esperado

Relatório CLARO com recomendação acionável:
- Se aprovar: código + parameters prontos para deploy
- Se rejeitar: razões + próximos passos
- Se inconclusivo: quais dados mais precisamos
