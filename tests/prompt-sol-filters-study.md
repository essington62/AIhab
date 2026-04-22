# Prompt: SOL Bot 4 — Estudo Estatístico de Filtros Estruturais

## Contexto

**Primeira trade SOL Bot 4 live fechou com -0.98% em 22/04/2026.**

Entry: $88.23 @ 16:15 UTC
Exit: $87.37 @ 19:43 UTC (TRAIL)
Duration: 3h 28min

**Problema identificado (análise visual + price action):**

O bot entrou em **fase 4 de um rally** (tardia) em vez do breakout inicial:
```
1. Lateral $83-85 (4/20-4/21)
2. Breakout 4/22 06:00 UTC
3. Aceleração até 15:00 UTC (peak ~$88.55)
4. ENTRY 16:15 UTC @ $88.23 ← fase tardia
5. Distribuição
6. Queda até $87.37 (TRAIL hit)
```

**Diagnóstico conceitual:**

Bot 4 vê features SNAPSHOT (ret_1d, taker_z, oi_z) mas não vê ESTRUTURA de mercado:
- Distância até resistência próxima
- Tempo desde breakout
- Extensão já realizada
- Volume pattern (crescendo vs exaustão)

**Trade real tinha:**
- Entry $88.23 a 0.36% de resistance 12h ($88.55)
- TP $89.99 exigia furar resistência +1.6% (estatisticamente baixa probabilidade)
- R/R estrutural: 0.09 (horrível)

## Objetivo do estudo

**Testar estatisticamente 4 filtros candidatos** em dados históricos SOL 2026+, identificando quais MELHORAM a strategy sem matar o edge original (Sharpe backtest 2.03).

### Filtros a testar (ordem de prioridade conceitual)

| # | Filtro | Conceito |
|---|--------|----------|
| 1 | **Structural (S/R)** | Distância até resistance/support 12h |
| 2 | **Volume_z** | Volume alto = exaustão (insight ETH transferido) |
| 3 | **Extension** | close/MA21 ratio + ret_1d combo |
| 4 | **Velocity** | ret_1h / ret_24h ratio |

## Input data

- **Arquivo:** `data/01_raw/spot/sol_1h.parquet` (SOL OHLCV 1h, 202 dias)
- **Período análise:** 2026+ (pós-regime change pós-ETFs)
- **Derivatives:** `data/01_raw/futures/sol_oi_4h.parquet`, `sol_taker_4h.parquet`
- **ETH context:** `data/01_raw/spot/eth_1h.parquet`
- **Strategy baseline:** `conf/parameters_sol.yml` (Phase 2 config)

## Metodologia

### Fase 1: Replicar backtest baseline

Carregar parameters SOL atual e rodar backtest baseline:

```python
# Baseline SOL strategy (Phase 2)
baseline_params = {
    "taker_z_4h_min": 0.3,
    "oi_z_24h_block": 2.0,
    "oi_z_1h_min": -0.5,
    "rsi_min": 60,
    "rsi_max": 80,
    "stablecoin_z_min": 1.3,
    "ret_1d_min": 0,
    "sl_pct": 0.015,
    "tp_pct": 0.020,
    "trail_pct": 0.010,
    # ... outros do parameters_sol.yml
}

# Rodar backtest com EXATAMENTE a lógica atual
baseline_results = run_sol_backtest(
    data=sol_2026_data,
    params=baseline_params,
    train_test_split=None,  # usar TUDO 2026+
)

print(f"Baseline Sharpe: {baseline_results['sharpe']:.2f}")
print(f"Baseline Trades: {baseline_results['n_trades']}")
print(f"Baseline WR: {baseline_results['win_rate']:.1%}")
```

**Validar:** Sharpe deve ser próximo de 2.03 (Phase 2 test set). Se muito diferente, revisar lógica.

### Fase 2: Compute features EXTRA pra cada trade entry

Para cada entry do baseline, calcular:

```python
def compute_structural_features(df, entry_time, lookback_hours=12):
    """Compute S/R e outros features estruturais."""
    
    # Janela lookback
    window = df[df.index < entry_time].tail(lookback_hours)
    entry_row = df[df.index == entry_time].iloc[0]
    entry_price = entry_row["close"]
    
    # === S/R ===
    resistance_12h = window["high"].max()
    support_12h = window["low"].min()
    
    dist_to_resistance_pct = (resistance_12h - entry_price) / entry_price
    dist_from_support_pct = (entry_price - support_12h) / entry_price
    
    structural_rr = (
        dist_to_resistance_pct / dist_from_support_pct 
        if dist_from_support_pct > 0 else float('inf')
    )
    
    # === Volume ===
    volume_rolling_mean = df["volume"].rolling(168).mean()  # 7d
    volume_rolling_std = df["volume"].rolling(168).std()
    volume_z = (entry_row["volume"] - volume_rolling_mean.loc[entry_time]) / volume_rolling_std.loc[entry_time]
    
    # === Extension ===
    ma21 = df["close"].rolling(21).mean()
    close_ma21_ratio = entry_price / ma21.loc[entry_time]
    
    # === Velocity ===
    ret_1h = df["close"].pct_change(1).loc[entry_time]
    ret_24h = df["close"].pct_change(24).loc[entry_time]
    velocity_ratio = ret_1h / (ret_24h / 24) if ret_24h > 0 else 0
    # velocity > 1 = movimento CONCENTRADO na última hora (exaustão?)
    
    return {
        "dist_to_resistance_pct": dist_to_resistance_pct,
        "dist_from_support_pct": dist_from_support_pct,
        "structural_rr": structural_rr,
        "volume_z": volume_z,
        "close_ma21_ratio": close_ma21_ratio,
        "ret_1h_prev": ret_1h,
        "velocity_ratio": velocity_ratio,
    }

# Aplicar a cada trade do baseline
for trade in baseline_trades:
    trade.update(compute_structural_features(
        sol_data, trade["entry_time"]
    ))
```

### Fase 3: Análise univariada

Para cada feature, agrupar trades por buckets e ver performance:

```python
import pandas as pd
import numpy as np

# Feature 1: dist_to_resistance_pct
buckets = [0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 1.0]
labels = ["<0.5%", "0.5-1%", "1-1.5%", "1.5-2%", "2-3%", "3-5%", ">5%"]

trades_df["resistance_bucket"] = pd.cut(
    trades_df["dist_to_resistance_pct"],
    bins=buckets,
    labels=labels
)

analysis = trades_df.groupby("resistance_bucket").agg(
    n_trades=("return_pct", "count"),
    win_rate=("return_pct", lambda x: (x > 0).mean()),
    avg_return=("return_pct", "mean"),
    total_return=("return_pct", lambda x: (1 + x/100).prod() - 1),
    sharpe=("return_pct", lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0),
)

print("=== Análise por distância até resistência ===")
print(analysis)

# Identificar bucket PIOR (= threshold candidato pra bloquear)
worst_bucket = analysis["sharpe"].idxmin()
print(f"\nPior bucket: {worst_bucket}")
```

**Repetir para cada feature:**
- `dist_to_resistance_pct` (P1)
- `structural_rr` (P1)
- `volume_z` (P2)
- `close_ma21_ratio` (P3)
- `ret_1h_prev` (P4)
- `velocity_ratio` (P4)

### Fase 4: Gerar gráficos diagnósticos

```python
import matplotlib.pyplot as plt

# Para cada feature, scatter: feature vs forward_return_2h
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

features_to_plot = [
    ("dist_to_resistance_pct", "Dist to Resistance"),
    ("structural_rr", "Structural R/R"),
    ("volume_z", "Volume Z-score"),
    ("close_ma21_ratio", "Close/MA21"),
    ("ret_1h_prev", "Return 1h prior"),
    ("velocity_ratio", "Velocity ratio"),
]

for ax, (feat, title) in zip(axes.flat, features_to_plot):
    ax.scatter(
        trades_df[feat],
        trades_df["return_pct"],
        alpha=0.5,
        c=trades_df["return_pct"] > 0,
        cmap="RdYlGn"
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(feat)
    ax.set_ylabel("Trade return %")
    ax.set_title(title)

plt.tight_layout()
plt.savefig("prompts/plots/sol_filters_study/scatter_features.png", dpi=100)
```

### Fase 5: Testar filtros aplicados

Para cada filtro CANDIDATO, re-rodar backtest e comparar:

```python
# Filtro 1: Block se resistance próxima
def filter_close_to_resistance(trade, threshold_pct=0.010):
    return trade["dist_to_resistance_pct"] > threshold_pct

# Filtro 2: Block se structural R/R ruim
def filter_bad_rr(trade, min_rr=0.3):
    return trade["structural_rr"] > min_rr

# Filtro 3: Block se volume extremo
def filter_high_volume(trade, max_z=1.0):
    return trade["volume_z"] < max_z

# ... etc

# Testar cada filtro isoladamente
configs_to_test = [
    ("baseline", {}),
    ("no_resistance_close", {"filter": filter_close_to_resistance}),
    ("no_bad_rr", {"filter": filter_bad_rr}),
    ("no_high_volume", {"filter": filter_high_volume}),
    ("no_extension", {"filter": filter_extension}),
    ("no_velocity_spike", {"filter": filter_velocity}),
    # Combinações
    ("combo_A", {"filters": [filter_close_to_resistance, filter_high_volume]}),
    ("combo_B", {"filters": [filter_bad_rr, filter_extension]}),
]

results = {}
for name, config in configs_to_test:
    result = run_sol_backtest_with_filter(
        data=sol_data,
        params=baseline_params,
        extra_filter=config.get("filter"),
        extra_filters=config.get("filters"),
    )
    results[name] = result

# Comparação
comparison_df = pd.DataFrame([
    {
        "config": name,
        "n_trades": r["n_trades"],
        "win_rate": r["win_rate"],
        "avg_return": r["avg_return"],
        "sharpe": r["sharpe"],
        "max_dd": r["max_dd"],
        "total_return": r["total_return"],
    }
    for name, r in results.items()
])

print(comparison_df.to_string())
```

### Fase 6: Grid search em thresholds

Para o filtro MAIS promissor, fazer grid search em thresholds:

```python
# Exemplo: se "dist_to_resistance" é promissor
thresholds_to_test = [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020]

grid_results = []
for thr in thresholds_to_test:
    result = run_with_resistance_filter(data, params, thr)
    grid_results.append({
        "threshold": thr,
        "n_trades": result["n_trades"],
        "sharpe": result["sharpe"],
        "win_rate": result["win_rate"],
    })

# Identificar melhor
best = max(grid_results, key=lambda x: x["sharpe"])
print(f"Melhor threshold: {best['threshold']*100:.1f}%")
print(f"Sharpe: {best['sharpe']:.2f}")
print(f"Trades: {best['n_trades']}")
```

## Output esperado

Gerar relatório em `prompts/sol_filters_study_report.md`:

```markdown
# SOL Bot 4 — Estudo de Filtros Estruturais

**Data:** 2026-04-22
**Trigger:** primeira trade live -0.98% (fase tardia rally)

## 1. Baseline

| Metric | Value |
|--------|-------|
| Sharpe | 2.03 |
| Trades | 21 |
| Win Rate | 57% |
| Avg Return | 0.4% |
| Max DD | -1.4% |

## 2. Análise univariada (por feature)

### Distance to Resistance (P1)
| Bucket | N | WR | Avg | Sharpe |
|--------|---|-----|-----|--------|
| <0.5%  | X | XX% | XX% | X.XX |
| 0.5-1% | X | XX% | XX% | X.XX |
| ...    |   |     |     |      |

**Insight:** [descrever relação]

### Structural R/R (P1)
...

### Volume Z (P2)
...

### Extension (P3)
...

### Velocity (P4)
...

## 3. Teste de filtros isolados

| Config | N | WR | Sharpe | Δ vs baseline |
|--------|---|-----|--------|---------------|
| Baseline | 21 | 57% | 2.03 | — |
| No resistance close (<1%) | X | XX | X.XX | +X.XX |
| No bad R/R (<0.3) | X | XX | X.XX | +X.XX |
| No high volume (>1.0) | X | XX | X.XX | +X.XX |
| ... | | | | |

## 4. Filtros combinados

| Combo | N | WR | Sharpe | Δ vs baseline |
|-------|---|-----|--------|---------------|
| resistance + volume | X | XX | X.XX | +X.XX |
| rr + extension | X | XX | X.XX | +X.XX |
| ... | | | | |

## 5. Grid search (melhor filtro)

[se filtro X venceu, mostrar threshold ótimo]

## 6. Análise da trade live (22/04)

Apply filtros à trade real:
- dist_to_resistance_pct: 0.36% → BLOQUEADO por "close_to_resistance < 1%" ✓
- structural_rr: 0.09 → BLOQUEADO por "min_rr 0.3" ✓
- Volume_z: X → [check]
- Extension: X → [check]

## 7. Recomendação

### Filtro RECOMENDADO (se aplicável)
[nome] com threshold [valor]

Justificativa:
- Sharpe melhorado de 2.03 para X.XX
- Trades mantidos: X/21 (manteve statistical power)
- Análise da trade real CONFIRMA
- Princípio conceitual sólido (não over-fit)

### Se NENHUM filtro recomendado:
- Trade atual foi outlier N=1
- Strategy original mantém-se válida
- Continuar monitorando

## 8. Implementação sugerida

```python
# Em src/trading/sol_bot4.py, adicionar em check_entry_signal():

def check_structural_guard(row, df, lookback=12):
    """Block entries perto da resistência."""
    recent = df.tail(lookback)
    resistance = recent["high"].max()
    dist_pct = (resistance - row["close"]) / row["close"]
    
    if dist_pct < 0.010:  # configurable
        return False, f"close_to_resistance_{dist_pct:.2%}"
    
    return True, None
```
```

## Tarefas resumo

1. ✅ Criar script `scripts/sol_filters_study.py`
2. ✅ Replicar backtest baseline (Sharpe ~2.03)
3. ✅ Computar features estruturais pra cada trade
4. ✅ Análise univariada (6 features, buckets, métricas)
5. ✅ Gerar 6 scatter plots + 2-3 bar charts
6. ✅ Testar filtros isolados (6 configs)
7. ✅ Testar filtros combinados (3-5 combos)
8. ✅ Grid search do melhor filtro
9. ✅ Aplicar à trade real (22/04) pra validar
10. ✅ Gerar relatório markdown final
11. ✅ Sugerir código de implementação

## Critérios de decisão

**APLICAR filtro se:**
- Sharpe melhora ≥ 0.3 vs baseline
- Mantém ≥ 60% dos trades (N ≥ 13)
- Filtra trade real 22/04 corretamente
- Princípio conceitual sólido (não over-fit)

**NÃO APLICAR se:**
- Sharpe melhora < 0.3 (insignificante)
- Mata > 40% dos trades (overfit)
- Nenhum princípio conceitual forte
- Complexidade >> benefício

## Input esperados

- Baseline Sharpe próximo 2.03
- 15-25 trades históricos SOL 2026
- Dados OHLCV sol_1h.parquet completos
- Dados derivatives SOL 4h

## Output esperado

Relatório `prompts/sol_filters_study_report.md` com:
- 7 seções (análise completa)
- Tabelas comparativas
- Gráficos salvos em `prompts/plots/sol_filters_study/`
- Recomendação CLARA: aplicar ou não, qual filtro, threshold

## Tempo estimado

- Setup + baseline: 20 min
- Compute features: 15 min
- Análise univariada: 20 min
- Plots: 10 min
- Filtros isolados + combos: 30 min
- Grid search: 15 min
- Relatório: 20 min
- **Total: ~2h de trabalho Claude Code**

## Arquivos esperados

```
scripts/sol_filters_study.py           ← script principal
prompts/sol_filters_study_report.md    ← relatório
prompts/plots/sol_filters_study/       ← gráficos
  scatter_features.png
  filter_comparison_bars.png
  grid_search_threshold.png
prompts/tables/sol_filters_study/      ← CSVs
  univariate_analysis.csv
  filter_comparison.csv
  grid_search_results.csv
```
